"""FastAPI route handlers."""

import json
from collections.abc import Mapping
from typing import Annotated, Any, cast

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from loguru import logger
from pydantic import ValidationError

from free_claude_code.application.provider_quota import ProviderQuotaService
from free_claude_code.application.errors import ApplicationError, InvalidRequestError
from free_claude_code.application.ports import ProviderResolver, RequestRuntimeLease
from free_claude_code.application.routing import ModelRouter, supports_native_messages
from free_claude_code.config.model_refs import parse_provider_type
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import (
    MessagesRequest,
    NativeTokenCountRequest,
    TokenCountRequest,
    get_token_count,
)
from free_claude_code.core.anthropic.native import (
    NativeMessagesError,
    validate_messages_json,
)
from free_claude_code.core.anthropic.passthrough import NativeMessagesRequest
from free_claude_code.core.gemini import GeminiRequest
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.trace import trace_event
from free_claude_code.application.usage_report import UsageDatabase

from .dependencies import (
    get_services,
    get_settings,
    require_anthropic_proxy_auth,
    require_gemini_proxy_auth,
    require_proxy_auth,
    resolve_provider,
)
from .handlers import MessagesHandler, ResponsesHandler, TokenCountHandler
from .handlers.gemini import GeminiHandler
from .model_catalog import (
    ModelCatalogView,
    ModelsListResponse,
    build_models_list_response,
    build_muse_models_list_response,
)
from .ports import ApiServices
from .request_errors import ordinary_application_error_response
from .request_ids import get_request_id
from .response_streams import bind_response_lifetime

router = APIRouter()


def _unique_json_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


@router.post("/v1beta/models/{resource:path}")
async def create_gemini_response(
    resource: str,
    request: Request,
    request_data: Annotated[dict[str, Any], Body()],
    services: ApiServices = Depends(get_services),
    _auth=Depends(require_gemini_proxy_auth),
):
    try:
        json.loads(await request.body(), object_pairs_hook=_unique_json_fields)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    model, separator, action = resource.rpartition(":")
    if not separator or action not in {"generateContent", "streamGenerateContent"}:
        raise HTTPException(status_code=404, detail="Unknown Gemini generation method")
    if not model.strip() or any(
        not segment or segment in {".", ".."} or any(char.isspace() for char in segment)
        for segment in model.split("/")
    ):
        raise HTTPException(status_code=400, detail="Invalid model resource")
    streaming = action == "streamGenerateContent"
    alt = request.query_params.getlist("alt")
    if (
        len(alt) > 1
        or (streaming and alt != ["sse"])
        or (not streaming and alt not in ([], ["json"]))
    ):
        raise HTTPException(
            status_code=400,
            detail="Streaming requires alt=sse; JSON generation accepts alt=json",
        )
    if "model" in request_data:
        raise HTTPException(status_code=400, detail="Specify the model in the URL only")
    try:
        data = GeminiRequest.model_validate({**request_data, "model": model})
    except ValidationError as error:
        raise RequestValidationError(
            [{**item, "loc": ("body", *item["loc"])} for item in error.errors()],
            body=request_data,
        ) from error
    lease = await services.requests.acquire()
    try:
        await lease.wait_for_token_estimation()
        handler = GeminiHandler(
            lease.settings,
            _provider_resolver(lease),
            generation_id=lease.generation_id,
            request_headers={
                key: value
                for key, value in request.headers.items()
                if key.lower() not in {"authorization", "x-api-key", "x-goog-api-key"}
            },
            model_info_lookup=lease.model_info,
        )
        response = await handler.create(
            data, streaming=streaming, request_id=get_request_id(request)
        )
    except BaseException:
        await lease.release()
        raise
    return await bind_response_lifetime(response, lease.release)


def _provider_resolver(lease: RequestRuntimeLease) -> ProviderResolver:
    return lambda provider_type, api_key_number: resolve_provider(
        provider_type, api_key_number, lease=lease
    )


async def _create_messages_response(
    services: ApiServices,
    request_data: MessagesRequest | JsonObject,
    *,
    request_id: str,
    request_headers: Mapping[str, str] | None = None,
) -> object:
    lease: RequestRuntimeLease | None = None
    try:
        lease = await services.requests.acquire()
        router = ModelRouter(lease.settings)
        raw = (
            request_data.model_dump(mode="json", exclude_unset=True)
            if isinstance(request_data, MessagesRequest)
            else request_data
        )
        model = raw.get("model")
        resolved = (
            router.resolve(model) if isinstance(model, str) and model.strip() else None
        )
        native = resolved is not None and supports_native_messages(
            resolved.primary.provider_id
        )
        if native:
            try:
                native_request = NativeMessagesRequest(raw)
            except NativeMessagesError as error:
                raise InvalidRequestError(str(error)) from error
        else:
            if not isinstance(request_data, MessagesRequest):
                try:
                    request_data = MessagesRequest.model_validate(raw)
                except ValidationError as error:
                    raise RequestValidationError(
                        [
                            {**item, "loc": ("body", *item["loc"])}
                            for item in error.errors()
                        ],
                        body=raw,
                    ) from error
            await lease.wait_for_token_estimation()
        handler = MessagesHandler(
            lease.settings,
            web_tools=services.web_tools,
            provider_resolver=_provider_resolver(lease),
            token_counter=get_token_count,
            generation_id=lease.generation_id,
            request_headers=request_headers,
            model_info_lookup=lease.model_info,
        )
        if native:
            assert resolved is not None
            response = await handler.create_native(
                router.route_native_messages(native_request, resolved),
                request_id=request_id,
            )
        else:
            assert isinstance(request_data, MessagesRequest)
            response = await handler.create(request_data, request_id=request_id)
    except ApplicationError as exc:
        if lease is not None:
            await lease.release()
        return ordinary_application_error_response(
            exc,
            wire_api="messages",
            request_id=request_id,
        )
    except BaseException:
        if lease is not None:
            await lease.release()
        raise
    assert lease is not None
    return await bind_response_lifetime(response, lease.release)


async def _create_responses_response(
    services: ApiServices,
    request_data: OpenAIResponsesRequest,
    *,
    request_id: str,
    request_headers: Mapping[str, str] | None = None,
) -> object:
    lease: RequestRuntimeLease | None = None
    try:
        lease = await services.requests.acquire()
        await lease.wait_for_token_estimation()
        handler = ResponsesHandler(
            lease.settings,
            provider_resolver=_provider_resolver(lease),
            generation_id=lease.generation_id,
            request_headers=request_headers,
            model_info_lookup=lease.model_info,
        )
        response = await handler.create(request_data, request_id=request_id)
    except ApplicationError as exc:
        if lease is not None:
            await lease.release()
        return ordinary_application_error_response(
            exc,
            wire_api="responses",
            request_id=request_id,
        )
    except BaseException:
        if lease is not None:
            await lease.release()
        raise
    assert lease is not None
    return await bind_response_lifetime(response, lease.release)


def _probe_response(allow: str) -> Response:
    return Response(status_code=204, headers={"Allow": allow})


@router.post("/v1/messages")
async def create_message(
    request: Request,
    request_data: Annotated[dict[str, Any], Body()],
    services: ApiServices = Depends(get_services),
    _auth=Depends(require_anthropic_proxy_auth),
):
    """Create a message (JSON by default; stream=true returns Anthropic SSE)."""
    return await _create_messages_response(
        services,
        cast(JsonObject, request_data),
        request_id=get_request_id(request),
        request_headers=request.headers,
    )


@router.api_route("/v1/messages", methods=["HEAD", "OPTIONS"])
async def probe_messages(_auth=Depends(require_anthropic_proxy_auth)):
    return _probe_response("POST, HEAD, OPTIONS")


@router.post("/v1/responses")
async def create_response(
    request: Request,
    request_data: OpenAIResponsesRequest,
    services: ApiServices = Depends(get_services),
    _auth=Depends(require_proxy_auth),
):
    """Create an OpenAI Responses-compatible response through this proxy."""
    return await _create_responses_response(
        services,
        request_data,
        request_id=get_request_id(request),
        request_headers=request.headers,
    )


@router.api_route("/v1/responses", methods=["HEAD", "OPTIONS"])
async def probe_responses(_auth=Depends(require_proxy_auth)):
    return _probe_response("POST, HEAD, OPTIONS")


@router.post("/v1/messages/count_tokens")
async def count_tokens(
    request: Request,
    request_data: Annotated[dict[str, Any], Body()],
    services: ApiServices = Depends(get_services),
    _auth=Depends(require_anthropic_proxy_auth),
):
    """Count tokens for a request."""
    lease = await services.requests.acquire()
    try:
        model_router = ModelRouter(lease.settings)
        model = request_data.get("model")
        if isinstance(model, str) and not model.strip():
            raise InvalidRequestError("Messages model must not be empty.")
        resolved = (
            model_router.resolve(model)
            if isinstance(model, str) and model.strip()
            else None
        )
        try:
            if resolved is not None and supports_native_messages(
                resolved.primary.provider_id
            ):
                validate_messages_json(request_data)
                counted = NativeTokenCountRequest.model_validate(request_data)
            else:
                counted = TokenCountRequest.model_validate(request_data)
        except NativeMessagesError as error:
            raise InvalidRequestError(str(error)) from error
        except ValidationError as error:
            raise RequestValidationError(
                [{**item, "loc": ("body", *item["loc"])} for item in error.errors()],
                body=request_data,
            ) from error
        await lease.wait_for_token_estimation()
        handler = TokenCountHandler(
            lease.settings, model_router=model_router, token_counter=get_token_count
        )
        return handler.count(
            counted, request_id=get_request_id(request), resolved=resolved
        )
    finally:
        await lease.release()


@router.api_route("/v1/messages/count_tokens", methods=["HEAD", "OPTIONS"])
async def probe_count_tokens(_auth=Depends(require_anthropic_proxy_auth)):
    return _probe_response("POST, HEAD, OPTIONS")


@router.get("/")
async def root(
    settings: Settings = Depends(get_settings),
    _auth=Depends(require_proxy_auth),
):
    return {
        "status": "ok",
        "provider": parse_provider_type(settings.model),
        "model": settings.model,
    }


@router.api_route("/", methods=["HEAD", "OPTIONS"])
async def probe_root():
    return _probe_response("GET, HEAD, OPTIONS")


@router.get("/health")
async def health():
    return {"status": "healthy"}


@router.api_route("/health", methods=["HEAD", "OPTIONS"])
async def probe_health():
    return _probe_response("GET, HEAD, OPTIONS")


@router.get("/api/usage")
async def usage_api(
    services: ApiServices = Depends(get_services),
):
    """Get quota data for all configured providers (public endpoint)."""
    quota_service = ProviderQuotaService(services.requests.current_settings())
    groups = await quota_service.get_all_quotas()
    return {
        "groups": [
            {
                "name": group.name,
                "display_name": group.display_name,
                "models": list(group.models),
                "providers": [
                    {
                        "provider_id": p.provider_id,
                        "display_name": p.display_name,
                        "group": p.group,
                        "error": p.error,
                        "last_checked": p.last_checked,
                        "windows": [
                            {
                                "label": w.label,
                                "used_pct": w.used_pct,
                                "remaining_pct": w.remaining_pct,
                                "resets_at": w.resets_at,
                                "refreshes_in": w.refreshes_in,
                                "limit_type": w.limit_type,
                                "raw_used": w.raw_used,
                                "raw_limit": w.raw_limit,
                            }
                            for w in p.windows
                        ],
                    }
                    for p in group.providers
                ],
            }
            for group in groups.values()
        ]
    }


async def _usage_report(request: Request, services: ApiServices) -> dict:
    from free_claude_code.application.usage_report import build_report

    from .admin_security import require_loopback_admin

    require_loopback_admin(request)
    params = request.query_params
    return await build_report(
        getattr(services.admin, "database", None),
        services.requests.current_settings(),
        session_id=params.get("session") or None,
        period=params.get("period") or "day",
        refresh=params.get("refresh") == "1",
    )


@router.get("/api/usage/report")
async def usage_report_api(
    request: Request,
    services: ApiServices = Depends(get_services),
):
    """Session totals, provider limits, and usage drivers (loopback only)."""
    return await _usage_report(request, services)


@router.get("/api/usage/text")
async def usage_report_text(
    request: Request,
    services: ApiServices = Depends(get_services),
):
    """The same report as plain text, for the /fcc-usage Claude Code command."""
    from fastapi.responses import PlainTextResponse

    from free_claude_code.application.usage_report import render_text

    return PlainTextResponse(render_text(await _usage_report(request, services)))


@router.get("/api/usage/session")
async def session_usage_api(
    services: ApiServices = Depends(get_services),
):
    """Get session usage analytics (cost, tokens, duration, cache stats)."""
    database: UsageDatabase | None = getattr(services.admin, "database", None)
    if database is None:
        return {
            "total_cost": 0.0,
            "total_duration_api_ms": 0,
            "total_duration_wall_ms": 0,
            "total_requests": 0,
            "by_model": [],
            "cache_stats": {
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "cache_savings_pct": 0.0,
            },
            "by_provider": [],
        }

    def operation(conn):
        # Get all usage data
        rows = conn.execute(
            """
            SELECT
                provider_id,
                model,
                wire_api,
                SUM(input_tokens) as total_input,
                SUM(output_tokens) as total_output,
                SUM(cache_read_tokens) as total_cache_read,
                SUM(cache_write_tokens) as total_cache_write,
                SUM(cost_usd) as total_cost,
                SUM(duration_ms) as total_duration,
                COUNT(*) as request_count,
                MAX(created_at) as last_request
            FROM request_usage
            GROUP BY provider_id, model, wire_api
            ORDER BY total_cost DESC
            """
        ).fetchall()

        # Overall totals
        totals = conn.execute(
            """
            SELECT
                SUM(cost_usd) as total_cost,
                SUM(duration_ms) as total_duration,
                SUM(input_tokens + output_tokens) as total_tokens,
                SUM(cache_read_tokens) as total_cache_read,
                SUM(cache_write_tokens) as total_cache_write,
                COUNT(*) as total_requests,
                MIN(created_at) as first_request,
                MAX(created_at) as last_request
            FROM request_usage
            """
        ).fetchone()

        return rows, totals

    rows, totals = await database.run(operation, write=False)

    by_model = []
    by_provider = {}

    for row in rows:
        provider_id = row["provider_id"]
        model = row["model"]
        wire_api = row["wire_api"]
        total_input = row["total_input"] or 0
        total_output = row["total_output"] or 0
        total_cache_read = row["total_cache_read"] or 0
        total_cache_write = row["total_cache_write"] or 0
        total_cost = row["total_cost"] or 0.0
        total_duration = row["total_duration"] or 0.0
        request_count = row["request_count"] or 0

        by_model.append({
            "provider_id": provider_id,
            "model": model,
            "wire_api": wire_api,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cache_read_tokens": total_cache_read,
            "cache_write_tokens": total_cache_write,
            "total_tokens": total_input + total_output + total_cache_read + total_cache_write,
            "cost_usd": round(total_cost, 6),
            "duration_ms": round(total_duration, 2),
            "request_count": request_count,
        })

        if provider_id not in by_provider:
            by_provider[provider_id] = {
                "provider_id": provider_id,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "cost_usd": 0.0,
                "duration_ms": 0.0,
                "request_count": 0,
            }
        p = by_provider[provider_id]
        p["input_tokens"] += total_input
        p["output_tokens"] += total_output
        p["cache_read_tokens"] += total_cache_read
        p["cache_write_tokens"] += total_cache_write
        p["cost_usd"] += total_cost
        p["duration_ms"] += total_duration
        p["request_count"] += request_count

    # Calculate cache stats
    total_cache_read = totals["total_cache_read"] or 0
    total_cache_write = totals["total_cache_write"] or 0
    total_input_all = sum(m["input_tokens"] for m in by_model)
    total_tokens_all = sum(m["total_tokens"] for m in by_model)

    cache_savings_pct = 0.0
    if total_input_all > 0:
        cache_savings_pct = round((total_cache_read / total_input_all) * 100, 1)

    total_cost = totals["total_cost"] or 0.0
    total_duration = totals["total_duration"] or 0.0
    total_requests = totals["total_requests"] or 0

    # Wall time estimate (time between first and last request)
    wall_duration = 0
    if totals["first_request"] and totals["last_request"]:
        wall_duration = (totals["last_request"] - totals["first_request"]) * 1000

    return {
        "total_cost": round(total_cost, 4),
        "total_duration_api_ms": round(total_duration, 2),
        "total_duration_wall_ms": wall_duration,
        "total_requests": total_requests,
        "total_tokens": total_tokens_all,
        "by_model": by_model,
        "cache_stats": {
            "cache_read_tokens": total_cache_read,
            "cache_write_tokens": total_cache_write,
            "cache_savings_pct": cache_savings_pct,
        },
        "by_provider": list(by_provider.values()),
    }


@router.get(
    "/v1/models",
    response_model=ModelsListResponse,
    response_model_exclude_none=True,
)
async def list_models(
    view: ModelCatalogView | None = None,
    x_fcc_model_view: ModelCatalogView | None = Header(default=None),
    services: ApiServices = Depends(get_services),
    _auth=Depends(require_proxy_auth),
):
    """List the model ids this proxy advertises to compatible clients."""
    trace_event(stage="ingress", event="free_claude_code.api.models.list", source="api")
    snapshot = services.requests.catalog_snapshot()
    return build_models_list_response(
        snapshot.settings,
        snapshot,
        view=view or x_fcc_model_view or ModelCatalogView.CLAUDE,
    )


@router.get("/usage")
async def usage_page(request: Request):
    """Serve the usage page (session, provider limits, what's contributing)."""
    from pathlib import Path

    from fastapi.responses import HTMLResponse

    from .admin_security import require_loopback_admin

    require_loopback_admin(request)
    page = Path(__file__).resolve().parent / "static" / "usage.html"
    return HTMLResponse(page.read_text(encoding="utf-8"))


@router.get(
    "/muse-code/models",
    response_model=ModelsListResponse,
    response_model_exclude_none=True,
)
async def list_muse_models(
    services: ApiServices = Depends(get_services),
    _auth=Depends(require_proxy_auth),
):
    """List the direct Responses models expected by Muse Code."""
    trace_event(stage="ingress", event="free_claude_code.api.models.list", source="api")
    snapshot = services.requests.catalog_snapshot()
    return build_muse_models_list_response(snapshot.settings, snapshot)


@router.post("/stop")
async def stop_cli(
    services: ApiServices = Depends(get_services),
    _auth=Depends(require_proxy_auth),
):
    """Stop all CLI sessions and pending tasks."""
    result = await services.tasks.stop_all()
    if result is None:
        raise HTTPException(status_code=503, detail="Messaging system not initialized")
    if result.source is not None:
        logger.info("STOP_CLI: source={} cancelled_count=N/A", result.source)
        return {"status": "stopped", "source": result.source}

    count = result.cancelled_count or 0
    trace_event(
        stage="ingress",
        event="free_claude_code.api.cli.stop_via_messaging_workflow",
        source="api",
        cancelled_nodes=count,
    )
    logger.info("STOP_CLI: source=messaging_workflow cancelled_count={}", count)
    return {"status": "stopped", "cancelled_count": count}
