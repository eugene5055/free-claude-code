"""Gemini JSON and SSE execution through the shared provider application."""

from collections.abc import Mapping, Sequence

from fastapi.responses import JSONResponse, Response
from loguru import logger

from free_claude_code.api.request_errors import (
    log_unexpected_api_exception,
    ordinary_application_error_response,
)
from free_claude_code.api.response_streams import (
    ManagedStreamingResponse,
    gemini_sse_streaming_response,
    terminal_execution_error_response,
    trace_terminal_execution_error,
)
from free_claude_code.api.stream_frames import decimal_json_response
from free_claude_code.application.errors import ApplicationError
from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.application.ports import ModelInfoLookup, ProviderResolver
from free_claude_code.application.routing import ModelRouter
from free_claude_code.application.web_tools.ports import WebToolsPort
from free_claude_code.application.web_tools.request import extract_query
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic.streaming.decoder import AnthropicSSEDecoder
from free_claude_code.core.diagnostics import safe_exception_message
from free_claude_code.core.failures import find_execution_failure
from free_claude_code.core.gemini import (
    GeminiConversionError,
    GeminiRequest,
    gemini_error_payload,
)
from free_claude_code.core.gemini.models import Content, Part, ToolConfig
from free_claude_code.core.gemini.response import GeminiResponseAccumulator
from free_claude_code.core.web_tools import WebSearchResult


def google_search_query(request: GeminiRequest) -> str:
    """The search query in the latest user turn (Antigravity's WebSearch prompt)."""
    for content in reversed(request.contents):
        if content.role in (None, "user"):
            text = "\n".join(
                part.text
                for part in content.parts
                if part.text is not None and not part.thought
            ).strip()
            if text:
                return extract_query(text)
    return ""


def search_results_text(
    query: str, results: Sequence[WebSearchResult] | None, error: str | None = None
) -> str:
    if error is not None:
        return (
            f"[FCC web search for {query!r} failed: {error}. "
            "Say that the search failed; do not invent results.]"
        )
    if not results:
        return f"[FCC web search found no results for {query!r}.]"
    lines = [
        f"[FCC ran Google Search for you. Results for {query!r}. "
        "Answer from these results and cite their URLs; "
        "say so if they do not answer the question.]"
    ]
    lines.extend(
        f"{index}. {result.title}\n   {result.url}"
        for index, result in enumerate(results, start=1)
    )
    return "\n".join(lines)


def without_google_search(request: GeminiRequest, note: str | None) -> GeminiRequest:
    """Drop googleSearch tools; append FCC's search results to the last user turn."""
    tools = [tool for tool in request.tools if tool.google_search is None]
    update: dict[str, object] = {"tools": tools}
    if not tools:
        update["tool_config"] = ToolConfig()
    if note is not None:
        contents = list(request.contents)
        last = contents[-1]
        if last.role in (None, "user"):
            contents[-1] = last.model_copy(
                update={"parts": [*last.parts, Part(text=note)]}
            )
        else:
            contents.append(Content(role="user", parts=[Part(text=note)]))
        update["contents"] = contents
    return request.model_copy(update=update)


class GeminiHandler:
    def __init__(
        self,
        settings: Settings,
        provider_resolver: ProviderResolver,
        *,
        generation_id: int | None = None,
        request_headers: Mapping[str, str] | None = None,
        model_info_lookup: ModelInfoLookup | None = None,
        web_tools: WebToolsPort | None = None,
    ) -> None:
        self._settings = settings
        self._web_tools = web_tools
        self._router = ModelRouter(settings)
        self._executor = ProviderExecutor(
            provider_resolver,
            generation_id=generation_id,
            request_headers=request_headers,
            model_info_lookup=model_info_lookup,
            log_raw_payloads=settings.log_raw_api_payloads,
        )

    async def create(
        self, request: GeminiRequest, *, streaming: bool, request_id: str
    ) -> Response:
        try:
            if request.requests_google_search:
                request = await self._apply_google_search(request)
            routed = self._router.resolve_gemini_request(request)
            body = self._executor.stream_gemini(
                routed,
                request_id=request_id,
                raw_log_payload=lambda: request.model_dump(
                    mode="json", by_alias=True, exclude_none=True
                ),
            )
            response = await gemini_sse_streaming_response(
                body,
                request_id=request_id,
                pre_start_error_response=lambda error: self._error(error, request_id),
            )
            if streaming or not isinstance(response, ManagedStreamingResponse):
                return response
            aggregate = GeminiResponseAccumulator()
            decoder = AnthropicSSEDecoder()
            try:
                async for chunk in response.body_iterator:
                    text = (
                        chunk
                        if isinstance(chunk, str)
                        else bytes(chunk).decode("utf-8")
                    )
                    for event in decoder.feed(text):
                        aggregate.feed(event.data)
                for event in decoder.finish():
                    aggregate.feed(event.data)
                return decimal_json_response(aggregate.result())
            finally:
                await response.aclose()
        except (Exception, BaseExceptionGroup) as error:
            return self._error(error, request_id)

    async def _apply_google_search(self, request: GeminiRequest) -> GeminiRequest:
        """Run Gemini's built-in search locally, as FCC does for Claude Code."""
        if any(tool.function_declarations for tool in request.tools):
            # Search offered beside client tools: the client tools win.
            return without_google_search(request, None)
        if not self._settings.enable_web_server_tools or self._web_tools is None:
            raise GeminiConversionError(
                "googleSearch needs FCC web tools (ENABLE_WEB_SERVER_TOOLS=true)."
            )
        query = google_search_query(request)
        if not query:
            raise GeminiConversionError("googleSearch requires a user text query")
        try:
            results = await self._web_tools.search(query)
        except Exception as error:
            logger.warning("Gemini googleSearch failed: {}", type(error).__name__)
            note = search_results_text(query, None, safe_exception_message(error))
        else:
            note = search_results_text(query, results)
        return without_google_search(request, note)

    def _error(self, error: BaseException, request_id: str) -> JSONResponse:
        if isinstance(error, ApplicationError):
            return ordinary_application_error_response(
                error, wire_api="gemini", request_id=request_id
            )
        if isinstance(error, GeminiConversionError):
            return JSONResponse(
                status_code=400, content=gemini_error_payload(400, str(error))
            )
        failure = find_execution_failure(error)
        status = failure.status_code if failure is not None else 500
        if failure is None:
            log_unexpected_api_exception(
                self._settings,
                error,
                context="CREATE_GEMINI_ERROR",
                request_id=request_id,
            )
        trace_terminal_execution_error(
            wire_api="gemini",
            request_id=request_id,
            status_code=status,
            error_type=failure.kind.value if failure else "api_error",
            error=error,
        )
        return terminal_execution_error_response(
            status_code=status,
            content=gemini_error_payload(
                status, safe_exception_message(failure or error)
            ),
        )
