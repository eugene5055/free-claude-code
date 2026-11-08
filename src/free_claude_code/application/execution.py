"""Provider execution shared by inbound API adapters."""

import asyncio
import json
import re
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from time import monotonic
from types import MappingProxyType
from typing import Literal

from loguru import logger

from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.core.anthropic import (
    anthropic_request_snapshot,
    get_token_count,
)
from free_claude_code.core.anthropic.tokens import TokenCounter
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.gemini.tokens import estimate_gemini_input_tokens
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    estimate_responses_input_tokens,
)
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.application import kimi_plan, model_usability, provider_health
from free_claude_code.application.provider_ratings import record_outcome, _kind_for_model
from free_claude_code.core.request_outcomes import record_request_route
from free_claude_code.core.stream_delivery import current_stream_delivery
from free_claude_code.core.stream_recovery import (
    ContinuationSeed,
    RecoveryAction,
    assess_recovery,
)
from free_claude_code.core.token_estimation import estimate_text_tokens
from free_claude_code.core.trace import (
    close_stream_input,
    trace_event,
    traced_async_stream,
)

from .ports import ModelInfoLookup, ProviderResolver
from .reasoning import (
    client_chat_reasoning_policy,
    client_reasoning_policy,
    client_responses_reasoning_policy,
    prepare_fallback_request,
    resolve_fallback_reasoning,
)
from .routing import (
    ProviderModelCandidate,
    ProviderModelTarget,
    ResolvedModelRoute,
    RoutedChatRequest,
    RoutedGeminiRequest,
    RoutedMessagesRequest,
    RoutedNativeMessagesRequest,
    RoutedResponsesRequest,
)

ResponsesTokenCounter = Callable[[OpenAIResponsesRequest], int]
WireApi = Literal["messages", "responses", "gemini", "chat"]
CandidateStreamOpener = Callable[
    [int, ProviderModelCandidate, ContinuationSeed | None],
    Awaitable[AsyncIterator[str]],
]


class ProviderExecutor:
    """Resolve a provider and execute one routed Anthropic Messages stream."""

    def __init__(
        self,
        provider_resolver: ProviderResolver,
        *,
        token_counter: TokenCounter = get_token_count,
        responses_token_counter: ResponsesTokenCounter = estimate_responses_input_tokens,
        generation_id: int | None = None,
        log_raw_payloads: bool = False,
        request_headers: Mapping[str, str] | None = None,
        model_info_lookup: ModelInfoLookup | None = None,
        local_providers: frozenset[str] = frozenset(),
    ) -> None:
        self._provider_resolver = provider_resolver
        # Providers with a local base URL (provider_locality.local_provider_ids): a request for one of them never
        # gets automatic cloud fallbacks.
        self._local_providers = local_providers
        self._model_info_lookup = model_info_lookup or (lambda _provider, _model: None)
        self._token_counter = token_counter
        self._responses_token_counter = responses_token_counter
        self._generation_id = generation_id
        self._log_raw_payloads = log_raw_payloads
        self._request_headers = MappingProxyType(dict(request_headers or {}))

    def _trace_fallback_started(
        self,
        *,
        request_id: str,
        wire_api: WireApi,
        failed: ProviderModelTarget,
        selected: ProviderModelTarget,
        failure: ExecutionFailure,
        candidate_index: int,
        candidate_count: int,
        recovery_action: str,
        recovery_reason: str | None,
    ) -> None:
        fields: dict[str, object] = {
            "stage": "execution",
            "event": "free_claude_code.model_fallback.started",
            "source": "application",
            "request_id": request_id,
            "wire_api": wire_api,
            "from_provider_model_ref": failed.provider_model_ref,
            "to_provider_model_ref": selected.provider_model_ref,
            "from_api_key_number": failed.api_key_number,
            "to_api_key_number": selected.api_key_number,
            "candidate_index": candidate_index,
            "candidate_count": candidate_count,
            "failure_kind": failure.kind.value,
            "status_code": failure.status_code,
            "provider_retryable": failure.retryable,
            "recovery_action": recovery_action,
            "recovery_reason": recovery_reason,
        }
        if self._generation_id is not None:
            fields["generation_id"] = self._generation_id
        trace_event(**fields)
        logger.info(
            "Model fallback: request_id={} from={} to={} candidate={}/{} "
            "failure_kind={} status_code={}",
            request_id,
            failed.provider_model_ref,
            selected.provider_model_ref,
            candidate_index,
            candidate_count,
            failure.kind.value,
            failure.status_code,
        )

    def _trace_fallback_selected(
        self,
        *,
        request_id: str,
        wire_api: WireApi,
        selected: ProviderModelTarget,
        candidate_index: int,
        candidate_count: int,
    ) -> None:
        fields: dict[str, object] = {
            "stage": "execution",
            "event": "free_claude_code.model_fallback.selected",
            "source": "application",
            "request_id": request_id,
            "wire_api": wire_api,
            "selected_provider_model_ref": selected.provider_model_ref,
            "api_key_number": selected.api_key_number,
            "candidate_index": candidate_index,
            "candidate_count": candidate_count,
        }
        if self._generation_id is not None:
            fields["generation_id"] = self._generation_id
        trace_event(**fields)

    def stream_native_messages(
        self,
        routed: RoutedNativeMessagesRequest,
        *,
        request_id: str,
    ) -> AsyncIterator[str]:
        native_reasoning = ReasoningPolicy.provider_default()

        async def open_candidate(
            index: int,
            candidate: ProviderModelCandidate,
            continuation: ContinuationSeed | None,
        ) -> AsyncIterator[str]:
            target = candidate.target
            provider = await self._provider_resolver(
                target.provider_id, target.api_key_number
            )
            return provider.stream_native_messages(
                routed.request.with_model(target.provider_model),
                request_id=request_id,
                response_model=routed.resolved.original_model,
                request_headers=self._request_headers,
                reasoning_override=(
                    None
                    if candidate.reasoning_preference
                    in {ReasoningPreference.INHERIT, ReasoningPreference.CLIENT}
                    else resolve_fallback_reasoning(
                        native_reasoning,
                        candidate.reasoning_preference,
                        client=native_reasoning,
                    )
                ),
                continuation=continuation,
            )

        messages = routed.request.body["messages"]
        assert isinstance(messages, list)
        return self._stream_candidates(
            resolved=routed.resolved,
            reasoning=native_reasoning,
            wire_api="messages",
            raw_log_label="FULL_NATIVE_MESSAGES_PAYLOAD",
            raw_log_payload=lambda: routed.request.body,
            request_snapshot=lambda: {
                "model": routed.request.model,
                "message_count": len(messages),
                "contract": "native",
            },
            ingress_count_name="message_count",
            ingress_count=len(messages),
            request_id=request_id,
            open_candidate=open_candidate,
            auto_fallbacks=False,
        )

    def stream_messages(
        self,
        routed: RoutedMessagesRequest,
        *,
        raw_log_payload: Callable[[], object],
        request_id: str,
    ) -> AsyncIterator[str]:
        """Execute one Anthropic Messages request."""

        primary_request = routed.request.model_copy(deep=True)
        client_reasoning = client_reasoning_policy(routed.request)
        input_tokens = self._token_counter(
            routed.request.messages,
            routed.request.system,
            routed.request.tools,
        )

        async def open_candidate(
            index: int,
            candidate: ProviderModelCandidate,
            continuation: ContinuationSeed | None,
        ) -> AsyncIterator[str]:
            target = candidate.target
            provider = await self._provider_resolver(
                target.provider_id, target.api_key_number
            )
            request = (
                primary_request
                if index == 0
                else prepare_fallback_request(
                    routed.request,
                    model=target.provider_model,
                    preference=candidate.reasoning_preference,
                )
            )
            return provider.stream_messages(
                request,
                input_tokens=input_tokens,
                request_id=request_id,
                response_model=routed.resolved.original_model,
                reasoning=resolve_fallback_reasoning(
                    routed.reasoning,
                    candidate.reasoning_preference,
                    client=client_reasoning,
                ),
                model_info=self._model_info_lookup(
                    target.provider_id, target.provider_model
                ),
                request_headers=self._request_headers,
                continuation=continuation,
            )

        return self._stream_candidates(
            resolved=routed.resolved,
            reasoning=routed.reasoning,
            wire_api="messages",
            raw_log_label="FULL_PAYLOAD",
            raw_log_payload=raw_log_payload,
            request_snapshot=lambda: anthropic_request_snapshot(routed.request),
            ingress_count_name="message_count",
            ingress_count=len(routed.request.messages),
            request_id=request_id,
            open_candidate=open_candidate,
        )

    def stream_responses(
        self,
        routed: RoutedResponsesRequest,
        *,
        raw_log_payload: Callable[[], object],
        request_id: str,
    ) -> AsyncIterator[str]:
        """Execute one native OpenAI Responses request."""

        primary_request = routed.request.model_copy(deep=True)
        client_reasoning = client_responses_reasoning_policy(routed.request)
        input_tokens = self._responses_token_counter(routed.request)

        async def open_candidate(
            index: int,
            candidate: ProviderModelCandidate,
            continuation: ContinuationSeed | None,
        ) -> AsyncIterator[str]:
            target = candidate.target
            provider = await self._provider_resolver(
                target.provider_id, target.api_key_number
            )
            request = (
                primary_request
                if index == 0
                else prepare_fallback_request(
                    routed.request,
                    model=target.provider_model,
                    preference=candidate.reasoning_preference,
                )
            )
            return provider.stream_responses(
                request,
                input_tokens=input_tokens,
                request_id=request_id,
                response_model=routed.resolved.original_model,
                reasoning=resolve_fallback_reasoning(
                    routed.reasoning,
                    candidate.reasoning_preference,
                    client=client_reasoning,
                ),
                request_headers=self._request_headers,
                continuation=continuation,
                model_info=self._model_info_lookup(
                    target.provider_id, target.provider_model
                ),
            )

        raw_input = routed.request.input
        input_item_count = (
            len(raw_input)
            if isinstance(raw_input, list)
            else int(raw_input is not None)
        )
        return self._stream_candidates(
            resolved=routed.resolved,
            reasoning=routed.reasoning,
            wire_api="responses",
            raw_log_label="FULL_RESPONSES_PAYLOAD",
            raw_log_payload=raw_log_payload,
            request_snapshot=lambda: {
                "model": routed.request.model,
                "input_item_count": input_item_count,
                "tool_count": len(routed.request.tools or ()),
            },
            ingress_count_name="input_item_count",
            ingress_count=input_item_count,
            request_id=request_id,
            open_candidate=open_candidate,
        )

    def stream_gemini(
        self,
        routed: RoutedGeminiRequest,
        *,
        raw_log_payload: Callable[[], object],
        request_id: str,
    ) -> AsyncIterator[str]:
        client_reasoning = routed.request.reasoning_policy()
        input_tokens = estimate_gemini_input_tokens(routed.request)

        async def open_candidate(
            index: int,
            candidate: ProviderModelCandidate,
            continuation: ContinuationSeed | None,
        ) -> AsyncIterator[str]:
            target = candidate.target
            provider = await self._provider_resolver(
                target.provider_id, target.api_key_number
            )
            request = prepare_fallback_request(
                routed.request,
                model=target.provider_model,
                preference=candidate.reasoning_preference,
            )
            return provider.stream_gemini(
                request,
                input_tokens=input_tokens,
                request_id=request_id,
                response_model=routed.resolved.original_model,
                reasoning=resolve_fallback_reasoning(
                    routed.reasoning,
                    candidate.reasoning_preference,
                    client=client_reasoning,
                ),
                request_headers=self._request_headers,
                continuation=continuation,
                model_info=self._model_info_lookup(
                    target.provider_id, target.provider_model
                ),
            )

        return self._stream_candidates(
            resolved=routed.resolved,
            reasoning=routed.reasoning,
            wire_api="gemini",
            raw_log_label="FULL_GEMINI_PAYLOAD",
            raw_log_payload=raw_log_payload,
            request_snapshot=lambda: {
                "model": routed.request.model,
                "content_count": len(routed.request.contents),
            },
            ingress_count_name="content_count",
            ingress_count=len(routed.request.contents),
            request_id=request_id,
            open_candidate=open_candidate,
        )

    def stream_chat(
        self,
        routed: RoutedChatRequest,
        *,
        raw_log_payload: Callable[[], object],
        request_id: str,
    ) -> AsyncIterator[str]:
        client_reasoning = client_chat_reasoning_policy(routed.request)
        input_tokens = sum(
            estimate_text_tokens(json.dumps(message)[:1_000_000])
            for message in routed.request.messages
        )

        async def open_candidate(
            index: int,
            candidate: ProviderModelCandidate,
            continuation: ContinuationSeed | None,
        ) -> AsyncIterator[str]:
            target = candidate.target
            provider = await self._provider_resolver(
                target.provider_id, target.api_key_number
            )
            request = prepare_fallback_request(
                routed.request,
                model=target.provider_model,
                preference=candidate.reasoning_preference,
            )
            return provider.stream_chat(
                request,
                input_tokens=input_tokens,
                request_id=request_id,
                response_model=routed.resolved.original_model,
                reasoning=resolve_fallback_reasoning(
                    routed.reasoning,
                    candidate.reasoning_preference,
                    client=client_reasoning,
                ),
                request_headers=self._request_headers,
                continuation=continuation,
                model_info=self._model_info_lookup(
                    target.provider_id, target.provider_model
                ),
            )

        return self._stream_candidates(
            resolved=routed.resolved,
            reasoning=routed.reasoning,
            wire_api="chat",
            raw_log_label="FULL_CHAT_PAYLOAD",
            raw_log_payload=raw_log_payload,
            request_snapshot=lambda: {
                "model": routed.request.model,
                "message_count": len(routed.request.messages),
            },
            ingress_count_name="message_count",
            ingress_count=len(routed.request.messages),
            request_id=request_id,
            open_candidate=open_candidate,
        )

    def _stream_candidates(
        self,
        *,
        resolved: ResolvedModelRoute,
        reasoning: ReasoningPolicy,
        wire_api: WireApi,
        raw_log_label: str,
        raw_log_payload: Callable[[], object],
        request_snapshot: Callable[[], dict[str, object]],
        ingress_count_name: str,
        ingress_count: int,
        request_id: str,
        open_candidate: CandidateStreamOpener,
        auto_fallbacks: bool = True,
    ) -> AsyncIterator[str]:
        """Start and consume candidates through one protocol-blind lifecycle."""

        primary = resolved.primary
        # (open index, candidate): the index tells open_candidate whether it is the primary (0) or a fallback, so it
        # stays with the candidate when rate-limited providers move to the end (provider_health). A grouped model
        # name only ever switches providers of the same model, so it gets no automatic fallbacks.
        candidates, automatic, spare = _ordered_candidates(
            (ProviderModelCandidate(primary), *resolved.fallbacks),
            auto_fallbacks=auto_fallbacks and not resolved.same_model_only,
            request_id=request_id,
            local_providers=self._local_providers,
        )
        gateway_model = resolved.original_model
        route_trace: dict[str, object] = {
            "stage": "routing",
            "event": "free_claude_code.api.route.resolved",
            "source": "api",
            "request_id": request_id,
            "provider_id": primary.provider_id,
            "provider_model": primary.provider_model,
            "provider_model_ref": primary.provider_model_ref,
            "fallback_count": len(resolved.fallbacks),
            "gateway_model": gateway_model,
            "reasoning_control": reasoning.control.value,
            "reasoning_effort": (
                reasoning.effort.value if reasoning.effort is not None else None
            ),
            "reasoning_budget_tokens": reasoning.budget_tokens,
        }
        if wire_api != "messages":
            route_trace["wire_api"] = wire_api
        if self._generation_id is not None:
            route_trace["generation_id"] = self._generation_id
        trace_event(**route_trace)

        ingress_trace: dict[str, object] = {
            "stage": "ingress",
            "event": (
                "free_claude_code.api.responses.request.received"
                if wire_api == "responses"
                else "free_claude_code.api.request.received"
            ),
            "source": "api",
            "request_id": request_id,
            ingress_count_name: ingress_count,
        }
        trace_event(
            lambda: {"snapshot": {**request_snapshot(), "model": gateway_model}},
            **ingress_trace,
        )

        if self._log_raw_payloads:
            logger.opt(lazy=True).debug(
                f"{raw_log_label} [{{}}]: {{}}", lambda: request_id, raw_log_payload
            )

        async def provider_body() -> AsyncIterator[str]:
            nonlocal spare
            delivery = current_stream_delivery()
            continuation: ContinuationSeed | None = None
            configured_failure: ExecutionFailure | None = None
            for position, (index, candidate) in enumerate(candidates):
                target = candidate.target
                record_request_route(target.provider_id, target.provider_model)
                candidate_started = monotonic()
                provider_stream: AsyncIterator[str] | None = None
                candidate_committed = False
                candidate_failure: ExecutionFailure | None = None
                try:
                    try:
                        provider_stream = await open_candidate(
                            index, candidate, continuation
                        )
                    except ExecutionFailure as failure:
                        candidate_failure = failure
                    except Exception as error:
                        # An automatic fallback (a model that worked before) whose provider is no longer configured
                        # or cannot start is skipped like a failed candidate, never failing the request.
                        if candidate.target.provider_model_ref not in automatic:
                            raise
                        candidate_failure = ExecutionFailure(
                            kind=FailureKind.UNAVAILABLE,
                            status_code=503,
                            message=f"Automatic fallback unavailable: {type(error).__name__}",
                            retryable=True,
                        )

                    if provider_stream is None and candidate_failure is None:
                        raise TypeError(
                            "provider stream method must return an async iterator"
                        )
                    while provider_stream is not None:
                        try:
                            chunk = await anext(provider_stream)
                        except StopAsyncIteration:
                            break
                        except ExecutionFailure as failure:
                            candidate_failure = failure
                            break
                        if not chunk:
                            await asyncio.sleep(0)
                            continue
                        if not candidate_committed:
                            candidate_committed = True
                            model_usability.record_latency(
                                target.provider_model_ref, monotonic() - candidate_started
                            )
                            if position > 0:
                                self._trace_fallback_selected(
                                    request_id=request_id,
                                    wire_api=wire_api,
                                    selected=target,
                                    candidate_index=position + 1,
                                    candidate_count=len(candidates),
                                )
                        yield chunk
                finally:
                    if provider_stream is not None:
                        active_error = sys.exception()
                        preserved_error = active_error or candidate_failure
                        await close_stream_input(
                            provider_stream,
                            owner="provider_executor",
                            source="api",
                            preserved_error=preserved_error,
                        )

                if candidate_failure is None:
                    record_outcome(
                        target.provider_id, _kind_for_model(target.provider_model), True,
                        monotonic() - candidate_started,
                    )
                    model_usability.record_success(target.provider_model_ref)
                    provider_health.record_success(target.provider_id, target.provider_model)
                    return
                if candidate_failure.kind is not FailureKind.CONTEXT_WINDOW_EXCEEDED:
                    # A prompt larger than the model's window is not the provider's fault.
                    record_outcome(
                        target.provider_id, _kind_for_model(target.provider_model), False,
                        monotonic() - candidate_started,
                    )
                model_usability.record_failure(target.provider_model_ref, candidate_failure)
                if (
                    candidate_failure.kind is FailureKind.RATE_LIMIT
                    or candidate_failure.status_code == 429
                ):
                    provider_health.mark_limited(target.provider_id)
                    plan = kimi_plan.exhausted(target.provider_id)
                    if plan is not None:
                        # Say when Kimi Code is back instead of "retry shortly".
                        candidate_failure = ExecutionFailure(
                            kind=FailureKind.RATE_LIMIT,
                            status_code=429,
                            message=plan[1],
                            retryable=False,
                        )
                    # A rate limit: switch to models of other providers that worked recently (provider_health), tried
                    # before any candidate whose provider is limited. Only here, so other failures behave as before.
                    if spare:
                        _insert_spare(candidates, position, spare, automatic)
                        spare = []
                if target.provider_model_ref not in automatic:
                    configured_failure = _roomiest_failure(configured_failure, candidate_failure)
                recovery_action = "continue" if continuation is not None else "restart"
                recovery_reason = "no_candidate_output"
                context = candidate_failure.stream_context
                if delivery is None or context is None:
                    if candidate_committed:
                        raise candidate_failure
                else:
                    assessment = assess_recovery(
                        content_released=delivery.content_released,
                        prefix=delivery.prefix,
                        source=context.source,
                        retryable=candidate_failure.retryable,
                    )
                    if assessment.action is RecoveryAction.STOP:
                        raise candidate_failure
                    if assessment.action is RecoveryAction.HANDOFF:
                        delivery.begin_continuation(handoff=True)
                        return
                    continuation = assessment.seed
                    recovery_action = assessment.action.value
                    recovery_reason = assessment.reason
                if position + 1 >= len(candidates):
                    # When the automatic fallbacks failed too, report what the configured model said; a prompt no
                    # window holds is refused as too long, so the client compacts and retries.
                    raise configured_failure or candidate_failure
                if delivery is not None and continuation is not None:
                    delivery.begin_continuation()
                next_target = candidates[position + 1][1].target
                self._trace_fallback_started(
                    request_id=request_id,
                    wire_api=wire_api,
                    failed=target,
                    selected=next_target,
                    failure=candidate_failure,
                    candidate_index=position + 2,
                    candidate_count=len(candidates),
                    recovery_action=recovery_action,
                    recovery_reason=recovery_reason,
                )

        stream_trace: dict[str, object] = {
            "request_id": request_id,
            "initial_provider_id": primary.provider_id,
            "gateway_model": gateway_model,
        }
        if self._generation_id is not None:
            stream_trace["generation_id"] = self._generation_id

        return traced_async_stream(
            provider_body(),
            stage="egress",
            source="api",
            complete_event=(
                "free_claude_code.api.responses.stream_completed"
                if wire_api == "responses"
                else "free_claude_code.api.response.stream_completed"
            ),
            interrupted_event=(
                "free_claude_code.api.responses.stream_interrupted"
                if wire_api == "responses"
                else "free_claude_code.api.response.stream_interrupted"
            ),
            chunk_event=None,
            extra=stream_trace,
        )


_CONTEXT_MAXIMUM = re.compile(r"> (\d+) maximum")


def _roomiest_failure(previous: ExecutionFailure | None, failure: ExecutionFailure) -> ExecutionFailure:
    """The failure to report: the latest, except that among "prompt is too long" refusals the largest window wins.

    When no sibling can hold a prompt, Claude Code compacts down to the reported maximum, so reporting the smaller
    window would throw away room the larger one has.
    """
    if (
        previous is None
        or previous.kind is not FailureKind.CONTEXT_WINDOW_EXCEEDED
        or failure.kind is not FailureKind.CONTEXT_WINDOW_EXCEEDED
    ):
        return failure
    return previous if _context_maximum(previous) > _context_maximum(failure) else failure


def _context_maximum(failure: ExecutionFailure) -> int:
    found = _CONTEXT_MAXIMUM.search(failure.message or "")
    return int(found.group(1)) if found else 0


def _ordered_candidates(
    configured: tuple[ProviderModelCandidate, ...],
    *,
    auto_fallbacks: bool,
    request_id: str,
    local_providers: frozenset[str] = frozenset(),
) -> tuple[list[tuple[int, ProviderModelCandidate]], set[str], list[tuple[int, ProviderModelCandidate]]]:
    """The order to try candidates in, the refs of the automatic ones, and the spare automatic ones.

    Without a rate limit this is the configured order, unchanged. Candidates of a provider that is rate limited right
    now go last (soonest reset first), and then models of other providers that recently completed requests
    (provider_health.proven_models, up to three) are tried before them. The spare ones join only once a configured
    candidate is refused with a rate limit during this request (_insert_spare).

    A request whose configured candidates are all local providers (local_providers) only gets local automatic
    fallbacks: a local model never falls back to a cloud model by itself. Cloud requests may still use local ones.
    """
    now = time.time()
    entries = list(enumerate(configured))
    rests = {e[1].target.provider_id: provider_health.limited_until(e[1].target.provider_id, now) for e in entries}
    free = [e for e in entries if rests[e[1].target.provider_id] is None]
    limited = sorted((e for e in entries if rests[e[1].target.provider_id] is not None),
                     key=lambda e: rests[e[1].target.provider_id])
    pool: list[tuple[int, ProviderModelCandidate]] = []
    automatic: set[str] = set()
    if auto_fallbacks:
        known = {candidate.target.provider_model_ref for candidate in configured}
        local_only = bool(configured) and all(
            candidate.target.provider_id in local_providers for candidate in configured
        )
        if local_only:
            proven = provider_health.proven_models(known, now, limit=sys.maxsize)
            kept = [model for model in proven if model[0] in local_providers][: provider_health._AUTO_FALLBACKS]
            skipped = [model for model in proven if model[0] not in local_providers]
            if skipped:
                logger.info(
                    "Local request: request_id={} skips cloud automatic fallbacks {}",
                    request_id,
                    ", ".join(f"{provider_id}/{model}" for provider_id, model in skipped),
                )
        else:
            kept = provider_health.proven_models(known, now)
        for offset, (provider_id, provider_model) in enumerate(kept):
            ref = f"{provider_id}/{provider_model}"
            target = ProviderModelTarget(provider_id=provider_id, provider_model=provider_model, provider_model_ref=ref)
            pool.append((len(entries) + offset, ProviderModelCandidate(target)))
            automatic.add(ref)
    all_local = bool(configured) and all(c.target.provider_id in local_providers for c in configured)
    queued: list[tuple[int, ProviderModelCandidate]] = []
    if not all_local:
        # A busy local queue yields: its fallbacks go after every other free candidate (never the first one asked for).
        def yields(entry: tuple[int, ProviderModelCandidate]) -> bool:
            pid = entry[1].target.provider_id
            return entry[0] != 0 and pid in local_providers and provider_health.busy(pid)

        queued = [e for e in free + pool if yields(e)]
        if queued:
            free = [e for e in free if e not in queued]
            pool = [e for e in pool if e not in queued]
            logger.info(
                "Local queue: request_id={} {} has {} waiting; other providers are tried first",
                request_id,
                queued[0][1].target.provider_model_ref,
                provider_health.queue_depth(queued[0][1].target.provider_id),
            )
    if limited:
        logger.info(
            "Rate limit: request_id={} tries {} first; {} rests until {}",
            request_id,
            (free or pool or queued or limited)[0][1].target.provider_model_ref,
            ", ".join(sorted({e[1].target.provider_id for e in limited})),
            time.strftime("%H:%M:%S", time.localtime(min(rests[e[1].target.provider_id] for e in limited))),
        )
        return free + pool + queued + limited, automatic, []
    return free + [e for e in queued if e[1].target.provider_model_ref not in automatic], automatic, pool + [
        e for e in queued if e[1].target.provider_model_ref in automatic]


def _insert_spare(
    candidates: list[tuple[int, ProviderModelCandidate]],
    position: int,
    spare: list[tuple[int, ProviderModelCandidate]],
    automatic: set[str],
) -> None:
    """Put the spare automatic fallbacks after the current candidate, before any later one of a limited provider."""
    now = time.time()
    at = next((i for i in range(position + 1, len(candidates))
               if provider_health.limited_until(candidates[i][1].target.provider_id, now) is not None), len(candidates))
    candidates[at:at] = [entry for entry in spare if entry[1].target.provider_model_ref in automatic]
