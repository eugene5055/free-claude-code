"""Custom endpoint resource ownership using FCC's existing wire transports."""

import sqlite3
from collections.abc import AsyncIterator, Mapping

import httpx
import httpx2
from loguru import logger

from free_claude_code.application import context_vault
from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.custom_providers import CustomProviderDefinition
from free_claude_code.core.anthropic import (
    MessagesRequest,
    ReasoningReplayMode,
    get_token_count,
)
from free_claude_code.core.chat_completions.request import ChatRequest
from free_claude_code.core.gemini import GeminiRequest
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    responses_reasoning_policy,
)
from free_claude_code.core.reasoning import (
    DEFAULT_REASONING_POLICY,
    ReasoningEffort,
    ReasoningPolicy,
)
from free_claude_code.core.stream_recovery import ContinuationSeed
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderOperationKind,
)
from free_claude_code.providers.anthropic_messages.discovery import (
    list_messages_models,
    messages_model_info,
)
from free_claude_code.providers.anthropic_messages.request_policy import (
    MessagesModelCapabilities,
)
from free_claude_code.providers.anthropic_messages.transport import (
    AnthropicMessagesTransport,
)
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.endpoint_types import HttpEndpoint
from free_claude_code.providers.failure_policy import (
    context_window_exceeded_provider_failure,
)
from free_claude_code.providers.history_negotiation import HistoryModeCache
from free_claude_code.providers.model_listing import (
    extract_openai_model_infos,
    model_infos_from_ids,
)
from free_claude_code.providers.openai_chat import (
    NO_REASONING,
    ChatTemplateReasoning,
    NamedEffortReasoning,
    OpenAIChatBehavior,
    OpenAIChatProfile,
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
    ReasoningEncoder,
    ReasoningObject,
    ThinkingObjectReasoning,
    create_chat_client,
)
from free_claude_code.providers.openai_client import OpenAIRequestClient
from free_claude_code.providers.openai_responses import OpenAIResponsesTransport


def _reasoning_encoder(name: str) -> ReasoningEncoder:
    efforts = tuple(
        (effort, "xhigh" if effort is ReasoningEffort.MAX else effort.value)
        for effort in ReasoningEffort
    )
    limited = tuple(
        (
            effort,
            "low"
            if effort is ReasoningEffort.MINIMAL
            else "high"
            if effort in {ReasoningEffort.XHIGH, ReasoningEffort.MAX}
            else effort.value,
        )
        for effort in ReasoningEffort
    )
    return {
        "provider_default": NO_REASONING,
        "openai_effort": NamedEffortReasoning(
            efforts, disabled_value="none", enabled_value="medium"
        ),
        "limited_effort": NamedEffortReasoning(limited, enabled_value="medium"),
        "reasoning_object": ReasoningObject(limited),
        "thinking": ThinkingObjectReasoning({"type": "enabled"}, {"type": "disabled"}),
        "chat_template": ChatTemplateReasoning("enable_thinking"),
    }[name]


async def _omit_cookies(request: httpx.Request) -> None:
    request.headers.pop("cookie", None)


_CONTEXT_ESTIMATE_FACTOR_PCT = 115
_CONTEXT_REPLY_RESERVE_CAP = 16384


class CustomProvider(BaseProvider):
    def __init__(
        self,
        config: ProviderConfig,
        *,
        definition: CustomProviderDefinition,
        admission: ProviderAdmissionController,
        history_cache: HistoryModeCache | None = None,
        openai_transport: httpx2.AsyncBaseTransport | None = None,
        messages_transport: httpx.AsyncBaseTransport | None = None,
    ):
        super().__init__(config, history_cache=history_cache)
        self._definition = definition
        self._admission = admission
        self._snapshot = HttpEndpoint(
            base_url=config.base_url, api_key=config.api_key, headers={}
        )
        self._openai_transport = openai_transport
        self._sdk = None
        self._http = None
        self._chat = None
        self._responses = None
        self._messages = None
        if definition.api_format == "anthropic_messages":
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    config.http_read_timeout,
                    connect=config.http_connect_timeout,
                    write=config.http_write_timeout,
                ),
                transport=messages_transport,
                follow_redirects=False,
                event_hooks={"request": [_omit_cookies]},
            )
            mode = definition.reasoning_format
            capabilities = MessagesModelCapabilities(
                adaptive_thinking="unsupported"
                if mode == "messages_manual"
                else "required"
                if mode == "messages_adaptive"
                else None,
                supports_output_effort=False
                if mode == "messages_manual"
                else True
                if mode == "messages_adaptive"
                else None,
                supported_efforts=("low", "medium", "high", "xhigh", "max")
                if mode == "messages_adaptive"
                else None,
            )
            self._messages = AnthropicMessagesTransport(
                client=self._http,
                admission=admission,
                history_cache=self._history_cache,
                provider_name=definition.provider_id,
                replay_scope=definition.provider_id,
                read_timeout_s=config.http_read_timeout,
                capabilities=capabilities,
            )
        else:

            async def credential() -> str:
                return config.api_key or ""

            self._sdk = create_chat_client(
                config,
                base_url=config.base_url,
                provider_name=definition.provider_id,
                api_key_provider=credential,
            )
            if definition.api_format == "openai_chat":
                profile = OpenAIChatProfile(
                    OpenAIChatRequestPolicy(
                        definition.provider_id,
                        ReasoningReplayMode(definition.reasoning_history_format),
                    ),
                    _reasoning_encoder(definition.reasoning_format),
                    reasoning_delta_fallback_field="reasoning",
                )
                self._chat = OpenAIChatTransport(
                    client=self._sdk,
                    admission=admission,
                    history_cache=self._history_cache,
                    behavior=OpenAIChatBehavior(profile),
                    read_timeout_s=config.http_read_timeout,
                    log_raw_sse_events=config.log_raw_sse_events,
                    log_api_error_tracebacks=config.log_api_error_tracebacks,
                    endpoint_transport=openai_transport,
                )
            else:
                self._responses = OpenAIResponsesTransport(
                    client=self._sdk,
                    admission=admission,
                    history_cache=self._history_cache,
                    provider_name=definition.provider_id,
                    read_timeout_s=config.http_read_timeout,
                    log_raw_sse_events=config.log_raw_sse_events,
                    endpoint_transport=openai_transport,
                )

    async def endpoint(self, *, force_refresh: bool = False) -> HttpEndpoint:
        return self._snapshot

    async def cleanup(self) -> None:
        if self._sdk is not None:
            await self._sdk.close()
        if self._http is not None:
            await self._http.aclose()
        if self._openai_transport is not None:
            await self._openai_transport.aclose()

    def _check_context_window(self, input_tokens: int, max_tokens: int | None) -> None:
        """Refuse a prompt the served window cannot hold, before it reaches the server.

        input_tokens is a cl100k estimate, which undercounts local tokenizers (~5% for Qwen), so it is scaled
        up; room is kept for the reply. The wording matches Anthropic's, which Claude Code parses to compact.
        """
        limit = self._context_limit(max_tokens)
        if limit is None or input_tokens <= 0:
            return
        needed = -(-input_tokens * _CONTEXT_ESTIMATE_FACTOR_PCT // 100)
        if needed > limit:
            raise context_window_exceeded_provider_failure(
                f"prompt is too long: {needed} tokens > {limit} maximum"
            )

    def _context_limit(self, max_tokens: int | None) -> int | None:
        """Estimated prompt tokens the served window holds, with room kept for the reply."""
        window = self._definition.context_window_tokens
        if window is None:
            return None
        return window - min(max_tokens or _CONTEXT_REPLY_RESERVE_CAP, max(4096, window // 8))

    def _reasoning(self, reasoning: ReasoningPolicy) -> ReasoningPolicy:
        mode = self._definition.reasoning_format
        if mode == "provider_default":
            return DEFAULT_REASONING_POLICY
        if reasoning.budget_tokens is not None and mode in {
            "openai_effort",
            "limited_effort",
            "native_responses",
            "messages_adaptive",
        }:
            raise InvalidRequestError(
                "The selected reasoning format cannot represent an exact token budget"
            )
        return reasoning

    def stream_messages(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
        continuation: ContinuationSeed | None = None,
    ) -> AsyncIterator[str]:
        """Stream a reply. With a declared window, nothing the conversation held is lost (context_vault): every
        request is archived; one no window holds is fitted by the roomiest sibling (FIT) instead of refused, and
        after Claude Code compacted, the model can recall what it dropped."""
        limit = self._context_limit(request.max_tokens)
        options = {
            "request_id": request_id,
            "response_model": response_model,
            "reasoning": reasoning,
            "model_info": model_info,
        }
        if limit is None:
            return self._stream_raw(request, input_tokens, continuation=continuation, **options)
        fitting = context_vault.FIT.get()
        if not fitting:
            self._check_context_window(input_tokens, request.max_tokens)
        vault = context_vault.default_vault()
        session = context_vault.session_key(request, request_headers)
        try:
            present = vault.archive(session, request)
            recalling = fitting or context_vault.needs_vault(request, vault, session, present)
        except sqlite3.Error as error:
            logger.warning("Context vault unavailable: {}", error)
            if fitting:
                raise context_window_exceeded_provider_failure(
                    f"prompt is too long: {input_tokens} tokens > {limit} maximum (context vault unavailable)"
                ) from error
            recalling = False
        if not recalling:
            return self._stream_raw(request, input_tokens, continuation=continuation, **options)

        def open_round(prepared: MessagesRequest, number: int) -> AsyncIterator[str]:
            tokens = get_token_count(prepared.messages, prepared.system, prepared.tools)
            return self._stream_raw(
                prepared, tokens, continuation=continuation if number == 1 else None, **options
            )

        return context_vault.recall_stream(
            request, vault=vault, session=session, limit=limit, open_round=open_round
        )

    def _stream_raw(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        model_info: ProviderModelInfo | None = None,
        continuation: ContinuationSeed | None = None,
    ) -> AsyncIterator[str]:
        reasoning = self._reasoning(reasoning)
        if (
            self._definition.reasoning_format == "provider_default"
            and self._messages is None
        ):
            request = request.model_copy(
                update={
                    "thinking": None,
                    "output_config": {
                        key: value
                        for key, value in (request.output_config or {}).items()
                        if key != "effort"
                    },
                }
            )
        if self._messages is not None:
            return self._messages.stream_messages(
                request,
                endpoint_context=self,
                request_id=request_id,
                response_model=response_model,
                reasoning=reasoning,
                model_info=model_info,
                preserve_native_controls=self._definition.reasoning_format
                == "provider_default",
                continuation=continuation,
            )
        transport = self._chat if self._chat is not None else self._responses
        assert transport is not None
        return transport.stream_messages(
            request,
            input_tokens=input_tokens,
            endpoint_context=self,
            request_id=request_id,
            response_model=response_model or request.model,
            reasoning=reasoning,
            model_info=model_info,
            continuation=continuation,
        )

    def stream_chat(
        self,
        request: ChatRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
        continuation: ContinuationSeed | None = None,
    ) -> AsyncIterator[str]:
        reasoning = self._reasoning(reasoning)
        if self._messages is not None:
            return self._messages.stream_chat(
                request,
                endpoint_context=self,
                request_id=request_id,
                response_model=response_model,
                reasoning=reasoning,
                model_info=model_info,
                continuation=continuation,
            )
        transport = self._chat if self._chat is not None else self._responses
        assert transport is not None
        return transport.stream_chat(
            request,
            input_tokens=input_tokens,
            endpoint_context=self,
            request_id=request_id,
            response_model=response_model or request.model,
            reasoning=reasoning,
            continuation=continuation,
        )

    def stream_gemini(
        self,
        request: GeminiRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
        continuation: ContinuationSeed | None = None,
    ) -> AsyncIterator[str]:
        reasoning = self._reasoning(reasoning)
        if self._messages is not None:
            return self._messages.stream_gemini(
                request,
                endpoint_context=self,
                request_id=request_id,
                response_model=response_model,
                reasoning=reasoning,
                model_info=model_info,
                continuation=continuation,
            )
        transport = self._chat if self._chat is not None else self._responses
        assert transport is not None
        return transport.stream_gemini(
            request,
            input_tokens=input_tokens,
            endpoint_context=self,
            request_id=request_id,
            response_model=response_model or request.model,
            reasoning=reasoning,
            continuation=continuation,
        )

    def stream_responses(
        self,
        request: OpenAIResponsesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
        continuation: ContinuationSeed | None = None,
    ) -> AsyncIterator[str]:
        reasoning = (
            responses_reasoning_policy(request.reasoning)
            if self._definition.reasoning_format == "provider_default"
            and self._responses is not None
            else self._reasoning(reasoning)
        )
        if (
            self._definition.reasoning_format == "provider_default"
            and self._responses is None
        ):
            request = request.model_copy(
                update={
                    "reasoning": {
                        key: value
                        for key, value in (request.reasoning or {}).items()
                        if key != "effort"
                    }
                }
            )
        if self._messages is not None:
            return self._messages.stream_responses(
                request,
                endpoint_context=self,
                request_id=request_id,
                response_model=response_model,
                reasoning=reasoning,
                model_info=model_info,
                continuation=continuation,
            )
        transport = self._chat if self._chat is not None else self._responses
        assert transport is not None
        return transport.stream_responses(
            request,
            input_tokens=input_tokens,
            endpoint_context=self,
            request_id=request_id,
            response_model=response_model or request.model,
            reasoning=reasoning,
            continuation=continuation,
        )

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        if self._definition.model_ids:
            return model_infos_from_ids(self._definition.model_ids)
        if self._sdk is not None:
            binding = OpenAIRequestClient(self._openai_transport)
            try:
                client = binding.for_endpoint(self._sdk, self._snapshot)
                payload = await self._admission.start_execution(model=None).run_call(
                    lambda: client.models.list(extra_headers=binding.openai_headers()),
                    operation_kind=ProviderOperationKind.MODEL_DISCOVERY,
                )
                return extract_openai_model_infos(
                    payload, provider_name=self._definition.provider_id
                )
            finally:
                await binding.aclose()
        assert self._http is not None
        headers = {"anthropic-version": "2023-06-01"}
        if self._config.api_key:
            headers["x-api-key"] = self._config.api_key
        records = await list_messages_models(
            self._http, self._admission, base_url=self._config.base_url, headers=headers
        )
        return frozenset(
            messages_model_info(item, self._definition.provider_id) for item in records
        )
