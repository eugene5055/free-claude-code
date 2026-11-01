"""Middleware to track token usage from SSE response streams."""

import json
import uuid
from time import monotonic
from typing import Any

from loguru import logger
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from free_claude_code.core.anthropic.stream_contracts import SSEEvent
from free_claude_code.core.anthropic.streaming.decoder import AnthropicSSEDecoder
from free_claude_code.core.request_outcomes import (
    RequestOutcome,
    current_request_outcome,
    record_request_exception,
    record_request_failure,
)
from free_claude_code.application.usage_pricing import MODEL_PRICING
from free_claude_code.application.usage_report import UsageDatabase


# Claude Code sends X-Claude-Code-Session-Id; the rest are SDK/trace variants.
_SESSION_HEADERS = frozenset(
    {
        "x-claude-code-session-id",
        "anthropic-session-id",
        "x-anthropic-session-id",
        "claude-session-id",
        "x-claude-session-id",
    }
)


class UsageTrackingMiddleware:
    """Track token usage from Anthropic SSE response streams."""

    def __init__(self, app: ASGIApp, database: UsageDatabase) -> None:
        self._app = app
        self._database = database

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") not in {"/v1/messages", "/v1/responses"}
        ):
            await self._app(scope, receive, send)
            return

        started = monotonic()
        # Share the outcome RequestOutcomeMiddleware already set, so routing
        # fields land in both places instead of shadowing the outer log.
        existing = current_request_outcome.get()
        outcome = existing if existing is not None else RequestOutcome()
        status_code: int | None = None
        completed = False
        decoder: AnthropicSSEDecoder | None = None
        request_id = scope.get("headers", [])
        # Extract request_id from headers
        request_id_str = None
        session_id: str | None = None
        for k, v in scope.get("headers", []):
            name = k.decode().lower()
            if name == "x-request-id":
                request_id_str = v.decode()
            elif name in _SESSION_HEADERS and not session_id:
                session_id = v.decode() or None
        if not request_id_str:
            request_id_str = str(uuid.uuid4())

        # Track token usage from stream
        usage_data = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }

        token = current_request_outcome.set(outcome)
        try:
            async def send_wrapper(message: Message) -> None:
                nonlocal status_code, completed, decoder, usage_data
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                    content_type = ""
                    for k, v in message.get("headers", []):
                        if k.decode() == "content-type":
                            content_type = v.decode()
                            break
                    if content_type.startswith("text/event-stream"):
                        decoder = AnthropicSSEDecoder(event_names={"message_start", "message_delta", "message_stop", "error"})
                    await send(message)
                elif message["type"] == "http.response.body":
                    body = message.get("body", b"")
                    more_body = message.get("more_body", False)
                    completed = not more_body

                    if decoder is not None and body:
                        try:
                            text = body.decode("utf-8", errors="replace")
                            for event in decoder.feed(text):
                                self._process_sse_event(event, usage_data)
                            if completed:
                                for event in decoder.finish():
                                    self._process_sse_event(event, usage_data)
                        except Exception as e:
                            logger.debug(f"Failed to parse SSE for usage tracking: {e}")

                    await send(message)
                else:
                    await send(message)

            await self._app(scope, receive, send_wrapper)
        except Exception as error:
            if not completed and status_code is None:
                status_code = 500
            record_request_exception(error)
            raise
        finally:
            current_request_outcome.reset(token)

            duration_ms = round((monotonic() - started) * 1000, 2)
            wire_api = "responses" if scope.get("path") == "/v1/responses" else "messages"
            provider_id = outcome.provider_id or "unknown"
            model = outcome.model or "unknown"
            failure_reason = outcome.failure_reason

            if failure_reason is not None or (status_code is not None and status_code >= 400):
                result = "failure"
                error_type = failure_reason or f"http_{status_code}"
            else:
                result = "success"
                error_type = None

            # Store usage data in database
            try:
                await self._store_usage(
                    request_id=request_id_str,
                    session_id=session_id,
                    provider_id=provider_id,
                    model=model,
                    wire_api=wire_api,
                    usage=usage_data,
                    duration_ms=duration_ms,
                    status_code=status_code,
                    error_type=error_type,
                )
            except Exception as e:
                logger.warning(f"Failed to store usage data: {e}")

    def _process_sse_event(self, event: SSEEvent, usage_data: dict[str, int]) -> None:
        """Process SSE event to extract token usage."""
        kind = event.event or event.data.get("type")
        if kind == "message_delta":
            usage = event.data.get("usage")
            if isinstance(usage, dict):
                # Update usage from message_delta
                if "input_tokens" in usage:
                    usage_data["input_tokens"] = usage["input_tokens"]
                if "output_tokens" in usage:
                    usage_data["output_tokens"] = usage["output_tokens"]
                if "cache_read_input_tokens" in usage:
                    usage_data["cache_read_tokens"] = usage["cache_read_input_tokens"]
                if "cache_creation_input_tokens" in usage:
                    usage_data["cache_write_tokens"] = usage["cache_creation_input_tokens"]
        elif kind == "message_start":
            message = event.data.get("message")
            if isinstance(message, dict):
                usage = message.get("usage")
                if isinstance(usage, dict):
                    if "input_tokens" in usage:
                        usage_data["input_tokens"] = usage["input_tokens"]
                    if "output_tokens" in usage:
                        usage_data["output_tokens"] = usage["output_tokens"]
                    if "cache_read_input_tokens" in usage:
                        usage_data["cache_read_tokens"] = usage["cache_read_input_tokens"]
                    if "cache_creation_input_tokens" in usage:
                        usage_data["cache_write_tokens"] = usage["cache_creation_input_tokens"]

    async def _store_usage(
        self,
        request_id: str,
        session_id: str | None,
        provider_id: str,
        model: str,
        wire_api: str,
        usage: dict[str, int],
        duration_ms: float,
        status_code: int | None,
        error_type: str | None,
    ) -> None:
        """Store usage data in database."""
        # Calculate cost based on model pricing
        cost_usd = self._calculate_cost(model, usage)

        def operation(conn):
            conn.execute(
                """
                INSERT OR REPLACE INTO request_usage (
                    request_id, session_id, provider_id, model, wire_api,
                    input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
                    cost_usd, duration_ms, status_code, error_type, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%s', 'now'))
                """,
                (
                    request_id,
                    session_id,
                    provider_id,
                    model,
                    wire_api,
                    usage.get("input_tokens", 0),
                    usage.get("output_tokens", 0),
                    usage.get("cache_read_tokens", 0),
                    usage.get("cache_write_tokens", 0),
                    cost_usd,
                    duration_ms,
                    status_code,
                    error_type,
                ),
            )

        # The shared database opens lazily (Code/messaging start it); start()
        # is idempotent, so make sure it is open before the first write.
        await self._database.start()
        await self._database.run(operation, write=True)

    def _calculate_cost(self, model: str, usage: dict[str, int]) -> float:
        """Calculate cost based on model pricing."""
        # Try to find matching pricing
        input_price = 0.0
        output_price = 0.0
        for key, (inp, out) in MODEL_PRICING.items():
            if key in model.lower():
                input_price = inp
                output_price = out
                break

        if input_price == 0 and output_price == 0:
            # Default pricing
            input_price = 3.00
            output_price = 15.00

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cache_read = usage.get("cache_read_tokens", 0)
        cache_write = usage.get("cache_write_tokens", 0)

        # Cache reads are typically cheaper (e.g., 10% of input price)
        # Cache writes are typically more expensive (e.g., 25% of input price)
        cost = (
            (input_tokens - cache_read - cache_write) * input_price / 1_000_000
            + output_tokens * output_price / 1_000_000
            + cache_read * input_price * 0.1 / 1_000_000
            + cache_write * input_price * 0.25 / 1_000_000
        )
        return round(cost, 6)
