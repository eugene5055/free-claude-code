"""Tool calls missing schema-required arguments are retried once upstream."""

import json
from typing import Any

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.openai_chat import (
    NO_REASONING,
    OpenAIChatBehavior,
    OpenAIChatProfile,
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
)
from free_claude_code.providers.openai_chat.tool_call_repair import (
    required_argument_defect,
)
from tests.providers.request_factory import make_messages_request

pytestmark = pytest.mark.asyncio

JS_SCHEMA = {
    "type": "object",
    "properties": {"code": {"type": "string"}},
    "required": ["code"],
}


def _transport(client: AsyncOpenAI, *, repair: bool = True) -> OpenAIChatTransport:
    return OpenAIChatTransport(
        client=client,
        admission=ProviderAdmissionController(
            provider_name="TEST",
            rate_limit=100,
            rate_window=1,
            max_concurrency=1,
            max_attempts=3,
        ),
        behavior=OpenAIChatBehavior(
            OpenAIChatProfile(
                OpenAIChatRequestPolicy("TEST", ReasoningReplayMode.REASONING_CONTENT),
                NO_REASONING,
            )
        ),
        read_timeout_s=2,
        log_raw_sse_events=False,
        log_api_error_tracebacks=False,
        repair_tool_calls=repair,
    )


def _chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    return {
        "id": "chat_test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _sse(*chunks: dict[str, Any]) -> httpx2.Response:
    text = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    return httpx2.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text=text + "data: [DONE]\n\n",
    )


def _tool_call(arguments: str, *, text: str = "", call_id: str = "call_1"):
    chunks = []
    if text:
        chunks.append(_chunk({"role": "assistant", "content": text}))
    chunks.append(
        _chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "js", "arguments": ""},
                    }
                ]
            }
        )
    )
    chunks.append(
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": arguments}}]})
    )
    chunks.append(_chunk({}, "tool_calls"))
    return _sse(*chunks)


class _Upstream:
    def __init__(self, *responses: httpx2.Response) -> None:
        self._responses = list(responses)
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.bodies.append(json.loads(request.content))
        return self._responses[min(len(self.bodies), len(self._responses)) - 1]


def _client(upstream: _Upstream) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="test",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(upstream)),
    )


def _messages_request():
    return make_messages_request(
        "model",
        thinking={"enabled": False},
        tools=[{"name": "js", "description": "Run JS", "input_schema": JS_SCHEMA}],
    )


async def _messages(upstream: _Upstream, *, repair: bool = True) -> list[dict]:
    async with _client(upstream) as client:
        stream = _transport(client, repair=repair).stream_messages(_messages_request())
        raw = "".join([event async for event in stream])
    return [
        json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: ")
    ]


def _tool_inputs(payloads: list[dict]) -> list[str]:
    inputs: list[str] = []
    for payload in payloads:
        if (
            payload.get("type") == "content_block_start"
            and payload["content_block"]["type"] == "tool_use"
        ):
            inputs.append("")
        delta = payload.get("delta", {})
        if delta.get("type") == "input_json_delta":
            inputs[-1] += delta["partial_json"]
    return inputs


def _text(payloads: list[dict]) -> str:
    return "".join(p.get("delta", {}).get("text", "") for p in payloads)


async def test_empty_arguments_are_retried_once_and_fixed_call_reaches_client():
    upstream = _Upstream(_tool_call("{}"), _tool_call('{"code": "1+1"}'))

    payloads = await _messages(upstream)

    assert len(upstream.bodies) == 2
    note = upstream.bodies[1]["messages"][-1]
    assert note["role"] == "user"
    assert "`js`" in note["content"] and "`code`" in note["content"]
    assert upstream.bodies[1]["messages"][:-1] == upstream.bodies[0]["messages"]
    assert [json.loads(raw) for raw in _tool_inputs(payloads)] == [{"code": "1+1"}]
    assert payloads[-1]["type"] == "message_stop"


async def test_retry_that_is_still_defective_passes_through_unchanged():
    upstream = _Upstream(_tool_call("{}"), _tool_call("{}", call_id="call_2"))

    payloads = await _messages(upstream)

    assert len(upstream.bodies) == 2
    assert [json.loads(raw or "{}") for raw in _tool_inputs(payloads)] == [{}]
    assert payloads[-1]["type"] == "message_stop"


async def test_valid_call_is_not_retried():
    upstream = _Upstream(_tool_call('{"code": "2"}'))

    payloads = await _messages(upstream)

    assert len(upstream.bodies) == 1
    assert [json.loads(raw) for raw in _tool_inputs(payloads)] == [{"code": "2"}]


async def test_repair_can_be_disabled():
    upstream = _Upstream(_tool_call("{}"), _tool_call('{"code": "1"}'))

    payloads = await _messages(upstream, repair=False)

    assert len(upstream.bodies) == 1
    assert [json.loads(raw or "{}") for raw in _tool_inputs(payloads)] == [{}]


async def test_streamed_text_is_kept_in_context_and_not_repeated():
    upstream = _Upstream(
        _tool_call("{}", text="Running it."),
        _tool_call('{"code": "3"}'),
    )

    payloads = await _messages(upstream)

    assert len(upstream.bodies) == 2
    assistant, note = upstream.bodies[1]["messages"][-2:]
    assert (assistant["role"], assistant["content"]) == ("assistant", "Running it.")
    assert note["role"] == "user"
    assert _text(payloads) == "Running it."
    assert [json.loads(raw) for raw in _tool_inputs(payloads)] == [{"code": "3"}]


async def test_failed_retry_releases_the_original_call():
    upstream = _Upstream(
        _tool_call("{}"),
        httpx2.Response(400, json={"error": {"message": "bad request"}}),
    )

    payloads = await _messages(upstream)

    assert len(upstream.bodies) >= 2
    assert [json.loads(raw or "{}") for raw in _tool_inputs(payloads)] == [{}]
    assert payloads[-1]["type"] == "message_stop"


async def test_responses_ingress_repairs_the_incident_call():
    upstream = _Upstream(_tool_call("{}"), _tool_call('{"code": "4"}'))
    request = OpenAIResponsesRequest(
        model="model",
        input="run js",
        tools=[{"type": "function", "name": "js", "parameters": JS_SCHEMA}],
    )

    async with _client(upstream) as client:
        stream = _transport(client).stream_responses(request)
        raw = "".join([event async for event in stream])

    assert len(upstream.bodies) == 2
    payloads = [
        json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: ")
    ]
    calls = [
        payload["item"]
        for payload in payloads
        if payload.get("type") == "response.output_item.done"
        and payload["item"].get("type") == "function_call"
    ]
    assert [json.loads(call["arguments"]) for call in calls] == [{"code": "4"}]


@pytest.mark.parametrize(
    ("arguments", "schema", "expected"),
    [
        ("{}", JS_SCHEMA, (("code",), False)),
        ("", JS_SCHEMA, (("code",), False)),
        ('{"code": "x"}', JS_SCHEMA, None),
        ("[]", JS_SCHEMA, (("code",), True)),
        ("null", JS_SCHEMA, (("code",), True)),
        ("not json", JS_SCHEMA, None),
        ('{"code": "abandoned', JS_SCHEMA, None),
        ("{}", {"type": "object", "properties": {}}, None),
        ("", {"type": "object"}, None),
        ("[]", {"type": "object"}, None),
    ],
)
async def test_required_argument_defect(arguments, schema, expected):
    assert required_argument_defect(arguments, schema) == expected
