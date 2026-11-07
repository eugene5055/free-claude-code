"""llama.cpp custom provider: Qwen template efforts, budgets and dropped features."""

import json

import httpx2
import pytest

from free_claude_code.config.custom_providers import REASONING_FORMATS
from free_claude_code.core.anthropic import MessagesRequest
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.openai_chat.llamacpp_compat import (
    NON_THINKING_TEMPERATURE,
    NON_THINKING_TOP_P,
    apply_llamacpp_request_compat,
)
from free_claude_code.providers.openai_chat.reasoning import (
    LLAMACPP_BUDGET_MESSAGE,
    llamacpp_reasoning_fields,
)
from tests.providers.test_custom_provider import completion, definition, provider

# Values Qwen3.x chat templates accept; anything else raises in the template.
TEMPLATE_EFFORTS = {"none", "low", "medium", "xhigh"}


@pytest.mark.parametrize(
    "policy,expected",
    [
        (ReasoningPolicy.off(), {"reasoning_effort": "none"}),
        (ReasoningPolicy.prefer_off(), {"reasoning_effort": "none"}),
        (ReasoningPolicy.provider_default(), {}),
        (ReasoningPolicy.on(), {"reasoning_effort": "xhigh"}),
        (ReasoningPolicy.on(effort=ReasoningEffort.LOW), {"reasoning_effort": "low"}),
        (
            ReasoningPolicy.on(effort=ReasoningEffort.MEDIUM),
            {"reasoning_effort": "medium"},
        ),
        (
            ReasoningPolicy.on(effort=ReasoningEffort.HIGH),
            {"reasoning_effort": "xhigh"},
        ),
        (ReasoningPolicy.on(effort=ReasoningEffort.MAX), {"reasoning_effort": "xhigh"}),
        (
            ReasoningPolicy.on(effort=ReasoningEffort.MINIMAL),
            {
                "reasoning_effort": "low",
                "thinking_budget_tokens": 512,
                "reasoning_budget_message": LLAMACPP_BUDGET_MESSAGE,
            },
        ),
        (
            ReasoningPolicy.on(budget_tokens=1024),
            {
                "reasoning_effort": "low",
                "thinking_budget_tokens": 1024,
                "reasoning_budget_message": LLAMACPP_BUDGET_MESSAGE,
            },
        ),
        (
            ReasoningPolicy.on(budget_tokens=8192),
            {
                "reasoning_effort": "medium",
                "thinking_budget_tokens": 8192,
                "reasoning_budget_message": LLAMACPP_BUDGET_MESSAGE,
            },
        ),
        (
            ReasoningPolicy.on(budget_tokens=31999),
            {
                "reasoning_effort": "xhigh",
                "thinking_budget_tokens": 31999,
                "reasoning_budget_message": LLAMACPP_BUDGET_MESSAGE,
            },
        ),
    ],
)
def test_reasoning_fields_use_only_template_efforts(policy, expected):
    fields = llamacpp_reasoning_fields(policy)
    assert fields == expected
    if "reasoning_effort" in fields:
        assert fields["reasoning_effort"] in TEMPLATE_EFFORTS


def test_llamacpp_is_an_openai_chat_reasoning_format():
    assert "llamacpp" in REASONING_FORMATS["openai_chat"]


def _request(**kwargs):
    base = {
        "model": "m",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Hi"}],
    }
    base.update(kwargs)
    return MessagesRequest.model_validate(base)


def _tool(name):
    return {"type": "function", "function": {"name": name, "parameters": {}}}


def test_named_tool_choice_offers_only_that_tool_and_requires_it():
    body = {
        "tools": [_tool("Read"), _tool("Bash")],
        "tool_choice": {"type": "function", "function": {"name": "Bash"}},
    }
    request = _request(
        tools=[
            {"name": "Read", "input_schema": {"type": "object"}},
            {"name": "Bash", "input_schema": {"type": "object"}},
        ],
        tool_choice={"type": "tool", "name": "Bash"},
    )
    apply_llamacpp_request_compat(body, request, ReasoningPolicy.provider_default())
    assert body["tool_choice"] == "required"
    assert [tool["function"]["name"] for tool in body["tools"]] == ["Bash"]
    assert body["parallel_tool_calls"] is False


@pytest.mark.parametrize(
    "tool_choice,expected",
    [
        (None, True),
        ({"type": "auto"}, True),
        ({"type": "auto", "disable_parallel_tool_use": True}, False),
        ({"type": "any", "disable_parallel_tool_use": True}, False),
    ],
)
def test_parallel_tool_calls_follow_anthropic_flag(tool_choice, expected):
    body = {"tools": [_tool("Read"), _tool("Bash")], "tool_choice": "auto"}
    kwargs = {"tools": [{"name": "Read", "input_schema": {"type": "object"}}]}
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    apply_llamacpp_request_compat(
        body, _request(**kwargs), ReasoningPolicy.provider_default()
    )
    assert body["parallel_tool_calls"] is expected


def test_no_tools_or_none_choice_leave_parallel_unset():
    body = {}
    apply_llamacpp_request_compat(body, _request(), ReasoningPolicy.provider_default())
    assert "parallel_tool_calls" not in body
    body = {"tools": [_tool("Read")], "tool_choice": "none"}
    apply_llamacpp_request_compat(body, _request(), ReasoningPolicy.provider_default())
    assert "parallel_tool_calls" not in body


def test_top_k_and_structured_output_are_forwarded():
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    body = {}
    request = _request(
        top_k=20, output_config={"format": {"type": "json_schema", "schema": schema}}
    )
    apply_llamacpp_request_compat(body, request, ReasoningPolicy.provider_default())
    assert body["extra_body"] == {"top_k": 20}
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "structured_output", "schema": schema, "strict": True},
    }


def test_non_thinking_sampling_defaults_only_when_client_is_silent():
    body = {}
    apply_llamacpp_request_compat(body, _request(), ReasoningPolicy.off())
    assert body["temperature"] == NON_THINKING_TEMPERATURE
    assert body["top_p"] == NON_THINKING_TOP_P
    assert body["presence_penalty"] == 1.5
    assert body["extra_body"] == {"top_k": 20}

    body = {"temperature": 0.2}
    apply_llamacpp_request_compat(
        body, _request(temperature=0.2), ReasoningPolicy.off()
    )
    assert body == {"temperature": 0.2}

    body = {}
    apply_llamacpp_request_compat(body, _request(), ReasoningPolicy.on())
    assert "temperature" not in body and "top_p" not in body


async def _wire_body(request, reasoning):
    bodies = []

    def reply(http_request):
        bodies.append(json.loads(http_request.content))
        return httpx2.Response(
            200,
            text=completion("openai_chat"),
            headers={"content-type": "text/event-stream"},
        )

    instance = provider(
        definition(api_format="openai_chat", reasoning_format="llamacpp"),
        openai_handler=reply,
    )
    try:
        assert [
            chunk
            async for chunk in instance.stream_messages(request, reasoning=reasoning)
        ]
    finally:
        await instance.cleanup()
    return bodies[0]


@pytest.mark.asyncio
async def test_wire_body_carries_template_effort_and_exact_budget():
    body = await _wire_body(
        _request(max_tokens=32000), ReasoningPolicy.on(budget_tokens=4096)
    )
    assert body["reasoning_effort"] == "medium"
    assert body["thinking_budget_tokens"] == 4096
    assert body["reasoning_budget_message"] == LLAMACPP_BUDGET_MESSAGE


@pytest.mark.asyncio
async def test_wire_body_forwards_top_k_through_extra_body():
    body = await _wire_body(_request(top_k=20), ReasoningPolicy.on())
    assert body["top_k"] == 20
    assert "extra_body" not in body


@pytest.mark.asyncio
async def test_wire_body_high_effort_never_sends_high():
    body = await _wire_body(_request(), ReasoningPolicy.on(effort=ReasoningEffort.HIGH))
    assert body["reasoning_effort"] == "xhigh"


@pytest.mark.asyncio
async def test_wire_body_off_disables_thinking_and_forces_named_tool():
    request = _request(
        tools=[
            {"name": "Read", "input_schema": {"type": "object"}},
            {"name": "Bash", "input_schema": {"type": "object"}},
        ],
        tool_choice={"type": "tool", "name": "Read"},
    )
    body = await _wire_body(request, ReasoningPolicy.off())
    assert body["reasoning_effort"] == "none"
    assert body["tool_choice"] == "required"
    assert [tool["function"]["name"] for tool in body["tools"]] == ["Read"]
    assert body["parallel_tool_calls"] is False
