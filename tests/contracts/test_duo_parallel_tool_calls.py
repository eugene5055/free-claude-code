"""Duo: every probed provider that runs several tool calls per turn is asked to (parallel_tool_calls)."""

import pytest

from free_claude_code.providers.openai_chat.profiles import OPENAI_CHAT_PROFILES
from free_claude_code.providers.openai_chat.request_policy import (
    _request_parallel_tool_calls,
)

# Probed 2026-11-02: each accepted parallel_tool_calls=true and returned two tool calls.
PARALLEL = ("qwencloud", "qwencloud_coding", "kimi_code", "zai", "zai_api")


@pytest.mark.parametrize("provider", PARALLEL)
def test_parallel_tool_calls_requested(provider: str) -> None:
    assert OPENAI_CHAT_PROFILES[provider].request_policy.parallel_tool_calls is True


def test_body_asks_for_parallel_calls_unless_the_caller_disables_them() -> None:
    body = {"tools": [{"type": "function"}]}
    _request_parallel_tool_calls(body, None)
    assert body["parallel_tool_calls"] is True
    body = {"tools": [{"type": "function"}]}
    _request_parallel_tool_calls(body, {"type": "auto", "disable_parallel_tool_use": True})
    assert body["parallel_tool_calls"] is False
    body = {}
    _request_parallel_tool_calls(body, None)
    assert "parallel_tool_calls" not in body
