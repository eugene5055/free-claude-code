"""Requests without an output budget get a default instead of running unbounded."""

from free_claude_code.core.anthropic.conversion import ReasoningReplayMode
from free_claude_code.providers.custom import CUSTOM_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.providers.openai_chat import OpenAIChatRequestPolicy
from free_claude_code.providers.openai_chat.request_policy import (
    apply_openai_chat_body_policy,
)


def _policy(**kwargs):
    return OpenAIChatRequestPolicy(
        "custom_test", ReasoningReplayMode.REASONING_CONTENT, **kwargs
    )


def test_missing_budget_gets_default():
    body = {"messages": []}
    apply_openai_chat_body_policy(body, _policy(default_max_tokens=1234))
    assert body["max_tokens"] == 1234


def test_null_budget_gets_default():
    body = {"messages": [], "max_tokens": None}
    apply_openai_chat_body_policy(body, _policy(default_max_tokens=1234))
    assert body["max_tokens"] == 1234


def test_explicit_budget_is_kept():
    body = {"messages": [], "max_tokens": 50}
    apply_openai_chat_body_policy(body, _policy(default_max_tokens=1234))
    assert body["max_tokens"] == 50


def test_explicit_completion_budget_is_kept():
    body = {"messages": [], "max_completion_tokens": 70}
    apply_openai_chat_body_policy(body, _policy(default_max_tokens=1234))
    assert body.get("max_tokens") is None
    assert body["max_completion_tokens"] == 70


def test_no_default_leaves_body_alone():
    body = {"messages": []}
    apply_openai_chat_body_policy(body, _policy())
    assert "max_tokens" not in body


def test_custom_default_is_bounded():
    assert 4096 <= CUSTOM_DEFAULT_MAX_OUTPUT_TOKENS <= 65536
