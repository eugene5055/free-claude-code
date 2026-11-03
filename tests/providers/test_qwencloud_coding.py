"""Tests for the QwenCloud Coding Plan OpenAI-chat provider profile."""

from types import SimpleNamespace

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.provider_catalog import QWENCLOUD_CODING_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import (
    REASONING_OFF,
    REASONING_ON,
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)


@pytest.fixture
def qwencloud_coding_provider() -> OpenAIChatProvider:
    return profiled_provider(
        "qwencloud_coding",
        make_provider_config(
            api_key="test-qwencloud-coding-key",
            base_url=QWENCLOUD_CODING_DEFAULT_BASE,
        ),
        admission=immediate_admission(provider_name="qwencloud_coding"),
    )


def test_init_uses_openai_chat_provider(
    qwencloud_coding_provider: OpenAIChatProvider,
) -> None:
    assert qwencloud_coding_provider._api_key == "test-qwencloud-coding-key"
    assert qwencloud_coding_provider._base_url == QWENCLOUD_CODING_DEFAULT_BASE
    assert qwencloud_coding_provider._provider_name == "QWENCLOUD_CODING"


def test_build_request_body_preserves_tools_and_images_without_inventing_a_cap(
    qwencloud_coding_provider: OpenAIChatProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "qwen3.7-plus",
            "system": "Be concise.",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Inspect this image."},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "aW1hZ2U=",
                            },
                        },
                    ],
                }
            ],
            "tools": [
                {
                    "name": "inspect_image",
                    "description": "Inspect an image",
                    "input_schema": {
                        "type": "object",
                        "properties": {"detail": {"type": "string"}},
                    },
                }
            ],
        }
    )

    body = qwencloud_coding_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assert body["model"] == "qwen3.7-plus"
    assert "max_tokens" not in body
    assert body["messages"][0] == {"role": "system", "content": "Be concise."}
    assert body["messages"][1]["content"] == [
        {"type": "text", "text": "Inspect this image."},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
        },
    ]
    assert body["tools"][0]["function"]["name"] == "inspect_image"
    assert "tool_stream" not in body


def test_build_request_body_preserves_explicit_client_cap(
    qwencloud_coding_provider: OpenAIChatProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "qwen3.7-plus",
            "max_tokens": 321,
            "messages": [{"role": "user", "content": "Hello"}],
        }
    )

    body = qwencloud_coding_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assert body["max_tokens"] == 321


@pytest.mark.parametrize(
    ("reasoning", "effort"),
    [
        (REASONING_OFF, "none"),
        (REASONING_ON, "high"),
        (ReasoningPolicy.on(effort=ReasoningEffort.MAX), "high"),
    ],
)
def test_build_request_body_sends_explicit_effort_without_vendor_thinking_fields(
    qwencloud_coding_provider: OpenAIChatProvider,
    reasoning: ReasoningPolicy,
    effort: str,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "qwen3.7-plus",
            "messages": [{"role": "user", "content": "Hello"}],
        }
    )

    body = qwencloud_coding_provider._chat._build_request_body(
        request, reasoning=reasoning
    )
    extra_body = body.get("extra_body", {})
    # Duo explicit-reasoning contract: the standard OpenAI reasoning_effort is
    # always sent (lowest level when reasoning is off) so Token Plan models
    # never fall back to their maximum default thinking; vendor-specific
    # thinking fields are still never invented.
    assert body["reasoning_effort"] == effort

    for field in (
        "enable_thinking",
        "preserve_thinking",
        "reasoning",
        "thinking",
        "thinking_budget",
        "tool_stream",
    ):
        assert field not in body
        assert field not in extra_body


def test_build_request_body_replays_prior_reasoning_content(
    qwencloud_coding_provider: OpenAIChatProvider,
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "qwen3.7-plus",
            "messages": [
                {"role": "user", "content": "Solve it."},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Work through it."},
                        {"type": "text", "text": "The answer is 42."},
                    ],
                },
                {"role": "user", "content": "Continue."},
            ],
        }
    )

    body = qwencloud_coding_provider._chat._build_request_body(
        request,
        reasoning=reasoning_for(request),
    )

    assert body["messages"][1] == {
        "role": "assistant",
        "content": "The answer is 42.",
        "reasoning_content": "Work through it.",
    }
    assert (
        qwencloud_coding_provider._profile.reasoning_delta(
            SimpleNamespace(reasoning_content="next thought")
        )
        == "next thought"
    )


@pytest.mark.asyncio
async def test_model_catalog_uses_standard_endpoint_base_url_and_auth(
    qwencloud_coding_provider: OpenAIChatProvider,
) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": "qwen3.7-plus",
                        "object": "model",
                        "created": 0,
                        "owned_by": "qwencloud",
                    },
                    {
                        "id": "kimi-k2.5",
                        "object": "model",
                        "created": 0,
                        "owned_by": "qwencloud",
                    },
                ],
            },
        )

    await qwencloud_coding_provider._client.close()
    qwencloud_coding_provider._client = AsyncOpenAI(
        api_key="wire-qwencloud-coding-key",
        base_url=QWENCLOUD_CODING_DEFAULT_BASE,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    try:
        model_infos = await qwencloud_coding_provider.list_model_infos()
    finally:
        await qwencloud_coding_provider.cleanup()

    assert model_infos == frozenset(
        {
            ProviderModelInfo("qwen3.7-plus"),
            ProviderModelInfo("kimi-k2.5"),
        }
    )
    assert len(requests) == 1
    assert str(requests[0].url) == (
        "https://coding-intl.dashscope.aliyuncs.com/v1/models"
    )
    assert requests[0].headers["authorization"] == ("Bearer wire-qwencloud-coding-key")
