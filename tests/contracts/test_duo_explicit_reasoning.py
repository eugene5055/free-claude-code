"""Duo contract: every route Duo uses sends an explicit reasoning control."""

import pytest

from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.config.settings import parse_reasoning_provider_policies
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.openai_chat.profiles import (
    KIMI_CODE_REASONING,
    OPENAI_CHAT_PROFILES,
    QWENCLOUD_REASONING,
)
from free_claude_code.providers.openai_chat.request_policy import (
    build_openai_chat_request_body,
)

OFF = ReasoningPolicy.off()
DEFAULT = ReasoningPolicy.provider_default()


def _encode(encoder, model: str, policy: ReasoningPolicy) -> dict:
    body: dict = {"model": model}
    encoder.encode(body, policy)
    return body


@pytest.mark.parametrize(
    ("model", "policy", "expected"),
    [
        # GLM refuses none/minimal/medium on Token Plan; low yields no reasoning.
        ("glm-5.3", OFF, "low"),
        ("glm-5.3", DEFAULT, "low"),
        ("glm-5.3", ReasoningPolicy.on(effort=ReasoningEffort.MEDIUM), "low"),
        ("glm-5.3", ReasoningPolicy.on(effort=ReasoningEffort.HIGH), "high"),
        ("glm-5.3", ReasoningPolicy.on(effort=ReasoningEffort.MAX), "max"),
        ("glm-5.3", ReasoningPolicy.on(), "high"),
        ("qwen3.8-max", OFF, "none"),
        ("qwen3.8-max", DEFAULT, "none"),
        ("qwen3.8-max", ReasoningPolicy.on(effort=ReasoningEffort.MINIMAL), "minimal"),
        ("qwen3.8-max", ReasoningPolicy.on(effort=ReasoningEffort.MAX), "high"),
        ("deepseek-v4.1-flash-1017", OFF, "none"),
        ("deepseek-v4-pro", DEFAULT, "none"),
        ("deepseek-v4-pro", ReasoningPolicy.on(effort=ReasoningEffort.MINIMAL), "low"),
        # Unknown families get the value every probed family accepts.
        ("minimax-m3", OFF, "low"),
    ],
)
def test_qwencloud_clamps_to_each_family_vocabulary(model, policy, expected):
    assert _encode(QWENCLOUD_REASONING, model, policy)["reasoning_effort"] == expected


@pytest.mark.parametrize(
    ("model", "policy", "expected"),
    [
        ("k3.1-pro", OFF, "low"),
        ("k3.1-pro", DEFAULT, "low"),
        ("k3.1-pro", ReasoningPolicy.on(effort=ReasoningEffort.LOW), "low"),
        ("k3.1-pro", ReasoningPolicy.on(), "max"),
        ("k3", OFF, "none"),
        ("k3", DEFAULT, "none"),
    ],
)
def test_kimi_code_keeps_k3_pro_thinking(model, policy, expected):
    assert _encode(KIMI_CODE_REASONING, model, policy)["reasoning_effort"] == expected


@pytest.mark.parametrize("provider_id", ["qwencloud", "qwencloud_coding"])
def test_token_plan_requests_parallel_tool_calls(provider_id):
    policy = OPENAI_CHAT_PROFILES[provider_id].request_policy
    tool = {"name": "read", "input_schema": {"type": "object", "properties": {}}}
    request = MessagesRequest.model_validate(
        {
            "model": "glm-5.3",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [tool],
        }
    )
    body = build_openai_chat_request_body(request, reasoning=OFF, policy=policy)
    assert body["parallel_tool_calls"] is True

    request = request.model_copy(
        update={"tool_choice": {"type": "auto", "disable_parallel_tool_use": True}}
    )
    body = build_openai_chat_request_body(request, reasoning=OFF, policy=policy)
    assert body["parallel_tool_calls"] is False


def test_provider_reasoning_policies_parse():
    assert parse_reasoning_provider_policies(" kimi_code=low, gemini=LOW ") == {
        "kimi_code": ReasoningPreference.LOW,
        "gemini": ReasoningPreference.LOW,
    }
    assert parse_reasoning_provider_policies("") == {}
    with pytest.raises(ValueError):
        parse_reasoning_provider_policies("kimi_code=inherit")
    with pytest.raises(ValueError):
        parse_reasoning_provider_policies("kimi_code")


# Routes Duo sends traffic to, with a representative model for each.
DUO_ROUTES = {
    "qwencloud": "glm-5.3",
    "qwencloud_coding": "qwen3.8-max",
    "kimi_code": "k3.1-pro",
    "nvidia_nim": "nvidia/nemotron-3-ultra-550b-a55b",
    "open_router": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "groq": "openai/gpt-oss-120b",
    "cloudflare": "@cf/zai-org/glm-4.7-flash",
    "mistral": "devstral-latest",
    "deepseek": "deepseek-v4-flash",
    "gemini": "models/gemini-3-flash-preview",
}


def _has_path(body: dict, path: tuple[str, ...]) -> bool:
    target: object = body
    for key in path:
        if not isinstance(target, dict) or key not in target:
            return False
        target = target[key]
    return True


@pytest.mark.parametrize("provider_id", sorted(DUO_ROUTES))
@pytest.mark.parametrize(
    "policy",
    [OFF, ReasoningPolicy.on(effort=ReasoningEffort.LOW)],
    ids=["off", "low"],
)
def test_every_duo_route_sends_an_explicit_reasoning_control(
    provider_id, policy, duo_route_provider
):
    """Fails when a route Duo uses would leave reasoning to the provider default."""
    provider = duo_route_provider(provider_id)
    request = MessagesRequest.model_validate(
        {
            "model": DUO_ROUTES[provider_id],
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    chat = provider._chat
    body = chat._build_request_body(request, reasoning=policy)
    off_fields = chat._behavior.reasoning_off_fields
    assert off_fields, f"{provider_id} has no reasoning control field"
    if policy.control.value == "off":
        assert any(_has_path(body, path) for path in off_fields), (
            f"{provider_id} sent no reasoning control under OFF: {body}"
        )
    else:
        sent = {k: v for k, v in body.items() if k not in {"model", "messages", "max_tokens", "max_completion_tokens", "stream", "stream_options"}}
        assert sent, f"{provider_id} sent no reasoning control under LOW: {body}"


@pytest.fixture
def duo_route_provider():
    from free_claude_code.config.settings import Settings
    from free_claude_code.providers.openai_chat import create_openai_chat_provider
    from free_claude_code.providers.runtime.factory import (
        _SPECIAL_PROVIDER_FACTORIES,
    )
    from tests.providers.support import immediate_admission, make_provider_config

    settings = Settings.model_construct(cloudflare_account_id="test-account")

    def build(provider_id: str):
        config = make_provider_config(
            api_key="test-key", base_url="https://provider.invalid/v1"
        )
        admission = immediate_admission(provider_name=provider_id)
        loader = _SPECIAL_PROVIDER_FACTORIES.get(provider_id)
        if loader is None:
            return create_openai_chat_provider(provider_id, config, admission)
        return loader()(config, settings, admission)

    return build
