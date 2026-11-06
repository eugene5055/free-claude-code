"""Claude's auto-mode safety classifier never runs on local custom providers.

User, 2026-11-06: a duo-local:stable session had a user-approved command blocked
by the local GPU model acting as classifier; "make sure it never happens again".
"""

from unittest.mock import AsyncMock

import pytest
from fastapi.responses import StreamingResponse

from free_claude_code.api.handlers import MessagesHandler
from free_claude_code.application.routing import ModelRouter
from free_claude_code.config.custom_providers import CustomProviderDefinition
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from tests.api.test_api_handlers import (
    _CLASSIFIER_USER,
    _CURRENT_CLASSIFIER_SYSTEM,
    _LEGACY_CLASSIFIER_SYSTEM,
    FakeProvider,
    _streaming_body_text,
)
from tests.web_tools_support import StubWebToolsClient

LOCAL = "custom_" + "a" * 32


def _settings(model: str = "nvidia_nim/cloud-model") -> Settings:
    definition = CustomProviderDefinition.model_validate(
        {
            "provider_id": LOCAL,
            "display_name": "RTX 4090",
            "base_url": "http://127.0.0.1:11435/v1",
        }
    )
    return Settings(custom_providers=(definition,), model=model, model_fallbacks=())


async def _send(settings: Settings, *, system: str, user: str):
    provider = FakeProvider()
    calls: list[str] = []

    def resolve(provider_id, _api_key_number):
        calls.append(provider_id)
        return provider

    handler = MessagesHandler(
        settings,
        provider_resolver=AsyncMock(side_effect=resolve),
        web_tools=StubWebToolsClient(),
    )
    response = await handler.create(
        MessagesRequest(
            model=f"{LOCAL}/local-model",
            max_tokens=64,
            stream=True,
            system=system,
            messages=[Message(role="user", content=user)],
        )
    )
    assert isinstance(response, StreamingResponse)
    await _streaming_body_text(response)
    return calls, provider


@pytest.mark.asyncio
@pytest.mark.parametrize("system", [_CURRENT_CLASSIFIER_SYSTEM, _LEGACY_CLASSIFIER_SYSTEM])
async def test_classifier_on_local_model_runs_on_cloud_model(system: str) -> None:
    calls, provider = await _send(_settings(), system=system, user=_CLASSIFIER_USER)
    assert calls and not any(c.startswith("custom_") for c in calls)
    assert calls[0] == "nvidia_nim"
    assert provider.requests[0].model == "cloud-model"


@pytest.mark.asyncio
async def test_ordinary_requests_stay_on_local_model() -> None:
    calls, provider = await _send(_settings(), system="Be helpful.", user="hello")
    assert calls[0] == LOCAL
    assert provider.requests[0].model == "local-model"


def test_classifier_route_never_lists_a_custom_fallback() -> None:
    router = ModelRouter(_settings())
    routed = router.resolve_messages_request(
        MessagesRequest(
            model=f"{LOCAL}/local-model",
            max_tokens=8,
            messages=[Message(role="user", content=_CLASSIFIER_USER)],
        )
    )
    moved = router.classifier_messages_request(routed)
    assert moved is not None
    targets = [moved.resolved.primary, *(c.target for c in moved.resolved.fallbacks)]
    assert not any(t.provider_id.startswith("custom_") for t in targets)
    assert moved.resolved.original_model == f"{LOCAL}/local-model"


def test_classifier_stays_put_without_a_cloud_model() -> None:
    router = ModelRouter(_settings(model=f"{LOCAL}/local-model"))
    routed = router.resolve_messages_request(
        MessagesRequest(
            model=f"{LOCAL}/local-model",
            max_tokens=8,
            messages=[Message(role="user", content=_CLASSIFIER_USER)],
        )
    )
    assert router.classifier_messages_request(routed) is None
