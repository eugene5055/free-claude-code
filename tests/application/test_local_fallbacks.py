"""A request for a local model never falls back to a cloud model by itself (execution._ordered_candidates)."""

import pytest

from free_claude_code.application import execution, provider_health
from free_claude_code.application.provider_locality import (
    is_local_url,
    local_provider_ids,
)
from free_claude_code.application.routing import (
    ProviderModelCandidate,
    ProviderModelTarget,
)
from free_claude_code.config.custom_providers import CustomProviderDefinition
from free_claude_code.config.settings import Settings

LOCAL = "custom_" + "a" * 32
LOCAL_2 = "custom_" + "b" * 32
LOCALS = frozenset({LOCAL, LOCAL_2})


def _candidate(provider_id: str, model: str) -> ProviderModelCandidate:
    ref = f"{provider_id}/{model}"
    return ProviderModelCandidate(
        ProviderModelTarget(
            provider_id=provider_id, provider_model=model, provider_model_ref=ref
        )
    )


@pytest.fixture
def proven(monkeypatch):
    """Recently proven models, newest first, as provider_health.proven_models returns them."""
    models: list[tuple[str, str]] = []

    def fake(exclude_refs, now=None, limit=provider_health._AUTO_FALLBACKS):
        out = [m for m in models if f"{m[0]}/{m[1]}" not in exclude_refs]
        return out[:limit]

    monkeypatch.setattr(provider_health, "proven_models", fake)
    monkeypatch.setattr(provider_health, "limited_until", lambda _p, _now=None: None)
    return models


def _refs(entries) -> list[str]:
    return [entry[1].target.provider_model_ref for entry in entries]


def test_local_primary_gets_no_cloud_automatic_candidates(proven):
    proven.extend([("nvidia_nim", "big"), ("openrouter", "free"), ("groq", "fast")])
    messages: list[str] = []
    handler = execution.logger.add(lambda m: messages.append(str(m)), level="INFO")
    try:
        candidates, automatic, spare = execution._ordered_candidates(
            (_candidate(LOCAL, "duo-local-combined"),),
            auto_fallbacks=True,
            request_id="r1",
            local_providers=LOCALS,
        )
    finally:
        execution.logger.remove(handler)
    assert _refs(candidates) == [f"{LOCAL}/duo-local-combined"]
    assert spare == [] and automatic == set()
    assert any("skips cloud automatic fallbacks" in m for m in messages)


def test_local_primary_rate_limited_gets_no_cloud_candidates(proven, monkeypatch):
    proven.extend([("nvidia_nim", "big"), (LOCAL_2, "small")])
    monkeypatch.setattr(
        provider_health,
        "limited_until",
        lambda p, _now=None: 9e12 if p == LOCAL else None,
    )
    candidates, automatic, spare = execution._ordered_candidates(
        (_candidate(LOCAL, "duo-local-combined"),),
        auto_fallbacks=True,
        request_id="r2",
        local_providers=LOCALS,
    )
    assert _refs(candidates) == [f"{LOCAL_2}/small", f"{LOCAL}/duo-local-combined"]
    assert automatic == {f"{LOCAL_2}/small"}
    assert spare == []


def test_local_primary_may_get_local_automatic_candidate(proven):
    # Cloud models proven more recently must not crowd the local one out of the three slots.
    proven.extend(
        [("nvidia_nim", "a"), ("openrouter", "b"), ("groq", "c"), (LOCAL_2, "small")]
    )
    candidates, automatic, spare = execution._ordered_candidates(
        (_candidate(LOCAL, "duo-local-combined"),),
        auto_fallbacks=True,
        request_id="r3",
        local_providers=LOCALS,
    )
    assert _refs(candidates) == [f"{LOCAL}/duo-local-combined"]
    assert _refs(spare) == [f"{LOCAL_2}/small"]
    assert automatic == {f"{LOCAL_2}/small"}


def test_cloud_primary_is_unchanged_and_can_get_local_candidate(proven):
    proven.extend([("openrouter", "free"), (LOCAL, "duo-local-combined")])
    configured = (_candidate("nvidia_nim", "big"),)
    with_locality = execution._ordered_candidates(
        configured, auto_fallbacks=True, request_id="r4", local_providers=LOCALS
    )
    without_locality = execution._ordered_candidates(
        configured, auto_fallbacks=True, request_id="r4"
    )
    assert with_locality == without_locality
    _candidates, automatic, spare = with_locality
    assert _refs(spare) == ["openrouter/free", f"{LOCAL}/duo-local-combined"]
    assert automatic == {"openrouter/free", f"{LOCAL}/duo-local-combined"}


def test_mixed_local_and_cloud_configured_keeps_cloud_automatic(proven):
    proven.extend([("openrouter", "free")])
    _candidates, automatic, _spare = execution._ordered_candidates(
        (_candidate(LOCAL, "m"), _candidate("nvidia_nim", "big")),
        auto_fallbacks=True,
        request_id="r5",
        local_providers=LOCALS,
    )
    assert automatic == {"openrouter/free"}


@pytest.mark.parametrize(
    ("url", "local"),
    [
        ("http://127.0.0.1:11437/v1", True),
        ("http://localhost:8080/v1", True),
        ("http://[::1]:8080/v1", True),
        ("http://169.254.98.247:11436/v1", True),
        ("http://192.168.68.58:11437/v1", True),
        ("http://10.0.0.5/v1", True),
        ("http://172.16.3.4/v1", True),
        ("http://gpu-box.local:8080/v1", True),
        ("https://openrouter.ai/api/v1", False),
        ("https://8.8.8.8/v1", False),
        ("https://local.example.com/v1", False),
        ("", False),
        (None, False),
    ],
)
def test_is_local_url(url, local):
    assert is_local_url(url) is local


def test_local_provider_ids_from_settings():
    settings = Settings(
        custom_providers=(
            CustomProviderDefinition(
                provider_id=LOCAL,
                display_name="GPU",
                base_url="http://127.0.0.1:11437/v1",
            ),
            CustomProviderDefinition(
                provider_id=LOCAL_2,
                display_name="Cloud gateway",
                base_url="https://gateway.example/v1",
            ),
        ),
    )
    ids = local_provider_ids(settings)
    assert LOCAL in ids
    assert LOCAL_2 not in ids
    assert {"lmstudio", "llamacpp"} <= ids
    assert "nvidia_nim" not in ids and "openrouter" not in ids
    assert local_provider_ids(None) == frozenset()
