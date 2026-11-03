"""PROVIDER_ADMISSION_LIMITS: a provider with a higher real limit gets its own admission limits."""

import pytest

from free_claude_code.config.settings import Settings, parse_provider_admission_limits
from free_claude_code.providers.admission_policy import ProviderAdmissionLimits
from free_claude_code.providers.admission_registry import ProviderAdmissionRegistry


def test_parse_pairs() -> None:
    assert parse_provider_admission_limits("qwencloud=10/1/8, kimi_code=4/1/8") == {
        "qwencloud": (10, 1.0, 8),
        "kimi_code": (4, 1.0, 8),
    }
    assert parse_provider_admission_limits("") == {}


@pytest.mark.parametrize("bad", ["qwencloud", "qwencloud=10/1", "qwencloud=0/1/8", "=1/1/1", "x=1/0/1"])
def test_parse_rejects(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_provider_admission_limits(bad)


def test_from_settings_overrides_one_provider_only() -> None:
    settings = Settings.model_validate(
        {"PROVIDER_ADMISSION_LIMITS": "qwencloud=10/1/8"}
    )
    limits = ProviderAdmissionLimits.from_settings(settings)
    assert limits.for_provider("qwencloud") == ProviderAdmissionLimits(10, 1.0, 8)
    assert limits.for_provider("nvidia_nim") == ProviderAdmissionLimits(
        settings.provider_rate_limit,
        settings.provider_rate_window,
        settings.provider_max_concurrency,
    )


def test_settings_rejects_bad_value() -> None:
    with pytest.raises(ValueError):
        Settings.model_validate({"PROVIDER_ADMISSION_LIMITS": "qwencloud=fast"})


@pytest.mark.asyncio
async def test_registry_uses_and_reconfigures_override() -> None:
    base = ProviderAdmissionLimits(1, 2.0, 2)
    registry = ProviderAdmissionRegistry(
        ProviderAdmissionLimits(1, 2.0, 2, (("qwencloud", ProviderAdmissionLimits(10, 1.0, 8)),))
    )
    registry.get("qwencloud")
    registry.get("nvidia_nim")
    assert registry._capacities["qwencloud"]._limits == ProviderAdmissionLimits(10, 1.0, 8)
    assert registry._capacities["nvidia_nim"]._limits == base
    registry.reconfigure(
        ProviderAdmissionLimits(1, 2.0, 2, (("qwencloud", ProviderAdmissionLimits(20, 1.0, 12)),))
    )
    assert registry._capacities["qwencloud"]._limits == ProviderAdmissionLimits(20, 1.0, 12)
    assert registry._capacities["nvidia_nim"]._limits == base
    registry.close()
