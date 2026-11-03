"""Validated capacity policy, independent of clients and controller lifetime."""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from free_claude_code.config.settings import Settings


@dataclass(frozen=True, slots=True)
class ProviderAdmissionLimits:
    rate_limit: int
    rate_window: float
    max_concurrency: int
    # (provider_id, limits) pairs from PROVIDER_ADMISSION_LIMITS; every other
    # provider uses the limits above.
    overrides: tuple[tuple[str, "ProviderAdmissionLimits"], ...] = ()

    def for_provider(self, provider_id: str) -> "ProviderAdmissionLimits":
        """This provider's limits: its override, else the global ones."""
        for name, limits in self.overrides:
            if name == provider_id:
                return limits
        if not self.overrides:
            return self
        return ProviderAdmissionLimits(
            self.rate_limit, self.rate_window, self.max_concurrency
        )

    def __post_init__(self) -> None:
        for name in ("rate_limit", "max_concurrency"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.rate_window) or self.rate_window <= 0:
            raise ValueError("rate_window must be finite and > 0")

    @classmethod
    def from_settings(cls, settings: Settings) -> ProviderAdmissionLimits:
        from free_claude_code.config.settings import (
            parse_provider_admission_limits,
        )

        overrides = tuple(
            sorted(
                (name, cls(rate, window, concurrency))
                for name, (rate, window, concurrency) in (
                    parse_provider_admission_limits(
                        settings.provider_admission_limits
                    ).items()
                )
            )
        )
        return cls(
            settings.provider_rate_limit,
            settings.provider_rate_window,
            settings.provider_max_concurrency,
            overrides,
        )
