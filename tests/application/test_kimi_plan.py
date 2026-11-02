import time

import pytest

from free_claude_code.application import kimi_plan, provider_health


NOW = 1_800_000_000.0


def _body(ratio_5h, ratio_month):
    iso = lambda t: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))
    return {
        "usages": {
            "limit_5h": {"used_ratio": ratio_5h, "reset_time": iso(NOW + 3600)},
            "limit_month_total": {"used_ratio": ratio_month, "reset_time": iso(NOW + 20 * 86400)},
        }
    }


@pytest.fixture(autouse=True)
def _fresh():
    kimi_plan.reset_cache()
    yield
    kimi_plan.reset_cache()


def test_used_up_window_says_when_kimi_is_back():
    found = kimi_plan.parse_windows(_body(1, 0.05), NOW)
    reset, message = kimi_plan.describe(found, NOW)
    assert reset == NOW + 3600
    assert "5-hour limit used up" in message
    assert "monthly 95% left" in message


def test_window_with_room_is_not_exhausted():
    assert kimi_plan.describe(kimi_plan.parse_windows(_body(0.4, 0.05), NOW), NOW) is None


def test_only_kimi_code_is_checked(monkeypatch):
    calls = []
    monkeypatch.setattr(kimi_plan, "_fetch", lambda now: calls.append(now) or [])
    assert kimi_plan.exhausted("nvidia_nim", NOW) is None
    assert calls == []


def test_usage_is_cached(monkeypatch):
    calls = []
    found = kimi_plan.parse_windows(_body(1, 0.05), NOW)
    monkeypatch.setattr(kimi_plan, "_fetch", lambda now: calls.append(now) or found)
    assert kimi_plan.exhausted("kimi_code", NOW) is not None
    assert kimi_plan.exhausted("kimi_code", NOW + 10) is not None
    assert len(calls) == 1


def test_unknown_usage_is_not_exhausted(monkeypatch):
    monkeypatch.setattr(kimi_plan, "_fetch", lambda now: None)
    assert kimi_plan.exhausted("kimi_code", NOW) is None


def test_limited_kimi_rests_until_the_plan_reset(monkeypatch):
    found = kimi_plan.parse_windows(_body(1, 0.05), NOW)
    monkeypatch.setattr(kimi_plan, "_fetch", lambda now: found)
    assert provider_health._header_reset("kimi_code", NOW) == NOW + 3600
