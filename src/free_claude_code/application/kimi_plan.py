"""Kimi Code plan windows (Duo).

Kimi Code's plan has a rolling 5-hour window and a monthly allowance. When a window is used up Kimi answers
429 without rate-limit headers, so FCC used to back off for a minute and re-probe for as long as the window
lasted. This reads the plan's own usage endpoint so a 429 during a used-up window rests kimi_code until the
real reset, fails fast instead of retrying, and tells the client when Kimi is back.
"""


import threading
import time
from datetime import datetime

import httpx
from loguru import logger

PROVIDER_ID = "kimi_code"
USAGE_URL = "https://api.kimi.com/coding/v1/usages"
WINDOWS = (("limit_5h", "5-hour"), ("limit_month_total", "monthly"))
_CACHE_S = 60.0
_TIMEOUT_S = 4.0

_lock = threading.Lock()
_cache: dict = {"at": float("-inf"), "windows": None}


def _credentials() -> tuple[str | None, str | None]:
    try:
        from free_claude_code.config.loader import get_settings

        settings = get_settings()
    except Exception:
        return None, None
    keys = getattr(settings, "kimi_code_api_keys", None) or ()
    key = None
    for item in keys if isinstance(keys, (list, tuple)) else ():
        value = item.get_secret_value() if hasattr(item, "get_secret_value") else item
        if value:
            key = str(value)
            break
    proxy = getattr(settings, "kimi_code_proxy", None)
    proxy = proxy.get_secret_value() if hasattr(proxy, "get_secret_value") else proxy
    return key, (str(proxy) if proxy else None)


def _epoch(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def parse_windows(body: dict, now: float) -> list[tuple[float, str, float]]:
    """(percent left, label, reset epoch) for each plan window that has not reset yet."""
    usages = body.get("usages") if isinstance(body, dict) else None
    out = []
    for name, label in WINDOWS:
        window = usages.get(name) if isinstance(usages, dict) else None
        if not isinstance(window, dict):
            continue
        try:
            ratio = float(window.get("used_ratio"))
        except (TypeError, ValueError):
            continue
        reset = _epoch(window.get("reset_time"))
        if reset is None or reset <= now:
            continue
        out.append((max(0.0, min(100.0, (1 - ratio) * 100)), label, reset))
    return out


def _fetch(now: float) -> list[tuple[float, str, float]] | None:
    key, proxy = _credentials()
    if not key:
        return None
    try:
        response = httpx.get(
            USAGE_URL,
            headers={"Authorization": f"Bearer {key}"},
            timeout=_TIMEOUT_S,
            proxy=proxy,
        )
        response.raise_for_status()
        return parse_windows(response.json(), now)
    except Exception as error:
        logger.debug("Kimi Code plan usage unavailable: {}", type(error).__name__)
        return None


def windows(now: float | None = None) -> list[tuple[float, str, float]] | None:
    now = time.time() if now is None else now
    with _lock:
        if now - _cache["at"] < _CACHE_S:
            return _cache["windows"]
    found = _fetch(now)
    with _lock:
        _cache.update(at=now, windows=found)
    return found


def describe(found: list[tuple[float, str, float]], now: float) -> tuple[float, str] | None:
    """(reset epoch, message) when a plan window is used up, else None."""
    used = [w for w in found if w[0] <= 0]
    if not used:
        return None
    reset = max(w[2] for w in used)
    when = time.strftime("%H:%M" if reset - now < 86400 else "%b %d %H:%M", time.localtime(reset))
    labels = " and ".join(w[1] for w in used)
    rest = [f"{w[1]} {w[0]:.0f}% left" for w in found if w[0] > 0]
    tail = f" ({', '.join(rest)})" if rest else ""
    return reset, f"Kimi Code {labels} limit used up; back at {when}{tail}."


def exhausted(provider_id: str, now: float | None = None) -> tuple[float, str] | None:
    """When provider_id is Kimi Code and a plan window is used up: (reset epoch, message)."""
    if provider_id != PROVIDER_ID:
        return None
    now = time.time() if now is None else now
    found = windows(now)
    return describe(found, now) if found else None


def reset_cache() -> None:
    with _lock:
        _cache.update(at=float("-inf"), windows=None)
