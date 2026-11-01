"""Remember upstream rate-limit headers so /usage can show live limit bars.

Most providers have no quota API, but many report their limits on every
inference response (OpenAI-style ``x-ratelimit-*``, Anthropic
``anthropic-ratelimit-*``, IETF ``ratelimit-*``). FCC sends all upstream
traffic through httpx clients, so one hook on ``AsyncClient.send`` sees them.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from loguru import logger

from free_claude_code.config.paths import config_dir_path
from free_claude_code.core.request_outcomes import current_request_outcome

_STORE_PATH = config_dir_path() / "rate_limits.json"
_ROLES = ("limit", "remaining", "reset", "utilization", "used")
_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")

_lock = threading.Lock()
_state: dict[str, dict[str, dict[str, float | None]]] = {}
_loaded = False
_installed = False


def _load() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        data = json.loads(_STORE_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            _state.update(data)
    except (OSError, ValueError):
        pass


def _save() -> None:
    tmp = _STORE_PATH.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(_state), encoding="utf-8")
        os.replace(tmp, _STORE_PATH)
    except OSError as exc:
        logger.debug("Could not save rate limits: {}", exc)


def _parse_reset(value: str, now: float) -> float | None:
    """Return the reset as epoch seconds from any of the common encodings."""
    text = value.strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        number = None
    if number is not None:
        if number > 1e12:
            return number / 1000  # epoch milliseconds (OpenRouter)
        if number > 1e9:
            return number  # epoch seconds
        return now + number  # delta seconds
    parts = _DURATION.findall(text)
    if parts and "".join(n + u for n, u in parts) == text.replace(" ", ""):
        scale = {"h": 3600, "m": 60, "s": 1, "ms": 0.001}
        return now + sum(float(n) * scale[u] for n, u in parts)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _parse(headers) -> dict[str, dict[str, float | None]]:
    """Group rate-limit headers into windows keyed by what they measure."""
    now = time.time()
    windows: dict[str, dict[str, float | None]] = {}
    for raw_name, raw_value in headers.items():
        name = raw_name.lower()
        if "ratelimit" not in name:
            continue
        tokens = [t for t in name.split("-") if t not in ("x", "anthropic")]
        if "ratelimit" not in tokens:
            continue
        tokens.remove("ratelimit")
        role = next((t for t in tokens if t in _ROLES), None)
        if role is None:
            continue
        tokens.remove(role)
        key = "-".join(tokens) or "requests"
        if role == "reset":
            parsed = _parse_reset(raw_value, now)
        else:
            try:
                parsed = float(raw_value.strip().split(",")[0].split(";")[0])
            except ValueError:
                continue
        windows.setdefault(key, {})[role] = parsed
    for window in windows.values():
        window["observed_at"] = now
    return {
        key: window
        for key, window in windows.items()
        if "limit" in window or "utilization" in window
    }


def _provider_for(url: str) -> str | None:
    outcome = current_request_outcome.get()
    if outcome is not None and outcome.provider_id:
        return outcome.provider_id
    host = urlsplit(url).hostname or ""
    from free_claude_code.config.provider_catalog import PROVIDER_CATALOG

    for provider_id, descriptor in PROVIDER_CATALOG.items():
        base = getattr(descriptor, "default_base_url", None) or ""
        if host and urlsplit(base).hostname == host:
            return provider_id
    return None


def record(url: str, headers) -> None:
    windows = _parse(headers)
    if not windows:
        return
    provider_id = _provider_for(url)
    if provider_id is None:
        return
    with _lock:
        _load()
        _state.setdefault(provider_id, {}).update(windows)
        _save()


def snapshot() -> dict[str, dict[str, dict[str, float | None]]]:
    with _lock:
        _load()
        return json.loads(json.dumps(_state))


def install() -> None:
    """Wrap httpx (and the vendored httpx2) AsyncClient.send once per process."""
    global _installed
    if _installed:
        return
    _installed = True
    modules = []
    for module_name in ("httpx", "httpx2"):
        try:
            modules.append(__import__(module_name))
        except ImportError:
            continue
    for module in modules:
        original = module.AsyncClient.send

        async def send(self, request, *args, _original=original, **kwargs):
            response = await _original(self, request, *args, **kwargs)
            try:
                record(str(request.url), response.headers)
            except Exception as exc:  # never break a provider call
                logger.debug("Rate-limit header capture failed: {}", exc)
            return response

        module.AsyncClient.send = send
