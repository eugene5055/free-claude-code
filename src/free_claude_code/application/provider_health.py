"""Which providers are rate limited right now, and which models have recently worked.

The request executor uses both to switch models by itself when a provider refuses with a rate limit:
- a provider that answered 429 is "limited" until its reset (the soonest exhausted window from its own rate-limit
  headers, else a backoff that grows on repeated hits), so later requests try other candidates first and skip the
  provider's retry wait; it stays the last resort, never dropped;
- a model that completed a request is "proven", so models of other providers can be added as automatic fallbacks
  without guessing which ones the user's account can run with Claude Code's tools. Only offered (free, not blocked)
  models are used (model_usability.offered).

State lives in ~/.fcc/provider_health.json so it survives a restart of the server.
"""

from __future__ import annotations

import json
import os
import threading
import time

from loguru import logger

from free_claude_code.config import paths

from . import kimi_plan, model_usability, rate_limit_headers

_BACKOFF_START, _BACKOFF_MAX = 60.0, 900.0  # seconds a limited provider rests without a header reset
_HIT_MEMORY = 1800.0  # a 429 within this long of the last one counts as a repeat (longer backoff)
_PROVEN_MAX_AGE = 7 * 86400.0
_AUTO_FALLBACKS = 3

_lock = threading.Lock()
_state: dict[str, dict] = {"limited": {}, "proven": {}}
_loaded_from = None


def _path():
    """Resolved per call, so a changed config directory (tests, FCC_HOME) is honored."""
    return paths.config_dir_path() / "provider_health.json"


def _load() -> None:
    global _loaded_from
    path = _path()
    if path == _loaded_from:
        return
    _loaded_from = path
    _state.clear()
    _state.update({"limited": {}, "proven": {}})
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if isinstance(data, dict):
        for key in ("limited", "proven"):
            if isinstance(data.get(key), dict):
                _state[key] = data[key]


def _save() -> None:
    path = _path()
    tmp = path.with_suffix(".tmp")
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        tmp.write_text(json.dumps(_state), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        logger.debug("Could not save provider health: {}", exc)


def _header_reset(provider_id: str, now: float) -> float | None:
    """The soonest future reset among the provider's exhausted rate-limit windows, from its own headers."""
    windows = rate_limit_headers.snapshot().get(provider_id) or {}
    resets = []
    for window in windows.values():
        reset = window.get("reset")
        exhausted = window.get("remaining") == 0 or (window.get("utilization") or 0) >= 1
        if exhausted and isinstance(reset, (int, float)) and reset > now:
            resets.append(float(reset))
    # Kimi Code sends no rate-limit headers; its plan usage says when a used-up window resets.
    plan = kimi_plan.exhausted(provider_id, now)
    if plan is not None:
        resets.append(plan[0])
    return min(resets) if resets else None


def mark_limited(provider_id: str, now: float | None = None) -> float:
    """Record a rate-limit refusal; returns when the provider is expected to take requests again."""
    now = time.time() if now is None else now
    with _lock:
        _load()
        entry = _state["limited"].get(provider_id) or {}
        hits = entry.get("hits", 0) + 1 if now - entry.get("at", 0) < _HIT_MEMORY else 1
        until = _header_reset(provider_id, now) or now + min(_BACKOFF_START * 2 ** (hits - 1), _BACKOFF_MAX)
        _state["limited"][provider_id] = {"until": until, "hits": hits, "at": now}
        _save()
    logger.info(
        "Rate limit: provider {} rests until {} (hit {}); other models are tried first",
        provider_id,
        time.strftime("%H:%M:%S", time.localtime(until)),
        hits,
    )
    return until


def limited_until(provider_id: str, now: float | None = None) -> float | None:
    now = time.time() if now is None else now
    with _lock:
        _load()
        until = (_state["limited"].get(provider_id) or {}).get("until", 0)
    return until if until > now else None


def record_success(provider_id: str, provider_model: str, now: float | None = None) -> None:
    """A model completed a request: its provider takes requests again and the model counts as proven."""
    now = time.time() if now is None else now
    ref = f"{provider_id}/{provider_model}"
    with _lock:
        _load()
        changed = _state["limited"].pop(provider_id, None) is not None
        entry = _state["proven"].get(ref)
        if entry is None or now - entry.get("last", 0) > 60 or changed:
            _state["proven"][ref] = {
                "provider_id": provider_id,
                "provider_model": provider_model,
                "last": now,
                "count": (entry or {}).get("count", 0) + 1,
            }
            changed = True
        if changed:
            _save()


def proven_models(
    exclude_refs: set[str], now: float | None = None, limit: int = _AUTO_FALLBACKS
) -> list[tuple[str, str]]:
    """(provider_id, provider_model) of offered models that worked within the last week, newest first, one per
    provider, skipping limited providers and the given refs."""
    now = time.time() if now is None else now
    with _lock:
        _load()
        entries = sorted(_state["proven"].values(), key=lambda e: -e.get("last", 0))
        limited = {p for p, e in _state["limited"].items() if e.get("until", 0) > now}
    out, providers = [], set()
    for entry in entries:
        provider_id, model = entry.get("provider_id"), entry.get("provider_model")
        if not provider_id or not model or now - entry.get("last", 0) > _PROVEN_MAX_AGE:
            continue
        ref = f"{provider_id}/{model}"
        if provider_id in limited or provider_id in providers or ref in exclude_refs:
            continue
        if not model_usability.offered(ref):
            continue
        providers.add(provider_id)
        out.append((provider_id, model))
        if len(out) >= limit:
            break
    return out


def snapshot() -> dict:
    with _lock:
        _load()
        return json.loads(json.dumps(_state))
