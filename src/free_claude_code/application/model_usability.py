"""Which catalog models actually answer, so model pickers only list usable ones.

A model is "usable" once it completed a request, and "dead" when its provider refused it for a reason that
will not pass by itself: no credits, no access, bad key, unknown model. Rate limits and timeouts change
nothing, since they pass. Real traffic (application/execution.py) and the `fcc-probe-models` script both
feed this; model_catalog hides every model that is not usable for Claude clients. Each entry also keeps
the average time to first output (ttft), which model_groups uses to try the fastest provider first.

State lives in ~/.fcc/model_usability.json. Until that file exists nothing is hidden, so a fresh install
still lists its whole catalog.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time

from loguru import logger

from free_claude_code.config.paths import config_dir_path
from free_claude_code.core.failures import ExecutionFailure, FailureKind

PATH = config_dir_path() / "model_usability.json"
# 400s are usually about the request, so only these messages make a 400 count as a dead model.
_DEAD_400 = re.compile(
    r"credit|balance|billing|quota|insufficient|payment|not found|does not exist|no such model|"
    r"not supported|unsupported model|invalid model|not available|deprecated|decommissioned|no access",
    re.IGNORECASE,
)
_DEAD_STATUS = {401, 402, 403, 404, 410}
# Only free models are offered (user, 2026-10-23): providers with a free tier that cannot bill, plus OpenRouter's
# `:free` variants. Alibaba Cloud bills once its free quota runs out, so it counts as paid.
FREE_PROVIDERS = frozenset(
    {"nvidia_nim", "groq", "cloudflare", "gemini", "mistral", "cerebras", "llm7", "ollama", "lmstudio", "llamacpp"}
)
# Flat-rate subscriptions are offered too: the Alibaba Token Plan (qwencloud, user 2026-10-31) and the Kimi Code
# plan (kimi_code, user 2026-11-01) have fixed monthly prices with usage windows, so they cannot run up a bill.
FLAT_RATE_PROVIDERS = frozenset({"qwencloud", "kimi_code"})
_TTFT_WEIGHT = 0.3  # newest sample's share of the running time-to-first-output average

_lock = threading.Lock()
_state: dict[str, dict] = {}
_mtime: float | None = None


def _load() -> None:
    """Re-read the file when another process (the probe) changed it."""
    global _state, _mtime
    try:
        mtime = PATH.stat().st_mtime
    except OSError:
        _state, _mtime = {}, None
        return
    if mtime == _mtime:
        return
    try:
        data = json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    _state = data if isinstance(data, dict) else {}
    _mtime = mtime


def _save() -> None:
    global _mtime
    tmp = PATH.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(_state, indent=0, sort_keys=True), encoding="utf-8")
        os.replace(tmp, PATH)
        _mtime = PATH.stat().st_mtime
    except OSError as exc:
        logger.debug("Could not save model usability: {}", exc)


def _set(ref: str, usable: bool, error: str | None = None) -> None:
    with _lock:
        _load()
        entry = _state.get(ref) or {}
        if entry.get("usable") == usable and time.time() - entry.get("at", 0) < 600:
            return
        timing = {k: entry[k] for k in ("ttft", "ttft_n") if k in entry}
        _state[ref] = {"usable": usable, "at": time.time(), **timing, **({"error": error[:300]} if error else {})}
        _save()
    if not usable:
        logger.info("Model {} is not usable, hidden from model lists: {}", ref, (error or "")[:160])


def is_free(provider_model_ref: str) -> bool:
    provider = provider_model_ref.partition("/")[0]
    if provider == "open_router":
        return provider_model_ref.endswith(":free")
    return provider in FREE_PROVIDERS


# Never offered on any provider (user, 2026-10-22; Duo's model_selection.BLOCKED is the same rule).
_BLOCKED = re.compile(r"nemotron-[0-9.]+-(super|lightning)", re.IGNORECASE)


def offered(provider_model_ref: str) -> bool:
    """Free or flat-rate, and not blocked: the models fcc lists and routes to."""
    flat_rate = provider_model_ref.partition("/")[0] in FLAT_RATE_PROVIDERS
    return (is_free(provider_model_ref) or flat_rate) and not _BLOCKED.search(provider_model_ref)


def is_dead_failure(failure: ExecutionFailure) -> bool:
    if failure.kind in {FailureKind.RATE_LIMIT, FailureKind.TIMEOUT, FailureKind.OVERLOADED}:
        return False
    if failure.status_code in _DEAD_STATUS or failure.kind in {
        FailureKind.AUTHENTICATION,
        FailureKind.PERMISSION,
    }:
        return True
    return failure.status_code == 400 and bool(_DEAD_400.search(failure.message or ""))


def record_success(provider_model_ref: str) -> None:
    _set(provider_model_ref, True)


def record_failure(provider_model_ref: str, failure: ExecutionFailure) -> None:
    if is_dead_failure(failure):
        _set(provider_model_ref, False, failure.message)


def record_latency(provider_model_ref: str, seconds: float) -> None:
    """Seconds to the first streamed output, averaged (model_groups puts the fastest provider first)."""
    with _lock:
        _load()
        entry = _state.setdefault(provider_model_ref, {"usable": True, "at": time.time()})
        old = entry.get("ttft")
        entry["ttft"] = round(seconds if old is None else old + _TTFT_WEIGHT * (seconds - old), 3)
        entry["ttft_n"] = entry.get("ttft_n", 0) + 1
        _save()


def record_probe(provider_model_ref: str, usable: bool, error: str | None = None) -> None:
    _set(provider_model_ref, usable, error)


def usable_refs() -> set[str] | None:
    """Offered refs that answered; None when nothing was ever checked (then nothing is hidden)."""
    with _lock:
        _load()
        if _mtime is None:
            return None
        return {ref for ref, entry in _state.items() if entry.get("usable") and offered(ref)}


def snapshot() -> dict[str, dict]:
    with _lock:
        _load()
        return json.loads(json.dumps(_state))
