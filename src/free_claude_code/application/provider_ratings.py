"""Per-provider per-kind request ratings for fcc-claude.

Tracks success rate and completion time for each provider so ModelRouter can
pick the best provider dynamically instead of relying on a static MODEL.
Adapted from Duo's lesson_book.py ratings logic (DECAY, LEAST, RELIABLE,
TIME_TARGETS, best(), eligible()).

Never logs or prints API keys or credentials.
"""

import math
import sqlite3
from pathlib import Path

from free_claude_code.config import paths

DECAY = 0.97
LEAST = 5
RELIABLE = 0.7
TIME_TARGETS = {"messages": 300}
DEFAULT_TIME_TARGET = 300
TIMING_COVERAGE = 0.9


def _db_path():
    """Resolved per call, so a changed config directory (tests, FCC_HOME) is honored."""
    return paths.config_dir_path() / "provider_ratings.db"


def _connect():
    path = _db_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=5)
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS ratings (
            provider TEXT, kind TEXT, runs REAL DEFAULT 0, ok REAL DEFAULT 0,
            seconds REAL DEFAULT 0, timed REAL DEFAULT 0,
            PRIMARY KEY (provider, kind));
    """
    )
    return con


def record_outcome(provider, kind, passed, seconds=None):
    """Record the outcome of one provider request.

    passed: True when the request completed successfully, False on failure.
    seconds: elapsed wall-clock time (failed attempts included when known).
    """
    valid_time = (
        isinstance(seconds, (int, float))
        and not isinstance(seconds, bool)
        and math.isfinite(seconds)
        and seconds > 0
    )
    seconds = float(seconds) if valid_time else 0.0
    timed = 1.0 if valid_time else 0.0
    ok = 1.0 if passed else 0.0
    con = _connect()
    try:
        with con:
            con.execute(
                "INSERT INTO ratings (provider,kind,runs,ok,seconds,timed) "
                "VALUES (?,?,1,?,?,?) "
                "ON CONFLICT(provider, kind) DO UPDATE SET "
                "runs=runs*?+1, ok=ok*?+excluded.ok, "
                "seconds=seconds*?+excluded.seconds, timed=timed*?+excluded.timed",
                (provider, kind, ok, seconds, DECAY, DECAY, DECAY, DECAY, DECAY),
            )
    finally:
        con.close()


def _ratings_rows(con):
    return con.execute(
        "SELECT provider, kind, runs, ok, seconds, timed FROM ratings"
    ).fetchall()


def ratings(con=None, least=LEAST):
    """Per-kind/provider absolute ratings. Omits rows without sufficient data."""
    own = con is None
    if own:
        con = _connect()
    try:
        rows = _ratings_rows(con)
    finally:
        if own:
            con.close()
    table = {}
    for provider, kind, runs, ok, seconds, timed in rows:
        success = ok / runs if runs > 0 else 0.0
        coverage = timed / runs if runs > 0 else 0.0
        duration = seconds / timed / success if timed > 0 and success > 0 else None
        if runs < least or runs <= 0:
            reason = "not enough data"
        elif ok > 0 and coverage < TIMING_COVERAGE:
            reason = "not enough timing data"
        else:
            reason = None
        target = TIME_TARGETS.get(kind, DEFAULT_TIME_TARGET)
        score = (
            None
            if reason
            else 0
            if ok <= 0
            else round(100 * success * min(1.0, target / duration))
        )
        row = {
            "runs": round(runs, 1),
            "success": success,
            "seconds": duration,
            "score": score,
            "sufficient": reason is None,
            "reason": reason,
            "timing_coverage": coverage,
            "target_seconds": target,
        }
        if reason is None and duration is not None:
            table.setdefault(kind, {}).setdefault(provider, row)
    return table


def eligible(row):
    """Enough scoring evidence, reliable success and a usable completion time."""
    seconds = row.get("seconds")
    return (
        row.get("sufficient") is True
        and row.get("score") is not None
        and row.get("success", 0) >= RELIABLE
        and isinstance(seconds, (int, float))
        and not isinstance(seconds, bool)
        and math.isfinite(seconds)
        and seconds > 0
    )


def best(kind, candidates):
    """The highest-scoring reliable provider for kind, or None when none is rated yet.

    Ties favour quicker success (lower seconds_per_completion).
    """
    rated = ratings().get(kind, {})
    scored = [
        (rated[p]["score"], -rated[p]["seconds"], p)
        for p in candidates
        if p in rated and eligible(rated[p])
    ]
    return max(scored)[2] if scored else None


def expected_seconds(provider, kind):
    """How long this provider takes to finish this kind of request, or None until rated."""
    return ratings().get(kind, {}).get(provider, {}).get("seconds")


def _kind_for_model(model_name: str) -> str:
    """Map a Claude model name to a task kind for provider rating."""
    normalized = (model_name or "").lower()
    if "fable" in normalized or "haiku" in normalized:
        return "advice"
    if "opus" in normalized:
        return "edit"
    if "sonnet" in normalized:
        return "test"
    return "messages"


def _configured_providers(settings) -> list:
    """Return provider IDs that have credentials configured, in catalog order."""
    from free_claude_code.config.provider_catalog import PROVIDER_CATALOG

    configured = []
    for provider_id in settings.provider_ids:
        descriptor = PROVIDER_CATALOG.get(provider_id)
        if descriptor is None or descriptor.credential_attr is None:
            continue
        keys = getattr(settings, descriptor.credential_attr, None)
        if keys:
            configured.append(provider_id)
    return configured
