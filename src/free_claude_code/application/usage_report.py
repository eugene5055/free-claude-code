"""FCC usage report: session totals, provider limits, and what drives usage.

One builder feeds both the ``/usage`` web page (JSON) and the ``/fcc-usage``
Claude Code command (plain text laid out like Claude's own Usage tab).
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol, TypeVar

from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings
from free_claude_code.application.usage_pricing import has_known_price

from . import rate_limit_headers
from .provider_quota import ProviderQuotaService

_T = TypeVar("_T")


class UsageDatabase(Protocol):
    """What the usage report and tracker need of FCC's SQLite database (runtime.sqlite_database.SQLiteDatabase)."""

    async def start(self) -> None: ...

    async def run(self, operation: Callable[[Any], _T], *, write: bool = True) -> _T: ...


PERIODS = {"day": ("Last 24h", 86400), "week": ("Last 7 days", 7 * 86400)}
_LARGE_CONTEXT = 150_000
_LARGE_MISS = 100_000
_MISS_FLOOR = 1024
_BAR_WIDTH = 50
_QUOTA_TTL = 60.0
_quota_cache: tuple[float, dict[str, Any]] | None = None
_quota_lock = asyncio.Lock()


# ---------------------------------------------------------------- data access


async def _rows(database: UsageDatabase, sql: str, params: tuple = ()) -> list[dict]:
    await database.start()

    def operation(conn):
        return [dict(row) for row in conn.execute(sql, params).fetchall()]

    return await database.run(operation, write=False)


async def _session_rows(database: UsageDatabase, session_id: str | None) -> tuple[str | None, list[dict]]:
    if not session_id:
        latest = await _rows(
            database,
            "SELECT session_id FROM request_usage WHERE session_id IS NOT NULL "
            "ORDER BY created_at DESC LIMIT 1",
        )
        session_id = latest[0]["session_id"] if latest else None
    if not session_id:
        return None, []
    rows = await _rows(
        database,
        "SELECT * FROM request_usage WHERE session_id = ? ORDER BY created_at",
        (session_id,),
    )
    return session_id, rows


async def recent_sessions(database: UsageDatabase, limit: int = 20) -> list[dict]:
    return await _rows(
        database,
        "SELECT session_id, COUNT(*) AS requests, MIN(created_at) AS first, "
        "MAX(created_at) AS last, SUM(cost_usd) AS cost FROM request_usage "
        "WHERE session_id IS NOT NULL GROUP BY session_id ORDER BY last DESC LIMIT ?",
        (limit,),
    )


# ------------------------------------------------------------------- sections


def _session_section(session_id: str | None, rows: list[dict]) -> dict | None:
    if not rows:
        return None
    by_model: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = f"{row['provider_id']}/{row['model']}"
        entry = by_model.setdefault(
            label,
            {"model": label, "requests": 0, "input": 0, "output": 0,
             "cache_read": 0, "cache_write": 0, "cost": 0.0,
             "known_price": has_known_price(row["model"])},
        )
        entry["requests"] += 1
        entry["input"] += row["input_tokens"] or 0
        entry["output"] += row["output_tokens"] or 0
        entry["cache_read"] += row["cache_read_tokens"] or 0
        entry["cache_write"] += row["cache_write_tokens"] or 0
        entry["cost"] += row["cost_usd"] or 0.0

    input_total = sum(m["input"] + m["cache_read"] + m["cache_write"] for m in by_model.values())
    cache_read = sum(m["cache_read"] for m in by_model.values())
    misses = [
        row for row in rows[1:]
        if not row["cache_read_tokens"]
        and (row["input_tokens"] or 0) + (row["cache_write_tokens"] or 0) > _MISS_FLOOR
    ]
    now = time.time()
    return {
        "id": session_id,
        "requests": len(rows),
        "failed": sum(1 for row in rows if (row["status_code"] or 200) >= 400),
        "cost": sum(m["cost"] for m in by_model.values()),
        "unknown_pricing": any(not m["known_price"] for m in by_model.values()),
        "api_ms": sum(row["duration_ms"] or 0 for row in rows),
        "wall_ms": (rows[-1]["created_at"] - rows[0]["created_at"]) * 1000
        + (rows[0]["duration_ms"] or 0),
        "first": rows[0]["created_at"],
        "last": rows[-1]["created_at"],
        "by_model": sorted(by_model.values(), key=lambda m: -m["cost"]),
        "cache": {
            "cached_pct": 100 * cache_read / input_total if input_total else 0.0,
            "misses": len(misses),
            "last_miss_ago": now - misses[-1]["created_at"] if misses else None,
            "last_activity_ago": now - rows[-1]["created_at"],
        },
    }


def _header_windows(provider_windows: dict[str, dict]) -> list[dict]:
    now = time.time()
    labels = {
        "requests": "Requests",
        "tokens": "Tokens",
        "input-tokens": "Input tokens",
        "output-tokens": "Output tokens",
        "unified-5h": "Current session (5h)",
        "unified-7d": "Current week",
    }
    out = []
    for key, window in sorted(provider_windows.items()):
        limit, remaining = window.get("limit"), window.get("remaining")
        reset = window.get("reset")
        if window.get("utilization") is not None:
            used = 100 * float(window["utilization"])
            detail = None
        elif limit:
            used = 100 * (limit - (remaining or 0)) / limit
            detail = f"{_count(remaining or 0)} of {_count(limit)} left"
        else:
            continue
        if reset is not None and reset < now:
            used, detail = 0.0, "reset since the last request"
            reset = None
        out.append({
            "label": labels.get(key, key.replace("-", " ").capitalize()),
            "used_pct": max(0.0, min(100.0, used)),
            "resets_at": reset,
            "detail": detail,
            "source": "headers",
            "observed_at": window.get("observed_at"),
        })
    return out


async def _quota_groups(settings: Settings, refresh: bool) -> dict[str, Any]:
    global _quota_cache
    async with _quota_lock:
        if not refresh and _quota_cache and time.time() - _quota_cache[0] < _QUOTA_TTL:
            return _quota_cache[1]
        groups = await ProviderQuotaService(settings).get_all_quotas()
        _quota_cache = (time.time(), groups)
        return groups


async def _limits_section(settings: Settings, refresh: bool) -> list[dict]:
    groups = await _quota_groups(settings, refresh)
    headers = rate_limit_headers.snapshot()
    providers: dict[str, dict] = {}
    for group in groups.values():
        for quota in group.providers:
            providers[quota.provider_id] = {
                "provider_id": quota.provider_id,
                "display_name": quota.display_name,
                "group": group.display_name,
                "error": quota.error,
                "windows": [
                    {
                        "label": w.label.replace("_", " ").capitalize(),
                        "used_pct": max(0.0, min(100.0, w.used_pct)),
                        "resets_at": w.resets_at,
                        "detail": (
                            f"${w.raw_used:,.2f} of ${w.raw_limit:,.2f} used"
                            if "credit" in w.label and w.raw_limit
                            else f"{_count(w.raw_used)} of {_count(w.raw_limit)} used"
                            if w.raw_limit
                            else None
                        ),
                        "source": "quota api",
                    }
                    for w in quota.windows
                ],
            }
    for provider_id, windows in headers.items():
        descriptor = PROVIDER_CATALOG.get(provider_id)
        entry = providers.setdefault(
            provider_id,
            {"provider_id": provider_id,
             "display_name": descriptor.display_name if descriptor else provider_id,
             "group": "OTHER PROVIDERS", "error": None, "windows": []},
        )
        entry["windows"].extend(_header_windows(windows))
    return sorted(
        providers.values(),
        key=lambda p: (not p["windows"], -max((w["used_pct"] for w in p["windows"]), default=0), p["display_name"]),
    )


async def _contributing_section(database: UsageDatabase, period: str) -> dict:
    label, seconds = PERIODS[period]
    rows = await _rows(
        database,
        "SELECT provider_id, input_tokens, output_tokens, cache_read_tokens, "
        "cache_write_tokens FROM request_usage WHERE created_at >= ?",
        (int(time.time() - seconds),),
    )
    total = large = missed = 0
    by_provider: dict[str, int] = {}
    for row in rows:
        inp, out = row["input_tokens"] or 0, row["output_tokens"] or 0
        read, write = row["cache_read_tokens"] or 0, row["cache_write_tokens"] or 0
        tokens = inp + out + read + write
        total += tokens
        by_provider[row["provider_id"]] = by_provider.get(row["provider_id"], 0) + tokens
        if inp + read + write > _LARGE_CONTEXT:
            large += tokens
        if inp + write > _LARGE_MISS:
            missed += tokens
    pct = (lambda part: 100 * part / total if total else 0.0)
    return {
        "period": period,
        "label": label,
        "requests": len(rows),
        "tokens": total,
        "large_context_pct": pct(large),
        "cache_miss_pct": pct(missed),
        "by_provider": [
            {"provider_id": pid, "pct": pct(tokens)}
            for pid, tokens in sorted(by_provider.items(), key=lambda item: -item[1])
        ],
    }


async def build_report(
    database: UsageDatabase | None,
    settings: Settings,
    *,
    session_id: str | None = None,
    period: str = "day",
    refresh: bool = False,
) -> dict:
    period = period if period in PERIODS else "day"
    limits = await _limits_section(settings, refresh)
    if database is None:
        return {"generated_at": time.time(), "session": None, "sessions": [],
                "limits": limits, "contributing": None, "storage": "unavailable"}
    found_id, rows = await _session_rows(database, session_id)
    return {
        "generated_at": time.time(),
        "requested_session": session_id,
        "session": _session_section(found_id, rows),
        "sessions": await recent_sessions(database),
        "limits": limits,
        "contributing": await _contributing_section(database, period),
        "storage": "ok",
    }


# ------------------------------------------------------------------ rendering


def _count(value: float | int | None) -> str:
    value = value or 0
    for size, suffix in ((1e9, "b"), (1e6, "m"), (1e3, "k")):
        if value >= size:
            return f"{value / size:.1f}{suffix}"
    return f"{value:g}" if isinstance(value, float) else str(value)


def _duration(ms: float) -> str:
    seconds = int(ms / 1000)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{ms / 1000:.1f}s" if ms < 10000 else f"{secs}s"


def _ago(seconds: float | None) -> str:
    if seconds is None:
        return "never"
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h {seconds % 3600 // 60}m ago"
    return f"{seconds // 86400}d ago"


def _timezone_name() -> str:
    if os.environ.get("TZ"):
        return os.environ["TZ"].lstrip(":")
    try:
        target = os.readlink("/etc/localtime")
        return target.split("zoneinfo/", 1)[1]
    except (OSError, IndexError):
        return time.strftime("%Z")


def _reset_text(epoch: float) -> str:
    when = datetime.fromtimestamp(epoch)
    clock = when.strftime("%I:%M%p").lstrip("0").lower()
    if when.date() == datetime.now().date():
        text = clock
    else:
        text = f"{when.strftime('%b')} {when.day}, {clock}"
    return f"Resets {text} ({_timezone_name()})"


def _bar(pct: float) -> str:
    halves = round(pct / 100 * _BAR_WIDTH * 2)
    full, half = divmod(halves, 2)
    return ("█" * full + ("▌" if half else "")).ljust(_BAR_WIDTH)


def render_text(report: dict) -> str:
    lines: list[str] = []
    add = lines.append
    session = report.get("session")

    add("Session")
    add("")
    if session is None:
        wanted = report.get("requested_session")
        add(
            f"No fcc requests recorded for session {wanted[:8]}… yet."
            if wanted
            else "No fcc requests with a Claude session id have been recorded yet."
        )
    else:
        note = " (costs may be inaccurate due to usage of unknown models)" if session["unknown_pricing"] else ""
        add(f"Session id:            {session['id']}")
        add(f"Total cost:            ${session['cost']:.2f}{note}")
        add(f"Total duration (API):  {_duration(session['api_ms'])}")
        add(f"Total duration (wall): {_duration(session['wall_ms'])}")
        add(f"Requests:              {session['requests']}" + (f" ({session['failed']} failed)" if session["failed"] else ""))
        add("Usage by model:")
        width = max(len(m["model"]) for m in session["by_model"])
        for m in session["by_model"]:
            add(
                f"  {m['model'].rjust(width)}:  {_count(m['input'])} input, {_count(m['output'])} output, "
                f"{_count(m['cache_read'])} cache read, {_count(m['cache_write'])} cache write (${m['cost']:.2f})"
            )
        cache = session["cache"]
        misses = (
            "no misses"
            if not cache["misses"]
            else f"{cache['misses']} miss{'es' if cache['misses'] != 1 else ''} (last {_ago(cache['last_miss_ago'])})"
        )
        add(
            f"Prompt cache:          {session['requests']} request{'s' if session['requests'] != 1 else ''} · "
            f"{cache['cached_pct']:.0f}% of input tokens from cache · {misses} · "
            f"last activity {_ago(cache['last_activity_ago'])}"
        )

    add("")
    add("")
    add("Provider limits")
    add("Quota APIs plus the rate-limit headers each provider sends back with its responses.")
    no_data = []
    for provider in report.get("limits", []):
        if not provider["windows"]:
            no_data.append(provider["display_name"])
            continue
        for window in provider["windows"]:
            add("")
            add(f"{provider['display_name']} · {window['label']}")
            add(f"{_bar(window['used_pct'])} {window['used_pct']:.0f}% used")
            details = [window["detail"]] if window.get("detail") else []
            if window.get("resets_at"):
                details.append(_reset_text(window["resets_at"]))
            if details:
                add(" · ".join(details))
    if no_data:
        add("")
        add("No limit data yet: " + ", ".join(no_data))
        add(" These have no quota API and haven't sent rate-limit headers yet; bars appear after their next request.")

    contributing = report.get("contributing")
    if contributing:
        add("")
        add("")
        add("What's contributing to your fcc usage?")
        add("Every request through this fcc server, from all clients on this machine")
        add("")
        add(
            f"{contributing['label']} · {contributing['requests']} requests · {_count(contributing['tokens'])} tokens · "
            "these are independent characteristics of your usage, not a breakdown"
        )
        if contributing["tokens"]:
            add("")
            add(f"{contributing['large_context_pct']:.0f}% of your usage was at >150k context")
            add(" Longer sessions are more expensive even when cached. /compact mid-task, /clear when switching to new tasks.")
            add("")
            add(f"{contributing['cache_miss_pct']:.0f}% of your usage hit a >100k-token cache miss")
            add(" Uncached input is expensive, and often happens when sending a message to a session that has gone idle. /compact before stepping away keeps the cold-start small.")
            add("")
            add("By provider: " + " · ".join(
                f"{p['provider_id']} {p['pct']:.0f}%" for p in contributing["by_provider"][:6]
            ))
    add("")
    add("d to day · w to week · r to refresh · Esc to close")
    return "\n".join(lines) + "\n"
