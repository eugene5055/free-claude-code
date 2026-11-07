"""fcc-probe-models: send one small tool-calling request to each catalog model so model_usability knows
which ones answer. The server records the verdicts itself (application/execution.py), so a request
that a fallback answered proves nothing about the model it was meant for.

Usage: fcc-probe-models [--all] [--only PREFIX] [--url URL]
By default it checks models never checked, dead ones older than a day and usable ones older than a week.
"""


import argparse
import asyncio
import collections
import os
import sys
import time

import httpx

from free_claude_code.application import model_usability
from free_claude_code.config.paths import config_dir_path

# Progress for the Claude Code status line (~/.claude/statusline-duo.sh reads it).
_LOG = config_dir_path() / "probe.log"
_log_file = None


def _out(line: str) -> None:
    print(line, flush=True)
    if _log_file is not None:
        _log_file.write(line + "\n")
        _log_file.flush()


_DEAD_RECHECK, _USABLE_RECHECK = 86400.0, 7 * 86400.0
_PER_PROVIDER, _TOTAL = 2, 16
# Claude Code's own max_tokens: a paid model whose credit covers a tiny request but not a real one is dead.
_BODY = {
    "max_tokens": 32000,
    "tools": [
        {
            "name": "ping",
            "description": "Confirms the model can call tools.",
            "input_schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
        }
    ],
    "messages": [{"role": "user", "content": "Call the ping tool with ok=true."}],
}


def _due(ref: str, state: dict[str, dict], now: float) -> bool:
    entry = state.get(ref)
    if not entry:
        return True
    age = now - entry.get("at", 0)
    return age > (_USABLE_RECHECK if entry.get("usable") else _DEAD_RECHECK)


async def _probe(client, url, key, ref, sems, total, results) -> None:
    async with total, sems[ref.partition("/")[0]]:
        try:
            response = await client.post(
                f"{url}/v1/messages",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                json={"model": f"anthropic/{ref}", **_BODY},
                timeout=180,
            )
            detail = "" if response.status_code == 200 else response.text[:200].replace("\n", " ")
        except httpx.HTTPError as exc:
            detail = f"{type(exc).__name__}: {exc}"
    entry = model_usability.snapshot().get(ref) or {}
    verdict = "usable" if entry.get("usable") else ("dead" if entry else "unknown")
    results[verdict] += 1
    _out(f"{verdict:8} {ref}{'  ' + detail[:140] if detail and verdict != 'usable' else ''}")


async def _main(args) -> int:
    key = os.environ.get("FCC_PROBE_KEY", "freecc")
    async with httpx.AsyncClient() as client:
        listing = await client.get(
            f"{args.url}/v1/models", params={"view": "messages"}, headers={"x-api-key": key}, timeout=60
        )
        listing.raise_for_status()
        refs = sorted({m["provider_model_ref"] for m in listing.json()["data"] if m.get("provider_model_ref")})
        if args.only:
            refs = [r for r in refs if r.startswith(args.only)]
        state, now = model_usability.snapshot(), time.time()
        todo = refs if args.all else [r for r in refs if _due(r, state, now)]
        _out(f"{len(todo)} of {len(refs)} models to check")
        sems = collections.defaultdict(lambda: asyncio.Semaphore(_PER_PROVIDER))
        total, results = asyncio.Semaphore(_TOTAL), collections.Counter()
        await asyncio.gather(*(_probe(client, args.url, key, r, sems, total, results) for r in todo))
    usable = model_usability.usable_refs() or set()
    _out(f"done: {dict(results)}; {len(usable & set(refs))} usable models listed")
    return 0


def main() -> int:
    global _log_file
    parser = argparse.ArgumentParser(prog="fcc-probe-models", description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", help="recheck every model")
    parser.add_argument("--only", help="only refs starting with this, e.g. nvidia_nim/")
    parser.add_argument("--url", default="http://127.0.0.1:8082")
    args = parser.parse_args()
    _LOG.unlink(missing_ok=True)
    with _LOG.open("w", encoding="utf-8") as _log_file:
        return asyncio.run(_main(args))


if __name__ == "__main__":
    sys.exit(main())
