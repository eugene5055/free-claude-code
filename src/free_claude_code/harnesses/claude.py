"""Shared Claude Code environment policy for FCC client surfaces."""

import os
from collections.abc import Mapping
from pathlib import Path

from free_claude_code.harnesses.environment import client_environment

CLAUDE_CODE_AUTO_COMPACT_WINDOW = "190000"
# Set FCC_AUTO_COMPACT_WINDOW (tokens) to compact at a larger window, e.g. 262144
# for the pooled RTX 4090+5090 model.
AUTO_COMPACT_WINDOW_ENV = "FCC_AUTO_COMPACT_WINDOW"


def auto_compact_window(env: Mapping[str, str] | None = None) -> str:
    """Claude Code's compaction window: FCC_AUTO_COMPACT_WINDOW, else 190000."""
    value = (
        (os.environ if env is None else env).get(AUTO_COMPACT_WINDOW_ENV, "").strip()
    )
    return (
        value
        if value.isdigit() and int(value) >= 1024
        else CLAUDE_CODE_AUTO_COMPACT_WINDOW
    )


CLAUDE_BINARY_NAME = "claude"
# Claude Code mods FCC ships: the fcc-usage panel (/fcc-usage).
CLAUDE_MODS = (Path(__file__).resolve().parent / "claude_mods" / "fcc-usage",)


def claude_proxy_values(proxy_root_url: str, auth_token: str) -> dict[str, str]:
    """Settings shared by launched Claude sessions and the VS Code extension."""
    return {
        "ANTHROPIC_BASE_URL": proxy_root_url,
        "ANTHROPIC_AUTH_TOKEN": auth_token,
        "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1",
        # FCC uses Claude's separate classifier requests, not Anthropic server checks.
        "CLAUDE_CODE_AUTO_MODE_SERVER": "0",
        "CLAUDE_CODE_DISABLE_ADVISOR_TOOL": "1",
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW": auto_compact_window(),
        # Keep Claude waiting while FCC handles provider recovery.
        "CLAUDE_ENABLE_STREAM_WATCHDOG": "0",
        "CLAUDE_ENABLE_BYTE_WATCHDOG": "0",
        "API_FORCE_IDLE_TIMEOUT": "0",
        # Zero retains Claude's 5-minute non-streaming timeout. Use its timer cap.
        "API_TIMEOUT_MS": "2147483647",
        "DISABLE_AUTOUPDATER": "1",
        "DISABLE_FEEDBACK_COMMAND": "1",
        "DISABLE_ERROR_REPORTING": "1",
    }


def build_claude_proxy_env(
    *,
    proxy_root_url: str,
    auth_token: str,
    base_env: Mapping[str, str],
) -> dict[str, str]:
    """Return the canonical environment for Claude Code proxy sessions."""

    # Claude's aggregate traffic flag also suppresses gateway model discovery.
    updates = claude_proxy_values(proxy_root_url, auth_token)
    existing = [
        d for d in base_env.get("CLAUDE_CODE_PLUGIN_DIRS", "").split(os.pathsep) if d
    ]
    mods = [str(d) for d in CLAUDE_MODS if d.is_dir() and str(d) not in existing]
    if mods:
        updates["CLAUDE_CODE_PLUGIN_DIRS"] = os.pathsep.join([*existing, *mods])
    return client_environment(
        base_env,
        proxy_root_url=proxy_root_url,
        remove_prefixes=("ANTHROPIC_",),
        remove_keys=("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",),
        updates=updates,
    )
