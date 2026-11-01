"""One model name for the same model on every provider that serves it.

`nvidia_nim/z-ai/glm-5.3`, `alibaba_cloud/ZHIPU/GLM-5.3` and `open_router/z-ai/glm-5.3:free` are all the
model "glm-5.3". Claude model pickers list each name once (api/model_catalog.py, as `anthropic/fcc/glm-5.3`),
and a request for it is routed to the providers that serve it, fastest first; the executor moves on to the
next one when a provider is rate limited or fails, but never to a different model.
Any other route whose model belongs to a group gets the group's other providers as its first fallbacks.

Members are the free refs model_usability saw answer, so a dead provider drops out by itself. They are tried
fastest first by measured time to first output; a provider not timed yet goes first now and then
(_EXPLORE) so it gets measured, and otherwise after the timed ones.
"""

from __future__ import annotations

import random
from collections.abc import Collection, Iterable

from . import model_usability

GROUP_PROVIDER = "fcc"
_EXPLORE = 0.15
_STRIP_SUFFIXES = (":free",)


def canonical_name(provider_model_ref: str) -> str:
    """The provider-independent model name: last path segment, lower case, no `:free`."""
    name = provider_model_ref.rpartition("/")[2].strip().lower()
    for suffix in _STRIP_SUFFIXES:
        name = name.removesuffix(suffix)
    return name


def display_name(refs: Iterable[str]) -> str:
    """The name as providers spell it, preferring their capitalisation (GLM-5.3 over glm-5.3)."""
    spellings = [ref.rpartition("/")[2].removesuffix(":free") for ref in refs]
    return max(spellings, key=lambda s: (sum(c.isupper() for c in s), -len(s)), default="")


def members(name: str, provider_ids: Collection[str]) -> list[str]:
    """Usable refs of the model `name`, fastest first (see the module docstring)."""
    state = model_usability.snapshot()
    found = {
        ref: entry
        for ref, entry in state.items()
        if entry.get("usable")
        and model_usability.offered(ref)
        and ref.partition("/")[0] in provider_ids
        and canonical_name(ref) == name
    }
    timed = sorted((r for r in found if found[r].get("ttft") is not None), key=lambda r: found[r]["ttft"])
    untimed = sorted((r for r in found if found[r].get("ttft") is None), key=lambda r: -found[r].get("at", 0))
    if untimed and timed and random.random() < _EXPLORE:
        return [untimed[0], *timed, *untimed[1:]]
    return timed + untimed


def group_by_name(refs: Iterable[str]) -> dict[str, list[str]]:
    """Refs grouped by canonical name, keeping first-seen order."""
    groups: dict[str, list[str]] = {}
    for ref in refs:
        groups.setdefault(canonical_name(ref), []).append(ref)
    return groups
