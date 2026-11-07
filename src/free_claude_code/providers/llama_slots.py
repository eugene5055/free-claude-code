"""Slot scheduling for custom providers served by llama-server.

A llama-server with ``-np N`` decodes N requests at once. More than N in flight
interleave, evict each other's prompt cache and re-read 100K-token prompts.
This module keeps at most ``total_slots`` (from ``/props``) requests in flight,
queues the rest by priority, pins each request to a slot with ``id_slot``, and
moves a conversation's KV cache to disk (``/slots/{id}?action=save``) when a
different conversation needs its slot, restoring it (``action=restore``) when
the conversation returns.

Priority, highest first (verified against Claude Code 2.1.319 requests):

0. MAIN: an interactive session's main thread, and its auto-mode classifier.
1. SUBAGENT: requests carrying ``x-claude-code-agent-id`` (or
   ``cc_is_subagent=true`` in the system billing block).
2. DUO: Duo's own jobs, whose session id starts with ``d0d0d0d0``.
3. BACKGROUND: haiku-class calls without tools or agent id (titles, summaries).

Waiting requests age one class every ``AGE_SECONDS`` so nothing starves. While
a request waits, the caller's stream emits keep-alive frames, so Claude Code's
byte-idle watchdog does not kill it. Nothing is ever routed elsewhere: the
queue order is the only lever.

Correctness of a restore does not depend on FCC's bookkeeping: llama-server
compares the restored token sequence with the new prompt and reuses only the
common prefix, so a stale file only costs time. FCC's message-chain check only
decides whether a restore is worth doing.
"""

import asyncio
import contextlib
import contextvars
import hashlib
import heapq
import ipaddress
import itertools
import json
import time
from collections import OrderedDict
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any
from urllib.parse import urlsplit

import httpx
from loguru import logger

DUO_SESSION_PREFIX = "d0d0d0d0"
SESSION_HEADER = "x-claude-code-session-id"
AGENT_HEADER = "x-claude-code-agent-id"
SUBAGENT_MARKER = "cc_is_subagent=true"

AGE_SECONDS = 120.0
KEEPALIVE_SECONDS = 10.0
PROPS_TTL_SECONDS = 60.0
PROPS_RETRY_SECONDS = 30.0
SAVE_RECHECK_SECONDS = 600.0
MIN_SAVE_TOKENS = 8192
SAVE_TIMEOUT_SECONDS = 180.0
BACKGROUND_MAX_TOKENS = 4096

MESSAGES_KEEPALIVE = 'event: ping\ndata: {"type": "ping"}\n\n'
COMMENT_KEEPALIVE = ": keepalive\n\n"


class RequestClass(IntEnum):
    MAIN = 0
    SUBAGENT = 1
    DUO = 2
    BACKGROUND = 3


@dataclass(frozen=True)
class SlotOptions:
    """Tunables carried on the custom provider definition."""

    enabled: bool = True
    save_budget_mb: int = 204800
    subagent_thinking_budget: int = 4096
    low_effort_thinking_budget: int = 1024


@dataclass(frozen=True)
class RequestProfile:
    klass: RequestClass
    conversation: str | None
    chain: tuple[str, ...]
    input_tokens: int
    extras: Mapping[str, Any]


def header(headers: Mapping[str, str] | None, name: str) -> str:
    if not headers:
        return ""
    for key, value in headers.items():
        if key.lower() == name:
            return value or ""
    return ""


def classify(
    headers: Mapping[str, str] | None,
    *,
    requested_model: str | None,
    system_text: str,
    has_tools: bool,
    is_classifier: bool,
) -> RequestClass:
    session = header(headers, SESSION_HEADER)
    if session.lower().startswith(DUO_SESSION_PREFIX):
        return RequestClass.DUO
    if is_classifier:
        return RequestClass.MAIN
    subagent = bool(header(headers, AGENT_HEADER)) or SUBAGENT_MARKER in system_text
    if subagent:
        return RequestClass.SUBAGENT
    if "haiku" in (requested_model or "").lower() and not has_tools:
        return RequestClass.BACKGROUND
    return RequestClass.MAIN


def _strip_cache_control(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_cache_control(item)
            for key, item in value.items()
            if key != "cache_control"
        }
    if isinstance(value, list):
        return [_strip_cache_control(item) for item in value]
    return value


def message_chain(head: Any, messages: Iterable[Any]) -> tuple[str, ...]:
    """Cumulative hashes of (system+tools, message 1, message 2, ...).

    ``cache_control`` markers move between turns, so they are ignored.
    """
    digest = hashlib.sha256()
    out: list[str] = []
    for item in itertools.chain((head,), messages):
        if hasattr(item, "model_dump"):
            item = item.model_dump(mode="json", exclude_none=True)
        digest.update(
            json.dumps(_strip_cache_control(item), sort_keys=True, default=str).encode()
        )
        digest.update(b"\x00")
        out.append(digest.copy().hexdigest()[:24])
    return tuple(out)


def common_prefix(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    n = 0
    for left, right in zip(a, b, strict=False):
        if left != right:
            break
        n += 1
    return n


def restore_worthwhile(saved: tuple[str, ...], new: tuple[str, ...]) -> bool:
    """The saved chain must share the system head and at least half its messages."""
    if not saved or not new:
        return False
    shared = common_prefix(saved, new)
    return shared >= 1 and shared >= max(2, (len(saved) + 1) // 2)


def is_local_url(url: str) -> bool:
    """True for loopback, private and link-local hosts (where llama servers run)."""
    host = urlsplit(url).hostname or ""
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_private or address.is_link_local


def conversation_key(
    headers: Mapping[str, str] | None, chain: tuple[str, ...]
) -> str | None:
    session = header(headers, SESSION_HEADER)
    agent = header(headers, AGENT_HEADER) or "main"
    head = chain[0][:12] if chain else ""
    if session:
        return f"{session}:{agent}:{head}"
    if len(chain) >= 2:
        return f"anon:{chain[1]}"
    return None


_EFFORT_LOW = {"minimal", "low"}


def thinking_extras(
    klass: RequestClass,
    *,
    effort: str | None,
    thinking_off: bool,
    options: SlotOptions,
    max_tokens: int | None,
) -> dict[str, Any]:
    """llama-server per-request fields: reasoning budget and output cap.

    The client's reasoning level is still sent untouched (every lane sends one);
    the budget only bounds how long llama.cpp lets the model think.
    """
    extras: dict[str, Any] = {}
    if thinking_off:
        return extras
    low = (effort or "").lower() in _EFFORT_LOW
    if klass is RequestClass.BACKGROUND:
        extras["thinking_budget_tokens"] = 0
        cap = BACKGROUND_MAX_TOKENS
        extras["max_tokens_cap"] = min(max_tokens or cap, cap)
    elif klass is RequestClass.MAIN:
        pass
    elif low and options.low_effort_thinking_budget > 0:
        extras["thinking_budget_tokens"] = options.low_effort_thinking_budget
    elif klass is RequestClass.SUBAGENT and options.subagent_thinking_budget > 0:
        extras["thinking_budget_tokens"] = options.subagent_thinking_budget
    return extras


# Fields the transport merges into the outgoing Chat Completions body.
_BODY_EXTRAS: contextvars.ContextVar[Mapping[str, Any] | None] = contextvars.ContextVar(
    "fcc_llama_body_extras", default=None
)


async def apply_body_extras(body: Mapping[str, Any]) -> None:
    """Prepared-body hook for OpenAIChatTransport: pin the slot, cap thinking.

    The transport passes its mutable create body; anything else is left alone.
    """
    extras = _BODY_EXTRAS.get()
    if not extras or not isinstance(body, dict):
        return
    # llama.cpp fields are not OpenAI SDK kwargs: they travel in extra_body,
    # which the SDK merges over the top-level request.
    current = body.get("extra_body")
    extra: dict[str, Any] = dict(current) if isinstance(current, dict) else {}
    for key, value in extras.items():
        if key == "max_tokens_cap":
            for name in ("max_tokens", "max_completion_tokens"):
                if isinstance(body.get(name), int) and body[name] > value:
                    body[name] = value
            continue
        if key == "thinking_budget_tokens":
            # Compose with an exact budget set elsewhere (a reasoning format
            # mapping or the client): the tighter cap wins, never a looser one.
            for holder in (body, extra):
                existing = holder.get(key)
                if isinstance(existing, int) and existing >= 0:
                    value = min(value, existing)
            if key in body:
                body[key] = value
        extra[key] = value
    body["extra_body"] = extra


@dataclass
class _Slot:
    id: int
    busy: bool = False
    resident: str | None = None
    chain: tuple[str, ...] = ()
    tokens: int = 0
    last_used: float = 0.0


@dataclass
class _Saved:
    filename: str
    chain: tuple[str, ...]
    size: int
    tokens: int


@dataclass
class SlotLease:
    slot: int | None
    extras: Mapping[str, Any]
    waited_s: float = 0.0
    restored: bool = False


@dataclass(order=True)
class _Waiter:
    klass: int
    seq: int
    enqueued: float = field(compare=False)
    profile: RequestProfile = field(compare=False)
    future: asyncio.Future[_Slot] = field(compare=False)


class LlamaSlotScheduler:
    """Admission, priority and KV persistence for one llama-server."""

    def __init__(
        self,
        base_url: str,
        *,
        options: SlotOptions,
        name: str = "llama",
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        api_key: str | None = None,
    ) -> None:
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        self._root = root
        self._options = options
        self._name = name
        self._clock = clock
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.AsyncClient(
            transport=transport,
            headers=headers,
            timeout=httpx.Timeout(SAVE_TIMEOUT_SECONDS, connect=5.0),
        )
        self._slots: list[_Slot] = []
        self._capacity = 0
        self._enabled: bool | None = None
        self._props_at = -1e18
        self._model_identity: str | None = None
        self._waiters: list[_Waiter] = []
        self._seq = itertools.count()
        self._lock = asyncio.Lock()
        self._saves: OrderedDict[str, _Saved] = OrderedDict()
        self._free_names: list[str] = []
        self._next_name = 0
        self._save_supported: bool | None = None
        self._save_checked_at = -1e18
        self.stats: dict[str, float] = {
            "granted": 0,
            "queued": 0,
            "saves": 0,
            "restores": 0,
            "restore_skipped": 0,
            "head_restores": 0,
        }

    @property
    def options(self) -> SlotOptions:
        return self._options

    async def aclose(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------ props
    async def _refresh(self) -> None:
        now = self._clock()
        ttl = PROPS_TTL_SECONDS if self._enabled else PROPS_RETRY_SECONDS
        if now - self._props_at < ttl:
            return
        self._props_at = now
        try:
            response = await self._http.get(f"{self._root}/props", timeout=5.0)
            response.raise_for_status()
            props = response.json()
            total = int(props.get("total_slots") or 0)
        except Exception as error:  # not llama-server, or unreachable
            if self._enabled is None:
                logger.info(
                    "LLAMA_SLOTS: {} has no usable /props ({}); not scheduling",
                    self._name,
                    type(error).__name__,
                )
            self._enabled = False if self._enabled is None else self._enabled
            return
        if total <= 0:
            self._enabled = False
            return
        identity = str(props.get("model_path") or props.get("model_alias") or "")
        if self._model_identity is not None and identity != self._model_identity:
            logger.info("LLAMA_SLOTS: {} model changed; dropping saved KV", self._name)
            for saved in self._saves.values():
                self._free_names.append(saved.filename)
            self._saves.clear()
            for slot in self._slots:
                slot.resident, slot.chain, slot.tokens = None, (), 0
        self._model_identity = identity
        if total != self._capacity:
            logger.info("LLAMA_SLOTS: {} serves {} slots", self._name, total)
        while len(self._slots) < total:
            self._slots.append(_Slot(id=len(self._slots)))
        self._capacity = total
        self._enabled = True
        self._grant_waiters()

    # -------------------------------------------------------------- admission
    def _free_slots(self) -> list[_Slot]:
        return [slot for slot in self._slots[: self._capacity] if not slot.busy]

    def _pick_slot(self, profile: RequestProfile, free: list[_Slot]) -> _Slot:
        for slot in free:
            if profile.conversation and slot.resident == profile.conversation:
                return slot
        # A free slot already holding the same system+tools head reuses it in place.
        if profile.chain:
            same_head = [slot for slot in free if slot.chain and slot.chain[0] == profile.chain[0]]
            if same_head:
                return max(same_head, key=lambda slot: slot.last_used)
        empty = [slot for slot in free if slot.resident is None]
        if empty:
            return empty[0]
        return min(free, key=lambda slot: slot.last_used)

    def _head_donor(self, slot: _Slot, profile: RequestProfile) -> str | None:
        """A saved conversation whose system+tools head matches a conversation with nothing saved of its own.

        A new Claude Code session starts with ~20K tokens of instructions and tools that the previous session
        in the same place also started with. Restoring that session's file lets llama-server reuse the shared
        token prefix (up to its last context checkpoint before the first difference) instead of re-reading it.
        None when the slot already holds the same head (llama-server reuses it in place) or nothing matches."""
        if not profile.chain:
            return None
        head = profile.chain[0]
        if slot.chain and slot.chain[0] == head:
            return None
        for key in reversed(self._saves):
            saved = self._saves[key]
            if saved.chain and saved.chain[0] == head:
                return key
        return None

    def _effective(self, waiter: _Waiter, now: float) -> float:
        return waiter.klass - (now - waiter.enqueued) / AGE_SECONDS

    def _grant_waiters(self) -> None:
        now = self._clock()
        while self._waiters:
            free = self._free_slots()
            if not free:
                return
            live = [w for w in self._waiters if not w.future.done()]
            if not live:
                self._waiters.clear()
                return
            best = min(live, key=lambda w: (self._effective(w, now), w.seq))
            # Among equally urgent waiters, prefer one whose KV is already resident.
            for waiter in sorted(live, key=lambda w: w.seq):
                if (
                    self._effective(waiter, now) <= self._effective(best, now) + 0.5
                    and waiter.profile.conversation
                    and any(s.resident == waiter.profile.conversation for s in free)
                ):
                    best = waiter
                    break
            self._waiters.remove(best)
            heapq.heapify(self._waiters)
            slot = self._pick_slot(best.profile, free)
            slot.busy = True
            best.future.set_result(slot)

    async def acquire(self, profile: RequestProfile) -> SlotLease:
        """Wait for a slot (by priority) and prepare its KV cache."""
        async with self._lock:
            await self._refresh()
        if not self._enabled:
            return SlotLease(slot=None, extras=profile.extras)
        started = self._clock()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[_Slot] = loop.create_future()
        waiter = _Waiter(int(profile.klass), next(self._seq), started, profile, future)
        heapq.heappush(self._waiters, waiter)
        self._grant_waiters()
        if not future.done():
            self.stats["queued"] += 1
            logger.info(
                "LLAMA_SLOTS: {} queued class={} waiting={} conv={}",
                self._name,
                profile.klass.name,
                len(self._waiters),
                (profile.conversation or "-")[:24],
            )
        try:
            slot = await future
        except BaseException:
            if future.done() and not future.cancelled():
                self._release_slot(future.result())
            elif waiter in self._waiters:
                self._waiters.remove(waiter)
                heapq.heapify(self._waiters)
            raise
        self.stats["granted"] += 1
        lease = SlotLease(
            slot=slot.id,
            extras={**profile.extras, "id_slot": slot.id},
            waited_s=self._clock() - started,
        )
        try:
            lease.restored = await self._prepare(slot, profile)
        except asyncio.CancelledError:
            self._release_slot(slot)
            raise
        except Exception as error:  # KV moves are an optimisation only
            logger.warning(
                "LLAMA_SLOTS: {} slot {} prepare failed: {}",
                self._name,
                slot.id,
                type(error).__name__,
            )
        slot.resident = profile.conversation
        slot.chain = profile.chain
        slot.tokens = profile.input_tokens
        slot.last_used = self._clock()
        return lease

    def _release_slot(self, slot: _Slot) -> None:
        slot.busy = False
        slot.last_used = self._clock()
        self._grant_waiters()

    def release(self, lease: SlotLease) -> None:
        if lease.slot is None or lease.slot >= len(self._slots):
            return
        self._release_slot(self._slots[lease.slot])

    # ----------------------------------------------------------- KV movement
    async def _save_available(self) -> bool:
        if self._options.save_budget_mb <= 0:
            return False
        if self._save_supported is False:
            if self._clock() - self._save_checked_at < SAVE_RECHECK_SECONDS:
                return False
            self._save_supported = None
        return True

    def _allocate_name(self) -> str:
        if self._free_names:
            return self._free_names.pop()
        name = f"fcc-{self._name}-{self._next_name:04d}.bin"
        self._next_name += 1
        return name

    def _evict_to_budget(self, keep: str) -> None:
        budget = self._options.save_budget_mb * 1024 * 1024
        total = sum(saved.size for saved in self._saves.values())
        while total > budget and len(self._saves) > 1:
            key, saved = next(iter(self._saves.items()))
            if key == keep:
                self._saves.move_to_end(key)
                continue
            del self._saves[key]
            self._free_names.append(saved.filename)
            total -= saved.size

    async def _slot_action(
        self, slot: int, action: str, filename: str
    ) -> dict[str, Any] | None:
        response = await self._http.post(
            f"{self._root}/slots/{slot}",
            params={"action": action},
            json={"filename": filename},
        )
        if response.status_code in (400, 404, 501) and (
            "slot-save-path" in response.text
            or "not supported" in response.text
            or response.status_code == 501
        ):
            self._save_supported = False
            self._save_checked_at = self._clock()
            logger.info(
                "LLAMA_SLOTS: {} has no --slot-save-path; KV stays in RAM only",
                self._name,
            )
            return None
        response.raise_for_status()
        self._save_supported = True
        return response.json()

    async def _prepare(self, slot: _Slot, profile: RequestProfile) -> bool:
        if not profile.conversation or slot.resident == profile.conversation:
            return False
        if not await self._save_available():
            return False
        outgoing = slot.resident
        if (
            outgoing
            and slot.tokens >= MIN_SAVE_TOKENS
            and not (
                outgoing in self._saves and self._saves[outgoing].chain == slot.chain
            )
        ):
            existing = self._saves.pop(outgoing, None)
            filename = existing.filename if existing else self._allocate_name()
            started = time.perf_counter()
            result = await self._slot_action(slot.id, "save", filename)
            if result is None:
                if existing is None:
                    self._free_names.append(filename)
                return False
            self.stats["saves"] += 1
            self._saves[outgoing] = _Saved(
                filename=filename,
                chain=slot.chain,
                size=int(result.get("n_written") or 0),
                tokens=int(result.get("n_saved") or slot.tokens),
            )
            self._evict_to_budget(keep=outgoing)
            logger.info(
                "LLAMA_SLOTS: {} saved slot {} ({} tokens, {} MB) in {:.1f}s",
                self._name,
                slot.id,
                result.get("n_saved"),
                int(result.get("n_written") or 0) // (1024 * 1024),
                time.perf_counter() - started,
            )
        saved = self._saves.get(profile.conversation)
        if saved is None:
            donor = self._head_donor(slot, profile)
            if donor is None:
                return False
            started = time.perf_counter()
            result = await self._slot_action(slot.id, "restore", self._saves[donor].filename)
            if result is None:
                return False
            self.stats["head_restores"] += 1
            logger.info(
                "LLAMA_SLOTS: {} seeded slot {} with {} tokens sharing the head of {} in {:.1f}s",
                self._name,
                slot.id,
                result.get("n_restored"),
                donor[:24],
                time.perf_counter() - started,
            )
            return True
        if not restore_worthwhile(saved.chain, profile.chain):
            self.stats["restore_skipped"] += 1
            return False
        started = time.perf_counter()
        result = await self._slot_action(slot.id, "restore", saved.filename)
        if result is None:
            return False
        self._saves.move_to_end(profile.conversation)
        self.stats["restores"] += 1
        logger.info(
            "LLAMA_SLOTS: {} restored {} tokens into slot {} in {:.1f}s",
            self._name,
            result.get("n_restored"),
            slot.id,
            time.perf_counter() - started,
        )
        return True

    # -------------------------------------------------------------- streaming
    async def stream(
        self,
        profile: RequestProfile,
        open_stream: Callable[[], AsyncIterator[str]],
        *,
        keepalive: str,
        interval: float = KEEPALIVE_SECONDS,
    ) -> AsyncGenerator[str]:
        """Queue for a slot, then relay the provider stream with keep-alives.

        Keep-alive frames go out while queued and whenever the server is silent
        (long prefill), so the client's idle watchdog never fires.
        """
        loop = asyncio.get_running_loop()
        acquiring = loop.create_task(self.acquire(profile))
        lease: SlotLease | None = None
        inner: AsyncIterator[str] | None = None
        pending: asyncio.Task[str] | None = None
        context = contextvars.copy_context()
        try:
            while True:
                done, _ = await asyncio.wait({acquiring}, timeout=interval)
                if done:
                    lease = acquiring.result()
                    break
                yield keepalive
            context.run(_BODY_EXTRAS.set, lease.extras)
            inner = context.run(open_stream)
            while True:
                pending = loop.create_task(_anext(inner), context=context)
                while True:
                    done, _ = await asyncio.wait({pending}, timeout=interval)
                    if done:
                        break
                    yield keepalive
                task, pending = pending, None
                try:
                    frame = task.result()
                except StopAsyncIteration:
                    break
                yield frame
        finally:
            if not acquiring.done():
                acquiring.cancel()
                with contextlib.suppress(BaseException):
                    await acquiring
            if (
                lease is None
                and acquiring.done()
                and not acquiring.cancelled()
                and acquiring.exception() is None
            ):
                lease = acquiring.result()
            if pending is not None and not pending.done():
                pending.cancel()
                with contextlib.suppress(BaseException):
                    await pending
            close = getattr(inner, "aclose", None)
            if callable(close):
                with contextlib.suppress(BaseException):
                    await loop.create_task(close(), context=context)
            if lease is not None:
                self.release(lease)


async def _anext(iterator: AsyncIterator[str]) -> str:
    return await iterator.__anext__()
