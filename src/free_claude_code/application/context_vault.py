"""Lossless context for models with a small served window (the FCC context vault).

User, 2026-11-06: "we need to make sure all compacted context is always preserved or else it is all for nothing".

Every request to a custom provider that declares context_window_tokens is archived per conversation (SQLite FTS5,
verbatim, 30 days). When a prompt fits no sibling's window, FCC fits it itself instead of refusing: the oldest (then
the largest) blocks become placeholders that say what was there, the archived passages that match the task are put
back verbatim, and the model gets a hidden fcc_recall tool that returns exact archived text. FCC runs those recall
calls itself and keeps streaming one answer to the client. When Claude Code compacted or cleared old tool results
itself, the same recall covers what it dropped, so nothing a conversation once held is lost.
"""

import contextvars
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from loguru import logger

from free_claude_code.core.anthropic import MessagesRequest
from free_claude_code.core.anthropic.models import Message, Tool
from free_claude_code.core.anthropic.tokens import get_token_count
from free_claude_code.core.token_estimation import estimate_text_tokens

RECALL_TOOL = "fcc_recall"
# Set by the executor when no candidate's window can hold the prompt: the roomiest candidate then fits it.
FIT = contextvars.ContextVar("fcc_context_vault_fit", default=False)

CHUNK_CHARS = 1500  # archived passage size
STUB_MIN_TOKENS = 200  # smaller blocks are never replaced by a placeholder
RECALL_SHARE = 6  # at most window/RECALL_SHARE tokens of recalled passages per request or recall call
MAX_ROUNDS = (
    4  # recall calls FCC answers within one client request before the model must answer
)
RETENTION_S = 30 * 86400
ESTIMATE_FACTOR_PCT = 115  # as providers/custom.py: cl100k undercounts local tokenizers
_SESSION_HEADERS = (
    "x-claude-code-session-id",
    "anthropic-session-id",
    "x-anthropic-session-id",
    "claude-session-id",
    "x-claude-session-id",
)
# Claude Code's own compaction leaves these behind; then archived content may be missing from the request.
_COMPACTION_MARKERS = (
    "This session is being continued from a previous conversation",
    "[Old tool result content cleared]",
)
_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)
_TERM = re.compile(r"[A-Za-z0-9_]{3,}")
_STOP = frozenset(
    [
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "have",
        "are",
        "was",
        "were",
        "you",
        "your",
        "not",
        "but",
        "all",
        "any",
        "can",
        "will",
        "just",
        "into",
        "than",
        "then",
        "them",
        "they",
        "what",
        "when",
        "which",
        "who",
        "how",
        "its",
        "also",
        "only",
        "each",
        "per",
        "one",
        "our",
        "out",
        "own",
        "via",
        "use",
        "used",
        "using",
        "get",
        "got",
        "has",
        "had",
        "did",
        "does",
        "should",
        "would",
        "could",
        "must",
        "may",
        "might",
        "there",
        "their",
        "here",
        "these",
        "those",
        "been",
        "being",
        "over",
        "under",
        "about",
        "after",
        "before",
    ]
)
_RECALL_DESCRIPTION = (
    "Search the verbatim text of this conversation that no longer fits in your context window: files and tool "
    "output it read, earlier messages, everything a placeholder marked [FCC vault ...] or a summary only mentions. "
    "Returns the exact archived passages that best match. Use specific words, names or numbers you need."
)


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()


def _get(block: Any, key: str, default: Any = None) -> Any:
    if isinstance(block, Mapping):
        return block.get(key, default)
    return getattr(block, key, default)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        content = [content]
    if isinstance(content, list):
        return "\n".join(
            str(_get(item, "text", ""))
            for item in content
            if _get(item, "type") == "text"
        )
    return ""


def _blocks(message: Message) -> list[Any]:
    content = message.content
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content or [])


def _block_text(block: Any) -> str:
    kind = _get(block, "type")
    if kind == "text":
        return str(_get(block, "text", ""))
    if kind == "tool_result":
        return _content_text(_get(block, "content", ""))
    if kind == "tool_use":
        return json.dumps(_get(block, "input", {}), ensure_ascii=False)
    if kind == "thinking":
        return str(_get(block, "thinking", ""))
    return ""


def _tool_labels(messages: list[Message]) -> dict[str, str]:
    labels = {}
    for message in messages:
        if message.role != "assistant" or isinstance(message.content, str):
            continue
        for block in message.content or []:
            if _get(block, "type") != "tool_use":
                continue
            arguments = _get(block, "input", {}) or {}
            target = next(
                (
                    str(arguments[key])
                    for key in (
                        "file_path",
                        "path",
                        "notebook_path",
                        "url",
                        "pattern",
                        "command",
                        "query",
                    )
                    if isinstance(arguments, Mapping) and arguments.get(key)
                ),
                "",
            )
            labels[str(_get(block, "id", ""))] = (
                str(_get(block, "name", "tool")) + " " + target[:160]
            ).strip()
    return labels


def _label(message: Message, block: Any, tools: Mapping[str, str]) -> str:
    kind = _get(block, "type")
    if kind == "tool_result":
        return "output of " + tools.get(
            str(_get(block, "tool_use_id", "")), "a tool call"
        )
    if kind == "tool_use":
        return "call " + str(_get(block, "name", "tool"))
    return message.role + " message"


def _chunks(text: str, label: str) -> list[tuple[str, str, str]]:
    pieces, start = [], 0
    while start < len(text):
        end = min(len(text), start + CHUNK_CHARS)
        if end < len(text):
            cut = text.rfind("\n", start + CHUNK_CHARS // 2, end)
            if cut > start:
                end = cut + 1
        body = text[start:end]
        if body.strip():
            pieces.append(
                (_sha(body), f"{label} (chars {start}-{end} of {len(text)})", body)
            )
        start = end
    return pieces


def request_pieces(request: MessagesRequest) -> list[tuple[str, str, str]]:
    """Every archivable passage of a request as (hash, label, text)."""
    tools = _tool_labels(request.messages)
    pieces = []
    for message in request.messages:
        for block in _blocks(message):
            text = _block_text(block)
            if text.strip() and not text.startswith("[FCC vault"):
                pieces.extend(_chunks(text, _label(message, block, tools)))
    return pieces


def query_terms(text: str, limit: int = 60) -> list[str]:
    seen: dict[str, None] = {}
    for term in _TERM.findall(_REMINDER.sub(" ", text).lower()):
        if term not in _STOP and term not in seen:
            seen[term] = None
            if len(seen) >= limit:
                break
    return list(seen)


def session_key(request: MessagesRequest, headers: Mapping[str, str] | None) -> str:
    lowered = {
        str(k).lower(): v for k, v in (headers or {}).items() if isinstance(v, str)
    }
    for name in _SESSION_HEADERS:
        if lowered.get(name):
            return "s:" + lowered[name]
    first = next((m for m in request.messages if m.role == "user"), None)
    return "h:" + _sha(_content_text(first.content) if first else "")[:20]


class Vault:
    """Verbatim passages per conversation, searchable with SQLite FTS5 (BM25)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._db: sqlite3.Connection | None = None

    def _conn(self) -> sqlite3.Connection:
        if self._db is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(str(self.path), check_same_thread=False)
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS chunks(
                    id INTEGER PRIMARY KEY, session TEXT NOT NULL, hash TEXT NOT NULL, label TEXT NOT NULL,
                    body TEXT NOT NULL, at REAL NOT NULL, UNIQUE(session, hash));
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(body, content='chunks', content_rowid='id');
                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, body) VALUES (new.id, new.body); END;
                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, body) VALUES ('delete', old.id, old.body); END;
                """
            )
            db.execute("DELETE FROM chunks WHERE at < ?", (time.time() - RETENTION_S,))
            db.commit()
            self._db = db
        return self._db

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None

    def archive(self, session: str, request: MessagesRequest) -> set[str]:
        """Store the request's passages; returns the hashes it holds."""
        pieces = request_pieces(request)
        if pieces:
            now = time.time()
            with self._lock:
                db = self._conn()
                db.executemany(
                    "INSERT OR IGNORE INTO chunks(session, hash, label, body, at) VALUES (?, ?, ?, ?, ?)",
                    [
                        (session, digest, label, body, now)
                        for digest, label, body in pieces
                    ],
                )
                db.commit()
        return {digest for digest, _, _ in pieces}

    def missing(self, session: str, present: set[str]) -> int:
        with self._lock:
            rows = (
                self._conn()
                .execute("SELECT hash FROM chunks WHERE session = ?", (session,))
                .fetchall()
            )
        return sum(1 for (digest,) in rows if digest not in present)

    def search(
        self, session: str, query: str, *, exclude: set[str], max_tokens: int
    ) -> list[tuple[str, str]]:
        terms = query_terms(query)
        if not terms or max_tokens <= 0:
            return []
        match = " OR ".join('"' + term + '"' for term in terms)
        with self._lock:
            rows = (
                self._conn()
                .execute(
                    "SELECT c.hash, c.label, c.body FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
                    "WHERE chunks_fts MATCH ? AND c.session = ? ORDER BY bm25(chunks_fts) LIMIT 80",
                    (match, session),
                )
                .fetchall()
            )
        found, used = [], 0
        for digest, label, body in rows:
            if digest in exclude:
                continue
            cost = estimate_text_tokens(body) + 20
            if used + cost > max_tokens:
                continue
            found.append((label, body))
            used += cost
        return found


_default: Vault | None = None


def default_vault() -> Vault:
    global _default
    path = Path(
        os.environ.get("FCC_CONTEXT_VAULT")
        or Path.home() / ".fcc" / "context_vault.sqlite3"
    )
    if _default is None or _default.path != path:
        if _default is not None:
            _default.close()
        _default = Vault(path)
    return _default


def needs_vault(
    request: MessagesRequest, vault: Vault, session: str, present: set[str]
) -> bool:
    """Whether Claude Code dropped archived content itself (its compaction markers plus missing passages)."""
    text = "\n".join(_block_text(b) for m in request.messages for b in _blocks(m))
    if not any(marker in text for marker in _COMPACTION_MARKERS):
        return False
    return vault.missing(session, present) > 0


def _estimate(request: MessagesRequest) -> int:
    tokens = get_token_count(request.messages, request.system, request.tools)
    return -(-tokens * ESTIMATE_FACTOR_PCT // 100)


def _stub(ref: str, label: str, text: str) -> str:
    head, tail = text[:400], text[-200:] if len(text) > 700 else ""
    return (
        f"[FCC vault {ref}: {label}, {len(text)} chars, kept verbatim outside your context window because the "
        f"conversation outgrew it. It starts:\n{head}\n"
        + (f"...\nand ends:\n{tail}\n" if tail else "")
        + f"Call {RECALL_TOOL} with words or numbers to look for to get its exact text back.]"
    )


def _task_text(messages: list[Message]) -> str:
    """The words that say what is being worked on: the latest user text, then the first user message."""
    users = [m for m in messages if m.role == "user"]
    latest = next(
        (
            t
            for m in reversed(users)
            for t in [_content_text(m.content)]
            if _REMINDER.sub("", t).strip()
        ),
        "",
    )
    first = _content_text(users[0].content) if users else ""
    return latest + "\n" + first


def fit(
    request: MessagesRequest, limit: int, vault: Vault, session: str
) -> MessagesRequest:
    """The request within `limit` estimated tokens, nothing lost: placeholders for what does not fit, the archived
    passages that match the task, and the recall tool. The request must already be archived."""
    tools = [t for t in (request.tools or []) if t.name != RECALL_TOOL] + [
        recall_tool()
    ]
    request = request.model_copy(update={"tools": tools})
    messages = [m.model_dump(exclude_none=True) for m in request.messages]
    for message in messages:
        if isinstance(message.get("content"), str):
            message["content"] = [{"type": "text", "text": message["content"]}]
    labels = _tool_labels(request.messages)
    first_user = next((i for i, m in enumerate(messages) if m["role"] == "user"), -1)
    last = len(messages) - 1
    recall_room = max(1024, limit // RECALL_SHARE)
    target = limit - recall_room
    needed = _estimate(request)
    candidates = []
    for i, message in enumerate(messages):
        for j, block in enumerate(message["content"]):
            text = _block_text(block)
            tokens = estimate_text_tokens(text) if text else 0
            if tokens < STUB_MIN_TOKENS or text.startswith("[FCC vault"):
                continue
            recalled = block.get("type") == "tool_result" and labels.get(
                str(block.get("tool_use_id", "")), ""
            ).startswith(RECALL_TOOL)
            protected = recalled or (i == first_user and block.get("type") == "text")
            phase = 2 if protected else 1 if i == last else 0
            candidates.append(
                ((phase, i if phase == 0 else -tokens, j), i, j, tokens, text)
            )
    candidates.sort(key=lambda item: item[0])
    replaced = 0
    for _, i, j, tokens, text in candidates:
        if needed <= target:
            break
        block = messages[i]["content"][j]
        owner = SimpleNamespace(role=messages[i]["role"])
        stub = _stub(_sha(text)[:10], _label(owner, block, labels), text)
        kind = block.get("type")
        if kind == "tool_result":
            block["content"] = stub
        elif kind == "tool_use":
            block["input"] = {"fcc_vault": stub}
        elif kind == "thinking":
            block["thinking"] = stub
        else:
            block["text"] = stub
        needed -= (tokens - estimate_text_tokens(stub)) * ESTIMATE_FACTOR_PCT // 100
        replaced += 1
    fitted = request.model_copy(
        update={"messages": [Message.model_validate(m) for m in messages]}
    )
    return _with_recalled(fitted, limit, vault, session, replaced)


# (session, anchor) -> passages chosen for it. The note keeps the same passages until the user types again.
_RECALLED: dict[tuple[str, str], list[tuple[str, str]]] = {}
_RECALLED_MAX = 64


def _typed(message: Message) -> bool:
    """Whether a user message holds words the user typed, not only tool results and system reminders."""
    if message.role != "user":
        return False
    return any(
        _get(block, "type") == "text"
        and _REMINDER.sub("", str(_get(block, "text", "") or "")).strip()
        for block in _blocks(message)
    )


def _anchor(messages: list[Message]) -> int | None:
    """Where the recall note goes: the user's latest typed message (else the latest user message)."""
    users = [i for i, m in enumerate(messages) if m.role == "user"]
    typed = [i for i in users if _typed(messages[i])]
    return (typed or users or [None])[-1]


def _with_recalled(
    request: MessagesRequest, limit: int, vault: Vault, session: str, replaced: int
) -> MessagesRequest:
    """The request plus a note with the archived passages that match the task.

    User, 2026-11-06: the note went on the newest message with passages chosen afresh every turn, so a local
    server re-read ~22K tokens on every tool call. It now goes on the user's latest typed message and keeps the
    same passages until the user types again, so the prompt prefix (and the server's cache) holds across turns."""
    messages = list(request.messages)
    index = _anchor(messages)
    room = min(limit - _estimate(request), max(1024, limit // RECALL_SHARE)) - 200
    budget = room * 100 // ESTIMATE_FACTOR_PCT
    anchor_text = _content_text(messages[index].content) if index is not None else ""
    key = (session, f"{index}:{_sha(anchor_text)}")
    passages = _RECALLED.get(key)
    if passages is None or sum(
        estimate_text_tokens(body) + 20 for _, body in passages
    ) > max(budget, 0):
        present = {digest for digest, _, _ in request_pieces(request)}
        passages = vault.search(
            session, _task_text(messages), exclude=present, max_tokens=budget
        )
        _RECALLED.pop(key, None)
        _RECALLED[key] = passages
        while len(_RECALLED) > _RECALLED_MAX:
            _RECALLED.pop(next(iter(_RECALLED)))
    note = (
        "[FCC vault: parts of this conversation are kept verbatim outside your context window"
        + (f" ({replaced} block(s) above are placeholders)" if replaced else "")
        + f". Call {RECALL_TOOL} to get the exact text of anything you need."
    )
    if passages:
        note += " Archived passages that match the task:\n\n" + "\n\n".join(
            f"### {label}\n{body}" for label, body in passages
        )
    note += "]"
    if index is None:
        messages.append(Message(role="user", content=note))
    else:
        blocks = [*_blocks(messages[index]), {"type": "text", "text": note}]
        messages[index] = Message.model_validate({"role": "user", "content": blocks})
    logger.info(
        "Context vault: session={} placeholders={} recalled_passages={} anchor={} limit={}",
        session[:24],
        replaced,
        len(passages),
        index,
        limit,
    )
    return request.model_copy(update={"messages": messages})

def recall_tool() -> Tool:
    return Tool(
        name=RECALL_TOOL,
        description=_RECALL_DESCRIPTION,
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Words, names or numbers to look for.",
                }
            },
            "required": ["query"],
        },
    )


def recall_text(vault: Vault, session: str, query: str, max_tokens: int) -> str:
    passages = vault.search(session, query, exclude=set(), max_tokens=max_tokens)
    if not passages:
        return f"No archived passage matches {query!r}. Try other words."
    return "\n\n".join(f"### {label}\n{body}" for label, body in passages)


def _event(name: str, data: Mapping[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _events(
    stream: AsyncIterator[str],
) -> AsyncIterator[tuple[dict[str, Any] | None, str]]:
    buffer = ""
    async for chunk in stream:
        buffer += chunk
        while "\n\n" in buffer:
            raw, buffer = buffer.split("\n\n", 1)
            raw += "\n\n"
            data = None
            for line in raw.splitlines():
                if line.startswith("data:"):
                    try:
                        data = json.loads(line[5:].strip())
                    except ValueError:
                        data = None
            yield (data if isinstance(data, dict) else None), raw
    if buffer.strip():
        yield None, buffer


async def recall_stream(
    request: MessagesRequest,
    *,
    vault: Vault,
    session: str,
    limit: int,
    open_round: Callable[[MessagesRequest, int], AsyncIterator[str]],
) -> AsyncIterator[str]:
    """Stream one answer while FCC itself answers the model's fcc_recall calls.

    Text and thinking stream live. Tool calls are held until the round ends: a round that calls fcc_recall gets
    the archived passages and the model continues in the same client message (its other calls that round are
    dropped; it repeats them once it has what it needed); otherwise the held calls are released as usual.
    """
    conversation = request
    started = False
    out = 0
    for round_number in range(1, MAX_ROUNDS + 1):
        final = round_number == MAX_ROUNDS
        prepared = fit(conversation, limit, vault, session)
        if final:
            prepared = prepared.model_copy(
                update={
                    "tools": [t for t in prepared.tools or [] if t.name != RECALL_TOOL]
                }
            )
        blocks: dict[int, dict[str, Any]] = {}
        ending: list[tuple[str, dict[str, Any]]] = []
        stream = open_round(prepared, round_number)
        try:
            async for data, raw in _events(stream):
                kind = data.get("type") if data else None
                if kind == "message_start":
                    if not started:
                        started = True
                        yield raw
                elif kind == "content_block_start":
                    block = dict(data.get("content_block") or {})
                    entry: dict[str, Any] = {"block": block, "parts": [], "events": []}
                    blocks[int(data.get("index", 0))] = entry
                    if block.get("type") != "tool_use":
                        entry["out"] = out
                        out += 1
                        yield _event(
                            "content_block_start", {**data, "index": entry["out"]}
                        )
                elif kind in ("content_block_delta", "content_block_stop"):
                    entry = blocks.get(int(data.get("index", 0)))
                    if entry is None:
                        continue
                    delta = data.get("delta") or {}
                    part = (
                        delta.get("text")
                        or delta.get("thinking")
                        or delta.get("partial_json")
                    )
                    if kind == "content_block_delta" and isinstance(part, str):
                        entry["parts"].append(part)
                    if "out" in entry:
                        yield _event(kind, {**data, "index": entry["out"]})
                    else:
                        entry["events"].append((kind, data))
                elif kind in ("message_delta", "message_stop"):
                    ending.append((kind, data))
                else:
                    yield raw
        finally:
            closer = getattr(stream, "aclose", None)
            if closer is not None:
                await closer()
        recalls = [
            e
            for e in blocks.values()
            if e["block"].get("type") == "tool_use"
            and e["block"].get("name") == RECALL_TOOL
        ]
        if recalls and not final:
            assistant: list[dict[str, Any]] = []
            results: list[dict[str, Any]] = []
            for entry in blocks.values():
                block = entry["block"]
                if block.get("type") == "text" and entry["parts"]:
                    assistant.append({"type": "text", "text": "".join(entry["parts"])})
            for entry in recalls:
                block = entry["block"]
                try:
                    arguments = (
                        json.loads("".join(entry["parts"]))
                        if entry["parts"]
                        else block.get("input") or {}
                    )
                except ValueError:
                    arguments = {}
                query = (
                    str(arguments.get("query") or "")
                    if isinstance(arguments, dict)
                    else ""
                )
                tool_id = str(block.get("id") or f"toolu_fcc_{uuid.uuid4().hex[:20]}")
                assistant.append(
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": RECALL_TOOL,
                        "input": {"query": query},
                    }
                )
                text = recall_text(
                    vault, session, query, max(1024, limit // RECALL_SHARE)
                )
                results.append(
                    {"type": "tool_result", "tool_use_id": tool_id, "content": text}
                )
                logger.info(
                    "Context vault: recall session={} query={!r} chars={}",
                    session[:24],
                    query[:80],
                    len(text),
                )
            conversation = conversation.model_copy(
                update={
                    "messages": [
                        *conversation.messages,
                        Message.model_validate(
                            {"role": "assistant", "content": assistant}
                        ),
                        Message.model_validate({"role": "user", "content": results}),
                    ]
                }
            )
            continue
        released = 0
        for _, entry in sorted(blocks.items()):
            if "out" in entry or entry["block"].get("name") == RECALL_TOOL:
                continue
            index = out
            out += 1
            released += 1
            yield _event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": entry["block"],
                },
            )
            for kind, data in entry["events"]:
                yield _event(kind, {**data, "index": index})
        for kind, data in ending:
            if (
                kind == "message_delta"
                and not released
                and (data.get("delta") or {}).get("stop_reason") == "tool_use"
            ):
                data = {**data, "delta": {**data["delta"], "stop_reason": "end_turn"}}
            yield _event(kind, data)
        return


CLASSIFIER_KEEP_TAIL = 6  # newest transcript entries a classifier trim never drops (the action being judged)
_CLASSIFIER_NOTE = (
    '{{"fcc_note":"{n} older tool-call entries omitted to fit this model context window; '
    'every user message is kept"}}\n'
)


def fit_classifier(request: MessagesRequest, limit: int) -> MessagesRequest | None:
    """Claude's safety classifier transcript within `limit`, or None if it cannot fit.

    Never placeholders or recall (the classifier must see what the user said): only the oldest non-user
    transcript entries are dropped. Every {"user": ...} entry, text outside the transcript (CLAUDE.md) and the
    newest entries stay verbatim (user, 2026-11-06: a vault-fitted transcript hid the user's approval and the
    local model blocked an approved command).
    """
    messages = [m.model_dump(exclude_none=True) for m in request.messages]
    for message in messages:
        if isinstance(message.get("content"), str):
            message["content"] = [{"type": "text", "text": message["content"]}]
    split: dict[tuple[int, int], list[str]] = {}
    entries: list[tuple[int, int, int]] = []
    inside = False
    for i, message in enumerate(messages):
        for j, block in enumerate(message["content"]):
            if block.get("type") != "text":
                continue
            parts = block.get("text", "").splitlines(keepends=True)
            split[(i, j)] = parts
            for k, line in enumerate(parts):
                if "<transcript>" in line:
                    inside = True
                elif "</transcript>" in line:
                    inside = False
                elif inside and line.strip():
                    entries.append((i, j, k))
    older = entries[:-CLASSIFIER_KEEP_TAIL] if len(entries) > CLASSIFIER_KEEP_TAIL else []
    droppable = [
        p for p in older if not split[p[:2]][p[2]].lstrip().startswith('{"user"')
    ]
    needed = _estimate(request)
    dropped: list[tuple[int, int, int]] = []
    for i, j, k in droppable:
        if needed <= limit:
            break
        needed -= estimate_text_tokens(split[(i, j)][k]) * ESTIMATE_FACTOR_PCT // 100
        split[(i, j)][k] = ""
        dropped.append((i, j, k))
    if dropped:
        i, j, k = dropped[0]
        split[(i, j)][k] = _CLASSIFIER_NOTE.format(n=len(dropped))
    for (i, j), parts in split.items():
        messages[i]["content"][j]["text"] = "".join(parts)
    fitted = request.model_copy(
        update={"messages": [Message.model_validate(m) for m in messages]}
    )
    if _estimate(fitted) > limit:
        return None
    if dropped:
        logger.info(
            "Safety classifier trim: dropped={} kept_user_entries=all limit={}",
            len(dropped),
            limit,
        )
    return fitted
