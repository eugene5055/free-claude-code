"""Nothing a conversation held is lost when it outgrows a local model's window (application/context_vault.py)."""

import json

import pytest

from free_claude_code.application import context_vault as cv
from free_claude_code.core.anthropic.models import MessagesRequest

NEEDLE = "The launch code for project heron is 7741-QUARTZ."


def filler(seed: int, chars: int) -> str:
    lines, n = [], 0
    while sum(map(len, lines)) < chars:
        lines.append(
            f"alpha{seed} row {n} value {n * 37 % 1009} padding text for the archive\n"
        )
        n += 1
    return "".join(lines)


def big_request(
    files: int = 6,
    chars: int = 30000,
    question: str = "What is the launch code for project heron?",
):
    messages = [{"role": "user", "content": "Read every file and answer: " + question}]
    calls, results = [], []
    for k in range(files):
        body = filler(k, chars)
        if k == 0:
            body = body[: chars // 2] + NEEDLE + "\n" + body[chars // 2 :]
        calls.append(
            {
                "type": "tool_use",
                "id": f"toolu_{k}",
                "name": "Read",
                "input": {"file_path": f"/tmp/f{k}"},
            }
        )
        results.append(
            {"type": "tool_result", "tool_use_id": f"toolu_{k}", "content": body}
        )
    messages += [
        {"role": "assistant", "content": calls},
        {"role": "user", "content": results},
    ]
    return MessagesRequest(model="m", max_tokens=4096, messages=messages)


def all_text(request: MessagesRequest) -> str:
    return json.dumps([m.model_dump() for m in request.messages])


@pytest.fixture
def vault(tmp_path):
    store = cv.Vault(tmp_path / "vault.sqlite3")
    yield store
    store.close()


def test_fit_keeps_the_prompt_in_the_window_and_brings_back_the_answer(vault):
    request = big_request()
    vault.archive("s", request)
    fitted = cv.fit(request, 12000, vault, "s")
    assert cv._estimate(fitted) <= 12000
    assert "[FCC vault" in all_text(fitted)
    assert NEEDLE in all_text(fitted)  # recalled verbatim from the archive
    assert any(tool.name == cv.RECALL_TOOL for tool in fitted.tools)
    uses = [b.id for m in fitted.messages if m.role == "assistant" for b in m.content]
    results = [
        b.tool_use_id
        for m in fitted.messages
        if not isinstance(m.content, str)
        for b in m.content
        if getattr(b, "type", None) == "tool_result"
    ]
    assert uses == results == [f"toolu_{k}" for k in range(6)]


def test_the_task_text_is_never_replaced_while_other_blocks_can_be(vault):
    request = big_request()
    vault.archive("s", request)
    fitted = cv.fit(request, 12000, vault, "s")
    assert fitted.messages[0].content[0].text.startswith("Read every file and answer")


def test_every_passage_is_archived_once(vault):
    request = big_request(files=2, chars=5000)
    first = vault.archive("s", request)
    again = vault.archive("s", request)
    assert first == again
    assert vault.missing("s", set()) == len(first)
    assert vault.missing("s", first) == 0


def test_sessions_never_see_each_others_passages(vault):
    vault.archive("a", big_request(files=1, chars=3000))
    assert vault.search("b", "launch code heron", exclude=set(), max_tokens=5000) == []
    assert any(
        NEEDLE in body
        for _, body in vault.search(
            "a", "launch code heron", exclude=set(), max_tokens=5000
        )
    )


def test_vault_mode_after_claude_code_compacted_and_dropped_content(vault):
    vault.archive("s", big_request(files=2, chars=5000))
    compacted = MessagesRequest(
        model="m",
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": "This session is being continued from a previous conversation that "
                "ran out of context. The summary below covers the earlier portion of the conversation. ...",
            }
        ],
    )
    present = vault.archive("s", compacted)
    assert cv.needs_vault(compacted, vault, "s", present)
    plain = MessagesRequest(
        model="m", max_tokens=100, messages=[{"role": "user", "content": "hello"}]
    )
    assert not cv.needs_vault(plain, vault, "s", vault.archive("s", plain))


def test_session_key_prefers_claude_code_session_header():
    request = MessagesRequest(
        model="m", max_tokens=10, messages=[{"role": "user", "content": "x"}]
    )
    assert cv.session_key(request, {"X-Claude-Code-Session-Id": "abc"}) == "s:abc"
    assert cv.session_key(request, {}) == cv.session_key(request, None)
    assert cv.session_key(request, None).startswith("h:")


def sse(name, data):
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def text_block(index, text):
    return [
        sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": text},
            },
        ),
        sse("content_block_stop", {"type": "content_block_stop", "index": index}),
    ]


def tool_block(index, tool_id, name, arguments):
    return [
        sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": name,
                    "input": {},
                },
            },
        ),
        sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(arguments),
                },
            },
        ),
        sse("content_block_stop", {"type": "content_block_stop", "index": index}),
    ]


def message(blocks, stop_reason):
    return [
        sse(
            "message_start",
            {
                "type": "message_start",
                "message": {"id": "msg", "role": "assistant", "content": []},
            },
        ),
        *blocks,
        sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason},
                "usage": {"output_tokens": 5},
            },
        ),
        sse("message_stop", {"type": "message_stop"}),
    ]


async def chunks(items):
    for item in items:
        # Split frames across chunks the way a network would.
        yield item[:7]
        yield item[7:]


def parsed(frames):
    return [
        json.loads(line[5:])
        for frame in "".join(frames).split("\n\n")
        for line in frame.splitlines()
        if line.startswith("data:")
    ]


@pytest.mark.asyncio
async def test_recall_calls_are_answered_by_fcc_and_the_client_sees_one_answer(vault):
    request = big_request()
    vault.archive("s", request)
    rounds = []

    def open_round(prepared, number):
        rounds.append(prepared)
        if number == 1:
            return chunks(
                message(
                    [
                        *text_block(0, "Let me look."),
                        *tool_block(
                            1, "toolu_r", cv.RECALL_TOOL, {"query": "launch code heron"}
                        ),
                    ],
                    "tool_use",
                )
            )
        return chunks(message(text_block(0, "It is 7741-QUARTZ."), "end_turn"))

    out = parsed(
        [
            f
            async for f in cv.recall_stream(
                request, vault=vault, session="s", limit=12000, open_round=open_round
            )
        ]
    )
    assert len(rounds) == 2
    last = rounds[1].messages[-1]
    assert any(
        getattr(b, "type", None) == "tool_result" and NEEDLE in str(b.content)
        for b in last.content
    )
    assert [e["type"] for e in out].count("message_start") == 1
    starts = [e for e in out if e["type"] == "content_block_start"]
    assert [e["index"] for e in starts] == [0, 1]
    assert all(e["content_block"]["type"] == "text" for e in starts)
    assert [e["delta"]["stop_reason"] for e in out if e["type"] == "message_delta"] == [
        "end_turn"
    ]


@pytest.mark.asyncio
async def test_client_tool_calls_pass_through_after_the_text(vault):
    request = big_request(files=1, chars=2000)
    vault.archive("s", request)

    def open_round(prepared, number):
        return chunks(
            message(
                [
                    *text_block(0, "Reading."),
                    *tool_block(1, "toolu_c", "Read", {"file_path": "/tmp/x"}),
                ],
                "tool_use",
            )
        )

    out = parsed(
        [
            f
            async for f in cv.recall_stream(
                request, vault=vault, session="s", limit=12000, open_round=open_round
            )
        ]
    )
    tool = [
        e
        for e in out
        if e["type"] == "content_block_start"
        and e["content_block"]["type"] == "tool_use"
    ]
    assert [(e["index"], e["content_block"]["name"]) for e in tool] == [(1, "Read")]
    deltas = [e for e in out if e["type"] == "content_block_delta" and e["index"] == 1]
    assert json.loads(deltas[0]["delta"]["partial_json"]) == {"file_path": "/tmp/x"}
    assert [e["delta"]["stop_reason"] for e in out if e["type"] == "message_delta"] == [
        "tool_use"
    ]


@pytest.mark.asyncio
async def test_the_last_round_has_no_recall_tool_so_the_model_must_answer(vault):
    request = big_request(files=1, chars=2000)
    vault.archive("s", request)
    tools = []

    def open_round(prepared, number):
        tools.append([t.name for t in prepared.tools or []])
        return chunks(
            message(
                tool_block(0, f"toolu_{number}", cv.RECALL_TOOL, {"query": "heron"}),
                "tool_use",
            )
        )

    out = parsed(
        [
            f
            async for f in cv.recall_stream(
                request, vault=vault, session="s", limit=12000, open_round=open_round
            )
        ]
    )
    assert len(tools) == cv.MAX_ROUNDS
    assert all(cv.RECALL_TOOL in names for names in tools[:-1])
    assert cv.RECALL_TOOL not in tools[-1]
    assert [e["delta"]["stop_reason"] for e in out if e["type"] == "message_delta"] == [
        "end_turn"
    ]


def _tool_turn(k: int):
    return [
        {"role": "assistant", "content": [{"type": "tool_use", "id": f"toolu_t{k}", "name": "Bash", "input": {"command": f"ls {k}"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"toolu_t{k}", "content": f"file{k}.txt"},
            {"type": "text", "text": "<system-reminder>\nKeep going.\n</system-reminder>"},
        ]},
    ]


def _note_index(request: MessagesRequest) -> list[int]:
    return [
        i for i, m in enumerate(request.messages)
        if "[FCC vault: parts of this conversation" in json.dumps(m.model_dump())
    ]


def _note(request: MessagesRequest, index: int) -> str:
    return next(
        b.text for b in request.messages[index].content
        if getattr(b, "type", None) == "text" and b.text.startswith("[FCC vault")
    )


def test_the_recall_note_rides_the_users_typed_message_and_stays_the_same_across_tool_turns(vault):
    vault.archive("anchor", big_request(files=2, chars=5000))
    typed = {"role": "user", "content": "Read every file alpha0 alpha1 and find the heron launch code"}
    first = MessagesRequest(model="m", max_tokens=100, messages=[typed, *_tool_turn(1)])
    vault.archive("anchor", first)
    a = cv._with_recalled(first, 230000, vault, "anchor", 0)
    later = MessagesRequest(model="m", max_tokens=100, messages=[typed, *_tool_turn(1), *_tool_turn(2)])
    vault.archive("anchor", later)  # the corpus grows every turn; the note must not change with it
    b = cv._with_recalled(later, 230000, vault, "anchor", 0)
    assert _note_index(a) == _note_index(b) == [0]  # not the newest tool-result message
    assert NEEDLE in _note(a, 0)
    assert _note(a, 0) == _note(b, 0)  # byte-identical: the server's prompt cache keeps it
    assert a.messages[0].model_dump() == b.messages[0].model_dump()


def test_a_new_typed_message_moves_the_recall_note_and_chooses_passages_again(vault):
    vault.archive("moves", big_request(files=2, chars=5000))
    typed = {"role": "user", "content": "Read every file alpha0 and summarise"}
    first = MessagesRequest(model="m", max_tokens=100, messages=[typed, *_tool_turn(1)])
    vault.archive("moves", first)
    cv._with_recalled(first, 230000, vault, "moves", 0)
    follow = {"role": "user", "content": "Now what is the heron launch code?"}
    second = MessagesRequest(
        model="m", max_tokens=100,
        messages=[typed, *_tool_turn(1), {"role": "assistant", "content": "Done."}, follow],
    )
    vault.archive("moves", second)
    out = cv._with_recalled(second, 230000, vault, "moves", 0)
    assert _note_index(out) == [len(second.messages) - 1]
    assert NEEDLE in _note(out, len(second.messages) - 1)


def test_a_cached_note_that_no_longer_fits_is_chosen_again(vault):
    vault.archive("shrink", big_request(files=4, chars=5000))
    typed = {"role": "user", "content": "Read every file alpha0 alpha1 alpha2 alpha3"}
    request = MessagesRequest(model="m", max_tokens=100, messages=[typed, *_tool_turn(1)])
    vault.archive("shrink", request)
    roomy = cv._with_recalled(request, 230000, vault, "shrink", 0)
    tight = cv._with_recalled(request, 6000, vault, "shrink", 0)
    assert len(_note(tight, 0)) < len(_note(roomy, 0))
