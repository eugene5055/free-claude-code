"""Claude's safety classifier on local models judges what the user actually said.

User, 2026-11-06: on duo-local:stable the classifier request shared the session's vault key,
its compaction summary switched the vault on, and placeholders hid the user's approval, so the
local model blocked an approved command. The classifier must never be vault-fitted.
"""

import json

import httpx2
import pytest

from free_claude_code.application import context_vault
from free_claude_code.core.anthropic import MessagesRequest
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.safety_classifier import is_safety_classifier
from tests.providers.test_custom_provider import completion, definition, provider

SYSTEM = "Classify the action. Reply <block>yes</block> or <block>no</block>."
APPROVAL = '{"user":"yes, go ahead and probe the 4090 host over ssh and fix it"}\n'


def _classifier(tool_lines: int, *, compacted: bool = True) -> MessagesRequest:
    summary = (
        '{"user":"This session is being continued from a previous conversation. '
        + "summary " * 50
        + '"}\n'
        if compacted
        else ""
    )
    old = "".join(
        json.dumps({"Bash": f"cat /tmp/file{n}.txt " + "x" * 400}) + "\n"
        for n in range(tool_lines)
    )
    blocks = [
        {"type": "text", "text": "The user's CLAUDE.md: be careful.\n"},
        {"type": "text", "text": "<transcript>\n"},
        {"type": "text", "text": summary + APPROVAL + old},
        {"type": "text", "text": '{"Bash":"ssh host ps aux"}\n'},
        {"type": "text", "text": "</transcript>\n"},
    ]
    return MessagesRequest(
        model="m",
        max_tokens=64,
        system=SYSTEM,
        messages=[{"role": "user", "content": blocks}],
    )


def test_detects_claude_codes_classifier():
    assert is_safety_classifier(_classifier(1))


def test_fit_classifier_keeps_every_user_entry_and_the_judged_action():
    fitted = context_vault.fit_classifier(_classifier(400), limit=8000)
    assert fitted is not None
    text = json.dumps(fitted.model_dump())
    assert "probe the 4090 host" in text
    assert "being continued" in text
    assert "ssh host ps aux" in text
    assert "CLAUDE.md" in text
    assert "fcc_note" in text
    assert "[FCC vault" not in text
    assert not fitted.tools


def test_fit_classifier_gives_up_rather_than_drop_user_words():
    assert context_vault.fit_classifier(_classifier(0), limit=50) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("fitting", [False, True])
async def test_local_classifier_never_gets_placeholders_or_recall(
    tmp_path, monkeypatch, fitting
):
    monkeypatch.setenv("FCC_CONTEXT_VAULT", str(tmp_path / "vault.sqlite3"))
    sent = []

    def reply(request):
        sent.append(request.content.decode())
        return httpx2.Response(
            200,
            text=completion("openai_chat"),
            headers={"content-type": "text/event-stream"},
        )

    instance = provider(
        definition(api_format="openai_chat", context_window_tokens=65536),
        openai_handler=reply,
    )
    # The session's vault already holds content the classifier transcript lacks.
    headers = {"x-claude-code-session-id": "same-session"}
    older = MessagesRequest(
        model="m",
        max_tokens=8,
        messages=[{"role": "user", "content": "an older turn compaction dropped " * 40}],
    )
    vault = context_vault.default_vault()
    vault.archive(context_vault.session_key(older, headers), older)
    token = context_vault.FIT.set(fitting)
    try:
        stream = instance.stream_messages(
            _classifier(3), input_tokens=2000, request_headers=headers
        )
    finally:
        context_vault.FIT.reset(token)
    try:
        assert [event async for event in stream]
        assert len(sent) == 1
        assert context_vault.RECALL_TOOL not in sent[0]
        assert "[FCC vault" not in sent[0]
        assert "probe the 4090 host" in sent[0]
    finally:
        await instance.cleanup()


@pytest.mark.asyncio
async def test_too_long_classifier_moves_to_a_roomier_sibling():
    instance = provider(
        definition(api_format="openai_chat", context_window_tokens=65536)
    )
    try:
        with pytest.raises(ExecutionFailure) as refused:
            instance.stream_messages(_classifier(3), input_tokens=60000)
        assert refused.value.kind is FailureKind.CONTEXT_WINDOW_EXCEEDED
    finally:
        await instance.cleanup()
