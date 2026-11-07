"""Claude's safety classifier on local models judges what the user actually said: its transcript is
sent as written, and one too long for the window is refused so a roomier sibling takes it."""

import json

import httpx2
import pytest

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


@pytest.mark.asyncio
async def test_local_classifier_is_sent_as_written():
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
    try:
        stream = instance.stream_messages(_classifier(3), input_tokens=2000)
        assert [event async for event in stream]
        assert len(sent) == 1
        assert "probe the 4090 host" in sent[0]
        assert "being continued" in sent[0]
        assert "ssh host ps aux" in sent[0]
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
