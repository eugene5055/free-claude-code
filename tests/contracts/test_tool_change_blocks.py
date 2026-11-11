"""Claude Code sends mid-conversation system messages whose content carries tool_addition / tool_removal blocks.

They used to fail request validation (HTTP 422). They are folded into the request's tool list instead.
"""

from free_claude_code.core.anthropic.models import MessagesRequest, TokenCountRequest


def _tool(name: str) -> dict:
    return {
        "name": name,
        "description": f"{name} tool",
        "input_schema": {"type": "object", "properties": {}},
    }


def _addition(name: str) -> dict:
    return {
        "type": "tool_addition",
        "tool": {"type": "tool_definition", "definition": _tool(name)},
    }


def _removal(name: str) -> dict:
    return {"type": "tool_removal", "tool": {"type": "tool_reference", "name": name}}


def _body(content: list[dict]) -> dict:
    return {
        "model": "claude-test",
        "max_tokens": 64,
        "tools": [_tool("read"), _tool("old")],
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "system", "content": content},
            {"role": "user", "content": "go on"},
        ],
    }


def _names(req) -> list[str]:
    return [t.name for t in (req.tools or [])]


def test_tool_addition_is_folded_into_tools():
    req = MessagesRequest.model_validate(_body([_addition("get_weather")]))
    assert _names(req) == ["read", "old", "get_weather"]


def test_tool_addition_of_a_known_tool_is_not_duplicated():
    req = MessagesRequest.model_validate(_body([_addition("read")]))
    assert _names(req) == ["read", "old"]


def test_tool_removal_drops_the_tool():
    req = MessagesRequest.model_validate(_body([_removal("old")]))
    assert _names(req) == ["read"]


def test_message_left_empty_is_removed_and_text_is_kept():
    req = MessagesRequest.model_validate(_body([_addition("get_weather")]))
    assert [m.role for m in req.messages] == ["user", "user"]
    req = MessagesRequest.model_validate(
        _body([{"type": "text", "text": "note"}, _removal("old")])
    )
    assert len(req.messages) == 3


def test_unknown_block_types_do_not_fail_validation():
    req = MessagesRequest.model_validate(
        _body([{"type": "some_future_block", "x": 1}, {"type": "text", "text": "note"}])
    )
    assert len(req.messages) == 3


def test_token_count_request_accepts_the_same_blocks():
    body = _body([_addition("get_weather"), _removal("old")])
    body.pop("max_tokens")
    req = TokenCountRequest.model_validate(body)
    assert _names(req) == ["read", "get_weather"]
