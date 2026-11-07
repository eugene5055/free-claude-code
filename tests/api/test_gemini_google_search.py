"""Antigravity's WebSearch sends Gemini's built-in googleSearch tool."""

import pytest
from pydantic import ValidationError

from free_claude_code.api.handlers.gemini import (
    google_search_query,
    search_results_text,
    without_google_search,
)
from free_claude_code.core.gemini import GeminiRequest
from free_claude_code.core.web_tools import WebSearchResult


def _request(tools: list[dict[str, object]], **extra: object) -> GeminiRequest:
    return GeminiRequest.model_validate(
        {
            "model": "m",
            "contents": [{"role": "user", "parts": [{"text": "current events today"}]}],
            "tools": tools,
            **extra,
        }
    )


def test_google_search_tool_is_accepted() -> None:
    request = _request([{"googleSearch": {}}])
    assert request.requests_google_search
    assert google_search_query(request) == "current events today"


def test_tool_needs_exactly_one_variant() -> None:
    with pytest.raises(ValidationError):
        _request([{}])
    with pytest.raises(ValidationError):
        _request([{"googleSearch": {}, "functionDeclarations": [{"name": "read"}]}])


def test_search_results_replace_the_tool() -> None:
    request = _request([{"googleSearch": {}}])
    note = search_results_text(
        "current events today", [WebSearchResult(title="News", url="https://n.example")]
    )
    grounded = without_google_search(request, note)
    assert grounded.tools == []
    parts = grounded.contents[-1].parts
    assert parts[0].text == "current events today"
    assert parts[1].text is not None and "https://n.example" in parts[1].text


def test_search_beside_function_tools_keeps_only_function_tools() -> None:
    request = _request(
        [{"googleSearch": {}}, {"functionDeclarations": [{"name": "read"}]}]
    )
    stripped = without_google_search(request, None)
    assert [tool.function_declarations[0].name for tool in stripped.tools] == ["read"]
    assert stripped.contents == request.contents


def test_failed_search_says_so() -> None:
    assert "failed" in search_results_text("q", None, "timeout")
    assert "no results" in search_results_text("q", [])
