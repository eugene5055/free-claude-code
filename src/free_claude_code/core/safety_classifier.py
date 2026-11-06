"""Detect known Claude auto-mode safety classifier requests (shared by api and providers)."""

from free_claude_code.core.anthropic import (
    MessagesRequest,
    extract_text_from_content,
)


def _request_system_text(request_data: MessagesRequest) -> str:
    """Return top-level and inline system text for request-shape detection."""
    parts: list[str] = []
    if request_data.system:
        text = extract_text_from_content(request_data.system)
        if text:
            parts.append(text)
    for message in request_data.messages:
        if message.role != "system":
            continue
        text = extract_text_from_content(message.content)
        if text:
            parts.append(text)
    return "\n".join(parts)


def detect_safety_classifier_stop_sequence(
    request_data: MessagesRequest,
) -> str | None:
    """Return the stop hint owned by a known Claude auto-mode classifier."""
    if request_data.tools:
        return None

    messages_text = "\n".join(
        extract_text_from_content(message.content)
        for message in request_data.messages
        if message.role != "system"
    )
    if "<transcript>" not in messages_text:
        return None

    system_text = _request_system_text(request_data)
    if "Output <severity>N</severity>" in system_text:
        return "</severity>"
    if "yes</block>" in system_text or "no</block>" in system_text:
        return "</block>"
    return None


def is_safety_classifier(request_data: MessagesRequest) -> bool:
    """Whether this is Claude Code's auto-mode safety classifier asking for a verdict."""
    return detect_safety_classifier_stop_sequence(request_data) is not None
