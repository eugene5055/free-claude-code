"""Anthropic request features llama.cpp's Chat Completions endpoint drops.

llama-server reads ``tool_choice`` only as a string (a named-function object
silently becomes "auto"), defaults ``parallel_tool_calls`` to the template's
capability, and takes sampling and structured-output fields the generic
conversion does not forward. This postprocessor runs for custom providers whose
reasoning format is ``llamacpp`` (the format declares a llama.cpp server).
"""

from typing import Any

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.reasoning import ReasoningControl, ReasoningPolicy

# Qwen3.8 model-card sampling for non-thinking (instruct) mode: temperature 0.7,
# top_p 0.8, top_k 20, presence_penalty 1.5. Thinking mode uses the server's
# defaults (temperature 1.0, top_p 0.95, top_k 20) from the GGUF's sampling
# metadata, which already match the card.
NON_THINKING_TEMPERATURE = 0.7
NON_THINKING_TOP_P = 0.8
NON_THINKING_TOP_K = 20
NON_THINKING_PRESENCE_PENALTY = 1.5
_STRUCTURED_OUTPUT_NAME = "structured_output"


def apply_llamacpp_request_compat(
    body: dict[str, Any], request: MessagesRequest, reasoning: ReasoningPolicy
) -> None:
    """Translate tool_choice, parallel calls, top_k, structured output, sampling."""
    _named_tool_choice(body)
    _parallel_tool_calls(body, request.tool_choice)
    if request.top_k is not None and request.top_k > 0:
        # Not an OpenAI SDK parameter: top-level unknown keys fail create().
        _extra_body(body)["top_k"] = request.top_k
    _structured_output(body, request.output_config)
    _non_thinking_sampling(body, request, reasoning)


def _extra_body(body: dict[str, Any]) -> dict[str, Any]:
    value = body.setdefault("extra_body", {})
    if not isinstance(value, dict):
        raise TypeError("OpenAI extra_body must be an object.")
    return value


def _named_tool_choice(body: dict[str, Any]) -> None:
    """Force one named tool: llama.cpp honors only "required", so offer just that tool."""
    choice = body.get("tool_choice")
    if not isinstance(choice, dict):
        return
    function = choice.get("function")
    name = function.get("name") if isinstance(function, dict) else None
    tools = body.get("tools")
    if not isinstance(name, str) or not isinstance(tools, list):
        return
    named = [
        tool
        for tool in tools
        if isinstance(tool, dict)
        and isinstance(tool.get("function"), dict)
        and tool["function"].get("name") == name
    ]
    if not named:
        return
    body["tools"] = named
    body["tool_choice"] = "required"


def _parallel_tool_calls(body: dict[str, Any], tool_choice: Any) -> None:
    if not body.get("tools") or body.get("tool_choice") == "none":
        return
    if body.get("tool_choice") == "required" and len(body["tools"]) == 1:
        # A forced single tool (Anthropic "tool" choice) is answered by one call.
        body["parallel_tool_calls"] = False
        return
    disabled = (
        isinstance(tool_choice, dict)
        and tool_choice.get("disable_parallel_tool_use") is True
    )
    body["parallel_tool_calls"] = not disabled


def _structured_output(body: dict[str, Any], output_config: Any) -> None:
    """Map Anthropic ``output_config.format`` to a llama.cpp JSON-schema grammar."""
    if not isinstance(output_config, dict):
        return
    fmt = output_config.get("format")
    if not isinstance(fmt, dict) or fmt.get("type") != "json_schema":
        return
    schema = fmt.get("schema")
    if not isinstance(schema, dict):
        return
    body["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": _STRUCTURED_OUTPUT_NAME,
            "schema": schema,
            "strict": True,
        },
    }


def _non_thinking_sampling(
    body: dict[str, Any], request: MessagesRequest, reasoning: ReasoningPolicy
) -> None:
    if reasoning.control not in {ReasoningControl.OFF, ReasoningControl.PREFER_OFF}:
        return
    if request.temperature is None and request.top_p is None:
        body.setdefault("temperature", NON_THINKING_TEMPERATURE)
        body.setdefault("top_p", NON_THINKING_TOP_P)
        body.setdefault("presence_penalty", NON_THINKING_PRESENCE_PENALTY)
        if request.top_k is None:
            _extra_body(body).setdefault("top_k", NON_THINKING_TOP_K)
