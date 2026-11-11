"""Pydantic models for the Anthropic Messages protocol."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _AnthropicBlockBase(BaseModel):
    """Pass through protocol extensions such as ``cache_control``."""

    model_config = ConfigDict(extra="allow")


class ContentBlockText(_AnthropicBlockBase):
    type: Literal["text"]
    text: str


class ContentBlockImage(_AnthropicBlockBase):
    type: Literal["image"]
    source: dict[str, Any]


class ContentBlockDocument(_AnthropicBlockBase):
    """Anthropic document block (e.g. PDF files via the Files API)."""

    type: Literal["document"]
    source: dict[str, Any]


class ContentBlockToolUse(_AnthropicBlockBase):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]


class ContentBlockToolResult(_AnthropicBlockBase):
    type: Literal["tool_result"]
    tool_use_id: str
    content: str | list[Any] | dict[str, Any]


class ContentBlockThinking(_AnthropicBlockBase):
    type: Literal["thinking"]
    thinking: str
    signature: str | None = None


class ContentBlockRedactedThinking(_AnthropicBlockBase):
    type: Literal["redacted_thinking"]
    data: str


class ContentBlockServerToolUse(_AnthropicBlockBase):
    """Anthropic server-side tool invocation (e.g. ``web_search``, ``web_fetch``)."""

    type: Literal["server_tool_use"]
    id: str
    name: str
    input: dict[str, Any]


class ContentBlockWebSearchToolResult(_AnthropicBlockBase):
    type: Literal["web_search_tool_result"]
    tool_use_id: str
    content: Any


class ContentBlockWebFetchToolResult(_AnthropicBlockBase):
    type: Literal["web_fetch_tool_result"]
    tool_use_id: str
    content: Any


class SystemContent(_AnthropicBlockBase):
    type: Literal["text"]
    text: str


_KNOWN_BLOCK_TYPES = frozenset(
    {
        "text",
        "image",
        "document",
        "tool_use",
        "tool_result",
        "thinking",
        "redacted_thinking",
        "server_tool_use",
        "web_search_tool_result",
        "web_fetch_tool_result",
    }
)


def _fold_tool_change_blocks(data: Any) -> Any:
    """Fold mid-conversation tool changes into ``tools``.

    Claude Code announces tools that load after the first turn with
    ``tool_addition`` / ``tool_removal`` blocks inside a system-role message.
    Providers behind this gateway only understand the top-level ``tools`` list,
    so apply the changes there in order and drop the blocks. Any other block
    type this gateway does not know is dropped too, instead of failing the
    whole request with a 422.
    """
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        return data
    tools = data.get("tools")
    tools = list(tools) if isinstance(tools, list) else []
    changed = False
    messages: list[Any] = []
    for message in data["messages"]:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            messages.append(message)
            continue
        kept: list[Any] = []
        for block in content:
            kind = block.get("type") if isinstance(block, dict) else None
            if not isinstance(block, dict) or kind in _KNOWN_BLOCK_TYPES:
                kept.append(block)
                continue
            changed = True
            tool = block.get("tool")
            if not isinstance(tool, dict):
                continue
            if kind == "tool_addition":
                definition = tool.get("definition", tool)
                name = definition.get("name") if isinstance(definition, dict) else None
                if isinstance(name, str) and not any(
                    isinstance(t, dict) and t.get("name") == name for t in tools
                ):
                    tools.append(definition)
            elif kind == "tool_removal":
                name = tool.get("name")
                tools = [
                    t
                    for t in tools
                    if not (isinstance(t, dict) and t.get("name") == name)
                ]
        if len(kept) == len(content):
            messages.append(message)
        elif kept:
            messages.append({**message, "content": kept})
    if not changed:
        return data
    data = {**data, "messages": messages}
    if tools or "tools" in data:
        data["tools"] = tools
    return data


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: (
        str
        | list[
            ContentBlockText
            | ContentBlockImage
            | ContentBlockDocument
            | ContentBlockToolUse
            | ContentBlockToolResult
            | ContentBlockThinking
            | ContentBlockRedactedThinking
            | ContentBlockServerToolUse
            | ContentBlockWebSearchToolResult
            | ContentBlockWebFetchToolResult
        ]
    )
    reasoning_content: str | None = None


class Tool(_AnthropicBlockBase):
    name: str
    type: str | None = None
    description: str | None = None
    input_schema: dict[str, Any] | None = None


class ThinkingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool | None = True
    type: str | None = None
    budget_tokens: int | None = None
    display: Literal["summarized", "omitted", "updates"] | None = None


class MessagesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    original_model: str | None = Field(default=None, exclude=True)
    resolved_provider_model: str | None = Field(default=None, exclude=True)
    max_tokens: int | None = None
    messages: list[Message]
    system: str | list[SystemContent] | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    metadata: dict[str, Any] | None = None
    tools: list[Tool] | None = None
    tool_choice: dict[str, Any] | None = None
    thinking: ThinkingConfig | None = None
    context_management: dict[str, Any] | None = None
    # Claude Code fast mode sends speed="fast"; accepted but not forwarded.
    speed: str | None = Field(default=None, exclude=True)
    output_config: dict[str, Any] | None = None
    mcp_servers: list[dict[str, Any]] | None = None
    extra_body: dict[str, Any] | None = None
    # Claude Code fast mode sends speed="fast"; accepted but never forwarded.
    speed: str | None = Field(default=None, exclude=True)
    betas: list[str] | None = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _fold_tool_changes(cls, data: Any) -> Any:
        return _fold_tool_change_blocks(data)

    @field_validator("max_tokens", mode="before")
    @classmethod
    def _reject_boolean_output_limit(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("max_tokens must not be a boolean")
        return value


class TokenCountRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    original_model: str | None = Field(default=None, exclude=True)
    resolved_provider_model: str | None = Field(default=None, exclude=True)
    messages: list[Message]
    system: str | list[SystemContent] | None = None
    tools: list[Tool] | None = None
    thinking: ThinkingConfig | None = None
    tool_choice: dict[str, Any] | None = None
    context_management: dict[str, Any] | None = None
    # Claude Code fast mode sends speed="fast"; accepted but not forwarded.
    speed: str | None = Field(default=None, exclude=True)
    output_config: dict[str, Any] | None = None
    mcp_servers: list[dict[str, Any]] | None = None
    betas: list[str] | None = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _fold_tool_changes(cls, data: Any) -> Any:
        return _fold_tool_change_blocks(data)


class TokenCountResponse(BaseModel):
    input_tokens: int


class NativeTokenCountMessage(BaseModel):
    """The outer message shape needed by local estimation, with opaque blocks."""

    model_config = ConfigDict(extra="allow", strict=True)

    role: str
    content: str | list[dict[str, Any]]


class NativeTokenCountRequest(BaseModel):
    """Count native inputs without applying compatibility protocol validation."""

    model_config = ConfigDict(extra="allow", strict=True)

    model: str = Field(min_length=1)
    messages: list[NativeTokenCountMessage]
    system: str | list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: list[
        ContentBlockText
        | ContentBlockToolUse
        | ContentBlockThinking
        | ContentBlockRedactedThinking
        | dict[str, Any]
    ]
    type: Literal["message"] = "message"
    stop_reason: (
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    ) = None
    stop_sequence: str | None = None
    usage: Usage
