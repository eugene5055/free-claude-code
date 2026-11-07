"""One-shot repair of tool calls that omit schema-required arguments.

Some models (notably small local ones) occasionally emit a tool call whose
arguments are ``{}`` or otherwise lack a property the tool's JSON schema marks
``required``. Clients then fail the call (for example an MCP server answering
``-32602: missing field `code```). :class:`ToolCallRepairGuard` lets the Chat
stream runner hold back tool-call output until the upstream response
completes, so a defective call can be retried once with a short corrective
note before anything about it reaches the client.

Streaming tradeoff: text and reasoning that precede the first tool-call delta
keep streaming to the client immediately. From the first tool-call delta on,
every chunk of that response is held. When the held calls are defective, the
retry request carries the already-streamed text back as an assistant message
followed by the corrective note, and the retry's output is appended to what
the client already saw (the client never sees the defective call or anything
after it). The retry is attempted at most once per request; if it still fails
validation its output passes through unchanged, and if it errors the original
response is released unchanged.

Only presence of ``required`` properties and object-ness of the arguments are
checked; this is deliberately not full JSON-schema validation. Tools without a
schema, tools whose schema has no ``required`` list, unknown tools and
freeform/custom tools are skipped, as are arguments that are not valid JSON
(those keep their existing transparent pass-through).
"""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from free_claude_code.core.anthropic.streaming import ToolSchema
from free_claude_code.core.openai_tool_names import OpenAIToolNameCodec

from .tool_calls import OpenAIToolCallCollector, restore_tool_argument_aliases


@dataclass(frozen=True, slots=True)
class ToolCallDefect:
    """One emitted tool call whose arguments violate its schema's basics."""

    wire_name: str
    name: str
    missing: tuple[str, ...]
    not_object: bool

    def describe(self) -> str:
        if self.not_object:
            required = ", ".join(f"`{field}`" for field in self.missing)
            suffix = f" containing {required}" if required else ""
            return (
                f"Your call to `{self.wire_name}` did not pass its arguments as a "
                f"JSON object; call it again with a JSON object{suffix}."
            )
        missing = ", ".join(f"`{field}`" for field in self.missing)
        return (
            f"Your call to `{self.wire_name}` was missing required argument(s) "
            f"{missing}; call it again with all required arguments."
        )


def required_argument_defect(
    arguments: str, schema: Mapping[str, Any]
) -> tuple[tuple[str, ...], bool] | None:
    """Return ``(missing, not_object)`` for defective arguments, else None."""
    raw_required = schema.get("required")
    required = (
        tuple(field for field in raw_required if isinstance(field, str))
        if isinstance(raw_required, list)
        else ()
    )
    if not required:
        return None
    if not arguments.strip():
        parsed: Any = {}
    else:
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            # Malformed or truncated JSON belongs to the existing stream
            # machinery, which passes it through transparently.
            return None
    if not isinstance(parsed, dict):
        return required, True
    missing = tuple(field for field in required if field not in parsed)
    return (missing, False) if missing else None


def tool_call_repair_body(
    body: Mapping[str, Any],
    defects: Iterable[ToolCallDefect],
    *,
    prior_text: str,
) -> dict[str, Any]:
    """Return one Chat body asking the model to redo its defective calls."""
    messages = list(body.get("messages") or [])
    if prior_text.strip():
        messages.append({"role": "assistant", "content": prior_text})
    note = " ".join(defect.describe() for defect in defects)
    if prior_text.strip():
        note += " Do not repeat text you already wrote."
    messages.append({"role": "user", "content": note})
    return {**body, "messages": messages}


class ToolCallRepairGuard:
    """Hold one Chat response's tool-call chunks until they can be validated."""

    def __init__(
        self,
        *,
        schemas: Mapping[str, ToolSchema],
        tool_names: OpenAIToolNameCodec,
        argument_aliases: Mapping[str, Mapping[str, str]],
        skip_names: frozenset[str] = frozenset(),
    ) -> None:
        self._schemas = schemas
        self._tool_names = tool_names
        self._argument_aliases = argument_aliases
        self._skip_names = skip_names
        self._collector = OpenAIToolCallCollector()
        self._held: list[Any] = []
        self._holding = False
        self._unjudgeable = False
        self._text_parts: list[str] = []
        self.finish_reason: Any = None

    @property
    def holding(self) -> bool:
        return self._holding

    @property
    def prior_text(self) -> str:
        """Visible content forwarded before the first tool-call delta."""
        return "".join(self._text_parts)

    def admit(self, chunk: Any) -> tuple[Any, ...]:
        """Return the chunks that may be forwarded to the client now."""
        choices = getattr(chunk, "choices", None) or ()
        choice = choices[0] if choices else None
        delta = getattr(choice, "delta", None) if choice is not None else None
        if choice is not None and choice.finish_reason is not None:
            self.finish_reason = choice.finish_reason
        tool_calls = getattr(delta, "tool_calls", None) if delta is not None else None
        if tool_calls:
            self._holding = True
            for tool_call in tool_calls:
                try:
                    self._collector.add(tool_call)
                except Exception:
                    # The stream assembler owns identity errors; an
                    # unjudgeable response simply passes through.
                    self._unjudgeable = True
        if self._holding:
            self._held.append(chunk)
            return ()
        content = getattr(delta, "content", None) if delta is not None else None
        if isinstance(content, str) and content:
            self._text_parts.append(content)
        return (chunk,)

    def release(self) -> tuple[Any, ...]:
        """Return and forget every held chunk."""
        held = tuple(self._held)
        self._held.clear()
        return held

    def defects(self) -> tuple[ToolCallDefect, ...]:
        """Return defective held calls; incomplete identities are skipped."""
        if self._unjudgeable:
            return ()
        found: list[ToolCallDefect] = []
        for wire_name, arguments in self._collector.raw_calls():
            name = self._tool_names.decode(wire_name)
            if name in self._skip_names:
                continue
            tool_schema = self._schemas.get(name)
            schema = tool_schema.input_schema if tool_schema is not None else None
            if not isinstance(schema, Mapping) or not schema:
                continue
            aliases = self._argument_aliases.get(name)
            if aliases and arguments.strip():
                restored = restore_tool_argument_aliases(arguments, dict(aliases))
                if restored is not None:
                    arguments = restored
            defect = required_argument_defect(arguments, schema)
            if defect is None:
                continue
            missing, not_object = defect
            found.append(
                ToolCallDefect(
                    wire_name=wire_name,
                    name=name,
                    missing=missing,
                    not_object=not_object,
                )
            )
        return tuple(found)
