"""Provider-owned reasoning translations for OpenAI-compatible APIs."""

import re
from dataclasses import dataclass
from typing import Any, Protocol

from free_claude_code.core.reasoning import (
    ReasoningControl,
    ReasoningEffort,
    ReasoningPolicy,
)

EffortValues = tuple[tuple[ReasoningEffort, str], ...]


class ReasoningEncoder(Protocol):
    """Translate provider-neutral reasoning intent into one wire shape."""

    def encode(self, body: dict[str, Any], policy: ReasoningPolicy) -> None: ...

    @property
    def off_field(self) -> tuple[str, ...] | None: ...


@dataclass(frozen=True, slots=True)
class NoReasoning:
    """Leave reasoning computation entirely to the upstream provider."""

    off_field = None

    def encode(self, body: dict[str, Any], policy: ReasoningPolicy) -> None:
        return


@dataclass(frozen=True, slots=True)
class NamedEffortReasoning:
    """Encode a provider's documented named-effort vocabulary."""

    efforts: EffortValues
    disabled_value: str | bool | None = None
    enabled_value: str | bool | None = None
    field: str = "reasoning_effort"
    budget_field: str | None = None
    use_extra_body: bool = False

    @property
    def off_field(self) -> tuple[str, ...] | None:
        if self.disabled_value is None:
            return None
        return ("extra_body", self.field) if self.use_extra_body else (self.field,)

    def encode(self, body: dict[str, Any], policy: ReasoningPolicy) -> None:
        target = _extra_body(body) if self.use_extra_body else body
        if policy.control is ReasoningControl.OFF:
            if self.disabled_value is not None:
                target[self.field] = self.disabled_value
            return

        if policy.budget_tokens is not None and self.budget_field is not None:
            target[self.budget_field] = policy.budget_tokens
            return

        effort = dict(self.efforts).get(policy.effort)
        if effort is not None:
            target[self.field] = effort
            return

        if policy.control is ReasoningControl.ON and self.enabled_value is not None:
            target[self.field] = self.enabled_value


@dataclass(frozen=True, slots=True)
class ReasoningObject:
    """Encode gateways that accept a top-level ``reasoning`` object."""

    efforts: EffortValues
    supports_budget: bool = True

    off_field = ("extra_body", "reasoning")

    def encode(self, body: dict[str, Any], policy: ReasoningPolicy) -> None:
        if policy.control is ReasoningControl.OFF:
            _extra_body(body)["reasoning"] = {"enabled": False}
            return

        reasoning: dict[str, Any] = {}
        if policy.budget_tokens is not None and self.supports_budget:
            reasoning["max_tokens"] = policy.budget_tokens
        elif effort := dict(self.efforts).get(policy.effort):
            reasoning["effort"] = effort
        elif policy.control is ReasoningControl.ON:
            reasoning["enabled"] = True

        if reasoning:
            _extra_body(body)["reasoning"] = reasoning


@dataclass(frozen=True, slots=True)
class ThinkingObjectReasoning:
    """Encode providers with an enabled/disabled ``thinking`` object."""

    enabled: dict[str, Any]
    disabled: dict[str, Any]

    off_field = ("extra_body", "thinking")

    def encode(self, body: dict[str, Any], policy: ReasoningPolicy) -> None:
        if policy.control is ReasoningControl.OFF:
            _extra_body(body)["thinking"] = dict(self.disabled)
        elif policy.requests_reasoning:
            _extra_body(body)["thinking"] = dict(self.enabled)


@dataclass(frozen=True, slots=True)
class ChatTemplateReasoning:
    """Encode a provider-wide chat-template boolean without model guessing."""

    field: str = "thinking"

    @property
    def off_field(self) -> tuple[str, ...]:
        return ("extra_body", "chat_template_kwargs", self.field)

    def encode(self, body: dict[str, Any], policy: ReasoningPolicy) -> None:
        if not policy.requests_reasoning and policy.control is not ReasoningControl.OFF:
            return
        kwargs = _nested_dict(_extra_body(body), "chat_template_kwargs")
        kwargs[self.field] = policy.control is not ReasoningControl.OFF


@dataclass(frozen=True, slots=True)
class LlamaCppReasoning:
    """Encode llama.cpp's per-request numeric thinking budget."""

    off_field = ("extra_body", "thinking_budget_tokens")

    def encode(self, body: dict[str, Any], policy: ReasoningPolicy) -> None:
        if policy.control is ReasoningControl.OFF:
            _extra_body(body)["thinking_budget_tokens"] = 0
        elif (budget := policy.numeric_budget_tokens) is not None:
            _extra_body(body)["thinking_budget_tokens"] = budget


# Qwen3.x chat templates accept only these reasoning_effort values and raise on
# any other; llama.cpp maps "none" to enable_thinking=false before templating.
_LLAMACPP_TEMPLATE_EFFORTS = {
    ReasoningEffort.MINIMAL: "low",
    ReasoningEffort.LOW: "low",
    ReasoningEffort.MEDIUM: "medium",
    ReasoningEffort.HIGH: "xhigh",
    ReasoningEffort.XHIGH: "xhigh",
    ReasoningEffort.MAX: "xhigh",
}
# Exact budgets also pick the template effort, so the prompt's guidance on how
# long to think agrees with the hard stop.
_LLAMACPP_BUDGET_EFFORTS = ((2_048, "low"), (8_192, "medium"))
LLAMACPP_BUDGET_MESSAGE = (
    "\n\nI have used my thinking budget, so I will stop thinking and answer now.\n"
)


def llamacpp_reasoning_fields(policy: ReasoningPolicy) -> dict[str, Any]:
    """Map FCC reasoning intent to llama.cpp request fields for Qwen templates.

    Returns one flat dict of the top-level ``reasoning_effort`` and llama.cpp's
    ``thinking_budget_tokens`` / ``reasoning_budget_message`` extensions, for
    any llama.cpp-facing caller:

    - off / prefer-off: ``reasoning_effort="none"`` (llama.cpp disables thinking)
    - exact budget: a template effort chosen from the budget, plus a hard stop
    - effort: template effort low / medium / xhigh; minimal also gets a hard stop.
      Higher efforts keep unbounded thinking, as Anthropic's adaptive mode does.
    - on without detail: ``xhigh`` (the template default, sent explicitly)
    - provider default: nothing, so the template default applies
    """
    if policy.control in {ReasoningControl.OFF, ReasoningControl.PREFER_OFF}:
        return {"reasoning_effort": "none"}
    if policy.budget_tokens is not None:
        effort = next(
            (
                name
                for limit, name in _LLAMACPP_BUDGET_EFFORTS
                if policy.budget_tokens <= limit
            ),
            "xhigh",
        )
        return {
            "reasoning_effort": effort,
            "thinking_budget_tokens": policy.budget_tokens,
            "reasoning_budget_message": LLAMACPP_BUDGET_MESSAGE,
        }
    if policy.effort is not None:
        fields: dict[str, Any] = {
            "reasoning_effort": _LLAMACPP_TEMPLATE_EFFORTS[policy.effort]
        }
        if policy.effort is ReasoningEffort.MINIMAL:
            fields["thinking_budget_tokens"] = policy.effort.budget_tokens
            fields["reasoning_budget_message"] = LLAMACPP_BUDGET_MESSAGE
        return fields
    if policy.control is ReasoningControl.ON:
        return {"reasoning_effort": "xhigh"}
    return {}


@dataclass(frozen=True, slots=True)
class LlamaCppTemplateReasoning:
    """Encode llama.cpp + Qwen-template reasoning: named effort and exact budgets."""

    off_field = ("reasoning_effort",)

    def encode(self, body: dict[str, Any], policy: ReasoningPolicy) -> None:
        for key, value in llamacpp_reasoning_fields(policy).items():
            if key == "reasoning_effort":
                body[key] = value
            else:
                _extra_body(body)[key] = value


@dataclass(frozen=True, slots=True)
class SplitReasoningOutput:
    """Request separate reasoning output where compute is not controllable."""

    off_field = None

    def encode(self, body: dict[str, Any], policy: ReasoningPolicy) -> None:
        _extra_body(body)["reasoning_split"] = True


@dataclass(frozen=True, slots=True)
class ModelFamilyReasoning:
    """Choose each model family's accepted vocabulary and always send a control.

    Multi-model gateways (Alibaba Token Plan, Kimi Code) accept different
    ``reasoning_effort`` values per model: GLM refuses ``none``/``minimal``/
    ``medium``, Qwen and DeepSeek accept ``none``, Kimi K3 Pro cannot stop
    thinking. Unsupported values are clamped to the nearest accepted one, and a
    request without client intent gets the family's lowest explicit level
    instead of the upstream default (which is maximum thinking).
    """

    families: tuple[tuple[str, NamedEffortReasoning], ...]
    fallback: NamedEffortReasoning

    @property
    def off_field(self) -> tuple[str, ...]:
        return (self.fallback.field,)

    def encoder_for(self, model: object) -> NamedEffortReasoning:
        name = str(model or "").lower().rpartition("/")[2]
        for pattern, encoder in self.families:
            if re.search(pattern, name):
                return encoder
        return self.fallback

    def encode(self, body: dict[str, Any], policy: ReasoningPolicy) -> None:
        if (
            policy.control is ReasoningControl.DEFAULT
            and policy.effort is None
            and policy.budget_tokens is None
        ):
            policy = ReasoningPolicy.off()
        self.encoder_for(body.get("model")).encode(body, policy)


def _extra_body(body: dict[str, Any]) -> dict[str, Any]:
    value = body.setdefault("extra_body", {})
    if not isinstance(value, dict):
        raise TypeError("OpenAI extra_body must be an object.")
    return value


def _nested_dict(container: dict[str, Any], key: str) -> dict[str, Any]:
    value = container.setdefault(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be an object.")
    return value


NO_REASONING = NoReasoning()
LLAMACPP_REASONING = LlamaCppReasoning()
LLAMACPP_TEMPLATE_REASONING = LlamaCppTemplateReasoning()
SPLIT_REASONING_OUTPUT = SplitReasoningOutput()
