"""Detect known Claude auto-mode classifier requests."""

from free_claude_code.core.safety_classifier import (
    detect_safety_classifier_stop_sequence,
    is_safety_classifier,
)

__all__ = ["detect_safety_classifier_stop_sequence", "is_safety_classifier"]
