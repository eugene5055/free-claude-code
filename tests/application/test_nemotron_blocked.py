"""Nemotron is never offered on any provider: Super and Lightning (user, 2026-10-22) and Ultra (user, 2026-11-08)."""

import pytest

from free_claude_code.application import model_usability


@pytest.mark.parametrize(
    "ref",
    [
        "nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b",
        "open_router/nvidia/nemotron-3-ultra-550b-a55b:free",
        "nvidia_nim/nvidia/nemotron-3-super-120b-a12b",
        "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b",
    ],
)
def test_every_nemotron_variant_is_blocked(ref):
    assert model_usability.blocked(ref)
    assert not model_usability.offered(ref)


@pytest.mark.parametrize(
    "ref", ["nvidia_nim/minimaxai/minimax-m3.1-flash", "kimi_code/k3.1-pro", "qwencloud/qwen3.8-max"]
)
def test_other_models_stay_offered(ref):
    assert not model_usability.blocked(ref)
    assert model_usability.offered(ref)


@pytest.mark.parametrize(
    "ref",
    [
        "groq/openai/gpt-oss-120b",
        "cloudflare/@cf/openai/gpt-oss-120b",
        "nvidia_nim/openai/gpt-oss-20b",
        "open_router/cohere/north-mini-code:free",
        "cloudflare/@cf/qwen/qwen3.8-27b",
        "cloudflare/@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "open_router/google/gemma-4-31b-it:free",
        "mistral/devstral-medium-latest",
        "mistral/mistral-small-latest",
    ],
)
def test_models_below_the_45_point_floor_are_blocked(ref):
    """User, 2026-11-08: only models with an LLM Stats score of at least 45."""
    assert model_usability.blocked(ref)
    assert not model_usability.offered(ref)


@pytest.mark.parametrize(
    "ref",
    ["custom_73e8e40b1fc34123ba862c9685cacfa5/duo-local-combined", "llamacpp/duo-local-combined", "mistral/mistral-large-latest"],
)
def test_floor_keeps_local_and_frontier_models(ref):
    assert not model_usability.blocked(ref)
