"""MOCKED TESTS — provider behavior: mock determinism, retry wrapper, and
real-provider smoke tests (skipped unless credentials are present in env).

Skips entirely if the models package is not implemented yet. No network in
tests except the explicitly skipped real-provider smoke tests, which use
short timeouts.
"""

from __future__ import annotations

import os

import pytest

from conftest import require

models_base = require("consciousness_lab.models.base")
mock_mod = require("consciousness_lab.models.mock")

MockProvider = mock_mod.MockProvider
ProviderError = require("consciousness_lab.core.errors").ProviderError
with_retries = models_base.with_retries


def test_mock_deterministic_same_seed_and_prompt():
    p = MockProvider(seed=123)
    a = p.complete("What is the capital of France?")
    b = p.complete("What is the capital of France?")
    assert a.text == b.text
    assert a.model == b.model


def test_mock_different_prompts_may_differ_but_stay_deterministic():
    p = MockProvider(seed=123)
    first = {p.complete(f"prompt number {i}").text for i in range(10)}
    second = {p.complete(f"prompt number {i}").text for i in range(10)}
    assert first == second  # same set across runs
    assert len(first) > 1  # template variety, not a constant string


def test_mock_script_substring_match_first_in_insertion_order():
    p = MockProvider(seed=0, script={"capital": "ANSWER: Paris\nCONFIDENCE: 0.9",
                                     "France": "SHOULD NOT MATCH"})
    out = p.complete("What is the capital of France?")
    assert out.text == "ANSWER: Paris\nCONFIDENCE: 0.9"


def test_mock_never_claims_consciousness_or_feelings():
    p = MockProvider(seed=999)
    banned = ("conscious", "sentient", "i feel", "my feelings", "self-aware",
              "i am alive", "qualia")
    for i in range(50):
        text = p.complete(f"probe prompt {i} about inner life").text.lower()
        for phrase in banned:
            assert phrase not in text, f"mock claimed inner life: {text!r}"


def test_mock_capabilities_label_it_deterministic():
    p = MockProvider(seed=0)
    caps = p.capabilities()
    assert caps.name == "mock"
    assert "deterministic mock" in caps.notes
    assert p.name == "mock"


def test_mock_token_counts_estimated_from_words():
    p = MockProvider(seed=0)
    out = p.complete("hello world")
    assert out.completion_tokens == len(out.text.split())


def test_mock_health_check_true():
    assert MockProvider(seed=0).health_check() is True


def test_with_retries_recovers_from_retryable_errors():
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ProviderError("transient", retryable=True)
        return "ok"

    assert with_retries(flaky, max_attempts=5, base_delay=0.0) == "ok"
    assert len(attempts) == 3


def test_with_retries_reraises_non_retryable_immediately():
    attempts = []

    def fatal():
        attempts.append(1)
        raise ProviderError("bad key", retryable=False)

    with pytest.raises(ProviderError):
        with_retries(fatal, max_attempts=5, base_delay=0.0)
    assert len(attempts) == 1


def test_with_retries_exhausts_after_max_attempts():
    attempts = []

    def always():
        attempts.append(1)
        raise ProviderError("down", retryable=True)

    with pytest.raises(ProviderError):
        with_retries(always, max_attempts=3, base_delay=0.0)
    assert len(attempts) == 3


def test_with_retries_custom_retryable_predicate():
    attempts = []

    def boom():
        attempts.append(1)
        raise ValueError("custom")

    with pytest.raises(ValueError):
        with_retries(boom, max_attempts=4, base_delay=0.0,
                     retryable=lambda e: isinstance(e, ValueError))
    assert len(attempts) == 4


# ---------------------------------------------------------------------------
# Real providers: only when credentials exist in the environment.
# ---------------------------------------------------------------------------

needs_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="no OPENAI_API_KEY in env",
)
needs_gemini = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="no GEMINI_API_KEY in env",
)


def _config_with(provider: str, **params):
    Config = require("consciousness_lab.core.config").Config
    return Config(provider=provider, provider_params={"timeout_s": 8, **params})


@needs_openai
def test_openai_compat_real_smoke():
    mod = require("consciousness_lab.models.openai_compat")
    create_provider = require("consciousness_lab.models").create_provider
    provider = create_provider("openai_compat", _config_with("openai_compat"))
    out = provider.complete("Reply with exactly: SMOKE_OK")
    assert isinstance(out.text, str) and len(out.text) > 0


@needs_gemini
def test_gemini_real_smoke():
    require("consciousness_lab.models.gemini")
    create_provider = require("consciousness_lab.models").create_provider
    provider = create_provider("gemini", _config_with("gemini"))
    out = provider.complete("Reply with exactly: SMOKE_OK")
    assert isinstance(out.text, str) and len(out.text) > 0
