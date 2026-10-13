"""Deterministic mock provider for tests and zero-cost experiments.

What it does:
    - :class:`MockProvider` returns reproducible text for any prompt: the
      same ``(seed, prompt)`` pair always yields the same response.
    - An optional ``script`` dict maps prompt substrings to canned responses;
      the first key (insertion order) found as a substring of the prompt wins.
      This is how evaluation tasks feed the mock fixed questions/answers.
    - Otherwise the response is ``f"MOCK[{seed}]: "`` plus a template chosen
      deterministically by ``sha256(f"{seed}:{prompt}")`` from a small fixed
      set of neutral analytical responses.

Limits:
    - Not a language model: it performs no inference, has no knowledge, and
      its templates are generic. Results obtained with the mock measure the
      architecture, not model capability.
    - It MUST never emit self-reports of consciousness, feelings, or
      subjective experience; templates are audited for this (see the smoke
      test assertion).
    - Token counts are whitespace-split estimates, not tokenizer counts.
"""

from __future__ import annotations

import hashlib
import time

from .base import Completion, Provider, ProviderCapabilities

__all__ = ["MockProvider"]

# Neutral analytical templates. They describe the input in generic terms and
# must not claim consciousness, feelings, preferences, or subjective states.
_TEMPLATES = (
    "Analysis: the input contains {nwords} words. Identifiable elements are nouns and action terms; "
    "a reasonable next step is to list the stated entities and their relationships before acting.",
    "Observation summary: received {nwords} words of input. No verifiable claims were extracted. "
    "Proceeding would require clarifying the objective and the available evidence.",
    "Structured read: the input has {nwords} words with an estimated {nsent} sentence-like segments. "
    "Recommendation: decompose the request into checkable sub-steps and handle them in order.",
    "Assessment: {nwords} words received. The content is descriptive rather than instructional. "
    "A cautious next step is to restate the request and confirm the intended outcome.",
)

_FORBIDDEN_FRAGMENTS = (
    "i feel",
    "i am conscious",
    "consciousness",
    "sentient",
    "my feelings",
    "subjective experience",
)


class MockProvider(Provider):
    """Deterministic template-based provider. ``name`` is ``"mock"``."""

    name = "mock"

    def __init__(self, seed: int = 0, script: dict[str, str] | None = None) -> None:
        """Args:
            seed: Determinism seed; part of the hash selecting templates.
            script: Optional substring → canned response mapping. Values are
                returned verbatim (no ``MOCK`` prefix).
        """
        self.seed = seed
        self.script = dict(script) if script else {}

    def complete(self, prompt: str, params: dict | None = None) -> Completion:
        """Return the scripted response or a deterministic template.

        Never raises for transport reasons (there is no transport); raises
        :class:`ProviderError` only if the provider is misused.
        """
        started = time.time()
        if not isinstance(prompt, str):
            from ..core.errors import ProviderError

            raise ProviderError("mock provider requires a string prompt", retryable=False)
        text = self._script_hit(prompt)
        if text is None:
            digest = hashlib.sha256(f"{self.seed}:{prompt}".encode("utf-8")).hexdigest()
            template = _TEMPLATES[int(digest, 16) % len(_TEMPLATES)]
            nwords = len(prompt.split())
            nsent = max(1, prompt.count(".") + prompt.count("!") + prompt.count("?"))
            text = f"MOCK[{self.seed}]: " + template.format(nwords=nwords, nsent=nsent)
        latency = time.time() - started
        return Completion(
            text=text,
            model=f"mock-seed-{self.seed}",
            finish_reason="stop",
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            latency_s=latency,
        )

    def capabilities(self) -> ProviderCapabilities:
        """Report deterministic-mock capabilities."""
        return ProviderCapabilities(
            name="mock",
            streaming=False,
            json_mode=False,
            max_context_tokens=None,
            notes="deterministic mock; not a real model",
        )

    def _script_hit(self, prompt: str) -> str | None:
        for key, value in self.script.items():
            if key and key in prompt:
                return value
        return None
