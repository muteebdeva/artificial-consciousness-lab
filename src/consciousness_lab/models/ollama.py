"""Ollama adapter over stdlib ``urllib``.

What it does:
    - :class:`OllamaProvider` POSTs to ``{base_url}/api/generate`` with
      ``{"model", "prompt", "stream": false}`` and returns the ``response``
      field as the completion text.

Limits (documented per contract):
    - Requires a running Ollama server; the default base URL is
      ``http://localhost:11434`` (override with ``OLLAMA_BASE_URL`` or the
      ``base_url`` argument).
    - No streaming support in this adapter; ``stream: false`` is always sent.
    - The model's context window is not validated here — overlong prompts
      fail server-side and surface as :class:`ProviderError`.
    - Retries apply only to HTTP 429/5xx (``retryable=True``). Auth is not
      used by Ollama; connection failures raise immediately.
    - No secrets are involved; prompts are never logged above DEBUG, and then
      only truncated to 200 characters.
"""

from __future__ import annotations

import logging
import os
import time

from ..core.errors import ProviderError
from .base import Completion, Provider, ProviderCapabilities, _http_post_json, with_retries

__all__ = ["OllamaProvider"]

logger = logging.getLogger("consciousness_lab.models")


class OllamaProvider(Provider):
    """Inference via a local Ollama server. ``name`` is ``"ollama"``."""

    name = "ollama"

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        """Args:
            model: Ollama model tag. Falls back to ``OLLAMA_MODEL`` env, then
                ``"llama3.1"`` (verify the tag exists on your server).
            base_url: Server base URL. Falls back to ``OLLAMA_BASE_URL`` env,
                then ``http://localhost:11434``.
            timeout_s: Per-request socket timeout in seconds.
        """
        self.model = model or os.environ.get("OLLAMA_MODEL") or "llama3.1"
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        self.timeout_s = float(timeout_s)

    def complete(self, prompt: str, params: dict | None = None) -> Completion:
        """Generate via ``/api/generate`` with retries on 429/5xx."""
        return with_retries(lambda: self._complete_once(prompt, params))

    def _complete_once(self, prompt: str, params: dict | None) -> Completion:
        url = f"{self.base_url}/api/generate"
        payload: dict = {"model": self.model, "prompt": prompt, "stream": False}
        options = (params or {}).get("options")
        if isinstance(options, dict):
            payload["options"] = options
        logger.debug("ollama POST %s model=%s prompt_chars=%d", url, self.model, len(prompt))
        started = time.time()
        _, data = _http_post_json(url, payload, {}, self.timeout_s)
        if isinstance(data.get("error"), str) and data["error"]:
            raise ProviderError(f"ollama error: {data['error'][:300]}", retryable=False)
        text = data.get("response")
        if not isinstance(text, str):
            raise ProviderError(
                "ollama returned malformed response (missing string 'response')", retryable=False
            )
        return Completion(
            text=text,
            model=self.model,
            finish_reason="stop" if data.get("done", True) else "length",
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
            latency_s=time.time() - started,
        )

    def capabilities(self) -> ProviderCapabilities:
        """Report Ollama adapter capabilities."""
        return ProviderCapabilities(
            name="ollama",
            streaming=False,
            json_mode=False,
            max_context_tokens=None,
            notes="local Ollama server required; context window depends on the model tag",
        )
