"""Google Gemini adapter over stdlib ``urllib``.

What it does:
    - :class:`GeminiProvider` POSTs to
      ``https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent``
      and returns the concatenated text parts of the first candidate.

Secrets handling:
    - The API key comes ONLY from the ``GEMINI_API_KEY`` environment variable
      (or the explicit ``api_key`` argument); never from config files.
    - **Documented limitation (per contract):** the Gemini REST API takes the
      key as a ``?key=`` query parameter, so the key travels in the URL and
      may appear in proxy/server access logs. We never log the URL with the
      key attached; prefer a restricted key.

Limits:
    - Model name falls back to ``GEMINI_MODEL`` env, then ``"gemini-2.0-flash"``
      (verify against Google's current model list; the default may be stale).
    - Retries apply only to HTTP 429/5xx. 400/401/403 (bad key, bad model,
      blocked prompt) raise immediately with ``retryable=False``.
    - A ``promptFeedback.blockReason`` in the response raises
      :class:`ProviderError` (not retryable) instead of returning empty text.
    - Full prompts are never logged at INFO; at DEBUG only a 200-character
      preview. Gemini ``generateContent`` does not return token usage in the
      basic response, so token counts are whitespace estimates.
"""

from __future__ import annotations

import logging
import os
import time
from urllib.parse import quote

from ..core.errors import ConfigError, ProviderError
from .base import Completion, Provider, ProviderCapabilities, _http_post_json, with_retries

__all__ = ["GeminiProvider"]

logger = logging.getLogger("consciousness_lab.models")
_API_HOST = "https://generativelanguage.googleapis.com"


class GeminiProvider(Provider):
    """Gemini ``generateContent`` adapter. ``name`` is ``"gemini"``."""

    name = "gemini"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        """Args:
            api_key: Falls back to ``GEMINI_API_KEY`` env. Required.
            model: Falls back to ``GEMINI_MODEL`` env, then ``"gemini-2.0-flash"``.
            timeout_s: Per-request socket timeout in seconds.

        Raises:
            ConfigError: If no API key is available from env or argument.
        """
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ConfigError(
                "GEMINI_API_KEY is not set; the gemini provider needs it in the environment"
            )
        self._api_key = key
        self.model = model or os.environ.get("GEMINI_MODEL") or "gemini-2.0-flash"
        self.timeout_s = float(timeout_s)

    def _url(self) -> str:
        # Documented limitation: the key travels as a query parameter.
        return f"{_API_HOST}/v1beta/models/{quote(self.model, safe='')}:generateContent?key={quote(self._api_key, safe='')}"

    def _redacted_url(self) -> str:
        return f"{_API_HOST}/v1beta/models/{self.model}:generateContent?key=***REDACTED***"

    def complete(self, prompt: str, params: dict | None = None) -> Completion:
        """Generate via ``generateContent`` with retries on 429/5xx."""
        return with_retries(lambda: self._complete_once(prompt, params))

    def _complete_once(self, prompt: str, params: dict | None) -> Completion:
        payload: dict = {"contents": [{"parts": [{"text": prompt}]}]}
        options = (params or {}).get("options")
        if isinstance(options, dict):
            payload.update(options)
        # Never log the key-bearing URL; never log the full prompt at INFO.
        logger.debug(
            "gemini POST %s model=%s prompt_chars=%d preview=%.200s",
            self._redacted_url(),
            self.model,
            len(prompt),
            prompt.replace("\n", " "),
        )
        started = time.time()
        _, data = _http_post_json(self._url(), payload, {}, self.timeout_s)
        feedback = data.get("promptFeedback") if isinstance(data.get("promptFeedback"), dict) else {}
        block_reason = feedback.get("blockReason")
        if isinstance(block_reason, str) and block_reason:
            raise ProviderError(
                f"gemini blocked the prompt: {block_reason}", retryable=False
            )
        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ProviderError(
                "gemini returned malformed response (no candidates)", retryable=False
            )
        first = candidates[0] if isinstance(candidates[0], dict) else {}
        content = first.get("content") if isinstance(first.get("content"), dict) else {}
        parts = content.get("parts") if isinstance(content.get("parts"), list) else []
        texts = [p.get("text") for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str)]
        if not texts:
            raise ProviderError(
                "gemini returned malformed response (no text parts)", retryable=False
            )
        text = "".join(texts)
        finish = first.get("finishReason", "STOP")
        return Completion(
            text=text,
            model=self.model,
            finish_reason="stop" if finish in ("STOP", "stop") else str(finish).lower(),
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            latency_s=time.time() - started,
        )

    def capabilities(self) -> ProviderCapabilities:
        """Report Gemini adapter capabilities."""
        return ProviderCapabilities(
            name="gemini",
            streaming=False,
            json_mode=False,
            max_context_tokens=None,
            notes="API key sent as URL query parameter (documented limitation); context window depends on the model",
        )
