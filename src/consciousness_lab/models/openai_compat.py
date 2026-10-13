"""OpenAI-compatible chat-completions adapter over stdlib ``urllib``.

What it does:
    - :class:`OpenAICompatibleProvider` POSTs to
      ``{base_url}/chat/completions`` with a single user message and returns
      ``choices[0].message.content``. Works against OpenAI and any
      OpenAI-compatible endpoint (override via ``OPENAI_BASE_URL``).

Secrets handling:
    - The API key comes ONLY from the ``OPENAI_API_KEY`` environment variable
      (or the explicit ``api_key`` argument); it is never read from config
      files and never logged. A missing key raises :class:`ConfigError` at
      construction time, not at call time.

Limits (documented per contract):
    - Bearer auth in the ``Authorization`` header; the key is not redacted in
      tracebacks by us, so never print exception chains containing headers
      (we do not include them).
    - Retries apply only to HTTP 429/5xx. 401/403 raise immediately with
      ``retryable=False`` (fix the key, don't retry).
    - Full prompts are never logged at INFO; at DEBUG only a 200-character
      preview is logged. Token usage comes from the server's ``usage`` block
      when present, else 0.
"""

from __future__ import annotations

import logging
import os
import time

from ..core.errors import ConfigError, ProviderError
from .base import Completion, Provider, ProviderCapabilities, _http_post_json, with_retries

__all__ = ["OpenAICompatibleProvider"]

logger = logging.getLogger("consciousness_lab.models")


class OpenAICompatibleProvider(Provider):
    """Chat-completions adapter. ``name`` is ``"openai_compat"``."""

    name = "openai_compat"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        """Args:
            api_key: Falls back to ``OPENAI_API_KEY`` env. Required.
            base_url: Falls back to ``OPENAI_BASE_URL`` env, then
                ``https://api.openai.com/v1``.
            model: Falls back to ``OPENAI_MODEL`` env, then ``"gpt-4o-mini"``.
            timeout_s: Per-request socket timeout in seconds.

        Raises:
            ConfigError: If no API key is available from env or argument.
        """
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ConfigError(
                "OPENAI_API_KEY is not set; the openai_compat provider needs it in the environment"
            )
        self._api_key = key
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        self.timeout_s = float(timeout_s)

    def complete(self, prompt: str, params: dict | None = None) -> Completion:
        """Generate via ``/chat/completions`` with retries on 429/5xx."""
        return with_retries(lambda: self._complete_once(prompt, params))

    def _complete_once(self, prompt: str, params: dict | None) -> Completion:
        url = f"{self.base_url}/chat/completions"
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        options = (params or {}).get("options")
        if isinstance(options, dict):
            payload.update(options)
        # Never log the key; never log the full prompt at INFO.
        logger.debug(
            "openai_compat POST model=%s prompt_chars=%d preview=%.200s",
            self.model,
            len(prompt),
            prompt.replace("\n", " "),
        )
        started = time.time()
        try:
            _, data = _http_post_json(
                url, payload, {"Authorization": f"Bearer {self._api_key}"}, self.timeout_s
            )
        except ProviderError as exc:
            if "HTTP 401" in str(exc) or "HTTP 403" in str(exc):
                raise ProviderError(
                    "openai_compat authentication failed (401/403); check OPENAI_API_KEY",
                    retryable=False,
                ) from exc
            raise
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError("openai_compat returned malformed response (no choices)", retryable=False)
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str):
            raise ProviderError(
                "openai_compat returned malformed response (no message content)", retryable=False
            )
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return Completion(
            text=text,
            model=data.get("model", self.model) if isinstance(data.get("model"), str) else self.model,
            finish_reason=choices[0].get("finish_reason", "stop")
            if isinstance(choices[0], dict)
            else "stop",
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_s=time.time() - started,
        )

    def capabilities(self) -> ProviderCapabilities:
        """Report OpenAI-compatible adapter capabilities."""
        return ProviderCapabilities(
            name="openai_compat",
            streaming=False,
            json_mode=False,
            max_context_tokens=None,
            notes="OpenAI-compatible chat completions; context window depends on the model",
        )
