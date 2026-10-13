"""Inference provider adapters.

What it does:
    - :func:`create_provider` builds the configured provider from a
      :class:`~consciousness_lab.core.config.Config`.

Provider selection (``config.provider``):
    - ``"mock"`` — deterministic template provider (default; no credentials).
    - ``"ollama"`` — local Ollama server (``OLLAMA_BASE_URL`` / ``OLLAMA_MODEL``).
    - ``"openai_compat"`` — OpenAI-compatible chat completions; requires
      ``OPENAI_API_KEY`` in the environment.
    - ``"gemini"`` — Google Gemini; requires ``GEMINI_API_KEY`` in the
      environment.

Limits:
    - Credentials are read from the environment only, by the adapters
      themselves; this module never handles secret values.
    - Unknown provider names raise :class:`ConfigError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Completion, Provider, ProviderCapabilities, with_retries
from .gemini import GeminiProvider
from .mock import MockProvider
from .ollama import OllamaProvider
from .openai_compat import OpenAICompatibleProvider

if TYPE_CHECKING:  # avoid a hard import; Config is only a type hint here
    from ..core.config import Config
    from ..core.errors import ConfigError

__all__ = [
    "Completion",
    "Provider",
    "ProviderCapabilities",
    "with_retries",
    "MockProvider",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "GeminiProvider",
    "create_provider",
]


def create_provider(name: str, config: "Config") -> Provider:
    """Create the named provider from ``config``.

    Args:
        name: One of ``mock`` | ``ollama`` | ``openai_compat`` | ``gemini``
            (case-insensitive).
        config: Lab config; supplies ``seed``, ``provider_model``, and
            ``provider_params`` (notably ``timeout_s`` and, for the mock,
            an optional ``script`` mapping).

    Raises:
        ConfigError: For unknown provider names, or when a real provider's
            required API key is missing from the environment.
    """
    from ..core.errors import ConfigError  # deferred: keep models importable standalone

    key = (name or "").strip().lower()
    params = config.provider_params if isinstance(config.provider_params, dict) else {}
    timeout_s = params.get("timeout_s", 30)
    try:
        timeout = float(timeout_s)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"provider_params.timeout_s must be numeric, got {timeout_s!r}") from exc

    if key == "mock":
        script = params.get("script")
        return MockProvider(seed=config.seed, script=script if isinstance(script, dict) else None)
    if key == "ollama":
        return OllamaProvider(
            model=config.provider_model or None,
            timeout_s=timeout,
        )
    if key == "openai_compat":
        return OpenAICompatibleProvider(
            model=config.provider_model or None,
            timeout_s=timeout,
        )
    if key == "gemini":
        return GeminiProvider(
            model=config.provider_model or None,
            timeout_s=timeout,
        )
    raise ConfigError(
        f"unknown provider {name!r}; expected one of: mock, ollama, openai_compat, gemini"
    )
