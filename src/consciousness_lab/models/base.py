"""Provider abstraction: completions, capabilities, retry helper, HTTP plumbing.

What it does:
    - :class:`Completion` / :class:`ProviderCapabilities` are the uniform
      result/capability records every adapter returns.
    - :class:`Provider` is the abstract inference interface with a default
      :meth:`health_check` (tiny probe call; any failure means ``False``).
    - :func:`with_retries` runs a thunk with exponential backoff + jitter.
    - :func:`_http_post_json` (private) performs one JSON POST over stdlib
      ``urllib`` and normalises transport/HTTP/JSON failures into
      :class:`ProviderError`.

Limits:
    - Retries only happen for errors the ``retryable`` predicate accepts
      (default: ``ProviderError`` with ``retryable=True``, i.e. HTTP 429/5xx).
      Everything else — auth failures, malformed responses, connection
      errors — raises immediately.
    - ``_http_post_json`` never logs URLs that may contain credentials or
      request bodies; callers must keep secrets and full prompts out of logs.
    - Jitter uses :mod:`random`; pass ``seed`` to :func:`with_retries` for
      deterministic test runs.
"""

from __future__ import annotations

import abc
import json
import logging
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

from ..core.errors import LabError, ProviderError

__all__ = [
    "Completion",
    "ProviderCapabilities",
    "Provider",
    "with_retries",
]

logger = logging.getLogger("consciousness_lab.models")


@dataclass
class Completion:
    """One completed provider call."""

    text: str
    model: str
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


@dataclass
class ProviderCapabilities:
    """Self-reported provider capabilities.

    These are reported by the adapter itself (or the provider's API) and are
    the only source the self-model may use when registering capabilities.
    """

    name: str
    streaming: bool = False
    json_mode: bool = False
    max_context_tokens: int | None = None
    notes: str = ""


class Provider(abc.ABC):
    """Abstract inference provider."""

    name: str

    @abc.abstractmethod
    def complete(self, prompt: str, params: dict | None = None) -> Completion:
        """Generate a completion for ``prompt``.

        Raises:
            ProviderError: On transport failures, HTTP errors, or malformed
                responses. ``retryable=True`` marks errors worth retrying.
        """

    @abc.abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Return this provider's self-reported capabilities."""

    def health_check(self) -> bool:
        """Return True if a tiny probe completion succeeds, else False.

        Any exception (including :class:`ProviderError`) becomes ``False``;
        this method never raises.
        """
        try:
            self.complete("ok", {"max_tokens": 1})
            return True
        except Exception:
            return False


def with_retries(
    fn: Callable[[], Completion],
    *,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    retryable: Callable[[Exception], bool] = lambda e: isinstance(e, ProviderError) and e.retryable,
    seed: int | None = None,
) -> Completion:
    """Call ``fn`` with exponential backoff and jitter.

    Args:
        fn: Zero-argument thunk returning a :class:`Completion`.
        max_attempts: Total attempts including the first (must be >= 1).
        base_delay: Delay before the first retry in seconds; doubles per retry.
        retryable: Predicate deciding whether an exception deserves a retry.
        seed: Optional seed for the jitter RNG (deterministic tests).

    Returns:
        The first successful result.

    Raises:
        LabError: If ``max_attempts < 1``.
        Exception: The last error, when attempts are exhausted or the error
            is not retryable.
    """
    if max_attempts < 1:
        raise LabError("with_retries requires max_attempts >= 1")
    rng = random.Random(seed)
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - predicate decides
            last_error = exc
            if attempt >= max_attempts or not retryable(exc):
                raise
            delay = base_delay * (2 ** (attempt - 1)) + rng.uniform(0, base_delay)
            logger.debug("retrying after %.2fs (attempt %d/%d): %s", delay, attempt, max_attempts, exc)
            time.sleep(delay)
    assert last_error is not None  # for type checkers; unreachable
    raise last_error


def _http_post_json(
    url: str, payload: dict, headers: dict[str, str], timeout_s: float
) -> tuple[int, dict]:
    """POST a JSON body with stdlib urllib; return ``(status, parsed_json)``.

    Raises:
        ProviderError: ``retryable=True`` for HTTP 429/5xx; ``retryable=False``
            for auth failures, other HTTP errors, transport failures, and
            malformed JSON. Never includes credentials or bodies in the
            message — only status codes and truncated server text.
    """
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status = int(response.status)
            raw = response.read()
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", "replace")[:300]
        except Exception:
            detail = ""
        retryable = exc.code == 429 or 500 <= exc.code < 600
        raise ProviderError(f"HTTP {exc.code} from provider: {detail}", retryable=retryable) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ProviderError(f"provider request failed: {exc}", retryable=False) from exc
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ProviderError(f"provider returned malformed JSON: {exc}", retryable=False) from exc
    if not isinstance(parsed, dict):
        raise ProviderError("provider returned a JSON value that is not an object", retryable=False)
    return status, parsed
