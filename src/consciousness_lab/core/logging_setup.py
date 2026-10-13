"""Structured single-line logging for the lab.

Every record is one line of ``key=value`` pairs::

    ts=2026-10-13T11:00:00+0000 level=INFO logger=consciousness_lab msg=built loop provider=mock

Values passed via the ``extra`` dict of a logging call are appended as
``k=v`` pairs, sorted by key. Any extra whose key contains (case-insensitive)
``key``, ``token``, ``secret``, ``password`` or ``api_key`` has its value
replaced with ``***REDACTED***`` before formatting.

What it does:
    - :func:`setup_logging` attaches a single stderr handler with the
      key=value formatter to the named logger and returns it.

Limits:
    - Redaction is key-based only. It does not scan the free-text message for
      secrets, so callers must still avoid interpolating secrets into messages.
    - Only one handler is ever attached; repeated calls are idempotent and
      will not duplicate output.
    - Timestamps use the local system clock rendered in ISO-8601; clock skew
      is not corrected.
"""

from __future__ import annotations

import logging
import sys

from .errors import ConfigError

__all__ = ["setup_logging"]

_REDACT_SUBSTRINGS = ("key", "token", "secret", "password", "api_key")
_REDACTED = "***REDACTED***"


def _standard_record_attrs() -> frozenset[str]:
    """Attribute names present on every LogRecord (excluded from k=v output)."""
    probe = logging.LogRecord("probe", logging.INFO, __file__, 1, "msg", (), None)
    return frozenset(probe.__dict__)


_STANDARD_ATTRS = _standard_record_attrs()


def _redact(key: str, value: object) -> object:
    lowered = key.lower()
    if any(part in lowered for part in _REDACT_SUBSTRINGS):
        return _REDACTED
    return value


class _KeyValueFormatter(logging.Formatter):
    """Format a record as a single line of ``key=value`` pairs."""

    def format(self, record: logging.LogRecord) -> str:
        parts = [
            f"ts={self.formatTime(record, '%Y-%m-%dT%H:%M:%S%z')}",
            f"level={record.levelname}",
            f"logger={record.name}",
            f"msg={record.getMessage()}",
        ]
        for attr in sorted(record.__dict__):
            if attr in _STANDARD_ATTRS:
                continue
            parts.append(f"{attr}={_redact(attr, record.__dict__[attr])}")
        formatted = " ".join(parts)
        if record.exc_info:
            formatted += f" exc={self.formatException(record.exc_info).replace(chr(10), ' | ')}"
        return formatted


def setup_logging(level: str = "INFO", *, name: str = "consciousness_lab") -> logging.Logger:
    """Configure and return the lab logger.

    Args:
        level: Logging level name (e.g. ``"INFO"``, ``"DEBUG"``).
        name: Logger name; child loggers (``consciousness_lab.core.loop``,
            etc.) inherit the handler through propagation.

    Returns:
        The configured :class:`logging.Logger`.

    Raises:
        ConfigError: If ``level`` is not a known logging level name.
    """
    numeric = getattr(logging, str(level).upper(), None)
    if not isinstance(numeric, int):
        raise ConfigError(f"unknown log level: {level!r}")
    logger = logging.getLogger(name)
    logger.setLevel(numeric)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_KeyValueFormatter())
        logger.addHandler(handler)
        # Avoid double-emission when the root logger is also configured.
        logger.propagate = False
    return logger
