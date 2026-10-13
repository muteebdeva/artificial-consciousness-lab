"""Perception: normalize raw external input into sanitized Observations.

What it does: converts ``str`` or ``dict`` input into an :class:`Observation`
with a validated modality, sanitized content, and metadata.

Limits:
- Text-only. There is no image/audio handling; "structured" modality is a
  plain text payload with a different label, not a parsed schema.
- Sanitization is lossy by design: control characters are stripped and
  content is truncated at ``MAX_CONTENT_CHARS`` (the truncation is flagged
  in ``metadata["truncated"]``).
- Modality is validated against a fixed allowlist; anything else raises
  ``ValueError``. This module never invents content.
"""

from __future__ import annotations

import logging
import time
import unicodedata
import uuid
from dataclasses import dataclass, field

try:  # core.logging_setup is owned by another agent; fall back until it lands
    from consciousness_lab.core.logging_setup import setup_logging

    _log = setup_logging()
except Exception:  # pragma: no cover - import fallback
    _log = logging.getLogger(__name__)

#: Modalities this module accepts. Anything else -> ValueError.
MODALITIES = ("text", "task", "structured")

#: Maximum characters kept in Observation.content after sanitization.
MAX_CONTENT_CHARS = 20_000

#: Characters that are never stripped (newline, tab).
_KEPT_CONTROLS = {"\n", "\t"}


def _strip_control_chars(text: str) -> str:
    """Remove Unicode control/surrogate characters, keeping \\n and \\t.

    Keeps characters whose ``unicodedata.category`` is "Cc" (control) or
    "Cs" (surrogate) out of the result, except newline and tab. This is a
    hygiene step, not a security boundary: it does not neutralize prompt
    injection, markup, or encoded payloads.
    """
    return "".join(
        ch
        for ch in text
        if ch in _KEPT_CONTROLS
        or unicodedata.category(ch) not in ("Cc", "Cs")
    )


def sanitize_text(text: str) -> tuple[str, bool]:
    """Sanitize free text; returns ``(sanitized, was_truncated)``.

    Strips control characters (except ``\\n`` and ``\\t``) and caps the
    result at :data:`MAX_CONTENT_CHARS` characters. Truncation is silent
    w.r.t. content but reported via the returned flag so callers can record
    it in metadata.
    """
    cleaned = _strip_control_chars(text)
    if len(cleaned) > MAX_CONTENT_CHARS:
        return cleaned[:MAX_CONTENT_CHARS], True
    return cleaned, False


@dataclass
class Observation:
    """A single normalized unit of perceived input."""

    id: str
    modality: str  # "text" | "task" | "structured"
    content: str
    timestamp: float
    metadata: dict = field(default_factory=dict)
    source: str = "external"


class Perception:
    """Ingest raw input into sanitized :class:`Observation` objects.

    Stateless apart from logging: no memory, no learning, no side effects
    beyond producing the Observation.
    """

    def ingest(self, raw: str | dict) -> Observation:
        """Normalize ``raw`` into an :class:`Observation`.

        - ``str`` input becomes modality ``"text"``.
        - ``dict`` input accepts keys ``modality``, ``content``,
          ``metadata``, ``source`` (all optional except ``content``;
          defaults: modality ``"text"``, metadata ``{}``,
          source ``"external"``).
        - Content is sanitized (control chars stripped, capped at
          :data:`MAX_CONTENT_CHARS`); truncation sets
          ``metadata["truncated"] = True``.

        Raises:
            TypeError: if ``raw`` is neither ``str`` nor ``dict``, or the
                dict's ``content`` is not a string.
            ValueError: if the modality is not in :data:`MODALITIES`.
        """
        if isinstance(raw, str):
            modality, content, metadata, source = "text", raw, {}, "external"
        elif isinstance(raw, dict):
            modality = raw.get("modality", "text")
            content = raw.get("content", "")
            metadata = dict(raw.get("metadata") or {})
            source = raw.get("source", "external")
            if not isinstance(content, str):
                raise TypeError(
                    f"dict observation 'content' must be str, got {type(content).__name__}"
                )
            if not isinstance(metadata, dict):
                raise TypeError(
                    f"dict observation 'metadata' must be dict, got {type(metadata).__name__}"
                )
        else:
            raise TypeError(
                f"raw observation must be str or dict, got {type(raw).__name__}"
            )

        if modality not in MODALITIES:
            raise ValueError(
                f"unknown modality {modality!r}; expected one of {MODALITIES}"
            )

        cleaned, truncated = sanitize_text(content)
        if truncated:
            metadata["truncated"] = True
            _log.warning(
                "perception truncated observation content",
                extra={"original_chars": len(content), "kept_chars": MAX_CONTENT_CHARS},
            )

        return Observation(
            id=uuid.uuid4().hex,
            modality=modality,
            content=cleaned,
            timestamp=time.time(),
            metadata=metadata,
            source=source,
        )
