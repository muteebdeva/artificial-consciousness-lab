"""Memory record primitives and shared retrieval helpers.

``MemoryRecord`` is the unit stored by episodic and semantic memory.
``recall_score`` implements the heuristic retrieval ranking shared by both
stores: keyword overlap * 0.5 + recency * 0.3 + confidence * 0.2. This is a
documented heuristic, NOT semantic search: it matches surface tokens only and
has no notion of meaning, paraphrase, or entailment.

``record_to_row`` / ``record_from_row`` map records to the SQLite column
layout used by the stores (``metadata`` is stored as a JSON string in the
``metadata_json`` column), so records round-trip losslessly through
``to_dict`` / ``from_dict``.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Literal

SourceType = Literal["verified", "user_provided", "model_hypothesis", "inferred"]

VALID_SOURCE_TYPES = ("verified", "user_provided", "model_hypothesis", "inferred")
VALID_KINDS = ("episodic", "semantic", "outcome", "working")


@dataclass
class MemoryRecord:
    """A single persisted memory entry.

    ``source_type`` is restricted to verified | user_provided |
    model_hypothesis | inferred. ``supersedes`` holds the id of a record this
    one explicitly replaces (set only via an explicit supersedes parameter,
    never implicitly). ``metadata`` carries arbitrary extra fields.
    """

    id: str
    kind: str  # episodic | semantic | outcome | working
    content: str
    provenance: str  # free text: where it came from
    confidence: float  # 0..1
    timestamp: float
    source_type: SourceType
    supersedes: str | None = None
    metadata: dict = field(default_factory=dict)

    def validate(self) -> None:
        """Check field invariants; raise ValueError on the first violation.

        Checks: kind is known, content is non-empty, confidence is a number
        in [0, 1], source_type is one of the four allowed values, timestamp
        is a finite number.
        """
        if self.kind not in VALID_KINDS:
            raise ValueError(
                f"kind must be one of {VALID_KINDS}, got {self.kind!r}"
            )
        if not self.content or not self.content.strip():
            raise ValueError("content must be non-empty")
        if (
            not isinstance(self.confidence, (int, float))
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence!r}"
            )
        if self.source_type not in VALID_SOURCE_TYPES:
            raise ValueError(
                f"source_type must be one of {VALID_SOURCE_TYPES}, "
                f"got {self.source_type!r}"
            )
        ts = self.timestamp
        if not isinstance(ts, (int, float)) or ts != ts or ts in (float("inf"), float("-inf")):
            raise ValueError(f"timestamp must be a finite number, got {ts!r}")

    def to_dict(self) -> dict:
        """Serialize to a plain dict; ``metadata`` stays a dict here."""
        return {
            "id": self.id,
            "kind": self.kind,
            "content": self.content,
            "provenance": self.provenance,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "source_type": self.source_type,
            "supersedes": self.supersedes,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryRecord":
        """Rebuild from a dict; ``metadata`` may be a dict or a JSON string."""
        metadata = d.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata) if metadata else {}
        return cls(
            id=d["id"],
            kind=d["kind"],
            content=d["content"],
            provenance=d.get("provenance", ""),
            confidence=d["confidence"],
            timestamp=d["timestamp"],
            source_type=d["source_type"],
            supersedes=d.get("supersedes"),
            metadata=dict(metadata),
        )


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set[str]:
    """Lowercase alphanumeric token set of ``text`` (for overlap scoring)."""
    return set(_TOKEN_RE.findall(text.lower()))


def recall_score(
    query: str, record: MemoryRecord, now: float | None = None
) -> float:
    """Heuristic retrieval score in [0, 1]-ish range.

    score = keyword_overlap * 0.5 + recency * 0.3 + confidence * 0.2, where
    keyword_overlap is the fraction of query tokens present in the record
    content and recency is 1 / (1 + age_hours). Documented heuristic, not
    semantic search: it cannot match paraphrases or infer meaning, and
    recency decays on a fixed 1-hour half-ish curve regardless of content.
    """
    now = time.time() if now is None else now
    query_tokens = tokenize(query)
    content_tokens = tokenize(record.content)
    overlap = (
        len(query_tokens & content_tokens) / len(query_tokens)
        if query_tokens
        else 0.0
    )
    age_hours = max(0.0, (now - record.timestamp)) / 3600.0
    recency = 1.0 / (1.0 + age_hours)
    return overlap * 0.5 + recency * 0.3 + record.confidence * 0.2


def record_to_row(record: MemoryRecord, namespace: str) -> dict:
    """Map a record to SQLite insert columns (metadata as JSON string)."""
    return {
        "id": record.id,
        "kind": record.kind,
        "content": record.content,
        "provenance": record.provenance,
        "confidence": record.confidence,
        "timestamp": record.timestamp,
        "source_type": record.source_type,
        "supersedes": record.supersedes,
        "metadata_json": json.dumps(record.metadata),
        "namespace": namespace,
    }


def record_from_row(row: dict) -> MemoryRecord:
    """Rebuild a record from a SQLite row dict (metadata_json parsed)."""
    d = dict(row)
    metadata_json = d.pop("metadata_json", "{}")
    metadata = json.loads(metadata_json) if metadata_json else {}
    d.pop("namespace", None)
    d["metadata"] = metadata
    return MemoryRecord.from_dict(d)
