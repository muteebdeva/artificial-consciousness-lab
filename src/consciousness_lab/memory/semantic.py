"""Semantic memory: persistent fact records with contradiction handling.

``upsert_fact`` never silently overwrites: when the new fact conflicts with
existing facts (per ``metacognition.ContradictionDetector``), BOTH records
are kept, the conflicts are returned, and the optional ``on_contradiction``
callback is invoked. ``supersedes`` is set ONLY when the caller passes an
explicit ``supersedes_id``.

The import of ``ContradictionDetector`` is function-local (deferred) inside
``upsert_fact`` to avoid an import cycle between the memory and metacognition
packages. If the metacognition module is unavailable, the ImportError
propagates to the caller rather than being silently swallowed.

Retrieval uses the same heuristic ``recall_score`` as episodic memory
(keyword overlap * 0.5 + recency * 0.3 + confidence * 0.2) — not semantic
search.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Callable

from consciousness_lab.core.errors import LabError
from consciousness_lab.memory.records import (
    MemoryRecord,
    SourceType,
    recall_score,
    record_from_row,
    record_to_row,
)
from consciousness_lab.persistence.sqlite_store import SqliteStore

SEMANTIC_DDL = """CREATE TABLE IF NOT EXISTS semantic (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    provenance TEXT NOT NULL,
    confidence REAL NOT NULL,
    timestamp REAL NOT NULL,
    source_type TEXT NOT NULL,
    supersedes TEXT,
    metadata_json TEXT NOT NULL,
    namespace TEXT NOT NULL
)"""

SEMANTIC_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_semantic_namespace ON semantic(namespace)",
    "CREATE INDEX IF NOT EXISTS idx_semantic_timestamp ON semantic(timestamp)",
)

_log: logging.Logger | None = None


def _get_log() -> logging.Logger:
    """Lab logger if core.logging_setup exists, else a plain stdlib logger."""
    global _log
    if _log is None:
        try:
            from consciousness_lab.core.logging_setup import setup_logging

            _log = setup_logging()
        except Exception:
            _log = logging.getLogger(__name__)
    return _log


class SemanticMemory:
    """Persistent store of fact records with contradiction detection."""

    def __init__(
        self,
        store: SqliteStore,
        *,
        namespace: str = "default",
        on_contradiction: Callable[[object], None] | None = None,
        on_delete: Callable[[str, str], None] | None = None,
        contradiction_detector: object | None | bool = None,
    ) -> None:
        """Create over ``store``; ensures the ``semantic`` table on init.

        ``on_contradiction`` is invoked once per detected ``Contradiction``.
        ``on_delete`` is an optional audit callback invoked as
        ``on_delete(record_id, reason)`` after a successful delete. Callback
        exceptions are caught and logged so they cannot break the store.

        ``contradiction_detector`` controls upsert-time contradiction
        detection: ``None`` (default) constructs the standard
        ``metacognition.ContradictionDetector``; pass an explicit detector
        instance to share one (e.g. the loop's); pass ``False`` to disable
        upsert-time detection entirely (used by the ``metacognition``
        ablation so disabling metacognition is honest and complete).
        """
        self._store = store
        self._namespace = namespace
        self._on_contradiction = on_contradiction
        self._on_delete = on_delete
        self._detector_arg = contradiction_detector
        store.ensure_table("semantic", SEMANTIC_DDL)
        for index_sql in SEMANTIC_INDEXES:
            store.execute_write(index_sql)

    def _detector(self):  # type: ignore[no-untyped-def]
        """Resolve the contradiction detector per constructor config."""
        if self._detector_arg is False:
            return None
        if self._detector_arg is not None:
            return self._detector_arg
        # Deferred import: avoids a memory <-> metacognition import cycle.
        from consciousness_lab.metacognition.contradiction import (
            ContradictionDetector,
        )

        return ContradictionDetector()

    def upsert_fact(
        self,
        content: str,
        *,
        provenance: str,
        confidence: float = 0.5,
        source_type: SourceType = "inferred",
        supersedes_id: str | None = None,
    ) -> tuple[MemoryRecord, list]:
        """Insert a fact, checking existing facts for contradictions.

        Returns ``(record, contradictions)``. On conflict both the old and
        new records are kept; ``record.supersedes`` is set only from the
        explicit ``supersedes_id`` parameter. Each contradiction is passed
        to ``on_contradiction`` (if set); callback failures are logged, not
        raised.

        Raises ValueError if the record fails validation. Raises ImportError
        if ``consciousness_lab.metacognition.contradiction`` is unavailable
        (unless detection was disabled via the constructor).
        """
        rec = MemoryRecord(
            id=uuid.uuid4().hex,
            kind="semantic",
            content=content,
            provenance=provenance,
            confidence=confidence,
            timestamp=time.time(),
            source_type=source_type,
            supersedes=supersedes_id,
            metadata={},
        )
        rec.validate()
        existing = self._all_records()
        detector = self._detector()
        contradictions = (
            detector.check(content, existing) if detector is not None else []
        )
        # The detector compares the new (not yet stored) content against
        # existing records, so it leaves record_a_id unset and fills
        # record_b_id with the existing record's id. Bind the new record's
        # id as side A; never overwrite an id the detector already set.
        for contradiction in contradictions:
            if contradiction.record_a_id is None:
                contradiction.record_a_id = rec.id
        self._store.insert("semantic", record_to_row(rec, self._namespace))
        if self._on_contradiction is not None:
            for contradiction in contradictions:
                try:
                    self._on_contradiction(contradiction)
                except Exception:
                    _get_log().exception(
                        "on_contradiction callback raised; continuing"
                    )
        return rec, list(contradictions)

    def query(self, q: str, k: int = 5) -> list[MemoryRecord]:
        """Return up to ``k`` facts ranked by the heuristic recall score.

        Not semantic search; see ``memory.records.recall_score``. Raises
        ValueError if k < 1.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k!r}")
        now = time.time()
        ranked = sorted(
            self._all_records(),
            key=lambda r: recall_score(q, r, now),
            reverse=True,
        )
        return ranked[:k]

    def get(self, record_id: str) -> MemoryRecord | None:
        """Return the fact with ``record_id``, or None if absent."""
        rows = self._store.execute_read(
            "SELECT * FROM semantic WHERE id = ? AND namespace = ?",
            (record_id, self._namespace),
        )
        if not rows:
            return None
        return record_from_row(rows[0])

    def delete(self, record_id: str, reason: str) -> None:
        """Delete a fact; ``reason`` is required (non-empty).

        Raises ValueError if ``reason`` is empty, LabError if the fact does
        not exist. On success the ``on_delete`` audit callback (if any) is
        invoked with ``(record_id, reason)``.
        """
        if not reason or not reason.strip():
            raise ValueError("delete() requires a non-empty reason")
        if self.get(record_id) is None:
            raise LabError(f"semantic record not found: {record_id!r}")
        self._store.execute_write(
            "DELETE FROM semantic WHERE id = ? AND namespace = ?",
            (record_id, self._namespace),
        )
        _get_log().info(
            "semantic record deleted id=%s reason=%s",
            record_id,
            reason.strip(),
        )
        if self._on_delete is not None:
            self._on_delete(record_id, reason)

    def _all_records(self) -> list[MemoryRecord]:
        rows = self._store.execute_read(
            "SELECT * FROM semantic WHERE namespace = ?", (self._namespace,)
        )
        return [record_from_row(row) for row in rows]
