"""Episodic memory: timestamped event records persisted in SQLite.

Retrieval ranking uses the heuristic ``recall_score`` from
``memory.records`` (keyword overlap * 0.5 + recency * 0.3 + confidence *
0.2) — documented as a surface-token heuristic, NOT semantic search.

Deletion requires a non-empty reason string, and an optional ``on_delete``
callback receives ``(record_id, reason)`` for audit logging. Deleting a
record that does not exist raises ``LabError``.
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

EPISODIC_DDL = """CREATE TABLE IF NOT EXISTS episodic (
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

EPISODIC_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_episodic_namespace ON episodic(namespace)",
    "CREATE INDEX IF NOT EXISTS idx_episodic_timestamp ON episodic(timestamp)",
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


class EpisodicMemory:
    """Persistent store of timestamped episodic records, namespaced."""

    def __init__(
        self,
        store: SqliteStore,
        *,
        namespace: str = "default",
        on_delete: Callable[[str, str], None] | None = None,
    ) -> None:
        """Create over ``store``; ensures the ``episodic`` table on init.

        ``on_delete`` is an optional audit callback invoked as
        ``on_delete(record_id, reason)`` after a successful delete.
        """
        self._store = store
        self._namespace = namespace
        self._on_delete = on_delete
        store.ensure_table("episodic", EPISODIC_DDL)
        for index_sql in EPISODIC_INDEXES:
            store.execute_write(index_sql)

    def record(
        self,
        content: str,
        *,
        provenance: str,
        confidence: float = 0.5,
        source_type: SourceType = "inferred",
        metadata: dict | None = None,
    ) -> MemoryRecord:
        """Persist one episodic record; returns the validated record.

        Raises ValueError if the record fails validation (bad confidence,
        bad source_type, empty content).
        """
        rec = MemoryRecord(
            id=uuid.uuid4().hex,
            kind="episodic",
            content=content,
            provenance=provenance,
            confidence=confidence,
            timestamp=time.time(),
            source_type=source_type,
            metadata=dict(metadata or {}),
        )
        rec.validate()
        self._store.insert("episodic", record_to_row(rec, self._namespace))
        return rec

    def recall(self, query: str, k: int = 5) -> list[MemoryRecord]:
        """Return up to ``k`` records ranked by the heuristic recall score.

        Scoring is keyword overlap * 0.5 + recency * 0.3 + confidence * 0.2
        (see ``memory.records.recall_score``). Not semantic search.
        Raises ValueError if k < 1.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k!r}")
        rows = self._store.execute_read(
            "SELECT * FROM episodic WHERE namespace = ?", (self._namespace,)
        )
        now = time.time()
        records = [record_from_row(row) for row in rows]
        ranked = sorted(
            records, key=lambda r: recall_score(query, r, now), reverse=True
        )
        return ranked[:k]

    def get(self, record_id: str) -> MemoryRecord | None:
        """Return the record with ``record_id``, or None if absent."""
        rows = self._store.execute_read(
            "SELECT * FROM episodic WHERE id = ? AND namespace = ?",
            (record_id, self._namespace),
        )
        if not rows:
            return None
        return record_from_row(rows[0])

    def delete(self, record_id: str, reason: str) -> None:
        """Delete a record; ``reason`` is required (non-empty).

        Raises ValueError if ``reason`` is empty, LabError if the record does
        not exist. On success the ``on_delete`` audit callback (if any) is
        invoked with ``(record_id, reason)``.
        """
        if not reason or not reason.strip():
            raise ValueError("delete() requires a non-empty reason")
        if self.get(record_id) is None:
            raise LabError(f"episodic record not found: {record_id!r}")
        self._store.execute_write(
            "DELETE FROM episodic WHERE id = ? AND namespace = ?",
            (record_id, self._namespace),
        )
        _get_log().info(
            "episodic record deleted id=%s reason=%s",
            record_id,
            reason.strip(),
        )
        if self._on_delete is not None:
            self._on_delete(record_id, reason)
