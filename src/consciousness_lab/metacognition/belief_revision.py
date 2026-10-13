"""Belief revision: update a stored belief's confidence in light of new evidence.

What it does
------------
``revise`` computes a new confidence as a weighted average of the old
confidence (weight 1) and the evidence confidence, then appends a NEW record
version with ``supersedes`` pointing at the old record id via the
``store_update`` callback. The old record is never mutated.

What it is NOT
--------------
Bayesian updating, learning, or understanding. Treating confidences as
probabilities and averaging them is a *chosen heuristic* (weights w_old=1,
w_ev=evidence_confidence), not a posterior. "Belief" here means "a stored
record with a confidence number" — nothing more. The provenance chain is a
text trail for audit, not evidence of reasoning.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # Avoid import cycles; memory.records imports nothing from here.
    from consciousness_lab.memory.records import MemoryRecord

log = logging.getLogger(__name__)

#: Weight given to the existing belief in the weighted-average update.
OLD_BELIEF_WEIGHT = 1.0


@dataclass
class RevisedBelief:
    """Result of revising a belief: the new version's identity and trail."""

    record_id: str  # id of the NEW record version
    old_confidence: float
    new_confidence: float
    evidence: str
    provenance_chain: list[str]  # ordered provenance strings, oldest first
    timestamp: float = field(default_factory=time.time)


class BeliefRevision:
    """Revise a stored record's confidence given new evidence.

    The revision is append-only: a new record version is created (with
    ``supersedes`` set to the old id) and handed to ``store_update``. The
    original record object is never modified in place.
    """

    def revise(
        self,
        record: "MemoryRecord",
        evidence: str,
        evidence_confidence: float,
        store_update: Callable[["MemoryRecord"], None],
    ) -> RevisedBelief:
        """Compute a revised confidence and append the new record version.

        Parameters
        ----------
        record:
            The existing record (duck-typed: needs ``.id``, ``.confidence``,
            ``.provenance``; ``.kind``, ``.content``, ``.source_type``,
            ``.metadata`` are copied to the new version when present).
        evidence:
            Human-readable description of the new evidence.
        evidence_confidence:
            Confidence in the evidence, 0..1. Also serves as its weight.
        store_update:
            Callback receiving the new record version for persistence. Never
            mutates the old record; if the callback is a no-op, nothing is
            persisted (documented, not hidden).

        Returns
        -------
        RevisedBelief with the new record id, old/new confidences, and the
        ordered provenance chain.
        """
        if not callable(store_update):
            raise TypeError("store_update must be callable")
        if not evidence or not str(evidence).strip():
            raise ValueError("evidence must be non-empty")
        if isinstance(evidence_confidence, bool) or not isinstance(
            evidence_confidence, (int, float)
        ):
            raise TypeError("evidence_confidence must be a number")
        evidence_confidence = float(evidence_confidence)
        if not 0.0 <= evidence_confidence <= 1.0:
            raise ValueError("evidence_confidence must be within 0..1")

        old_confidence = getattr(record, "confidence", None)
        if isinstance(old_confidence, bool) or not isinstance(
            old_confidence, (int, float)
        ):
            raise TypeError("record.confidence must be a number")
        old_confidence = float(old_confidence)

        # Weighted-average update (documented heuristic, not Bayes' rule).
        w_old = OLD_BELIEF_WEIGHT
        w_ev = evidence_confidence
        new_confidence = (old_confidence * w_old + evidence_confidence * w_ev) / (
            w_old + w_ev
        )

        new_id = uuid.uuid4().hex
        old_provenance = str(getattr(record, "provenance", ""))
        new_provenance = f"{old_provenance} | revised on evidence: {evidence}"
        now = time.time()

        new_record = self._new_version(
            record,
            new_id=new_id,
            confidence=new_confidence,
            provenance=new_provenance,
            timestamp=now,
        )
        store_update(new_record)

        return RevisedBelief(
            record_id=new_id,
            old_confidence=old_confidence,
            new_confidence=new_confidence,
            evidence=str(evidence),
            provenance_chain=[old_provenance, new_provenance],
            timestamp=now,
        )

    @staticmethod
    def _new_version(
        record: "MemoryRecord",
        *,
        new_id: str,
        confidence: float,
        provenance: str,
        timestamp: float,
    ) -> "MemoryRecord":
        """Build the new record version, preferring the real MemoryRecord type.

        Function-local import avoids import cycles. If the memory package is
        not yet available in this build (parallel development), falls back to
        ``dataclasses.replace`` on the record's own type — preserving the
        append-new-version-with-``supersedes`` semantics either way.
        """
        try:
            from consciousness_lab.memory.records import MemoryRecord
        except ImportError:  # memory package not present in this build
            MemoryRecord = None  # type: ignore[assignment]

        if MemoryRecord is not None:
            return MemoryRecord(
                id=new_id,
                kind=getattr(record, "kind", "semantic"),
                content=getattr(record, "content", ""),
                provenance=provenance,
                confidence=confidence,
                timestamp=timestamp,
                source_type=getattr(record, "source_type", "inferred"),
                supersedes=getattr(record, "id", None),
                metadata=dict(getattr(record, "metadata", {}) or {}),
            )
        import dataclasses

        return dataclasses.replace(
            record,
            id=new_id,
            confidence=confidence,
            provenance=provenance,
            timestamp=timestamp,
            supersedes=getattr(record, "id", None),
        )
