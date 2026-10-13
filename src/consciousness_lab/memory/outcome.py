"""Outcome memory: predictions, observed outcomes, and calibration.

``OutcomeMemory`` records probabilistic predictions and later scores them
against observed boolean outcomes. ``calibration()`` reports the Brier score
(mean squared error of predicted probabilities) and expected calibration
error (ECE) over 10 equal-width confidence bins.

Limits:
- Brier/ECE measure calibration of stated confidences, not correctness of
  the underlying claims, and say nothing about whether the predictions were
  well-posed.
- An empty calibration (no scored predictions) returns NaN scores rather
  than 0 — NaN means "unknown", which must not be read as "perfect".
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field

from consciousness_lab.core.errors import LabError
from consciousness_lab.persistence.sqlite_store import SqliteStore

OUTCOME_DDL = """CREATE TABLE IF NOT EXISTS outcome_predictions (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    confidence REAL NOT NULL,
    timestamp REAL NOT NULL,
    context_json TEXT NOT NULL,
    outcome INTEGER,
    outcome_timestamp REAL,
    outcome_note TEXT NOT NULL DEFAULT '',
    namespace TEXT NOT NULL
)"""

OUTCOME_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_outcome_namespace "
    "ON outcome_predictions(namespace)",
    "CREATE INDEX IF NOT EXISTS idx_outcome_pending "
    "ON outcome_predictions(namespace, outcome)",
)


@dataclass
class Prediction:
    """One probabilistic prediction, optionally scored against an outcome."""

    id: str
    content: str
    confidence: float  # 0..1 predicted probability
    timestamp: float
    context: dict = field(default_factory=dict)
    outcome: bool | None = None
    outcome_timestamp: float | None = None
    outcome_note: str = ""

    def to_dict(self) -> dict:
        """Serialize to a plain dict; ``context`` stays a dict here."""
        return {
            "id": self.id,
            "content": self.content,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "context": dict(self.context),
            "outcome": self.outcome,
            "outcome_timestamp": self.outcome_timestamp,
            "outcome_note": self.outcome_note,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Prediction":
        """Rebuild from a dict; ``context`` may be a dict or JSON string."""
        context = d.get("context", {})
        if isinstance(context, str):
            context = json.loads(context) if context else {}
        outcome = d.get("outcome")
        if outcome is not None:
            outcome = bool(outcome)
        return cls(
            id=d["id"],
            content=d["content"],
            confidence=d["confidence"],
            timestamp=d["timestamp"],
            context=dict(context),
            outcome=outcome,
            outcome_timestamp=d.get("outcome_timestamp"),
            outcome_note=d.get("outcome_note", ""),
        )


@dataclass
class CalibrationReport:
    """Calibration summary over scored predictions.

    ``bins`` holds one dict per confidence bin:
    ``{"bin_low", "bin_high", "n", "avg_confidence", "accuracy"}``.
    Empty bins report NaN for avg_confidence/accuracy. When ``n == 0``
    (no scored predictions), ``brier_score`` and ``ece`` are NaN and
    ``bins`` is empty.
    """

    n: int
    brier_score: float
    ece: float
    bins: list[dict]
    generated_at: float


def _prediction_to_row(pred: Prediction, namespace: str) -> dict:
    return {
        "id": pred.id,
        "content": pred.content,
        "confidence": pred.confidence,
        "timestamp": pred.timestamp,
        "context_json": json.dumps(pred.context),
        "outcome": None if pred.outcome is None else int(pred.outcome),
        "outcome_timestamp": pred.outcome_timestamp,
        "outcome_note": pred.outcome_note,
        "namespace": namespace,
    }


def _prediction_from_row(row: dict) -> Prediction:
    d = dict(row)
    context_json = d.pop("context_json", "{}")
    context = json.loads(context_json) if context_json else {}
    outcome = d.pop("outcome", None)
    outcome = None if outcome is None else bool(outcome)
    d.pop("namespace", None)
    d["context"] = context
    d["outcome"] = outcome
    return Prediction.from_dict(d)


class OutcomeMemory:
    """Persistent store of predictions and their observed outcomes."""

    def __init__(self, store: SqliteStore, *, namespace: str = "default") -> None:
        """Create over ``store``; ensures the ``outcome_predictions`` table."""
        self._store = store
        self._namespace = namespace
        store.ensure_table("outcome_predictions", OUTCOME_DDL)
        for index_sql in OUTCOME_INDEXES:
            store.execute_write(index_sql)

    def record_prediction(
        self,
        content: str,
        confidence: float,
        context: dict | None = None,
    ) -> Prediction:
        """Record a new prediction with a stated confidence in [0, 1].

        Raises ValueError on empty content or confidence outside [0, 1].
        """
        if not content or not content.strip():
            raise ValueError("content must be non-empty")
        if (
            not isinstance(confidence, (int, float))
            or not 0.0 <= confidence <= 1.0
        ):
            raise ValueError(
                f"confidence must be in [0, 1], got {confidence!r}"
            )
        pred = Prediction(
            id=uuid.uuid4().hex,
            content=content,
            confidence=float(confidence),
            timestamp=time.time(),
            context=dict(context or {}),
        )
        self._store.insert(
            "outcome_predictions", _prediction_to_row(pred, self._namespace)
        )
        return pred

    def record_outcome(
        self, prediction_id: str, observed: bool, note: str = ""
    ) -> Prediction:
        """Score a prediction against an observed boolean outcome.

        Raises ValueError if ``observed`` is not a bool; raises LabError if
        the prediction id is unknown. Re-recording an outcome overwrites the
        previous one (documented; used for corrections).
        """
        if not isinstance(observed, bool):
            raise ValueError(
                f"observed must be a bool, got {observed!r}"
            )
        rows = self._store.execute_read(
            "SELECT * FROM outcome_predictions WHERE id = ? AND namespace = ?",
            (prediction_id, self._namespace),
        )
        if not rows:
            raise LabError(f"prediction not found: {prediction_id!r}")
        self._store.execute_write(
            "UPDATE outcome_predictions SET outcome = ?, outcome_timestamp = ?, "
            "outcome_note = ? WHERE id = ? AND namespace = ?",
            (
                int(observed),
                time.time(),
                note,
                prediction_id,
                self._namespace,
            ),
        )
        updated = self._store.execute_read(
            "SELECT * FROM outcome_predictions WHERE id = ? AND namespace = ?",
            (prediction_id, self._namespace),
        )
        return _prediction_from_row(updated[0])

    def pending(self) -> list[Prediction]:
        """Return predictions with no recorded outcome, oldest first."""
        rows = self._store.execute_read(
            "SELECT * FROM outcome_predictions WHERE namespace = ? "
            "AND outcome IS NULL ORDER BY timestamp ASC",
            (self._namespace,),
        )
        return [_prediction_from_row(row) for row in rows]

    def calibration(self, n_bins: int = 10) -> CalibrationReport:
        """Compute Brier score and ECE over scored predictions.

        Brier = mean((confidence - outcome)^2). ECE = sum over bins of
        (n_bin / n) * |accuracy_bin - avg_confidence_bin| with 10 equal-width
        bins over [0, 1]. With no scored predictions, returns n=0 and NaN
        scores (documented: NaN means unknown, not perfect).
        """
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins!r}")
        rows = self._store.execute_read(
            "SELECT * FROM outcome_predictions WHERE namespace = ? "
            "AND outcome IS NOT NULL",
            (self._namespace,),
        )
        scored = [_prediction_from_row(row) for row in rows]
        generated_at = time.time()
        if not scored:
            return CalibrationReport(
                n=0,
                brier_score=float("nan"),
                ece=float("nan"),
                bins=[],
                generated_at=generated_at,
            )
        n = len(scored)
        brier = sum(
            (p.confidence - (1.0 if p.outcome else 0.0)) ** 2 for p in scored
        ) / n
        bins: list[dict] = []
        for i in range(n_bins):
            low = i / n_bins
            high = (i + 1) / n_bins
            in_bin = [
                p
                for p in scored
                if min(int(p.confidence * n_bins), n_bins - 1) == i
            ]
            if in_bin:
                avg_conf = sum(p.confidence for p in in_bin) / len(in_bin)
                accuracy = sum(1.0 if p.outcome else 0.0 for p in in_bin) / len(
                    in_bin
                )
            else:
                avg_conf = float("nan")
                accuracy = float("nan")
            bins.append(
                {
                    "bin_low": low,
                    "bin_high": high,
                    "n": len(in_bin),
                    "avg_confidence": avg_conf,
                    "accuracy": accuracy,
                }
            )
        ece = sum(
            (b["n"] / n) * abs(b["accuracy"] - b["avg_confidence"])
            for b in bins
            if b["n"] > 0
        )
        return CalibrationReport(
            n=n,
            brier_score=brier,
            ece=ece,
            bins=bins,
            generated_at=generated_at,
        )
