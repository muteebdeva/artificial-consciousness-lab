"""Error detection: score predictions against observed outcomes and scan traces.

What it does
------------
``evaluate_prediction`` turns a (prediction, observed outcome) pair into an
:class:`ErrorReport` with a Brier contribution and a miscalibration flag
(``|p - o| > 0.5``). ``scan_trace`` returns the failed entries of a
stage-trace list.

What it is NOT
--------------
A judgment of reasoning quality. ``correct`` means only "the predicted
outcome (p >= 0.5) matched the observation" — a lucky guess scores as
correct. Miscalibration flags badly-scaled confidences, not bad arguments.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # Avoid import cycles; memory.outcome imports nothing from here.
    from consciousness_lab.memory.outcome import Prediction

log = logging.getLogger(__name__)

#: A prediction is flagged miscalibrated when |confidence - outcome| exceeds this.
MISCALIBRATION_THRESHOLD = 0.5


@dataclass
class ErrorReport:
    """Outcome of scoring one prediction against what actually happened."""

    prediction_id: str
    predicted_confidence: float
    observed: bool
    correct: bool  # predicted outcome (p >= 0.5) matched the observation
    brier_contribution: float  # (p - o)^2 for this prediction
    miscalibrated: bool  # |p - o| > 0.5
    timestamp: float = field(default_factory=time.time)


class ErrorDetector:
    """Score predictions against outcomes; scan stage traces for failures."""

    def evaluate_prediction(self, prediction: "Prediction", observed: bool) -> ErrorReport:
        """Score ``prediction`` against the observed outcome.

        ``prediction`` is duck-typed (requires ``.id`` and ``.confidence``) so
        this module does not import the memory package at module level. A
        function-local import of ``consciousness_lab.memory.outcome.Prediction``
        is attempted for an explicit type check; when the memory package is
        not yet available (parallel development) the check is skipped and
        attributes are validated directly instead.
        """
        try:
            from consciousness_lab.memory.outcome import Prediction as _Prediction
        except ImportError:  # memory module not present in this build
            _Prediction = None  # type: ignore[assignment]

        if _Prediction is not None and not isinstance(prediction, _Prediction):
            raise TypeError(
                f"prediction must be a consciousness_lab.memory.outcome.Prediction, "
                f"got {type(prediction).__name__}"
            )

        prediction_id = str(getattr(prediction, "id", "unknown"))
        confidence = getattr(prediction, "confidence", None)
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise TypeError("prediction.confidence must be a number")
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("prediction.confidence must be within 0..1")

        observed_bool = bool(observed)
        outcome_value = float(observed_bool)
        correct = (confidence >= 0.5) == observed_bool
        brier_contribution = (confidence - outcome_value) ** 2
        miscalibrated = abs(confidence - outcome_value) > MISCALIBRATION_THRESHOLD

        return ErrorReport(
            prediction_id=prediction_id,
            predicted_confidence=confidence,
            observed=observed_bool,
            correct=correct,
            brier_contribution=brier_contribution,
            miscalibrated=miscalibrated,
            timestamp=time.time(),
        )

    def scan_trace(self, trace: list[dict]) -> list[dict]:
        """Return the failed stages of a trace.

        Each entry is expected to be ``{"stage": str, "ok": bool, "detail": str}``;
        entries with falsy ``ok`` are returned (as dict copies). Entries that
        are not dicts, or lack ``ok``, are treated as failed (missing the
        success signal is itself suspicious) — documented here, not silently
        swallowed.
        """
        failed: list[dict] = []
        for entry in trace:
            if not isinstance(entry, dict):
                failed.append({"stage": "unknown", "ok": False, "detail": f"malformed trace entry: {entry!r}"})
                continue
            if not entry.get("ok", False):
                failed.append(dict(entry))
        return failed


__all__ = ["ErrorDetector", "ErrorReport", "MISCALIBRATION_THRESHOLD"]
