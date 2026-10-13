"""Metacognition package: behavioral heuristics for confidence, contradiction,
error detection, and belief revision.

These are *heuristic computations over the system's own outputs and records*.
They are NOT measures of consciousness, self-awareness, or understanding —
each module's docstring documents what the heuristic does and where it breaks.
"""

from consciousness_lab.metacognition.belief_revision import BeliefRevision, RevisedBelief
from consciousness_lab.metacognition.confidence import Confidence, ConfidenceEstimator
from consciousness_lab.metacognition.contradiction import Contradiction, ContradictionDetector
from consciousness_lab.metacognition.error_detection import ErrorDetector, ErrorReport

__all__ = [
    "BeliefRevision",
    "Confidence",
    "ConfidenceEstimator",
    "Contradiction",
    "ContradictionDetector",
    "ErrorDetector",
    "ErrorReport",
    "RevisedBelief",
]
