"""Self-model: the system's model of its own capabilities, limitations, and track record.

What it does
------------
Maintains a registry of capabilities, limitations, tools, the current task,
resources, unresolved questions, and scored performance predictions about the
system's own future behavior. Every mutation records its ``source`` so the
provenance of every self-model claim is auditable.

What it is NOT
-------------
A measure of consciousness, self-awareness, or introspective access. It is a
plain bookkeeping structure. Per design decision DD-06 (docs/design_decisions.md),
self-model fields are updated only from actual system state, tool results, and
recorded outcomes. Capabilities come ONLY from ``register_tool`` /
``register_capability`` calls made by the loop factory with real
tools/providers — never invented strings.

Limits
------
- Stored in memory only; passing a ``store`` is accepted for API compatibility
  but no persistence is performed by this implementation.
- ``update_from_observation`` records counts/summaries; it does not interpret
  or understand observations.
- ``accuracy()`` is a descriptive statistic over the system's own predictions;
  a high value means the system predicted its behavior well, nothing more.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict, dataclass, field

log = logging.getLogger(__name__)

#: Exact source string required by the contract for limitations derived from
#: observed tool failures.
OBSERVED_FAILURE_SOURCE = "observed failure"

#: Source string used when a tool is executed with a recorded outcome.
TOOL_RESULT_SOURCE = "tool result"


@dataclass
class PerformancePrediction:
    """A prediction about the system's own future performance on a task.

    ``actual`` is ``None`` until :meth:`SelfModel.score_prediction` records the
    observed outcome. ``confidence`` is the system's stated probability (0..1)
    that the prediction is correct — a calibration input, not a fact.
    """

    id: str
    task: str
    predicted_success: bool
    confidence: float
    timestamp: float
    actual: bool | None = None


@dataclass
class _Capability:
    name: str
    source: str
    timestamp: float


@dataclass
class _Limitation:
    text: str
    source: str
    timestamp: float


@dataclass
class _ToolInfo:
    name: str
    description: str
    registered_at: float


@dataclass
class _Unresolved:
    question: str
    timestamp: float
    resolved: bool = False
    resolution: str = ""
    resolved_at: float | None = None


class SelfModel:
    """Source-attributed registry of what the system can/can't do and how it did.

    Parameters
    ----------
    store:
        Accepted for API compatibility with the persistence layer. Currently
        unused — all state is held in memory. Documented here so callers do not
        assume durability they do not get.
    """

    def __init__(self, store=None) -> None:
        self._store = store
        self._capabilities: dict[str, _Capability] = {}
        self._limitations: list[_Limitation] = []
        self._tools: dict[str, _ToolInfo] = {}
        self._task: str = ""
        self._objectives: list[str] = []
        self._resources: dict = {}
        self._unresolved: list[_Unresolved] = []
        self._predictions: list[PerformancePrediction] = []
        self._tool_history: list[dict] = []
        self._observation_count: int = 0
        self._last_observation_at: float | None = None

    # ------------------------------------------------------------------
    # Registration (the ONLY sources of capability/limitation entries)
    # ------------------------------------------------------------------

    def register_capability(self, name: str, source: str) -> None:
        """Register a capability, recording where the claim came from.

        Re-registering an existing name updates its source/timestamp (latest
        registration wins); the registry never contains duplicate names.
        """
        if not name or not name.strip():
            raise ValueError("capability name must be non-empty")
        if not source or not source.strip():
            raise ValueError("capability source must be non-empty")
        self._capabilities[name] = _Capability(
            name=name, source=source, timestamp=time.time()
        )
        log.debug("registered capability", extra={"name": name, "source": source})

    def register_limitation(self, text: str, source: str) -> None:
        """Register a limitation, recording where the claim came from."""
        if not text or not text.strip():
            raise ValueError("limitation text must be non-empty")
        if not source or not source.strip():
            raise ValueError("limitation source must be non-empty")
        self._limitations.append(
            _Limitation(text=text, source=source, timestamp=time.time())
        )
        log.debug("registered limitation", extra={"source": source})

    def register_tool(self, name: str, description: str) -> None:
        """Register a tool.

        Tools imply capabilities: registering a tool also registers a
        capability of the same name whose source is the tool registration.
        Per DD-06 the loop factory calls this with real tools only — this
        method never invents capability names on its own.
        """
        if not name or not name.strip():
            raise ValueError("tool name must be non-empty")
        self._tools[name] = _ToolInfo(
            name=name, description=description, registered_at=time.time()
        )
        self.register_capability(name, source=f"tool registration: {name}")

    # ------------------------------------------------------------------
    # Updates from system activity
    # ------------------------------------------------------------------

    def update_from_observation(self, obs: dict) -> None:
        """Record that an observation was processed.

        Records a count and timestamp only; the observation is not interpreted.
        ``obs`` is expected to be a mapping (e.g. from the perception stage)
        but any dict is accepted defensively.
        """
        if not isinstance(obs, dict):
            raise TypeError("obs must be a dict")
        self._observation_count += 1
        self._last_observation_at = time.time()

    def update_from_tool_result(self, result: dict) -> None:
        """Record a tool execution outcome.

        ``result`` must contain ``{"tool": str, "success": bool, "detail": str}``.
        Failures append a limitation entry whose source is exactly
        ``"observed failure"``. Successes are recorded in the tool history but
        do NOT create capabilities — capabilities are only ever added by
        :meth:`register_tool` / :meth:`register_capability`.
        """
        for key in ("tool", "success", "detail"):
            if key not in result:
                raise ValueError(f"tool result missing required key: {key!r}")
        entry = {
            "tool": result["tool"],
            "success": bool(result["success"]),
            "detail": result["detail"],
            "timestamp": time.time(),
            "source": TOOL_RESULT_SOURCE,
        }
        self._tool_history.append(entry)
        if not entry["success"]:
            self.register_limitation(
                f"Observed failure of tool {entry['tool']!r}: {entry['detail']}",
                source=OBSERVED_FAILURE_SOURCE,
            )

    # ------------------------------------------------------------------
    # Task / resources / open questions
    # ------------------------------------------------------------------

    def set_task(self, task: str, objectives: list[str]) -> None:
        """Record the current task and its objectives (source: task assignment)."""
        self._task = task
        self._objectives = list(objectives)

    def update_resources(self, budgets: dict) -> None:
        """Record current resource/budget state (source: resource update)."""
        if not isinstance(budgets, dict):
            raise TypeError("budgets must be a dict")
        self._resources = dict(budgets)

    def add_unresolved(self, question: str) -> None:
        """Add an open question the system cannot currently answer."""
        if not question or not question.strip():
            raise ValueError("question must be non-empty")
        self._unresolved.append(_Unresolved(question=question, timestamp=time.time()))

    def resolve_unresolved(self, question: str, resolution: str) -> None:
        """Mark a previously recorded open question as resolved.

        Raises ``ValueError`` if the question was never recorded.
        """
        for item in self._unresolved:
            if item.question == question and not item.resolved:
                item.resolved = True
                item.resolution = resolution
                item.resolved_at = time.time()
                return
        raise ValueError(f"no unresolved question recorded: {question!r}")

    # ------------------------------------------------------------------
    # Performance predictions about itself
    # ------------------------------------------------------------------

    @staticmethod
    def _check_confidence(confidence: float) -> float:
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            raise TypeError("confidence must be a number")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("confidence must be within 0..1")
        return float(confidence)

    def predict_performance(
        self, task: str, predicted_success: bool, confidence: float
    ) -> PerformancePrediction:
        """Record a prediction about the system's own future success on ``task``."""
        prediction = PerformancePrediction(
            id=uuid.uuid4().hex,
            task=task,
            predicted_success=bool(predicted_success),
            confidence=self._check_confidence(confidence),
            timestamp=time.time(),
        )
        self._predictions.append(prediction)
        return prediction

    def score_prediction(self, prediction_id: str, actual: bool) -> None:
        """Record the observed outcome for a previous performance prediction.

        Raises ``ValueError`` if no prediction with that id exists.
        """
        for prediction in self._predictions:
            if prediction.id == prediction_id:
                prediction.actual = bool(actual)
                return
        raise ValueError(f"unknown prediction id: {prediction_id!r}")

    def accuracy(self) -> dict:
        """Descriptive accuracy of scored performance predictions.

        Returns ``{"n", "accuracy", "brier"}`` over predictions with a recorded
        outcome. ``accuracy`` is the fraction whose ``predicted_success``
        matched ``actual``; ``brier`` is the mean squared error between the
        implied probability of success and the binary outcome, where the
        implied probability is ``confidence`` when ``predicted_success`` is
        True and ``1 - confidence`` when it is False (so a confident correct
        "will fail" prediction scores well). With no scored
        predictions, ``accuracy`` and ``brier`` are ``None`` (not fabricated).
        """
        scored = [p for p in self._predictions if p.actual is not None]
        n = len(scored)
        if n == 0:
            return {"n": 0, "accuracy": None, "brier": None}
        correct = sum(1 for p in scored if p.predicted_success == p.actual)
        brier = (
            sum(
                ((p.confidence if p.predicted_success else 1.0 - p.confidence)
                 - float(p.actual)) ** 2
                for p in scored
            )
            / n
        )
        return {"n": n, "accuracy": correct / n, "brier": brier}

    # ------------------------------------------------------------------
    # Snapshot for the API
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Return a plain-dict snapshot of the self-model for the API.

        Contains only JSON-safe values: lists of dicts, strings, numbers,
        booleans, and ``None``.
        """
        return {
            "capabilities": [asdict(c) for c in self._capabilities.values()],
            "limitations": [asdict(lim) for lim in self._limitations],
            "tools": [asdict(t) for t in self._tools.values()],
            "task": self._task,
            "objectives": list(self._objectives),
            "resources": dict(self._resources),
            "unresolved": [asdict(u) for u in self._unresolved],
            "observation_count": self._observation_count,
            "last_observation_at": self._last_observation_at,
            "tool_history": [dict(e) for e in self._tool_history],
            "performance_predictions": [asdict(p) for p in self._predictions],
            "performance_accuracy": self.accuracy(),
            "generated_at": time.time(),
        }


# Re-exported at package level for convenience.
__all__ = [
    "PerformancePrediction",
    "SelfModel",
    "OBSERVED_FAILURE_SOURCE",
    "TOOL_RESULT_SOURCE",
]
