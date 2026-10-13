"""The cognitive loop: 15-stage cycle, task runner, and loop factory.

What it does:
    - :class:`CognitiveLoop` runs one cognitive cycle per :meth:`step` call
      (perception → attention → workspace → metacognition → planning →
      execution → memory/self-model/learning updates → audit) and multi-cycle
      tasks via :meth:`run`.
    - :func:`build_loop` wires a full loop from a :class:`Config`: provider,
      persistence, and every cognitive component, honouring
      ``config.components`` flags.

Component model:
    - Sibling packages (``perception``, ``workspace``, ``memory``,
      ``self_model``, ``metacognition``, ``attention``, ``goals``,
      ``learning``, ``safety``) are imported **lazily inside functions** so
      this module never creates import cycles and keeps working while other
      modules are still being implemented.
    - A component passed as ``None`` degrades per the contract: no workspace
      → candidates pass through unfiltered; no memory → recall skipped; no
      self-model → empty snapshot section; no metacognition → confidence/
      contradiction stages skipped; no attention → input order kept; no
      learning → outcome learning skipped.
    - Perception, safety, goals, and planning are load-bearing: when their
      real modules are not importable, minimal contract-mirroring fallbacks
      (prefixed ``_Fallback``) are used instead, and :func:`build_loop` logs
      which fallbacks are active. Fallbacks are in-memory only and are
      replaced automatically once the real modules exist.

Limits:
    - Errors in non-critical stages are recorded in the cycle's ``errors``
      list and the cycle continues degraded; a failed critical stage
      (perception ingest, budget accounting) aborts the cycle, recorded as an
      ``ABORTED ...`` entry (``CycleResult`` carries no status field, so the
      abort is reported through ``errors``).
    - The loop never invents capabilities: the self-model is registered only
      with the provider's self-reported capabilities and the ``respond`` tool.
    - ``PREDICT:`` lines in provider text are the only prediction source;
      without them, no predictions are recorded (nothing is fabricated).
    - This module performs no network I/O itself; all inference goes through
      the provider adapter.
"""

from __future__ import annotations

import importlib
import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

from .config import BudgetConfig, Config
from .errors import BudgetExceeded, ConfigError, LabError, PermissionDenied, ProviderError, SafetyError

logger = logging.getLogger("consciousness_lab.core.loop")

__all__ = ["CycleResult", "TaskResult", "CognitiveLoop", "build_loop"]


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------

@dataclass
class CycleResult:
    """Outcome of a single cognitive cycle."""

    cycle: int
    observation_id: str
    selected_ids: list[str]
    action: dict | None  # Action converted via asdict(); None when planning failed
    provider_calls: int
    tokens_used: int
    predictions: list[dict]
    contradictions: list[dict]
    confidence: float | None
    errors: list[str]
    duration_s: float


@dataclass
class TaskResult:
    """Outcome of a multi-cycle :meth:`CognitiveLoop.run`."""

    task: str
    status: str  # completed | budget_exhausted | stopped | failed
    cycles: int
    tokens_used: int
    goal_status: dict
    predictions_scored: int
    errors: list[str]


# ---------------------------------------------------------------------------
# Lazy sibling-module loading
# ---------------------------------------------------------------------------

_lazy_cache: dict[str, Any] = {}


def _lazy(dotted: str) -> Any:
    """Import ``"package.module:Attribute"`` lazily.

    Returns the attribute, or ``None`` when the module (or attribute) is not
    importable. Successful loads are cached; failures are retried on the next
    call so modules that appear later are picked up. Only ``ImportError`` is
    treated as "unavailable" — anything else (e.g. ``SyntaxError``) propagates
    loudly rather than being mistaken for a missing module.
    """
    if dotted not in _lazy_cache:
        module_name, _, attr = dotted.partition(":")
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            return None
        _lazy_cache[dotted] = getattr(module, attr, None)
    return _lazy_cache[dotted]


# ---------------------------------------------------------------------------
# Contract-mirroring fallbacks (used only when the real module is missing)
# ---------------------------------------------------------------------------

@dataclass
class _FallbackWorkspaceItem:
    """Mirrors ``workspace.WorkspaceItem`` (id/content/source/salience/...)."""

    id: str
    content: str
    source: str
    salience: float = 0.5
    relevance: float | None = None
    novelty: float | None = None
    confidence: float = 0.5
    attention_score: float | None = None
    timestamp: float = field(default_factory=time.time)
    provenance: dict = field(default_factory=dict)

    def priority(self) -> float:
        """Salience × relevance × novelty, defaulting missing factors to 1.0."""
        return self.salience * (self.relevance or 1.0) * (self.novelty or 1.0)


@dataclass
class _FallbackAction:
    """Mirrors ``goals.planning.Action``."""

    id: str
    type: str  # respond | noop | tool | await_approval
    payload: dict = field(default_factory=dict)
    requires_approval: bool = False
    budget_cost: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class _FallbackObservation:
    """Mirrors ``perception.Observation``."""

    id: str
    modality: str
    content: str
    timestamp: float
    metadata: dict
    source: str = "external"


@dataclass
class _FallbackGoal:
    """Mirrors ``goals.Goal``."""

    id: str
    description: str
    success_criteria: str = ""
    status: str = "pending"
    parent_id: str | None = None
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class _FallbackPerception:
    """Minimal perception: string/dict ingest with sanitising.

    Mirrors ``perception.Perception.ingest``: strips control characters
    (except newline/tab), caps content at 20 000 chars, rejects unknown
    modalities with ``ValueError``.
    """

    def ingest(self, raw: str | dict) -> _FallbackObservation:
        if isinstance(raw, str):
            modality, content, metadata, source = "text", raw, {}, "external"
        elif isinstance(raw, dict):
            modality = raw.get("modality", "text")
            content = raw.get("content", "")
            metadata = raw.get("metadata", {}) or {}
            source = raw.get("source", "external")
            if modality not in ("text", "task", "structured"):
                raise ValueError(f"unknown modality: {modality!r}")
            if not isinstance(content, str):
                content = str(content)
            if not isinstance(metadata, dict):
                raise ValueError("observation metadata must be a mapping")
        else:
            raise ValueError(f"cannot ingest {type(raw).__name__}; expected str or dict")
        content = _CONTROL_CHARS_RE.sub("", content)
        if len(content) > 20_000:
            content = content[:20_000]
        return _FallbackObservation(
            id=uuid.uuid4().hex,
            modality=modality,
            content=content,
            timestamp=time.time(),
            metadata=dict(metadata),
            source=source,
        )


class _FallbackBudget:
    """Mirrors ``safety.Budget``: hard caps on steps/tokens/tool-calls/time."""

    def __init__(self, max_steps: int, max_tokens: int, max_wall_time_s: float, max_tool_calls: int) -> None:
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.max_wall_time_s = max_wall_time_s
        self.max_tool_calls = max_tool_calls
        self.steps_used = 0
        self.tokens_used = 0
        self.tool_calls_used = 0
        self.started_at = time.time()

    @classmethod
    def from_config(cls, c: BudgetConfig) -> "_FallbackBudget":
        return cls(c.max_steps, c.max_tokens, c.max_wall_time_s, c.max_tool_calls)

    def consume_step(self) -> None:
        if self.steps_used >= self.max_steps:
            raise BudgetExceeded(f"step budget exhausted ({self.steps_used}/{self.max_steps})")
        self.steps_used += 1

    def consume_tokens(self, n: int) -> None:
        if n < 0:
            raise LabError("cannot consume a negative token count")
        if self.tokens_used + n > self.max_tokens:
            raise BudgetExceeded(
                f"token budget exhausted ({self.tokens_used}+{n} > {self.max_tokens})"
            )
        self.tokens_used += n

    def consume_tool_call(self) -> None:
        if self.tool_calls_used >= self.max_tool_calls:
            raise BudgetExceeded("tool-call budget exhausted")
        self.tool_calls_used += 1

    def check_time(self) -> None:
        if time.time() - self.started_at > self.max_wall_time_s:
            raise BudgetExceeded("wall-time budget exhausted")

    def remaining(self) -> dict:
        return {
            "steps": self.max_steps - self.steps_used,
            "tokens": self.max_tokens - self.tokens_used,
            "tool_calls": self.max_tool_calls - self.tool_calls_used,
            "wall_time_s": max(0.0, self.max_wall_time_s - (time.time() - self.started_at)),
        }

    def exhausted(self) -> bool:
        return (
            self.steps_used >= self.max_steps
            or self.tokens_used >= self.max_tokens
            or self.tool_calls_used >= self.max_tool_calls
            or (time.time() - self.started_at) > self.max_wall_time_s
        )


class _FallbackPermissions:
    """Mirrors ``safety.Permissions``: allowlist that only a human can change."""

    def __init__(self, allowlist: tuple[str, ...] = (), *, locked: bool = True) -> None:
        self._allowlist = tuple(allowlist)
        self._locked = locked

    def allowed(self, tool_name: str) -> bool:
        return tool_name in self._allowlist

    def grant(self, tool_name: str, *, authorized: bool = False) -> None:
        if not authorized:
            raise SafetyError("grant() requires explicit human authorization (authorized=True)")
        if tool_name not in self._allowlist:
            self._allowlist = self._allowlist + (tool_name,)

    def revoke(self, tool_name: str, *, authorized: bool = False) -> None:
        if not authorized:
            raise SafetyError("revoke() requires explicit human authorization (authorized=True)")
        self._allowlist = tuple(t for t in self._allowlist if t != tool_name)


class _FallbackAuditLog:
    """In-memory append-only audit log (the real one persists to SQLite)."""

    def __init__(self, store: Any = None, path: str | None = None) -> None:
        self._entries: list[dict] = []

    def log(self, event: str, details: dict) -> None:
        if not isinstance(details, dict):
            raise LabError("audit details must be a mapping")
        self._entries.append(
            {"id": uuid.uuid4().hex, "ts": time.time(), "event": event, "details": details}
        )

    def read(self, limit: int = 100) -> list[dict]:
        return list(self._entries[-max(0, limit):])


class _FallbackSafetyController:
    """Mirrors ``safety.SafetyController`` over the fallback budget/permissions."""

    def __init__(self, budget: Any, permissions: Any, audit: Any, *, autonomous_enabled: bool = True) -> None:
        self.budget = budget
        self.permissions = permissions
        self.audit = audit
        self.autonomous_enabled = autonomous_enabled
        self._stopped = False

    def _ensure_running(self) -> None:
        if self._stopped:
            raise LabError("run has been stopped")

    def consume_step(self) -> None:
        self._ensure_running()
        self.budget.consume_step()

    def consume_tokens(self, n: int) -> None:
        self._ensure_running()
        self.budget.consume_tokens(n)

    def check_action(self, action: Any) -> None:
        self._ensure_running()
        if getattr(action, "requires_approval", False):
            raise PermissionDenied("human approval required")
        action_type = getattr(action, "type", "")
        if action_type == "tool":
            payload = getattr(action, "payload", {}) or {}
            tool = payload.get("tool", "")
            if not self.permissions.allowed(tool):
                raise PermissionDenied(f"tool {tool!r} is not in the permission allowlist")
        cost = getattr(action, "budget_cost", {}) or {}
        tokens = int(cost.get("tokens", 0) or 0)
        tool_calls = int(cost.get("tool_calls", 0) or 0)
        if tokens:
            self.budget.consume_tokens(tokens)
        for _ in range(tool_calls):
            self.budget.consume_tool_call()

    def stop(self, run_id: str) -> None:
        self._stopped = True
        try:
            self.audit.log("stop", {"run_id": run_id})
        except Exception:
            pass

    @property
    def stopped(self) -> bool:
        return self._stopped

    def disable_autonomous(self, *, authorized: bool = False) -> None:
        if not authorized:
            raise SafetyError("disable_autonomous() requires explicit human authorization")
        self.autonomous_enabled = False


class _FallbackGoalManager:
    """In-memory goal manager mirroring ``goals.GoalManager`` (no persistence)."""

    def __init__(self, store: Any = None) -> None:
        self._goals: dict[str, _FallbackGoal] = {}

    def create(self, description: str, success_criteria: str = "", parent_id: str | None = None) -> _FallbackGoal:
        if not description or not description.strip():
            raise LabError("goal description must be non-empty")
        goal = _FallbackGoal(
            id=uuid.uuid4().hex,
            description=description.strip(),
            success_criteria=success_criteria,
            parent_id=parent_id,
        )
        self._goals[goal.id] = goal
        return goal

    def get(self, goal_id: str) -> _FallbackGoal | None:
        return self._goals.get(goal_id)

    def active_goals(self) -> list[_FallbackGoal]:
        return [g for g in self._goals.values() if g.status in ("pending", "active")]

    def decompose(self, goal: _FallbackGoal) -> list[_FallbackGoal]:
        """Naive baseline: split on ';', ' then ', or numbered steps.

        Documented placeholder heuristic for the planner to improve later.
        """
        text = goal.description
        parts = [p.strip() for p in re.split(r"\s*;\s*", text) if p.strip()]
        if len(parts) == 1:
            parts = [p.strip() for p in re.split(r"\s+then\s+", text, flags=re.IGNORECASE) if p.strip()]
        if len(parts) == 1:
            parts = [p.strip() for p in re.split(r"\d+[.)]\s+", text) if p.strip()]
        if len(parts) <= 1:
            parts = [f"gather information for: {text}", f"produce result for: {text}"]
        return [self.create(p, parent_id=goal.id) for p in parts]

    def track_progress(self, goal_id: str, progress: float, note: str = "") -> _FallbackGoal:
        goal = self._goals.get(goal_id)
        if goal is None:
            raise LabError(f"unknown goal {goal_id!r}")
        goal.progress = max(0.0, min(1.0, float(progress)))
        if goal.status == "pending":
            goal.status = "active"
        return goal

    def mark_complete(self, goal_id: str, reason: str = "") -> _FallbackGoal:
        goal = self._goals.get(goal_id)
        if goal is None:
            raise LabError(f"unknown goal {goal_id!r}")
        goal.status = "completed"
        goal.progress = 1.0
        return goal

    def mark_abandoned(self, goal_id: str, reason: str) -> _FallbackGoal:
        if not reason:
            raise LabError("abandoning a goal requires a reason")
        goal = self._goals.get(goal_id)
        if goal is None:
            raise LabError(f"unknown goal {goal_id!r}")
        goal.status = "abandoned"
        return goal


class _FallbackPlanner:
    """Minimal planner mirroring ``goals.planning.Planner``.

    Default: a ``respond`` action (600-token budget estimate). ``noop`` when
    there is no active goal, ``None`` when the goal is complete, and a
    ``tool`` action requiring approval when the self-model names a needed tool
    outside the permission allowlist (convention: ``self_model`` snapshot key
    ``"needed_tools"``).
    """

    def __init__(self, budgets: BudgetConfig, permissions: Any) -> None:
        self.budgets = budgets
        self.permissions = permissions

    def next_action(self, goal: Any | None, workspace_state: Any, self_model: dict) -> Any | None:
        if goal is None:
            return _FallbackAction(
                id=uuid.uuid4().hex, type="noop", payload={"reason": "no active goal"}
            )
        status = getattr(goal, "status", "pending")
        progress = getattr(goal, "progress", 0.0) or 0.0
        if status == "completed" or progress >= 1.0:
            return None
        snapshot = self_model or {}
        for tool in snapshot.get("needed_tools") or []:
            if not self.permissions.allowed(tool):
                return _FallbackAction(
                    id=uuid.uuid4().hex,
                    type="tool",
                    payload={"tool": tool, "goal_id": getattr(goal, "id", "")},
                    requires_approval=True,
                    budget_cost={"tool_calls": 1},
                )
        return _FallbackAction(
            id=uuid.uuid4().hex,
            type="respond",
            payload={
                "goal_id": getattr(goal, "id", ""),
                "instruction": f"Work on goal: {getattr(goal, 'description', '')}",
            },
            requires_approval=False,
            budget_cost={"tokens": 600},
        )


# ---------------------------------------------------------------------------
# CognitiveLoop
# ---------------------------------------------------------------------------

_PREDICT_RE = re.compile(r"(?m)^PREDICT:\s*(.+?)\s*$")


class CognitiveLoop:
    """One cognitive cycle per :meth:`step`; multi-cycle tasks via :meth:`run`.

    Component keyword arguments (all optional, ``None`` means degraded):
    perception, workspace, attention, episodic_memory, semantic_memory,
    outcome_memory, working_memory, self_model, contradiction_detector,
    confidence_estimator, goal_manager, planner, learning, safety,
    permissions, budget, audit, store.
    """

    def __init__(self, config: Config, provider: Any, **components: Any) -> None:
        if config is None:
            raise ConfigError("CognitiveLoop requires a Config")
        if provider is None:
            raise ConfigError("CognitiveLoop requires a provider")
        self.config = config
        self.provider = provider
        self.components = dict(components)
        self.perception = components.get("perception")
        self.workspace = components.get("workspace")
        self.attention = components.get("attention")
        self.episodic_memory = components.get("episodic_memory")
        self.semantic_memory = components.get("semantic_memory")
        self.outcome_memory = components.get("outcome_memory")
        self.working_memory = components.get("working_memory")
        self.self_model = components.get("self_model")
        self.contradiction_detector = components.get("contradiction_detector")
        self.confidence_estimator = components.get("confidence_estimator")
        self.goal_manager = components.get("goal_manager")
        self.planner = components.get("planner")
        self.learning = components.get("learning")
        self.safety = components.get("safety")
        self.permissions = components.get("permissions")
        self.budget = components.get("budget")
        self.audit = components.get("audit")
        self.store = components.get("store")
        # Contradictions surfaced via SemanticMemory's on_contradiction hook.
        self.memory_contradictions: list = []
        self._cycle = 0
        self._tokens_used = 0
        self._provider_calls = 0
        self._predictions_recorded = 0
        self._recent_errors: list[str] = []
        self._confidence_history: list[dict] = []

    # -- internal helpers ------------------------------------------------

    def _safety(self) -> Any:
        if self.safety is None:
            raise LabError("safety controller is unavailable; refusing to run")
        return self.safety

    def _consume_step(self) -> None:
        """Consume one step from the run budget.

        Uses ``safety.consume_step()`` when the controller exposes it, else
        falls back to ``safety.budget.consume_step()`` (the interface the
        real ``SafetyController`` implements).
        """
        safety = self._safety()
        consume = getattr(safety, "consume_step", None)
        if callable(consume):
            consume()
        else:
            safety.budget.consume_step()

    def _consume_tokens(self, n: int) -> None:
        """Consume ``n`` tokens from the run budget (same dispatch as above)."""
        safety = self._safety()
        consume = getattr(safety, "consume_tokens", None)
        if callable(consume):
            consume(n)
        else:
            safety.budget.consume_tokens(n)

    def _check_time(self) -> None:
        """Enforce the wall-time budget (same dispatch as above)."""
        safety = self._safety()
        check = getattr(safety, "check_time", None)
        if callable(check):
            check()
        else:
            safety.budget.check_time()

    def _perception(self) -> Any:
        if self.perception is None:
            raise LabError("perception is unavailable; cannot ingest observations")
        return self.perception

    def _make_item(self, content: str, source: str, salience: float, provenance: dict | None = None) -> Any:
        cls = _lazy("consciousness_lab.workspace.workspace:WorkspaceItem") or _FallbackWorkspaceItem
        return cls(
            id=uuid.uuid4().hex,
            content=str(content),
            source=source,
            salience=float(salience),
            provenance=provenance or {},
        )

    @staticmethod
    def _action_to_dict(action: Any) -> dict | None:
        if action is None:
            return None
        if is_dataclass(action):
            return asdict(action)
        if isinstance(action, dict):
            return dict(action)
        return {"repr": repr(action)}

    def _build_prompt(self, obs: Any, selected: list, goal: Any, memory_snippets: list[str]) -> str:
        lines: list[str] = []
        if goal is not None:
            lines.append(f"Goal: {getattr(goal, 'description', '')}")
        if memory_snippets:
            lines.append("Recalled context:")
            for snippet in memory_snippets[:3]:
                lines.append(f"- {snippet[:400]}")
        if selected:
            lines.append("Salient items:")
            for item in selected[:7]:
                lines.append(f"- [{getattr(item, 'source', '?')}] {getattr(item, 'content', '')[:400]}")
        lines.append(f"Observation: {getattr(obs, 'content', '')[:2000]}")
        lines.append(
            "Respond concisely and factually. Do not claim consciousness, feelings, "
            "or subjective experience."
        )
        return "\n".join(lines)

    # -- the 15-stage cycle ------------------------------------------------

    def step(self, raw_observation: str | dict) -> CycleResult:
        """Run one cognitive cycle over ``raw_observation``.

        Each stage is guarded: non-critical failures are recorded in
        ``errors`` and the cycle continues degraded; a failed critical stage
        (perception ingest, budget accounting) aborts the cycle, recorded as
        an ``ABORTED ...`` error entry.
        """
        t_start = time.time()
        self._cycle += 1
        cycle = self._cycle
        errors: list[str] = []
        observation_id = ""
        selected_ids: list[str] = []
        action: Any = None
        provider_calls = 0
        tokens_used = 0
        predictions: list[dict] = []
        contradictions: list[dict] = []
        confidence_value: float | None = None
        response_text: str | None = None
        goal: Any = None
        memory_snippets: list[str] = []

        def finish() -> CycleResult:
            duration_s = time.time() - t_start
            if errors:
                self._recent_errors.extend(errors)
                del self._recent_errors[:-100]
            return CycleResult(
                cycle=cycle,
                observation_id=observation_id,
                selected_ids=selected_ids,
                action=self._action_to_dict(action),
                provider_calls=provider_calls,
                tokens_used=tokens_used,
                predictions=predictions,
                contradictions=contradictions,
                confidence=confidence_value,
                errors=errors,
                duration_s=duration_s,
            )

        # Stage 1: perception (critical) + budget (critical).
        try:
            obs = self._perception().ingest(raw_observation)
            observation_id = getattr(obs, "id", "")
            obs_content = getattr(obs, "content", "")
        except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
            errors.append(f"ABORTED stage=perception: {type(exc).__name__}: {exc}")
            return finish()
        try:
            self._consume_step()
        except BudgetExceeded as exc:
            errors.append(f"ABORTED stage=budget: {exc}")
            return finish()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"ABORTED stage=budget: {type(exc).__name__}: {exc}")
            return finish()

        # Stage 1b: episodic record of the observation (non-critical).
        # Episodic memory stores "recorded interactions and events"; each
        # ingested observation is such an event. Semantic memory is NOT
        # written here: observations are raw input, not vetted facts.
        if self.episodic_memory is not None:
            try:
                self.episodic_memory.record(
                    obs_content,
                    provenance=f"loop:observation:{observation_id}",
                    confidence=0.8,
                    source_type=(
                        "user_provided"
                        if getattr(obs, "source", "external") == "external"
                        else "inferred"
                    ),
                    metadata={
                        "modality": getattr(obs, "modality", ""),
                        "cycle": cycle,
                    },
                )
            except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
                errors.append(
                    f"stage=episodic_record: {type(exc).__name__}: {exc}"
                )

        # Stage 2: build candidate workspace items.
        candidates: list = []
        try:
            candidates.append(
                self._make_item(
                    obs_content,
                    "perception",
                    0.9,
                    {"observation_id": observation_id, "modality": getattr(obs, "modality", "")},
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"stage=candidates/observation: {type(exc).__name__}: {exc}")
        if self.working_memory is not None:
            try:
                for entry in self.working_memory.contents():
                    text = entry if isinstance(entry, str) else getattr(entry, "content", str(entry))
                    candidates.append(self._make_item(text, "memory:working", 0.6))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"stage=candidates/working-memory: {type(exc).__name__}: {exc}")
        if self.episodic_memory is not None:
            try:
                for record in self.episodic_memory.recall(obs_content, k=3):
                    memory_snippets.append(record.content)
                    candidates.append(
                        self._make_item(
                            record.content,
                            "memory:episodic",
                            0.5,
                            {"record_id": record.id, "confidence": record.confidence},
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"stage=candidates/episodic-recall: {type(exc).__name__}: {exc}")
        if self.semantic_memory is not None:
            try:
                for record in self.semantic_memory.query(obs_content, k=3):
                    memory_snippets.append(record.content)
                    candidates.append(
                        self._make_item(
                            record.content,
                            "memory:semantic",
                            0.5,
                            {"record_id": record.id, "confidence": record.confidence},
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"stage=candidates/semantic-query: {type(exc).__name__}: {exc}")
        if self.goal_manager is not None:
            try:
                active = self.goal_manager.active_goals()
                if active:
                    goal = active[0]
                for active_goal in active[:3]:
                    candidates.append(
                        self._make_item(
                            f"Active goal: {active_goal.description}",
                            "goal",
                            0.7,
                            {"goal_id": active_goal.id},
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"stage=candidates/goals: {type(exc).__name__}: {exc}")
        if self.self_model is not None:
            try:
                snapshot = self.self_model.snapshot() or {}
                for question in (snapshot.get("unresolved_questions") or [])[:3]:
                    text = question if isinstance(question, str) else str(question.get("question", question))
                    candidates.append(self._make_item(f"Open question: {text}", "self_model", 0.4))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"stage=candidates/self-model: {type(exc).__name__}: {exc}")

        # Stage 3: attention prioritization (or input order when attention is off).
        scored = candidates
        if self.attention is not None:
            try:
                goals_for_attention = self.goal_manager.active_goals() if self.goal_manager is not None else []
                workspace_state = self.workspace.snapshot() if self.workspace is not None else None
                scored = self.attention.prioritize(candidates, goals_for_attention, workspace_state)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"stage=attention: {type(exc).__name__}: {exc}")
                scored = candidates

        # Stage 4: workspace selection (or all candidates when workspace is off).
        selected = scored
        if self.workspace is not None:
            try:
                selected = self.workspace.select(scored)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"stage=workspace-select: {type(exc).__name__}: {exc}")
                selected = scored
        selected_ids = [getattr(item, "id", "") for item in selected]

        # Stage 5: broadcast.
        if self.workspace is not None:
            try:
                self.workspace.broadcast(selected)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"stage=broadcast: {type(exc).__name__}: {exc}")

        # Stage 6: contradiction detection.
        if self.contradiction_detector is not None:
            try:
                existing: list = []
                if self.semantic_memory is not None:
                    existing = self.semantic_memory.query(obs_content, k=5)
                found = self.contradiction_detector.check(obs_content, existing)
                for item in found or []:
                    contradictions.append(
                        {
                            "id": getattr(item, "id", ""),
                            "description": getattr(item, "description", ""),
                            "severity": getattr(item, "severity", 0.0),
                            "record_a_id": getattr(item, "record_a_id", None),
                            "record_b_id": getattr(item, "record_b_id", None),
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"stage=contradiction: {type(exc).__name__}: {exc}")

        # Stage 7: confidence estimation.
        if self.confidence_estimator is not None:
            try:
                confidences = [getattr(item, "confidence", 0.5) or 0.5 for item in selected]
                agreement = (
                    sum(1.0 for value in confidences if value >= 0.5) / len(confidences)
                    if confidences
                    else 0.5
                )
                context = {
                    "sources": [getattr(item, "source", "") for item in selected],
                    "n_evidence": len(selected),
                    "agreement": agreement,
                    "recency": 1.0,
                    "base_rate": 0.5,
                }
                estimate = self.confidence_estimator.estimate(context)
                confidence_value = float(estimate.value)
                self._confidence_history.append(
                    {"cycle": cycle, "value": confidence_value, "basis": getattr(estimate, "basis", "")}
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"stage=confidence: {type(exc).__name__}: {exc}")

        # Stage 8: planning.
        self_model_snapshot: dict = {}
        workspace_state_for_planner: Any = None
        try:
            workspace_state_for_planner = (
                self.workspace.snapshot()
                if self.workspace is not None
                else {"items": [self._action_to_dict(item) for item in selected], "cycle": cycle}
            )
            self_model_snapshot = self.self_model.snapshot() if self.self_model is not None else {}
            if self.planner is not None:
                action = self.planner.next_action(goal, workspace_state_for_planner, self_model_snapshot)
            else:
                action = _FallbackAction(
                    id=uuid.uuid4().hex,
                    type="respond",
                    payload={"instruction": "Respond to the observation."},
                    budget_cost={"tokens": 600},
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"stage=planning: {type(exc).__name__}: {exc}")
            action = None

        # Stage 9: approval gate — never execute an action requiring approval.
        if action is not None and getattr(action, "requires_approval", False):
            if self.audit is not None:
                try:
                    self.audit.log(
                        "action_pending_approval",
                        {"cycle": cycle, "action": self._action_to_dict(action)},
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"stage=audit: {type(exc).__name__}: {exc}")
            errors.append("stage=approval: action requires human approval; cycle ends without execution")
            return finish()

        # Stage 10: execution.
        action_type = getattr(action, "type", "") if action is not None else ""
        if action is not None and action_type == "respond":
            try:
                prompt = self._build_prompt(obs, selected, goal, memory_snippets)
                params = dict(self.config.provider_params) if isinstance(self.config.provider_params, dict) else {}
                completion = self.provider.complete(prompt, params)
                provider_calls = 1
                reported = (completion.prompt_tokens or 0) + (completion.completion_tokens or 0)
                tokens_used = reported if reported > 0 else len(prompt.split()) + len(completion.text.split())
                self._consume_tokens(tokens_used)
                response_text = completion.text
                self._provider_calls += 1
                self._tokens_used += tokens_used
            except (BudgetExceeded, PermissionDenied, ProviderError) as exc:
                errors.append(f"stage=execute: {type(exc).__name__}: {exc}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"stage=execute: {type(exc).__name__}: {exc}")
        elif action is not None and action_type == "noop":
            pass  # nothing to do
        elif action is not None and action_type == "tool":
            try:
                payload = getattr(action, "payload", {}) or {}
                tool_name = payload.get("tool", "")
                permissions = self.permissions
                if permissions is None:
                    raise LabError("permissions unavailable; cannot authorize tool action")
                if not permissions.allowed(tool_name):
                    raise PermissionDenied(f"tool {tool_name!r} is not in the permission allowlist")
                errors.append(
                    f"stage=execute: tool {tool_name!r} is permitted but no executor "
                    "is registered in v0.1; skipped"
                )
            except (PermissionDenied, LabError) as exc:
                errors.append(f"stage=execute: {type(exc).__name__}: {exc}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"stage=execute: {type(exc).__name__}: {exc}")
        elif action is not None:
            errors.append(f"stage=execute: unknown action type {action_type!r}; skipped")

        # Stage 11: record predictions from PREDICT: lines (if any).
        if response_text and self.outcome_memory is not None:
            try:
                for match in _PREDICT_RE.finditer(response_text):
                    content = match.group(1).strip()
                    if not content:
                        continue
                    record = self.outcome_memory.record_prediction(
                        content,
                        confidence=confidence_value if confidence_value is not None else 0.5,
                        context={"cycle": cycle, "observation_id": observation_id},
                    )
                    predictions.append(
                        {"id": record.id, "content": record.content, "confidence": record.confidence}
                    )
                    self._predictions_recorded += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"stage=prediction: {type(exc).__name__}: {exc}")

        # Stage 12: self-model updates.
        if self.self_model is not None:
            try:
                self.self_model.update_from_observation(
                    {
                        "id": observation_id,
                        "modality": getattr(obs, "modality", ""),
                        "content": getattr(obs, "content", ""),
                        "metadata": getattr(obs, "metadata", {}),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"stage=self-model/observation: {type(exc).__name__}: {exc}")
            try:
                succeeded = response_text is not None and not any(
                    e.startswith("stage=execute") for e in errors
                )
                self.self_model.update_from_tool_result(
                    {
                        "tool": action_type or "none",
                        "success": succeeded,
                        "detail": (response_text or "")[:200],
                    }
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"stage=self-model/tool-result: {type(exc).__name__}: {exc}")

        # Stage 13: learning observes the cycle.
        if self.learning is not None:
            try:
                self.learning.observe_cycle(
                    {
                        "action_type": action_type or None,
                        "success": response_text is not None,
                        "strategy": None,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"stage=learning: {type(exc).__name__}: {exc}")

        # Stage 14: audit.
        if self.audit is not None:
            try:
                self.audit.log(
                    "cycle",
                    {
                        "cycle": cycle,
                        "observation_id": observation_id,
                        "selected": len(selected_ids),
                        "action_type": action_type or None,
                        "provider_calls": provider_calls,
                        "tokens_used": tokens_used,
                        "n_predictions": len(predictions),
                        "n_contradictions": len(contradictions),
                        "n_errors": len(errors),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"stage=audit: {type(exc).__name__}: {exc}")

        # Stage 15: return the cycle result.
        return finish()

    # -- task runner -----------------------------------------------------

    def run(self, task: str, *, max_cycles: int | None = None) -> TaskResult:
        """Run ``task`` for up to ``max_cycles`` cycles (default: step budget).

        Creates a root goal, then feeds ``{"modality": "task", ...}``
        observations into :meth:`step` until the goal completes, the budget is
        exhausted, or safety stops the run.
        """
        t_start = time.time()
        if max_cycles is None:
            max_cycles = self.config.budgets.max_steps
        errors: list[str] = []
        goal_id: str | None = None
        if self.goal_manager is not None:
            try:
                goal = self.goal_manager.create(task)
                goal_id = goal.id
            except Exception as exc:  # noqa: BLE001
                errors.append(f"run: goal creation failed: {type(exc).__name__}: {exc}")

        status = "budget_exhausted"
        cycles = 0
        tokens_used = 0
        predictions_scored = 0
        for n in range(max_cycles):
            try:
                if self._safety().stopped:
                    status = "stopped"
                    break
            except LabError as exc:
                errors.append(f"run: {exc}")
                status = "failed"
                break
            try:
                self._check_time()
            except BudgetExceeded:
                status = "budget_exhausted"
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"run: time check failed: {type(exc).__name__}: {exc}")
                status = "failed"
                break
            observation = {"modality": "task", "content": task, "metadata": {"cycle": n}}
            result = self.step(observation)
            cycles += 1
            tokens_used += result.tokens_used
            predictions_scored += len(result.predictions)
            errors.extend(result.errors)
            if any(e.startswith("ABORTED stage=budget") for e in result.errors):
                status = "budget_exhausted"
                break
            if any(e.startswith("ABORTED") for e in result.errors):
                status = "failed"
                break
            if goal_id is not None and self.goal_manager is not None:
                try:
                    current = self.goal_manager.get(goal_id)
                    if current is not None and (
                        current.status == "completed" or (current.progress or 0.0) >= 1.0
                    ):
                        status = "completed"
                        break
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"run: goal poll failed: {type(exc).__name__}: {exc}")

        goal_status: dict = {}
        if goal_id is not None and self.goal_manager is not None:
            try:
                current = self.goal_manager.get(goal_id)
                if current is not None:
                    goal_status = {
                        "id": current.id,
                        "description": current.description,
                        "status": current.status,
                        "progress": current.progress,
                    }
            except Exception as exc:  # noqa: BLE001
                errors.append(f"run: goal status failed: {type(exc).__name__}: {exc}")

        logger.info(
            "run finished",
            extra={
                "task_chars": len(task),
                "status": status,
                "cycles": cycles,
                "tokens_used": tokens_used,
                "duration_s": round(time.time() - t_start, 3),
            },
        )
        return TaskResult(
            task=task,
            status=status,
            cycles=cycles,
            tokens_used=tokens_used,
            goal_status=goal_status,
            predictions_scored=predictions_scored,
            errors=errors,
        )

    # -- inspection ------------------------------------------------------

    def snapshot(self) -> dict:
        """Return live state for the API: workspace, self-model, goals, budgets."""
        data: dict = {}
        try:
            if self.workspace is not None:
                state = self.workspace.snapshot()
                data["workspace"] = asdict(state) if is_dataclass(state) else dict(state)
            else:
                data["workspace"] = {"items": [], "note": "workspace disabled"}
        except Exception as exc:  # noqa: BLE001
            data["workspace"] = {"error": f"{type(exc).__name__}: {exc}"}
        try:
            data["self_model"] = self.self_model.snapshot() if self.self_model is not None else {}
        except Exception as exc:  # noqa: BLE001
            data["self_model"] = {"error": f"{type(exc).__name__}: {exc}"}
        try:
            goals = self.goal_manager.active_goals() if self.goal_manager is not None else []
            data["goals"] = [
                {
                    "id": g.id,
                    "description": g.description,
                    "status": g.status,
                    "progress": g.progress,
                }
                for g in goals
            ]
        except Exception as exc:  # noqa: BLE001
            data["goals"] = {"error": f"{type(exc).__name__}: {exc}"}
        try:
            data["budgets"] = self.budget.remaining() if self.budget is not None else {}
        except Exception as exc:  # noqa: BLE001
            data["budgets"] = {"error": f"{type(exc).__name__}: {exc}"}
        data["provider"] = getattr(self.provider, "name", "?")
        data["cycle"] = self._cycle
        data["tokens_used"] = self._tokens_used
        data["provider_calls"] = self._provider_calls
        data["errors"] = list(self._recent_errors[-10:])
        data["memory_contradictions"] = len(self.memory_contradictions)
        return data


# ---------------------------------------------------------------------------
# build_loop factory
# ---------------------------------------------------------------------------

def build_loop(config: Config) -> CognitiveLoop:
    """Build a fully wired :class:`CognitiveLoop` from ``config``.

    Creates the provider via :func:`models.create_provider`, a
    :class:`SqliteStore` at ``config.db_path``, and every cognitive component.
    ``config.components`` flags are honoured: a disabled component is passed
    as ``None`` and the loop degrades gracefully per the contract.

    The self-model is registered only with real, observed capabilities: the
    provider's self-reported capabilities and the ``respond`` tool (the tool
    registry starts as ``["respond"]`` plus provider capabilities). The
    semantic memory's ``on_contradiction`` callback is wired to the loop's
    audit log and contradiction buffer (detection itself happens inside
    ``SemanticMemory.upsert_fact`` via the metacognition detector).

    When a sibling module is not importable, a contract-mirroring in-memory
    fallback is used for the load-bearing pieces (perception, safety,
    goals/planning); other components degrade to ``None``. A summary of
    fallbacks is logged at INFO.
    """
    from ..models import create_provider  # deferred: models must not import core.loop at module level
    from ..persistence import SqliteStore

    provider = create_provider(config.provider, config)
    flags = config.components or {}

    def enabled(name: str) -> bool:
        return bool(flags.get(name, True))

    fallbacks: list[str] = []

    def note_fallback(label: str) -> None:
        fallbacks.append(label)

    store = SqliteStore(config.db_path)

    # -- perception (load-bearing; fallback if module missing) -------------
    perception_cls = _lazy("consciousness_lab.perception.perception:Perception")
    if perception_cls is None:
        note_fallback("perception")
        perception = _FallbackPerception()
    else:
        perception = perception_cls()

    # -- workspace / attention / memory (None when disabled or unavailable) -
    workspace = None
    if enabled("workspace"):
        workspace_cls = _lazy("consciousness_lab.workspace.workspace:GlobalWorkspace")
        if workspace_cls is None:
            note_fallback("workspace (module unavailable; candidates pass through unfiltered)")
        else:
            workspace = workspace_cls(capacity=config.workspace_capacity)

    attention = None
    if enabled("attention"):
        attention_cls = _lazy("consciousness_lab.attention.attention:Attention")
        if attention_cls is None:
            note_fallback("attention (module unavailable; input order kept)")
        else:
            attention = attention_cls()

    episodic_memory = semantic_memory = outcome_memory = working_memory = None
    if enabled("memory"):
        episodic_cls = _lazy("consciousness_lab.memory.episodic:EpisodicMemory")
        semantic_cls = _lazy("consciousness_lab.memory.semantic:SemanticMemory")
        outcome_cls = _lazy("consciousness_lab.memory.outcome:OutcomeMemory")
        working_cls = _lazy("consciousness_lab.memory.working:WorkingMemory")
        if episodic_cls is None or semantic_cls is None or outcome_cls is None or working_cls is None:
            note_fallback("memory (module unavailable; recall skipped)")
        else:
            memory_contradiction_buffer: list = []

            def _on_contradiction(contradiction: Any) -> None:
                # Detection ran inside SemanticMemory.upsert_fact via the
                # metacognition ContradictionDetector; this hook forwards the
                # finding to the loop's metacognition-facing records.
                memory_contradiction_buffer.append(contradiction)
                try:
                    audit_ref.log(
                        "contradiction",
                        {
                            "id": getattr(contradiction, "id", ""),
                            "description": getattr(contradiction, "description", ""),
                            "severity": getattr(contradiction, "severity", 0.0),
                        },
                    )
                except Exception:
                    pass

            episodic_memory = episodic_cls(store, namespace="default")
            # When the metacognition ablation is active, upsert-time
            # contradiction detection must be disabled too; otherwise the
            # ablation would leak through SemanticMemory's internal detector.
            semantic_memory = semantic_cls(
                store,
                namespace="default",
                on_contradiction=_on_contradiction,
                contradiction_detector=(
                    False if not enabled("metacognition") else None
                ),
            )
            outcome_memory = outcome_cls(store, namespace="default")
            working_memory = working_cls()

    # -- self-model ---------------------------------------------------------
    self_model = None
    if enabled("self_model"):
        self_model_cls = _lazy("consciousness_lab.self_model.self_model:SelfModel")
        if self_model_cls is None:
            note_fallback("self_model (module unavailable; empty snapshot)")
        else:
            self_model = self_model_cls(store)

    # -- metacognition -------------------------------------------------------
    contradiction_detector = confidence_estimator = None
    if enabled("metacognition"):
        detector_cls = _lazy("consciousness_lab.metacognition.contradiction:ContradictionDetector")
        estimator_cls = _lazy("consciousness_lab.metacognition.confidence:ConfidenceEstimator")
        if detector_cls is None or estimator_cls is None:
            note_fallback("metacognition (module unavailable; checks skipped)")
        else:
            contradiction_detector = detector_cls()
            confidence_estimator = estimator_cls()

    # -- learning ------------------------------------------------------------
    learning = None
    if enabled("learning"):
        learning_cls = _lazy("consciousness_lab.learning.learning:Learning")
        if learning_cls is None:
            note_fallback("learning (module unavailable; outcome learning skipped)")
        else:
            learning = learning_cls(store)

    # -- safety stack (load-bearing; fallbacks if module missing) ------------
    budget_cls = _lazy("consciousness_lab.safety.safety:Budget") or _FallbackBudget
    permissions_cls = _lazy("consciousness_lab.safety.safety:Permissions") or _FallbackPermissions
    audit_cls = _lazy("consciousness_lab.safety.safety:AuditLog") or _FallbackAuditLog
    safety_cls = _lazy("consciousness_lab.safety.safety:SafetyController") or _FallbackSafetyController
    if _lazy("consciousness_lab.safety.safety:SafetyController") is None:
        note_fallback("safety")
    budget = budget_cls.from_config(config.budgets)
    permissions = permissions_cls(("respond",), locked=True)
    audit_ref = audit_cls(store)
    safety = safety_cls(budget, permissions, audit_ref, autonomous_enabled=config.autonomous_enabled)

    # -- goals + planner (load-bearing; fallbacks if module missing) --------
    goal_manager_cls = _lazy("consciousness_lab.goals.goals:GoalManager")
    if goal_manager_cls is None:
        note_fallback("goals")
        goal_manager = _FallbackGoalManager()
    else:
        goal_manager = goal_manager_cls(store)

    planner_cls = _lazy("consciousness_lab.goals.planning:Planner")
    if planner_cls is None:
        note_fallback("planning")
        planner = _FallbackPlanner(config.budgets, permissions)
    else:
        planner = planner_cls(config.budgets, permissions)

    loop = CognitiveLoop(
        config,
        provider,
        perception=perception,
        workspace=workspace,
        attention=attention,
        episodic_memory=episodic_memory,
        semantic_memory=semantic_memory,
        outcome_memory=outcome_memory,
        working_memory=working_memory,
        self_model=self_model,
        contradiction_detector=contradiction_detector,
        confidence_estimator=confidence_estimator,
        goal_manager=goal_manager,
        planner=planner,
        learning=learning,
        safety=safety,
        permissions=permissions,
        budget=budget,
        audit=audit_ref,
        store=store,
    )
    if enabled("memory") and semantic_memory is not None:
        # Expose contradictions surfaced through the on_contradiction hook.
        loop.memory_contradictions = memory_contradiction_buffer

    # -- register only real, observed capabilities on the self-model ---------
    if self_model is not None:
        try:
            capabilities = provider.capabilities()
            self_model.register_capability(
                f"text generation via provider '{provider.name}'",
                source="build_loop:provider.capabilities",
            )
            if capabilities.streaming:
                self_model.register_capability(
                    "streaming responses", source="build_loop:provider.capabilities"
                )
            if capabilities.json_mode:
                self_model.register_capability(
                    "JSON-mode responses", source="build_loop:provider.capabilities"
                )
            if capabilities.max_context_tokens:
                self_model.register_capability(
                    f"context window up to {capabilities.max_context_tokens} tokens (provider-reported)",
                    source="build_loop:provider.capabilities",
                )
            # The tool registry starts as ["respond"] plus provider capabilities.
            self_model.register_tool(
                "respond",
                "Generate a text response via the configured inference provider.",
            )
            if provider.name == "mock":
                self_model.register_limitation(
                    "mock provider performs no real inference; outputs are deterministic templates",
                    source="build_loop:provider.capabilities",
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("self-model capability registration failed: %s: %s", type(exc).__name__, exc)

    try:
        audit_ref.log(
            "loop_built",
            {
                "provider": provider.name,
                "model": config.provider_model or None,
                "fallbacks": fallbacks,
                "components": {name: enabled(name) for name in
                               ("memory", "self_model", "workspace", "metacognition", "attention", "learning")},
            },
        )
    except Exception:
        pass
    logger.info(
        "built cognitive loop",
        extra={"provider": provider.name, "fallbacks": fallbacks or "none"},
    )
    return loop
