"""Goals: Goal dataclass and GoalManager with optional sqlite persistence.

What it does: creates, retrieves, decomposes, and tracks the lifecycle of
goals (pending -> active -> completed | abandoned), persisting every
mutation to the ``goals`` table when a ``SqliteStore`` is given.

Limits:
- ``decompose`` is a naive textual baseline (split on ``;`` / "then" /
  numbered steps; otherwise two template subgoals). It does not understand
  task semantics and is explicitly a placeholder for a real planner.
- Persistence is last-write-wins via ``INSERT OR REPLACE``; there is no
  conflict detection across processes sharing one DB file.
- ``track_progress`` auto-completes a goal whose progress reaches 1.0;
  partial progress is a single float, not per-subgoal aggregation.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # persistence is owned by another agent; duck-typed at runtime
    from consciousness_lab.persistence.sqlite_store import SqliteStore

try:  # core.logging_setup is owned by another agent; fall back until it lands
    from consciousness_lab.core.logging_setup import setup_logging

    _log = setup_logging()
except Exception:  # pragma: no cover - import fallback
    _log = logging.getLogger(__name__)

#: DDL for the goals table, registered via SqliteStore.ensure_table.
GOALS_DDL = """
CREATE TABLE IF NOT EXISTS goals (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    success_criteria TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    parent_id TEXT,
    progress REAL NOT NULL DEFAULT 0.0,
    created_at REAL NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
)
"""

#: Statuses considered "active" by GoalManager.active_goals().
ACTIVE_STATUSES = ("pending", "active")

#: Valid goal statuses.
GOAL_STATUSES = ("pending", "active", "completed", "abandoned")

_NUMBERED_STEP_RE = re.compile(r"\d+[.)]\s*")


@dataclass
class Goal:
    """A unit of intended work tracked by the cognitive loop."""

    id: str
    description: str
    success_criteria: str = ""
    status: str = "pending"  # pending|active|completed|abandoned
    parent_id: str | None = None
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in GOAL_STATUSES:
            raise ValueError(
                f"invalid goal status {self.status!r}; expected one of {GOAL_STATUSES}"
            )
        if not isinstance(self.progress, (int, float)):
            raise TypeError("goal progress must be numeric")
        self.progress = max(0.0, min(1.0, float(self.progress)))

    def to_record(self) -> dict:
        """Serialize to a dict matching the ``goals`` table columns."""
        return {
            "id": self.id,
            "description": self.description,
            "success_criteria": self.success_criteria,
            "status": self.status,
            "parent_id": self.parent_id,
            "progress": self.progress,
            "created_at": self.created_at,
            "metadata_json": json.dumps(self.metadata),
        }

    @classmethod
    def from_record(cls, record: dict) -> "Goal":
        """Rebuild a Goal from a ``goals`` table row."""
        metadata = record.get("metadata_json") or "{}"
        return cls(
            id=record["id"],
            description=record["description"],
            success_criteria=record.get("success_criteria", ""),
            status=record.get("status", "pending"),
            parent_id=record.get("parent_id"),
            progress=float(record.get("progress", 0.0)),
            created_at=float(record.get("created_at", time.time())),
            metadata=json.loads(metadata) if isinstance(metadata, str) else dict(metadata),
        )


def _split_clauses(description: str) -> list[str]:
    """Naively split a goal description into sub-task clauses.

    Tries, in order: numbered steps (``1. ... 2. ...``), ``;`` separators,
    then the word "then". Returns ``[]`` when no split yields >= 2 clauses.
    This is a textual heuristic with no understanding of task semantics.
    """
    text = description.strip()
    # Numbered steps: "1. do X 2. do Y" or "1) do X 2) do Y". Require at
    # least two numbered markers so "version 2.0" style text is not split.
    if len(_NUMBERED_STEP_RE.findall(text)) >= 2:
        parts = [p.strip(" .") for p in _NUMBERED_STEP_RE.split(text)]
        parts = [p for p in parts if p]
        if len(parts) >= 2:
            return parts
    for sep in (";",):
        chunks = [c.strip(" .") for c in text.split(sep)]
        chunks = [c for c in chunks if c]
        if len(chunks) >= 2:
            return chunks
    chunks = [c.strip(" .") for c in re.split(r"\bthen\b", text, flags=re.IGNORECASE)]
    chunks = [c for c in chunks if c]
    if len(chunks) >= 2:
        return chunks
    return []


class GoalManager:
    """Create, decompose, and track goals, optionally persisted in sqlite.

    With ``store=None`` everything is in-memory only. With a store, every
    mutation is written through to the ``goals`` table, and constructing a
    new ``GoalManager`` on the same store reloads all previously saved
    goals (reopen support for interrupted tasks).
    """

    def __init__(self, store: "SqliteStore | None" = None) -> None:
        self._store = store
        self._goals: dict[str, Goal] = {}
        if store is not None:
            store.ensure_table("goals", GOALS_DDL)
            for row in store.execute_read("SELECT * FROM goals"):
                goal = Goal.from_record(row)
                self._goals[goal.id] = goal
            _log.info("goal manager loaded goals", extra={"n": len(self._goals)})

    def _save(self, goal: Goal) -> None:
        if self._store is None:
            return
        record = goal.to_record()
        cols = ", ".join(record.keys())
        placeholders = ", ".join("?" for _ in record)
        self._store.execute_write(
            f"INSERT OR REPLACE INTO goals ({cols}) VALUES ({placeholders})",
            tuple(record.values()),
        )

    def create(
        self,
        description: str,
        success_criteria: str = "",
        parent_id: str | None = None,
    ) -> Goal:
        """Create a goal (status ``"pending"``) and persist it."""
        if not description or not description.strip():
            raise ValueError("goal description must be non-empty")
        if parent_id is not None and parent_id not in self._goals:
            raise KeyError(f"unknown parent goal {parent_id!r}")
        goal = Goal(
            id=uuid.uuid4().hex,
            description=description.strip(),
            success_criteria=success_criteria,
            status="pending",
            parent_id=parent_id,
        )
        self._goals[goal.id] = goal
        self._save(goal)
        _log.info("goal created", extra={"goal_id": goal.id})
        return goal

    def get(self, goal_id: str) -> Goal | None:
        """Return the goal with ``goal_id``, or ``None`` if unknown."""
        return self._goals.get(goal_id)

    def active_goals(self) -> list[Goal]:
        """Goals with status pending/active, oldest first."""
        return sorted(
            (g for g in self._goals.values() if g.status in ACTIVE_STATUSES),
            key=lambda g: g.created_at,
        )

    def decompose(self, goal: Goal) -> list[Goal]:
        """Split ``goal`` into subgoals (naive textual baseline).

        Splits the description on numbered steps, ``;``, or "then"; when no
        split applies, creates two template subgoals ("gather information",
        "produce result"). Subgoals get ``parent_id=goal.id`` and are
        persisted. The goal itself is left unchanged.

        This is explicitly a placeholder heuristic: it performs surface
        text splitting with no semantic understanding of the task, and the
        template fallback invents generic subgoals rather than deriving
        them from the domain.
        """
        if goal.id not in self._goals:
            raise KeyError(f"unknown goal {goal.id!r}")
        clauses = _split_clauses(goal.description)
        if not clauses:
            clauses = ["gather information", "produce result"]
            _log.info(
                "goal decompose fell back to template subgoals",
                extra={"goal_id": goal.id},
            )
        subgoals = [
            self.create(
                f"{clause} (for goal: {goal.description[:80]})",
                parent_id=goal.id,
            )
            for clause in clauses
        ]
        return subgoals

    def track_progress(self, goal_id: str, progress: float, note: str = "") -> Goal:
        """Set progress (clamped to [0, 1]); reaching 1.0 auto-completes.

        Appends ``note`` to ``metadata["notes"]`` when non-empty.
        Raises ``KeyError`` for an unknown goal id.
        """
        goal = self._goals.get(goal_id)
        if goal is None:
            raise KeyError(f"unknown goal {goal_id!r}")
        try:
            progress = max(0.0, min(1.0, float(progress)))
        except (TypeError, ValueError) as exc:
            raise TypeError(f"progress must be numeric: {exc}") from exc
        goal.progress = progress
        if note:
            goal.metadata.setdefault("notes", []).append(
                {"progress": progress, "note": note, "at": time.time()}
            )
        if progress >= 1.0 and goal.status in ACTIVE_STATUSES:
            goal.status = "completed"
            goal.metadata["completion_reason"] = "progress reached 1.0"
        self._save(goal)
        return goal

    def mark_complete(self, goal_id: str, reason: str = "") -> Goal:
        """Mark a goal completed (progress set to 1.0)."""
        goal = self._goals.get(goal_id)
        if goal is None:
            raise KeyError(f"unknown goal {goal_id!r}")
        goal.status = "completed"
        goal.progress = 1.0
        if reason:
            goal.metadata["completion_reason"] = reason
        self._save(goal)
        _log.info("goal completed", extra={"goal_id": goal_id})
        return goal

    def mark_abandoned(self, goal_id: str, reason: str) -> Goal:
        """Mark a goal abandoned; ``reason`` is required and recorded."""
        if not reason or not reason.strip():
            raise ValueError("abandoning a goal requires a non-empty reason")
        goal = self._goals.get(goal_id)
        if goal is None:
            raise KeyError(f"unknown goal {goal_id!r}")
        goal.status = "abandoned"
        goal.metadata["abandon_reason"] = reason.strip()
        self._save(goal)
        _log.info("goal abandoned", extra={"goal_id": goal_id})
        return goal

    def all_goals(self) -> list[Goal]:
        """Every known goal, oldest first (convenience for snapshots/API)."""
        return sorted(self._goals.values(), key=lambda g: g.created_at)

    def __len__(self) -> int:
        return len(self._goals)
