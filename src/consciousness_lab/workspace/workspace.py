"""Global workspace: limited-capacity competitive selection and broadcast.

Implements the Global Workspace buffer from the architecture: candidate items
compete on priority (salience x relevance x novelty) for a small number of
slots; the winners are broadcast to registered consumers.

Limits:
- Selection is a plain sort on a heuristic priority score. It does not model
  attentional dynamics, inhibition between items, or coalition formation from
  Global Workspace Theory.
- Consumers run synchronously inside broadcast(); a slow consumer blocks the
  broadcast loop.
- State is in-memory only; persistence is the caller's responsibility.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

_log: logging.Logger | None = None


def _get_log() -> logging.Logger:
    """Return the lab logger, falling back to a plain stdlib logger.

    The import of core.logging_setup is deferred so this module loads even
    before that module is wired up; logging must never break the workspace.
    """
    global _log
    if _log is None:
        try:
            from consciousness_lab.core.logging_setup import setup_logging

            _log = setup_logging()
        except Exception:
            _log = logging.getLogger(__name__)
    return _log


@dataclass
class WorkspaceItem:
    """One candidate for broadcast through the global workspace.

    ``priority()`` is ``salience * relevance * novelty``, with ``None``
    relevance/novelty treated as 1.0. Ranges are validated: salience,
    relevance, novelty, confidence and attention_score must lie in [0, 1]
    (relevance/novelty/attention_score may also be None).
    """

    id: str
    content: str
    source: str  # e.g. "perception", "memory:episodic", "goal", "self_model"
    salience: float = 0.5  # 0..1
    relevance: float | None = None  # 0..1, default 1.0 when None
    novelty: float | None = None  # 0..1, default 1.0 when None
    confidence: float = 0.5  # 0..1
    attention_score: float | None = None
    timestamp: float = field(default_factory=time.time)
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("salience", "confidence"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")
        for name in ("relevance", "novelty", "attention_score"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0
            ):
                raise ValueError(
                    f"{name} must be in [0, 1] or None, got {value!r}"
                )
        if not self.id:
            raise ValueError("id must be non-empty")
        if not self.content:
            raise ValueError("content must be non-empty")

    def priority(self) -> float:
        """Return salience * (relevance or 1.0) * (novelty or 1.0).

        A heuristic ranking score, not a principled attention model; a 0.0
        relevance or novelty zeroes the priority, while None means "unknown"
        and is treated as neutral (1.0).
        """
        relevance = 1.0 if self.relevance is None else self.relevance
        novelty = 1.0 if self.novelty is None else self.novelty
        return self.salience * relevance * novelty


@dataclass
class WorkspaceState:
    """Point-in-time snapshot of the workspace's current contents."""

    items: list[WorkspaceItem]
    capacity: int
    cycle: int
    updated_at: float


class GlobalWorkspace:
    """Capacity-limited global workspace with competitive selection.

    ``select()`` ranks candidates by ``priority()`` (descending), breaks ties
    by earliest timestamp, keeps the top ``capacity`` items, stores them as the
    current state, and increments the cycle counter. ``broadcast()`` delivers
    items to every registered consumer.

    Consumer failures: a consumer that raises is caught, logged (with
    traceback), and counted in ``consumer_error_counts`` keyed by consumer
    index. The exception never propagates and never prevents the remaining
    consumers from receiving the broadcast. Each consumer gets its own shallow
    copy of the item list so one consumer cannot corrupt delivery to others.
    """

    def __init__(self, capacity: int = 7) -> None:
        """Create a workspace holding at most ``capacity`` items.

        Raises ValueError if capacity < 1.
        """
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity!r}")
        self._capacity = capacity
        self._items: list[WorkspaceItem] = []
        self._cycle = 0
        self._updated_at = time.time()
        self._consumers: list[Callable[[list[WorkspaceItem]], None]] = []
        #: consumer index -> number of exceptions raised during broadcast
        self.consumer_error_counts: dict[int, int] = {}

    @property
    def capacity(self) -> int:
        """Maximum number of items the workspace holds."""
        return self._capacity

    def select(self, candidates: list[WorkspaceItem]) -> list[WorkspaceItem]:
        """Select the top-``capacity`` candidates by priority.

        Ranking: ``priority()`` descending, ties broken by earliest
        timestamp. The winners become the current workspace state, the cycle
        counter increments, and a copy of the winners is returned. The input
        list is not mutated.
        """
        ranked = sorted(
            candidates, key=lambda item: (-item.priority(), item.timestamp)
        )
        selected = ranked[: self._capacity]
        self._items = list(selected)
        self._cycle += 1
        self._updated_at = time.time()
        return list(selected)

    def broadcast(self, items: list[WorkspaceItem]) -> None:
        """Deliver ``items`` to every registered consumer.

        A failing consumer is caught, logged, and counted; it never breaks
        the broadcast for the remaining consumers.
        """
        for index, consumer in enumerate(self._consumers):
            try:
                consumer(list(items))
            except Exception:
                self.consumer_error_counts[index] = (
                    self.consumer_error_counts.get(index, 0) + 1
                )
                _get_log().exception(
                    "workspace consumer %d raised during broadcast; continuing",
                    index,
                )

    def register_consumer(
        self, fn: Callable[[list[WorkspaceItem]], None]
    ) -> None:
        """Register a callable invoked with the broadcast item list.

        Raises ValueError if ``fn`` is not callable.
        """
        if not callable(fn):
            raise ValueError(f"consumer must be callable, got {fn!r}")
        self._consumers.append(fn)

    def snapshot(self) -> WorkspaceState:
        """Return a copy of the current workspace state."""
        return WorkspaceState(
            items=list(self._items),
            capacity=self._capacity,
            cycle=self._cycle,
            updated_at=self._updated_at,
        )

    def clear(self) -> None:
        """Empty the workspace. The cycle counter is preserved."""
        self._items = []
        self._updated_at = time.time()
