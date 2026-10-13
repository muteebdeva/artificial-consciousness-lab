"""Learning: memory/retrieval/strategy adaptation — NOT online ML training.

DESIGN DECISION DD-07 (prominent, do not remove):
=================================================
Nothing in this module trains, fine-tunes, or otherwise updates the weights
of any machine-learning model. There is no gradient computation, no
backpropagation, no optimizer state, and no change to any neural network
anywhere in this codebase. The word "learning" here means exactly three
measurable, non-parametric mechanisms:

1. **Strategy adaptation** — counting, per named strategy, how often it was
   used and how often it succeeded, then selecting the historically best
   strategy for future use. This is bookkeeping over observed outcomes, not
   model training.
2. **Retrieval improvement** — counting how often a stored memory record
   proved useful and up-weighting its retrieval priority accordingly.
   Records are re-ranked, never re-embedded or re-trained.
3. **Outcome observation** — recording per-cycle action/outcome pairs so
   other components (metacognition, evaluation) can measure calibration.

Any claim that this module performs "online learning" in the ML sense
would be false. See docs/design_decisions.md DD-07.

What it does: maintains strategy success statistics and per-record
usefulness counts, in memory and (when a ``SqliteStore`` is given) in the
``strategy_stats`` / ``record_usefulness`` tables, and uses them to select
strategies and weight retrieval.

Limits:
- Selection is by raw observed success rate with no confidence intervals,
  no exploration, and no context modeling: a strategy that succeeded once
  outranks one that succeeded 9/10 times. Ties and unseen strategies fall
  back to candidate order.
- Statistics are global, not conditioned on task type or context.
- Persistence is last-write-wins per strategy/record; concurrent writers
  can interleave counts.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # persistence is owned by another agent; duck-typed at runtime
    from consciousness_lab.persistence.sqlite_store import SqliteStore

try:  # core.logging_setup is owned by another agent; fall back until it lands
    from consciousness_lab.core.logging_setup import setup_logging

    _log = setup_logging()
except Exception:  # pragma: no cover - import fallback
    _log = logging.getLogger(__name__)

#: DDL for per-strategy outcome counts.
STRATEGY_STATS_DDL = """
CREATE TABLE IF NOT EXISTS strategy_stats (
    strategy TEXT PRIMARY KEY,
    uses INTEGER NOT NULL DEFAULT 0,
    successes INTEGER NOT NULL DEFAULT 0
)
"""

#: DDL for per-record usefulness counts.
RECORD_USEFULNESS_DDL = """
CREATE TABLE IF NOT EXISTS record_usefulness (
    record_id TEXT PRIMARY KEY,
    uses INTEGER NOT NULL DEFAULT 0
)
"""

#: Cap on the retrieval weight multiplier (see retrieval_weight).
MAX_RETRIEVAL_WEIGHT = 2.0

#: Weight increment per recorded useful use.
RETRIEVAL_WEIGHT_STEP = 0.1


class Learning:
    """Memory/retrieval/strategy adaptation. NOT online ML training (DD-07).

    Tracks which named strategies succeed and which stored records prove
    useful, and exposes that history for strategy selection and retrieval
    weighting. With ``store=None`` all state is in-memory only; with a
    store, counts persist across restarts in ``strategy_stats`` and
    ``record_usefulness``.
    """

    def __init__(self, store: "SqliteStore | None" = None) -> None:
        self._store = store
        # strategy -> {"uses": int, "successes": int}
        self._strategies: dict[str, dict[str, int]] = {}
        # record_id -> useful-use count
        self._usefulness: dict[str, int] = {}
        if store is not None:
            store.ensure_table("strategy_stats", STRATEGY_STATS_DDL)
            store.ensure_table("record_usefulness", RECORD_USEFULNESS_DDL)
            for row in store.execute_read(
                "SELECT strategy, uses, successes FROM strategy_stats"
            ):
                self._strategies[row["strategy"]] = {
                    "uses": int(row["uses"]),
                    "successes": int(row["successes"]),
                }
            for row in store.execute_read(
                "SELECT record_id, uses FROM record_usefulness"
            ):
                self._usefulness[row["record_id"]] = int(row["uses"])
            _log.info(
                "learning loaded persisted stats",
                extra={
                    "strategies": len(self._strategies),
                    "records": len(self._usefulness),
                },
            )

    # -- persistence helpers -------------------------------------------------
    def _save_strategy(self, strategy: str) -> None:
        if self._store is None:
            return
        stats = self._strategies[strategy]
        self._store.execute_write(
            "INSERT INTO strategy_stats (strategy, uses, successes) VALUES (?, ?, ?) "
            "ON CONFLICT(strategy) DO UPDATE SET uses=excluded.uses, successes=excluded.successes",
            (strategy, stats["uses"], stats["successes"]),
        )

    def _save_usefulness(self, record_id: str) -> None:
        if self._store is None:
            return
        self._store.execute_write(
            "INSERT INTO record_usefulness (record_id, uses) VALUES (?, ?) "
            "ON CONFLICT(record_id) DO UPDATE SET uses=excluded.uses",
            (record_id, self._usefulness[record_id]),
        )

    # -- strategy adaptation -------------------------------------------------
    def record_strategy_use(
        self, strategy: str, success: bool, context: str = ""
    ) -> None:
        """Record one observed use of ``strategy`` and whether it succeeded.

        ``context`` is accepted per the interface contract and logged at
        debug level, but statistics are intentionally global (not
        conditioned on context) in this version.
        """
        if not strategy or not str(strategy).strip():
            raise ValueError("strategy name must be non-empty")
        name = str(strategy).strip()
        stats = self._strategies.setdefault(name, {"uses": 0, "successes": 0})
        stats["uses"] += 1
        if success:
            stats["successes"] += 1
        self._save_strategy(name)
        _log.debug(
            "strategy use recorded",
            extra={"strategy": name, "success": bool(success), "context": context},
        )

    def select_strategy(self, candidates: list[str]) -> str:
        """Return the candidate with the highest observed success rate.

        Ties resolve to the earliest candidate in the list. Candidates
        with no recorded uses are treated as unknown and lose to any
        candidate with observed data; if none have data, the first
        candidate is returned. Raises ``ValueError`` on an empty list.
        """
        if not candidates:
            raise ValueError("select_strategy requires at least one candidate")
        best: str | None = None
        best_rate: float | None = None
        for candidate in candidates:
            stats = self._strategies.get(str(candidate))
            rate = (
                stats["successes"] / stats["uses"]
                if stats and stats["uses"] > 0
                else None
            )
            if rate is not None and (best_rate is None or rate > best_rate):
                best, best_rate = str(candidate), rate
        return best if best is not None else str(candidates[0])

    def strategy_stats(self) -> dict:
        """Per-strategy counts and success rates.

        Returns ``{strategy: {"uses": int, "successes": int,
        "success_rate": float | None}}``; ``success_rate`` is ``None`` when
        the strategy has no recorded uses.
        """
        return {
            name: {
                "uses": stats["uses"],
                "successes": stats["successes"],
                "success_rate": (
                    stats["successes"] / stats["uses"] if stats["uses"] > 0 else None
                ),
            }
            for name, stats in self._strategies.items()
        }

    # -- retrieval improvement ------------------------------------------------
    def note_useful_record(self, record_id: str) -> None:
        """Increment the usefulness count for a stored memory record."""
        if not record_id or not str(record_id).strip():
            raise ValueError("record_id must be non-empty")
        rid = str(record_id).strip()
        self._usefulness[rid] = self._usefulness.get(rid, 0) + 1
        self._save_usefulness(rid)

    def retrieval_weight(self, record_id: str) -> float:
        """Retrieval priority multiplier: ``1.0 + 0.1 * uses``, capped at 2.0.

        Unknown records weigh 1.0. This re-ranks records; it does not
        modify record content or any model.
        """
        uses = self._usefulness.get(str(record_id), 0)
        return min(MAX_RETRIEVAL_WEIGHT, 1.0 + RETRIEVAL_WEIGHT_STEP * uses)

    # -- outcome observation ---------------------------------------------------
    def observe_cycle(self, info: dict) -> None:
        """Record a cycle outcome for strategy statistics.

        ``info`` keys: ``action_type`` (str), ``success`` (bool | None),
        ``strategy`` (str | None). A cycle with no named strategy, or with
        ``success=None`` (unknown outcome), is ignored: only observed
        (strategy, success) pairs update statistics.
        """
        if not isinstance(info, dict):
            raise TypeError(f"observe_cycle info must be a dict, got {type(info).__name__}")
        strategy = info.get("strategy")
        success = info.get("success")
        if strategy and success is not None:
            self.record_strategy_use(str(strategy), bool(success))
        _log.debug(
            "cycle observed",
            extra={
                "action_type": info.get("action_type"),
                "success": success,
                "strategy": strategy,
            },
        )
