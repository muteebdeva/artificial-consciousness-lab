"""Planning: choose the next Action for the active goal within budgets.

What it does: :class:`Planner.next_action` maps ``(goal, workspace_state,
self_model)`` to a single :class:`Action` per the interface contract:
``noop`` when there is no goal, ``None`` when the goal is done or budgets
are exhausted, a gated ``tool`` action when the self-model names a needed
tool outside the permissions allowlist, otherwise a default ``respond``.

Limits:
- This is a rule-based stub planner, not a reasoning engine: it always
  returns the default ``respond`` action for an active goal and never
  synthesizes multi-step plans. The ``workspace_state`` argument is
  accepted per the contract but does not yet influence the decision.
- Budget enforcement here is planner-local accounting of tokens/tool calls
  for actions *this planner issued*. The authoritative budget lives in
  ``safety.Budget`` (owned by another agent); the loop checks that first.
- ``self_model`` is read for a ``"needed_tools"`` key (list of tool-name
  strings, as produced by ``SelfModel.snapshot()``). Any other self-model
  shape is ignored for tool gating.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from consciousness_lab.core.config import BudgetConfig

if TYPE_CHECKING:  # safety is owned by another agent; duck-typed at runtime
    from consciousness_lab.safety.safety import Permissions

try:  # core.logging_setup is owned by another agent; fall back until it lands
    from consciousness_lab.core.logging_setup import setup_logging

    _log = setup_logging()
except Exception:  # pragma: no cover - import fallback
    _log = logging.getLogger(__name__)

#: Valid action types per the interface contract.
ACTION_TYPES = ("respond", "noop", "tool", "await_approval")

#: Default token budget charged for a "respond" action (planner-local estimate).
DEFAULT_RESPOND_TOKENS = 600

#: Self-model snapshot key listing tools the system believes it needs.
NEEDED_TOOLS_KEY = "needed_tools"


@dataclass
class Action:
    """A single executable step proposed by the planner."""

    id: str
    type: str  # "respond" | "noop" | "tool" | "await_approval"
    payload: dict = field(default_factory=dict)
    requires_approval: bool = False
    budget_cost: dict = field(default_factory=dict)  # {"tokens": int, "tool_calls": int}
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.type not in ACTION_TYPES:
            raise ValueError(
                f"invalid action type {self.type!r}; expected one of {ACTION_TYPES}"
            )
        for key in ("tokens", "tool_calls"):
            if key in self.budget_cost:
                try:
                    value = int(self.budget_cost[key])
                except (TypeError, ValueError) as exc:
                    raise TypeError(f"budget_cost[{key!r}] must be an int: {exc}") from exc
                if value < 0:
                    raise ValueError(f"budget_cost[{key!r}] must be >= 0")
                self.budget_cost[key] = value


class Planner:
    """Select the next action for a goal without exceeding budgets.

    Args:
        budgets: hard limits; the planner never issues an action whose
            token cost exceeds its own remaining token estimate, and
            returns ``None`` once tokens are exhausted.
        permissions: allowlist checked when the self-model names a needed
            tool (imported function-locally to avoid import cycles with
            the safety package).
    """

    def __init__(self, budgets: BudgetConfig, permissions: Any) -> None:
        from consciousness_lab.safety.safety import Permissions  # noqa: PLC0415

        if not isinstance(budgets, BudgetConfig):
            raise TypeError(
                f"budgets must be a BudgetConfig, got {type(budgets).__name__}"
            )
        if not isinstance(permissions, Permissions):
            raise TypeError(
                "permissions must be a safety.Permissions instance, "
                f"got {type(permissions).__name__}"
            )
        self.budgets = budgets
        self._permissions = permissions
        # Planner-local spend accounting for actions this planner issued.
        self._tokens_spent = 0
        self._tool_calls_spent = 0

    @property
    def remaining(self) -> dict:
        """Planner-local remaining budget estimate.

        Tracks only what this planner has issued; the safety layer's
        ``Budget`` is authoritative for the whole loop.
        """
        return {
            "tokens": max(0, self.budgets.max_tokens - self._tokens_spent),
            "tool_calls": max(0, self.budgets.max_tool_calls - self._tool_calls_spent),
            "tokens_spent": self._tokens_spent,
            "tool_calls_spent": self._tool_calls_spent,
        }

    def _charge(self, action: Action) -> None:
        self._tokens_spent += int(action.budget_cost.get("tokens", 0))
        self._tool_calls_spent += int(action.budget_cost.get("tool_calls", 0))

    def next_action(
        self,
        goal: Any | None,
        workspace_state: Any,
        self_model: dict,
    ) -> Action | None:
        """Choose the next action for ``goal``.

        - ``goal is None`` -> ``Action("noop")`` (nothing to do).
        - Goal finished (``progress >= 1.0`` or status completed/abandoned)
          -> ``None`` (no further action).
        - Planner-local token budget exhausted -> ``None``.
        - ``self_model["needed_tools"]`` names a tool not in the permissions
          allowlist -> ``Action("tool", requires_approval=True)``.
        - Otherwise -> default ``Action("respond", budget_cost={"tokens": 600})``,
          or ``None`` if fewer than 600 tokens remain.

        ``workspace_state`` is accepted per the contract but does not yet
        influence the decision (documented limitation of this stub planner).
        """
        _ = workspace_state  # reserved for future context-aware planning

        if goal is None:
            action = Action(
                id=uuid.uuid4().hex,
                type="noop",
                payload={"reason": "no active goal"},
                budget_cost={},
            )
            return action

        progress = float(getattr(goal, "progress", 0.0) or 0.0)
        status = str(getattr(goal, "status", "active"))
        if progress >= 1.0 or status in ("completed", "abandoned"):
            return None

        if self.remaining["tokens"] <= 0:
            _log.warning("planner token budget exhausted; no action issued")
            return None

        needed_tools = []
        if isinstance(self_model, dict):
            raw_tools = self_model.get(NEEDED_TOOLS_KEY) or []
            needed_tools = [str(t) for t in raw_tools if str(t).strip()]
        for tool_name in needed_tools:
            if not self._permissions.allowed(tool_name):
                _log.info(
                    "planner gating tool action",
                    extra={"tool": tool_name, "goal_id": getattr(goal, "id", None)},
                )
                action = Action(
                    id=uuid.uuid4().hex,
                    type="tool",
                    payload={
                        "tool": tool_name,
                        "goal_id": getattr(goal, "id", None),
                        "reason": "needed tool is not in the permissions allowlist",
                    },
                    requires_approval=True,
                    budget_cost={"tool_calls": 1},
                )
                self._charge(action)
                return action

        if self.remaining["tokens"] < DEFAULT_RESPOND_TOKENS:
            _log.warning(
                "planner cannot afford default respond action",
                extra={"remaining_tokens": self.remaining["tokens"]},
            )
            return None

        action = Action(
            id=uuid.uuid4().hex,
            type="respond",
            payload={
                "goal_id": getattr(goal, "id", None),
                "instruction": f"Advance goal: {getattr(goal, 'description', '')}",
            },
            requires_approval=False,
            budget_cost={"tokens": DEFAULT_RESPOND_TOKENS},
        )
        self._charge(action)
        return action
