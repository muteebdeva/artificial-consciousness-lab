"""MOCKED TESTS — goals lifecycle and planner action selection.

Skips if consciousness_lab.goals is not implemented yet.
"""

from __future__ import annotations

import pytest

from conftest import require

goals_mod = require("consciousness_lab.goals.goals")
planning_mod = require("consciousness_lab.goals.planning")

GoalManager = goals_mod.GoalManager
Planner = planning_mod.Planner


def test_goal_lifecycle_statuses():
    gm = GoalManager()
    g = gm.create("write a report", success_criteria="report exists")
    assert g.status == "pending"
    assert gm.get(g.id) is not None
    assert g.id in [x.id for x in gm.active_goals()]
    gm.track_progress(g.id, 0.5, note="halfway")
    assert gm.get(g.id).progress == pytest.approx(0.5)
    gm.mark_complete(g.id, reason="done")
    assert gm.get(g.id).status == "completed"
    assert g.id not in [x.id for x in gm.active_goals()]


def test_goal_abandon_requires_reason():
    gm = GoalManager()
    g = gm.create("doomed task")
    gm.mark_abandoned(g.id, reason="no longer relevant")
    assert gm.get(g.id).status == "abandoned"


def test_decompose_splits_on_semicolons():
    gm = GoalManager()
    g = gm.create("gather data; analyze data; write summary")
    subs = gm.decompose(g)
    assert len(subs) == 3
    assert all(s.parent_id == g.id for s in subs)


def test_decompose_fallback_creates_two_subgoals():
    gm = GoalManager()
    g = gm.create("do the thing")
    subs = gm.decompose(g)
    assert len(subs) == 2


def test_goals_persist_across_manager_instances(store):
    gm = GoalManager(store)
    g = gm.create("persistent goal")
    gm2 = GoalManager(store)
    assert gm2.get(g.id) is not None
    assert gm2.get(g.id).description == "persistent goal"


def _planner():
    budgets_mod = require("consciousness_lab.core.config")
    safety_mod = require("consciousness_lab.safety.safety")
    return Planner(budgets_mod.BudgetConfig(), safety_mod.Permissions())


def _workspace_state(items):
    ws_mod = require("consciousness_lab.workspace.workspace")
    return ws_mod.WorkspaceState(items=items, capacity=7, cycle=1, updated_at=0.0)


def test_planner_no_goal_returns_noop():
    action = _planner().next_action(None, _workspace_state([]), {})
    assert action is not None
    assert action.type == "noop"


def test_planner_completed_goal_returns_none():
    gm = GoalManager()
    g = gm.create("done task")
    gm.track_progress(g.id, 1.0)
    assert _planner().next_action(g, _workspace_state([]), {}) is None


def test_planner_default_is_respond_with_token_budget():
    gm = GoalManager()
    g = gm.create("answer the question")
    action = _planner().next_action(g, _workspace_state([]), {})
    assert action.type == "respond"
    assert action.requires_approval is False
    assert action.budget_cost.get("tokens") == 600
    assert action.payload.get("goal_id") == g.id


def test_planner_tool_needing_unlisted_permission_requires_approval():
    gm = GoalManager()
    g = gm.create("search the web")
    self_model = {"needed_tools": ["web_search"]}
    action = _planner().next_action(g, _workspace_state([]), self_model)
    assert action.type == "tool"
    assert action.requires_approval is True
