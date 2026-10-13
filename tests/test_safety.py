"""MOCKED TESTS — safety: budgets, permissions, append-only audit log, and the
safety controller's approval/budget enforcement.

Skips if consciousness_lab.safety.safety is not implemented yet.
"""

from __future__ import annotations

import time

import pytest

from conftest import require

safety_mod = require("consciousness_lab.safety.safety")
errors_mod = require("consciousness_lab.core.errors")

Budget = safety_mod.Budget
Permissions = safety_mod.Permissions
AuditLog = safety_mod.AuditLog
SafetyController = safety_mod.SafetyController
BudgetExceeded = errors_mod.BudgetExceeded
PermissionDenied = errors_mod.PermissionDenied
SafetyError = errors_mod.SafetyError


def _budget(**over):
    from consciousness_lab.core.config import BudgetConfig
    kwargs = {"max_steps": 3, "max_tokens": 100, "max_wall_time_s": 60.0,
              "max_tool_calls": 2}
    kwargs.update(over)
    return Budget.from_config(BudgetConfig(**kwargs))


def test_budget_consume_step_raises_at_limit():
    b = _budget()
    b.consume_step(); b.consume_step(); b.consume_step()
    with pytest.raises(BudgetExceeded):
        b.consume_step()
    assert b.exhausted() is True


def test_budget_consume_tokens_raises_at_limit():
    b = _budget()
    b.consume_tokens(100)
    with pytest.raises(BudgetExceeded):
        b.consume_tokens(1)


def test_budget_consume_tool_calls_raises_at_limit():
    b = _budget()
    b.consume_tool_call(); b.consume_tool_call()
    with pytest.raises(BudgetExceeded):
        b.consume_tool_call()


def test_budget_check_time_raises_when_expired():
    b = _budget(max_wall_time_s=0.01)
    b.started_at = time.time() - 10  # pretend we started long ago
    with pytest.raises(BudgetExceeded):
        b.check_time()


def test_budget_remaining_reports_counts():
    b = _budget()
    b.consume_step()
    b.consume_tokens(10)
    rem = b.remaining()
    assert rem["steps"] == 2 or rem.get("max_steps", 3) - rem.get("steps_used", 1) == 2


def test_permissions_default_deny():
    p = Permissions()
    assert p.allowed("web_search") is False


def test_permissions_grant_requires_authorization():
    p = Permissions()
    with pytest.raises(SafetyError):
        p.grant("web_search")  # system cannot self-authorize
    p.grant("web_search", authorized=True)
    assert p.allowed("web_search") is True


def test_permissions_revoke_requires_authorization():
    p = Permissions()
    p.grant("web_search", authorized=True)
    with pytest.raises(SafetyError):
        p.revoke("web_search")
    p.revoke("web_search", authorized=True)
    assert p.allowed("web_search") is False


def test_audit_log_append_only(tmp_path):
    audit = AuditLog(path=str(tmp_path / "audit.jsonl"))
    audit.log("cycle", {"n": 1})
    audit.log("memory_write", {"id": "abc"})
    entries = audit.read(limit=10)
    assert len(entries) == 2
    assert {e["event"] for e in entries} == {"cycle", "memory_write"}
    # no update/delete API surface
    assert not hasattr(audit, "update")
    assert not hasattr(audit, "delete")


def _controller(tmp_path):
    audit = AuditLog(path=str(tmp_path / "audit.jsonl"))
    return SafetyController(budget=_budget(), permissions=Permissions(),
                            audit=audit, autonomous_enabled=True)


def _action(action_type="respond", requires_approval=False):
    planning_mod = require("consciousness_lab.goals.planning")
    return planning_mod.Action(id="a1", type=action_type,
                               requires_approval=requires_approval,
                               budget_cost={"tokens": 10})


def test_controller_rejects_approval_actions_always(tmp_path):
    c = _controller(tmp_path)
    with pytest.raises(PermissionDenied):
        c.check_action(_action(requires_approval=True))


def test_controller_rejects_unlisted_tool(tmp_path):
    c = _controller(tmp_path)
    with pytest.raises(PermissionDenied):
        c.check_action(_action(action_type="tool"))


def test_controller_allows_listed_tool(tmp_path):
    c = _controller(tmp_path)
    c.permissions.grant("calculator", authorized=True)
    action = _action(action_type="tool")
    action.payload["tool"] = "calculator"
    c.check_action(action)  # must not raise


def test_controller_consumes_budget_for_action(tmp_path):
    c = _controller(tmp_path)
    for _ in range(10):
        c.check_action(_action())  # 10 tokens each; budget is 100
    with pytest.raises(BudgetExceeded):
        c.check_action(_action())


def test_controller_stop_halts_run(tmp_path):
    c = _controller(tmp_path)
    assert c.stopped is False
    c.stop("run-1")
    assert c.stopped is True


def test_disable_autonomous_requires_authorization(tmp_path):
    c = _controller(tmp_path)
    with pytest.raises(SafetyError):
        c.disable_autonomous()
    c.disable_autonomous(authorized=True)
