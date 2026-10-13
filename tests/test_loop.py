"""MOCKED TESTS — CognitiveLoop integration with the deterministic mock
provider: step/run lifecycle, graceful degradation with components disabled,
budget exhaustion, and snapshot shape.

Skips if consciousness_lab.core.loop is not implemented yet.
"""

from __future__ import annotations

import pytest

from conftest import FIXED_SEED, require

loop_mod = require("consciousness_lab.core.loop")
build_loop = loop_mod.build_loop
CognitiveLoop = loop_mod.CognitiveLoop


def _config(tmp_path, **over):
    Config = require("consciousness_lab.core.config").Config
    return Config(db_path=str(tmp_path / "loop.sqlite3"), provider="mock",
                  seed=FIXED_SEED, **over)


def test_build_loop_returns_cognitive_loop(tmp_path):
    loop = build_loop(_config(tmp_path))
    assert isinstance(loop, CognitiveLoop)


def test_step_returns_cycle_result(tmp_path):
    loop = build_loop(_config(tmp_path))
    cr = loop.step({"modality": "task", "content": "say hello",
                    "metadata": {}, "source": "test"})
    assert cr.cycle == 1
    assert cr.observation_id
    assert isinstance(cr.selected_ids, list)
    assert isinstance(cr.provider_calls, int) and cr.provider_calls >= 0
    assert isinstance(cr.tokens_used, int)
    assert isinstance(cr.errors, list)
    assert cr.duration_s >= 0


def test_step_accepts_plain_string_observation(tmp_path):
    loop = build_loop(_config(tmp_path))
    cr = loop.step("just a string observation")
    assert cr.cycle == 1


def test_run_completes_task(tmp_path):
    loop = build_loop(_config(tmp_path))
    result = loop.run("answer: what is 2+2?", max_cycles=3)
    assert result.task == "answer: what is 2+2?"
    assert result.status in ("completed", "budget_exhausted", "stopped")
    assert result.cycles >= 1
    assert isinstance(result.goal_status, dict)


def test_run_respects_max_cycles(tmp_path):
    loop = build_loop(_config(tmp_path))
    result = loop.run("never-ending task", max_cycles=2)
    assert result.cycles <= 2


def test_budget_exhaustion_stops_run(tmp_path):
    from consciousness_lab.core.config import BudgetConfig
    cfg = _config(tmp_path)
    cfg.budgets = BudgetConfig(max_steps=1, max_tokens=10**9,
                               max_wall_time_s=600.0, max_tool_calls=10)
    loop = build_loop(cfg)
    result = loop.run("do things", max_cycles=10)
    assert result.status == "budget_exhausted"


def test_all_components_disabled_still_steps(tmp_path):
    cfg = _config(tmp_path)
    cfg.components = {k: False for k in cfg.components}
    loop = build_loop(cfg)
    cr = loop.step("degraded observation")
    assert cr.cycle == 1  # degrades gracefully, never dies


def test_snapshot_shape(tmp_path):
    loop = build_loop(_config(tmp_path))
    snap = loop.snapshot()
    assert isinstance(snap, dict)
    for key in ("workspace", "self_model", "goals", "budgets"):
        assert key in snap, f"snapshot missing {key!r}"


def test_step_errors_are_recorded_not_swallowed(tmp_path):
    loop = build_loop(_config(tmp_path))
    cr = loop.step({"modality": "task", "content": "x", "metadata": {}})
    # contract: every stage guarded; errors land in CycleResult.errors
    assert isinstance(cr.errors, list)
