"""MOCKED TESTS — Learning: strategy selection by observed success rate,
usefulness-weighted retrieval, and cycle observation.

Skips if consciousness_lab.learning.learning is not implemented yet. Note:
Learning is memory/retrieval adaptation, NOT online ML training.
"""

from __future__ import annotations

import pytest

from conftest import require

Learning = require("consciousness_lab.learning.learning").Learning


def test_select_strategy_picks_highest_success_rate():
    lg = Learning()
    for _ in range(3):
        lg.record_strategy_use("careful", success=True, context="qa")
    lg.record_strategy_use("careful", success=False, context="qa")  # 0.75
    lg.record_strategy_use("fast", success=True, context="qa")      # 1.0 (1 sample)
    lg.record_strategy_use("fast", success=True, context="qa")      # 1.0
    assert lg.select_strategy(["careful", "fast"]) == "fast"


def test_select_strategy_tie_goes_to_first():
    lg = Learning()
    lg.record_strategy_use("a", success=True)
    lg.record_strategy_use("b", success=True)
    assert lg.select_strategy(["a", "b"]) == "a"


def test_select_strategy_unknown_defaults_to_first_candidate():
    lg = Learning()
    assert lg.select_strategy(["novel", "other"]) == "novel"


def test_retrieval_weight_grows_with_use_and_caps():
    lg = Learning()
    assert lg.retrieval_weight("r1") == pytest.approx(1.0)
    lg.note_useful_record("r1")
    lg.note_useful_record("r1")
    assert lg.retrieval_weight("r1") == pytest.approx(1.2)
    for _ in range(100):
        lg.note_useful_record("r1")
    assert lg.retrieval_weight("r1") == pytest.approx(2.0)  # capped


def test_observe_cycle_feeds_strategy_stats():
    lg = Learning()
    lg.observe_cycle({"action_type": "respond", "success": True, "strategy": "careful"})
    lg.observe_cycle({"action_type": "respond", "success": False, "strategy": "careful"})
    stats = lg.strategy_stats()
    assert "careful" in stats
    assert stats["careful"]["uses"] == 2
    assert stats["careful"]["successes"] == 1


def test_strategy_stats_empty_initially():
    lg = Learning()
    assert lg.strategy_stats() == {}
