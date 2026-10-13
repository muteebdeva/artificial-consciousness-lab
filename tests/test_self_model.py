"""MOCKED TESTS — SelfModel grounding: capabilities come only from explicit
registration, never invented; tool failures become observed limitations;
performance predictions are scored honestly.

Skips if consciousness_lab.self_model.self_model is not implemented yet.
"""

from __future__ import annotations

import pytest

from conftest import require

SelfModel = require("consciousness_lab.self_model.self_model").SelfModel


def test_capabilities_only_from_registration(store):
    sm = SelfModel(store)
    snap = sm.snapshot()
    caps = snap.get("capabilities", [])
    assert caps == []  # nothing invented before registration
    sm.register_tool("calculator", "evaluates arithmetic expressions")
    sm.register_capability("summarization", source="benchmark run 2026-10-01")
    snap = sm.snapshot()
    names = [c["name"] if isinstance(c, dict) else c for c in snap["capabilities"]]
    assert "calculator" in names
    assert "summarization" in names
    assert len(names) == 2  # exactly what was registered, nothing more


def test_tool_failure_becomes_observed_limitation(store):
    sm = SelfModel(store)
    sm.register_tool("web_search", "searches the web")
    sm.update_from_tool_result({"tool": "web_search", "success": False,
                                "detail": "connection timed out"})
    snap = sm.snapshot()
    lims = snap.get("limitations", [])
    texts = [l["text"] if isinstance(l, dict) else l for l in lims]
    assert any("web_search" in t for t in texts)
    sources = [l.get("source") if isinstance(l, dict) else "" for l in lims]
    assert "observed failure" in sources


def test_mutations_record_source(store):
    sm = SelfModel(store)
    sm.register_limitation("no long-term planning beyond 20 steps", source="design doc")
    snap = sm.snapshot()
    lims = snap["limitations"]
    assert lims[0].get("source") == "design doc"


def test_performance_prediction_accuracy(store):
    sm = SelfModel(store)
    p1 = sm.predict_performance("easy task", predicted_success=True, confidence=0.9)
    p2 = sm.predict_performance("hard task", predicted_success=False, confidence=0.8)
    p3 = sm.predict_performance("medium task", predicted_success=True, confidence=0.6)
    sm.score_prediction(p1.id, actual=True)
    sm.score_prediction(p2.id, actual=False)
    sm.score_prediction(p3.id, actual=False)  # wrong
    acc = sm.accuracy()
    assert acc["n"] == 2 + 1
    assert acc["accuracy"] == pytest.approx(2 / 3)
    # NOTE: the implementation scores brier as mean((confidence - actual)^2),
    # using confidence directly as P(success) without flipping for
    # predicted_success=False. A correct "will fail" prediction with high
    # confidence therefore contributes (0.8-0)^2 instead of (0.2-0)^2.
    # The contract does not pin the probability mapping, so this test only
    # asserts structural sanity; the mapping choice is flagged in the report.
    assert acc["brier"] is not None
    assert 0.0 <= acc["brier"] <= 1.0


def test_accuracy_empty_is_none_not_crash(store):
    sm = SelfModel(store)
    acc = sm.accuracy()
    assert acc["n"] == 0
    assert acc["accuracy"] is None
    assert acc["brier"] is None


def test_task_resources_and_unresolved(store):
    sm = SelfModel(store)
    sm.set_task("write report", objectives=["draft", "review"])
    sm.update_resources({"steps_remaining": 5})
    sm.add_unresolved("is the data current?")
    snap = sm.snapshot()
    assert snap["task"] == "write report" or "write report" in str(snap)
    assert "is the data current?" in str(snap.get("unresolved", []))
    sm.resolve_unresolved("is the data current?", "checked timestamp, yes")
    snap = sm.snapshot()
    # resolved items are marked resolved (kept as history), not silently dropped
    unresolved = snap.get("unresolved", [])
    entry = next(u for u in unresolved
                 if (u.get("question") if isinstance(u, dict) else u)
                 == "is the data current?")
    assert entry.get("resolved") is True
    assert entry.get("resolution") == "checked timestamp, yes"
    open_items = [u for u in unresolved
                  if isinstance(u, dict) and not u.get("resolved")]
    assert not any("is the data current?" in str(u) for u in open_items)


def test_snapshot_is_plain_dict(store):
    sm = SelfModel(store)
    snap = sm.snapshot()
    assert isinstance(snap, dict)
    import json
    json.dumps(snap)  # must be JSON-serializable for the API
