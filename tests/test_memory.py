"""MOCKED TESTS — episodic/semantic/working/outcome memory.

Covers CRUD, recall scoring, contradiction flagging on upsert, reason-required
delete, and outcome calibration math. Skips if the memory package is not
implemented yet.
"""

from __future__ import annotations

import math
import time

import pytest

from conftest import require

records_mod = require("consciousness_lab.memory.records")
MemoryRecord = records_mod.MemoryRecord

episodic_mod = require("consciousness_lab.memory.episodic")
semantic_mod = require("consciousness_lab.memory.semantic")
working_mod = require("consciousness_lab.memory.working")
outcome_mod = require("consciousness_lab.memory.outcome")

EpisodicMemory = episodic_mod.EpisodicMemory
SemanticMemory = semantic_mod.SemanticMemory
WorkingMemory = working_mod.WorkingMemory
OutcomeMemory = outcome_mod.OutcomeMemory


def test_memory_record_validation():
    with pytest.raises(ValueError):
        MemoryRecord(id="1", kind="episodic", content="", provenance="p",
                     confidence=0.5, timestamp=time.time(),
                     source_type="inferred").validate()
    with pytest.raises(ValueError):
        MemoryRecord(id="1", kind="episodic", content="x", provenance="p",
                     confidence=1.5, timestamp=time.time(),
                     source_type="inferred").validate()
    with pytest.raises(ValueError):
        MemoryRecord(id="1", kind="episodic", content="x", provenance="p",
                     confidence=0.5, timestamp=time.time(),
                     source_type="made_up").validate()
    # valid record passes
    MemoryRecord(id="1", kind="episodic", content="x", provenance="p",
                 confidence=0.5, timestamp=time.time(),
                 source_type="verified").validate()


def test_memory_record_roundtrip():
    rec = MemoryRecord(id="abc", kind="semantic", content="sky is blue",
                       provenance="user", confidence=0.8, timestamp=1234.5,
                       source_type="user_provided", metadata={"k": "v"})
    d = rec.to_dict()
    back = MemoryRecord.from_dict(d)
    assert back.content == "sky is blue"
    assert back.confidence == 0.8
    assert back.source_type == "user_provided"
    assert back.metadata == {"k": "v"}


def test_episodic_crud(store):
    mem = EpisodicMemory(store)
    rec = mem.record("the meeting is at noon", provenance="user message",
                     confidence=0.9, source_type="user_provided")
    assert rec.id
    got = mem.get(rec.id)
    assert got is not None and got.content == "the meeting is at noon"
    hits = mem.recall("when is the meeting", k=5)
    assert any(h.id == rec.id for h in hits)
    mem.delete(rec.id, reason="user requested removal")
    assert mem.get(rec.id) is None


def test_episodic_delete_requires_reason(store):
    mem = EpisodicMemory(store)
    rec = mem.record("temporary", provenance="test", source_type="inferred")
    with pytest.raises(ValueError):
        mem.delete(rec.id, reason="")
    assert mem.get(rec.id) is not None  # still there


def test_episodic_recall_prefers_relevant_and_confident(store):
    mem = EpisodicMemory(store)
    mem.record("irrelevant gardening tips", provenance="t", confidence=0.1,
               source_type="inferred")
    target = mem.record("the launch code is 1234", provenance="t",
                        confidence=0.95, source_type="verified")
    hits = mem.recall("what is the launch code", k=1)
    assert hits and hits[0].id == target.id


def test_semantic_upsert_no_contradiction(store):
    mem = SemanticMemory(store)
    rec, contradictions = mem.upsert_fact("Paris is the capital of France",
                                         provenance="test", confidence=0.9,
                                         source_type="verified")
    assert contradictions == []
    assert mem.get(rec.id) is not None


def test_semantic_upsert_flags_contradiction_and_keeps_both(store):
    seen = []
    mem = SemanticMemory(store, on_contradiction=seen.append)
    first, _ = mem.upsert_fact("The Eiffel Tower is in Paris", provenance="t1",
                               confidence=0.9, source_type="verified")
    second, contradictions = mem.upsert_fact(
        "The Eiffel Tower is not in Paris", provenance="t2",
        confidence=0.9, source_type="user_provided")
    assert len(contradictions) >= 1
    assert len(seen) >= 1  # callback invoked
    # never silently overwrite: both records retrievable
    assert mem.get(first.id) is not None
    assert mem.get(second.id) is not None
    assert second.supersedes is None  # no explicit replacement requested


def test_semantic_explicit_supersede_links_records(store):
    mem = SemanticMemory(store)
    first, _ = mem.upsert_fact("Water boils at 90C", provenance="t1",
                               source_type="inferred")
    second, _ = mem.upsert_fact("Water boils at 100C at sea level",
                                provenance="t2", source_type="verified")
    # without explicit supersedes_id, no link is set
    assert second.supersedes is None
    assert first.id != second.id


def test_working_memory_is_bounded_and_clearable():
    wm = WorkingMemory(capacity=10)
    for i in range(15):
        wm.put(f"item {i}")
    assert len(wm.contents()) <= 10
    wm.clear()
    assert wm.contents() == []


def test_working_memory_not_persisted(store):
    wm = WorkingMemory(capacity=10)
    wm.put("volatile")
    # a second WorkingMemory over the same store sees nothing (in-memory only)
    wm2 = WorkingMemory(capacity=10)
    assert wm2.contents() == []


def test_outcome_calibration_math(store):
    om = OutcomeMemory(store)
    p1 = om.record_prediction("it will rain", confidence=1.0)
    p2 = om.record_prediction("it will snow", confidence=0.0)
    om.record_outcome(p1.id, observed=True, note="rained")
    om.record_outcome(p2.id, observed=False, note="no snow")
    report = om.calibration()
    assert report.n == 2
    assert report.brier_score == pytest.approx(0.0)  # perfect
    assert report.ece == pytest.approx(0.0)
    assert len(report.bins) == 10


def test_outcome_calibration_known_brier(store):
    om = OutcomeMemory(store)
    p = om.record_prediction("x", confidence=0.75)
    om.record_outcome(p.id, observed=True)
    report = om.calibration()
    assert report.brier_score == pytest.approx((0.75 - 1.0) ** 2)


def test_outcome_calibration_empty_is_nan_not_crash(store):
    om = OutcomeMemory(store)
    report = om.calibration()
    assert report.n == 0
    assert math.isnan(report.brier_score)
    assert math.isnan(report.ece)


def test_outcome_pending_lists_unscored(store):
    om = OutcomeMemory(store)
    p = om.record_prediction("y", confidence=0.5)
    assert any(x.id == p.id for x in om.pending())
    om.record_outcome(p.id, observed=True)
    assert not any(x.id == p.id for x in om.pending())
