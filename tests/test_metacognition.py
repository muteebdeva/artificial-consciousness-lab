"""MOCKED TESTS — metacognition: confidence heuristics, contradiction
detection, error detection, and belief revision.

Skips if consciousness_lab.metacognition is not implemented yet.
"""

from __future__ import annotations

import pytest

from conftest import require

meta_conf = require("consciousness_lab.metacognition.confidence")
meta_contra = require("consciousness_lab.metacognition.contradiction")
meta_error = require("consciousness_lab.metacognition.error_detection")
meta_belief = require("consciousness_lab.metacognition.belief_revision")

ConfidenceEstimator = meta_conf.ConfidenceEstimator
ContradictionDetector = meta_contra.ContradictionDetector
ErrorDetector = meta_error.ErrorDetector
BeliefRevision = meta_belief.BeliefRevision

records_mod = require("consciousness_lab.memory.records")
MemoryRecord = records_mod.MemoryRecord


def _record(content: str, confidence: float = 0.5) -> MemoryRecord:
    import time
    return MemoryRecord(id=f"r-{abs(hash(content)) % 10**8}", kind="semantic",
                        content=content, provenance="test", confidence=confidence,
                        timestamp=time.time(), source_type="inferred")


def test_confidence_verified_sources_raise_estimate():
    est = ConfidenceEstimator()
    high = est.estimate({"sources": ["verified", "verified"], "n_evidence": 2,
                         "agreement": 1.0, "recency": 1.0, "base_rate": 0.5})
    low = est.estimate({"sources": ["model_hypothesis"], "n_evidence": 1,
                        "agreement": 1.0, "recency": 1.0, "base_rate": 0.5})
    assert high.value > low.value
    assert high.basis and isinstance(high.basis, str)


def test_confidence_clamped_to_range():
    est = ConfidenceEstimator()
    for ctx in (
        {"sources": ["verified"] * 20, "n_evidence": 50, "agreement": 1.0,
         "recency": 1.0, "base_rate": 0.99},
        {"sources": ["model_hypothesis"] * 10, "n_evidence": 0, "agreement": 0.0,
         "recency": 0.0, "base_rate": 0.01},
    ):
        c = est.estimate(ctx)
        assert 0.01 <= c.value <= 0.99


def test_confidence_agreement_scales_estimate():
    est = ConfidenceEstimator()
    agree = est.estimate({"sources": ["verified"], "n_evidence": 2,
                          "agreement": 1.0, "recency": 1.0})
    disagree = est.estimate({"sources": ["verified"], "n_evidence": 2,
                             "agreement": 0.0, "recency": 1.0})
    assert agree.value >= disagree.value


def test_contradiction_detected_on_negation():
    det = ContradictionDetector()
    existing = [_record("The Eiffel Tower is in Paris", confidence=0.9)]
    hits = det.check("The Eiffel Tower is not in Paris", existing)
    assert len(hits) >= 1
    assert 0.0 <= hits[0].severity <= 1.0
    assert hits[0].description


def test_contradiction_detected_on_numeric_mismatch():
    det = ContradictionDetector()
    c = det.check_pair("Water boils at 100 degrees Celsius",
                       "Water boils at 50 degrees Celsius")
    assert c is not None


def test_no_contradiction_for_compatible_statements():
    det = ContradictionDetector()
    c = det.check_pair("The Eiffel Tower is in Paris",
                       "Paris is the capital of France")
    assert c is None


def test_check_pair_returns_none_for_unrelated():
    det = ContradictionDetector()
    assert det.check_pair("Cats are mammals", "Quantum computing uses qubits") is None


def test_error_detector_flags_miscalibration():
    det = ErrorDetector()
    outcome_mod = require("consciousness_lab.memory.outcome")
    Prediction = outcome_mod.Prediction
    import time
    pred = Prediction(id="p1", content="x", confidence=0.95, timestamp=time.time())
    report = det.evaluate_prediction(pred, observed=False)
    assert report.miscalibrated is True  # |0.95 - 0| > 0.5
    assert report.correct is False
    assert report.brier_contribution == pytest.approx(0.95 ** 2)
    ok_pred = Prediction(id="p2", content="y", confidence=0.9, timestamp=time.time())
    ok_report = det.evaluate_prediction(ok_pred, observed=True)
    assert ok_report.miscalibrated is False
    assert ok_report.correct is True


def test_scan_trace_returns_failed_stages():
    det = ErrorDetector()
    trace = [
        {"stage": "perception", "ok": True, "detail": "fine"},
        {"stage": "planning", "ok": False, "detail": "no goal"},
        {"stage": "respond", "ok": False, "detail": "provider timeout"},
    ]
    failed = det.scan_trace(trace)
    assert [f["stage"] for f in failed] == ["planning", "respond"]


def test_belief_revision_weighted_update():
    rev = BeliefRevision()
    rec = _record("old belief", confidence=0.6)
    updated = []

    result = rev.revise(rec, evidence="new study", evidence_confidence=0.5,
                        store_update=updated.append)
    # documented heuristic: new = (old*w_old + ev*w_ev) / (w_old + w_ev)
    # with w_old=1, w_ev=evidence_confidence, ev=evidence_confidence
    # (the only numeric evidence the signature provides)
    expected = (0.6 * 1 + 0.5 * 0.5) / (1 + 0.5)
    assert result.old_confidence == pytest.approx(0.6)
    assert result.new_confidence == pytest.approx(expected)
    assert result.evidence == "new study"
    assert len(result.provenance_chain) >= 1
    # new version appended via callback with supersedes link, never silent
    assert len(updated) == 1
    assert updated[0].supersedes == rec.id
    assert result.record_id == updated[0].id
