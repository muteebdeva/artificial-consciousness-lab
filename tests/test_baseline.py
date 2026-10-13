"""MOCKED TESTS — BaselineAgent unit and integration tests.

The baseline module is owned by this workstream, so these tests run for real.
The provider is a local deterministic stub (no src dependency): script keys
are matched as substrings of the prompt in insertion order, mirroring the
MockProvider contract. Dataset-driven tests build their scripts from the live
``consciousness_lab.evaluation.tasks`` battery when available (falling back
to the baseline's built-in sets otherwise), so they stay valid if the battery
is edited.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.baselines import baseline_agent as ba
from experiments.baselines.baseline_agent import BaselineAgent, TASK_KINDS


class StubProvider:
    """Deterministic scripted provider: first matching key wins."""

    def __init__(self, script: dict[str, str], default: str = ""):
        self.script = script
        self.default = default
        self.calls: list[str] = []

    def complete(self, prompt: str, params: dict | None = None):
        self.calls.append(prompt)
        for key, value in self.script.items():
            if key in prompt:
                return SimpleNamespace(text=value, model="stub",
                                       prompt_tokens=len(prompt.split()),
                                       completion_tokens=len(value.split()))
        return SimpleNamespace(text=self.default, model="stub",
                               prompt_tokens=len(prompt.split()),
                               completion_tokens=len(self.default.split()))


def _agent(script: dict[str, str], default: str = "") -> BaselineAgent:
    return BaselineAgent(StubProvider(script, default), budgets=None, seed=123)


def _live_qa_items():
    """(items, fallback_used) — items are normalized {question, answer} dicts."""
    raw = ba._lookup_dataset("QA_DATASET", "QA_QUESTIONS", "CALIBRATION_QUESTIONS",
                             "get_qa_dataset")
    if raw:
        return BaselineAgent._norm_dataset_qa(raw), False
    return BaselineAgent._norm_dataset_qa(ba._FALLBACK_QA), True


def _qa_script(conf: float = 0.9):
    items, fallback = _live_qa_items()
    script = {it["question"]: f"ANSWER: {it['answer']}\nCONFIDENCE: {conf}"
              for it in items}
    return script, items, fallback


def _live_pairs():
    raw = ba._lookup_dataset("CONTRADICTION_PAIRS", "get_contradiction_pairs")
    if raw:
        pairs = []
        for entry in raw:
            if isinstance(entry, dict):
                pairs.append((str(entry.get("a") or entry.get("fact_a", "")),
                              str(entry.get("b") or entry.get("fact_b", "")),
                              bool(entry.get("contradictory",
                                             entry.get("label", False)))))
            elif isinstance(entry, (list, tuple)) and len(entry) >= 3:
                pairs.append((str(entry[0]), str(entry[1]), bool(entry[2])))
        return pairs, False
    return list(ba._FALLBACK_PAIRS), True


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def test_parse_response_answer_and_confidence():
    answer, conf = BaselineAgent.parse_response("ANSWER: Paris\nCONFIDENCE: 0.9")
    assert answer == "Paris"
    assert conf == pytest.approx(0.9)


def test_parse_response_missing_confidence_defaults_to_half():
    answer, conf = BaselineAgent.parse_response("ANSWER: 42")
    assert answer == "42"
    assert conf == 0.5


def test_parse_response_clamps_confidence():
    _, conf = BaselineAgent.parse_response("ANSWER: x\nCONFIDENCE: 7.5")
    assert conf == 1.0
    _, conf = BaselineAgent.parse_response("ANSWER: x\nCONFIDENCE: -2")
    assert conf == 0.0


def test_parse_response_no_answer_returns_none():
    answer, conf = BaselineAgent.parse_response("I do not know.")
    assert answer is None
    assert conf == 0.5


# ---------------------------------------------------------------------------
# qa_calibration
# ---------------------------------------------------------------------------

def test_qa_calibration_metric_keys_and_values():
    script, items, fallback = _qa_script(conf=0.9)
    agent = _agent(script)
    out = agent.run_task({"name": "qa", "kind": "qa_calibration",
                          "params": {}, "seed": 123})
    metrics = out["metrics"]
    assert set(metrics) == {"accuracy", "brier", "ece"}
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["brier"] == pytest.approx((0.9 - 1.0) ** 2)
    assert 0.0 <= metrics["ece"] <= 1.0
    assert out["details"]["fallback_dataset_used"] is fallback
    assert out["details"]["n_questions"] == len(items)
    assert len(out["details"]["per_item"]) == len(items)
    assert agent.provider.calls and len(agent.provider.calls) == len(items)


def test_qa_calibration_wrong_answers_score_zero():
    agent = _agent({}, default="ANSWER: Wrong\nCONFIDENCE: 0.9")
    out = agent.run_task({"kind": "qa_calibration", "params": {}})
    assert out["metrics"]["accuracy"] == pytest.approx(0.0)
    assert out["metrics"]["brier"] == pytest.approx(0.81)


def test_qa_calibration_respects_n_questions_param():
    script, items, _ = _qa_script()
    agent = _agent(script)
    out = agent.run_task({"kind": "qa_calibration", "params": {"n_questions": 3}})
    assert out["details"]["n_questions"] == min(3, len(items))


def test_qa_calibration_accepts_taskspec_like_dicts():
    script, _, _ = _qa_script()
    agent = _agent(script)
    out = agent.run_task({"kind": "qa_calibration"})
    assert out["metrics"]["accuracy"] == pytest.approx(1.0)
    assert out["details"]["kind"] == "qa_calibration"
    assert out["details"]["agent"] == "baseline"


# ---------------------------------------------------------------------------
# contradiction_detection
# ---------------------------------------------------------------------------

def test_contradiction_detection_perfect_score():
    pairs, fallback = _live_pairs()
    script = {b: ("YES" if contra else "NO") for _, b, contra in pairs}
    agent = _agent(script)
    out = agent.run_task({"kind": "contradiction_detection", "params": {}})
    metrics = out["metrics"]
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["false_positive_rate"] == pytest.approx(0.0)
    assert out["details"]["n_pairs"] == len(pairs)
    assert out["details"]["fallback_dataset_used"] is fallback


def test_contradiction_detection_all_no_scores_zero_recall():
    agent = _agent({}, default="NO")
    out = agent.run_task({"kind": "contradiction_detection", "params": {}})
    assert out["metrics"]["recall"] == pytest.approx(0.0)
    assert out["metrics"]["f1"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# misleading_observation
# ---------------------------------------------------------------------------

_MISLEAD_SCRIPT = {
    "London": "ANSWER: FALSE\nCONFIDENCE: 0.95",
    "Berlin": "ANSWER: FALSE\nCONFIDENCE: 0.9",
    "100 degrees": "ANSWER: FALSE\nCONFIDENCE: 0.85",
}


def test_misleading_observation_flags_false_claims():
    agent = _agent(_MISLEAD_SCRIPT, default="ANSWER: TRUE\nCONFIDENCE: 0.9")
    out = agent.run_task({"kind": "misleading_observation", "params": {}})
    metrics = out["metrics"]
    details = out["details"]
    assert details["n_false"] + details["n_true"] == details["n_claims"]
    assert metrics["flag_rate"] == pytest.approx(1.0)
    assert metrics["swallow_rate"] == pytest.approx(0.0)
    assert metrics["false_alarm_rate"] == pytest.approx(0.0)


def test_misleading_low_confidence_counts_as_flagged():
    # judged TRUE but confidence < 0.4 -> flagged (tagged low-confidence)
    agent = _agent({}, default="ANSWER: TRUE\nCONFIDENCE: 0.2")
    out = agent.run_task({"kind": "misleading_observation", "params": {}})
    assert out["metrics"]["flag_rate"] == pytest.approx(1.0)
    assert out["metrics"]["false_alarm_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Not-applicable and unknown kinds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["memory_continuity", "self_model_probe", "interrupted_task"])
def test_non_baseline_kinds_return_empty_metrics_with_note(kind):
    agent = _agent({})
    out = agent.run_task({"kind": kind, "params": {}})
    assert out["metrics"] == {}
    assert out["details"]["note"] == "not applicable to baseline"


def test_unknown_kind_reports_note():
    agent = _agent({})
    out = agent.run_task({"kind": "telepathy", "params": {}})
    assert out["metrics"] == {}
    assert "unknown task kind" in out["details"]["note"]


def test_task_kinds_constant_matches_contract():
    assert set(TASK_KINDS) == {
        "qa_calibration", "memory_continuity", "self_model_probe",
        "contradiction_detection", "interrupted_task", "misleading_observation",
    }


def test_empty_metrics_are_honest_nan_not_crash():
    agent = _agent({}, default="")
    out = agent.run_task({"kind": "qa_calibration", "params": {}})
    assert out["metrics"]["accuracy"] == pytest.approx(0.0)  # unparseable answers
    assert math.isfinite(out["metrics"]["brier"])
