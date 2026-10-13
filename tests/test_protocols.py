"""REAL TESTS — experiment protocol JSON schema and content validation.

No src imports, no mocks: validates the 10 protocol files against the schema
in INTERFACES.md (required keys, types, task-kind mapping, falsifiability
content requirements).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = REPO_ROOT / "experiments" / "protocols"

REQUIRED_KEYS = {
    "name", "version", "research_question", "hypothesis", "failure_condition",
    "task", "baseline", "ablations", "metrics", "trials", "seed",
    "analysis", "limitations",
}
KNOWN_KINDS = (
    "qa_calibration", "memory_continuity", "self_model_probe",
    "contradiction_detection", "interrupted_task", "misleading_observation",
)
KNOWN_COMPONENTS = {"memory", "self_model", "workspace", "metacognition", "attention"}

# Contract mapping: protocol file -> expected task kind.
EXPECTED_KINDS = {
    "01_memory_continuity.json": "memory_continuity",
    "02_self_model_accuracy.json": "self_model_probe",
    "03_calibration.json": "qa_calibration",
    "04_error_contradiction_detection.json": "contradiction_detection",
    "05_prediction_accuracy.json": "qa_calibration",
    "06_task_recovery.json": "interrupted_task",
    "07_cross_module_integration.json": "qa_calibration",
    "08_adaptation_after_feedback.json": "qa_calibration",
    "09_robustness_misleading.json": "misleading_observation",
    "10_component_ablation.json": "qa_calibration",
}


def _load_all():
    files = sorted(PROTOCOL_DIR.glob("*.json"))
    return {f.name: json.loads(f.read_text(encoding="utf-8")) for f in files}


def test_all_ten_protocols_exist():
    protocols = _load_all()
    assert set(protocols) == set(EXPECTED_KINDS), (
        f"expected exactly the 10 protocol files, got: {sorted(protocols)}"
    )


@pytest.mark.parametrize("filename", sorted(EXPECTED_KINDS))
def test_protocol_schema(filename):
    protocol = _load_all()[filename]
    missing = REQUIRED_KEYS - set(protocol)
    assert not missing, f"{filename} missing keys: {sorted(missing)}"
    assert isinstance(protocol["name"], str) and protocol["name"]
    assert isinstance(protocol["version"], str) and protocol["version"]
    assert isinstance(protocol["task"], dict)
    assert protocol["task"].get("kind") in KNOWN_KINDS, (
        f"{filename}: unknown task kind {protocol['task'].get('kind')!r}"
    )
    assert isinstance(protocol["task"].get("params"), dict)
    assert protocol["baseline"] is True
    assert isinstance(protocol["ablations"], list)
    assert set(protocol["ablations"]) <= KNOWN_COMPONENTS, (
        f"{filename}: unknown ablation component(s): {protocol['ablations']}"
    )
    assert isinstance(protocol["metrics"], list) and len(protocol["metrics"]) >= 1
    assert all(isinstance(m, str) and m for m in protocol["metrics"])
    assert isinstance(protocol["trials"], int) and protocol["trials"] >= 1
    assert isinstance(protocol["seed"], int)


@pytest.mark.parametrize("filename,expected_kind", sorted(EXPECTED_KINDS.items()))
def test_protocol_task_kind_mapping(filename, expected_kind):
    protocol = _load_all()[filename]
    assert protocol["task"]["kind"] == expected_kind, (
        f"{filename}: expected kind {expected_kind!r}, got {protocol['task']['kind']!r}"
    )


@pytest.mark.parametrize("filename", sorted(EXPECTED_KINDS))
def test_protocol_is_scientifically_honest(filename):
    """Every protocol must state a real question, a falsifiable hypothesis, an
    explicit failure condition, and genuine limitations — no placeholders."""
    protocol = _load_all()[filename]
    for key in ("research_question", "hypothesis", "failure_condition",
                "analysis", "limitations"):
        value = protocol[key]
        assert isinstance(value, str) and len(value.strip()) >= 50, (
            f"{filename}: {key!r} is too short to be a serious statement"
        )
        lowered = value.lower()
        assert "todo" not in lowered and "tbd" not in lowered, (
            f"{filename}: {key!r} contains a placeholder"
        )


def test_protocol_10_ablates_all_five_components():
    protocol = _load_all()["10_component_ablation.json"]
    assert set(protocol["ablations"]) == KNOWN_COMPONENTS


def test_protocol_05_is_prediction_focused():
    params = _load_all()["05_prediction_accuracy.json"]["task"]["params"]
    assert params.get("prediction_focused") is True
    assert params.get("require_predict_lines") is True


def test_protocol_08_runs_twice_with_feedback():
    params = _load_all()["08_adaptation_after_feedback.json"]["task"]["params"]
    assert params.get("run_twice") is True
    assert params["feedback_injection"]["inject_into"] == "semantic_memory"


def test_protocol_07_notes_integration_design():
    params = _load_all()["07_cross_module_integration.json"]["task"]["params"]
    assert "integration_note" in params and len(params["integration_note"]) > 20


def test_protocol_names_match_filenames():
    for filename, protocol in _load_all().items():
        stem = filename[:-len(".json")]
        # filename is NN_name; protocol name is the bare name
        assert protocol["name"] == stem[3:], (
            f"{filename}: protocol name {protocol['name']!r} != {stem[3:]!r}"
        )
