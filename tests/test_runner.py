"""MOCKED TESTS — ExperimentRunner: protocol execution, per-trial seeds,
versioned JSON output, error capture without fabrication, and the
never-fabricate guarantee.

Uses small trials=2 and the mock provider so runs stay fast and offline.
Skips if consciousness_lab.evaluation.runner is not implemented yet.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from conftest import FIXED_SEED, REPO_ROOT, require

runner_mod = require("consciousness_lab.evaluation.runner")
ExperimentRunner = runner_mod.ExperimentRunner

PROTOCOL = str(REPO_ROOT / "experiments" / "protocols" / "03_calibration.json")


def _runner(tmp_path):
    Config = require("consciousness_lab.core.config").Config
    cfg = Config(db_path=str(tmp_path / "exp.sqlite3"), provider="mock",
                 seed=FIXED_SEED, results_dir=str(tmp_path / "results"))
    return ExperimentRunner(cfg, results_dir=str(tmp_path / "results"))


def test_run_protocol_small_trial_count(tmp_path):
    runner = _runner(tmp_path)
    result = runner.run_protocol(PROTOCOL, trials=2, conditions=["full", "baseline"])
    assert result.name == "calibration"
    # the run seed comes from the protocol file, not the Config
    assert result.seed == json.loads(Path(PROTOCOL).read_text())["seed"]
    assert len(result.trials) == 4  # 2 trials x 2 conditions
    conditions = {t["condition"] for t in result.trials}
    assert conditions == {"full", "baseline"}
    # paired design: both conditions of trial i share one seed (comparability),
    # and seeds differ across trial indices (independence)
    by_trial: dict[int, set[int]] = {}
    for t in result.trials:
        by_trial.setdefault(t["trial"], set()).add(t["seed"])
    assert all(len(s) == 1 for s in by_trial.values())
    assert len({next(iter(s)) for s in by_trial.values()}) == 2


def test_trial_dicts_have_expected_shape(tmp_path):
    runner = _runner(tmp_path)
    result = runner.run_protocol(PROTOCOL, trials=2, conditions=["baseline"])
    for trial in result.trials:
        assert "trial" in trial and "seed" in trial and "condition" in trial
        assert "metrics" in trial or "error" in trial or "details" in trial


def test_summary_aggregates_and_deltas(tmp_path):
    runner = _runner(tmp_path)
    result = runner.run_protocol(PROTOCOL, trials=2, conditions=["full", "baseline"])
    summary = result.summary
    assert "conditions" in summary
    for condition in ("full", "baseline"):
        cond = summary["conditions"][condition]
        assert cond["n_trials_ok"] == 2
        assert cond["n_errors"] == 0
        for metric in ("accuracy", "brier", "ece"):
            agg = cond["metrics"][metric]
            assert "mean" in agg and "stdev" in agg
    assert "delta_vs_baseline" in summary


def test_save_writes_versioned_json(tmp_path):
    runner = _runner(tmp_path)
    result = runner.run_protocol(PROTOCOL, trials=2, conditions=["baseline"])
    path = runner.save(result)
    assert Path(path).exists()
    assert re.search(r"calibration_\d{8}T\d{6}_", Path(path).name), (
        f"filename not versioned as <name>_<ts>_<seed>.json: {path}"
    )
    saved = json.loads(Path(path).read_text(encoding="utf-8"))
    assert saved["name"] == "calibration"
    assert "config_snapshot" in saved
    assert "protocol" in saved or "protocol_path" in saved
    assert "trials" in saved and len(saved["trials"]) == 2
    assert "summary" in saved
    assert saved.get("lab_version")  # version recorded


def test_condition_error_recorded_not_raised(tmp_path, monkeypatch):
    """A failing condition must be captured as an error in its trial.

    The runner/tasks layer records the failure (top-level "error" or nested
    under details) with empty metrics instead of raising or fabricating.
    """
    runner = _runner(tmp_path)
    baseline_mod = require("experiments.baselines.baseline_agent")

    def boom(self, task, *, max_cycles=5):
        raise RuntimeError("simulated provider collapse")

    monkeypatch.setattr(baseline_mod.BaselineAgent, "run_task", boom)
    result = runner.run_protocol(PROTOCOL, trials=2, conditions=["baseline"])
    assert len(result.trials) == 2
    for trial in result.trials:
        err = trial.get("error") or trial.get("details", {}).get("error")
        assert err, f"expected error capture, got: {trial}"
        assert "simulated provider collapse" in err
        assert trial["metrics"] == {}
    # save still works on a partially-failed result
    path = runner.save(result)
    assert Path(path).exists()


def test_ablation_condition_disables_component(tmp_path):
    runner = _runner(tmp_path)
    result = runner.run_protocol(PROTOCOL, trials=2,
                                 conditions=["full", "ablation:memory"])
    conditions = {t["condition"] for t in result.trials}
    assert conditions == {"full", "ablation:memory"}
    assert "ablation:memory" in result.summary["conditions"]
