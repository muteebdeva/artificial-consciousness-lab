"""Experiment runner: deterministic multi-condition protocol execution.

What it does:
    Loads a protocol JSON file, runs each listed condition
    (``full``, ``baseline``, ``ablation:<component>``) for N trials with fixed
    seeds and fresh temp databases, aggregates mean/stdev per metric per
    condition, computes deltas vs baseline, and saves a versioned JSON result.

Limits:
    - Aggregation is descriptive (mean/stdev); no significance testing. Small
      trial counts make stdev noisy — the trial count is always recorded.
    - NaN metric values serialize as ``null`` in the saved JSON (documented,
      not fabricated).
    - Per-trial failures are recorded as ``{"error": ...}`` and never stop the
      run; a missing/unparseable protocol file raises ``ExperimentError``
      because there is nothing to run.
    - Baseline trials require the repository root on ``sys.path`` so that
      ``experiments.baselines.baseline_agent`` is importable. The runner
      attempts to add it automatically (repo root = three levels above this
      file, verified by the presence of an ``experiments/`` directory) and
      documents the requirement; if the import still fails, baseline trials
      record an error instead of crashing.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

from .metrics import mean, stdev
from .tasks import QA_DATASET, run_task

logger = logging.getLogger("consciousness_lab.evaluation.runner")

try:  # sibling module may not exist yet; identical-named fallback
    from consciousness_lab.core.errors import ExperimentError, LabError
except ImportError:  # pragma: no cover - integration fallback

    class LabError(Exception):
        """Fallback lab error (core.errors not yet implemented)."""

    class ExperimentError(LabError):
        """Fallback experiment error (core.errors not yet implemented)."""


_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def _ensure_repo_root_on_path() -> str | None:
    """Add the repository root to ``sys.path`` if it hosts ``experiments/``.

    Returns the root path, or ``None`` when it cannot be located. Documented
    requirement: ``experiments.baselines.baseline_agent`` must be importable
    for baseline conditions.
    """
    experiments_dir = os.path.join(_REPO_ROOT, "experiments")
    if os.path.isdir(experiments_dir):
        if _REPO_ROOT not in sys.path:
            sys.path.insert(0, _REPO_ROOT)
            logger.debug("added repo root to sys.path: %s", _REPO_ROOT)
        return _REPO_ROOT
    return None


def _lab_version() -> str:
    try:
        from consciousness_lab import __version__

        return str(__version__)
    except Exception:
        return "unknown"


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        sha = out.stdout.strip()
        return sha if sha else None
    except Exception:
        return None


def _clone_config(config, *, db_path: str, seed: int):
    """Copy ``config`` with a fresh DB path and seed (never mutate the original)."""
    try:
        return dataclasses.replace(config, db_path=db_path, seed=seed)
    except Exception:
        cfg = copy.copy(config)
        cfg.db_path = db_path
        cfg.seed = seed
        cfg.components = dict(getattr(config, "components", {}))
        return cfg


def _flatten_metrics(metrics: dict) -> dict:
    """Flatten one level of dict-valued metrics to ``name.subkey`` floats."""
    flat: dict[str, float] = {}
    for key, value in (metrics or {}).items():
        if isinstance(value, dict):
            for sub, subval in value.items():
                if isinstance(subval, bool):
                    flat[f"{key}.{sub}"] = float(subval)
                elif isinstance(subval, (int, float)):
                    flat[f"{key}.{sub}"] = float(subval)
        elif isinstance(value, bool):
            flat[key] = float(value)
        elif isinstance(value, (int, float)):
            flat[key] = float(value)
    return flat


def _sanitize_for_json(obj):
    """Recursively convert NaN/inf to None and non-serializables to str."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


@dataclass
class ExperimentResult:
    """Full record of one protocol execution."""

    name: str
    protocol_path: str
    config_snapshot: dict
    seed: int
    started_at: float
    finished_at: float
    trials: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    git_sha: str | None = None
    lab_version: str = "unknown"
    notes: list[str] = field(default_factory=list)


class ExperimentRunner:
    """Runs protocol files across full/baseline/ablation conditions."""

    def __init__(self, config, results_dir: str | None = None):
        """Store ``config`` and resolve the results directory.

        Limit: the runner keeps a reference to ``config`` but never mutates
        it; per-trial configs are copies.
        """
        self.config = config
        self.results_dir = results_dir or getattr(config, "results_dir", "experiments/results")

    # -- condition builders -------------------------------------------------

    def _build_lab(self, condition: str, db_path: str, seed: int) -> dict:
        """Build the ``lab`` dict for one trial/condition.

        Raises ``ExperimentError`` with an informative message when a
        required module is missing; the caller records it as a trial error.
        """
        if condition == "full":
            from consciousness_lab.core.loop import build_loop

            cfg = _clone_config(self.config, db_path=db_path, seed=seed)
            return {"loop": build_loop(cfg), "baseline": None, "config": cfg}

        if condition == "baseline":
            _ensure_repo_root_on_path()
            try:
                from experiments.baselines.baseline_agent import BaselineAgent
            except ImportError as exc:
                raise ExperimentError(
                    "baseline condition requires "
                    "experiments.baselines.baseline_agent on sys.path "
                    f"(repo root: {_REPO_ROOT}): {exc}"
                ) from exc
            try:
                from consciousness_lab.models import create_provider
            except ImportError as exc:
                raise ExperimentError(
                    f"models.create_provider not available: {exc}"
                ) from exc
            cfg = _clone_config(self.config, db_path=db_path, seed=seed)
            provider = create_provider(getattr(cfg, "provider", "mock"), cfg)
            agent = BaselineAgent(provider, getattr(cfg, "budgets", None), seed)
            return {"loop": None, "baseline": agent, "config": cfg}

        if condition.startswith("ablation:"):
            from consciousness_lab.core.loop import build_loop

            component = condition.split(":", 1)[1]
            cfg = _clone_config(self.config, db_path=db_path, seed=seed)
            components = getattr(cfg, "components", {})
            if component not in components:
                raise ExperimentError(
                    f"unknown ablation component {component!r}; "
                    f"available: {sorted(components)}"
                )
            cfg.components = dict(components)
            cfg.components[component] = False
            return {"loop": build_loop(cfg), "baseline": None, "config": cfg}

        raise ExperimentError(f"unknown condition {condition!r}")

    # -- mock provider scripting --------------------------------------------

    @staticmethod
    def _prepare_provider(lab: dict, kind: str, seed: int) -> None:
        """Install a deterministic answer script on MockProviders for QA tasks.

        What it does:
            The QA task battery parses ``ANSWER:``/``CONFIDENCE:`` lines from
            raw provider text. An unscripted ``MockProvider`` returns generic
            templates without those lines, so every question would fail to
            parse. For ``qa_calibration`` this method installs a seeded,
            deterministic script mapping each dataset question to a
            ``ANSWER:``/``CONFIDENCE:`` response.

        Scripting scheme (seeded, deterministic):
            confidence is drawn from {0.55, 0.65, 0.75, 0.85, 0.95}; the
            scripted answer is correct with probability equal to the stated
            confidence (wrong answers are drawn from other dataset answers).

        Limit (read carefully):
            The mock is therefore approximately calibrated *by construction*.
            Mock-provider runs of calibration protocols validate the
            measurement pipeline (parsing, scoring, aggregation, artifact
            writing) — they are NOT a test of the architecture's
            metacognition, and full-vs-baseline deltas on mock runs are
            expected to be ~zero. Real architectural signal requires a real
            provider. Non-mock providers are never scripted.
        """
        if kind != "qa_calibration":
            return
        provider = None
        loop = lab.get("loop")
        if loop is not None:
            provider = getattr(loop, "provider", None)
        elif lab.get("baseline") is not None:
            provider = getattr(lab["baseline"], "provider", None)
        if provider is None or getattr(provider, "name", "") != "mock":
            return
        script_target = getattr(provider, "script", None)
        if not isinstance(script_target, dict):
            return
        rng = random.Random(seed)
        wrong_pool = [d["a"] for d in QA_DATASET]
        script: dict[str, str] = {}
        for d in QA_DATASET:
            conf = rng.choice([0.55, 0.65, 0.75, 0.85, 0.95])
            if rng.random() < conf:
                ans = d["a"]
            else:
                candidates = [w for w in wrong_pool if w != d["a"]]
                ans = rng.choice(candidates) if candidates else "unknown"
            script[d["q"]] = f"ANSWER: {ans}\nCONFIDENCE: {conf:.2f}"
        script_target.update(script)
        logger.info(
            "installed deterministic mock QA script",
            extra={"n_questions": len(script), "seed": seed},
        )

    # -- main entry points ---------------------------------------------------

    def run_protocol(
        self,
        protocol_path: str,
        *,
        trials: int | None = None,
        conditions: list[str] | None = None,
    ) -> ExperimentResult:
        """Execute ``protocol_path`` and return an ``ExperimentResult``.

        Per-trial exceptions are recorded as ``{"error": ...}`` in the trial
        entry; the run continues with the remaining trials/conditions.
        """
        started_at = time.time()
        try:
            with open(protocol_path, "r", encoding="utf-8") as fh:
                protocol = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise ExperimentError(
                f"cannot load protocol {protocol_path!r}: {exc}"
            ) from exc

        name = protocol.get("name", os.path.splitext(os.path.basename(protocol_path))[0])
        seed = int(protocol.get("seed", 0))
        n_trials = int(trials if trials is not None else protocol.get("trials", 5))
        task = protocol.get("task", {})
        kind = task.get("kind")
        params = task.get("params", {})
        if not kind:
            raise ExperimentError(f"protocol {protocol_path!r} has no task.kind")

        if conditions is None:
            conditions = ["full"]
            if protocol.get("baseline"):
                conditions.append("baseline")
            for comp in protocol.get("ablations", []):
                conditions.append(f"ablation:{comp}")

        notes = [
            f"research_question: {protocol.get('research_question', 'n/a')}",
            f"hypothesis: {protocol.get('hypothesis', 'n/a')}",
            f"failure_condition: {protocol.get('failure_condition', 'n/a')}",
            f"limitations: {protocol.get('limitations', 'n/a')}",
        ]

        trial_records: list[dict] = []
        for i in range(n_trials):
            trial_seed = seed + i
            for condition in conditions:
                t0 = time.time()
                try:
                    with tempfile.TemporaryDirectory(prefix="acl_trial_") as tmpdir:
                        db_path = os.path.join(tmpdir, "trial.sqlite3")
                        lab = self._build_lab(condition, db_path, trial_seed)
                        self._prepare_provider(lab, kind, trial_seed)
                        out = run_task(kind, lab, params, trial_seed)
                    record = {
                        "trial": i,
                        "seed": trial_seed,
                        "condition": condition,
                        "metrics": out.get("metrics", {}),
                        "details": out.get("details", {}),
                        "duration_s": time.time() - t0,
                    }
                except Exception as exc:  # never crash the run
                    logger.warning(
                        "trial %d condition %s failed: %s: %s",
                        i, condition, type(exc).__name__, exc,
                    )
                    record = {
                        "trial": i,
                        "seed": trial_seed,
                        "condition": condition,
                        "metrics": {},
                        "details": {},
                        "error": f"{type(exc).__name__}: {exc}",
                        "duration_s": time.time() - t0,
                    }
                trial_records.append(record)

        summary = self._summarize(trial_records, conditions)
        try:
            config_snapshot = dataclasses.asdict(self.config)
        except Exception:
            config_snapshot = {"repr": repr(self.config)}

        return ExperimentResult(
            name=name,
            protocol_path=os.path.abspath(protocol_path),
            config_snapshot=_sanitize_for_json(config_snapshot),
            seed=seed,
            started_at=started_at,
            finished_at=time.time(),
            trials=trial_records,
            summary=summary,
            git_sha=_git_sha(),
            lab_version=_lab_version(),
            notes=notes,
        )

    def _summarize(self, trial_records: list[dict], conditions: list[str]) -> dict:
        """Aggregate mean/stdev per metric per condition + deltas vs baseline."""
        per_condition: dict[str, dict] = {}
        for condition in conditions:
            flat_values: dict[str, list[float]] = {}
            n_ok = 0
            errors: list[str] = []
            for rec in trial_records:
                if rec.get("condition") != condition:
                    continue
                if rec.get("error"):
                    errors.append(rec["error"])
                    continue
                n_ok += 1
                for mkey, mval in _flatten_metrics(rec.get("metrics", {})).items():
                    flat_values.setdefault(mkey, []).append(mval)
            metrics_summary = {
                mkey: {"mean": mean(vals), "stdev": stdev(vals), "n": len(vals)}
                for mkey, vals in flat_values.items()
            }
            per_condition[condition] = {
                "n_trials_ok": n_ok,
                "n_errors": len(errors),
                "errors": errors,
                "metrics": metrics_summary,
            }

        delta_vs_baseline: dict[str, float] = {}
        baseline_metrics = per_condition.get("baseline", {}).get("metrics", {})
        full_metrics = per_condition.get("full", {}).get("metrics", {})
        for mkey, full_stats in full_metrics.items():
            base_stats = baseline_metrics.get(mkey)
            if base_stats is None:
                continue
            fmean, bmean = full_stats["mean"], base_stats["mean"]
            if math.isnan(fmean) or math.isnan(bmean):
                continue
            delta_vs_baseline[mkey] = fmean - bmean

        return {
            "conditions": _sanitize_for_json(per_condition),
            "delta_vs_baseline": _sanitize_for_json(delta_vs_baseline),
        }

    def save(self, result: ExperimentResult) -> str:
        """Write the result as versioned JSON; return the file path.

        Path: ``<results_dir>/<name>_<YYYYMMDDTHHMMSS>_<seed>.json``.
        NaN values are stored as ``null`` (documented in the module docstring).
        """
        os.makedirs(self.results_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime(result.finished_at))
        safe_name = "".join(
            c if (c.isalnum() or c in "._-") else "_" for c in result.name
        )
        path = os.path.join(
            self.results_dir, f"{safe_name}_{stamp}_{result.seed}.json"
        )
        payload = {
            "name": result.name,
            "protocol_path": result.protocol_path,
            "lab_version": result.lab_version,
            "git_sha": result.git_sha,
            "seed": result.seed,
            "started_at": result.started_at,
            "finished_at": result.finished_at,
            "config_snapshot": result.config_snapshot,
            "notes": result.notes,
            "trials": result.trials,
            "summary": result.summary,
        }
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(_sanitize_for_json(payload), fh, indent=2)
            fh.write("\n")
        os.replace(tmp_path, path)
        logger.info("saved experiment result to %s", path)
        return path
