#!/usr/bin/env python3
"""Run an experiment protocol through the ExperimentRunner.

Usage:
    python scripts/run_experiment.py --protocol experiments/protocols/03_calibration.json \
        --config configs/mock.yaml --trials 5 --conditions full,baseline --seed 0

Prints a summary table (mean +/- stdev per metric per condition, deltas vs
baseline), saves the versioned result JSON, and exits non-zero if any trial
failed or if the evaluation package is not implemented yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

REQUIRED_PROTOCOL_KEYS = {
    "name", "version", "research_question", "hypothesis", "failure_condition",
    "task", "baseline", "ablations", "metrics", "trials", "seed",
    "analysis", "limitations",
}


def load_protocol(path: str) -> dict:
    """Load and schema-check a protocol JSON (needs no src imports)."""
    with open(path, "r", encoding="utf-8") as f:
        protocol = json.load(f)
    missing = REQUIRED_PROTOCOL_KEYS - set(protocol)
    if missing:
        raise ValueError(f"protocol {path} missing keys: {sorted(missing)}")
    task = protocol["task"]
    if not isinstance(task, dict) or "kind" not in task:
        raise ValueError(f"protocol {path}: task must be an object with 'kind'")
    return protocol


def print_summary(result) -> None:
    """Print the runner's aggregated summary as a table."""
    summary = result.summary or {}
    print(f"\nExperiment: {result.name}")
    print(f"Protocol:   {result.protocol_path}")
    print(f"Seed:       {result.seed}   Trials: {len(result.trials)}")
    print("-" * 78)
    header = f"{'condition':<22}{'metric':<22}{'mean':>10}{'stdev':>10}{'n':>6}"
    print(header)
    print("-" * 78)
    per_condition = summary.get("conditions", summary.get("per_condition", {}))
    for condition in sorted(per_condition):
        cond = per_condition[condition]
        metrics = cond.get("metrics", cond)  # support both summary shapes
        n_ok = cond.get("n_trials_ok", "?")
        rows = 0
        for metric in sorted(metrics):
            agg = metrics[metric]
            if not isinstance(agg, dict):
                continue
            mean = agg.get("mean")
            stdev = agg.get("stdev")
            n = agg.get("n", n_ok)
            mean_s = f"{mean:.4f}" if isinstance(mean, (int, float)) else "n/a"
            sd_s = f"{stdev:.4f}" if isinstance(stdev, (int, float)) else "n/a"
            print(f"{condition:<22}{metric:<22}{mean_s:>10}{sd_s:>10}{str(n):>6}")
            rows += 1
        if cond.get("n_errors"):
            print(f"{condition:<22}{'! errors':<22}{cond['n_errors']:>10}")
        elif rows == 0:
            print(f"{condition:<22}{'(not applicable to this task)':<22}")
    print("-" * 78)
    deltas = summary.get("delta_vs_baseline", {})
    if deltas:
        print("Deltas vs baseline (full - baseline):")
        for metric in sorted(deltas):
            print(f"  {metric}: {deltas[metric]:+.4f}")
    errors = sum(1 for t in result.trials
                 if t.get("error") or t.get("details", {}).get("error"))
    print(f"Trials with errors: {errors}/{len(result.trials)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run an experiment protocol.")
    ap.add_argument("--protocol", required=True, help="Path to protocol JSON.")
    ap.add_argument("--config", default="configs/default.yaml",
                    help="Path to lab config YAML.")
    ap.add_argument("--trials", type=int, default=None,
                    help="Override protocol trial count.")
    ap.add_argument("--conditions", default=None,
                    help="Comma-separated conditions, e.g. full,baseline,ablation:memory. "
                         "If omitted, conditions come from the protocol file "
                         "(baseline flag + ablations list).")
    ap.add_argument("--seed", type=int, default=None,
                    help="Override protocol seed.")
    args = ap.parse_args(argv)

    try:
        protocol = load_protocol(args.protocol)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    try:
        from consciousness_lab.core.config import Config
        from consciousness_lab.evaluation.runner import ExperimentRunner
    except ImportError as e:
        print(f"ERROR: evaluation package not implemented yet ({e}); "
              f"protocol '{protocol['name']}' validated but cannot run.",
              file=sys.stderr)
        return 2

    try:
        config = Config.load(args.config)
    except Exception as e:  # ConfigError and friends
        print(f"ERROR: cannot load config {args.config}: {e}", file=sys.stderr)
        return 2

    conditions = (
        [c.strip() for c in args.conditions.split(",") if c.strip()]
        if args.conditions
        else None  # runner derives conditions from the protocol file
    )
    runner = ExperimentRunner(config)
    try:
        result = runner.run_protocol(
            args.protocol,
            trials=args.trials,
            conditions=conditions,
        )
    except Exception as e:
        print(f"ERROR: experiment failed: {e}", file=sys.stderr)
        return 1

    saved = runner.save(result)
    print(f"Saved: {saved}")
    print_summary(result)

    errors = sum(1 for t in result.trials
                 if t.get("error") or t.get("details", {}).get("error"))
    if errors:
        print(f"FAILURE: {errors} trial(s) errored.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
