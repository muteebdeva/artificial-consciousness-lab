#!/usr/bin/env python3
"""Run the matched baseline agent on one task kind and print metrics JSON.

Usage:
    python scripts/run_baseline.py --task-kind qa_calibration --config configs/mock.yaml

The baseline uses the same provider and budgets as the full loop but none of
the cognitive components. Exit 0 when metrics were produced, 1 when the task
is not applicable to the baseline, 2 when the environment is not ready.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the baseline agent on a task.")
    ap.add_argument("--task-kind", required=True,
                    help="Task kind, e.g. qa_calibration, contradiction_detection, "
                         "misleading_observation.")
    ap.add_argument("--config", default="configs/mock.yaml",
                    help="Path to lab config YAML (provider + budgets).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-cycles", type=int, default=5)
    ap.add_argument("--params", default="{}",
                    help="JSON object of task params.")
    args = ap.parse_args(argv)

    try:
        params = json.loads(args.params)
        if not isinstance(params, dict):
            raise ValueError("params must be a JSON object")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"ERROR: invalid --params: {e}", file=sys.stderr)
        return 2

    try:
        from consciousness_lab.core.config import Config
        from consciousness_lab.models import create_provider
    except ImportError as e:
        print(f"ERROR: core package not implemented yet ({e}).", file=sys.stderr)
        return 2

    try:
        config = Config.load(args.config)
    except Exception as e:
        print(f"ERROR: cannot load config {args.config}: {e}", file=sys.stderr)
        return 2

    try:
        provider = create_provider(config.provider, config)
    except Exception as e:
        print(f"ERROR: cannot create provider '{config.provider}': {e}",
              file=sys.stderr)
        return 2

    from experiments.baselines.baseline_agent import BaselineAgent

    agent = BaselineAgent(provider, budgets=config.budgets, seed=args.seed)
    # For qa_calibration with the mock provider, install the same deterministic
    # answer script the experiment runner uses (see evaluation.runner), so the
    # standalone baseline run measures the same pipeline. Non-mock providers
    # and other task kinds are unaffected.
    try:
        from consciousness_lab.evaluation.runner import ExperimentRunner
        ExperimentRunner._prepare_provider(
            {"loop": None, "baseline": agent, "config": config},
            args.task_kind,
            args.seed,
        )
    except Exception as e:  # scripting is best-effort; the task still runs
        print(f"WARNING: mock scripting unavailable: {e}", file=sys.stderr)
    out = agent.run_task(
        {"name": args.task_kind, "kind": args.task_kind,
         "params": params, "seed": args.seed},
        max_cycles=args.max_cycles,
    )
    print(json.dumps(out, indent=2, default=str))

    if not out.get("metrics"):
        note = out.get("details", {}).get("note", "no metrics produced")
        print(f"NOTE: {note}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
