#!/usr/bin/env python3
"""Demo the cognitive loop for a few cycles with the mock provider.

Usage:
    python scripts/demo_loop.py --config configs/mock.yaml --cycles 5

Builds the loop via build_loop(), steps it manually on a demo task, and prints
a per-cycle summary (selected items, confidence, contradictions, provider
calls, errors). Exit non-zero if the core package is not implemented yet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Demo the cognitive loop.")
    ap.add_argument("--config", default="configs/mock.yaml",
                    help="Path to lab config YAML.")
    ap.add_argument("--cycles", type=int, default=5,
                    help="Number of loop cycles to run.")
    ap.add_argument("--task", default="Summarize the key open questions in AI safety research.",
                    help="Task text fed as the observation each cycle.")
    args = ap.parse_args(argv)

    if args.cycles < 1:
        print("ERROR: --cycles must be >= 1", file=sys.stderr)
        return 2

    try:
        from consciousness_lab.core.config import Config
        from consciousness_lab.core.loop import build_loop
    except ImportError as e:
        print(f"ERROR: core package not implemented yet ({e}).", file=sys.stderr)
        return 2

    try:
        config = Config.load(args.config)
    except Exception as e:
        print(f"ERROR: cannot load config {args.config}: {e}", file=sys.stderr)
        return 2

    try:
        loop = build_loop(config)
    except Exception as e:
        print(f"ERROR: build_loop failed: {e}", file=sys.stderr)
        return 1

    print(f"Demo task: {args.task}")
    print(f"Provider:  {config.provider}   cycles: {args.cycles}")
    print("-" * 72)
    failures = 0
    # Create a goal so the planner selects real "respond" actions (without an
    # active goal the planner correctly returns noop and the demo would show
    # zero provider calls).
    try:
        goal = loop.goal_manager.create(
            args.task, success_criteria="demo cycles completed"
        )
    except Exception as e:
        print(f"ERROR: goal creation failed: {e}", file=sys.stderr)
        return 1
    for n in range(1, args.cycles + 1):
        obs = {"modality": "task", "content": args.task,
               "metadata": {"cycle": n}, "source": "demo"}
        try:
            cr = loop.step(obs)
        except Exception as e:  # a failed cycle aborts the demo, honestly
            print(f"cycle {n}: FAILED to step: {e}", file=sys.stderr)
            failures += 1
            break
        conf = f"{cr.confidence:.3f}" if cr.confidence is not None else "n/a"
        n_contra = len(cr.contradictions or [])
        n_err = len(cr.errors or [])
        print(
            f"cycle {cr.cycle}: selected={len(cr.selected_ids)} "
            f"confidence={conf} contradictions={n_contra} "
            f"provider_calls={cr.provider_calls} tokens={cr.tokens_used} "
            f"errors={n_err} duration={cr.duration_s:.2f}s"
        )
        if cr.errors:
            for err in cr.errors:
                print(f"    error: {err}")
        try:
            loop.goal_manager.track_progress(goal.id, n / args.cycles)
        except Exception:
            pass
    print("-" * 72)
    if failures:
        print("DEMO FAILED: loop.step raised.", file=sys.stderr)
        return 1
    try:
        loop.goal_manager.mark_complete(goal.id, reason="demo finished")
    except Exception:
        pass
    print("Demo complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
