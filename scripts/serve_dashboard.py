#!/usr/bin/env python3
"""Serve the research dashboard backed by a live CognitiveLoop.

Usage:
    .venv/bin/python scripts/serve_dashboard.py --config configs/mock.yaml [--port 8765]

Opens the dashboard at http://127.0.0.1:8765/ (loopback only).
All displayed state comes from the live backend; inferred/model-generated
values are labeled as such in the UI.
"""

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Serve the ACL research dashboard.")
    ap.add_argument("--config", default="configs/mock.yaml",
                    help="Path to lab config YAML.")
    ap.add_argument("--port", type=int, default=8765, help="Port to bind.")
    ap.add_argument("--host", default="127.0.0.1",
                    help="Host to bind (default loopback only).")
    args = ap.parse_args(argv)

    try:
        from consciousness_lab.core.config import Config
        from consciousness_lab.core.loop import build_loop
        from consciousness_lab.api import serve
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
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

    lab = {
        "loop": loop,
        "safety": getattr(loop, "safety", None),
        "audit": getattr(loop, "audit", None),
        "config": config,
        "runner": None,  # experiment runner attached separately if needed
    }
    print(f"Dashboard: http://{args.host}:{args.port}/  (config={args.config})")
    serve(lab, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
