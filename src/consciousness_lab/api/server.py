"""Minimal stdlib HTTP API exposing live research state.

What it does:
    ``create_handler(lab)`` builds a ``BaseHTTPRequestHandler`` subclass bound
    to a ``lab`` dict ``{"loop", "config", "audit", "safety", "runner"}``;
    ``serve(lab, host, port)`` runs it with ``ThreadingHTTPServer``.
    Endpoints (all GET, JSON) plus ``POST /api/stop`` are listed in the
    contract; every value comes from the live objects — nothing is invented.

Limits:
    - Read-only except ``POST /api/stop``. No authentication: bind to
      127.0.0.1 (the default) and do not expose this to untrusted networks.
    - Missing optional pieces return ``503 {"error": ...}``, never a crash;
      ``runner=None`` makes ``/api/experiments`` a 503 per the contract.
    - The loop's internals are accessed by duck typing (documented attribute
      names); when the loop does not expose what an endpoint needs, that
      endpoint returns 503 instead of fabricating values.
    - ``model_generated`` tagging is heuristic: a record/item is labeled
      ``true`` when its ``source_type`` is ``"model_hypothesis"`` or its
      ``source``/``provenance`` mentions the model/provider. It marks
      *provenance as recorded by the system*, not a verified fact about the
      content's origin.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import logging
import math
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger("consciousness_lab.api")


def _lab_version() -> str:
    try:
        from consciousness_lab import __version__

        return str(__version__)
    except Exception:
        return "unknown"


def _sanitize(obj):
    """Make ``obj`` JSON-safe: NaN/inf -> None, dataclasses -> dicts."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _sanitize(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return _sanitize(obj.to_dict())
        except Exception:
            pass
    return str(obj)


def _tag_model_generated(item: dict) -> dict:
    """Heuristically label provider-originated content (see module docstring)."""
    source_type = str(item.get("source_type", ""))
    source = str(item.get("source", "")).lower()
    provenance = str(item.get("provenance", "")).lower()
    item["model_generated"] = bool(
        source_type == "model_hypothesis"
        or "model" in source
        or "provider" in source
        or "model" in provenance
    )
    return item


def create_handler(lab: dict) -> type[BaseHTTPRequestHandler]:
    """Build a request-handler class bound to ``lab``.

    ``lab`` keys: ``loop``, ``config``, ``audit``, ``safety``,
    ``runner`` (may be ``None``).
    """

    class LabHandler(BaseHTTPRequestHandler):
        server_version = "ConsciousnessLabAPI/0.1"

        # -- plumbing ----------------------------------------------------
        def log_message(self, fmt, *args):  # route away from stderr print
            logger.info("%s - %s", self.address_string(), fmt % args)

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(_sanitize(payload)).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str) -> None:
            self._send_json({"error": message}, status=status)

        def _loop(self):
            return lab.get("loop")

        # -- routing -----------------------------------------------------
        def _serve_dashboard(self):
            """Serve web/index.html (the research dashboard)."""
            here = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(here, "..", "..", "..", "..", "web", "index.html"),
                os.path.join(os.getcwd(), "web", "index.html"),
            ]
            for candidate in candidates:
                candidate = os.path.normpath(candidate)
                if os.path.isfile(candidate):
                    with open(candidate, "rb") as fh:
                        body = fh.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
            self._error(404, "dashboard not found (web/index.html missing)")

        def do_GET(self):  # noqa: N802 (http.server convention)
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            if path == "/":
                self._serve_dashboard()
                return
            routes = {
                "/api/health": self._get_health,
                "/api/workspace": self._get_workspace,
                "/api/memory": lambda: self._get_memory(query),
                "/api/self-model": self._get_self_model,
                "/api/goals": self._get_goals,
                "/api/uncertainty": self._get_uncertainty,
                "/api/predictions": self._get_predictions,
                "/api/traces": lambda: self._get_traces(query),
                "/api/experiments": self._get_experiments,
                "/api/resources": self._get_resources,
                "/api/errors": self._get_errors,
            }
            handler = routes.get(path)
            if handler is None:
                self._error(404, f"unknown endpoint {path!r}")
                return
            try:
                handler()
            except Exception as exc:  # never crash the server on a bad endpoint
                logger.exception("endpoint %s failed", path)
                self._error(500, f"{type(exc).__name__}: {exc}")

        def do_POST(self):  # noqa: N802 (http.server convention)
            parsed = urlparse(self.path)
            if parsed.path != "/api/stop":
                self._error(404, f"unknown endpoint {parsed.path!r}")
                return
            length = int(self.headers.get("Content-Length", 0) or 0)
            if length:
                self.rfile.read(length)  # drain body
            safety = lab.get("safety")
            if safety is None:
                self._error(503, "safety controller not configured")
                return
            try:
                safety.stop("api")
            except Exception as exc:
                logger.exception("POST /api/stop failed")
                self._error(500, f"{type(exc).__name__}: {exc}")
                return
            self._send_json({"ok": True, "stopped": bool(safety.stopped)})

        # -- endpoints ---------------------------------------------------
        def _get_health(self):
            self._send_json({"ok": True, "version": _lab_version()})

        def _get_workspace(self):
            workspace = getattr(self._loop(), "workspace", None)
            if workspace is None:
                self._error(503, "loop does not expose a workspace")
                return
            snap = _sanitize(workspace.snapshot())
            items = [_tag_model_generated(i) for i in snap.get("items", [])]
            self._send_json({
                "capacity": snap.get("capacity"),
                "cycle": snap.get("cycle"),
                "updated_at": snap.get("updated_at"),
                "n_items": len(items),
                "items": items,
            })

        def _get_memory(self, query):
            loop = self._loop()
            q = query.get("q", [""])[0]
            kind = query.get("kind", [None])[0]
            try:
                k = max(1, min(50, int(query.get("k", ["5"])[0])))
            except ValueError:
                k = 5
            episodic = getattr(loop, "episodic_memory", None) or getattr(loop, "episodic", None)
            semantic = getattr(loop, "semantic_memory", None) or getattr(loop, "semantic", None)
            records: list[dict] = []
            try:
                if kind == "episodic" and episodic is not None:
                    records = [_sanitize(r) for r in episodic.recall(q, k)]
                elif kind == "semantic" and semantic is not None:
                    records = [_sanitize(r) for r in semantic.query(q, k)]
                elif kind is None:
                    if episodic is not None:
                        records += [_sanitize(r) for r in episodic.recall(q, k)]
                    if semantic is not None:
                        records += [_sanitize(r) for r in semantic.query(q, k)]
                else:
                    self._error(400, f"unknown memory kind {kind!r}; use episodic|semantic")
                    return
            except Exception as exc:
                self._error(500, f"memory query failed: {type(exc).__name__}: {exc}")
                return
            if kind is None and episodic is None and semantic is None:
                self._error(503, "loop does not expose episodic or semantic memory")
                return
            if kind in ("episodic", "semantic") and not records and (
                (kind == "episodic" and episodic is None)
                or (kind == "semantic" and semantic is None)
            ):
                self._error(503, f"loop does not expose {kind} memory")
                return
            self._send_json({
                "q": q, "kind": kind, "k": k, "n": len(records),
                "records": [_tag_model_generated(r) for r in records],
            })

        def _get_self_model(self):
            loop = self._loop()
            sm = getattr(loop, "self_model", None) or getattr(loop, "selfmodel", None)
            if sm is None:
                self._error(503, "loop does not expose a self-model")
                return
            self._send_json({"self_model": _sanitize(sm.snapshot())})

        def _get_goals(self):
            loop = self._loop()
            gm = getattr(loop, "goal_manager", None) or getattr(loop, "goals", None)
            if gm is None:
                self._error(503, "loop does not expose a goal manager")
                return
            goals = [_sanitize(g) for g in gm.active_goals()]
            self._send_json({"n_active": len(goals), "goals": goals})

        def _get_uncertainty(self):
            loop = self._loop()
            om = getattr(loop, "outcome_memory", None) or getattr(loop, "outcome", None)
            calibration = _sanitize(om.calibration()) if om is not None else None
            recent = getattr(loop, "recent_confidences", None)
            if recent is None:
                sm = getattr(loop, "self_model", None)
                try:
                    recent = (sm.snapshot() or {}).get("uncertainty_estimates", [])
                except Exception:
                    recent = []
            if calibration is None and not recent:
                self._error(503, "no uncertainty data available from the loop")
                return
            self._send_json({
                "calibration": calibration,
                "recent_confidence_estimates": _sanitize(recent or []),
            })

        def _get_predictions(self):
            loop = self._loop()
            om = getattr(loop, "outcome_memory", None) or getattr(loop, "outcome", None)
            if om is None:
                self._error(503, "loop does not expose an outcome memory")
                return
            pending = [_sanitize(p) for p in om.pending()]
            self._send_json({
                "n_pending": len(pending),
                "pending": [_tag_model_generated(p) for p in pending],
                "calibration": _sanitize(om.calibration()),
            })

        def _get_traces(self, query):
            audit = lab.get("audit")
            if audit is None:
                self._error(503, "audit log not configured")
                return
            try:
                limit = max(1, min(500, int(query.get("limit", ["50"])[0])))
            except ValueError:
                limit = 50
            self._send_json({"limit": limit, "entries": _sanitize(audit.read(limit))})

        def _get_experiments(self):
            runner = lab.get("runner")
            if runner is None:
                self._error(503, "experiment runner not configured")
                return
            config = lab.get("config")
            results_dir = getattr(runner, "results_dir", None) or getattr(
                config, "results_dir", "experiments/results")
            files = []
            latest_summary = None
            try:
                paths = sorted(glob.glob(os.path.join(results_dir, "*.json")))
                for p in paths:
                    files.append({
                        "name": os.path.basename(p),
                        "path": p,
                        "modified_at": os.path.getmtime(p),
                    })
                if paths:
                    newest = max(paths, key=os.path.getmtime)
                    with open(newest, "r", encoding="utf-8") as fh:
                        doc = json.load(fh)
                    latest_summary = {
                        "file": os.path.basename(newest),
                        "name": doc.get("name"),
                        "finished_at": doc.get("finished_at"),
                        "summary": doc.get("summary"),
                    }
            except Exception as exc:
                logger.warning("reading experiment results failed: %s", exc)
            self._send_json({
                "results_dir": results_dir,
                "n_files": len(files),
                "files": files,
                "latest_summary": latest_summary,
            })

        def _get_resources(self):
            safety = lab.get("safety")
            config = lab.get("config")
            budget = getattr(safety, "budget", None)
            provider = getattr(self._loop(), "provider", None)
            provider_name = getattr(provider, "name", None) or getattr(
                config, "provider", None)
            self._send_json({
                "budget_remaining": _sanitize(budget.remaining()) if budget else None,
                "budget_exhausted": bool(budget.exhausted()) if budget else None,
                "provider": provider_name,
                "autonomous_enabled": getattr(safety, "autonomous_enabled", None),
                "stopped": bool(getattr(safety, "stopped", False)),
            })

        def _get_errors(self):
            loop = self._loop()
            errors = None
            for attr in ("_recent_errors", "recent_errors", "errors", "error_log"):
                candidate = getattr(loop, attr, None)
                if isinstance(candidate, list):
                    errors = candidate
                    break
            if errors is None:
                self._error(503, "loop does not expose a recent error list")
                return
            self._send_json({"n": len(errors), "errors": _sanitize(errors)})

    return LabHandler


def serve(lab: dict, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve the lab API forever (blocking).

    Binds to ``host`` (default loopback only — do not expose to untrusted
    networks; there is no authentication). ``port=0`` selects an ephemeral
    port (useful for tests).
    """
    handler = create_handler(lab)
    server = ThreadingHTTPServer((host, port), handler)
    bound = server.server_address
    logger.info("serving lab API on %s:%s", bound[0], bound[1])
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("API server interrupted; shutting down")
    finally:
        server.server_close()
