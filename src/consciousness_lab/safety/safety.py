"""Safety controls: budgets, permissions, append-only audit, controller.

What it does:
    - ``Budget``: hard caps on steps, tokens, wall-time, and tool calls.
      Exceeding any cap raises ``BudgetExceeded`` with an informative message.
    - ``Permissions``: tool allowlist. ``grant``/``revoke`` require explicit
      human authorization (``authorized=True``); the system cannot
      self-authorize. Default-deny for anything not listed.
    - ``AuditLog``: append-only record of events. Exposes ``log()`` and
      ``read()`` ONLY — there are deliberately no update or delete methods.
      Backed by a ``SqliteStore``-compatible store, a JSONL file, or memory.
    - ``SafetyController``: gates actions. ``requires_approval=True`` ALWAYS
      raises ``PermissionDenied("human approval required")`` — there is no
      auto-approval path. Tool actions outside the allowlist are denied.
      Budget costs are consumed per action.

Limits:
    - Budgets are cooperative: they only bind code that calls the ``consume_*``
      methods. They do not sandbox the OS process.
    - ``Permissions.grant`` with ``authorized=True`` models a human operator's
      decision; this module cannot verify who set the flag — the caller is
      trusted to pass it only for genuine human authorization.
    - The in-memory ``AuditLog`` backend is for tests/demos only; it is not
      durable across restarts. Prefer the store or file backend.
    - Error classes are imported from ``consciousness_lab.core.errors`` when
      available; identical-named fallbacks are defined otherwise so this
      module never fails to import before sibling modules exist.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field

try:
    from consciousness_lab.core.errors import (
        BudgetExceeded,
        PermissionDenied,
        SafetyError,
    )
except ImportError:  # pragma: no cover - integration fallback

    class BudgetExceeded(Exception):
        """Fallback: raised when a budget cap is exceeded."""

    class PermissionDenied(Exception):
        """Fallback: raised when an action is not permitted."""

    class SafetyError(Exception):
        """Fallback: raised for safety-control misuse."""


try:
    from consciousness_lab.core.logging_setup import setup_logging

    logger = setup_logging(name="consciousness_lab.safety")
except ImportError:  # pragma: no cover - integration fallback
    logger = logging.getLogger("consciousness_lab.safety")

_AUDIT_DDL = (
    "CREATE TABLE IF NOT EXISTS audit_log "
    "(id TEXT PRIMARY KEY, ts REAL NOT NULL, event TEXT NOT NULL, "
    "details_json TEXT NOT NULL)"
)


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


@dataclass
class Budget:
    """Hard execution caps. All ``consume_*`` methods raise ``BudgetExceeded``.

    Semantics are check-then-consume: a call that would exceed a cap raises
    *without* consuming, so the caller always knows the exact state.
    """

    max_steps: int
    max_tokens: int
    max_wall_time_s: float
    max_tool_calls: int
    steps_used: int = 0
    tokens_used: int = 0
    tool_calls_used: int = 0
    started_at: float = field(default_factory=time.time)

    @classmethod
    def from_config(cls, c) -> "Budget":
        """Build from a ``BudgetConfig`` (duck-typed attribute access)."""
        return cls(
            max_steps=int(getattr(c, "max_steps", 20)),
            max_tokens=int(getattr(c, "max_tokens", 8000)),
            max_wall_time_s=float(getattr(c, "max_wall_time_s", 120.0)),
            max_tool_calls=int(getattr(c, "max_tool_calls", 10)),
        )

    def consume_step(self) -> None:
        """Consume one step; raise ``BudgetExceeded`` if the cap is reached."""
        if self.steps_used >= self.max_steps:
            raise BudgetExceeded(
                f"step budget exhausted: {self.steps_used}/{self.max_steps} steps used"
            )
        self.steps_used += 1

    def consume_tokens(self, n: int) -> None:
        """Consume ``n`` tokens; raise ``BudgetExceeded`` if that would exceed the cap."""
        n = int(n)
        if n < 0:
            raise ValueError(f"cannot consume a negative token count: {n}")
        if self.tokens_used + n > self.max_tokens:
            raise BudgetExceeded(
                f"token budget would be exceeded: {self.tokens_used}+{n} > "
                f"{self.max_tokens} tokens"
            )
        self.tokens_used += n

    def consume_tool_call(self) -> None:
        """Consume one tool call; raise ``BudgetExceeded`` if the cap is reached."""
        if self.tool_calls_used >= self.max_tool_calls:
            raise BudgetExceeded(
                f"tool-call budget exhausted: {self.tool_calls_used}/"
                f"{self.max_tool_calls} tool calls used"
            )
        self.tool_calls_used += 1

    def check_time(self) -> None:
        """Raise ``BudgetExceeded`` if wall-time has passed the cap."""
        elapsed = time.time() - self.started_at
        if elapsed > self.max_wall_time_s:
            raise BudgetExceeded(
                f"wall-time budget exceeded: {elapsed:.1f}s > "
                f"{self.max_wall_time_s:.1f}s"
            )

    def remaining(self) -> dict:
        """Remaining budget per dimension (never negative)."""
        elapsed = time.time() - self.started_at
        return {
            "steps": max(0, self.max_steps - self.steps_used),
            "tokens": max(0, self.max_tokens - self.tokens_used),
            "tool_calls": max(0, self.max_tool_calls - self.tool_calls_used),
            "wall_time_s": max(0.0, self.max_wall_time_s - elapsed),
        }

    def exhausted(self) -> bool:
        """True if any cap is reached (checked against current time)."""
        try:
            self.check_time()
        except BudgetExceeded:
            return True
        return (
            self.steps_used >= self.max_steps
            or self.tokens_used >= self.max_tokens
            or self.tool_calls_used >= self.max_tool_calls
        )


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------


class Permissions:
    """Tool allowlist with default-deny.

    ``grant``/``revoke`` mutate the allowlist only with explicit human
    authorization (``authorized=True``); otherwise they raise ``SafetyError``.
    The ``locked`` flag is recorded for audit metadata; it does not change the
    authorized-only mutation rule.
    """

    def __init__(self, allowlist: tuple[str, ...] = (), *, locked: bool = True):
        self._allowlist: set[str] = set(allowlist)
        self.locked = bool(locked)

    @property
    def allowlist(self) -> tuple[str, ...]:
        """Current allowlist as a tuple (read-only view)."""
        return tuple(sorted(self._allowlist))

    def allowed(self, tool_name: str) -> bool:
        """True iff ``tool_name`` is on the allowlist (default-deny)."""
        return tool_name in self._allowlist

    def grant(self, tool_name: str, *, authorized: bool = False) -> None:
        """Add ``tool_name`` to the allowlist.

        Raises ``SafetyError`` unless ``authorized=True``: the system cannot
        self-authorize. The caller is trusted to pass ``authorized=True`` only
        for genuine human authorization.
        """
        if not authorized:
            raise SafetyError(
                f"grant({tool_name!r}) requires explicit human authorization "
                "(authorized=True); the system cannot self-authorize"
            )
        self._allowlist.add(tool_name)
        logger.info("permission granted for tool=%s (authorized)", tool_name)

    def revoke(self, tool_name: str, *, authorized: bool = False) -> None:
        """Remove ``tool_name`` from the allowlist (idempotent).

        Raises ``SafetyError`` unless ``authorized=True``.
        """
        if not authorized:
            raise SafetyError(
                f"revoke({tool_name!r}) requires explicit human authorization "
                "(authorized=True); the system cannot self-authorize"
            )
        self._allowlist.discard(tool_name)
        logger.info("permission revoked for tool=%s (authorized)", tool_name)


# ---------------------------------------------------------------------------
# AuditLog
# ---------------------------------------------------------------------------


class AuditLog:
    """Append-only audit log. ``log()`` and ``read()`` are the only methods.

    Backends (first available wins):
      1. ``store`` — any object with ``ensure_table``/``execute_write``/
         ``execute_read`` (e.g. ``SqliteStore``); table ``audit_log``.
      2. ``path`` — JSONL file, one record per line.
      3. neither — in-memory list (tests/demos only; not durable).

    There are deliberately no update or delete methods: audit history must
    not be rewritable through this class.
    """

    def __init__(self, store=None, path: str | None = None):
        self._store = store
        self._path = path
        self._memory: list[dict] = []
        if self._store is not None:
            self._store.ensure_table("audit_log", _AUDIT_DDL)
        elif self._path is not None:
            parent = os.path.dirname(os.path.abspath(self._path))
            os.makedirs(parent, exist_ok=True)

    def log(self, event: str, details: dict) -> None:
        """Append one audit record. Never updates or deletes."""
        record = {
            "id": uuid.uuid4().hex,
            "ts": time.time(),
            "event": str(event),
            "details": dict(details or {}),
        }
        if self._store is not None:
            self._store.execute_write(
                "INSERT INTO audit_log (id, ts, event, details_json) VALUES (?, ?, ?, ?)",
                (record["id"], record["ts"], record["event"],
                 json.dumps(record["details"])),
            )
        elif self._path is not None:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        else:
            self._memory.append(record)
        logger.debug("audit event=%s", event)

    def read(self, limit: int = 100) -> list[dict]:
        """Return up to ``limit`` most recent records, newest first."""
        limit = max(1, int(limit))
        if self._store is not None:
            rows = self._store.execute_read(
                "SELECT id, ts, event, details_json FROM audit_log "
                "ORDER BY ts DESC LIMIT ?",
                (limit,),
            )
            return [
                {"id": r["id"], "ts": r["ts"], "event": r["event"],
                 "details": json.loads(r["details_json"])}
                for r in rows
            ]
        if self._path is not None:
            records: list[dict] = []
            try:
                with open(self._path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            records.append(json.loads(line))
            except FileNotFoundError:
                return []
            return records[-limit:][::-1]
        return self._memory[-limit:][::-1]


# ---------------------------------------------------------------------------
# SafetyController
# ---------------------------------------------------------------------------


class SafetyController:
    """Gates actions against approvals, permissions, and budgets.

    - ``requires_approval=True`` → ALWAYS ``PermissionDenied("human approval
      required")``. There is no auto-approval path and no override flag.
    - ``type == "tool"`` with a tool outside the allowlist → ``PermissionDenied``.
    - ``action.budget_cost`` (``{"tokens": int, "tool_calls": int}``) is
      consumed; exhaustion raises ``BudgetExceeded``.
    """

    def __init__(self, budget: Budget, permissions: Permissions,
                 audit: AuditLog, *, autonomous_enabled: bool = True):
        self.budget = budget
        self.permissions = permissions
        self.audit = audit
        self._autonomous_enabled = bool(autonomous_enabled)
        self._stopped_runs: set[str] = set()

    @property
    def autonomous_enabled(self) -> bool:
        """Whether autonomous execution is currently enabled."""
        return self._autonomous_enabled

    @property
    def stopped(self) -> bool:
        """True if any run has been stopped via ``stop()``."""
        return len(self._stopped_runs) > 0

    def check_action(self, action) -> None:
        """Gate ``action`` (duck-typed: ``requires_approval``, ``type``,
        ``payload``, ``budget_cost``). Returns ``None`` on approval; raises
        ``PermissionDenied`` or ``BudgetExceeded`` otherwise."""
        action_id = getattr(action, "id", "?")
        action_type = getattr(action, "type", "?")
        requires_approval = bool(getattr(action, "requires_approval", False))

        if requires_approval:
            self.audit.log("action_denied", {
                "action_id": action_id, "action_type": action_type,
                "reason": "human approval required",
            })
            raise PermissionDenied(
                f"action {action_id!r} requires human approval; "
                "auto-approval is not permitted"
            )

        if action_type == "tool":
            payload = getattr(action, "payload", {}) or {}
            tool_name = payload.get("tool") or payload.get("name")
            if not tool_name:
                self.audit.log("action_denied", {
                    "action_id": action_id, "reason": "tool action without a tool name"})
                raise PermissionDenied(
                    f"tool action {action_id!r} names no tool; denied by default"
                )
            if not self.permissions.allowed(tool_name):
                self.audit.log("action_denied", {
                    "action_id": action_id, "tool": tool_name,
                    "reason": "not in allowlist"})
                raise PermissionDenied(
                    f"tool {tool_name!r} is not in the permissions allowlist"
                )

        budget_cost = getattr(action, "budget_cost", {}) or {}
        tokens = int(budget_cost.get("tokens", 0))
        tool_calls = int(budget_cost.get("tool_calls", 0))
        if tokens:
            self.budget.consume_tokens(tokens)  # raises BudgetExceeded
        for _ in range(tool_calls):
            self.budget.consume_tool_call()  # raises BudgetExceeded

        self.audit.log("action_checked", {
            "action_id": action_id, "action_type": action_type,
            "tokens": tokens, "tool_calls": tool_calls,
        })

    def stop(self, run_id: str) -> None:
        """Halt a run; recorded so ``stopped`` becomes True."""
        self._stopped_runs.add(str(run_id))
        self.audit.log("run_stopped", {"run_id": str(run_id)})
        logger.info("run stopped: run_id=%s", run_id)

    def disable_autonomous(self, *, authorized: bool = False) -> None:
        """Disable autonomous execution globally.

        Raises ``SafetyError`` unless ``authorized=True`` — the system cannot
        change its own safety controls without explicit human authorization.
        """
        if not authorized:
            raise SafetyError(
                "disable_autonomous requires explicit human authorization "
                "(authorized=True); the system cannot alter its own safety controls"
            )
        self._autonomous_enabled = False
        self.audit.log("autonomous_disabled", {})
        logger.info("autonomous execution disabled (authorized)")
