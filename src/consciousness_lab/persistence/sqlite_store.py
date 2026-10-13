"""SQLite persistence layer: single file, WAL mode, transaction-per-write.

What it does:
    - :class:`SqliteStore` wraps one SQLite database file with WAL journaling,
      a busy timeout, parameterized queries, and one transaction per write.
    - Schema versioning via the ``_schema(version INTEGER)`` table and
      :meth:`SqliteStore.migrate`.
    - Owning modules register their own tables with :meth:`ensure_table`;
      the DDL strings live in those modules, not here.

Limits:
    - Single-process oriented: a re-entrant lock serialises access, but
      concurrent writers in other processes rely on SQLite's own locking and
      the busy timeout rather than application-level coordination.
    - ``PRAGMA synchronous=NORMAL`` is used with WAL; this is the documented
      safe combination for WAL mode, but it is not ``FULL`` durability.
    - Table/column names in :meth:`insert` are validated as identifiers and
      quoted; values are always bound parameters. Raw SQL passed to
      :meth:`execute_write`/``execute_read`` is the caller's responsibility.
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Iterator

from ..core.errors import LabError

__all__ = ["SqliteStore"]

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BUSY_TIMEOUT_MS = 5000


def _check_identifier(name: str, what: str) -> str:
    if not isinstance(name, str) or not _IDENTIFIER_RE.match(name):
        raise LabError(f"{what} must be a plain SQL identifier, got {name!r}")
    return name


class SqliteStore:
    """A single SQLite file with WAL mode and transaction-per-write semantics.

    Args:
        path: Database file path. Parent directories are created.
    """

    def __init__(self, path: str) -> None:
        if not path or not isinstance(path, str):
            raise LabError("SqliteStore path must be a non-empty string")
        parent = os.path.dirname(os.path.abspath(path))
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as exc:
            raise LabError(f"cannot create directory for {path!r}: {exc}") from exc
        self._path = path
        self._lock = threading.RLock()
        self._tx_depth = 0
        try:
            # isolation_level=None => autocommit; transactions are managed
            # explicitly with BEGIN/COMMIT so transaction-per-write and the
            # transaction() context manager behave identically.
            self._conn = sqlite3.connect(path, timeout=30.0, check_same_thread=False, isolation_level=None)
        except sqlite3.Error as exc:
            raise LabError(f"cannot open sqlite database {path!r}: {exc}") from exc
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            self._conn.execute("PRAGMA synchronous=NORMAL")

    # -- schema ----------------------------------------------------------

    def ensure_table(self, name: str, ddl: str) -> None:
        """Create a table if it does not exist.

        Args:
            name: Table name (validated as a plain identifier).
            ddl: Full ``CREATE TABLE`` statement, owned by the calling module.
        """
        _check_identifier(name, "table name")
        if not ddl or not isinstance(ddl, str):
            raise LabError("ddl must be a non-empty string")
        self.execute_write(ddl)

    # -- reads/writes ----------------------------------------------------

    def execute_write(self, sql: str, params: tuple = ()) -> int:
        """Execute one write statement in its own transaction.

        Returns:
            The cursor's ``lastrowid``.
        """
        with self._lock:
            try:
                if self._tx_depth == 0:
                    self._conn.execute("BEGIN")
                cursor = self._conn.execute(sql, params)
                rowid = cursor.lastrowid
                if self._tx_depth == 0:
                    self._conn.execute("COMMIT")
                return rowid if rowid is not None else 0
            except Exception:
                if self._tx_depth == 0:
                    try:
                        self._conn.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                raise

    def execute_read(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        """Execute a read query; returns rows as a list of dicts."""
        with self._lock:
            try:
                cursor = self._conn.execute(sql, params)
                return [dict(row) for row in cursor.fetchall()]
            except sqlite3.Error as exc:
                raise LabError(f"sqlite read failed: {exc}") from exc

    def insert(self, table: str, record: dict[str, Any]) -> str:
        """Insert one record with bound parameters.

        Args:
            table: Table name (identifier-validated and quoted).
            record: Column-name → value mapping (names identifier-validated).

        Returns:
            The inserted row id as a string (``""`` when the table has no
            integer row id to report).
        """
        _check_identifier(table, "table name")
        if not record:
            raise LabError("insert requires a non-empty record")
        columns = [_check_identifier(col, "column name") for col in record.keys()]
        placeholders = ", ".join("?" for _ in columns)
        quoted = ", ".join(f'"{col}"' for col in columns)
        sql = f'INSERT INTO "{table}" ({quoted}) VALUES ({placeholders})'
        rowid = self.execute_write(sql, tuple(record[col] for col in record.keys()))
        return str(rowid) if rowid else ""

    @contextmanager
    def transaction(self) -> Iterator["SqliteStore"]:
        """Context manager for multi-statement writes.

        Commits when the block exits cleanly, rolls back on exception.
        Nested use joins the outer transaction.
        """
        with self._lock:
            if self._tx_depth == 0:
                self._conn.execute("BEGIN")
            self._tx_depth += 1
        try:
            yield self
        except Exception:
            with self._lock:
                self._tx_depth -= 1
                if self._tx_depth == 0:
                    self._conn.execute("ROLLBACK")
            raise
        else:
            with self._lock:
                self._tx_depth -= 1
                if self._tx_depth == 0:
                    self._conn.execute("COMMIT")

    # -- migrations ------------------------------------------------------

    @property
    def schema_version(self) -> int:
        """Current schema version; 0 when the ``_schema`` table is absent."""
        tables = self.execute_read(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='_schema'"
        )
        if not tables:
            return 0
        rows = self.execute_read("SELECT version FROM _schema ORDER BY version DESC LIMIT 1")
        return int(rows[0]["version"]) if rows else 0

    def migrate(self, migrations: dict[int, list[str]]) -> None:
        """Apply pending migrations in version order, recording each in ``_schema``.

        Args:
            migrations: ``{version: [sql statements]}``. Only versions above
                the current one run, each inside a single transaction.
        """
        if not migrations:
            return
        self.ensure_table("_schema", "CREATE TABLE IF NOT EXISTS _schema (version INTEGER NOT NULL)")
        current = self.schema_version
        with self.transaction():
            for version in sorted(migrations):
                if version <= current:
                    continue
                statements = migrations[version]
                if not statements:
                    raise LabError(f"migration {version} has no statements")
                for statement in statements:
                    self.execute_write(statement)
                self.execute_write("INSERT INTO _schema (version) VALUES (?)", (version,))

    def close(self) -> None:
        """Close the database connection. Safe to call more than once."""
        with self._lock:
            if getattr(self, "_conn", None) is not None:
                try:
                    self._conn.close()
                except sqlite3.Error:
                    pass
                self._conn = None  # type: ignore[assignment]
