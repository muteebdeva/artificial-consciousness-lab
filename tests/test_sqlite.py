"""REAL TESTS — SqliteStore durability and transaction semantics.

sqlite3 is stdlib, so these are real tests (not mocked) of the store itself;
they skip only if consciousness_lab.persistence.sqlite_store is not
implemented yet.
"""

from __future__ import annotations

import sqlite3

import pytest

from conftest import require

SqliteStore = require("consciousness_lab.persistence.sqlite_store").SqliteStore

DDL = "CREATE TABLE IF NOT EXISTS kv (id INTEGER PRIMARY KEY AUTOINCREMENT, k TEXT, v TEXT)"


def test_wal_mode_enabled(tmp_db):
    store = SqliteStore(tmp_db)
    try:
        mode = store.execute_read("PRAGMA journal_mode")[0]["journal_mode"]
        assert mode.lower() == "wal"
    finally:
        store.close()


def test_write_then_read_roundtrip(tmp_db):
    store = SqliteStore(tmp_db)
    try:
        store.ensure_table("kv", DDL)
        rowid = store.execute_write("INSERT INTO kv (k, v) VALUES (?, ?)", ("a", "b"))
        assert isinstance(rowid, int) and rowid >= 1
        rows = store.execute_read("SELECT k, v FROM kv WHERE id = ?", (rowid,))
        assert rows == [{"k": "a", "v": "b"}]
    finally:
        store.close()


def test_insert_helper_returns_id(tmp_db):
    store = SqliteStore(tmp_db)
    try:
        store.ensure_table("kv", DDL)
        rowid = store.insert("kv", {"k": "x", "v": "y's \"quoted\" value"})
        rows = store.execute_read("SELECT v FROM kv WHERE id = ?", (rowid,))
        assert rows[0]["v"] == "y's \"quoted\" value"  # parameterized, no injection
    finally:
        store.close()


def test_data_survives_close_and_reopen(tmp_db):
    store = SqliteStore(tmp_db)
    store.ensure_table("kv", DDL)
    store.execute_write("INSERT INTO kv (k, v) VALUES (?, ?)", ("persist", "me"))
    store.close()

    reopened = SqliteStore(tmp_db)  # new object, same path
    try:
        rows = reopened.execute_read("SELECT v FROM kv WHERE k = ?", ("persist",))
        assert rows and rows[0]["v"] == "me"
    finally:
        reopened.close()


def test_transaction_commits_all_statements(tmp_db):
    store = SqliteStore(tmp_db)
    try:
        store.ensure_table("kv", DDL)
        # transaction() yields the store itself for multi-statement writes
        with store.transaction() as tx:
            tx.execute_write("INSERT INTO kv (k, v) VALUES (?, ?)", ("t1", "1"))
            tx.execute_write("INSERT INTO kv (k, v) VALUES (?, ?)", ("t2", "2"))
        rows = store.execute_read("SELECT COUNT(*) AS n FROM kv")
        assert rows[0]["n"] == 2
    finally:
        store.close()


def test_transaction_rolls_back_on_error(tmp_db):
    store = SqliteStore(tmp_db)
    try:
        store.ensure_table("kv", DDL)
        with pytest.raises(RuntimeError):
            with store.transaction() as tx:
                tx.execute_write("INSERT INTO kv (k, v) VALUES (?, ?)", ("t1", "1"))
                raise RuntimeError("boom")
        rows = store.execute_read("SELECT COUNT(*) AS n FROM kv")
        assert rows[0]["n"] == 0
    finally:
        store.close()


def test_busy_timeout_is_set(tmp_db):
    store = SqliteStore(tmp_db)
    try:
        row = store.execute_read("PRAGMA busy_timeout")[0]
        timeout = row.get("busy_timeout", row.get("timeout"))
        assert timeout is not None and timeout > 0
    finally:
        store.close()


def test_schema_version_and_migrate(tmp_db):
    store = SqliteStore(tmp_db)
    try:
        assert store.schema_version == 0
        store.migrate({1: [DDL]})
        assert store.schema_version == 1
        # migrate is idempotent: re-running does not fail or duplicate
        store.migrate({1: [DDL]})
        assert store.schema_version == 1
    finally:
        store.close()


def test_ensure_table_idempotent(tmp_db):
    store = SqliteStore(tmp_db)
    try:
        store.ensure_table("kv", DDL)
        store.ensure_table("kv", DDL)  # must not raise
    finally:
        store.close()


def test_creates_parent_directories(tmp_path):
    nested = str(tmp_path / "a" / "b" / "c.sqlite3")
    store = SqliteStore(nested)
    try:
        store.ensure_table("kv", DDL)
        store.execute_write("INSERT INTO kv (k, v) VALUES (?, ?)", ("k", "v"))
    finally:
        store.close()
    # verify with a raw connection that the file is a real sqlite db
    conn = sqlite3.connect(nested)
    try:
        n = conn.execute("SELECT COUNT(*) FROM kv").fetchone()[0]
        assert n == 1
    finally:
        conn.close()
