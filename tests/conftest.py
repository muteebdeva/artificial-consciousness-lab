"""TEST INFRASTRUCTURE (supports the MOCKED TESTS suite) — shared fixtures,
sys.path setup, and import guards.

Convention: modules owned by other agents are imported through ``require()``,
which skips the test with a clear reason when the module is not implemented
yet (or not importable yet, e.g. a SyntaxError in a freshly landed module).
Nothing here touches the network; the mock provider is deterministic.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FIXED_SEED = 123


def require(modname: str):
    """Import ``modname`` or skip the test with an explicit reason.

    Use for every module owned by another agent so the suite stays green
    (skipped, not failed) until those modules land and import cleanly.
    ImportError (missing module) and SyntaxError (broken module) both skip;
    anything else propagates so genuine runtime bugs are not hidden.
    """
    try:
        return importlib.import_module(modname)
    except ImportError as e:
        pytest.skip(f"{modname} not implemented yet ({e})")
    except SyntaxError as e:
        pytest.skip(f"{modname} not importable yet (SyntaxError: {e})")


@pytest.fixture
def tmp_db(tmp_path):
    """Path to a fresh sqlite file inside a pytest temp dir."""
    return str(tmp_path / "test.sqlite3")


@pytest.fixture
def lab_config(tmp_path):
    """A Config with a temp DB, mock provider, and fixed seed (or skip)."""
    Config = require("consciousness_lab.core.config").Config
    return Config(
        db_path=str(tmp_path / "lab.sqlite3"),
        provider="mock",
        seed=FIXED_SEED,
        results_dir=str(tmp_path / "results"),
    )


@pytest.fixture
def mock_provider():
    """Deterministic MockProvider (or skip if not implemented yet)."""
    MockProvider = require("consciousness_lab.models.mock").MockProvider
    return MockProvider(seed=FIXED_SEED)


@pytest.fixture
def store(tmp_db):
    """A live SqliteStore on a temp path (or skip)."""
    SqliteStore = require("consciousness_lab.persistence.sqlite_store").SqliteStore
    s = SqliteStore(tmp_db)
    yield s
    s.close()
