"""MOCKED TESTS — GlobalWorkspace selection, capacity, and broadcast.

Skips if consciousness_lab.workspace.workspace is not implemented yet.
"""

from __future__ import annotations

import time

import pytest

from conftest import require

ws_mod = require("consciousness_lab.workspace.workspace")
GlobalWorkspace = ws_mod.GlobalWorkspace
WorkspaceItem = ws_mod.WorkspaceItem


def _item(content: str, salience: float = 0.5, relevance=None, novelty=None,
          ts: float | None = None) -> WorkspaceItem:
    return WorkspaceItem(
        id=f"id-{content[:8]}-{salience}",
        content=content,
        source="perception",
        salience=salience,
        relevance=relevance,
        novelty=novelty,
        timestamp=ts if ts is not None else time.time(),
    )


def test_priority_multiplies_components_with_none_defaulting_to_one():
    item = _item("x", salience=0.5, relevance=None, novelty=None)
    assert item.priority() == pytest.approx(0.5)
    item2 = _item("x", salience=0.5, relevance=0.5, novelty=0.5)
    assert item2.priority() == pytest.approx(0.125)


def test_select_ranks_by_priority_descending():
    ws = GlobalWorkspace(capacity=7)
    items = [_item("low", salience=0.1), _item("high", salience=0.9),
             _item("mid", salience=0.5)]
    selected = ws.select(items)
    assert [i.content for i in selected] == ["high", "mid", "low"]


def test_select_respects_capacity():
    ws = GlobalWorkspace(capacity=3)
    items = [_item(f"item{i}", salience=i / 10) for i in range(10)]
    selected = ws.select(items)
    assert len(selected) == 3
    assert selected[0].content == "item9"  # highest salience first


def test_select_tie_break_earliest_timestamp():
    ws = GlobalWorkspace(capacity=2)
    old = _item("old", salience=0.5, ts=1000.0)
    new = _item("new", salience=0.5, ts=2000.0)
    selected = ws.select([new, old])
    assert [i.content for i in selected] == ["old", "new"]


def test_broadcast_reaches_all_consumers():
    ws = GlobalWorkspace(capacity=7)
    seen: list[list] = []
    ws.register_consumer(lambda items: seen.append([i.content for i in items]))
    ws.register_consumer(lambda items: seen.append(["second"]))
    items = ws.select([_item("a"), _item("b")])
    ws.broadcast(items)
    assert seen[0] == ["a", "b"]
    assert seen[1] == ["second"]


def test_failing_consumer_does_not_break_broadcast():
    ws = GlobalWorkspace(capacity=7)
    calls = []

    def bad(items):
        raise RuntimeError("consumer exploded")

    ws.register_consumer(bad)
    ws.register_consumer(lambda items: calls.append(True))
    ws.broadcast(ws.select([_item("a")]))
    assert calls == [True]  # good consumer still ran


def test_snapshot_reports_state_and_cycle_increments():
    ws = GlobalWorkspace(capacity=4)
    assert ws.snapshot().cycle == 0
    ws.select([_item("a"), _item("b")])
    snap = ws.snapshot()
    assert snap.capacity == 4
    assert snap.cycle == 1
    assert len(snap.items) == 2


def test_clear_empties_state():
    ws = GlobalWorkspace(capacity=4)
    ws.select([_item("a")])
    ws.clear()
    assert ws.snapshot().items == []
