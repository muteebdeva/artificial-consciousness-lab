"""MOCKED TESTS — perception sanitization and attention prioritization.

Skips if consciousness_lab.perception / consciousness_lab.attention are not
implemented yet.
"""

from __future__ import annotations

import pytest

from conftest import require

perception_mod = require("consciousness_lab.perception.perception")
attention_mod = require("consciousness_lab.attention.attention")
ws_mod = require("consciousness_lab.workspace.workspace")

Perception = perception_mod.Perception
Attention = attention_mod.Attention
WorkspaceItem = ws_mod.WorkspaceItem


def _item(content: str, salience: float = 0.5) -> WorkspaceItem:
    return WorkspaceItem(id=f"i-{abs(hash(content)) % 10**8}", content=content,
                         source="perception", salience=salience)


def test_ingest_plain_string():
    obs = Perception().ingest("hello world")
    assert obs.content == "hello world"
    assert obs.modality == "text"
    assert obs.timestamp > 0
    assert obs.id


def test_ingest_strips_control_chars_but_keeps_newlines():
    obs = Perception().ingest("line1\x00\x01\nline2\tend")
    assert "\x00" not in obs.content
    assert "\x01" not in obs.content
    assert "\n" in obs.content
    assert "\t" in obs.content


def test_ingest_caps_length_at_20000():
    obs = Perception().ingest("x" * 25_000)
    assert len(obs.content) <= 20_000


def test_ingest_dict_form():
    obs = Perception().ingest({"modality": "task", "content": "do things",
                               "metadata": {"cycle": 1}, "source": "demo"})
    assert obs.modality == "task"
    assert obs.content == "do things"
    assert obs.metadata == {"cycle": 1}
    assert obs.source == "demo"


def test_ingest_unknown_modality_raises():
    with pytest.raises(ValueError):
        Perception().ingest({"modality": "telepathy", "content": "x"})


def test_attention_prioritizes_goal_relevant_items():
    att = Attention()
    goals = [{"description": "summarize the quarterly revenue report"}]
    # equal salience so the goal-overlap term decides the order
    items = [
        _item("the weather is sunny", salience=0.5),
        _item("quarterly revenue grew 12 percent", salience=0.5),
    ]
    ranked = att.prioritize(items, goals, None)
    assert ranked[0].content == "quarterly revenue grew 12 percent"
    assert all(i.attention_score is not None for i in ranked)


def test_attention_sorted_descending_by_score():
    att = Attention()
    items = [_item("a", salience=0.1), _item("b", salience=0.9), _item("c", salience=0.5)]
    ranked = att.prioritize(items, [], None)
    scores = [i.attention_score for i in ranked]
    assert scores == sorted(scores, reverse=True)


def test_attention_accepts_goal_objects():
    goals_mod = require("consciousness_lab.goals.goals")
    att = Attention()
    goal = goals_mod.Goal(id="g1", description="plan the revenue forecast")
    # equal salience so the goal-overlap term decides the order
    items = [_item("revenue forecast draft", salience=0.5),
             _item("unrelated poem", salience=0.5)]
    ranked = att.prioritize(items, [goal], None)
    assert ranked[0].content == "revenue forecast draft"
