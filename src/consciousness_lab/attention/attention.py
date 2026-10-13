"""Attention: score candidate workspace items against active goals.

What it does: ranks :class:`~consciousness_lab.workspace.workspace.WorkspaceItem`
candidates so goal-relevant items surface first, setting each item's
``attention_score`` in place and returning the items sorted descending.

Heuristic (documented, not learned):
``score = salience * (1 + goal_overlap)``, where ``goal_overlap`` is the
fraction of the goal's keyword tokens also present in the item's content
(the maximum over all goals when several are active).

Limits:
- Pure keyword overlap: no stemming, no synonyms, no semantic similarity.
  "revenue" and "revenues" do not match; paraphrase is invisible to this
  scorer. A learned ranker may replace it later.
- Items and goals are duck-typed (only ``salience``/``content``/
  ``attention_score`` on items, ``description`` or ``["description"]`` on
  goals), so this module never imports the workspace or goals packages and
  cannot create an import cycle.
- Sorting is deterministic: ties break on ``(timestamp, id)``.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable

try:  # core.logging_setup is owned by another agent; fall back until it lands
    from consciousness_lab.core.logging_setup import setup_logging

    _log = setup_logging()
except Exception:  # pragma: no cover - import fallback
    _log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    """Lowercase alphanumeric word tokens of ``text``."""
    return set(_TOKEN_RE.findall(text.lower()))


def _goal_description(goal: Any) -> str:
    """Extract the description from a Goal object or a dict with "description"."""
    if isinstance(goal, dict):
        return str(goal.get("description", ""))
    return str(getattr(goal, "description", ""))


def _overlap_fraction(goal_tokens: set[str], item_tokens: set[str]) -> float:
    """Fraction of goal keyword tokens present in the item's tokens."""
    if not goal_tokens:
        return 0.0
    return len(goal_tokens & item_tokens) / len(goal_tokens)


class Attention:
    """Prioritize workspace candidates by goal keyword overlap.

    Stateless: scoring uses only the items, goals, and salience values
    passed in. ``workspace_state`` is accepted per the interface contract
    but currently unused (reserved for future context-aware scoring);
    passing ``None`` is fine.
    """

    def prioritize(
        self,
        items: list,
        goals: Iterable,
        workspace_state: Any | None = None,
    ) -> list:
        """Score, annotate, and sort ``items`` by goal relevance.

        Sets ``item.attention_score`` on every item (in place) and returns
        the items sorted by score descending. Items lacking a numeric
        ``salience`` are treated as salience 0. An empty ``goals`` iterable
        reduces the score to raw salience, preserving input order via the
        deterministic tie-break.

        Args:
            items: candidates with ``salience``, ``content``,
                ``attention_score`` (settable), ``timestamp``, ``id``.
            goals: ``Goal`` objects or dicts with a ``"description"`` key.
            workspace_state: accepted for interface compatibility; unused.
        """
        _ = workspace_state  # reserved for future context-aware scoring
        goal_token_sets = [
            _tokens(_goal_description(g)) for g in (goals or [])
        ]
        goal_token_sets = [t for t in goal_token_sets if t]

        scored: list[tuple[float, float, str, Any]] = []
        for item in items:
            salience = getattr(item, "salience", 0.0)
            try:
                salience = max(0.0, min(1.0, float(salience)))
            except (TypeError, ValueError):
                salience = 0.0
            content = str(getattr(item, "content", ""))
            item_tokens = _tokens(content)
            overlap = (
                max(
                    _overlap_fraction(gt, item_tokens)
                    for gt in goal_token_sets
                )
                if goal_token_sets
                else 0.0
            )
            score = salience * (1.0 + overlap)
            try:
                item.attention_score = score
            except AttributeError:
                _log.warning(
                    "attention could not set attention_score",
                    extra={"item_id": getattr(item, "id", None)},
                )
            scored.append(
                (score, float(getattr(item, "timestamp", 0.0) or 0.0),
                 str(getattr(item, "id", "")), item)
            )

        scored.sort(key=lambda t: (-t[0], t[1], t[2]))
        return [t[3] for t in scored]
