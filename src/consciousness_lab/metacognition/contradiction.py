"""Contradiction detection: a documented token-level heuristic.

What it does
------------
Flags pairs of statements that share many of the same words but differ in a
negation cue (``not``, ``never``, ``no``, ``n't``, ...) or in an explicit
number. ``check`` compares new content against a list of stored records;
``check_pair`` compares two raw strings.

What it is NOT
--------------
Natural-language processing or understanding. It has no notion of meaning,
paraphrase, scope, irony, or context: "Paris is not boring" vs "Paris is not
safe" share tokens and both contain ``not``, so no asymmetry is flagged — but
"The Eiffel Tower is not in Paris" vs "The Eiffel Tower is in Paris" is. It
will MISS contradictions expressed with different wording ("Paris is the
capital of France" vs "Lyon is the capital of France" has low overlap) and may
FALSE-POSITIVE on contrastive but compatible statements ("Coffee is hot,
however tea is better" contains ``however`` with high token overlap). Treat
every flag as a candidate for review, not a verdict.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # Avoid a hard dependency on the memory package at import time.
    from consciousness_lab.memory.records import MemoryRecord

log = logging.getLogger(__name__)

#: Negation / contrast cues. Presence in one statement but not the other
#: (negation asymmetry) is the primary contradiction signal.
NEGATION_CUES = (
    "not",
    "never",
    "no",
    "n't",
    "false",
    "incorrect",
    "opposite",
    "however",
    "but",
    "despite",
    "contradicts",
    "contradiction",
    "contrary",
    "disagree",
    "denies",
    "deny",
    "wrong",
    "neither",
    "nor",
    "cannot",
)

#: Minimum token-overlap (Jaccard) before a pair is even considered.
OVERLAP_THRESHOLD = 0.5

#: Cue strength used when the signal is an explicit numeric mismatch.
NUMERIC_MISMATCH_STRENGTH = 0.9

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class Contradiction:
    """A flagged candidate contradiction between two pieces of content."""

    id: str
    record_a_id: str | None
    record_b_id: str | None
    description: str
    severity: float  # 0..1
    timestamp: float = field(default_factory=time.time)


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _numbers(text: str) -> set[str]:
    return set(_NUMBER_RE.findall(text.lower()))


def _cues(tokens: set[str]) -> set[str]:
    found: set[str] = set()
    for token in tokens:
        for cue in NEGATION_CUES:
            if cue == "n't":
                if "n't" in token or token.endswith("n't"):
                    found.add("n't")
            elif token == cue:
                found.add(cue)
    return found


class ContradictionDetector:
    """Flag candidate contradictions with a token-overlap + negation-cue heuristic."""

    def check(self, new_content: str, existing: list) -> list[Contradiction]:
        """Compare ``new_content`` against each record in ``existing``.

        ``existing`` is typed as ``list[MemoryRecord]`` in the contract; at
        runtime any object exposing ``.content`` and ``.id`` works (duck-typed
        so this module never imports the memory package — avoiding cycles).
        The new content is not a stored record, so ``record_a_id`` is ``None``
        and ``record_b_id`` is the matched record's id.
        """
        contradictions: list[Contradiction] = []
        for record in existing:
            content = getattr(record, "content", "") or ""
            hit = self.check_pair(new_content, content)
            if hit is not None:
                hit.record_a_id = None
                hit.record_b_id = getattr(record, "id", None)
                contradictions.append(hit)
        return contradictions

    def check_pair(self, a: str, b: str) -> Contradiction | None:
        """Return a :class:`Contradiction` if ``a`` and ``b`` look contradictory.

        Flags when token overlap (Jaccard) >= ``OVERLAP_THRESHOLD`` AND either
        (negation asymmetry) one side contains negation cues the other lacks,
        or (numeric mismatch) both contain numbers that differ.
        ``severity = overlap * cue_strength``, in 0..1.
        """
        tokens_a, tokens_b = _tokens(a or ""), _tokens(b or "")
        union = tokens_a | tokens_b
        if not union:
            return None
        overlap = len(tokens_a & tokens_b) / len(union)
        if overlap < OVERLAP_THRESHOLD:
            return None

        cues_a, cues_b = _cues(tokens_a), _cues(tokens_b)
        asymmetric_cues = cues_a ^ cues_b
        negation_asymmetry = bool(asymmetric_cues)

        nums_a, nums_b = _numbers(a or ""), _numbers(b or "")
        numeric_mismatch = bool(nums_a and nums_b and nums_a != nums_b)

        if not (negation_asymmetry or numeric_mismatch):
            return None

        if numeric_mismatch:
            cue_strength = NUMERIC_MISMATCH_STRENGTH
            reason = (
                f"numeric mismatch ({sorted(nums_a)} vs {sorted(nums_b)}) "
                f"at token overlap {overlap:.2f}"
            )
        else:
            cue_strength = min(1.0, 0.5 + 0.2 * len(asymmetric_cues))
            reason = (
                f"negation asymmetry (cues {sorted(asymmetric_cues)} present in "
                f"only one statement) at token overlap {overlap:.2f}"
            )

        severity = max(0.0, min(1.0, overlap * cue_strength))
        return Contradiction(
            id=uuid.uuid4().hex,
            record_a_id=None,
            record_b_id=None,
            description=f"Candidate contradiction: {reason}: {a[:120]!r} vs {b[:120]!r}",
            severity=severity,
            timestamp=time.time(),
        )
