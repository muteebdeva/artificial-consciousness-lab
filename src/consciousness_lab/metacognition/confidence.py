"""Confidence estimation: a documented heuristic for scoring how much the system
should trust a piece of information.

What it does
------------
Combines evidence-source types, agreement across sources, and a base rate into
a single number in ``[0.01, 0.99]`` plus a human-readable ``basis`` string
explaining the computation.

What it is NOT
--------------
A calibrated probability, a measure of truth, or evidence of understanding.
The weights (+0.1 per verified source, −0.15 for hypothesis-only evidence,
scaling by agreement) are *chosen constants*, not learned or validated. The
number is only meaningful relative to other numbers produced by this same
heuristic. ``recency`` and ``n_evidence`` are reported in the basis/factors
but do not change the value (documented so callers don't assume otherwise).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

#: Source types understood by the heuristic. Unknown strings are tolerated and
#: treated as neutral (no bonus, no penalty).
VERIFIED_SOURCE = "verified"
HYPOTHESIS_SOURCE = "model_hypothesis"

#: Chosen constant: bonus per verified source, capped by MAX_VERIFIED_COUNT.
VERIFIED_BONUS = 0.1
MAX_VERIFIED_COUNT = 3

#: Chosen constant: penalty when ALL evidence is model hypotheses.
HYPOTHESIS_ONLY_PENALTY = 0.15

MIN_CONFIDENCE = 0.01
MAX_CONFIDENCE = 0.99


@dataclass
class Confidence:
    """Result of a confidence estimate.

    ``value`` is the heuristic score (0.01..0.99); ``basis`` is a
    human-readable justification; ``factors`` records the numeric inputs so the
    computation is auditable.
    """

    value: float  # 0..1
    basis: str  # human-readable justification
    factors: dict = field(default_factory=dict)


class ConfidenceEstimator:
    """Estimate confidence from evidence-source context via a fixed heuristic.

    ``context`` keys (per contract):
      - ``sources``: list of source-type strings (``"verified"``,
        ``"user_provided"``, ``"model_hypothesis"``, ``"inferred"``)
      - ``n_evidence``: number of evidence items (reported, not weighted)
      - ``agreement``: 0..1 agreement across sources
      - ``recency``: recency score (reported, not weighted)
      - ``base_rate``: prior, default 0.5
    """

    def estimate(self, context: dict) -> Confidence:
        """Compute a confidence score from ``context``.

        Heuristic (fixed, documented): start at ``base_rate``; add
        ``VERIFIED_BONUS`` per verified source (capped at
        ``MAX_VERIFIED_COUNT`` sources); subtract ``HYPOTHESIS_ONLY_PENALTY``
        when the evidence consists solely of model hypotheses; multiply by
        ``(0.5 + 0.5 * agreement)`` to scale toward 0.5 when sources disagree;
        clamp to ``[0.01, 0.99]``.
        """
        if not isinstance(context, dict):
            raise TypeError("context must be a dict")

        sources = context.get("sources", []) or []
        n_evidence = int(context.get("n_evidence", 0))
        agreement = self._clamp01(context.get("agreement", 0.5), "agreement")
        recency = self._clamp01(context.get("recency", 0.5), "recency")
        base_rate = self._clamp01(context.get("base_rate", 0.5), "base_rate")

        n_verified = sum(1 for s in sources if s == VERIFIED_SOURCE)
        verified_bonus = VERIFIED_BONUS * min(n_verified, MAX_VERIFIED_COUNT)

        hypothesis_only = bool(sources) and all(
            s == HYPOTHESIS_SOURCE for s in sources
        )
        hypothesis_penalty = HYPOTHESIS_ONLY_PENALTY if hypothesis_only else 0.0

        agreement_multiplier = 0.5 + 0.5 * agreement

        value = (base_rate + verified_bonus - hypothesis_penalty) * agreement_multiplier
        value = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, value))

        parts = [f"base_rate={base_rate:.2f}"]
        if verified_bonus:
            parts.append(f"+{verified_bonus:.2f} for {n_verified} verified source(s)")
        if hypothesis_penalty:
            parts.append(f"-{hypothesis_penalty:.2f} (evidence is model hypotheses only)")
        parts.append(f"scaled by agreement={agreement:.2f} (x{agreement_multiplier:.2f})")
        parts.append(f"n_evidence={n_evidence}, recency={recency:.2f} (reported only)")
        parts.append(f"clamped to [{MIN_CONFIDENCE}, {MAX_CONFIDENCE}]")
        basis = "; ".join(parts) + f" -> {value:.3f}"

        factors = {
            "base_rate": base_rate,
            "n_verified": n_verified,
            "verified_bonus": verified_bonus,
            "hypothesis_only": hypothesis_only,
            "hypothesis_penalty": hypothesis_penalty,
            "agreement": agreement,
            "agreement_multiplier": agreement_multiplier,
            "n_evidence": n_evidence,
            "recency": recency,
        }
        return Confidence(value=value, basis=basis, factors=factors)

    @staticmethod
    def _clamp01(raw: object, name: str) -> float:
        try:
            value = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a number") from exc
        return max(0.0, min(1.0, value))
