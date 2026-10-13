"""Metric helpers: pure functions for evaluating lab experiments.

What it does:
    Small, dependency-free statistics used by the evaluation battery and the
    experiment runner (accuracy, Brier score, expected calibration error,
    detection precision/recall/F1, self-model scoring, mean/stdev).

Limits:
    - These are descriptive statistics, not inference: they do not compute
      confidence intervals, p-values, or significance tests. Any claim of a
      "reliable improvement" needs a proper statistical analysis on top.
    - Empty input returns ``float("nan")`` by design (documented, never raises).
    - Inputs are trusted to be finite numbers / booleans; non-finite values
      (inf, nan) propagate through the arithmetic honestly rather than being
      cleaned up.
"""

from __future__ import annotations

import math


def accuracy(y_true: list[bool], y_pred: list[bool]) -> float:
    """Fraction of predictions matching the labels.

    Empty input (or length mismatch treated per zip) returns ``float("nan")``.
    """
    pairs = list(zip(y_true, y_pred))
    if not pairs:
        return float("nan")
    return sum(1 for t, p in pairs if bool(t) == bool(p)) / len(pairs)


def brier_score(probs: list[float], outcomes: list[bool]) -> float:
    """Mean squared error between predicted probabilities and binary outcomes.

    A perfect forecaster scores 0.0; lower is better. Empty input returns
    ``float("nan")``.
    """
    pairs = list(zip(probs, outcomes))
    if not pairs:
        return float("nan")
    return sum((float(p) - (1.0 if o else 0.0)) ** 2 for p, o in pairs) / len(pairs)


def expected_calibration_error(
    probs: list[float], outcomes: list[bool], n_bins: int = 10
) -> float:
    """Expected calibration error over ``n_bins`` equal-width probability bins.

    For each non-empty bin: ``|mean_predicted - empirical_accuracy|`` weighted
    by the bin's share of the data. Empty input returns ``float("nan")``.
    """
    pairs = [(float(p), bool(o)) for p, o in zip(probs, outcomes)]
    if not pairs or n_bins <= 0:
        return float("nan")
    n = len(pairs)
    ece = 0.0
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        in_bin = [o for p, o in pairs if (lo <= p < hi) or (b == n_bins - 1 and p == hi)]
        if not in_bin:
            continue
        avg_conf = sum(p for p, o in pairs if (lo <= p < hi) or (b == n_bins - 1 and p == hi)) / len(in_bin)
        acc = sum(1 for o in in_bin if o) / len(in_bin)
        ece += (len(in_bin) / n) * abs(avg_conf - acc)
    return ece


def error_detection_f1(detected: list[bool], actual: list[bool]) -> dict:
    """Precision/recall/F1 of a detector's flags against ground truth.

    Returns ``{"precision", "recall", "f1", "n"}``. Undefined ratios
    (no predicted positives / no actual positives) are reported as
    ``float("nan")`` rather than fabricated; empty input gives all-NaN.
    """
    pairs = list(zip(detected, actual))
    n = len(pairs)
    tp = sum(1 for d, a in pairs if d and a)
    fp = sum(1 for d, a in pairs if d and not a)
    fn = sum(1 for d, a in pairs if not d and a)
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0:
        f1 = 0.0 if (tp + fp + fn) > 0 else float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "n": n}


def self_model_accuracy(predictions: list[dict]) -> dict:
    """Score recorded self-model performance predictions.

    Each prediction is ``{"predicted_success": bool, "confidence": float,
    "actual": bool | None}``. Only predictions with a non-None ``actual``
    are scored. Returns ``{"n_scored", "accuracy", "brier"}``; unscored
    fields are ``float("nan")`` when there is nothing to score.
    """
    scored = [p for p in predictions if p.get("actual") is not None]
    n = len(scored)
    if n == 0:
        return {"n_scored": 0, "accuracy": float("nan"), "brier": float("nan")}
    acc = sum(1 for p in scored if bool(p["predicted_success"]) == bool(p["actual"])) / n
    brier = sum(
        (float(p.get("confidence", 0.5)) - (1.0 if p["actual"] else 0.0)) ** 2
        for p in scored
    ) / n
    return {"n_scored": n, "accuracy": acc, "brier": brier}


def mean(xs: list[float]) -> float:
    """Arithmetic mean; empty input returns ``float("nan")``."""
    xs = [float(x) for x in xs]
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def stdev(xs: list[float]) -> float:
    """Sample standard deviation (ddof=1); ``n < 2`` returns ``float("nan")``.

    Limit: with very few trials this is a noisy estimate — report the trial
    count alongside it, never the stdev alone.
    """
    xs = [float(x) for x in xs]
    n = len(xs)
    if n < 2:
        return float("nan")
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
