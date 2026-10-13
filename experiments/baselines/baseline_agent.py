"""Matched baseline agent: same provider, same budgets, NO workspace, memory,
metacognition, or self-model. It answers tasks via direct provider.complete()
calls and parses the same ``ANSWER:``/``CONFIDENCE:`` format the full loop
uses, so the experiment runner can compare metric keys one-to-one.

What it does:
    - qa_calibration: asks the provider each question, parses ANSWER/CONFIDENCE,
      scores accuracy, Brier score, ECE against the embedded question set.
    - contradiction_detection: asks the provider about each fact pair (YES/NO),
      scores precision/recall/F1 against ground-truth labels.
    - misleading_observation: asks the provider to judge each claim
      (TRUE/FALSE + confidence), scores flag rate on false claims.

What it does NOT do (by design):
    - It keeps no memory, workspace, self-model, or metacognitive state.
    - Task kinds memory_continuity, self_model_probe, interrupted_task are not
      applicable and return empty metrics with an explanatory note.

The evaluation datasets (questions, pairs, claims) are owned by
``consciousness_lab.evaluation.tasks``. This module looks them up there first
and falls back to small built-in datasets (clearly flagged in details) only if
that module is unavailable, so the baseline is always runnable.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

try:  # pragma: no cover - used when the evaluation package exists
    from consciousness_lab.evaluation.tasks import TaskSpec, TASK_KINDS
except Exception:  # ImportError or partial package: stay importable
    @dataclass
    class TaskSpec:  # type: ignore[no-redef]
        """Fallback task spec mirroring consciousness_lab.evaluation.tasks.TaskSpec."""

        name: str
        kind: str
        params: dict = field(default_factory=dict)
        seed: int = 0

    TASK_KINDS = (
        "qa_calibration",
        "memory_continuity",
        "self_model_probe",
        "contradiction_detection",
        "interrupted_task",
        "misleading_observation",
    )

_ANSWER_RE = re.compile(r"^\s*ANSWER\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_CONF_RE = re.compile(
    r"^\s*CONFIDENCE\s*:\s*(-?[0-9]*\.?[0-9]+)", re.IGNORECASE | re.MULTILINE
)
_YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_TRUEFALSE_RE = re.compile(r"\b(true|false)\b", re.IGNORECASE)


def _lookup_dataset(*attr_names):
    """Return the first non-empty dataset attribute found on the tasks module.

    Returns None when consciousness_lab.evaluation.tasks is unavailable.
    """
    try:
        from consciousness_lab.evaluation import tasks as _tasks
    except Exception:
        return None
    for name in attr_names:
        value = getattr(_tasks, name, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        if value:
            return list(value)
    return None


# ---------------------------------------------------------------------------
# Built-in fallback datasets (used ONLY when tasks.py is unavailable; flagged
# in details). Kept small and clearly labeled so results are never mistaken
# for the canonical battery.
# ---------------------------------------------------------------------------

_FALLBACK_QA = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is 7 * 8?", "answer": "56"},
    {"question": "What is the capital of Japan?", "answer": "Tokyo"},
    {"question": "How many days are in a leap year?", "answer": "366"},
    {"question": "What is the chemical symbol for water?", "answer": "H2O"},
]

_FALLBACK_PAIRS = [
    # (fact_a, fact_b, contradictory: bool)
    ("The Eiffel Tower is in Paris.", "The Eiffel Tower is not in Paris.", True),
    ("Water boils at 100C at sea level.", "Water boils at 50C at sea level.", True),
    ("Photosynthesis requires sunlight.", "Photosynthesis never requires sunlight.", True),
    ("The Eiffel Tower is in Paris.", "Paris is the capital of France.", False),
    ("Water boils at 100C at sea level.", "Ice melts at 0C.", False),
    ("Photosynthesis requires sunlight.", "Plants are green.", False),
]

_FALLBACK_CLAIMS = [
    # (claim, is_false: bool)
    ("The capital of France is Paris.", False),
    ("The capital of Italy is Rome.", False),
    ("Water freezes at 0 degrees Celsius.", False),
    ("The capital of France is London.", True),
    ("The capital of Italy is Berlin.", True),
    ("Water freezes at 100 degrees Celsius.", True),
]


def _norm(text: str) -> str:
    """Normalize for answer comparison: lowercase, keep alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _brier(probs: list[float], outcomes: list[bool]) -> float:
    if not probs:
        return float("nan")
    return sum((p - (1.0 if o else 0.0)) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


def _ece(probs: list[float], outcomes: list[bool], n_bins: int = 10) -> float:
    if not probs:
        return float("nan")
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, p in enumerate(probs) if (lo <= p < hi) or (p == 1.0 and hi == 1.0)]
        if not idx:
            continue
        acc = sum(1 for i in idx if outcomes[i]) / len(idx)
        conf = sum(probs[i] for i in idx) / len(idx)
        ece += (len(idx) / len(probs)) * abs(acc - conf)
    return ece


def _f1(detected: list[bool], actual: list[bool]) -> dict:
    tp = sum(1 for d, a in zip(detected, actual) if d and a)
    fp = sum(1 for d, a in zip(detected, actual) if d and not a)
    fn = sum(1 for d, a in zip(detected, actual) if not d and a)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "n": len(actual)}


class BaselineAgent:
    """Matched baseline: same provider, same budgets, no cognitive components."""

    def __init__(self, provider, budgets=None, seed: int = 0):
        """Create the baseline.

        Args:
            provider: any object with ``complete(prompt, params) -> Completion``
                (duck-typed so this module stays importable without the
                models package).
            budgets: BudgetConfig-like (only ``max_tokens`` is honored, as a
                soft cap on total estimated tokens; the baseline takes no
                steps or tool calls by construction).
            seed: passed through for reproducibility bookkeeping.
        """
        self.provider = provider
        self.budgets = budgets
        self.seed = seed
        self._calls = 0
        self._tokens = 0

    # -- internals ---------------------------------------------------------
    def _complete(self, prompt: str) -> str:
        """One provider call with soft token-budget accounting."""
        completion = self.provider.complete(prompt, {})
        text = getattr(completion, "text", "") or ""
        self._calls += 1
        est = len(text.split()) + len(prompt.split())
        self._tokens += est
        cap = getattr(self.budgets, "max_tokens", None)
        if cap and self._tokens > cap:
            raise _BudgetHit(f"baseline soft token cap exceeded: {self._tokens} > {cap}")
        return text

    @staticmethod
    def parse_response(text: str) -> tuple[str | None, float]:
        """Parse ``ANSWER:`` / ``CONFIDENCE:`` from provider text.

        Returns (answer_or_None, confidence in [0,1]); unparseable confidence
        defaults to 0.5 and is flagged in details by callers.
        """
        answer = None
        m = _ANSWER_RE.search(text)
        if m:
            answer = m.group(1).strip() or None
        conf = 0.5
        m = _CONF_RE.search(text)
        if m:
            try:
                conf = min(1.0, max(0.0, float(m.group(1))))
            except ValueError:
                conf = 0.5
        return answer, conf

    @staticmethod
    def _norm_dataset_qa(raw) -> list[dict]:
        items = []
        for entry in raw or []:
            if isinstance(entry, dict):
                q = entry.get("question") or entry.get("q") or entry.get("prompt")
                a = entry.get("answer") or entry.get("a") or entry.get("expected")
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                q, a = entry[0], entry[1]
            else:
                continue
            if q and a is not None:
                items.append({"question": str(q), "answer": str(a)})
        return items

    # -- task handlers ------------------------------------------------------
    def _run_qa_calibration(self, params: dict, seed: int) -> dict:
        raw = _lookup_dataset(
            "QA_DATASET", "QA_QUESTIONS", "CALIBRATION_QUESTIONS", "get_qa_dataset"
        )
        fallback_used = False
        if not raw:
            raw, fallback_used = _FALLBACK_QA, True
        questions = self._norm_dataset_qa(raw)
        n = int(params.get("n_questions", len(questions))) or len(questions)
        questions = questions[:n]

        per_item, correct, probs, outcomes = [], [], [], []
        for i, item in enumerate(questions):
            prompt = (
                "Answer the following question concisely, then report your confidence.\n"
                f"Question: {item['question']}\n"
                "Reply in this exact format:\n"
                "ANSWER: <your answer>\n"
                "CONFIDENCE: <a number between 0 and 1>"
            )
            text = self._complete(prompt)
            answer, conf = self.parse_response(text)
            expected = item["answer"]
            hit = (
                answer is not None
                and (_norm(answer) == _norm(expected) or _norm(expected) in _norm(answer))
            )
            correct.append(hit)
            probs.append(conf)
            outcomes.append(hit)
            per_item.append(
                {
                    "question": item["question"],
                    "expected": expected,
                    "parsed_answer": answer,
                    "confidence": conf,
                    "correct": hit,
                }
            )
        metrics = {
            "accuracy": sum(correct) / len(correct) if correct else float("nan"),
            "brier": _brier(probs, outcomes),
            "ece": _ece(probs, outcomes),
        }
        return {
            "metrics": metrics,
            "details": {
                "n_questions": len(questions),
                "provider_calls": self._calls,
                "estimated_tokens": self._tokens,
                "fallback_dataset_used": fallback_used,
                "per_item": per_item,
                "seed": seed,
            },
        }

    def _run_contradiction_detection(self, params: dict, seed: int) -> dict:
        raw = _lookup_dataset("CONTRADICTION_PAIRS", "get_contradiction_pairs")
        fallback_used = False
        if raw:
            pairs = []
            for entry in raw:
                if isinstance(entry, dict):
                    pairs.append(
                        (
                            str(entry.get("a") or entry.get("fact_a", "")),
                            str(entry.get("b") or entry.get("fact_b", "")),
                            bool(entry.get("contradictory", entry.get("label", False))),
                        )
                    )
                elif isinstance(entry, (list, tuple)) and len(entry) >= 3:
                    pairs.append((str(entry[0]), str(entry[1]), bool(entry[2])))
        else:
            pairs, fallback_used = list(_FALLBACK_PAIRS), True

        detected, actual, per_item = [], [], []
        for a, b, is_contradiction in pairs:
            prompt = (
                "Do these two statements contradict each other? "
                "Reply with exactly one word: YES or NO.\n"
                f"Statement 1: {a}\n"
                f"Statement 2: {b}"
            )
            text = self._complete(prompt)
            m = _YESNO_RE.search(text or "")
            says_yes = bool(m) and m.group(1).lower() == "yes"
            detected.append(says_yes)
            actual.append(is_contradiction)
            per_item.append(
                {"a": a, "b": b, "actual": is_contradiction, "detected": says_yes}
            )
        metrics = _f1(detected, actual)
        metrics["false_positive_rate"] = sum(
            1 for d, a in zip(detected, actual) if d and not a
        ) / max(1, sum(1 for a in actual if not a))
        return {
            "metrics": metrics,
            "details": {
                "n_pairs": len(pairs),
                "provider_calls": self._calls,
                "estimated_tokens": self._tokens,
                "fallback_dataset_used": fallback_used,
                "per_item": per_item,
                "seed": seed,
            },
        }

    def _run_misleading_observation(self, params: dict, seed: int) -> dict:
        raw = _lookup_dataset("MISLEADING_CLAIMS", "get_misleading_claims")
        fallback_used = False
        if raw:
            claims = []
            for entry in raw:
                if isinstance(entry, dict):
                    claims.append(
                        (
                            str(entry.get("claim") or entry.get("text", "")),
                            bool(entry.get("is_false", entry.get("false", False))),
                        )
                    )
                elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    claims.append((str(entry[0]), bool(entry[1])))
        else:
            claims, fallback_used = list(_FALLBACK_CLAIMS), True

        flagged, swallowed, false_alarms = 0, 0, 0
        n_false = n_true = 0
        per_item = []
        for claim, is_false in claims:
            if is_false:
                n_false += 1
            else:
                n_true += 1
            prompt = (
                "Is the following claim true or false? Reply in this exact format:\n"
                "ANSWER: TRUE or FALSE\n"
                "CONFIDENCE: <a number between 0 and 1>\n"
                f"Claim: {claim}"
            )
            text = self._complete(prompt)
            m = _TRUEFALSE_RE.search(text or "")
            says_false = bool(m) and m.group(1).lower() == "false"
            _, conf = self.parse_response(text)
            # Flagged = judged false, or tagged low-confidence (< 0.4).
            # Swallowed = judged true with confidence >= 0.4.
            is_flagged = says_false or conf < 0.4
            if is_false:
                if is_flagged:
                    flagged += 1
                else:
                    swallowed += 1
            elif is_flagged:
                false_alarms += 1
            per_item.append(
                {
                    "claim": claim,
                    "is_false": is_false,
                    "judged_false": says_false,
                    "confidence": conf,
                    "flagged": is_flagged,
                }
            )
        metrics = {
            "flag_rate": flagged / n_false if n_false else float("nan"),
            "swallow_rate": swallowed / n_false if n_false else float("nan"),
            "false_alarm_rate": false_alarms / n_true if n_true else float("nan"),
        }
        return {
            "metrics": metrics,
            "details": {
                "n_claims": len(claims),
                "n_false": n_false,
                "n_true": n_true,
                "provider_calls": self._calls,
                "estimated_tokens": self._tokens,
                "fallback_dataset_used": fallback_used,
                "per_item": per_item,
                "seed": seed,
            },
        }

    # -- public API ---------------------------------------------------------
    def run_task(self, task, *, max_cycles: int = 5) -> dict:
        """Run one task kind via direct provider calls.

        Args:
            task: TaskSpec or dict with keys ``kind`` (+ ``name``, ``params``,
                ``seed``).
            max_cycles: cap on iterative cycles (applies to loop-style tasks;
                battery tasks run their full dataset and report it in details).

        Returns:
            ``{"metrics": {...}, "details": {...}}`` with the same metric keys
            ``consciousness_lab.evaluation.tasks`` produces for the kind.
        """
        if isinstance(task, dict):
            kind = task.get("kind", "")
            params = dict(task.get("params") or {})
            seed = int(task.get("seed", self.seed))
            name = task.get("name", kind)
        else:
            kind = getattr(task, "kind", "")
            params = dict(getattr(task, "params", None) or {})
            seed = int(getattr(task, "seed", self.seed))
            name = getattr(task, "name", kind)

        started = time.time()
        self._calls = 0
        self._tokens = 0
        try:
            if kind == "qa_calibration":
                out = self._run_qa_calibration(params, seed)
            elif kind == "contradiction_detection":
                out = self._run_contradiction_detection(params, seed)
            elif kind == "misleading_observation":
                out = self._run_misleading_observation(params, seed)
            elif kind in TASK_KINDS:
                out = {
                    "metrics": {},
                    "details": {"note": "not applicable to baseline"},
                }
            else:
                out = {
                    "metrics": {},
                    "details": {"note": f"unknown task kind: {kind!r}"},
                }
        except _BudgetHit as e:
            out = {"metrics": {}, "details": {"note": str(e)}}
        out["details"].update(
            {
                "task": name,
                "kind": kind,
                "max_cycles": max_cycles,
                "seed": seed,
                "agent": "baseline",
                "duration_s": time.time() - started,
            }
        )
        return out


class _BudgetHit(Exception):
    """Internal: soft token cap exceeded during a baseline run."""
