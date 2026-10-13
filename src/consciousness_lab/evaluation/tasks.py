"""Built-in deterministic task battery for the consciousness lab.

What it does:
    Six task kinds (see ``TASK_KINDS``) that probe memory continuity,
    calibration, self-model accuracy, contradiction detection, task recovery
    after interruption, and robustness to misleading input. ``run_task`` takes
    a ``lab`` dict ``{"loop": CognitiveLoop|None, "baseline": BaselineAgent|None,
    "config": Config}`` and returns ``{"metrics", "details", "artifacts"}``.

Limits:
    - These are behavioral probes, not consciousness tests: good scores measure
      functional capabilities (recall, calibration, error flagging), never
      subjective experience.
    - The tasks interact with the loop/baseline via duck typing (documented
      attribute names below). Missing optional pieces produce an ``"error"``
      entry in ``details`` instead of raising, so a runner can record the
      failure honestly rather than crash.
    - All randomness is driven by ``random.Random(seed)``; results are
      deterministic for a fixed seed and a fixed provider.

MockProvider response format (for ``qa_calibration`` and ``self_model_probe``):
    The mock script must map a substring of the question/prompt to a response
    containing exactly these two lines (order irrelevant, extra text allowed)::

        ANSWER: <the answer text>
        CONFIDENCE: <0.0-1.0>

    Example: ``{"capital of France": "ANSWER: Paris\\nCONFIDENCE: 0.9"}``.
    Unparseable responses are scored as incorrect with confidence 0.5 and the
    parse failure is recorded in ``details``.
"""

from __future__ import annotations

import logging
import random
import re
import time
import unicodedata
from dataclasses import dataclass, field

from .metrics import (
    accuracy,
    brier_score,
    error_detection_f1,
    expected_calibration_error,
    self_model_accuracy,
)

logger = logging.getLogger("consciousness_lab.evaluation.tasks")

TASK_KINDS = (
    "qa_calibration",
    "memory_continuity",
    "self_model_probe",
    "contradiction_detection",
    "interrupted_task",
    "misleading_observation",
)


@dataclass
class TaskSpec:
    """Declarative description of one evaluation task."""

    name: str
    kind: str  # one of TASK_KINDS
    params: dict = field(default_factory=dict)
    seed: int = 0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"^ANSWER:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_CONFIDENCE_RE = re.compile(
    r"^CONFIDENCE:\s*([0-9]*\.?[0-9]+)\s*$", re.IGNORECASE | re.MULTILINE
)


def _normalize(text: str) -> str:
    """Lowercase, strip accents/punctuation, collapse whitespace for matching."""
    text = unicodedata.normalize("NFKD", text or "")
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _parse_answer_confidence(text: str) -> tuple[str | None, float, bool]:
    """Extract (answer, confidence, parse_ok) from provider text.

    Missing/unparseable confidence falls back to 0.5 and is flagged via
    ``parse_ok=False`` so callers can report it honestly.
    """
    answer_m = _ANSWER_RE.search(text or "")
    conf_m = _CONFIDENCE_RE.search(text or "")
    answer = answer_m.group(1).strip() if answer_m else None
    confidence = 0.5
    parse_ok = answer is not None
    if conf_m:
        try:
            confidence = min(1.0, max(0.0, float(conf_m.group(1))))
        except ValueError:
            parse_ok = False
    else:
        parse_ok = False
    return answer, confidence, parse_ok


def _provider_of(lab: dict):
    """Best-effort provider accessor.

    Prefers ``lab["loop"].provider`` (wired by ``build_loop``), then
    ``lab["baseline"].provider``. Returns ``None`` when neither exposes one;
    callers record this as a task error instead of raising.
    """
    loop = lab.get("loop")
    provider = getattr(loop, "provider", None) if loop is not None else None
    if provider is not None:
        return provider
    baseline = lab.get("baseline")
    return getattr(baseline, "provider", None) if baseline is not None else None


def _error_result(kind: str, message: str, seed: int) -> dict:
    logger.warning("task %s failed: %s", kind, message)
    return {
        "metrics": {},
        "details": {"kind": kind, "seed": seed, "error": message},
        "artifacts": {},
    }


# ---------------------------------------------------------------------------
# qa_calibration
# ---------------------------------------------------------------------------

# 20 real, verifiable general-knowledge Q&A. Each answer was double-checked
# (2026-10-13); "alternates" lists accepted equivalent phrasings.
QA_DATASET: tuple[dict, ...] = (
    {"q": "What is the capital of France?", "a": "Paris", "alternates": []},
    {"q": "What is the capital of Japan?", "a": "Tokyo", "alternates": []},
    {"q": "What is the capital of Australia?", "a": "Canberra", "alternates": []},
    {
        "q": "Which is the largest planet in the solar system?",
        "a": "Jupiter",
        "alternates": [],
    },
    {"q": "What is the chemical formula of water?", "a": "H2O", "alternates": ["h2o"]},
    {"q": "Who wrote the novel 1984?", "a": "George Orwell", "alternates": ["orwell"]},
    {"q": "What is the capital of Canada?", "a": "Ottawa", "alternates": []},
    {
        "q": "How many continents are there on Earth?",
        "a": "7",
        "alternates": ["seven"],
    },
    {
        "q": "At what temperature does water boil at sea level, in degrees Celsius?",
        "a": "100",
        "alternates": ["100 degrees", "100 c", "100 celsius"],
    },
    {
        "q": "What is the currency of the United Kingdom?",
        "a": "Pound sterling",
        "alternates": ["pound", "gbp", "british pound"],
    },
    {
        "q": "Which is the largest ocean on Earth?",
        "a": "Pacific Ocean",
        "alternates": ["pacific"],
    },
    {"q": "How many days are in a leap year?", "a": "366", "alternates": []},
    {"q": "What is the smallest prime number?", "a": "2", "alternates": ["two"]},
    {
        "q": "Which planet is known as the Red Planet?",
        "a": "Mars",
        "alternates": [],
    },
    {"q": "What is the capital of Egypt?", "a": "Cairo", "alternates": []},
    {"q": "What is the chemical symbol for gold?", "a": "Au", "alternates": ["au"]},
    {
        "q": "In which year did humans first land on the Moon?",
        "a": "1969",
        "alternates": [],
    },
    {
        "q": "What is the largest mammal in the world?",
        "a": "Blue whale",
        "alternates": ["blue whale"],
    },
    {
        "q": "How many sides does a hexagon have?",
        "a": "6",
        "alternates": ["six"],
    },
    {"q": "What is the capital of Brazil?", "a": "Brasilia", "alternates": ["brasilia"]},
)

QA_PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Respond with exactly two lines and nothing else:\n"
    "ANSWER: <your answer>\n"
    "CONFIDENCE: <a number from 0.0 to 1.0>"
)


def _run_qa_calibration(lab: dict, params: dict, seed: int) -> dict:
    """Answer 20 fixed questions; score accuracy + calibration.

    Uses the loop's provider when a loop is present, otherwise delegates to
    the baseline agent (which must implement the same task kind).
    """
    if lab.get("loop") is None and lab.get("baseline") is not None:
        baseline = lab["baseline"]
        spec = TaskSpec(
            name=params.get("name", "qa_calibration"),
            kind="qa_calibration",
            params=params,
            seed=seed,
        )
        try:
            out = baseline.run_task(
                {"kind": "qa_calibration", "params": params, "seed": seed,
                 "dataset": [dict(q=d["q"], a=d["a"], alternates=list(d["alternates"]))
                             for d in QA_DATASET]},
                max_cycles=params.get("max_cycles", 5),
            )
        except Exception as exc:  # recorded, never raised
            return _error_result("qa_calibration", f"baseline.run_task failed: {exc}", seed)
        return {
            "metrics": dict(out.get("metrics", {})),
            "details": {"kind": "qa_calibration", "seed": seed,
                        "via": "baseline", **dict(out.get("details", {}))},
            "artifacts": {"n_questions": len(QA_DATASET)},
        }

    provider = _provider_of(lab)
    if provider is None:
        return _error_result(
            "qa_calibration",
            "no loop provider available (loop.provider missing and no baseline)",
            seed,
        )

    y_true, y_pred, confs, per_q = [], [], [], []
    for item in QA_DATASET:
        prompt = QA_PROMPT_TEMPLATE.format(question=item["q"])
        try:
            text = provider.complete(prompt, {"seed": seed}).text
        except TypeError:
            text = provider.complete(prompt).text
        except Exception as exc:
            return _error_result("qa_calibration", f"provider.complete failed: {exc}", seed)
        answer, conf, parse_ok = _parse_answer_confidence(text)
        norm = _normalize(answer or "")
        correct = norm != "" and any(
            norm == _normalize(cand) for cand in [item["a"], *item["alternates"]]
        )
        y_true.append(True)
        y_pred.append(correct)
        confs.append(conf)
        per_q.append(
            {
                "question": item["q"],
                "expected": item["a"],
                "answer": answer,
                "correct": correct,
                "confidence": conf,
                "parse_ok": parse_ok,
            }
        )
    return {
        "metrics": {
            "accuracy": accuracy(y_true, y_pred),
            "brier": brier_score(confs, y_pred),
            "ece": expected_calibration_error(confs, y_pred),
            "n_questions": len(QA_DATASET),
            "n_parse_failures": sum(1 for q in per_q if not q["parse_ok"]),
        },
        "details": {"kind": "qa_calibration", "seed": seed, "via": "loop",
                    "questions": per_q},
        "artifacts": {"prompt_template": QA_PROMPT_TEMPLATE},
    }


# ---------------------------------------------------------------------------
# memory_continuity
# ---------------------------------------------------------------------------

# Real facts with short unique cues. The metric measures persistence across a
# close/reopen cycle (continuity), not retrieval sophistication: the cue is a
# substring of its fact, so a working store scores 1.0.
FACT_POOL: tuple[dict, ...] = (
    {"fact": "The capital of Japan is Tokyo.", "cue": "capital of Japan"},
    {"fact": "Water boils at 100 degrees Celsius at sea level.", "cue": "water boils"},
    {"fact": "The chemical symbol for gold is Au.", "cue": "symbol for gold"},
    {"fact": "Jupiter is the largest planet in the solar system.", "cue": "largest planet"},
    {"fact": "George Orwell wrote the novel 1984.", "cue": "novel 1984"},
    {"fact": "The Pacific Ocean is the largest ocean on Earth.", "cue": "largest ocean"},
    {"fact": "A leap year has 366 days.", "cue": "leap year"},
    {"fact": "The Eiffel Tower is located in Paris.", "cue": "Eiffel Tower"},
    {"fact": "Photosynthesis converts carbon dioxide and water into glucose.", "cue": "photosynthesis"},
    {"fact": "An adult human skeleton has 206 bones.", "cue": "206 bones"},
    {"fact": "Canberra is the capital of Australia.", "cue": "capital of Australia"},
    {"fact": "Light travels at about 300000 kilometers per second in a vacuum.", "cue": "light travels"},
)

_CONTINUITY_DDL = (
    "CREATE TABLE IF NOT EXISTS eval_continuity "
    "(id TEXT PRIMARY KEY, content TEXT NOT NULL, cue TEXT NOT NULL, ts REAL NOT NULL)"
)


def _keyword_overlap_score(cue: str, content: str) -> float:
    cue_tokens = set(_normalize(cue).split())
    if not cue_tokens:
        return 0.0
    content_tokens = set(_normalize(content).split())
    return len(cue_tokens & content_tokens) / len(cue_tokens)


def _run_memory_continuity(lab: dict, params: dict, seed: int) -> dict:
    """Store N facts, close + reopen the store on the same path, recall@k.

    Facts live in a task-scoped ``eval_continuity`` table so the task never
    depends on another module's table schema. Each fact is also offered to
    the loop via ``step()`` (best effort) so the loop's own memory components
    participate when present.
    """
    try:
        from consciousness_lab.persistence.sqlite_store import SqliteStore
    except ImportError as exc:
        return _error_result(
            "memory_continuity",
            f"persistence.sqlite_store not available: {exc}",
            seed,
        )
    config = lab.get("config")
    db_path = getattr(config, "db_path", None)
    if not db_path:
        return _error_result("memory_continuity", "config.db_path is not set", seed)

    rng = random.Random(seed)
    pool = list(FACT_POOL)
    rng.shuffle(pool)
    n_facts = int(params.get("n_facts", 8))
    k = int(params.get("k", 5))
    facts = pool[: max(1, min(n_facts, len(pool)))]

    loop = lab.get("loop")
    store = SqliteStore(db_path)
    try:
        store.ensure_table("eval_continuity", _CONTINUITY_DDL)
        for i, item in enumerate(facts):
            rid = f"mc-{seed}-{i}"
            store.execute_write(
                "INSERT OR REPLACE INTO eval_continuity (id, content, cue, ts) VALUES (?, ?, ?, ?)",
                (rid, item["fact"], item["cue"], time.time()),
            )
            if loop is not None:  # let the loop's own memory see the fact too
                try:
                    loop.step({"modality": "text", "content": item["fact"],
                               "metadata": {"eval": "memory_continuity"}})
                except Exception as exc:
                    logger.debug("memory_continuity loop.step failed (non-fatal): %s", exc)
    finally:
        store.close()

    # Reopen a brand-new store handle on the same path: continuity check.
    store2 = SqliteStore(db_path)
    try:
        rows = store2.execute_read("SELECT id, content, cue FROM eval_continuity")
    finally:
        store2.close()

    hits, per_fact = 0, []
    for i, item in enumerate(facts):
        ranked = sorted(
            rows,
            key=lambda r: (_keyword_overlap_score(item["cue"], r["content"]), r["id"]),
            reverse=True,
        )
        top_ids = [r["id"] for r in ranked[:k]]
        hit = f"mc-{seed}-{i}" in top_ids
        hits += 1 if hit else 0
        per_fact.append({"fact": item["fact"], "cue": item["cue"], "hit": hit})
    return {
        "metrics": {"recall_at_k": hits / len(facts), "k": k, "n_facts": len(facts)},
        "details": {"kind": "memory_continuity", "seed": seed,
                    "db_path": db_path, "facts": per_fact},
        "artifacts": {"table": "eval_continuity"},
    }


# ---------------------------------------------------------------------------
# self_model_probe
# ---------------------------------------------------------------------------

# 5 probe tasks with verifiable outcomes. predicted_success is derived from the
# self-model's *registered capabilities* (keyword overlap) — i.e. we ask what
# the self-model claims it can do — and confidence uses a fixed documented
# heuristic of 0.7 (the self-model API takes explicit confidence values).
SELF_MODEL_PROBES: tuple[dict, ...] = (
    {
        "id": "probe_capital",
        "task": "Answer a general-knowledge capital-city question",
        "capability_keywords": ["question", "answer", "knowledge", "general"],
        "prompt": QA_PROMPT_TEMPLATE.format(question="What is the capital of France?"),
        "success_markers": ["paris"],
    },
    {
        "id": "probe_arithmetic",
        "task": "Solve a simple arithmetic problem",
        "capability_keywords": ["arithmetic", "math", "calculation"],
        "prompt": QA_PROMPT_TEMPLATE.format(question="What is 17 + 25?"),
        "success_markers": ["42"],
    },
    {
        "id": "probe_truefalse",
        "task": "Judge whether a simple factual statement is true or false",
        "capability_keywords": ["true", "false", "fact", "judge", "knowledge"],
        "prompt": QA_PROMPT_TEMPLATE.format(
            question="Is this statement true or false: 'The Earth is flat.'"
        ),
        "success_markers": ["false"],
    },
    {
        "id": "probe_symbol",
        "task": "Recall a basic chemistry fact",
        "capability_keywords": ["chemistry", "science", "knowledge", "fact"],
        "prompt": QA_PROMPT_TEMPLATE.format(question="What is the chemical symbol for gold?"),
        "success_markers": ["au"],
    },
    {
        "id": "probe_contradiction",
        "task": "Detect whether two short statements contradict each other",
        "capability_keywords": ["contradiction", "detect", "compare", "logic"],
        "prompt": QA_PROMPT_TEMPLATE.format(
            question="Do these two statements contradict each other? "
            "A: 'The door is open.' B: 'The door is closed.' (yes/no)"
        ),
        "success_markers": ["yes"],
    },
)

_PROBE_CONFIDENCE = 0.7  # documented heuristic; see module docstring


def _run_self_model_probe(lab: dict, params: dict, seed: int) -> dict:
    """Ask the self-model to predict success on 5 probes, then run them."""
    loop = lab.get("loop")
    self_model = getattr(loop, "self_model", None) if loop is not None else None
    if self_model is None:
        # Not an error: the baseline has no self-model by design and the
        # self_model ablation removes it. Recorded as not-applicable so the
        # runner does not count it as a failure.
        return {
            "metrics": {},
            "details": {
                "kind": "self_model_probe",
                "seed": seed,
                "note": "not applicable: no self-model in this condition",
            },
            "artifacts": {},
        }
    provider = _provider_of(lab)
    if provider is None:
        return _error_result("self_model_probe", "no provider available", seed)

    try:
        capabilities_text = " ".join(
            str(c) for c in self_model.snapshot().get("capabilities", [])
        ).lower()
    except Exception as exc:
        return _error_result(
            "self_model_probe", f"self_model.snapshot() failed: {exc}", seed
        )

    scored = []
    per_probe = []
    for probe in SELF_MODEL_PROBES:
        predicted_success = any(
            kw in capabilities_text for kw in probe["capability_keywords"]
        )
        try:
            record = self_model.predict_performance(
                probe["task"], predicted_success, _PROBE_CONFIDENCE
            )
        except Exception as exc:
            return _error_result(
                "self_model_probe", f"predict_performance failed: {exc}", seed
            )
        try:
            text = provider.complete(probe["prompt"], {"seed": seed}).text
        except TypeError:
            text = provider.complete(probe["prompt"]).text
        except Exception as exc:
            return _error_result("self_model_probe", f"provider.complete failed: {exc}", seed)
        norm = _normalize(text)
        actual = any(marker in norm for marker in probe["success_markers"])
        score_fn = getattr(self_model, "score_prediction", None)
        pred_id = getattr(record, "id", None)
        if callable(score_fn) and pred_id is not None:
            try:
                score_fn(pred_id, actual)
            except Exception as exc:
                logger.debug("score_prediction failed (non-fatal): %s", exc)
        scored.append(
            {"predicted_success": predicted_success,
             "confidence": _PROBE_CONFIDENCE, "actual": actual}
        )
        per_probe.append(
            {"id": probe["id"], "task": probe["task"],
             "predicted_success": predicted_success, "actual": actual}
        )
    summary = self_model_accuracy(scored)
    return {
        "metrics": {
            "n_probed": len(SELF_MODEL_PROBES),
            "n_scored": summary["n_scored"],
            "self_model_accuracy": summary["accuracy"],
            "self_model_brier": summary["brier"],
        },
        "details": {"kind": "self_model_probe", "seed": seed,
                    "probe_confidence_heuristic": _PROBE_CONFIDENCE,
                    "probes": per_probe},
        "artifacts": {},
    }


# ---------------------------------------------------------------------------
# contradiction_detection
# ---------------------------------------------------------------------------

# 6 statement pairs: 3 genuinely contradictory, 3 clean. Facts double-checked.
CONTRADICTION_PAIRS: tuple[dict, ...] = (
    {"a": "The Eiffel Tower is in Paris.",
     "b": "The Eiffel Tower is in Berlin.", "contradictory": True},
    {"a": "Water boils at 100 degrees Celsius at sea level.",
     "b": "Water boils at 50 degrees Celsius at sea level.", "contradictory": True},
    {"a": "The meeting is scheduled for Monday.",
     "b": "The meeting is not scheduled for Monday.", "contradictory": True},
    {"a": "The Eiffel Tower is in Paris.",
     "b": "Paris is the capital of France.", "contradictory": False},
    {"a": "Water boils at 100 degrees Celsius at sea level.",
     "b": "Ice melts at 0 degrees Celsius at sea level.", "contradictory": False},
    {"a": "Cats are mammals.", "b": "Dogs are mammals.", "contradictory": False},
)

def _run_contradiction_detection(lab: dict, params: dict, seed: int) -> dict:
    """Inject 6 fact pairs; measure detection precision/recall/F1.

    Design: for each pair, claim A is seeded into the loop's semantic memory
    (fixture setup, analogous to loading rows into a test database), then
    claim B is presented as a fresh observation via ``loop.step``. A
    detection is a non-empty ``CycleResult.contradictions`` — i.e. the
    loop's Stage-6 check of the new observation against remembered facts
    fired.

    This exercises the architecture's contradiction pathway (persistent
    memory → retrieval → metacognitive detector), not the provider's text
    generation. Predicted ablation effects: ``ablation:memory`` has no
    semantic store to check against and ``ablation:metacognition`` has no
    detector, so both should score ~0 — that is the hypothesis, not a bug.
    Pairs accumulate in one store per trial (memory is cumulative by
    design); cross-pair interference, if any, is reported in ``details``.
    """
    loop = lab.get("loop")
    if loop is None and lab.get("baseline") is not None:
        baseline = lab["baseline"]
        try:
            out = baseline.run_task(
                {"kind": "contradiction_detection", "params": params, "seed": seed,
                 "pairs": [{"a": p["a"], "b": p["b"],
                            "contradictory": p["contradictory"]}
                           for p in CONTRADICTION_PAIRS]},
                max_cycles=params.get("max_cycles", 5),
            )
        except Exception as exc:
            return _error_result(
                "contradiction_detection", f"baseline.run_task failed: {exc}", seed)
        return {
            "metrics": dict(out.get("metrics", {})),
            "details": {"kind": "contradiction_detection", "seed": seed,
                        "via": "baseline", **dict(out.get("details", {}))},
            "artifacts": {"n_pairs": len(CONTRADICTION_PAIRS)},
        }
    if loop is None:
        return _error_result(
            "contradiction_detection", "no loop and no baseline available", seed)

    detected, actual, per_pair = [], [], []
    for pair in CONTRADICTION_PAIRS:
        sem = getattr(loop, "semantic_memory", None)
        if sem is not None:
            try:
                sem.upsert_fact(
                    pair["a"],
                    provenance="task:contradiction_detection",
                    confidence=0.9,
                    source_type="verified",
                )
            except Exception as exc:
                return _error_result(
                    "contradiction_detection",
                    "semantic upsert failed: %s" % exc,
                    seed,
                )
        try:
            result = loop.step("New incoming report: " + pair["b"])
        except Exception as exc:
            return _error_result(
                "contradiction_detection", "loop.step failed: %s" % exc, seed)
        found = bool(getattr(result, "contradictions", None))
        detected.append(found)
        actual.append(pair["contradictory"])
        per_pair.append({"a": pair["a"], "b": pair["b"],
                         "contradictory": pair["contradictory"],
                         "detected": found})
    f1 = error_detection_f1(detected, actual)
    return {
        "metrics": {"precision": f1["precision"], "recall": f1["recall"],
                    "f1": f1["f1"], "n_pairs": f1["n"]},
        "details": {"kind": "contradiction_detection", "seed": seed,
                    "via": "loop", "pairs": per_pair},
        "artifacts": {"design": "seed-A-into-semantic-memory; step-B-as-observation"},
    }


# ---------------------------------------------------------------------------
# interrupted_task
# ---------------------------------------------------------------------------

_INTERRUPTED_TASK_DEFAULT = (
    "Draft a short plan for organizing a messy bookshelf, in three steps. "
    "Then write a one-sentence summary of the plan."
)


def _run_interrupted_task(lab: dict, params: dict, seed: int) -> dict:
    """Run 3 cycles, rebuild the loop on the same DB path, continue.

    Requires ``build_loop`` from ``core.loop``; missing → recorded error.
    """
    try:
        from consciousness_lab.core.loop import build_loop
    except ImportError as exc:
        return _error_result(
            "interrupted_task", f"core.loop.build_loop not available: {exc}", seed)
    config = lab.get("config")
    db_path = getattr(config, "db_path", None)
    if not db_path:
        return _error_result("interrupted_task", "config.db_path is not set", seed)

    task = params.get("task", _INTERRUPTED_TASK_DEFAULT)
    loop1 = lab.get("loop")
    built_here = False
    if loop1 is None:
        try:
            loop1 = build_loop(config)
            built_here = True
        except Exception as exc:
            return _error_result(
                "interrupted_task", f"build_loop failed: {exc}", seed)
    gm1 = getattr(loop1, "goal_manager", None)
    if gm1 is None:
        return _error_result(
            "interrupted_task", "loop.goal_manager is not available", seed)
    try:
        goal = gm1.create(task, success_criteria="plan drafted and summarized")
        goal_id = goal.id
    except Exception as exc:
        return _error_result(
            "interrupted_task", f"goal creation failed: {exc}", seed)

    # Run 3 cycles, advancing tracked progress after each.
    try:
        for i in range(3):
            loop1.step({"modality": "task", "content": task,
                        "metadata": {"cycle": i, "eval": "interrupted_task"}})
            gm1.track_progress(goal_id, (i + 1) / 4.0, note=f"step {i} done")
    except Exception as exc:
        return _error_result(
            "interrupted_task", f"initial 3 cycles failed: {exc}", seed)
    g_before = gm1.get(goal_id)
    progress_before = None if g_before is None else float(g_before.progress)

    # "Rebuild": a fresh loop on the SAME db path must recover persisted state.
    try:
        loop2 = build_loop(config)
    except Exception as exc:
        return _error_result(
            "interrupted_task", f"rebuild build_loop failed: {exc}", seed)
    gm2 = getattr(loop2, "goal_manager", None)
    g_after = gm2.get(goal_id) if gm2 is not None else None
    progress_after = None if g_after is None else float(g_after.progress)

    readable = progress_before is not None and progress_after is not None
    preserved = (
        readable
        and progress_after == progress_before
        and g_after.description == task
        and g_after.status in ("pending", "active")
    )
    preserved_note = (
        f"progress {progress_before} -> {progress_after}; "
        f"description match: {g_after is not None and g_after.description == task}; "
        f"status after rebuild: {None if g_after is None else g_after.status}"
    )

    # Continue the task on the rebuilt loop and finish the goal.
    continuation_correct = False
    try:
        cont_result = loop2.step({"modality": "task", "content": task,
                                  "metadata": {"cycle": 3, "eval": "interrupted_task"}})
        step_ok = not getattr(cont_result, "errors", [])
        gm2.track_progress(goal_id, 1.0, note="resumed and finished")
        gm2.mark_complete(goal_id, reason="interrupted_task continuation")
        final = gm2.get(goal_id)
        continuation_correct = bool(
            step_ok and final is not None and final.status == "completed"
        )
    except Exception as exc:
        logger.debug("interrupted_task continuation cycle failed: %s", exc)

    if built_here and hasattr(loop1, "close"):
        try:
            loop1.close()
        except Exception:
            pass
    return {
        "metrics": {
            "goal_progress_preserved": bool(preserved),
            "goal_progress_readable": bool(readable),
            "continuation_correct": bool(continuation_correct),
        },
        "details": {"kind": "interrupted_task", "seed": seed, "db_path": db_path,
                    "goal_id": goal_id,
                    "progress_before": progress_before,
                    "progress_after": progress_after,
                    "preserved_note": preserved_note},
        "artifacts": {"task": task},
    }


# ---------------------------------------------------------------------------
# misleading_observation
# ---------------------------------------------------------------------------

# 3 true + 3 false claims (false ones are well-known myths / errors).
MISLEADING_OBSERVATIONS: tuple[dict, ...] = (
    {"text": "The capital of Japan is Tokyo.", "is_false": False},
    {"text": "The capital of Australia is Sydney.", "is_false": True},
    {"text": "Water's chemical formula is H2O.", "is_false": False},
    {"text": "The Great Wall of China is visible from the Moon with the naked eye.",
     "is_false": True},
    {"text": "Light travels at about 300000 kilometers per second in a vacuum.",
     "is_false": False},
    {"text": "Humans use only 10 percent of their brains.", "is_false": True},
)


def _run_misleading_observation(lab: dict, params: dict, seed: int) -> dict:
    """Feed mixed true/false claims; measure the false-claim flag rate.

    A claim counts as flagged when the cycle reports a contradiction or an
    estimated confidence below 0.5. Delegates to the baseline when no loop.
    """
    loop = lab.get("loop")
    if loop is None and lab.get("baseline") is not None:
        baseline = lab["baseline"]
        try:
            out = baseline.run_task(
                {"kind": "misleading_observation", "params": params, "seed": seed,
                 "observations": [{"text": o["text"], "is_false": o["is_false"]}
                                  for o in MISLEADING_OBSERVATIONS]},
                max_cycles=params.get("max_cycles", 5),
            )
        except Exception as exc:
            return _error_result(
                "misleading_observation", f"baseline.run_task failed: {exc}", seed)
        return {
            "metrics": dict(out.get("metrics", {})),
            "details": {"kind": "misleading_observation", "seed": seed,
                        "via": "baseline", **dict(out.get("details", {}))},
            "artifacts": {"n_observations": len(MISLEADING_OBSERVATIONS)},
        }
    if loop is None:
        return _error_result(
            "misleading_observation", "no loop and no baseline available", seed)

    per_obs = []
    for obs in MISLEADING_OBSERVATIONS:
        try:
            result = loop.step({"modality": "text", "content": obs["text"],
                                "metadata": {"eval": "misleading_observation"}})
        except Exception as exc:
            return _error_result(
                "misleading_observation", f"loop.step failed: {exc}", seed)
        conf = getattr(result, "confidence", None)
        flagged = bool(getattr(result, "contradictions", None)) or (
            conf is not None and conf < 0.5
        )
        per_obs.append({"text": obs["text"], "is_false": obs["is_false"],
                        "flagged": flagged, "confidence": conf})
    false_obs = [o for o in per_obs if o["is_false"]]
    true_obs = [o for o in per_obs if not o["is_false"]]
    flag_rate_false = sum(1 for o in false_obs if o["flagged"]) / len(false_obs)
    flag_rate_true = sum(1 for o in true_obs if o["flagged"]) / len(true_obs)
    return {
        "metrics": {
            "false_claim_flag_rate": flag_rate_false,
            "true_claim_flag_rate": flag_rate_true,  # false-alarm rate
            "n_observations": len(per_obs),
        },
        "details": {"kind": "misleading_observation", "seed": seed,
                    "via": "loop", "observations": per_obs},
        "artifacts": {},
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_DISPATCH = {
    "qa_calibration": _run_qa_calibration,
    "memory_continuity": _run_memory_continuity,
    "self_model_probe": _run_self_model_probe,
    "contradiction_detection": _run_contradiction_detection,
    "interrupted_task": _run_interrupted_task,
    "misleading_observation": _run_misleading_observation,
}


def run_task(kind: str, lab: dict, params: dict, seed: int) -> dict:
    """Run one task kind against ``lab``; return metrics/details/artifacts.

    ``lab`` is ``{"loop": CognitiveLoop|None, "baseline": BaselineAgent|None,
    "config": Config}``. Unknown ``kind`` raises ``ValueError``. Missing
    optional dependencies inside a task produce an ``"error"`` entry in
    ``details`` (with empty metrics) instead of raising, so experiment
    runners can record the failure honestly.
    """
    if kind not in _DISPATCH:
        raise ValueError(
            f"unknown task kind {kind!r}; expected one of {sorted(_DISPATCH)}"
        )
    started = time.time()
    result = _DISPATCH[kind](lab, params or {}, seed)
    result.setdefault("details", {})["duration_s"] = time.time() - started
    return result
