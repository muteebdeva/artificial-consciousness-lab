# Candidate Hypotheses: Considered and Decided

**Document:** research/hypotheses.md
**Date:** 2026-10-13
**Status:** v1.0 — records the selection process for the initial thesis.
**See also:** `research_thesis.md` (the chosen hypothesis, H1),
`literature_review.md` (background).

> **Purpose.** To show the work: which hypotheses were considered, why most
> were rejected, and why H1 was selected. A hypothesis that cannot state its
> own falsification conditions was rejected outright.

---

## H1 — Coupled workspace–metacognition improves calibrated task performance ✅ SELECTED

**Statement:** A language-agent architecture combining a competitive
limited-capacity global workspace with a metacognitive layer whose
uncertainty estimates drive belief revision and evidence-seeking will achieve
lower ECE and higher confidence-vs-correctness AUROC than matched ablations
(workspace-only, metacognition-only, neither) on held-out QA and multi-step
task batteries.

**Why selected:**
- **Narrow and falsifiable.** Specifies the intervention (two components),
  the comparison (three matched ablations), the metrics (ECE, AUROC), the
  statistics (permutation test, α = 0.05, ≥30 trials/condition), and the exact
  failure condition (no significant improvement over the best ablation).
- **Fills a documented gap.** No published work ablates workspace integration
  and metacognitive evaluation in LLM agents under matched conditions
  (literature_review.md §10). LIDA implements GWT without the ablation
  battery; CoALA describes without implementing; ReAct/Reflexion implement
  without measuring.
- **Theoretically grounded without overclaiming.** The components map
  directly onto GNWT (competition/broadcast), higher-order theories
  (metacognition driving belief revision — Butlin et al.'s HOT-3), RPT
  (recurrence), and AST (attention schema in the self-model), but the
  hypothesis tests only their *functional* contribution to calibration.
- **Valuable even if falsified.** A null result constrains the design space
  and is independently publishable; better-calibrated agents are valuable
  regardless of consciousness debates.
- **Feasible with available resources.** Runs on CPU, SQLite, and local or
  free-tier providers; the deterministic mock provider enables the full
  battery without paid APIs.

**Risks acknowledged:** the effect may be small; prompt-content confounds
must be controlled by matched prompt budgets; the result is
implementation-specific and should be reported as such.

---

## H2 — Self-model accuracy predicts calibration quality ✅ SELECTED (secondary)

**Statement:** Conditioning uncertainty estimates on a grounded self-model
(actual capabilities, limitations, resources, outcomes) reduces ECE vs. an
ungrounded/absent self-model; self-model accuracy (vs. ground-truth
capability inventory) correlates with ECE across runs.

**Why selected:** directly implementable, measurable against ground truth,
and addresses a real gap — "self-model" in agent literature is usually a
prompted persona, never scored. Falsifiable via the no-difference condition.

**Why secondary, not primary:** it is a sub-mechanism of H1's metacognitive
layer; testing it first would leave the headline architectural question
unanswered.

---

## H3 — Workspace competition improves contradiction detection ✅ SELECTED (secondary)

**Statement:** On deliberately misleading/incomplete observations, explicit
workspace competition (losing hypotheses retained with provenance) yields
higher contradiction-detection F1 than single-interpretation baselines.

**Why selected:** tests a distinctive prediction of the workspace design
(competition surfaces disagreement) that a plain ReAct baseline cannot match
by construction. Falsifiable on the misleading-input subset.

---

## H4 — Episodic outcome memory improves self-performance prediction ✅ SELECTED (secondary)

**Statement:** Predictions of own task success are better calibrated (Brier
score) when informed by episodic outcome memory (past predictions vs.
outcomes).

**Why selected:** minimal, clean test of the outcome-memory component;
directly measures the "learning from experience" claim without invoking
weight updates. Falsifiable.

---

## Rejected candidates

### R1 — "The system will satisfy N of Butlin et al.'s 14 indicators"
**Rejected as a hypothesis** (kept as an assessment exercise instead).
Satisfying indicators is a *design checklist*, not a falsifiable prediction
about the world — and Butlin et al. explicitly warn that satisfying
indicators would not establish consciousness. It belongs in evaluation
(`experiments/`), not in the thesis. Scoring the system on the checklist is
still planned, but as descriptive assessment, not hypothesis testing.

### R2 — "Recurrent processing is necessary for the system's performance"
**Rejected as stated; partially absorbed into H1d.** "Necessary" is too
strong — a feedforward system might match performance on many tasks. The
testable version (recurrence matters for multi-step error recovery, not for
single-step QA) is retained as prediction H1d. Claiming necessity would
require exhaustive task coverage we cannot provide.

### R3 — "Precision-weighted prediction errors improve attention allocation"
**Rejected for the initial build; deferred.** A genuine hypothesis from the
predictive-processing literature, but implementing full precision-weighting
requires a probabilistic generative model of the task domain that we do not
yet have. The project implements the *conceptual* version (reliability-gated
updates) and defers the formal version to the roadmap. Testing a weak
implementation would produce an uninformative null.

### R4 — "The system exhibits integrated information (Φ) above baseline"
**Rejected.** Φ is uncomputable at relevant scale (IIT 3.0 acknowledges the
~dozen-element limit); any reported number would be a toy computation with no
evidentiary value. Butlin et al. exclude IIT from their functionalist
framework. Retained only as conceptual background.

### R5 — "Metacognitive efficiency (meta-d′/d′) will exceed 1.0"
**Rejected as stated; absorbed into measurement practice.** Meta-d′/d′ >
1.0 would mean metacognition adds information beyond first-order
performance — an interesting finding if observed, but there is no theoretical
basis to *predict* it for this architecture, and the human literature shows
efficiency is typically ≤ 1.0. The project will *measure* meta-d′/d′ as a
descriptive statistic without predicting its value.

### R6 — "The architecture will outperform commercial agent frameworks on general benchmarks"
**Rejected.** Benchmark-maxing is not the research question; commercial
frameworks differ in model, tools, and tuning, making matched conditions
impossible; and winning a benchmark would say nothing about the thesis.
The project's baselines are *internal matched ablations*, not leaderboard
competitors.

### R7 — "Workspace broadcast enables emergent communication between modules not explicitly programmed"
**Rejected as too vague to falsify.** "Emergent communication" has no agreed
operationalization here; any information flow could be redescribed as
emergent. If a precise version becomes formulable (e.g., modules solving a
novel coordination task with no shared protocol), it can be promoted to a
hypothesis in a later phase.

### R8 — "The system will develop a stable, human-like sense of self"
**Rejected.** Not measurable against ground truth, invites anthropomorphism
(Shanahan 2024), and edges toward the consciousness claims the project
explicitly refuses. The self-model is defined strictly as a structured,
grounded representation of system state — nothing more.

---

## Selection summary

| ID | Hypothesis | Status | Falsifiability |
|---|---|---|---|
| H1 | Workspace–metacognition coupling improves calibration | **Primary** | No significant gain over best ablation → falsified |
| H2 | Grounded self-model improves calibration | Secondary | No ECE difference / zero correlation → falsified |
| H3 | Workspace competition improves contradiction detection | Secondary | No F1 difference on misleading inputs → falsified |
| H4 | Outcome memory improves self-prediction | Secondary | No Brier-score improvement → falsified |
| R1 | Satisfy N indicators | Rejected (→ assessment) | Not a prediction |
| R2 | Recurrence is necessary | Rejected (→ H1d) | Too strong |
| R3 | Precision weighting | Deferred | Needs generative model |
| R4 | Φ above baseline | Rejected | Uncomputable |
| R5 | meta-d′/d′ > 1.0 | Rejected (→ descriptive) | No basis to predict |
| R6 | Beat commercial frameworks | Rejected | Wrong question |
| R7 | Emergent inter-module communication | Rejected | Unfalsifiable as stated |
| R8 | Human-like sense of self | Rejected | Unmeasurable; off-mission |

**Decision rule applied throughout:** a candidate became a hypothesis only
if it named an intervention, a comparison, a metric, and a failure
condition. Everything else was reclassified as assessment, measurement
practice, deferred work, or rejected.
