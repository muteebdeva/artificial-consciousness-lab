# Research Thesis: Coupled Workspace–Metacognition in Language Agents

**Document:** research/research_thesis.md
**Date:** 2026-10-13
**Status:** v1.0 — hypothesis registered before implementation.
**Companion documents:** `literature_review.md` (background),
`hypotheses.md` (candidates considered), `limitations.md` (assessment limits).

---

## 1. The scientific question

Whether computational architectures can support properties associated with
consciousness is an open scientific question. This project does **not** attempt
to settle it, and it does **not** claim that any system it builds is conscious.
Phenomenal consciousness (subjective experience) is distinguished throughout
from **measurable cognitive capabilities** (calibration, error detection,
belief revision, self-model accuracy).

The tractable scientific question this project investigates is narrower:

> **Do the functional mechanisms that leading theories of consciousness
> identify — specifically, a competitive global workspace coupled to
> metacognitive monitoring — causally improve an AI agent's measurable
> cognitive performance, and if so, through which component?**

This question is answerable. It converts the Butlin et al. (2023) indicator
checklist from a static scorecard into a **causal research program**: build the
indicators, ablate them under matched conditions, and measure what changes.

---

## 2. Motivation

Three findings from the literature jointly motivate this thesis:

1. **The best empirical test of the leading theories produced a draw.**
   The COGITATE adversarial collaboration (Nature, 2025) preregistered
   predictions of IIT and GNWT and confirmed neither fully: no sustained
   posterior synchronization (IIT), no offset ignition and weak prefrontal
   content decoding (GNWT). The functional pieces — integration, broadcast,
   recurrence — look real; the grand theories overclaim. The productive move
   is to test the *mechanisms*, not the *metaphysics*.

2. **No published work causally tests whether a global workspace improves
   agent cognition.** LIDA implements GWT as software but has no published
   ablation against matched baselines on modern tasks. Bengio's consciousness
   prior (2017), VanRullen & Kanai (2021), and Juliani et al. (2021) argue for
   workspace-like structure in deep learning, but as architectural proposals,
   not controlled experiments. Goldstein & Kirk-Giannini (2024) argue
   conditionally that GWT-satisfying language agents might be conscious — a
   claim that is *testable in its functional half*: does the architecture
   actually do anything measurable?

3. **Metacognition in AI is measured but rarely made causally effective.**
   The LLM uncertainty literature (Kadavath et al. 2022; Kuhn et al. 2023;
   Tian et al. 2023) measures calibration and sensitivity, but confidence
   estimates seldom *drive* behavior — belief revision, information-seeking,
   or task abortion. Higher-order theories (Lau & Rosenthal 2011) and Butlin
   et al.'s HOT-3 indicator (belief revision driven by metacognition) point to
   exactly this gap: metacognition that only produces verbal confidence
   scores is decorative.

**The synthesis:** a competitive global workspace (GNWT mechanism) coupled to
a metacognitive layer whose uncertainty estimates *cause* belief revision and
evidence-seeking (higher-order mechanism), with recurrence feeding both back
into perception and memory (RPT mechanism), and an explicit attention schema
(AST mechanism) in the self-model. The original contribution is not the
combination of buzzwords — it is the **falsifiable causal test** of whether
this coupling improves measurable cognition under matched conditions.

---

## 3. The primary hypothesis (H1)

> **H1 — Coupled workspace–metacognition improves calibrated task
> performance.**
> A language-agent architecture combining
> (a) a **limited-capacity global workspace** with explicit competition among
> module outputs and broadcast of the winner to all modules, and
> (b) a **metacognitive layer** that estimates confidence over workspace
> contents and *revises beliefs and actions* on the basis of those estimates
> (including requesting additional evidence when uncertainty is high),
> will achieve **lower expected calibration error (ECE)** and **higher AUROC
> of confidence-vs-correctness** on held-out factual-QA and multi-step
> tool-use batteries than each of three matched ablations —
> workspace-only (no metacognition), metacognition-only (no
> workspace competition/broadcast), and neither (standard ReAct-style
> baseline) — with the base model, tools, prompt budget, and compute held
> constant.

### Predicted outcomes (directional, not numerical claims)
- **H1a:** Full system < workspace-only < metacognition-only ≈ baseline on
  ECE (lower is better). Rationale: broadcast integrates heterogeneous
  evidence (memory, tools, self-model) into one decision context; without it,
  metacognitive estimates are computed over fragmented state.
- **H1b:** Full system > all ablations on AUROC (confidence discriminating
  correct vs. incorrect). Rationale: workspace competition surfaces
  *disagreement* between modules, which is the informative signal for
  uncertainty.
- **H1c:** The metacognition-only ablation improves ECE modestly over
  baseline but does not improve error-detection F1 (detecting its own
  errors on held-out traces), because its confidence estimates do not feed
  back into action selection. This tests the "decorative vs. causal
  metacognition" distinction.
- **H1d:** Removing recurrence (feedforward-only variant) degrades
  multi-step task recovery after injected errors, but leaves single-step QA
  calibration unaffected — testing RPT's functional claim at the task level.

### Failure conditions (what falsifies H1)
- **Falsified** if, with ≥30 trials per condition and matched budgets, the
  full system shows **no statistically significant improvement** (permutation
  test, α = 0.05) over the best-performing ablation on the primary metrics
  (ECE, AUROC).
- **Weakened** if the full system beats baseline but no single ablation
  differs from full — i.e., the gain is not attributable to the hypothesized
  components (suggesting a confound such as total compute or prompt content).
- **Refined, not falsified,** if ablations show a graded pattern different
  from H1a–H1d (e.g., workspace-only matches full): the thesis then updates
  to credit the responsible component.

### Alternative explanations (controlled by design)
| Alternative | Control |
|---|---|
| The gain is just more compute/tokens | Fixed per-trial token and tool-call budgets across conditions |
| The gain is prompt engineering (metacognitive prompts) | Baseline receives equivalent prompt length and reflection prompts without the architectural loop |
| The gain is base-model calibration, not architecture | Same model and provider in all conditions; provider-variation study is separate |
| Dataset leakage / benchmark gaming | Held-out task sets generated after architecture freeze; task batteries documented in `experiments/protocols/` |
| Researcher degrees of freedom | Preregistered metrics, seeds, and trial counts per experiment; results versioned in `experiments/results/` |

---

## 4. Secondary hypotheses

### H2 — Self-model accuracy predicts calibration quality
> Conditioning uncertainty estimates on a **grounded self-model** (actual
> capabilities, limitations, resources, recent outcomes — updated from tool
> results, not self-report) reduces ECE relative to conditioning on an
> ungrounded or absent self-model.
> **Falsification:** no significant ECE difference between grounded-self-model
> and no-self-model conditions; or self-model accuracy score (measured
> against ground-truth capability inventory) shows zero correlation with ECE
> across runs.

### H3 — Workspace competition improves error detection under misleading input
> On trials with deliberately misleading or incomplete observations, the
> workspace's explicit competition (losing hypotheses retained with
> provenance) yields higher contradiction-detection F1 than a baseline that
> commits to a single interpretation.
> **Falsification:** no F1 difference on the misleading-input subset.

### H4 — Episodic outcome memory improves prediction of own performance
> The system's predictions of its own task success (recorded before
> execution) are better calibrated when informed by episodic outcome memory
> (past predictions vs. outcomes) than without.
> **Falsification:** no improvement in prediction–outcome calibration
> (Brier score) with outcome memory enabled.

All secondary hypotheses inherit the same matched-condition, preregistered
discipline as H1.

---

## 5. Evaluation criteria

**Primary metrics (per experiment, preregistered):**
- **ECE** (expected calibration error) with 10 equal-width bins; reported
  with bootstrap 95% CIs.
- **AUROC** of confidence scores discriminating correct vs. incorrect
  responses.
- **Error-detection F1** on held-out execution traces with labeled errors.
- **Brier score** for self-performance predictions (H4).

**Design requirements:**
- ≥30 trials per condition per experiment (power for medium effects at
  α = 0.05; exact power analysis recorded per protocol).
- Matched conditions: identical base model, tools, token/tool budgets,
  task sets, seeds.
- Deterministic mock provider for unit tests; real providers for experiments
  with provider identity and version logged.
- **Never fabricated:** any experiment not run is labeled PENDING; any
  blocked experiment documents the blocker and what is needed.

**Interpretation discipline:**
- A positive result supports a claim about **functional cognitive
  capabilities only**: calibrated uncertainty, error detection, belief
  revision.
- A positive result is **not evidence of phenomenal consciousness** and will
  never be presented as such. See `limitations.md` for the full false-positive
  analysis.

---

## 6. Comparison with existing work (why this is not buzzword assembly)

- **vs. LIDA:** LIDA implements GWT faithfully but (i) predates LLMs,
  (ii) has no published matched ablation battery, (iii) is unmaintained
  Java. We implement the same functional roles (competition, broadcast,
  learning-from-broadcast) in an LLM-native, Python, provider-abstracted
  system *with the ablation battery LIDA never had*.
- **vs. CoALA:** CoALA is a descriptive framework, not an executable system
  with experiments. We adopt its memory/action/decision vocabulary and build
  the executable instance plus the evaluation harness it does not provide.
- **vs. ReAct / Reflexion:** these demonstrate reasoning traces and verbal
  self-reflection; they do not measure calibration, sensitivity, and
  efficiency separately, and their "self-correction" is not ablated against
  architectural alternatives.
- **vs. Butlin et al. (2023):** they provide the indicator checklist and the
  assessment methodology; we provide a system designed to be *scored and
  ablated* on it. Their framework is the yardstick; our contribution is the
  measured artifact.
- **vs. Bengio's consciousness prior:** a learning bias proposal without an
  agent-level implementation or evaluation. Ours is an agent-level
  implementation with evaluation.

The claim to originality is therefore narrow and checkable: **the first
open, reproducible ablation battery causally testing whether
workspace–metacognition coupling improves calibrated agent cognition, with
results reported win or lose.**

---

## 7. Scope boundaries (what this thesis does NOT claim)

1. No claim about phenomenal consciousness, subjective experience, or
   sentience — in either direction. The project is agnostic by design.
2. No claim that the architecture is necessary or sufficient for
   consciousness under any theory.
3. No claim that beating a baseline on calibration constitutes progress
   toward machine consciousness — only progress toward better-calibrated
   agents, which is independently valuable (safer deployment, better
   human–AI teaming).
4. IIT is out of architectural scope (uncomputable Φ; excluded by Butlin et
   al.'s functionalist framing), retained only as conceptual background.
5. Full variational active inference is out of scope for the initial build;
   precision-weighting is implemented in standard statistical form.

---

## 8. Milestones tied to this thesis

- **M1:** Core architecture runs end-to-end on the deterministic mock
  provider; all unit tests pass.
- **M2:** Experiment runner executes H1 battery (full + 3 ablations +
  baseline) on mock provider with ≥30 trials/condition; results versioned.
- **M3:** H1 battery re-run on at least one real provider (Ollama local or
  API); provider identity logged; results compared with mock.
- **M4:** H2–H4 batteries executed; self-model accuracy measured against
  ground truth.
- **M5:** Butlin-indicator self-assessment published with per-indicator
  evidence or explicit absence; limitations and blockers disclosed.

Each milestone reports **actual results**, including null results. A null
result on H1 is a successful scientific outcome: it would constitute
evidence that workspace–metacognition coupling, as implemented, does not
causally improve calibrated cognition — a genuine, publishable constraint
on the design space.
