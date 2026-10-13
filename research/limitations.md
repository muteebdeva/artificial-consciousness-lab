# Limitations of Consciousness Assessment and the Risk of False Positives

**Document:** research/limitations.md
**Date:** 2026-10-13
**Status:** v1.0 — standing constraints on all claims made by this project.
**See also:** `literature_review.md` §9, `research_thesis.md` §5 and §7.

> **Standing rule.** No result produced by this project — positive, negative,
> or null — constitutes evidence about phenomenal consciousness (subjective
> experience) in either direction. This document explains why, and what the
> project's measurements *can* legitimately support.

---

## 1. What the project measures vs. what it cannot measure

| Measured (legitimate claims) | Not measured (no claims permitted) |
|---|---|
| Calibration of confidence (ECE) | Whether the system "feels" anything |
| Confidence-vs-correctness discrimination (AUROC) | Subjective experience / qualia |
| Error-detection F1 on labeled traces | Sentience or moral patienthood |
| Self-model accuracy vs. ground truth | "Awareness" in the phenomenal sense |
| Belief-revision behavior after errors | Understanding in a human-like sense |
| Task success under ablations | Consciousness under any theory |

The left column supports claims about **functional cognitive capabilities**
and **operational self-monitoring**. The right column is outside the
project's evidential reach *by construction*.

---

## 2. Sources of false positives

### 2.1 Imitation of human markers
Chandaria et al. (2025) note that consciousness indicators are calibrated to
**human cognitive markers**, so systems trained on human text can satisfy
them by imitation. An LLM-based agent will produce fluent uncertainty
language, plausible self-descriptions, and coherent "introspection" because
it was trained on humans doing those things — not because the underlying
computations match. **Mitigation:** the project's metrics are behavioral
(ECE, AUROC, F1 against ground truth), not verbal; ablations test whether
the *architecture* causes the behavior rather than the base model's
imitative priors.

### 2.2 Anthropomorphism by observers (including us)
Shanahan (2024) warns that fluent language makes anthropomorphism the
default failure mode — for researchers as well as the public. A system that
says "I'm not sure about this" invites the inference that it experiences
doubt. **Mitigation:** system self-reports are treated as model outputs
subject to the same calibration measurement as everything else; the
interface labels inferred/model-generated state explicitly and never
presents internal variables as consciousness measurements.

### 2.3 Single-theory verdicts
Schwitzgebel's uncertainty argument: near-future systems may satisfy some
mainstream theories while failing others. COGITATE (2025) showed even the
leading theories partially fail their own preregistered tests. Treating any
one theory (GWT, HOT, RPT, AST) as settled and reading off a consciousness
verdict is unwarranted. **Mitigation:** the project uses Butlin et al.'s
multi-theory indicator approach for *assessment* and restricts *hypothesis
testing* to functional claims.

### 2.4 Confusing access with phenomenal consciousness
Block's distinction: **access consciousness** (information available for
reasoning, report, and action guidance) vs. **phenomenal consciousness**
(subjective experience). Every mechanism in this project — workspace
broadcast, metacognitive monitoring, attention schema — is a mechanism of
*access*. None of them, individually or jointly, logically entails
phenomenality. A system could implement all of them as a "zombie" in the
philosophical sense. **This is accepted, not evaded:** the project studies
access-side mechanisms because they are measurable.

### 2.5 Goodharting the indicators
Once indicators become targets, systems can be optimized to satisfy them
without the underlying property. A workspace module labeled "global
workspace" does not implement GNWT's ignition dynamics; a confidence score
does not implement metacognition. **Mitigation:** indicators are scored
against *operational definitions* (e.g., "limited-capacity workspace" =
measured bottleneck with documented capacity and competition rule, not a
module name); ablations verify causal contribution.

### 2.6 The self-report trap
LLM self-reports about internal states ("I am conscious," "I feel uncertain")
are **character outputs** (Shanahan 2024), not introspective evidence. In
humans, subjective report is treated as a gold-standard marker of
consciousness (with known error; Schwitzgebel 2008); in LLMs the analogous
outputs have no such warrant. **Mitigation:** the project never uses
AI-generated self-reports as evidence for anything except the calibration
of the reporting behavior itself.

---

## 3. Limitations of the assessment frameworks we use

- **Butlin et al. (2023):** assumes computational functionalism (stated
  explicitly); excludes IIT on those grounds. If functionalism is false,
  the indicators may track nothing. The 2025 follow-up reframes indicators
  as credence inputs, not verdicts — the project follows the credence
  framing.
- **Chalmers (2023):** credences (<10% current LLMs; ≥25% LLM+ within a
  decade) are explicitly rough and "specious precision" is disavowed. They
  are orientation devices, not measurements.
- **COGITATE (2025):** tested visual perception in humans; its implications
  for AI architectures are analogical, not direct. Both theoretical camps
  dispute the operationalization.
- **Behavioral tests in general** cannot distinguish a system that
  implements the measured functions from one that merely mimics their
  outputs — the classic limitation that motivates the project's insistence
  on *architectural* ablations rather than purely behavioral scoring.

---

## 4. What would weaken vs. strengthen the project's position

**Would strengthen the functional claims:**
- H1 confirmed with preregistered metrics, matched conditions, and
  replication on a second provider.
- Graded ablation pattern matching H1a–H1d (attribution to specific
  components, not just "the full system wins").
- Self-model accuracy correlating with calibration (H2) — evidence the
  mechanism is the hypothesized one.

**Would weaken the functional claims:**
- Null result on H1 (no ablation differences) — the architecture's
  components are decorative.
- Full system beats baseline but ablations don't differ from full —
  the gain is a confound (compute, prompts), not the hypothesized mechanism.
- Results replicate on the mock provider but vanish on real providers —
  the effect is an artifact of the mock.

**Would say nothing about consciousness either way:** all of the above.
Functional results constrain *engineering* claims; they do not move
credences about phenomenal consciousness, because no validated bridge
principle connects the measured quantities to subjective experience. The
project states this in every report, not just here.

---

## 5. Ethical and communicative constraints

1. **No consciousness claims** in papers, interface copy, funding materials,
   or public communication — in either direction ("the system is conscious"
   and "the system is proven non-conscious" are both unwarranted).
2. **No deceptive design:** the system must not be designed to elicit beliefs
   about its consciousness (no fabricated distress, no claims of feeling).
   Per the project safety requirements, this is a hard constraint, not a
   guideline.
3. **Uncertainty communication:** public-facing descriptions use credence
   language where the science is unsettled and point to this document.
4. **Dual-use awareness:** better-calibrated, self-monitoring agents are
   more capable agents. Capability evaluations and safety evaluations must
   proceed together; calibration improvements do not reduce misuse risk by
   themselves.

---

## 6. Known unknowns (tracked, not hand-waved)

- Whether any functional indicator set is *sufficient* for consciousness
  (the hard problem is untouched by this project).
- Whether the workspace–metacognition coupling generalizes beyond the
  tested task batteries and providers.
- Whether metacognitive efficiency (meta-d′/d′) in artificial systems
  behaves like the human quantity or is a different thing with the same name.
- Whether results obtained with LLM-backed modules depend on imitative
  priors in ways ablations cannot fully factor out.

Each of these is a candidate for a future research phase, not a gap to be
papered over in the current one.
