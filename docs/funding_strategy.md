# Funding Strategy

## The specific scientific problem

Leading theories of consciousness (GNWT, IIT, higher-order theories) make
architectural claims about cognition, but no published work causally tests
whether the mechanisms they identify — competitive global workspace,
metacognitive monitoring that drives belief revision — actually improve an AI
agent's measurable cognitive performance under matched conditions. The
COGITATE adversarial collaboration (Nature, 2025) tested the theories'
neuroscientific predictions and confirmed neither fully, leaving
mechanism-level testing as the productive path forward.

## Why existing approaches are insufficient

- **LIDA** implements GWT as software but has no published ablations against
  matched baselines on modern tasks.
- **Bengio's consciousness prior / VanRullen & Kanai / Juliani et al.** are
  architectural proposals, not controlled experiments.
- **LLM uncertainty literature** (Kadavath 2022; Kuhn 2023; Tian 2023) measures
  calibration but rarely makes confidence estimates *causally effective*.
- **Butlin et al. (2023)** provide an indicator checklist, but as a static
  scorecard — no causal program attaches.

## Proposed technical contribution

This platform converts the indicator checklist into a causal research program:
build the indicators as modules, ablate each under matched conditions, and
measure what changes. The registered hypothesis (H1) is falsifiable by design
(permutation test, α=0.05, ≥30 trials/condition).

## What evidence would support or weaken the thesis

- **Support:** full architecture beats the best single-component ablation on
  calibration (ECE, AUROC) with compute controlled.
- **Weaken:** no significant gain; gains fully explained by extra tokens/compute;
  or ablations show the workspace adds nothing once metacognition is present
  (which would itself be a publishable negative result).

## Minimum viable research milestones

1. ✅ v0.1 engine + 10-protocol battery + mock-validated pipeline (this repo)
2. Real-provider runs (Ollama local, then Gemini API): first genuine
   architectural signal on protocols 03/04/10
3. Factorial ablation design (interaction effects, not just single ablations)
4. Pre-registered confirmatory run (n≥30/condition) + write-up

## Compute, staffing, infrastructure

- **Compute:** v0.1–v0.2 run on CPU-only hardware; LLM inference via local
  Ollama or API credits. No GPU training planned (no fine-tuning until
  experiments justify it).
- **Staffing:** 1 research engineer (current); a second person for independent
  replication would materially strengthen credibility.
- **Infrastructure:** commodity — SQLite, stdlib Python, versioned JSON results.

## Low-cost initial path

The entire v0.1 milestone cost ₹0: no paid APIs, no GPUs, no services.
v0.2 needs only API credits or a local Ollama host. This is deliberate —
the science must not depend on funding to begin.

## Potential academic collaborations, grants, fellowships

*Directions to explore, not commitments:* cognitive-science and AI-safety labs
working on machine consciousness indicators; adversarial-collaboration style
replication (following the COGITATE model); open-science grants supporting
reproducible AI research infrastructure. No partnerships, endorsements, or
funding commitments exist at this time, and none are claimed.

## How a product might emerge (only if results justify it)

If the architecture reliably improves calibration and error detection in
agents, the metacognitive layer could become a reliability product for
high-stakes agent deployments (calibrated abstention, contradiction flagging,
audit trails). This is a *conditional* possibility, not a plan — the research
comes first, and any product framing must not distort the science.

## What this document does not claim

No funding commitments, partnerships, institutional endorsements, patent
protection, or market validation exist. A promising proposal is distinguished
from demonstrated performance throughout.
