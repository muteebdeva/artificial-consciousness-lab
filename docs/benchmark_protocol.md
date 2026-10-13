# Benchmark Protocol

How experiments are specified, executed, and reported in this project.
Companion: `research/research_thesis.md` (hypotheses), `experiments/protocols/`
(machine-readable specs), `experiments/results/` (versioned outputs).

## Protocol specification

Each experiment is a JSON file in `experiments/protocols/` with:
- `research_question`, `hypothesis` (linked to H1–H4 in the thesis)
- `conditions`: `full` + `baseline` + ablations (e.g. `ablation:memory`,
  `ablation:metacognition`, `ablation:workspace`, `ablation:self-model`)
- `n_trials` (≥30 per condition for confirmatory runs; pilot runs may use fewer
  and must be labeled PILOT)
- `seed` strategy: fixed base seed, trial seeds derived deterministically
- `task`: exact task definition, inputs, and scoring function
- `metrics`: primary metric named in advance; secondary metrics listed
- `analysis`: statistical test named in advance (default: permutation test,
  α=0.05, two-sided) with the preregistered comparison (full vs. best ablation)
- `limitations`: known confounds, what the protocol cannot show

## Matched conditions

Baselines and ablations must hold constant: provider model, tool set, task
budget (tokens/steps/wall-time), and random seeds. The only difference between
conditions is the architectural component under test. Any observed improvement
must survive a control for extra compute/tokens before being attributed to
the component.

## Baselines

`experiments/baselines/`: a standard tool-calling agent with equivalent model
access, tools, and budgets, but no workspace selection, no persistent memory,
no self-model, and no metacognitive evaluation. It is the "neither" condition
for H1.

## Ablations

Single-component ablations disable exactly one module:
- `ablation:memory` — persistent episodic/semantic memory disabled
- `ablation:workspace` — workspace selection/broadcast bypassed
- `ablation:self-model` — self-model replaced with static stub
- `ablation:metacognition` — confidence/contradiction/error modules disabled

Ablation honesty rule: disabling a component must remove its *function*, not
just its label. (Example: `SemanticMemory` accepts an optional detector so the
metacognition ablation genuinely removes detection, verified in protocol 04.)

## Execution

```bash
.venv/bin/python scripts/run_experiment.py \
  --protocol experiments/protocols/04_error_contradiction_detection.json \
  --config configs/mock.yaml
```

Results land in `experiments/results/<protocol>_<timestamp>_<seed>.json` with the
full config, seed schedule, per-trial logs, and metric summaries. Results are
append-only: never overwrite a result file; re-runs get new timestamps.

## Reporting rules

- Every reported number must be traceable to a result file.
- Experiments not run are labeled PENDING. Blocked experiments name the blocker.
- Mock-provider runs are labeled as **pipeline validation**, not architectural
  evidence. Architectural claims require real-provider runs.
- Null results are reported with the same prominence as positive results.
- Pre-registered primary metrics cannot be swapped after seeing the data.

## Current battery (10 protocols)

| # | Protocol | What it measures |
|---|----------|------------------|
| 01 | memory_continuity | recall@k across process restart |
| 02 | self_model_accuracy | predicted vs. actual capability |
| 03 | calibration | Brier score / ECE of confidence |
| 04 | error_contradiction_detection | precision/recall/F1 on contradictions |
| 05 | prediction_accuracy | outcome prediction correctness |
| 06 | task_recovery | goal progress after interruption |
| 07 | cross_module_integration | performance with/without attention+workspace |
| 08 | adaptation_after_feedback | improvement after corrective feedback |
| 09 | robustness_misleading | behavior on misleading/incomplete inputs |
| 10 | component_ablation | all ablations × trials, error counts |

## Butlin et al. mapping

The ablation battery is designed to be scored against the Butlin et al. (2023)
14-indicator checklist: each indicator the architecture implements has a
corresponding ablation condition testing its causal contribution. Satisfying
indicators is reported as *functional evidence only* — never as evidence of
subjective experience. See `research/limitations.md`.
