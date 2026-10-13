# artificial-consciousness-lab

A research-grade platform for investigating whether computational
architectures can support properties associated with consciousness —
integrated cognition, persistent self-modeling, metacognition, attention,
memory, and autonomous cognitive processes.

**Scientific stance (read first):** this platform does *not* assume
consciousness has been achieved. Subjective experience is treated as an open
scientific question. We distinguish measurable functional capabilities
(memory, planning, calibration, error detection) from claims about
phenomenal consciousness, and we never treat fluent self-reports
("I am conscious") as evidence. See `docs/architecture.md` and
`research/` for the full framing (literature review, registered thesis H1–H4, limitations).

## Working hypothesis (v0.1)

> A cognitive architecture combining (a) a limited-capacity global
> workspace with competitive attention-based selection and broadcast,
> (b) persistent episodic + semantic memory with provenance and
> confidence, and (c) metacognitive confidence calibration and
> contradiction detection, will measurably outperform a matched baseline
> (same model, tools, task budget; no workspace/memory/metacognition) on
> memory continuity, self-model accuracy, confidence calibration, and
> error/contradiction detection. Ablating any single component degrades
> the capability it supports.

## Repository layout

```
src/consciousness_lab/   # research engine (stdlib only)
  core/        config, logging, errors, CognitiveLoop (15-stage cycle)
  models/      provider abstraction: mock | ollama | openai_compat | gemini
  persistence/ SQLite store (WAL, parameterized, versioned schema)
  workspace/   capacity-limited global workspace, competitive selection
  memory/      episodic / semantic / working / outcome (+calibration)
  self_model/  capability/limitation model grounded in observed state
  metacognition/ confidence, contradiction, error detection, belief revision
  perception/  text + structured observation ingest
  attention/   task-relevant prioritization
  goals/       goal lifecycle + bounded planner
  learning/    strategy selection, retrieval re-weighting (NOT online ML)
  evaluation/  metrics, task battery, experiment runner
  safety/      budgets, permissions allowlist, append-only audit, kill switch
  api/         stdlib HTTP API over live backend state
experiments/
  protocols/   10 falsifiable experiment protocols (JSON)
  baselines/   matched no-component baseline agent
  ablations/   (via configs/ablation_*.yaml)
  results/     versioned experiment results (JSON)
configs/       default.yaml, mock.yaml, ablation_*.yaml
scripts/       run_experiment.py, run_baseline.py, demo_loop.py
tests/         unit + integration tests (mock-labeled)
docs/          architecture, safety, design decisions, roadmap
```

## Install

Requires Python ≥ 3.10. The engine itself is **stdlib-only**; only the
test runner needs installing.

```bash
cd artificial-consciousness-lab
python3 -m venv .venv
.venv/bin/pip install pytest
```

No API keys are needed: the default provider is a deterministic mock, so the
full test suite and all 10 experiment protocols run with ₹0 cost, no
network, and no GPU.

## Run

```bash
# 1. Cognitive-loop demo (mock provider, 3 cycles)
.venv/bin/python scripts/demo_loop.py --config configs/mock.yaml --cycles 3

# 2. Full test suite (204 tests)
.venv/bin/python -m pytest -q

# 3. One experiment: calibration, full vs baseline vs ablated metacognition
.venv/bin/python scripts/run_experiment.py \
  --protocol experiments/protocols/03_calibration.json \
  --config configs/mock.yaml \
  --conditions full,baseline,ablation:metacognition

# 4. Full component-ablation study (protocol decides the conditions)
.venv/bin/python scripts/run_experiment.py \
  --protocol experiments/protocols/10_component_ablation.json \
  --config configs/mock.yaml

# 5. Standalone baseline on the QA task
.venv/bin/python scripts/run_baseline.py \
  --task-kind qa_calibration --config configs/mock.yaml

# 6. Research API (stdlib HTTP, live backend state, no UI required)
.venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
from consciousness_lab.core.config import Config
from consciousness_lab.core.loop import build_loop
from consciousness_lab.api.server import serve
cfg = Config.load('configs/mock.yaml')
loop = build_loop(cfg)
lab = {'loop': loop, 'config': cfg, 'audit': loop.audit,
       'safety': loop.safety, 'runner': None}
serve(lab, port=8765)"
# then: curl localhost:8765/api/workspace | head -c 400
```

Results land in `experiments/results/<name>_<timestamp>_<seed>.json`
(config snapshot, seeds, per-trial metrics, deltas vs baseline).

## Real providers

Copy `.env.example` to `.env` (never commit it) and set credentials —
**environment variables only, never in config files**:

| Provider | Env vars |
|---|---|
| Ollama (local) | `OLLAMA_BASE_URL`, `OLLAMA_MODEL` |
| OpenAI-compatible | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` |
| Gemini | `GEMINI_API_KEY`, `GEMINI_MODEL` |

```bash
ACL_PROVIDER=ollama OLLAMA_MODEL=qwen3:4b \
  .venv/bin/python scripts/demo_loop.py --config configs/default.yaml --cycles 3
```

Documented limits: Ollama needs a local server; Gemini sends the key as a
query parameter (see `models/gemini.py`); all adapters retry 429/5xx with
backoff and raise `ProviderError` (never leak secrets into logs).

## Configuration

`configs/default.yaml` documents every key. The YAML parser accepts a
documented subset only (mappings, `- ` lists, scalars); anything else —
including anchors — raises `ConfigError`. Environment overrides
(`ACL_PROVIDER`, `ACL_DB_PATH`, `ACL_SEED`, `ACL_MAX_STEPS`, …) win over
the file. Component ablations are config flags:

```yaml
components:
  memory: false   # disables episodic/semantic/outcome/working memory
```

## What the experiments actually show (2026-10-13, mock provider)

Honest summary — mock runs validate the *measurement pipeline*; the mock's
behavior is scripted by construction, so architectural signal needs a real
provider:

- **Contradiction detection (protocol 04):** full system F1 = 0.80
  (precision 1.00, recall 0.67 — catches negation and numeric-mismatch
  cases, misses same-slot substitutions like Paris/Berlin, a documented
  heuristic limit). `ablation:memory` and `ablation:metacognition` both
  score 0.00, exactly the predicted ablation effect. Baseline 0.00.
- **Calibration (protocol 03):** full vs baseline deltas ≈ 0 on the mock —
  expected, since the QA task calls the provider directly and the mock is
  calibrated by construction. The pipeline (parsing, Brier/ECE, aggregation)
  is verified end-to-end.
- **Task recovery (protocol 06):** goal state survives a loop rebuild on the
  same SQLite DB: progress preserved 1.00, continuation correct 1.00.
- **Memory continuity (protocol 01):** recall@5 = 1.00 across close/reopen.
- **Self-model probe (protocol 02):** accuracy 0.92 over 5 probes (heuristic
  probe design; see `evaluation/tasks.py` limits).

Full per-trial data: `experiments/results/`. Every protocol file states its
research question, falsifiable hypothesis, failure condition, metrics, and
limitations.

## Safety

Bounded execution (steps/tokens/wall-time/tool calls), tool allowlist with
default-deny, `requires_approval` actions always raise `PermissionDenied`
(no auto-approval), append-only audit log, `SafetyController.stop()`, and
autonomous execution disable via config. The system cannot modify its own
safety controls (`grant`/`revoke`/`disable_autonomous` require
`authorized=True`, which the system cannot grant itself). External content
is treated as untrusted input. See `docs/safety.md`.

## Status and next milestones

- [x] Phase 4: smallest complete functioning cognitive engine
- [x] Phase 5: experiment runner + baselines + ablations (mock-verified)
- [ ] Phase 2/3: literature review + formal research thesis (`research/`)
- [ ] Real-provider experiment runs (needs API access)
- [x] Web dashboard over the live API (`web/index.html`, served at `/` by `scripts/serve_dashboard.py`)
- [ ] Factorial ablation design (interaction effects)

## License

MIT — see `LICENSE`.
