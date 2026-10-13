# Roadmap

## v0.1 — Research engine MVP (current milestone)
- [x] Repo scaffold, architecture spec, safety policy
- [ ] Literature review + falsifiable thesis (research agent in progress)
- [ ] Core modules: config, logging, errors, SQLite persistence
- [ ] Provider abstraction: mock (deterministic), Ollama, OpenAI-compatible, Gemini adapters
- [ ] Workspace (capacity-limited selection + broadcast), attention, perception
- [ ] Memory: episodic, semantic, working, outcome; provenance + contradiction handling
- [ ] Self-model grounded in observables; metacognition (confidence, contradiction, error detection, belief revision)
- [ ] Goals/planning with budgets and approval gates
- [ ] Experiment runner: fixed configs, seeds, versioned JSON results
- [ ] Baselines (matched LLM agent) + ablations (no-memory, no-self-model, no-workspace, no-metacognition)
- [ ] Unit + integration tests (mock provider), persistence tests, provider failure tests
- [ ] 10 protocol experiments (at least 1 baseline + 1 ablation executed in this environment)
- [ ] Minimal web UI backed by real API data

## v0.2 — Measurement hardening
- Calibration on held-out tasks (Brier score, ECE); held-out evaluation sets
- Statistical analysis helpers (paired tests, confidence intervals)
- Larger protocol battery; pre-registration of hypotheses

## v0.3 — External validity
- Real-provider runs (Ollama local; OpenAI/Gemini only with explicit credentials)
- Academic collaboration outreach (see docs/funding_strategy.md)

## Non-goals
- Claims of consciousness or subjective experience.
- Unbounded autonomous execution.
- Fine-tuning / RL until experiments justify the cost.
