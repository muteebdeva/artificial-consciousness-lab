# Architecture

## Scientific stance

This platform investigates whether computational architectures can support properties
associated with consciousness. It does **not** assume consciousness has been achieved.
Subjective experience is treated as an open scientific question.

We distinguish:
- **Functional cognitive capabilities** (measurable: memory, planning, calibration)
- **Operational self-monitoring** (measurable: self-model accuracy, error detection)
- **Behavioral indicators** potentially relevant to consciousness theories (e.g., global
  broadcast, metacognitive sensitivity) — interpreted cautiously
- **Theoretical interpretations** (explicitly labeled as interpretation, not evidence)
- **Insufficient evidence**: fluent conversation, persistent memory, goal-directed
  behavior, self-reference, or elaborate UI do **not** establish subjective experience.
  AI self-reports ("I am conscious") are never treated as evidence.

## Working hypothesis (v0.1, to be refined by research/research_thesis.md)

> A cognitive architecture combining (a) a limited-capacity global workspace with
> competitive attention-based selection and broadcast, (b) persistent episodic +
> semantic memory with provenance and confidence, and (c) metacognitive confidence
> calibration and contradiction detection, will measurably outperform a matched
> baseline (same model, tools, task budget, no workspace/memory/metacognition) on:
> 1. Cross-session memory continuity
> 2. Self-model accuracy (predicting its own capabilities/limits)
> 3. Confidence calibration (predicted vs. observed accuracy)
> 4. Error and contradiction detection
>
> Ablating any single component will degrade the specific capability it supports.
> Failure condition: no statistically reliable improvement over baseline on any
> of the four, or improvements fully explained by extra compute/tokens.

## System overview

```
                  +-------------------+
                  |   Perception      |  text + structured observations
                  +--------+----------+
                           |
                  +--------v----------+      +------------------+
                  |    Attention      |----->|  Working Memory  |
                  | (task-relevant   |      |  (limited span)  |
                  |  prioritization) |      +--------+---------+
                  +--------+----------+               |
                           |                          |
                  +--------v--------------------------v---------+
                  |            Global Workspace                   |
                  |  limited capacity, competitive selection,     |
                  |  broadcast to consumers                       |
                  +--+------+------+------+------+------+--------+
                     |      |      |      |      |      |
                     v      v      v      v      v      v
                  Memory  Self-  Meta-  Goals/ Planning Learning
                  (episodic,    model  cog.   (decomp,  (memory
                   semantic,           (conf,  progress, updates,
                   outcome)      uncertainty) action   belief
                                              select)  revision)
                     |      |      |      |      |      |
                     +------+--+---+--+---+--+---+--+---+
                                |
                        +-------v--------+
                        | Inference      |
                        | Engine         |
                        | (provider      |
                        |  abstraction)  |
                        +----------------+
```

Information flow per cognitive cycle:
1. Perception ingests observation (text/structured) → normalized `Observation`.
2. Attention scores relevance against active goals + workspace state → prioritized items.
3. Workspace selects top-k items under capacity limit, resolves conflicts, broadcasts.
4. Memory retrieves relevant episodic/semantic context (with provenance + confidence).
5. Self-model provides grounded capability/limitation snapshot.
6. Metacognition attaches uncertainty estimates, flags contradictions / missing info.
7. Goals/Planning decompose task, select next action within budgets and permissions.
8. Inference engine executes model call via provider abstraction.
9. Outcome recorded: prediction vs. observation → learning updates (memory, beliefs).
10. Safety layer enforces budgets, permissions, audit logging throughout.

## Module interfaces

### core
- `CognitiveLoop`: `step(observation) -> CycleResult`; `run(task, budgets) -> TaskResult`
- `Config`: typed config from env/file; `get(key, default)`
- Errors: typed exceptions (`ProviderError`, `BudgetExceeded`, `PermissionDenied`,
  `ContradictionError`); never silently swallow.

### workspace
- `GlobalWorkspace(capacity: int)`
  - `select(candidates: list[WorkspaceItem]) -> list[WorkspaceItem]`: competitive
    selection by salience × relevance × novelty, capacity-limited.
  - `broadcast(items) -> None`: distribute to registered consumers.
  - `snapshot() -> WorkspaceState`: inspectable current content.
- `WorkspaceItem`: `{id, content, source, salience, confidence, timestamp, provenance}`

### memory
- Backed by SQLite (`persistence.SqliteStore`) with transactions.
- `EpisodicMemory`: `record(event)`, `recall(query, k) -> list[MemoryRecord]`
- `SemanticMemory`: `upsert_fact(fact, provenance, confidence)`, `query(q, k)`
- `WorkingMemory`: in-memory, capacity-limited, task-scoped.
- `OutcomeMemory`: `record_prediction(pred)`, `record_outcome(pred_id, observed)`,
  `calibration() -> CalibrationReport`
- `MemoryRecord`: `{id, kind, content, provenance, confidence, timestamp,
  source_type: verified|user_provided|model_hypothesis|inferred, supersedes}`
- Contradiction handling: new fact contradicting high-confidence existing fact →
  flag for metacognition, do not silently overwrite; keep both with provenance.
- Deletion: `delete(record_id, reason)` with audit log.

### self_model
- `SelfModel`: structured, continuously updated from **actual** system state.
  - `capabilities: list[str]` — derived from registered tools/providers, not invented.
  - `limitations: list[str]` — from config, provider capability reports, past failures.
  - `current_task`, `objectives`, `resources` (budgets remaining).
  - `recent_actions`, `outcomes`, `uncertainty_estimates`.
  - `unresolved_questions`, `performance_predictions`.
- `update_from_observation(obs)`, `update_from_tool_result(result)`,
  `predict_performance(task) -> Prediction` (recorded, later scored).

### metacognition
- `ConfidenceEstimator`: `estimate(context) -> Confidence {value, basis}`.
- `ContradictionDetector`: `check(new_info, memory) -> list[Contradiction]`.
- `ErrorDetector`: `evaluate_prediction(prediction, outcome) -> ErrorReport`.
- `BeliefRevision`: `revise(belief, evidence) -> RevisedBelief` with provenance chain.
- Calibration tracked on held-out tasks: Brier score, expected calibration error.

### perception / attention
- `Perception`: `ingest(raw) -> Observation {modality, content, timestamp, metadata}`.
  Text + structured first; image/audio are optional extensions.
- `Attention`: `prioritize(items, goals, workspace_state) -> scored items`.
  Evaluate: does attention+workspace improve task performance vs. no-attention
  baseline? (Experiment 7.)

### goals / planning
- `Goal`: `{id, description, success_criteria, status, parent_id, created_at}`.
- `GoalManager`: `create`, `decompose(goal) -> subgoals`, `track_progress`,
  `mark_complete/abandoned` with reason.
- `Planner`: `next_action(goal, workspace_state, self_model) -> Action | None`.
- `Action`: `{type, payload, requires_approval, budget_cost}`.
- Bounded tasks: max steps, max tokens, max wall-time; explicit execution limits.

### learning
- Measurable mechanisms only: memory updates, retrieval improvement (e.g., upweight
  useful records), strategy selection (choose among strategies by past success),
  explicit belief revision.
- Do **not** claim online learning when the system merely stores new text.

### evaluation
- `ExperimentRunner`: fixed configs, seeds, versioned results, structured logs.
- Protocols in `experiments/protocols/`; results in `experiments/results/` (JSON).
- Baselines: standard LLM agent with equivalent model access, tools, task budgets.
- Ablations: disable persistent memory, self-model, workspace integration,
  metacognitive evaluation — matched conditions.

### safety
- `Budget`: tokens, steps, wall-time, tool calls.
- `Permissions`: tool allowlist; external/destructive/financial/sensitive actions
  require explicit approval (`requires_approval=True` → human gate).
- `AuditLog`: append-only record of actions, approvals, memory writes.
- Controls: stop active run, disable autonomous execution, inspect/delete memories.
- The system cannot alter its own safety controls without explicit authorization.
- No uncontrolled self-replication, unrestricted resource acquisition, or
  unrestricted autonomous execution.

### models (inference engine)
- `Provider` abstract: `complete(prompt, params) -> Completion`,
  `capabilities() -> ProviderCapabilities`.
- Implementations: `MockProvider` (deterministic, for tests), `OllamaProvider`,
  `OpenAICompatibleProvider`, `GeminiProvider` (adapters, documented limits).
- Config via env; never hardcode keys; no secrets in logs.
- Handle timeouts, rate limits, malformed responses, retries with backoff.
- Meaningful subset of experiments must run on mock provider (no paid APIs/GPU).

### persistence
- `SqliteStore`: single SQLite file, WAL mode, parameterized queries,
  transaction-per-write, schema versioned with migrations.

### api
- Minimal HTTP API (stdlib only) exposing research state: workspace snapshot,
  memory query + provenance, self-model, goals/progress, uncertainty, predictions
  vs. outcomes, module traces, experiment configs/results, resource usage, errors.
- Backend runs without the UI. UI displays actual backend data only; inferred or
  model-generated state is labeled as such. No decorative charts with invented values.

## Design decisions (summary)
See `docs/design_decisions.md` for full rationale.
- SQLite over vector DB: zero-dependency, transactional, sufficient for v0.1;
  retrieval via keyword + recency + confidence scoring (no claim of semantic search).
- Stdlib-first: no numpy/torch dependency for core; keeps CPU-only execution viable.
- Mock provider default: reproducibility and ₹0-cost operation.
- Capacity-limited workspace: forces selection pressure, the theoretically
  interesting property (cf. GNWT), and makes ablation meaningful.
