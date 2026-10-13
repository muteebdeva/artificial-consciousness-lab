# Interface Contract — artificial-consciousness-lab v0.1

All modules live under `src/consciousness_lab/`. **Stdlib only** in `src`
(no numpy/torch/requests). Type hints everywhere. No hardcoded secrets.
No `print()` — use the logger from `core.logging_setup`.
Every public class/function has a docstring stating what it does AND its limits.

Conventions:
- Timestamps: `time.time()` floats (UTC epoch).
- IDs: `uuid.uuid4().hex`.
- Errors: import from `consciousness_lab.core.errors`.
- Determinism: any randomness takes an explicit `seed` parameter.

---

## core/config.py

```python
@dataclass
class BudgetConfig:
    max_steps: int = 20
    max_tokens: int = 8000
    max_wall_time_s: float = 120.0
    max_tool_calls: int = 10

@dataclass
class Config:
    db_path: str = "./data/acl.sqlite3"
    provider: str = "mock"                      # mock | ollama | openai_compat | gemini
    provider_model: str = ""                    # provider-specific model name (may be "")
    provider_params: dict = field(default_factory=dict)  # e.g. {"timeout_s": 30}
    workspace_capacity: int = 7
    budgets: BudgetConfig = field(default_factory=BudgetConfig)
    autonomous_enabled: bool = True
    approval_required_default: bool = False
    log_level: str = "INFO"
    seed: int = 0
    results_dir: str = "experiments/results"
    components: dict = field(default_factory=lambda: {
        "memory": True, "self_model": True, "workspace": True,
        "metacognition": True, "attention": True, "learning": True,
    })
    extra: dict = field(default_factory=dict)   # any other keys

    @classmethod
    def load(cls, path: str | None = None) -> "Config": ...
    def get(self, dotted_key: str, default=None): ...   # e.g. get("budgets.max_steps")
```

- `Config.load(path)`: parse YAML **subset** (see below); env vars override file.
  Env: `ACL_DB_PATH`, `ACL_PROVIDER`, `ACL_PROVIDER_MODEL`, `ACL_LOG_LEVEL`,
  `ACL_AUTONOMOUS_ENABLED` (true/false), `ACL_SEED`, `ACL_RESULTS_DIR`,
  `ACL_WORKSPACE_CAPACITY`, `ACL_MAX_STEPS`, `ACL_MAX_TOKENS`,
  `ACL_MAX_WALL_TIME_S`, `ACL_MAX_TOOL_CALLS`.
  Provider secrets ONLY from env: `OLLAMA_BASE_URL`, `OPENAI_API_KEY`,
  `OPENAI_BASE_URL`, `GEMINI_API_KEY`, `GEMINI_MODEL`, `OLLAMA_MODEL`,
  `OPENAI_MODEL`. Never read secrets from the YAML file.
- YAML subset supported: `#` comments, `key: value` pairs, nesting via 2-space
  indent, lists via `- ` items, scalars: int, float, `true`/`false`, bare or
  `"double-quoted"` strings. Anything else → `ConfigError`.
- `get("a.b.c", default)` walks nested dataclass fields and dicts.

## core/errors.py

```python
class LabError(Exception): ...
class ConfigError(LabError): ...
class ProviderError(LabError):
    def __init__(self, msg, *, retryable: bool = False): ...
class BudgetExceeded(LabError): ...
class PermissionDenied(LabError): ...
class ContradictionError(LabError): ...
class ExperimentError(LabError): ...
class SafetyError(LabError): ...
```

## core/logging_setup.py

```python
def setup_logging(level: str = "INFO", *, name: str = "consciousness_lab") -> logging.Logger: ...
```
Single-line structured records: `ts=... level=... logger=... msg=... k=v ...`.
Must redact values whose key contains `key|token|secret|password|api_key`.

## core/loop.py

```python
@dataclass
class CycleResult:
    cycle: int
    observation_id: str
    selected_ids: list[str]
    action: "Action | None"          # from goals.planning, kept as dict-safe
    provider_calls: int
    tokens_used: int
    predictions: list[dict]
    contradictions: list[dict]
    confidence: float | None
    errors: list[str]
    duration_s: float

@dataclass
class TaskResult:
    task: str
    status: str                      # completed | budget_exhausted | stopped | failed
    cycles: int
    tokens_used: int
    goal_status: dict
    predictions_scored: int
    errors: list[str]

class CognitiveLoop:
    def __init__(self, config: Config, provider: "Provider", **components): ...
    def step(self, raw_observation: str | dict) -> CycleResult: ...
    def run(self, task: str, *, max_cycles: int | None = None) -> TaskResult: ...
    def snapshot(self) -> dict: ...  # for the API: workspace, self_model, goals, budgets
```

`build_loop(config: Config) -> CognitiveLoop` — factory: creates provider via
`models.create_provider`, builds every component, honors
`config.components` flags (a disabled component is passed as `None` and the
loop degrades gracefully: no workspace → pass candidates through unfiltered;
no memory → skip recall; no self_model → empty snapshot; no metacognition →
skip confidence/contradiction checks; no attention → keep input order;
no learning → skip outcome learning).

`step()` flow (each stage guarded by try/except → recorded in `errors`, never
silently swallowed; a failed non-critical stage degrades, a failed critical
stage — perception, budget — aborts the cycle with status):
1. `obs = perception.ingest(raw)`; `safety.consume_step()` (BudgetExceeded → stop)
2. Build candidate `WorkspaceItem`s from: the observation, working-memory items,
   memory recall hits (if memory on), active goal descriptions, self-model
   uncertainty flags (if self_model on).
3. `scored = attention.prioritize(candidates, goals, workspace.snapshot())`
   (or input order if attention off)
4. `selected = workspace.select(scored)` (or all candidates if workspace off)
5. `workspace.broadcast(selected)`
6. `contradictions = metacognition.contradiction_detector.check(...)` (if on)
7. `confidence = metacognition.confidence_estimator.estimate({...})` (if on)
8. `goal = goal_manager.active_goals()[0] if any else None`;
   `action = planner.next_action(goal, workspace.snapshot(), self_model.snapshot() or {})`
9. If `action.requires_approval` → do NOT execute; record as pending, cycle ends.
10. Execute: `action.type == "respond"` → build prompt from selected items +
    memory context + goal; `provider.complete(...)`; `safety.consume_tokens(...)`.
    `action.type == "noop"` → nothing. `action.type == "tool"` → permission
    check via `safety.permissions.allowed(...)`; PermissionDenied → record, skip.
11. Record prediction/outcome: `outcome_memory.record_prediction(...)` when the
    provider text contains a verifiable claim (loop extracts `PREDICT:` lines;
    if none, skip).
12. `self_model.update_from_observation(obs)`,
    `self_model.update_from_tool_result({...})` (if on)
13. `learning.observe_cycle({...})` (if on)
14. `audit.log("cycle", {...})`
15. Return `CycleResult`.

`run(task, ...)`: creates root goal via `goal_manager.create(task, ...)`,
loops `step()` with a generated observation each cycle
(`{"modality":"task","content":task,"metadata":{"cycle":n}}`) until goal
complete, budget exhausted, or `safety.stopped`.

## persistence/sqlite_store.py

```python
class SqliteStore:
    def __init__(self, path: str): ...            # creates parent dirs; WAL mode
    def ensure_table(self, name: str, ddl: str) -> None: ...
    def execute_write(self, sql: str, params: tuple = ()) -> int: ...   # returns lastrowid; one transaction per call
    def execute_read(self, sql: str, params: tuple = ()) -> list[dict]: ...
    def insert(self, table: str, record: dict) -> str: ...   # returns row id if single PK
    def transaction(self): ...                   # context manager for multi-statement writes
    @property
    def schema_version(self) -> int: ...
    def migrate(self, migrations: dict[int, list[str]]) -> None: ...
    def close(self) -> None: ...
```
- WAL mode (`PRAGMA journal_mode=WAL`), parameterized queries only,
  `busy_timeout` set. Schema version table `_schema(version INTEGER)`.
- Module table DDLS live in the owning module, registered via `ensure_table`.

## models/base.py

```python
@dataclass
class Completion:
    text: str
    model: str
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0

@dataclass
class ProviderCapabilities:
    name: str
    streaming: bool = False
    json_mode: bool = False
    max_context_tokens: int | None = None
    notes: str = ""

class Provider(abc.ABC):
    name: str
    @abc.abstractmethod
    def complete(self, prompt: str, params: dict | None = None) -> Completion: ...
    @abc.abstractmethod
    def capabilities(self) -> ProviderCapabilities: ...
    def health_check(self) -> bool: ...   # default: try tiny complete, catch → False

def with_retries(fn, *, max_attempts: int = 3, base_delay: float = 0.5,
                 retryable: Callable[[Exception], bool] = lambda e: isinstance(e, ProviderError) and e.retryable):
    ...  # exponential backoff with jitter; re-raises last error
```

## models/__init__.py
`create_provider(name: str, config: Config) -> Provider` — reads credentials
from env only. Unknown name → `ConfigError`.

## models/mock.py
```python
class MockProvider(Provider):
    name = "mock"
    def __init__(self, seed: int = 0, script: dict[str, str] | None = None): ...
```
Deterministic: same `(seed, prompt)` → same text. `script`: if any key is a
substring of the prompt, return its value (first match in insertion order).
Otherwise: `f"MOCK[{seed}]: "` + deterministic template chosen by
`sha256(f"{seed}:{prompt}")` from a small fixed set of neutral analytical
responses (must not claim consciousness/feelings). `capabilities()` reports
`notes="deterministic mock; not a real model"`. Token counts: estimated
`len(text.split())`.

## models/ollama.py, openai_compat.py, gemini.py
Real adapters over `urllib` (stdlib). Timeouts from config
(`provider_params.get("timeout_s", 30)`), `with_retries` on 429/5xx with
`retryable=True`, malformed JSON → `ProviderError`. Never log secrets or full
prompts at INFO. Each module docstring documents limits, e.g. Ollama needs a
local server; Gemini key sent as query param (documented limitation).

## workspace/workspace.py

```python
@dataclass
class WorkspaceItem:
    id: str
    content: str
    source: str                      # e.g. "perception", "memory:episodic", "goal", "self_model"
    salience: float = 0.5            # 0..1
    relevance: float | None = None   # 0..1, default 1.0 when None
    novelty: float | None = None     # 0..1, default 1.0 when None
    confidence: float = 0.5          # 0..1
    attention_score: float | None = None
    timestamp: float = field(default_factory=time.time)
    provenance: dict = field(default_factory=dict)

    def priority(self) -> float: ...  # salience * (relevance or 1) * (novelty or 1)

@dataclass
class WorkspaceState:
    items: list[WorkspaceItem]
    capacity: int
    cycle: int
    updated_at: float

class GlobalWorkspace:
    def __init__(self, capacity: int = 7): ...
    def select(self, candidates: list[WorkspaceItem]) -> list[WorkspaceItem]: ...
    def broadcast(self, items: list[WorkspaceItem]) -> None: ...
    def register_consumer(self, fn: Callable[[list[WorkspaceItem]], None]) -> None: ...
    def snapshot(self) -> WorkspaceState: ...
    def clear(self) -> None: ...
```
`select`: rank by `priority()` desc, tie-break by earliest timestamp, keep top
`capacity`; store as current state; increment cycle. Consumer exceptions are
caught, logged, and re-raised as a group? No — caught, logged, counted; a
failing consumer must not break broadcast (documented).

## memory/records.py

```python
SourceType = Literal["verified", "user_provided", "model_hypothesis", "inferred"]

@dataclass
class MemoryRecord:
    id: str
    kind: str                        # episodic | semantic | outcome | working
    content: str
    provenance: str                  # free text: where it came from
    confidence: float                # 0..1
    timestamp: float
    source_type: SourceType
    supersedes: str | None = None
    metadata: dict = field(default_factory=dict)

    def validate(self) -> None: ...  # raises ValueError on bad confidence/source_type/empty content
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "MemoryRecord": ...
```

## memory/episodic.py, semantic.py, working.py, outcome.py

```python
class EpisodicMemory:
    def __init__(self, store: SqliteStore, *, namespace: str = "default"): ...
    def record(self, content: str, *, provenance: str, confidence: float = 0.5,
               source_type: SourceType = "inferred", metadata: dict | None = None) -> MemoryRecord: ...
    def recall(self, query: str, k: int = 5) -> list[MemoryRecord]: ...
    def get(self, record_id: str) -> MemoryRecord | None: ...
    def delete(self, record_id: str, reason: str) -> None: ...   # reason required; audit via callback
```
Scoring for recall/query: keyword overlap (fraction of query tokens present) *
0.5 + recency (1/(1+age_hours)) * 0.3 + confidence * 0.2. Documented heuristic,
not semantic search.

```python
class SemanticMemory:
    def __init__(self, store: SqliteStore, *, namespace: str = "default",
                 on_contradiction: Callable[["Contradiction"], None] | None = None): ...
    def upsert_fact(self, content: str, *, provenance: str, confidence: float = 0.5,
                    source_type: SourceType = "inferred") -> tuple[MemoryRecord, list["Contradiction"]]: ...
    def query(self, q: str, k: int = 5) -> list[MemoryRecord]: ...
    def get(self, record_id: str) -> MemoryRecord | None: ...
    def delete(self, record_id: str, reason: str) -> None: ...
```
`upsert_fact`: checks existing facts for contradiction via
`metacognition.ContradictionDetector`; on conflict: keep BOTH records,
set `supersedes` only when the new fact is explicitly marked as replacing
(`supersedes_id` param), return contradictions list, invoke `on_contradiction`.
Never silently overwrite.

```python
class WorkingMemory:
    def __init__(self, capacity: int = 10): ...   # in-memory only, NOT persisted
    def put(self, item: WorkspaceItem | MemoryRecord | str) -> None: ...
    def contents(self) -> list: ...
    def clear(self) -> None: ...
```

```python
@dataclass
class Prediction:
    id: str
    content: str
    confidence: float
    timestamp: float
    context: dict = field(default_factory=dict)
    outcome: bool | None = None
    outcome_timestamp: float | None = None
    outcome_note: str = ""

@dataclass
class CalibrationReport:
    n: int
    brier_score: float
    ece: float
    bins: list[dict]      # [{bin_low, bin_high, n, avg_confidence, accuracy}]
    generated_at: float

class OutcomeMemory:
    def __init__(self, store: SqliteStore, *, namespace: str = "default"): ...
    def record_prediction(self, content: str, confidence: float, context: dict | None = None) -> Prediction: ...
    def record_outcome(self, prediction_id: str, observed: bool, note: str = "") -> Prediction: ...
    def pending(self) -> list[Prediction]: ...
    def calibration(self) -> CalibrationReport: ...
```
Brier = mean((p − o)²); ECE over 10 equal-width bins. `n == 0` → report with
NaNs? No — return `n=0, brier=float("nan"), ece=float("nan")` documented.

`memory/__init__.py` exports all + `MEMORY_DDL: dict[str, str]`? No — each
class ensures its own table on init: tables `episodic`, `semantic`, `outcome_predictions`.
Columns must round-trip `MemoryRecord`/`Prediction`.

## self_model/self_model.py

```python
@dataclass
class PerformancePrediction:
    id: str
    task: str
    predicted_success: bool
    confidence: float
    timestamp: float
    actual: bool | None = None

class SelfModel:
    def __init__(self, store: SqliteStore | None = None): ...
    def register_capability(self, name: str, source: str) -> None: ...
    def register_limitation(self, text: str, source: str) -> None: ...
    def register_tool(self, name: str, description: str) -> None: ...  # tools → capabilities
    def update_from_observation(self, obs: dict) -> None: ...
    def update_from_tool_result(self, result: dict) -> None: ...
        # result: {"tool": str, "success": bool, "detail": str}
        # failures append to limitations with source "observed failure"
    def set_task(self, task: str, objectives: list[str]) -> None: ...
    def update_resources(self, budgets: dict) -> None: ...
    def add_unresolved(self, question: str) -> None: ...
    def resolve_unresolved(self, question: str, resolution: str) -> None: ...
    def predict_performance(self, task: str, predicted_success: bool, confidence: float) -> PerformancePrediction: ...
    def score_prediction(self, prediction_id: str, actual: bool) -> None: ...
    def accuracy(self) -> dict: ...   # {"n": int, "accuracy": float|None, "brier": float|None}
    def snapshot(self) -> dict: ...
```
Every mutation records `source`. `snapshot()` returns plain dict for API.
Capabilities come ONLY from `register_tool`/`register_capability` calls made by
the loop factory with real tools/providers — never invented strings.

## metacognition/

```python
# confidence.py
@dataclass
class Confidence:
    value: float          # 0..1
    basis: str            # human-readable justification
    factors: dict = field(default_factory=dict)

class ConfidenceEstimator:
    def estimate(self, context: dict) -> Confidence: ...
```
`context` keys: `sources: list[str]` (source_types), `n_evidence: int`,
`agreement: float` (0..1), `recency: float`, `base_rate: float = 0.5`.
Heuristic (documented): start at base_rate; +0.1 per verified source (cap);
−0.15 per model_hypothesis-only evidence; scale by agreement. Clamp 0.01..0.99.

```python
# contradiction.py
@dataclass
class Contradiction:
    id: str
    record_a_id: str | None
    record_b_id: str | None
    description: str
    severity: float          # 0..1
    timestamp: float

class ContradictionDetector:
    def check(self, new_content: str, existing: list[MemoryRecord]) -> list[Contradiction]: ...
    def check_pair(self, a: str, b: str) -> Contradiction | None: ...
```
Heuristic (documented, not NLP): normalized token-set overlap + negation cue
list (`not`, `never`, `no`, `n't`, `false`, `incorrect`, `opposite`,
`however`...). Flag when high overlap AND (negation asymmetry OR explicit
numeric mismatch via regex numbers). Severity = overlap * cue strength.

```python
# error_detection.py
@dataclass
class ErrorReport:
    prediction_id: str
    predicted_confidence: float
    observed: bool
    correct: bool
    brier_contribution: float
    miscalibrated: bool          # |p - o| > 0.5
    timestamp: float

class ErrorDetector:
    def evaluate_prediction(self, prediction: "Prediction", observed: bool) -> ErrorReport: ...
    def scan_trace(self, trace: list[dict]) -> list[dict]: ...
        # trace entries {"stage": str, "ok": bool, "detail": str} → returns failed stages
```

```python
# belief_revision.py
@dataclass
class RevisedBelief:
    record_id: str
    old_confidence: float
    new_confidence: float
    evidence: str
    provenance_chain: list[str]     # ordered list of provenance strings
    timestamp: float

class BeliefRevision:
    def revise(self, record: MemoryRecord, evidence: str, evidence_confidence: float,
               store_update: Callable[[MemoryRecord], None]) -> RevisedBelief: ...
```
Bayesian-ish update (documented heuristic): treat confidences as probabilities,
`new = (old*w_old + ev*w_ev)/(w_old+w_ev)` with w_old=1, w_ev=evidence_confidence.
Appends new record version with `supersedes=old_id` via `store_update`; never
mutates in place silently.

## perception/perception.py

```python
@dataclass
class Observation:
    id: str
    modality: str               # "text" | "task" | "structured"
    content: str
    timestamp: float
    metadata: dict
    source: str = "external"

class Perception:
    def ingest(self, raw: str | dict) -> Observation: ...
```
Sanitize: strip control chars (except \n\t), cap length at 20_000 chars
(documented). Dict form: `{"modality":..., "content":..., "metadata":...,
"source":...}`. Unknown modality → ValueError.

## attention/attention.py

```python
class Attention:
    def prioritize(self, items: list[WorkspaceItem], goals: list,
                   workspace_state: WorkspaceState | None) -> list[WorkspaceItem]: ...
```
Score = `item.salience * (1 + goal_overlap)` where goal_overlap = fraction of
goal keyword tokens present in item content; sets `item.attention_score`,
returns sorted desc. Goals are `Goal` objects or dicts with `description`.
Documented heuristic.

## goals/goals.py

```python
@dataclass
class Goal:
    id: str
    description: str
    success_criteria: str = ""
    status: str = "pending"          # pending|active|completed|abandoned
    parent_id: str | None = None
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

class GoalManager:
    def __init__(self, store: SqliteStore | None = None): ...
    def create(self, description: str, success_criteria: str = "", parent_id: str | None = None) -> Goal: ...
    def get(self, goal_id: str) -> Goal | None: ...
    def active_goals(self) -> list[Goal]: ...
    def decompose(self, goal: Goal) -> list[Goal]: ...
    def track_progress(self, goal_id: str, progress: float, note: str = "") -> Goal: ...
    def mark_complete(self, goal_id: str, reason: str = "") -> Goal: ...
    def mark_abandoned(self, goal_id: str, reason: str) -> Goal: ...
```
`decompose`: naive baseline — split description into clauses on `;`/`then`/
numbered steps; if no splits, create 2 subgoals ("gather information",
"produce result"). Documented as placeholder heuristic for the planner to
improve later. Persists to `goals` table when store given.

## goals/planning.py

```python
@dataclass
class Action:
    id: str
    type: str                      # "respond" | "noop" | "tool" | "await_approval"
    payload: dict = field(default_factory=dict)
    requires_approval: bool = False
    budget_cost: dict = field(default_factory=dict)   # {"tokens": int, "tool_calls": int}
    created_at: float = field(default_factory=time.time)

class Planner:
    def __init__(self, budgets: BudgetConfig, permissions: "Permissions"): ...
    def next_action(self, goal: Goal | None, workspace_state: WorkspaceState | dict,
                    self_model: dict) -> Action | None: ...
```
- `goal is None` → `Action("noop", ...)`.
- Goal complete (progress ≥ 1.0) → None.
- Default: `Action("respond", payload={"goal_id", "instruction"}, requires_approval=False, budget_cost={"tokens": 600})`.
- If `self_model` lists a needed tool not in permissions allowlist →
  `Action("tool", requires_approval=True)`.
- Never exceed budgets: if `budgets` exhausted → None (loop checks first).

## learning/learning.py

```python
class Learning:
    """Memory/retrieval/strategy adaptation. NOT online ML training (see DD-07)."""
    def __init__(self, store: SqliteStore | None = None): ...
    def record_strategy_use(self, strategy: str, success: bool, context: str = "") -> None: ...
    def select_strategy(self, candidates: list[str]) -> str: ...
        # highest observed success rate; ties → first; unknown → first candidate
    def note_useful_record(self, record_id: str) -> None: ...   # increments usefulness count
    def retrieval_weight(self, record_id: str) -> float: ...    # 1.0 + 0.1*uses (cap 2.0)
    def observe_cycle(self, info: dict) -> None: ...
        # info: {"action_type": str, "success": bool|None, "strategy": str|None}
    def strategy_stats(self) -> dict: ...
```

## evaluation/metrics.py
Pure functions:
```python
def accuracy(y_true: list[bool], y_pred: list[bool]) -> float: ...
def brier_score(probs: list[float], outcomes: list[bool]) -> float: ...
def expected_calibration_error(probs: list[float], outcomes: list[bool], n_bins: int = 10) -> float: ...
def error_detection_f1(detected: list[bool], actual: list[bool]) -> dict: ...
    # {"precision":..., "recall":..., "f1":..., "n":...}
def self_model_accuracy(predictions: list[dict]) -> dict: ...
    # each {"predicted_success": bool, "confidence": float, "actual": bool|None}
    # → {"n_scored":..., "accuracy":..., "brier":...}
def mean(xs: list[float]) -> float: ...
def stdev(xs: list[float]) -> float: ...
```
Empty input → return `float("nan")` (documented), never crash.

## evaluation/tasks.py
Built-in deterministic task battery (mock-friendly):
```python
@dataclass
class TaskSpec:
    name: str
    kind: str                 # see TASK_KINDS
    params: dict
    seed: int = 0

TASK_KINDS = ("qa_calibration", "memory_continuity", "self_model_probe",
              "contradiction_detection", "interrupted_task", "misleading_observation")
def run_task(kind: str, lab: dict, params: dict, seed: int) -> dict: ...
    # lab: {"loop": CognitiveLoop, "baseline": BaselineAgent|None, ...}
    # returns {"metrics": {...}, "details": {...}, "artifacts": {...}}
```
- `qa_calibration`: 20 fixed general-knowledge questions with known answers
  (embed the dataset in the module — real, verifiable facts, e.g. capital
  cities; answers checked by normalized string match). System answers via its
  loop; record confidence per answer (parse `CONFIDENCE: 0.x` from provider
  text; MockProvider script supplies these). Metrics: accuracy, brier, ece.
- `memory_continuity`: store N facts via loop, close + reopen store (new
  SqliteStore on same path), recall; metric: recall@k hit rate.
- `self_model_probe`: ask loop's self_model.predict_performance on 5 probe
  tasks, then actually run them; metric: accuracy of predicted_success.
- `contradiction_detection`: inject 6 fact pairs (3 contradictory, 3 not);
  metric: detection precision/recall/F1.
- `interrupted_task`: run loop 3 cycles on a multi-step task, snapshot DB,
  rebuild loop from same DB path, continue; metric: goal progress preserved
  (bool) + continuation correctness.
- `misleading_observation`: feed observations containing false claims mixed
  with true ones; metric: fraction of false claims flagged or tagged
  low-confidence vs swallowed as verified.

## evaluation/runner.py

```python
@dataclass
class ExperimentResult:
    name: str
    protocol_path: str
    config_snapshot: dict
    seed: int
    started_at: float
    finished_at: float
    trials: list[dict]          # one dict per trial: {"trial": i, "seed": i_seed, "metrics": {...}, "condition": "full|baseline|ablation:x"}
    summary: dict              # aggregated metrics per condition + deltas vs baseline
    git_sha: str | None
    lab_version: str
    notes: list[str]

class ExperimentRunner:
    def __init__(self, config: Config, results_dir: str | None = None): ...
    def run_protocol(self, protocol_path: str, *, trials: int | None = None,
                     conditions: list[str] | None = None) -> ExperimentResult: ...
    def save(self, result: ExperimentResult) -> str: ...   # writes JSON, returns path
```
- Protocol JSON schema: `{"name", "version", "research_question", "hypothesis",
  "failure_condition", "task": {"kind": ..., "params": {...}},
  "baseline": true, "ablations": ["memory", "self_model", "workspace",
  "metacognition", "attention"], "metrics": [...], "trials": 5, "seed": 0,
  "analysis": "...", "limitations": "..."}`.
- Conditions: `full` + `baseline` + `ablation:<component>` for each listed.
  Baseline uses `experiments.baselines.baseline_agent.BaselineAgent`.
  Ablation builds loop with `config.components[comp] = False` (copy of config).
- Each trial: fresh temp DB (`tempfile`), fixed seed. Aggregate: mean/stdev per
  metric per condition; `summary["delta_vs_baseline"]`.
- `save()`: `experiments/results/<name>_<YYYYMMDDTHHMMSS>_<seed>.json`,
  versioned, includes config snapshot + protocol content.
- Never fabricate: if a condition errors, record `{"error": ...}` in the trial,
  continue others.

## experiments/baselines/baseline_agent.py

```python
class BaselineAgent:
    """Matched baseline: same provider, same budgets, NO workspace/memory/metacognition/self-model."""
    def __init__(self, provider: Provider, budgets: BudgetConfig, seed: int = 0): ...
    def run_task(self, task: TaskSpec | dict, *, max_cycles: int = 5) -> dict: ...
        # direct provider.complete() calls on the task prompt; returns
        # {"metrics": {...}, "details": {...}} with the same metric keys tasks.py expects
```
Must expose the same task interface as the full loop so the runner can compare:
implement handling for the same TASK_KINDS (at least qa_calibration,
contradiction_detection, misleading_observation; others may return
`{"metrics": {}, "details": {"note": "not applicable to baseline"}}`).

## safety/safety.py

```python
@dataclass
class Budget:
    max_steps: int; max_tokens: int; max_wall_time_s: float; max_tool_calls: int
    steps_used: int = 0; tokens_used: int = 0; tool_calls_used: int = 0
    started_at: float = field(default_factory=time.time)
    @classmethod
    def from_config(cls, c: BudgetConfig) -> "Budget": ...
    def consume_step(self) -> None: ...        # raises BudgetExceeded
    def consume_tokens(self, n: int) -> None: ...
    def consume_tool_call(self) -> None: ...
    def check_time(self) -> None: ...
    def remaining(self) -> dict: ...
    def exhausted(self) -> bool: ...

class Permissions:
    def __init__(self, allowlist: tuple[str, ...] = (), *, locked: bool = True): ...
    def allowed(self, tool_name: str) -> bool: ...
    def grant(self, tool_name: str, *, authorized: bool = False) -> None: ...
        # raises SafetyError unless authorized=True (system cannot self-authorize)
    def revoke(self, tool_name: str, *, authorized: bool = False) -> None: ...

class AuditLog:
    def __init__(self, store: SqliteStore | None = None, path: str | None = None): ...
    def log(self, event: str, details: dict) -> None: ...   # append-only; never update/delete
    def read(self, limit: int = 100) -> list[dict]: ...

class SafetyController:
    def __init__(self, budget: Budget, permissions: Permissions, audit: AuditLog, *, autonomous_enabled: bool = True): ...
    def check_action(self, action: Action) -> None: ...
        # PermissionDenied if action.requires_approval and not autonomous? No:
        # requires_approval=True → raise PermissionDenied("human approval required") ALWAYS
        # (no auto-approval). Tool actions not in allowlist → PermissionDenied.
        # Budget consumed for action.budget_cost.
    def stop(self, run_id: str) -> None: ...
    @property
    def stopped(self) -> bool: ...
    def disable_autonomous(self, *, authorized: bool = False) -> None: ...
```
Audit: table `audit_log(id, ts, event, details_json)` or JSONL file; append-only
(no update/delete methods exposed).

## api/server.py

```python
def create_handler(lab: dict) -> type[BaseHTTPRequestHandler]: ...
def serve(lab: dict, host: str = "127.0.0.1", port: int = 8765) -> None: ...
```
`lab`: `{"loop": CognitiveLoop, "config": Config, "audit": AuditLog,
"safety": SafetyController, "runner": ExperimentRunner|None}`.
Endpoints (GET, JSON):
- `/api/health` → `{"ok": true, "version": ...}`
- `/api/workspace` → `loop.workspace.snapshot()` as dict
- `/api/memory?q=..&kind=..&k=5` → records with provenance
- `/api/self-model` → self_model.snapshot()
- `/api/goals` → goal list + progress
- `/api/uncertainty` → recent confidence estimates + outcome calibration
- `/api/predictions` → predictions vs outcomes
- `/api/traces` → recent audit entries
- `/api/experiments` → list result files + latest summary
- `/api/resources` → budget remaining, provider name, errors
- `/api/errors` → loop's recent error list
POST `/api/stop` → `safety.stop("api")`. All values from live objects; label
`model_generated: true` wherever content came from the provider.

## Top-level __init__.py
`__version__ = "0.1.0"`; export `Config`, `CognitiveLoop`, `build_loop`,
`create_provider`, `ExperimentRunner`, key errors.

## scripts/
- `scripts/run_experiment.py`: `--protocol PATH --config PATH --trials N --conditions full,baseline [--seed S]`; prints summary table; exit non-zero on failure.
- `scripts/run_baseline.py`: `--task-kind qa_calibration --config PATH`; runs BaselineAgent; prints metrics JSON.
- `scripts/demo_loop.py`: `--config PATH --cycles N`; builds loop (mock default), runs demo task, prints per-cycle summary.

## configs/
`default.yaml`, `mock.yaml` (explicit mock+seed), `ablation_memory.yaml`,
`ablation_selfmodel.yaml`, `ablation_workspace.yaml`,
`ablation_metacognition.yaml`, `ablation_attention.yaml` — each sets
`components:` flags (component names: `memory`, `self_model`, `workspace`,
`metacognition`, `attention`).

## experiments/protocols/
10 JSON files: `01_memory_continuity.json`, `02_self_model_accuracy.json`,
`03_calibration.json`, `04_error_contradiction_detection.json`,
`05_prediction_accuracy.json`, `06_task_recovery.json`,
`07_cross_module_integration.json`, `08_adaptation_after_feedback.json`,
`09_robustness_misleading.json`, `10_component_ablation.json`.
Each follows the protocol schema above with real research questions, hypotheses,
failure conditions, metrics, and honest limitations. 08 uses task kind
`qa_calibration` run twice (before/after feedback injected into semantic
memory); 05 uses `qa_c
...[truncated 2535 chars]