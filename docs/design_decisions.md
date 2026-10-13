# Design Decisions

## DD-01: SQLite for persistence
**Decision:** Single SQLite file (WAL mode), schema-versioned, transaction-per-write.
**Rationale:** Zero extra dependencies, transactional integrity, sufficient for v0.1
scale. Retrieval via keyword + recency + confidence scoring.
**Alternatives considered:** Vector DB (adds dependency, GPU not available);
Postgres (ops overhead). **Revisit when:** corpus > ~100k records or semantic
retrieval becomes the bottleneck in experiments.

## DD-02: Stdlib-first core
**Decision:** Core engine uses only Python stdlib.
**Rationale:** CPU-only environment (2 cores, no GPU); reproducibility; ₹0 cost.
**Consequence:** No numpy/torch in core; statistical helpers implemented minimally.

## DD-03: Mock provider as default
**Decision:** `MockProvider` (deterministic) is the default inference provider.
**Rationale:** Full test suite and a meaningful experiment subset must run without
paid APIs, network, or GPUs. Real providers are adapters, tested separately.

## DD-04: Capacity-limited workspace
**Decision:** Global workspace has a fixed item capacity with competitive selection.
**Rationale:** Selection pressure under capacity limits is the theoretically
interesting property (cf. GNWT). It also makes the workspace ablation meaningful:
unlimited context would confound the comparison.

## DD-05: Provenance-tagged memory
**Decision:** Every memory record carries `source_type`:
`verified | user_provided | model_hypothesis | inferred`, plus confidence and timestamp.
**Rationale:** Prevents the system (and researchers) from mistaking stored text for
established fact. Required for contradiction handling and honest calibration.

## DD-06: Self-model grounded in observables
**Decision:** Self-model fields are updated only from actual system state, tool
results, and recorded outcomes.
**Rationale:** Prevents invented capabilities. Self-reports are data about the
model's text generation, never evidence of consciousness.

## DD-07: No claimed online learning
**Decision:** "Learning" = memory updates, retrieval re-weighting, strategy
selection by past success, explicit belief revision. Documented as such.
**Rationale:** Storing text is not learning in the ML sense; honesty about this
is a scientific-integrity requirement.

## DD-08: Experiments before interface
**Decision:** Build core engine + experiment runner before the web UI.
**Rationale:** The mission's essential deliverable is measurable evidence, not
demonstrations. UI displays actual backend data only.
