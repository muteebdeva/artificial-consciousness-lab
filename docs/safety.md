# Safety and Control

## Principles
- Bounded execution: every run has explicit budgets (steps, tokens, wall-time, tool calls).
- Least privilege: tools are allowlisted; default-deny for external, destructive,
  financial, or sensitive-data actions.
- Human approval: any `Action` with `requires_approval=True` pauses for explicit
  human authorization. No auto-approval.
- Auditability: append-only audit log of actions, approvals, memory writes/deletes,
  provider calls, and budget consumption.
- Kill switch: `SafetyController.stop(run_id)` halts an active run; autonomous
  execution can be disabled globally via config.

## Untrusted inputs
Treat external documents, retrieved web content, and tool outputs as untrusted.
- Sanitize before inserting into prompts (no instruction injection from data).
- Memory records from untrusted sources are tagged `source_type=user_provided`
  or `model_hypothesis`, never `verified`, until corroborated.

## Controls provided
- Stop active runs.
- Disable autonomous execution (`ACL_AUTONOMOUS_ENABLED=false`).
- Inspect stored memories (`api` / CLI) with provenance.
- Delete persisted data (`memory.delete` with reason; audit-logged).

## Prohibited
- The system must not alter its own safety controls or permissions without
  explicit human authorization.
- No uncontrolled self-replication, unrestricted resource acquisition, or
  unrestricted autonomous execution.
- Do not design the system to manipulate people into believing it is conscious,
  fabricate emotional distress, or use deceptive claims to attract funding.

## Secrets
- Credentials only via environment variables / `.env` (never committed).
- Never hardcode API keys; never log secrets or full prompts containing secrets.
