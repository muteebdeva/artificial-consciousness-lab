"""Typed exception hierarchy for the artificial-consciousness-lab platform.

All recoverable failures in the lab surface as one of these exceptions so that
callers (the cognitive loop, the experiment runner, the API) can distinguish
configuration problems, provider failures, budget exhaustion, permission
denials, and safety violations without parsing message strings.

Limits:
    - These carry no structured payload beyond ``ProviderError.retryable``;
      richer diagnostics belong in the audit log, not in exception objects.
    - They must never be used to silently swallow errors: every ``except``
      site in the loop records the error in the cycle's error list.
"""

from __future__ import annotations


class LabError(Exception):
    """Base class for all lab errors.

    Catch this only when you genuinely want to handle every lab failure the
    same way (e.g. recording it in a cycle's error list). Do not catch it
    merely to suppress it.
    """


class ConfigError(LabError):
    """Raised for invalid configuration: bad YAML, unknown provider name,
    missing required settings, or bad environment-variable values."""


class ProviderError(LabError):
    """Raised when an inference provider call fails.

    The ``retryable`` flag tells the retry helper whether another attempt is
    likely to succeed (rate limits, 5xx) or pointless (auth failures,
    malformed responses, unreachable hosts that already failed fast).
    """

    def __init__(self, msg: str, *, retryable: bool = False) -> None:
        super().__init__(msg)
        self.retryable = retryable


class BudgetExceeded(LabError):
    """Raised when a run exceeds its step, token, tool-call, or wall-time budget."""


class PermissionDenied(LabError):
    """Raised when an action requires a permission that was not granted."""


class ContradictionError(LabError):
    """Raised when a belief/memory update cannot proceed due to an
    unresolvable contradiction. (Detection itself does not raise; it returns
    ``Contradiction`` records. This is for hard conflicts in update paths.)"""


class ExperimentError(LabError):
    """Raised for experiment-runner failures: bad protocol files, unknown
    task kinds, or result-saving failures."""


class SafetyError(LabError):
    """Raised for safety-control violations, e.g. attempting to grant a
    permission without explicit human authorization."""
