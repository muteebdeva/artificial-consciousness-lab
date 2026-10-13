"""artificial-consciousness-lab: a research platform for AI consciousness studies.

A serious, honest research instrument — not a chatbot wrapper. It investigates
whether computational architectures (global workspace, persistent memory with
provenance, metacognition, grounded self-modeling) support measurable
functional properties associated with consciousness. It does not assume
consciousness has been achieved, and AI self-reports are never treated as
evidence of subjective experience.

Top-level exports: :class:`Config`, :class:`CognitiveLoop`,
:func:`build_loop`, :func:`create_provider`, :class:`ExperimentRunner`,
and the key error types. ``ExperimentRunner`` is imported lazily so this
package stays importable before the evaluation module is implemented.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .core.config import BudgetConfig, Config
from .core.errors import (
    BudgetExceeded,
    ConfigError,
    ContradictionError,
    ExperimentError,
    LabError,
    PermissionDenied,
    ProviderError,
    SafetyError,
)
from .core.loop import CognitiveLoop, CycleResult, TaskResult, build_loop
from .models import create_provider

__all__ = [
    "__version__",
    "Config",
    "BudgetConfig",
    "CognitiveLoop",
    "CycleResult",
    "TaskResult",
    "build_loop",
    "create_provider",
    "ExperimentRunner",
    "LabError",
    "ConfigError",
    "ProviderError",
    "BudgetExceeded",
    "PermissionDenied",
    "ContradictionError",
    "ExperimentError",
    "SafetyError",
]


def __getattr__(name: str):
    # Deferred import: evaluation.runner imports only core, so this cannot
    # cycle — but the module may not exist yet while other agents implement it.
    if name == "ExperimentRunner":
        from .evaluation.runner import ExperimentRunner

        return ExperimentRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
