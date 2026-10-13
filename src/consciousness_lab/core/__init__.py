"""Core engine: configuration, errors, logging, and the cognitive loop."""

from .config import BudgetConfig, Config
from .errors import (
    BudgetExceeded,
    ConfigError,
    ContradictionError,
    ExperimentError,
    LabError,
    PermissionDenied,
    ProviderError,
    SafetyError,
)
from .logging_setup import setup_logging
from .loop import CognitiveLoop, CycleResult, TaskResult, build_loop

__all__ = [
    "BudgetConfig",
    "Config",
    "LabError",
    "ConfigError",
    "ProviderError",
    "BudgetExceeded",
    "PermissionDenied",
    "ContradictionError",
    "ExperimentError",
    "SafetyError",
    "setup_logging",
    "CognitiveLoop",
    "CycleResult",
    "TaskResult",
    "build_loop",
]
