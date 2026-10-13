"""Evaluation package: task battery, metrics, and experiment runner."""

from .metrics import (
    accuracy,
    brier_score,
    error_detection_f1,
    expected_calibration_error,
    mean,
    self_model_accuracy,
    stdev,
)
from .runner import ExperimentResult, ExperimentRunner
from .tasks import TASK_KINDS, TaskSpec, run_task

__all__ = [
    "accuracy",
    "brier_score",
    "error_detection_f1",
    "expected_calibration_error",
    "mean",
    "self_model_accuracy",
    "stdev",
    "ExperimentResult",
    "ExperimentRunner",
    "TASK_KINDS",
    "TaskSpec",
    "run_task",
]
