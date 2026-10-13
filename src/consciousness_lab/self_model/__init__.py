"""Self-model package: the system's source-attributed model of its own capabilities.

This is a bookkeeping structure, NOT a measure of consciousness or
self-awareness. Per DD-06, every entry records where it came from, and
capabilities are never invented — they are registered only via
``register_tool`` / ``register_capability`` calls made by the loop factory
with real tools/providers.
"""

from consciousness_lab.self_model.self_model import PerformancePrediction, SelfModel

__all__ = ["PerformancePrediction", "SelfModel"]
