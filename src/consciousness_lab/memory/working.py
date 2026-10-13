"""Working memory: in-memory, capacity-limited, task-scoped buffer.

NOT persisted: contents live only in this process and are lost on restart.
When over capacity, the oldest item is evicted first (FIFO). Accepts
``WorkspaceItem``, ``MemoryRecord``, or plain ``str`` (a string is wrapped
in a ``MemoryRecord`` of kind "working" with provenance "working_memory").

Limits: no retrieval ranking is applied here — ``contents()`` returns items
in insertion order. Any prioritization happens in the workspace/attention
modules, not in this buffer.
"""

from __future__ import annotations

import time
import uuid

from consciousness_lab.memory.records import MemoryRecord
from consciousness_lab.workspace.workspace import WorkspaceItem


class WorkingMemory:
    """In-memory scratch buffer with FIFO eviction at capacity."""

    def __init__(self, capacity: int = 10) -> None:
        """Create an empty buffer holding at most ``capacity`` items.

        Raises ValueError if capacity < 1.
        """
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity!r}")
        self._capacity = capacity
        self._items: list = []

    @property
    def capacity(self) -> int:
        """Maximum number of items held."""
        return self._capacity

    def put(self, item: WorkspaceItem | MemoryRecord | str) -> None:
        """Append ``item``; evict oldest items while over capacity.

        Plain strings are wrapped as ``MemoryRecord(kind="working",
        provenance="working_memory", source_type="inferred")``. Raises
        ValueError for any other input type.
        """
        if isinstance(item, str):
            item = MemoryRecord(
                id=uuid.uuid4().hex,
                kind="working",
                content=item,
                provenance="working_memory",
                confidence=0.5,
                timestamp=time.time(),
                source_type="inferred",
            )
        elif not isinstance(item, (WorkspaceItem, MemoryRecord)):
            raise ValueError(
                "item must be WorkspaceItem, MemoryRecord, or str; "
                f"got {type(item).__name__}"
            )
        self._items.append(item)
        while len(self._items) > self._capacity:
            self._items.pop(0)

    def contents(self) -> list:
        """Return the current items in insertion order (a copy)."""
        return list(self._items)

    def clear(self) -> None:
        """Drop all items."""
        self._items = []
