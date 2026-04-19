"""Thin Python wrapper around librnsg_ipc.so.

Two classes are exposed:

- ``Producer`` (lease/publish) for the RangeNet inference loop. ``lease()``
  returns numpy zero-copy views into the slot's points/labels arrays so the
  caller can write directly to shared memory without an intermediate buffer.
- ``Consumer`` (next/close) for completeness; the SG-SLAM consumer normally
  links the C library directly.

The shared library is located via the ``RNSG_IPC_LIB`` environment variable,
or by searching alongside this package.
"""

from __future__ import annotations

from .core import (
    Consumer,
    Frame,
    Producer,
    RingBusy,
    RingError,
    RingTimeout,
    SlotView,
    DEFAULT_CAPACITY_POINTS,
    DEFAULT_SLOT_COUNT,
    FLAG_RAW_SEMANTICKITTI_LABELS,
    unlink,
)

__all__ = [
    "Consumer",
    "Frame",
    "Producer",
    "RingBusy",
    "RingError",
    "RingTimeout",
    "SlotView",
    "DEFAULT_CAPACITY_POINTS",
    "DEFAULT_SLOT_COUNT",
    "FLAG_RAW_SEMANTICKITTI_LABELS",
    "unlink",
]
