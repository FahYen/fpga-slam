"""ctypes binding for librnsg_ipc.so.

Numpy is required. Producer slot views and consumer frame views are both
exposed as numpy arrays that alias the underlying shared memory directly;
no bytes are copied by this layer.

Discipline:
    * Producer: between ``Producer.lease()`` and ``Producer.publish(...)``,
      the returned ``SlotView.points`` / ``.labels`` arrays are valid for
      writing. After publish, the producer must not touch those arrays.
    * Consumer: between ``Consumer.acquire()`` and ``Consumer.release()``,
      the returned ``Frame.points`` / ``.labels`` arrays are valid for
      reading. After release, they become dangling and must not be touched.
      Use ``with consumer.frame(...) as frame:`` to enforce release.
"""

from __future__ import annotations

import contextlib
import ctypes as C
import ctypes.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

DEFAULT_SLOT_COUNT = 8
DEFAULT_CAPACITY_POINTS = 200000
FLAG_RAW_SEMANTICKITTI_LABELS = 0x1

_RNSG_OK = 0
_RNSG_TIMEOUT = 1
_RNSG_E_BUSY = -9


# ---------------------------------------------------------------- library load


def _load_library() -> C.CDLL:
    explicit = os.environ.get("RNSG_IPC_LIB")
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    here = Path(__file__).resolve().parent
    candidates.extend(
        [
            here / "librnsg_ipc.so",
            here.parent / "librnsg_ipc.so",
            here.parent.parent / "build" / "librnsg_ipc.so",
        ]
    )
    for path in candidates:
        if path.is_file():
            return C.CDLL(str(path))
    found = ctypes.util.find_library("rnsg_ipc")
    if found:
        return C.CDLL(found)
    raise RuntimeError(
        "librnsg_ipc.so not found. Set RNSG_IPC_LIB to its path or build the "
        "library under rangenet_sgslam_ipc/build/."
    )


_lib = _load_library()


# ---------------------------------------------------------------- C structs


class _SlotView(C.Structure):
    _fields_ = [
        ("slot_idx", C.c_uint32),
        ("capacity_points", C.c_uint32),
        ("header", C.c_void_p),
        ("points", C.POINTER(C.c_float)),
        ("labels", C.POINTER(C.c_int32)),
    ]


class _FrameView(C.Structure):
    _fields_ = [
        ("slot_idx", C.c_uint32),
        ("num_points", C.c_uint32),
        ("frame_id", C.c_uint64),
        ("consumed_index", C.c_uint64),
        ("capture_ns", C.c_uint64),
        ("publish_ns", C.c_uint64),
        ("flags", C.c_uint32),
        ("reserved0", C.c_uint32),
        ("skipped_before", C.c_uint64),
        ("points", C.POINTER(C.c_float)),
        ("labels", C.POINTER(C.c_int32)),
    ]


# ---------------------------------------------------------------- bind


def _bind(name, restype, argtypes):
    fn = getattr(_lib, name)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


_rnsg_create = _bind(
    "rnsg_create",
    C.c_int,
    [C.c_char_p, C.c_uint32, C.c_uint32, C.POINTER(C.c_void_p)],
)
_rnsg_open = _bind("rnsg_open", C.c_int, [C.c_char_p, C.POINTER(C.c_void_p)])
_rnsg_close = _bind("rnsg_close", None, [C.c_void_p])
_rnsg_unlink = _bind("rnsg_unlink", C.c_int, [C.c_char_p])
_rnsg_slot_count = _bind("rnsg_slot_count", C.c_uint32, [C.c_void_p])
_rnsg_capacity_points = _bind("rnsg_capacity_points", C.c_uint32, [C.c_void_p])
_rnsg_drop_count = _bind("rnsg_drop_count", C.c_uint64, [C.c_void_p])
_rnsg_head_seq = _bind("rnsg_head_seq", C.c_uint64, [C.c_void_p])
_rnsg_tail_seq = _bind("rnsg_tail_seq", C.c_uint64, [C.c_void_p])
_rnsg_producer_lease = _bind(
    "rnsg_producer_lease", C.c_int, [C.c_void_p, C.POINTER(_SlotView)]
)
_rnsg_producer_publish = _bind(
    "rnsg_producer_publish",
    C.c_int,
    [C.c_void_p, C.c_uint32, C.c_uint64, C.c_uint64, C.c_uint32],
)
_rnsg_consumer_acquire = _bind(
    "rnsg_consumer_acquire",
    C.c_int,
    [C.c_void_p, C.c_int64, C.POINTER(_FrameView)],
)
_rnsg_consumer_release = _bind("rnsg_consumer_release", C.c_int, [C.c_void_p])


# ---------------------------------------------------------------- exceptions


class RingError(RuntimeError):
    """Errors from the underlying C library."""


class RingTimeout(TimeoutError):
    """Returned by ``Consumer.acquire()`` when the wait deadline expires."""


class RingBusy(RuntimeError):
    """Acquire while already holding, or release while not holding."""


def _check(rc: int, what: str) -> None:
    if rc < 0:
        if rc == _RNSG_E_BUSY:
            raise RingBusy(f"{what} returned BUSY")
        raise RingError(f"{what} failed (rc={rc}, errno={C.get_errno()})")


# ---------------------------------------------------------------- helpers


def _array_from_ptr(ptr, count: int, ctype) -> np.ndarray:
    """Numpy array aliasing a raw C pointer. No copy."""
    if count == 0:
        return np.empty(0, dtype=np.dtype(ctype).newbyteorder("="))
    addr = C.cast(ptr, C.c_void_p).value
    if addr is None:
        return np.empty(0, dtype=np.dtype(ctype).newbyteorder("="))
    buf_type = ctype * count
    buf = buf_type.from_address(addr)
    return np.frombuffer(buf, dtype=np.dtype(ctype).newbyteorder("="))


# ---------------------------------------------------------------- public types


@dataclass
class SlotView:
    """Producer-side zero-copy view over the leased slot.

    Both arrays are sized at the ring's configured capacity. The producer
    fills only the first ``num_points`` rows before calling ``publish``.
    """

    slot_idx: int
    capacity_points: int
    points: np.ndarray  # (capacity, 4) float32, columns: x, y, z, remission
    labels: np.ndarray  # (capacity,) int32, raw SemanticKITTI ids per contract


@dataclass
class Frame:
    """Consumer-side zero-copy view of a held frame.

    ``points`` and ``labels`` are read-only numpy aliases into shared memory,
    sized exactly ``num_points``. They are valid only until the matching
    ``Consumer.release()``.

    Index semantics:
        - ``frame_id``       : SPARSE producer-assigned id; jumps by
                               ``skipped_before + 1`` under drop-oldest.
                               Use only as a back-reference to the producer
                               source (e.g. KITTI scan number).
        - ``consumed_index`` : DENSE per-consumer counter (0, 1, 2, ...).
                               Increments by exactly 1 on every successful
                               acquire, even when frames were dropped. Use
                               for structural indexing (pose vectors, graph
                               keys, per-frame artifact filenames).
        - ``skipped_before`` : count of producer frames overwritten between
                               the previous successful acquire and this one.
    """

    slot_idx: int
    frame_id: int
    consumed_index: int
    capture_ns: int
    publish_ns: int
    flags: int
    skipped_before: int
    points: np.ndarray  # (num_points, 4) float32, read-only
    labels: np.ndarray  # (num_points,) int32, read-only

    @property
    def num_points(self) -> int:
        return int(self.labels.shape[0])


# ---------------------------------------------------------------- API


def unlink(name: str) -> None:
    """Remove the named POSIX shm region. No-op if it does not exist."""
    rc = _rnsg_unlink(name.encode("utf-8"))
    if rc == 0 or rc == -2:
        return
    raise RingError(f"unlink({name}) failed (rc={rc})")


class _Ring:
    def __init__(self, handle: C.c_void_p, name: str):
        self._handle = handle
        self._name = name

    def close(self) -> None:
        if self._handle:
            _rnsg_close(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def slot_count(self) -> int:
        return int(_rnsg_slot_count(self._handle))

    @property
    def capacity_points(self) -> int:
        return int(_rnsg_capacity_points(self._handle))

    @property
    def drop_count(self) -> int:
        return int(_rnsg_drop_count(self._handle))

    @property
    def head_seq(self) -> int:
        return int(_rnsg_head_seq(self._handle))

    @property
    def tail_seq(self) -> int:
        return int(_rnsg_tail_seq(self._handle))


class Producer(_Ring):
    """RangeNet-side writer."""

    @classmethod
    def create(
        cls,
        name: str,
        slot_count: int = DEFAULT_SLOT_COUNT,
        capacity_points: int = DEFAULT_CAPACITY_POINTS,
    ) -> "Producer":
        handle = C.c_void_p()
        rc = _rnsg_create(
            name.encode("utf-8"),
            C.c_uint32(slot_count),
            C.c_uint32(capacity_points),
            C.byref(handle),
        )
        _check(rc, f"rnsg_create({name})")
        return cls(handle, name)

    @classmethod
    def open(cls, name: str) -> "Producer":
        handle = C.c_void_p()
        rc = _rnsg_open(name.encode("utf-8"), C.byref(handle))
        _check(rc, f"rnsg_open({name})")
        return cls(handle, name)

    def lease(self) -> SlotView:
        """Return a zero-copy view on the next slot to write."""
        view = _SlotView()
        rc = _rnsg_producer_lease(self._handle, C.byref(view))
        _check(rc, "rnsg_producer_lease")
        cap = int(view.capacity_points)
        points = _array_from_ptr(view.points, cap * 4, C.c_float).reshape(cap, 4)
        labels = _array_from_ptr(view.labels, cap, C.c_int32)
        return SlotView(
            slot_idx=int(view.slot_idx),
            capacity_points=cap,
            points=points,
            labels=labels,
        )

    def publish(
        self,
        num_points: int,
        capture_ns: int = 0,
        frame_id: Optional[int] = None,
        flags: int = FLAG_RAW_SEMANTICKITTI_LABELS,
    ) -> None:
        if frame_id is None:
            frame_id = (1 << 64) - 1  # sentinel: library assigns
        rc = _rnsg_producer_publish(
            self._handle,
            C.c_uint32(num_points),
            C.c_uint64(capture_ns),
            C.c_uint64(frame_id),
            C.c_uint32(flags),
        )
        _check(rc, "rnsg_producer_publish")


class Consumer(_Ring):
    """SG-SLAM-side reader. Zero-copy acquire/release lifetime."""

    @classmethod
    def open(cls, name: str) -> "Consumer":
        handle = C.c_void_p()
        rc = _rnsg_open(name.encode("utf-8"), C.byref(handle))
        _check(rc, f"rnsg_open({name})")
        return cls(handle, name)

    def acquire(self, timeout_s: Optional[float] = None) -> Frame:
        """Block until the next frame is available; return a zero-copy view.

        Caller must ``release()`` (or use the ``frame()`` context manager)
        before calling ``acquire`` again.

        ``timeout_s``: ``None`` blocks forever, ``0`` is non-blocking,
        ``>0`` is a bounded wait. Raises ``RingTimeout`` on timeout.
        """
        if timeout_s is None:
            timeout_ns = -1
        elif timeout_s <= 0:
            timeout_ns = 0
        else:
            timeout_ns = int(timeout_s * 1e9)

        view = _FrameView()
        rc = _rnsg_consumer_acquire(
            self._handle, C.c_int64(timeout_ns), C.byref(view)
        )
        if rc == _RNSG_TIMEOUT:
            raise RingTimeout(f"acquire timed out after {timeout_s}s")
        _check(rc, "rnsg_consumer_acquire")

        n = int(view.num_points)
        points = _array_from_ptr(view.points, n * 4, C.c_float).reshape(n, 4)
        labels = _array_from_ptr(view.labels, n, C.c_int32)
        # Mark read-only at the numpy layer so accidental writes raise.
        points.setflags(write=False)
        labels.setflags(write=False)

        return Frame(
            slot_idx=int(view.slot_idx),
            frame_id=int(view.frame_id),
            consumed_index=int(view.consumed_index),
            capture_ns=int(view.capture_ns),
            publish_ns=int(view.publish_ns),
            flags=int(view.flags),
            skipped_before=int(view.skipped_before),
            points=points,
            labels=labels,
        )

    def release(self) -> None:
        rc = _rnsg_consumer_release(self._handle)
        _check(rc, "rnsg_consumer_release")

    @contextlib.contextmanager
    def frame(self, timeout_s: Optional[float] = None) -> Iterator[Frame]:
        """Acquire-then-auto-release context manager.

        Usage:
            with consumer.frame(timeout_s=1.0) as frame:
                process(frame.points, frame.labels)
        """
        f = self.acquire(timeout_s=timeout_s)
        try:
            yield f
        finally:
            self.release()
