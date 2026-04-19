# rangenet_sgslam_ipc

Shared-memory FIFO of complete labeled LiDAR frames between the RangeNet
producer and the SG-SLAM consumer. Replaces the per-scan `.label` file
handoff with an in-memory ring on POSIX shared memory.

This component is intentionally isolated from RangeNet's training tree and
from SG-SLAM's ROS layer so it can build and unit-test on its own, with no
ROS, no CUDA, and no Vitis dependencies.

See `Docs/Design/contract.md` for the wire-level contract this implements.

## Variant

- **SPSC** (single producer, single consumer).
- **Drop-oldest**: the producer never blocks. When all unheld slots are
  occupied, the next publish overwrites the oldest unheld slot and the
  consumer sees `skipped_before > 0` on its next acquire.
- **Zero-copy reads** with **acquire/release lifetime**: the consumer
  receives const pointers directly into shared memory and reads in place.
  While a slot is held, the producer is forbidden from overwriting it; the
  producer's cursor walks past held slots when picking a victim.
- **Frame atomicity**: each FIFO element is a complete `(scan, labels)`
  pair. The consumer never observes a partial frame.

## Layout

- `include/rnsg_ipc.h` - single C ABI header. Both Python (`ctypes`) and C++
  link against this. No C++-only or Python-only types in the wire format.
- `src/rnsg_ring.c` - POSIX shm implementation of the control region + slot
  arena and the SPSC drop-oldest acquire/release algorithm.
- `python/rnsg_ipc/` - thin ctypes wrapper. Exposes `Producer.lease()` /
  `Producer.publish()` and `Consumer.acquire()` / `Consumer.release()` /
  `Consumer.frame()` (context manager) with numpy zero-copy views over
  slot memory.
- `tests/` - C round-trip, drop-oldest overrun, held-slot invariant under
  overrun, and a two-process Python round-trip.

## Per-slot atomic encoding

Each slot has a single `_Atomic uint64_t slot_seq[i]` in shm:

| Value                       | Meaning                                                  |
|-----------------------------|----------------------------------------------------------|
| `0`                         | empty / mid-write / freshly released                     |
| `seq + 1` (low 63 bits)     | published with sequence `seq`, available for acquire     |
| `HELD_BIT \| (seq + 1)`      | currently held by consumer; producer must not overwrite  |

The producer claims a slot via `CAS(cur -> 0)`; the consumer claims via
`CAS(seq+1 -> seq+1 | HELD_BIT)`. CAS contention is bounded to one retry
under SPSC.

## Build

```bash
cd rangenet_sgslam_ipc
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

This produces `build/librnsg_ipc.so` and the test binaries. The Python tests
import `python/rnsg_ipc` and locate the shared library via
`RNSG_IPC_LIB=/path/to/librnsg_ipc.so` (the test runner sets this for you).

## Usage sketch

Producer (Python, RangeNet):

```python
from rnsg_ipc import Producer, FLAG_RAW_SEMANTICKITTI_LABELS

prod = Producer.create("/rnsg_main", slot_count=4, capacity_points=200_000)
view = prod.lease()
view.points[:n] = scan_xyzr        # zero-copy write into shm
view.labels[:n] = raw_label_ids
prod.publish(num_points=n, capture_ns=ts, frame_id=fid,
             flags=FLAG_RAW_SEMANTICKITTI_LABELS)
```

Consumer (Python, for tests):

```python
from rnsg_ipc import Consumer

cons = Consumer.open("/rnsg_main")
with cons.frame(timeout_s=1.0) as frame:
    # frame.points / frame.labels are read-only numpy views aliasing shm
    process(frame.points, frame.labels)
# slot is automatically released on __exit__
```

Consumer (C++, SG-SLAM):

```cpp
rnsg_frame_view fv;
if (rnsg_consumer_acquire(ring, /*timeout_ns=*/-1, &fv) == RNSG_OK) {
    // fv.points / fv.labels point directly into shm; build Eigen vectors etc.
    process(fv.points, fv.labels, fv.num_points);
    rnsg_consumer_release(ring);
}
```

## Defaults

- `slot_count = 8` (effective working depth `slot_count - 1 = 7` while a
  slot is held)
- `capacity_points = 200000` (covers HDL-64 + OS1-128 worst case with margin)

Each slot is `sizeof(header) + capacity_points * (16 + 4)` bytes ~ 4 MB.
Total ring footprint ~ `slot_count * slot_bytes` ~ 32 MB.

Sizing rule (see `Docs/Design/contract.md` for the derivation):

```
slot_count >= ceil(consumer_latency * producer_rate) + 1
```

The default of 8 was chosen to absorb the p99 SG-SLAM front-end latency
(~400 ms measured on the AWS dev box) at a 10 Hz scan rate without
dropping frames the consumer hasn't yet seen. Bump `slot_count` higher if
your consumer's tail latency exceeds ~700 ms or your producer rate exceeds
10 Hz; lower if you can tolerate occasional `skipped_before > 0` events on
slow frames.
