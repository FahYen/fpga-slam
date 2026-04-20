# Shared-Memory RangeNet -> SG-SLAM Contract

This replaces the temporary file/basename `.label` handoff for the live GPU
path. The interface is a shared-memory FIFO carrying complete labeled LiDAR
frames from `RangeNet` to `SG-SLAM`, implemented under
`rangenet_sgslam_ipc/`.

## Variant

- **SPSC**: one `RangeNet` producer, one `SG-SLAM` consumer.
- **Drop-oldest**: producer never blocks. It first reuses any free
  unreadable slot (`slot_seq == 0`); only when all unheld slots still
  contain unread data does the next publish overwrite the oldest unheld
  slot. The consumer observes this on the next acquire as
  `skipped_before > 0` and the global `drop_count` atomic is incremented
  by exactly the number of lost frames.
- **Zero-copy reads with acquire/release lifetime**: the consumer receives
  const pointers directly into shared memory and reads in place. While a
  slot is held, the producer is forbidden from overwriting it; the
  producer's cursor walks past held slots when picking a victim.
- **Frame atomicity**: each FIFO element is a complete `(scan, labels)`
  pair. The consumer never observes a partial frame.

## Frame Payload

Each FIFO element contains:

- `num_points`
- point cloud in original point order, `float[num_points][4]` (x, y, z, remission)
- one `int32` semantic label per point, same order
- `frame_id`, `capture_ns` (sensor timestamp), `publish_ns` (CLOCK_MONOTONIC)

Label rules:

- raw `SemanticKITTI` IDs (no pre-remap to `SG-SLAM` reduced IDs)
- `SG-SLAM` keeps ownership of label remap and filtering
- the `flags` field carries `RNSG_FLAG_RAW_SEMANTICKITTI_LABELS` so the
  consumer can detect a producer that ever ships pre-remapped IDs

### Frame indexing: sparse `frame_id` vs dense `consumed_index`

The consumer-side `rnsg_frame_view` exposes two distinct counters per
acquired frame; mixing them up corrupts SG-SLAM's pose graph silently.

| Field            | Source                  | Density under drops      | Use for                                                      |
|------------------|-------------------------|--------------------------|--------------------------------------------------------------|
| `frame_id`       | producer-assigned       | SPARSE (gaps on drops)   | back-reference to producer source (KITTI scan id, log line)  |
| `consumed_index` | consumer-side counter   | DENSE (0, 1, 2, ...)     | structural indexing: pose vectors, GTSAM keys, file outputs  |
| `skipped_before` | per-acquire delta       | n/a                      | adapting motion priors, gating thresholds, drop diagnostics  |

`frame_id` is whatever the producer stamped at publish time. Under
drop-oldest the consumer sees `frame_id` jump by `(skipped_before + 1)`
on every successful acquire that follows a drop. It is therefore unsafe
to use `frame_id` as an index into any vector grown one-per-acquire on
the consumer side.

`consumed_index` is the IPC layer's per-consumer dense counter. It starts
at 0 on the first successful acquire and increments by exactly 1 on
every successful acquire, regardless of how many producer frames were
overwritten between acquires. This is the index SG-SLAM must use for:

- `poses_vec_[consumed_index]` and the `poses_vec_[consumed_index - 1]`
  previous-pose lookup feeding the GTSAM `BetweenFactor`.
- GTSAM keys: `BetweenFactor<Pose3>(consumed_index - 1, consumed_index, ...)`
  and `initial.insert(consumed_index, ...)`. Using `frame_id` here would
  create orphaned keys (e.g. keys 1..7 missing after a 7-frame drop)
  and the BetweenFactor would connect existing keys across the gap with
  the wrong relative pose.
- Boolean per-frame vectors such as `is_keyframe_vec[consumed_index]`.
- Per-frame artifact filenames (trajectory dumps, scan snapshots) where
  the consumer wants a contiguous index regardless of producer-side gaps.

If you need both — a dense GTSAM key and the producer's `frame_id` for
traceability back to the source scan file — store both in the
back-end's per-frame struct, e.g. `{cloud_id = consumed_index,
source_frame_id = frame_id, ...}`. The current `SemGraphSLAM` back-end
already routes a `cloud_id` value through its queue; that `cloud_id` is
exactly the field that should be set to `consumed_index`, not
`frame_id`.

## Wire layout

One POSIX shm region per ring. Layout:

```
[ struct rnsg_ctl              ]   fixed-size control block
[ slot_seq[slot_count]         ]   per-slot atomic state (see below)
[ pad to page                  ]
[ slot 0: header + points + labels ]
[ slot 1: ... ]
...
```

`rnsg_slot_header` is fixed-layout with explicit `points_offset` and
`labels_offset` so future versions can extend the payload without breaking
older readers (`magic` + `version` are checked on every acquire).

## Per-slot atomic encoding

Each slot has one `_Atomic uint64_t slot_seq[i]`:

| Value                       | Meaning                                                  |
|-----------------------------|----------------------------------------------------------|
| `0`                         | empty / mid-write / freshly released                     |
| `seq + 1` (low 63 bits)     | published with sequence `seq`, available for acquire     |
| `HELD_BIT | (seq + 1)`      | currently held by consumer; producer must not overwrite  |

The `+1` offset reserves `0` as the "no readable frame" sentinel.

Producer claim: prefer the first `0` slot in cursor order; otherwise
`CAS(oldest_seq+1 -> 0)` on the oldest non-`HELD` published slot.
Consumer claim: `CAS(seq+1 -> seq+1 | HELD_BIT)` on the slot with the
lowest `seq >= consumer_next`. Under SPSC, CAS contention is bounded to a
single retry per operation.

## Ring rules

- If no slot has `seq >= consumer_next`, `SG-SLAM` waits on a POSIX
  semaphore that the producer posts after every publish.
- Producer never blocks: drop-oldest. Held slots are always preserved.
- Each slot is atomic — never deliver a scan without its labels.
- Start, stop, and reset only at frame boundaries. `rnsg_close()`
  automatically releases any still-held slot so a clean shutdown does not
  permanently lose ring capacity.

Sizing rule:

```
slot_count - 1 >= ceil(segmentation_latency / scan_period) + margin
```

The `-1` accounts for the slot the consumer holds during processing. With
SPSC and 1 held slot, effective working depth is `slot_count - 1`.
`slot_count` must be a power of two and `>= 2`.

## Concurrency Model

- Producer-only writers: `head_seq`, the seq portion of `slot_seq[i]`
  during publish, and the slot's payload bytes.
- Consumer-only writers: `tail_seq` (observability mirror), the `HELD_BIT`
  on the slot it claims, and the release-store of `0` to free a slot.
- Atomics use the C11 memory model (`memory_order_acquire` /
  `memory_order_release` / `memory_order_acq_rel`). No handwritten
  ISA-specific fences.
- Producer publish ordering: write payload -> release-store seq into
  `slot_seq[i]` -> release-store `head_seq` -> `sem_post`.
- Consumer acquire ordering: acquire-load `head_seq` -> scan
  `slot_seq[*]` with acquire -> CAS `slot_seq[best]` with acquire on
  success -> read header from shm.
- Consumer release ordering: release-store `0` to `slot_seq[held]` so all
  earlier reads from the slot happen-before the producer's next claim.
- Re-evaluate the design if the queue becomes MPMC or crosses a
  CPU <-> FPGA DMA/MMIO boundary. The current acquire/release lifetime
  carries over directly to AIE/PL backends: only the slot allocator
  changes (POSIX shm -> XRT buffer object).

## API surface

C ABI (`rangenet_sgslam_ipc/include/rnsg_ipc.h`), reused from both Python
(via `ctypes`) and C++:

- Lifecycle: `rnsg_create`, `rnsg_open`, `rnsg_close`, `rnsg_unlink`.
- Producer: `rnsg_producer_lease(ring, &slot_view)` returns mutable
  pointers into the next claimed slot; `rnsg_producer_publish(ring,
  num_points, capture_ns, frame_id, flags)` commits.
- Consumer: `rnsg_consumer_acquire(ring, timeout_ns, &frame_view)`
  returns `const` pointers directly into shm;
  `rnsg_consumer_release(ring)` returns the slot to the producer pool.
- Introspection: `rnsg_drop_count`, `rnsg_head_seq`, `rnsg_tail_seq`,
  `rnsg_slot_count`, `rnsg_capacity_points`, `rnsg_slot_bytes`.
- Errors: `RNSG_TIMEOUT`, `RNSG_E_BUSY` (acquire-while-holding or
  release-while-not-holding), plus standard invalid/IO/version codes.

Single-producer and single-consumer enforced at the API boundary by
`RNSG_E_BUSY` on lease-while-leased and acquire-while-holding.

## SG-SLAM Integration

- The non-ROS consumer is implemented as
  `SG-SLAM/cpp/semgraph_slam/apps/sgslam_ipc_runner.cpp`.
- Input selection now sits behind a `FrameSource` abstraction:
  `FileFrameSource` preserves the existing `.bin` + `.label` replay path and
  `IpcFrameSource` wraps `rnsg_consumer_acquire()` /
  `rnsg_consumer_release()`.
- `SemGraphSLAM` now has a borrowed-frame ingress that accepts a
  `BorrowedFrameView {const float *points_xyzi, const int32_t *raw_labels,
  num_points}`. This keeps the IPC transport zero-copy up to the SG-SLAM
  front-end boundary instead of forcing an eager copy in the adapter.
- The consumer still must finish all reads before calling `release`. The
  `FrameLease` RAII wrapper in the app layer enforces this on normal and
  exceptional exits.
- Remove artificial playback pacing from the hot path; process each frame
  as soon as a complete labeled frame is ready.
- **Set `cloudInd = rnsg_frame_view::consumed_index`** (the dense
  per-consumer counter), not `frame_id`. This keeps `poses_vec_`,
  `is_keyframe_vec`, the GTSAM `BetweenFactor(cloudInd-1, cloudInd, ...)`
  keys, and `initial.insert(cloudInd, ...)` all referring to a contiguous
  index space even when the IPC layer drops frames under producer overrun.
  Carry the producer's `frame_id` separately (e.g. add a `source_frame_id`
  field to the back-end queue struct) if you need traceability back to the
  source scan file.
- Dropped frames do not create a wrong "velocity output" in `SG-SLAM`, but
  they do make registration harder: the front-end warm-starts scan matching
  from the last relative pose only, with no explicit scaling by elapsed time
  or `skipped_before`. Regular decimation may still converge, but large or
  uneven gaps can make the initial guess worse and reduce scan-to-map
  convergence robustness.
- The file-based `.label` path stays available behind the same
  `FrameSource` polymorphism for offline replay and CI.

## Producer Integration

- The live GPU producer is implemented as
  `RangeNet/train/tasks/semantic/stream_sgslam_ipc.py`.
- It reuses the existing `RangeNet` model loading and `predict_scan()`
  inference path, but publishes directly to shared memory instead of writing
  `.label` files.
- The producer reads KITTI `.bin` scans in file order, preserves original
  `XYZI` point order, paces playback at the requested scan rate (default
  `10 Hz`), and publishes raw SemanticKITTI labels with
  `RNSG_FLAG_RAW_SEMANTICKITTI_LABELS`.

## Operational Artifacts

The implemented live path writes searchable traces on disk so failures can be
debugged later without repeating a full GPU run:

- Producer JSONL trace: per-frame publish record with scan path, `frame_id`,
  point count, inference latency, publish latency, drop counters, device, and
  model path.
- Producer manifest JSON: run-level metadata such as ring name, slot count,
  capacity, scan root, and selected model directory.
- Consumer CSV trace: `status`, `source`, `frame_id`, `consumed_index`,
  `skipped_before`, `num_points`, acquire wait, front-end latency,
  capture-to-consume latency, and publish-to-consume latency.

## Notes

- The key invariant remains: `scan_i` must always be paired with
  `label_i`.
- `frame_id` is mandatory in this implementation (sized into the slot
  header) and lets the consumer detect any out-of-order delivery cheaply.
  It is producer-assigned and SPARSE under drop-oldest. Use
  `consumed_index` for any consumer-side structural indexing; see
  "Frame indexing" above.
- Defaults: `slot_count = 8`, `capacity_points = 200000`. Each slot is
  `sizeof(header) + capacity_points * (16 + 4)` bytes ~ 4 MB; total ring
  footprint is therefore roughly `8 * slot_bytes` ~ 32 MB.
