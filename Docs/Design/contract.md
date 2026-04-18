# Shared-Memory RangeNet -> SG-SLAM Contract Plan

This replaces the temporary file/basename `.label` handoff. The target interface is a shared-memory FIFO carrying complete labeled LiDAR frames from `RangeNet` to `SG-SLAM`.

## Contract

- Producer: `RangeNet` runtime writes complete labeled frames into a shared-memory ring/FIFO.
- Consumer: `SG-SLAM` blocks on `blocking_pop()` and processes the next ready frame.
- One FIFO element = one scan plus its semantic labels.
- Delivery must remain in capture order during normal operation.
- Pipeline latency may be multiple scan periods; correctness depends on in-order delivery of complete frames.

## Frame Payload

Each FIFO element must contain:

- `num_points`
- point cloud in original point order
- one semantic label per point, same order
- optional `timestamp` and `frame_id` for diagnostics

Label rules:

- use raw `SemanticKITTI` IDs
- store one `int32` label per point
- do not pre-remap to `SG-SLAM` reduced IDs
- `SG-SLAM` keeps ownership of label remap and filtering

## FIFO Rules

- If FIFO is empty, `SG-SLAM` waits.
- If FIFO is full, producer or upstream capture stalls.
- Do not drop only scans or only labels.
- Start, stop, and reset only at frame boundaries.

Sizing rule:

`fifo_depth >= ceil(segmentation_latency / scan_period) + margin`

## Concurrency Model

- Preferred topology: `SPSC` (`RangeNet` producer, `SG-SLAM` consumer).
- If implemented lock-free, write against the language memory model, not handwritten ISA-specific synchronization.
- Producer writes payload first, then publishes readiness or tail update with `release`.
- Consumer observes readiness or tail with `acquire`, then reads the payload.
- `Rust` or `C++` are both acceptable if they use correct atomics.
- Re-evaluate the design if the queue becomes `MPMC` or crosses a CPU <-> FPGA DMA/MMIO boundary.

## SG-SLAM Integration

- Replace basename-based `loadCloud(<frame>.bin, <frame>.label)` with shared-memory `blocking_pop()`.
- Keep `mainProcess(frame, labels, timestamps, dataset)` unchanged as much as possible.
- Remove artificial playback pacing from the hot path; process each frame as soon as a complete labeled frame is ready.

## Notes

- A sideband `frame_id` is optional if the pipeline is strictly in-order, but recommended as a cheap sync check.
- The key invariant is simple: `scan_i` must always be paired with `label_i`.
