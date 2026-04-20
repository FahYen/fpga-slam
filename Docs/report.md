# RangeNet DarkNet53 FPGA Quantization Feasibility

## Setup

- **Model**: DarkNet53 backbone (40.6M) + decoder (9.8M) + head (5.8K) = 50.4M params
- **Architecture**: Conv2d, BN, LeakyReLU, ConvTranspose2d, residual adds (all standard fixed-point ops)
- **Input**: 5 x 64 x 2048 (range, xyz, remission), ~124K points/frame
- **Golden reference**: 700 GPU-produced labels (`data/rangenet_gpu_labels/`), CRF disabled
- **Test**: 10 frames (seq 00, frames 000000-000009); original INT8/INT16 study on NVIDIA A10G, INT6/INT4 extension rerun on CPU because no working local NVIDIA driver was available
- **Quantization method**: Post-training symmetric; per-output-channel weights; per-layer and per-channel activations

## Results

| Scenario | Mean Mismatch | Max | Min |
|---|---|---|---|
| FP32 vs golden | 0.00% | 0.00% | 0.00% |
| INT16 wt only | 0.00% | 0.01% | 0.00% |
| INT16 wt + INT16 act (per-layer) | 0.01% | 0.01% | 0.00% |
| INT8 wt only | 0.20% | 0.22% | 0.17% |
| INT8 wt + INT8 act (per-channel) | 1.11% | 1.73% | 0.86% |
| INT6 wt + INT6 act (per-channel) | 20.73% | 22.10% | 18.96% |
| INT4 wt + INT4 act (per-channel) | 81.97% | 82.72% | 81.21% |

## Findings

1. **FP32 remains the reference** — the original GPU baseline reproduced the golden labels exactly, and the CPU rerun used for INT6/INT4 stayed within 0.005% of golden, so the new sub-8-bit results are dominated by quantization error rather than device drift.
2. **INT16×INT16 is essentially lossless** — 0.01% mean. Simplest viable FPGA datapath.
3. **INT8 weight-only is safe** — 0.2% mismatch (~245 pixels/frame near decision boundaries).
4. **INT8×INT8 per-channel activations are still the best PTQ trade-off** — 1.11% mean. Requires a per-channel scale LUT per layer but enables 8-bit multipliers.
5. **INT6×INT6 per-channel PTQ is not viable** — 20.73% mean, almost identical to the failed INT8 per-layer case. Per-channel scaling alone does not rescue 6-bit post-training quantization here.
6. **INT8×INT8 per-layer activations are not viable** — 20.6% mismatch. Per-layer min/max is too coarse.
7. **INT4×INT4 per-channel PTQ is unusable** — 81.97% mean mismatch. This precision needs quantization-aware training and/or mixed precision, not straight PTQ.

## FPGA Design Options

| Design | Multiplier | Accuracy | Complexity |
|---|---|---|---|
| INT16×INT16 per-layer | 16-bit | ~lossless (0.01%) | Simplest — one scale per layer |
| INT8×INT8 per-channel | 8-bit | ~1% loss | Per-channel scale LUT per layer |
| INT6×INT6 per-channel | 6-bit | ~20.7% loss | Same LUT scheme as INT8, but accuracy is too poor to justify deployment |
| INT4×INT4 per-channel | 4-bit | ~82.0% loss | Not viable without QAT or mixed precision |

| Component | Note |
|---|---|
| Softmax | Skip — argmax(logits) = argmax(softmax) |
| CRF | Skip — disabled in config |
| BatchNorm | Fold into Conv at export time |

## Next Steps

- Export BN-folded weights in FPGA-consumable binary format (layer-by-layer, little-endian)
- Profile per-layer compute/memory to guide FPGA resource allocation
- Keep the plain PTQ hardware target at INT8×INT8 per-channel or INT16×INT16; do not pursue full INT6/INT4 PTQ as-is
- If sub-8-bit compute is still required, evaluate QAT or mixed precision (for example INT8 stem/head plus lower-bit body) instead of more plain min/max PTQ

## Artifacts

- Script: `RangeNet/train/tasks/semantic/quantize_feasibility.py`
- v1 report: `aws_runs/20260405T180709Z/quantize_feasibility_report.json`
- v2 report (with INT16 act + per-channel INT8 act): `aws_runs/20260405T180709Z/quantize_v2_report.json`
- v3 report (INT6/INT4 per-channel, CPU rerun): `aws_runs/20260412T001429Z_int6_int4_perchan/quantize_int6_int4_perchan_10frames.json`
- v3 log: `aws_runs/20260412T001429Z_int6_int4_perchan/quantize_int6_int4_perchan_10frames.log`
- v3 environment receipts: `aws_runs/20260412T001429Z_int6_int4_perchan/torch-check.txt`, `aws_runs/20260412T001429Z_int6_int4_perchan/nvidia-smi.txt`, `aws_runs/20260412T001429Z_int6_int4_perchan/pip-freeze.txt`
- AWS run script: `aws_runs/20260405T180709Z/remote_quantize.sh`

## AWS GPU IPC Integration

### Setup

- **Date**: `2026-04-19`
- **Run id**: `aws_runs/20260419T162450Z`
- **Instance**: `g5.xlarge`
- **GPU**: `NVIDIA A10G`
- **Torch/CUDA**: `torch 2.11.0+cu128`, `cuda_available=True`
- **Flow tested**: `KITTI seq 00 -> RangeNet GPU -> rnsg_ipc -> SG-SLAM IPC consumer`

### End-to-End Result

- The non-ROS `RangeNet -> IPC -> SG-SLAM` path ran successfully on AWS GPU.
- Producer completed all `4541` KITTI scans.
- SG-SLAM consumer processed `4344` frames and wrote `4344` odometry lines.
- Producer-side dropped frames: `196`
- Consumer-side skipped frames observed: `197`
- Shutdown behavior was clean: the consumer stopped after `10` consecutive timeouts once the producer finished.

### Timing Notes

- **RangeNet point-load average**: `161.3 ms`
- **RangeNet infer average**: `70.1 ms`
- **RangeNet infer p95**: `76.2 ms`
- **SG-SLAM frontend average**: `70.5 ms`
- **SG-SLAM frontend p95**: `179.3 ms`

Producer-side per-frame time before IPC publish was approximately:

```text
161.3 ms (point load) + 70.1 ms (inference) ~= 231.4 ms
```

That means this setup did **not** sustain a true `10 Hz` end-to-end rate for the full run. The producer trace ended with about `613.6 s` of accumulated playback lag.

`p95` means the 95th percentile: `95%` of measured frames were at or below that latency, and the slowest `5%` were above it.

### Caveat

- This run confirmed the live GPU producer, IPC handoff, and SG-SLAM IPC consumer/odometry path.
- `kitti_slam_00.txt`, `graph_map_00.txt`, and `graph_edge_00.txt` were not produced in this AWS run, so the validated path here is strongest for the front-end/odometry side rather than final back-end map export.

### Integration Artifacts

- Run summary: `aws_runs/20260419T162450Z/remote-results/run-summary.txt`
- Producer manifest: `aws_runs/20260419T162450Z/remote-results/rangenet_ipc_manifest.json`
- Producer trace: `aws_runs/20260419T162450Z/remote-results/rangenet_ipc_trace.jsonl`
- Consumer trace: `aws_runs/20260419T162450Z/remote-results/sgslam_consumer_trace.csv`
- Odometry output: `aws_runs/20260419T162450Z/remote-results/kitti_odometry_00.txt`
- Environment receipts: `aws_runs/20260419T162450Z/remote-results/torch-check-gtsam.txt`, `aws_runs/20260419T162450Z/remote-results/nvidia-smi.txt`, `aws_runs/20260419T162450Z/remote-results/pip-freeze.txt`
