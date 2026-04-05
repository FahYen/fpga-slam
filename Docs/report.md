# RangeNet DarkNet53 FPGA Quantization Feasibility

## Setup

- **Model**: DarkNet53 backbone (40.6M) + decoder (9.8M) + head (5.8K) = 50.4M params
- **Architecture**: Conv2d, BN, LeakyReLU, ConvTranspose2d, residual adds (all standard fixed-point ops)
- **Input**: 5 x 64 x 2048 (range, xyz, remission), ~124K points/frame
- **Golden reference**: 700 GPU-produced labels (`data/rangenet_gpu_labels/`), CRF disabled
- **Test**: 10 frames (seq 00, frames 000000-000009), NVIDIA A10G
- **Quantization method**: Post-training symmetric; per-output-channel weights; per-layer and per-channel activations

## Results

| Scenario | Mean Mismatch | Max | Min |
|---|---|---|---|
| FP32 vs golden | 0.00% | 0.00% | 0.00% |
| INT16 wt only | 0.00% | 0.01% | 0.00% |
| INT16 wt + INT16 act (per-layer) | 0.01% | 0.01% | 0.00% |
| INT8 wt only | 0.20% | 0.22% | 0.17% |
| INT8 wt + INT8 act (per-channel) | 1.11% | 1.73% | 0.86% |
| INT8 wt + INT8 act (per-layer) | 20.59% | 24.51% | 16.83% |

## Findings

1. **FP32 reproduces golden labels exactly** — sanity check passes.
2. **INT16×INT16 is essentially lossless** — 0.01% mean. Simplest viable FPGA datapath.
3. **INT8 weight-only is safe** — 0.2% mismatch (~245 pixels/frame near decision boundaries).
4. **INT8×INT8 per-channel activations are viable** — 1.11% mean. Requires a per-channel scale LUT per layer but enables 8-bit multipliers.
5. **INT8×INT8 per-layer activations are not viable** — 20.6% mismatch. Per-layer min/max is too coarse.

## FPGA Design Options

| Design | Multiplier | Accuracy | Complexity |
|---|---|---|---|
| INT16×INT16 per-layer | 16-bit | ~lossless (0.01%) | Simplest — one scale per layer |
| INT8×INT8 per-channel | 8-bit | ~1% loss | Per-channel scale LUT per layer |

| Component | Note |
|---|---|
| Softmax | Skip — argmax(logits) = argmax(softmax) |
| CRF | Skip — disabled in config |
| BatchNorm | Fold into Conv at export time |

## Next Steps

- Export BN-folded weights in FPGA-consumable binary format (layer-by-layer, little-endian)
- Profile per-layer compute/memory to guide FPGA resource allocation
- If targeting INT8×INT8: evaluate histogram-based (percentile) activation calibration for further improvement

## Artifacts

- Script: `RangeNet/train/tasks/semantic/quantize_feasibility.py`
- v1 report: `aws_runs/20260405T180709Z/quantize_feasibility_report.json`
- v2 report (with INT16 act + per-channel INT8 act): `aws_runs/20260405T180709Z/quantize_v2_report.json`
- AWS run script: `aws_runs/20260405T180709Z/remote_quantize.sh`
