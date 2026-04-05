# RangeNet to SG-SLAM Contract

This is the frozen interface for the first GPU baseline.

## Contract

- Input scan path: `/data/kitti/sequences/00/velodyne/<frame>.bin`
- Output label path: `/data/rangenet_sgslam/kitti/00/predictions/<frame>.label`
- Do not use `data/SegNet4D_predictions/` for exporter output: in this repo that tree is the bundled SegNet4D reference (ground-truth) labels. Pick a separate root such as `data/rangenet_sgslam/`.
- Filename contract: the `.label` basename must exactly match the `.bin` basename.
- Cardinality contract: one semantic label per LiDAR point, in original point order.
- Binary format: little-endian `int32`, 4 bytes per point.
- Bit layout: semantic class in the lower 16 bits, upper 16 bits set to zero.
- Label space: write raw SemanticKITTI-style IDs from `RangeNet` `learning_map_inv`.
- These `.label` files are `RangeNet` predictions, not SemanticKITTI ground-truth labels.
- Do not pre-remap to SG-SLAM reduced IDs. `SG-SLAM` already remaps raw labels on load in `ros/ros1/InsUtils.hpp` and `ros/ros2/InsUtils.hpp`.
- For correctness-only FPGA checks, a bounded deterministic subset is acceptable. Prefer the first `N` scans in sorted basename order via `--max-scans-per-sequence`.

## Raw Labels That Matter To SG-SLAM

SG-SLAM consumes raw labels, then reduces them internally.

- Vehicle-like objects: raw `10`, `18`, `20` -> reduced `1`, `4`, `5`
- Trunk: raw `71` -> reduced `16`
- Pole: raw `80` -> reduced `18`
- Background used downstream: raw `40`, `50`, `51`, `70` -> reduced `9`, `13`, `14`, `15`

Notes:

- Current `RangeNet` export through `learning_map_inv` emits canonical raw classes, so moving-class IDs such as `252` and `258` are not expected in exported predictions.
- `SG-SLAM` later filters reduced labels `>= 20` during preprocessing, so the contract should stay in the raw label space and let SG-SLAM own the reduction step.

## Exporter

Use `RangeNet/train/tasks/semantic/export_sgslam_labels.py`.

What it does:

- runs `RangeNet` directly on unlabeled scan folders
- writes SG-SLAM-compatible `.label` files
- writes `sgslam_contract_manifest.json`
- appends `sgslam_export_trace.jsonl` with per-scan timing and label summaries

## KITTI Example

Layout matches what SG-SLAM expects (`<sequence>/<predictions_subdir>/*.label`), but use a **different root** than `data/SegNet4D_predictions/`, which holds the bundled SegNet4D reference labels.

```bash
cd /home/ubuntu/src/slam/RangeNet/train/tasks/semantic
OMP_NUM_THREADS=1 python3 export_sgslam_labels.py \
  --scan-root /data/kitti/sequences \
  --output-root /data/rangenet_sgslam/kitti \
  --model /models/rangenet_darknet53 \
  --sequences 00 \
  --scan-subdir velodyne \
  --label-subdir predictions \
  --device cuda
```

Point SG-SLAM at the export directory (update `label_path` in your launch file or config):

```bash
label_path=/data/rangenet_sgslam/kitti/00/predictions/
```

For a bounded correctness run:

```bash
cd /home/ubuntu/src/slam/RangeNet/train/tasks/semantic
OMP_NUM_THREADS=1 python3 export_sgslam_labels.py \
  --scan-root /data/kitti/sequences \
  --output-root /data/rangenet_sgslam_700/kitti \
  --model /models/rangenet_darknet53 \
  --sequences 00 \
  --scan-subdir velodyne \
  --label-subdir predictions \
  --max-scans-per-sequence 700 \
  --device cuda
```

This deterministically yields `000000.label` through `000699.label`.

## Reproducibility Notes

- Keep `arch_cfg.yaml` and `data_cfg.yaml` alongside the pretrained weights in the model directory.
- Keep the generated manifest and JSONL trace with any benchmark run so you can compare FPGA results against the exact GPU baseline without rerunning the export.
- For bounded subset runs, record the subset size and basename range so the FPGA side consumes the exact same scans.
- On AWS, use a CUDA-capable instance for this stage. The exporter is meant to establish the baseline before any `dora-rs` or FPGA integration work.
