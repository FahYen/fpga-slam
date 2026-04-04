# RangeNet to SG-SLAM Contract

This is the frozen interface for the first GPU baseline.

## Contract

- Input scan path: `<scan_root>/<sequence>/<scan_subdir>/<frame>.bin`
- Output label path: `<output_root>/<sequence>/<label_subdir>/<frame>.label`
- Filename contract: the `.label` basename must exactly match the `.bin` basename.
- Cardinality contract: one semantic label per LiDAR point, in original point order.
- Binary format: little-endian `int32`, 4 bytes per point.
- Bit layout: semantic class in the lower 16 bits, upper 16 bits set to zero.
- Label space: write raw SemanticKITTI-style IDs from `RangeNet` `learning_map_inv`.
- Do not pre-remap to SG-SLAM reduced IDs. `SG-SLAM` already remaps raw labels on load in `ros/ros1/InsUtils.hpp` and `ros/ros2/InsUtils.hpp`.

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

This produces labels in the same layout expected by the default SG-SLAM KITTI launch file.

```bash
cd /home/ubuntu/src/slam/RangeNet/train/tasks/semantic
OMP_NUM_THREADS=1 python3 export_sgslam_labels.py \
  --scan-root /data/kitti/sequences \
  --output-root /data/SegNet4D_predictions/kitti \
  --model /models/rangenet_darknet53 \
  --sequences 00 \
  --scan-subdir velodyne \
  --label-subdir predictions
```

Then point SG-SLAM at:

```bash
label_path=/data/SegNet4D_predictions/kitti/00/predictions/
```

## MulRAN Example

This matches the existing SG-SLAM MulRAN launch layout where scans live under `Ouster` and labels are read from `segnet4d`.

```bash
cd /home/ubuntu/src/slam/RangeNet/train/tasks/semantic
OMP_NUM_THREADS=1 python3 export_sgslam_labels.py \
  --scan-root /data/mulran_in_kitti/sequences \
  --output-root /data/mulran_in_kitti/sequences \
  --model /models/rangenet_darknet53 \
  --sequences 00 \
  --scan-subdir Ouster \
  --label-subdir segnet4d
```

Then point SG-SLAM at:

```bash
label_path=/data/mulran_in_kitti/sequences/00/segnet4d/
```

## Reproducibility Notes

- Keep `arch_cfg.yaml` and `data_cfg.yaml` alongside the pretrained weights in the model directory.
- Keep the generated manifest and JSONL trace with any benchmark run so you can compare FPGA results against the exact GPU baseline without rerunning the export.
- On AWS, use a CUDA-capable instance for this stage. The exporter is meant to establish the baseline before any `dora-rs` or FPGA integration work.
