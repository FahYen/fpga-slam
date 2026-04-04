# AWS GPU Baseline Runbook

This runbook gets you to a reproducible `RangeNet -> .label -> SG-SLAM` GPU baseline before any `dora-rs` or FPGA work.

## Recommended Instance

- Start with `g5.xlarge` for smoke tests and contract validation.
- Use `g5.2xlarge` or larger if you want faster full-sequence export throughput.
- Prefer an image that already has NVIDIA drivers and CUDA working. A Deep Learning AMI or a CUDA-ready Ubuntu image is the least risky choice.
- Attach enough storage for datasets and artifacts. `200 GB` gp3 is a practical minimum if you will keep KITTI, MulRAN, model files, logs, and traces on the instance.

## What To Persist

Before the expensive run, capture searchable artifacts on disk:

- environment snapshot
- `nvidia-smi`
- Python package list
- exporter stdout/stderr log
- `sgslam_contract_manifest.json`
- `sgslam_export_trace.jsonl`

These are your baseline receipts. Keep them with the generated `.label` files.

## Directory Convention

Use a stable layout so later FPGA comparisons are apples-to-apples:

```text
/workspace/slam
/workspace/data/kitti/sequences
/workspace/data/mulran_in_kitti/sequences
/workspace/models/rangenet_darknet53
/workspace/runs/<timestamp>/
```

## Bootstrap

Clone the workspace:

```bash
mkdir -p /workspace
cd /workspace
git clone <your-repo-url> slam
```

Create a Python environment:

```bash
sudo apt-get update
sudo apt-get install -y git tmux build-essential python3 python3-venv python3-pip
python3 -m venv /workspace/venv-rangenet
source /workspace/venv-rangenet/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

Install PyTorch with CUDA support using the current command from the official PyTorch selector, then install the remaining Python deps needed by `RangeNet`.

Minimum verification:

```bash
OMP_NUM_THREADS=1 python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY
```

If that does not report CUDA availability, stop there and fix the image or driver before doing anything else.

## Model Directory

The exporter expects:

```text
/workspace/models/rangenet_darknet53/
  arch_cfg.yaml
  data_cfg.yaml
  backbone
  segmentation_decoder
  segmentation_head
```

If KNN post-processing is enabled in `arch_cfg.yaml`, the exporter will apply it automatically.

## Capture Baseline Metadata

Create a run directory and dump environment metadata before inference:

```bash
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR=/workspace/runs/$RUN_ID
mkdir -p "$RUN_DIR"

uname -a > "$RUN_DIR/uname.txt"
nvidia-smi > "$RUN_DIR/nvidia-smi.txt"
python3 --version > "$RUN_DIR/python-version.txt"
OMP_NUM_THREADS=1 python3 -m pip freeze > "$RUN_DIR/pip-freeze.txt"
```

## Smoke Test On KITTI

Run a short export first:

```bash
cd /workspace/slam/RangeNet/train/tasks/semantic
OMP_NUM_THREADS=1 python3 export_sgslam_labels.py \
  --scan-root /workspace/data/kitti/sequences \
  --output-root /workspace/data/SegNet4D_predictions/kitti \
  --model /workspace/models/rangenet_darknet53 \
  --sequences 00 \
  --scan-subdir velodyne \
  --label-subdir predictions \
  --max-scans-per-sequence 10 \
  2>&1 | tee "$RUN_DIR/kitti-smoke.log"
```

Quick checks:

- verify `.label` files appear under `/workspace/data/SegNet4D_predictions/kitti/00/predictions/`
- verify `sgslam_contract_manifest.json` and `sgslam_export_trace.jsonl` were written
- verify the number of bytes in each `.label` file is `4 * number_of_points`

## Full KITTI Baseline

Once the smoke test passes:

```bash
cd /workspace/slam/RangeNet/train/tasks/semantic
OMP_NUM_THREADS=1 python3 export_sgslam_labels.py \
  --scan-root /workspace/data/kitti/sequences \
  --output-root /workspace/data/SegNet4D_predictions/kitti \
  --model /workspace/models/rangenet_darknet53 \
  --sequences 00 \
  --scan-subdir velodyne \
  --label-subdir predictions \
  2>&1 | tee "$RUN_DIR/kitti-full.log"
```

Then point `SG-SLAM` at:

```text
/workspace/data/SegNet4D_predictions/kitti/00/predictions/
```

## MulRAN Baseline

For the existing SG-SLAM MulRAN layout:

```bash
cd /workspace/slam/RangeNet/train/tasks/semantic
OMP_NUM_THREADS=1 python3 export_sgslam_labels.py \
  --scan-root /workspace/data/mulran_in_kitti/sequences \
  --output-root /workspace/data/mulran_in_kitti/sequences \
  --model /workspace/models/rangenet_darknet53 \
  --sequences 00 \
  --scan-subdir Ouster \
  --label-subdir segnet4d \
  2>&1 | tee "$RUN_DIR/mulran-full.log"
```

Then point `SG-SLAM` at:

```text
/workspace/data/mulran_in_kitti/sequences/00/segnet4d/
```

## SG-SLAM Integration Check

At this stage, do not introduce `dora-rs` yet.

The goal is only:

- `RangeNet` generates `.label`
- `SG-SLAM` reads `.label`
- trajectory quality and runtime are recorded

Only after that baseline is stable should you decide whether to wrap the stages with a middleware layer.

## What To Save For FPGA Comparison

Keep these together:

- exported `.label` directories
- `sgslam_contract_manifest.json`
- `sgslam_export_trace.jsonl`
- `kitti-smoke.log`, `kitti-full.log`, `mulran-full.log`
- `nvidia-smi.txt`
- `pip-freeze.txt`
- exact model directory used

This gives you a fixed software baseline to compare against later HLS and F1 runs.
