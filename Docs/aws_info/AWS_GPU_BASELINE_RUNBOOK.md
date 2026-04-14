# AWS GPU Baseline Runbook

This runbook gets you to a reproducible `RangeNet -> .label -> SG-SLAM` GPU baseline before any `dora-rs` or FPGA work.

## Recommended Instance

- Start with `g5.xlarge` for smoke tests and contract validation.
- Use `g5.2xlarge` or larger if you want faster full-sequence export throughput.
- Prefer an image that already has NVIDIA drivers and CUDA working. A Deep Learning AMI or a CUDA-ready Ubuntu image is the least risky choice.
- Attach enough gp3 for OS, Python env, models, your `data/` tree, and run outputs. If you only stage the KITTI sequences you need (not a full mirror), **`50–80 GB`** is usually sufficient.
- A single `g5.xlarge` with an NVIDIA A10G was sufficient for a `700`-scan correctness run.

## What To Persist

Before the expensive run, capture searchable artifacts on disk:

- environment snapshot
- `nvidia-smi`
- Python package list
- exporter stdout/stderr log
- `sgslam_contract_manifest.json`
- `sgslam_export_trace.jsonl`
- subset selection note if you intentionally stop at fewer than all scans

These are your baseline receipts. Keep them with the generated `.label` files.

## Directory Convention

Use a stable layout so later FPGA comparisons are apples-to-apples:

```text
/workspace/slam
/workspace/data/kitti/sequences
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

For exporter-only work, a minimal environment is enough:

```bash
python -m pip install --upgrade pip wheel setuptools
python -m pip install "numpy<1.24" PyYAML scipy
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Notes:

- `scipy` is required even when CRF is disabled because `tasks.semantic.postproc.CRF` is imported at startup.
- Some GPU AMIs still do not ship with a ready-to-use Python environment for this repo, so expect to create your own venv.
- Adjust the PyTorch CUDA wheel selector if you use a different driver or CUDA stack.

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

Download the Bonn pretrained `darknet53` bundle and unpack it into the model directory:

```bash
mkdir -p /workspace/models
curl -fL http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/darknet53.tar.gz \
  -o /workspace/models/darknet53.tar.gz
tar -xzf /workspace/models/darknet53.tar.gz -C /workspace/models
mv /workspace/models/darknet53 /workspace/models/rangenet_darknet53
```

The exporter expects:

```text
/workspace/models/rangenet_darknet53/
  arch_cfg.yaml
  data_cfg.yaml
  backbone
  segmentation_decoder
  segmentation_head
```

These weights are loaded directly by `Segmentator`.

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
  --output-root /workspace/data/rangenet_sgslam/kitti \
  --model /workspace/models/rangenet_darknet53 \
  --sequences 00 \
  --scan-subdir velodyne \
  --label-subdir predictions \
  --max-scans-per-sequence 10 \
  --device cuda \
  2>&1 | tee "$RUN_DIR/kitti-smoke.log"
```

Quick checks:

- verify `.label` files appear under `/workspace/data/rangenet_sgslam/kitti/00/predictions/`
- verify `sgslam_contract_manifest.json` and `sgslam_export_trace.jsonl` were written
- verify the number of bytes in each `.label` file is `4 * number_of_points`

Important:

- `export_sgslam_labels.py` writes `RangeNet` predictions, not dataset ground-truth labels.

## Correctness Subset

For FPGA correctness checks, keep the export bounded and deterministic instead of paying for the full sequence:

```bash
cd /workspace/slam/RangeNet/train/tasks/semantic
OMP_NUM_THREADS=1 python3 export_sgslam_labels.py \
  --scan-root /workspace/data/kitti/sequences \
  --output-root /workspace/data/rangenet_sgslam_700/kitti \
  --model /workspace/models/rangenet_darknet53 \
  --sequences 00 \
  --scan-subdir velodyne \
  --label-subdir predictions \
  --max-scans-per-sequence 700 \
  --device cuda \
  2>&1 | tee "$RUN_DIR/kitti-700.log"
```

This deterministically yields `000000.label` through `000699.label` under `/workspace/data/rangenet_sgslam_700/kitti/00/predictions/`.

## Full KITTI Baseline

Once the smoke test passes:

```bash
cd /workspace/slam/RangeNet/train/tasks/semantic
OMP_NUM_THREADS=1 python3 export_sgslam_labels.py \
  --scan-root /workspace/data/kitti/sequences \
  --output-root /workspace/data/rangenet_sgslam/kitti \
  --model /workspace/models/rangenet_darknet53 \
  --sequences 00 \
  --scan-subdir velodyne \
  --label-subdir predictions \
  --device cuda \
  2>&1 | tee "$RUN_DIR/kitti-full.log"
```

Then point `SG-SLAM` at:

```text
/workspace/data/rangenet_sgslam/kitti/00/predictions/
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
- if you use a bounded subset, record the exact subset size and basename range
- `kitti-smoke.log`, `kitti-full.log`
- `nvidia-smi.txt`
- `pip-freeze.txt`
- exact model directory used

This gives you a fixed software baseline to compare against later HLS and F1 runs.
