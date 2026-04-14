# Vitis AI Integration for RangeNet DarkNet53

Accelerates the RangeNet semantic segmentation CNN on Xilinx DPU using
Vitis AI 3.0 with INT8 per-channel quantization.

## Overview

```
┌─────────────────────────────────────────────────────────┐
│  g5.xlarge (GPU instance)                               │
│  Vitis AI 3.0 Docker                                    │
│                                                         │
│  1. quantize_vitisai.py --quant-mode calib              │
│     └── Calibrates INT8 scales on KITTI frames          │
│                                                         │
│  2. quantize_vitisai.py --quant-mode test               │
│     └── Evaluates accuracy, exports .xmodel             │
│                                                         │
│  3. compile_vitisai.sh                                  │
│     └── Compiles for DPUCADF8H (AWS F1 DPU)            │
└────────────────────────────┬────────────────────────────┘
                             │ S3 transfer
                             ▼
┌─────────────────────────────────────────────────────────┐
│  f1.2xlarge (FPGA instance)                             │
│                                                         │
│  4. infer_vitisai.py                                    │
│     └── Runs on DPU, writes .label files for SG-SLAM   │
└─────────────────────────────────────────────────────────┘
```

## Files

| File | Purpose | Where it runs |
|------|---------|---------------|
| `quantize_vitisai.py` | PTQ calibration + INT8 accuracy eval + xmodel export | Vitis AI Docker (GPU instance) |
| `compile_vitisai.sh` | Compiles quantized xmodel for target DPU | Vitis AI Docker (GPU instance) |
| `infer_vitisai.py` | VART runtime inference, produces .label files | F1 instance with DPU overlay |
| `docker_run.sh` | Launches the Vitis AI Docker container | GPU instance |

## Prerequisites

- **AWS GPU instance** (g5.xlarge or larger) with Docker + NVIDIA Container Toolkit
- **AWS F1 instance** (f1.2xlarge) with DPUCADF8H overlay
- **KITTI data** at `/workspace/data/kitti/sequences/00/velodyne/`
- **Pretrained model** at `/workspace/models/rangenet_darknet53/`
- **Vitis AI 3.0 Docker image**: `xilinx/vitis-ai-pytorch-gpu:3.0.0`

## Step-by-Step

### 1. Setup (GPU instance)

```bash
# Pull the Docker image (~15 GB)
docker pull xilinx/vitis-ai-pytorch-gpu:3.0.0

# Launch the container
bash fpga/vitis/docker_run.sh
```

### 2. Calibrate (inside Docker)

```bash
python fpga/vitis/quantize_vitisai.py \
    --model /workspace/models/rangenet_darknet53 \
    --scan-root /workspace/data/kitti/sequences \
    --sequence 00 \
    --num-calib-frames 200 \
    --output-dir /workspace/vitis_output/quantize_result \
    --device cuda \
    --quant-mode calib
```

This runs 200 KITTI frames through the model to determine per-channel INT8 scale factors.
Output: `quant_info.json` in the output directory.

### 3. Test + Export (inside Docker)

```bash
python fpga/vitis/quantize_vitisai.py \
    --model /workspace/models/rangenet_darknet53 \
    --scan-root /workspace/data/kitti/sequences \
    --sequence 00 \
    --num-test-frames 50 \
    --output-dir /workspace/vitis_output/quantize_result \
    --device cuda \
    --quant-mode test
```

This evaluates INT8 vs FP32 mismatch and exports the quantized `.xmodel`.

### 4. Compile (inside Docker)

```bash
bash fpga/vitis/compile_vitisai.sh /workspace/vitis_output/quantize_result
```

Output: `compiled/rangenet_darknet53.xmodel` ready for the DPU.

### 5. Transfer to F1

```bash
aws s3 cp /workspace/vitis_output/quantize_result/compiled/rangenet_darknet53.xmodel \
    s3://your-bucket/vitis-models/
```

### 6. Run on DPU (F1 instance)

```bash
aws s3 cp s3://your-bucket/vitis-models/rangenet_darknet53.xmodel \
    /workspace/models/

python fpga/vitis/infer_vitisai.py \
    --xmodel /workspace/models/rangenet_darknet53.xmodel \
    --model /workspace/models/rangenet_darknet53 \
    --scan-root /workspace/data/kitti/sequences \
    --output-root /workspace/data/rangenet_vitisai/kitti \
    --sequence 00
```

Output: `.label` files compatible with SG-SLAM contract.

## Verifying Results

Compare DPU labels against the GPU baseline:

```bash
# On the GPU instance, the baseline labels are at:
#   /workspace/data/rangenet_sgslam/kitti/00/predictions/

# On F1, the DPU labels are at:
#   /workspace/data/rangenet_vitisai/kitti/00/predictions/

# Use the existing quantize_feasibility.py comparison logic,
# or a simple binary diff:
python -c "
import numpy as np
from pathlib import Path

gpu_dir = Path('/workspace/data/rangenet_sgslam/kitti/00/predictions')
dpu_dir = Path('/workspace/data/rangenet_vitisai/kitti/00/predictions')

total, mismatch = 0, 0
for lf in sorted(dpu_dir.glob('*.label')):
    gpu_lf = gpu_dir / lf.name
    if gpu_lf.exists():
        a = np.fromfile(str(gpu_lf), dtype=np.int32)
        b = np.fromfile(str(lf), dtype=np.int32)
        total += len(a)
        mismatch += int(np.sum(a != b))

print(f'Total points: {total}')
print(f'Mismatched:   {mismatch}')
print(f'Mismatch %:   {100*mismatch/max(total,1):.3f}%')
"
```

## Troubleshooting

- **`pytorch_nndct not found`**: You're not inside the Vitis AI Docker. Run `docker_run.sh`.
- **`No DPU subgraph found`**: The xmodel was compiled for a different DPU target than what's on the FPGA. Re-compile with the correct `arch.json`.
- **`xir/vart not found`**: VART runtime not installed on the F1 instance. Install the `vitis-ai-runtime` packages.
- **High mismatch rate (>5%)**: Try increasing `--num-calib-frames` to 500, or consider quantization-aware training (QAT) via `vai_q_pytorch` in `qat` mode.
- **ConvTranspose2d partitioned to CPU**: The decoder's asymmetric transpose convolutions may not map to the DPU. Check compiler logs for CPU-fallback warnings. If performance is an issue, replace with `nn.Upsample + nn.Conv2d`.
