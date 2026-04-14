# S3 Data Reference

**Bucket:** `s3://sgslam-data-448792657895`
**Region:** `us-east-1`

## Contents

| S3 prefix | Local path | Size | Description |
|---|---|---|---|
| `data/kitti/` | `data/kitti/` | 8.3 GB | KITTI odometry sequence 00 — raw LiDAR `.bin` scans and pose ground-truth used by the SLAM pipeline |
| `data/SegNet4D_predictions/` | `data/SegNet4D_predictions/` | 2.1 GB | Per-scan `.label` files from SegNet4D inference on KITTI seq 00 (semantic labels consumed by SG-SLAM) |
| `data/pretrained_darknet53_weights/` | `data/pretrained_darknet53_weights/` | 385 MB | DarkNet-53 backbone weights (`model.onnx`, YAML configs, SHA256 checksums) for RangeNet++ inference |
| `data/rangenet_gpu_labels/` | `data/rangenet_gpu_labels/` | 324 MB | Pre-computed RangeNet++ GPU label outputs for KITTI seq 00 |
| `fpga/golden_vectors/` | `fpga/golden_vectors/` | 11 GB | Layer-by-layer golden I/O vectors (3 frames × ~246 tensors each) for bitwise verification of HLS kernels |
| `fpga/weights/` | `fpga/weights/` | 193 MB | Flattened `.bin` weight/bias tensors for every conv/BN layer — used by FPGA kernels and HLS testbenches |
| `venv-rangenet/` | `.venv-rangenet/` | 885 MB | Pre-built Python 3 virtualenv with RangeNet++ dependencies (torch, numpy, etc.) |

## Pull commands

Configure credentials first:

```bash
aws configure
# Access Key ID:     (your key)
# Secret Access Key: (your secret)
# Region:            us-east-1
# Output:            json
```

### Pull everything

```bash
cd ~/slam
aws s3 sync s3://sgslam-data-448792657895/data/          data/
aws s3 sync s3://sgslam-data-448792657895/fpga/golden_vectors/ fpga/golden_vectors/
aws s3 sync s3://sgslam-data-448792657895/fpga/weights/   fpga/weights/
aws s3 sync s3://sgslam-data-448792657895/venv-rangenet/  .venv-rangenet/
```

### Pull only what you need

KITTI raw scans + poses (required for any SLAM run):

```bash
aws s3 sync s3://sgslam-data-448792657895/data/kitti/ data/kitti/
```

Semantic labels (needed if you skip live RangeNet++ inference):

```bash
aws s3 sync s3://sgslam-data-448792657895/data/SegNet4D_predictions/ data/SegNet4D_predictions/
aws s3 sync s3://sgslam-data-448792657895/data/rangenet_gpu_labels/  data/rangenet_gpu_labels/
```

FPGA development (HLS golden vectors + weights):

```bash
aws s3 sync s3://sgslam-data-448792657895/fpga/golden_vectors/ fpga/golden_vectors/
aws s3 sync s3://sgslam-data-448792657895/fpga/weights/        fpga/weights/
```

RangeNet++ inference environment:

```bash
aws s3 sync s3://sgslam-data-448792657895/data/pretrained_darknet53_weights/ data/pretrained_darknet53_weights/
aws s3 sync s3://sgslam-data-448792657895/venv-rangenet/ .venv-rangenet/
source .venv-rangenet/bin/activate
```
