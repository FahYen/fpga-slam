# GPU Full Integration

This repo now has reusable scripts for the non-ROS
`RangeNet -> IPC -> SG-SLAM` path.

The default test path should be the `10`-frame smoke run, not the full KITTI
sequence.

## Assumptions

- repo root: `/home/ubuntu/src/slam`
- KITTI scans live under `data/kitti/sequences/00/velodyne`
- pretrained weights live under `data/pretrained_darknet53_weights`
- the target machine has a working NVIDIA driver

## Recommended Default

Use this order:

1. environment setup
2. data sync
3. `10`-frame smoke run

Only run the full sequence when you explicitly need it.

## Scripts

### 1. Optional: package the local GTSAM install

This is optional.

If the target machine does not already have GTSAM under `/usr/local`, the setup
script can now build and install it from `third_party/gtsam` automatically.

Creating the archive is still the faster path on a fresh machine:

```bash
cd /home/ubuntu/src/slam
./scripts/package_local_gtsam.sh
```

That writes:

```text
scripts/gtsam-local-install.tar.gz
```

### 2. Environment setup

This installs the Python/C++ dependencies, restores the packaged GTSAM archive
if present, otherwise builds and installs GTSAM from `third_party/gtsam`,
verifies CUDA `torch`, and builds both the IPC library and the standalone
SG-SLAM consumer:

```bash
cd /home/ubuntu/src/slam
./scripts/gpu_setup_env.sh
```

### 3. Sync only the test data you need

This pulls KITTI seq `00` and the pretrained weights from the project S3 bucket:

```bash
cd /home/ubuntu/src/slam
./scripts/gpu_sync_test_data.sh
```

### 4. Run the default smoke test

This starts the producer first, waits for the IPC ring to exist, then starts
the SG-SLAM consumer. It runs only `10` scans and asks SG-SLAM to consume only
`10` frames.

```bash
cd /home/ubuntu/src/slam
./scripts/gpu_smoke_10frames.sh
```

Artifacts are written under:

```text
sgslam_runs/gpu_smoke_10frames/<timestamp>/
```

## What Each Script Does

- `scripts/package_local_gtsam.sh`
  Packages the current `/usr/local` GTSAM install plus the matching
  `libmetis-gtsam.so` helper into a tarball for reuse on another machine.

- `scripts/gpu_setup_env.sh`
  Installs apt dependencies, creates `.venv-rangenet`, installs CUDA-enabled
  `torch`, verifies CUDA visibility, restores GTSAM from the archive when
  available or builds it from source when not, and builds:
  - `rangenet_sgslam_ipc`
  - `SG-SLAM/cpp/semgraph_slam/build/apps/sgslam_ipc_runner`

- `scripts/gpu_sync_test_data.sh`
  Syncs:
  - `data/kitti/`
  - `data/pretrained_darknet53_weights/`

- `scripts/gpu_smoke_10frames.sh`
  Runs the integrated `10`-frame GPU smoke test with durable logs and traces.

- `scripts/aws_gpu_smoke_10frames.sh`
  Launches a fresh AWS GPU instance, syncs the repo, runs the same `10`-frame
  smoke path remotely, copies artifacts back under `aws_runs/<timestamp>/`, and
  cleans up the temporary instance, key pair, and security group.

## If You Still Want Manual Commands

### Producer, capped at 10 scans

```bash
cd /home/ubuntu/src/slam/RangeNet/train/tasks/semantic
OMP_NUM_THREADS=1 /home/ubuntu/src/slam/.venv-rangenet/bin/python3 stream_sgslam_ipc.py \
  --scan-root /home/ubuntu/src/slam/data/kitti/sequences \
  --sequence 00 \
  --scan-subdir velodyne \
  --model /home/ubuntu/src/slam/data/pretrained_darknet53_weights \
  --device cuda \
  --ipc-name /rnsg_kitti_00 \
  --hz 10 \
  --slot-count 8 \
  --capacity-points 200000 \
  --max-scans 10 \
  --unlink-existing \
  --trace-path /tmp/rangenet_ipc_trace.jsonl \
  --manifest-path /tmp/rangenet_ipc_manifest.json
```

### Consumer, capped at 10 frames

```bash
cd /home/ubuntu/src/slam/SG-SLAM/cpp/semgraph_slam
./build/apps/sgslam_ipc_runner \
  --input-mode ipc \
  --dataset kitti \
  --ipc-name /rnsg_kitti_00 \
  --acquire-timeout-s 1 \
  --max-idle-timeouts 10 \
  --max-frames 10 \
  --result-path /tmp/kitti_odometry_00.txt \
  --pgo-result-path /tmp/kitti_slam_00.txt \
  --graph-map-path /tmp/graph_map_00.txt \
  --graph-edge-path /tmp/graph_edge_00.txt \
  --trace-path /tmp/sgslam_consumer_trace.csv
```

## Artifacts To Keep

For smoke runs, keep:

- `rangenet_ipc_manifest.json`
- `rangenet_ipc_trace.jsonl`
- `sgslam_consumer_trace.csv`
- `kitti_odometry_00.txt`
- producer and consumer logs

For full runs, keep those plus any PGO/map outputs that are produced.

## One-Command AWS Smoke Run

If you want the whole `10`-frame test on a fresh AWS GPU instance:

```bash
cd /home/ubuntu/src/slam
./scripts/aws_gpu_smoke_10frames.sh
```

What it does:

- verifies local AWS CLI access
- packages the local `/usr/local` GTSAM install first when available, for the
  faster setup path
- otherwise relies on `gpu_setup_env.sh` to build GTSAM from `third_party/gtsam`
- launches a `g5.xlarge` in `us-east-1`
- syncs the repo to the instance
- runs:
  - `./scripts/gpu_setup_env.sh`
  - `./scripts/gpu_sync_test_data.sh`
  - `./scripts/gpu_smoke_10frames.sh`
- copies remote smoke artifacts back under `aws_runs/<timestamp>/remote-results/`
- terminates the instance and removes the temporary key/security group

Important defaults:

- smoke only: `10` scans / `10` consumer frames
- root volume: `200 GB`
- region: `us-east-1`
- instance type: `g5.xlarge`
