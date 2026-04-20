#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv-rangenet}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/sgslam_runs/gpu_smoke_10frames}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$RUN_ROOT/$RUN_ID"
IPC_NAME="${IPC_NAME:-/rnsg_kitti_00}"
SCAN_ROOT="${SCAN_ROOT:-$REPO_ROOT/data/kitti/sequences}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/data/pretrained_darknet53_weights}"

mkdir -p "$RUN_DIR/logs"

pkill -f stream_sgslam_ipc.py || true
pkill -f sgslam_ipc_runner || true
rm -f "/dev/shm/${IPC_NAME#/}"

PRODUCER_LOG="$RUN_DIR/logs/producer.log"
CONSUMER_LOG="$RUN_DIR/logs/consumer.log"

OMP_NUM_THREADS=1 "$VENV_DIR/bin/python3" \
  "$REPO_ROOT/RangeNet/train/tasks/semantic/stream_sgslam_ipc.py" \
  --scan-root "$SCAN_ROOT" \
  --sequence 00 \
  --scan-subdir velodyne \
  --model "$MODEL_DIR" \
  --device cuda \
  --ipc-name "$IPC_NAME" \
  --hz 10 \
  --slot-count 8 \
  --capacity-points 200000 \
  --max-scans 10 \
  --unlink-existing \
  --trace-path "$RUN_DIR/rangenet_ipc_trace.jsonl" \
  --manifest-path "$RUN_DIR/rangenet_ipc_manifest.json" \
  >"$PRODUCER_LOG" 2>&1 &
PRODUCER_PID=$!

trap 'kill "$PRODUCER_PID" "$CONSUMER_PID" 2>/dev/null || true' EXIT

python3 - <<PY
from pathlib import Path
import time
path = Path("/dev/shm/${IPC_NAME#/}")
deadline = time.time() + 30.0
while time.time() < deadline:
    if path.exists():
        raise SystemExit(0)
    time.sleep(0.05)
raise SystemExit("Timed out waiting for IPC ring creation")
PY

"$REPO_ROOT/SG-SLAM/cpp/semgraph_slam/build/apps/sgslam_ipc_runner" \
  --input-mode ipc \
  --dataset kitti \
  --ipc-name "$IPC_NAME" \
  --acquire-timeout-s 1 \
  --max-idle-timeouts 10 \
  --max-frames 10 \
  --result-path "$RUN_DIR/kitti_odometry_00.txt" \
  --pgo-result-path "$RUN_DIR/kitti_slam_00.txt" \
  --graph-map-path "$RUN_DIR/graph_map_00.txt" \
  --graph-edge-path "$RUN_DIR/graph_edge_00.txt" \
  --trace-path "$RUN_DIR/sgslam_consumer_trace.csv" \
  >"$CONSUMER_LOG" 2>&1 &
CONSUMER_PID=$!

wait "$PRODUCER_PID"
PRODUCER_RC=$?
wait "$CONSUMER_PID"
CONSUMER_RC=$?

trap - EXIT

{
  echo "producer_rc=$PRODUCER_RC"
  echo "consumer_rc=$CONSUMER_RC"
  echo "run_dir=$RUN_DIR"
} > "$RUN_DIR/run-summary.txt"

if [[ "$PRODUCER_RC" -ne 0 || "$CONSUMER_RC" -ne 0 ]]; then
  echo "Smoke test failed. See $RUN_DIR/logs" >&2
  exit 1
fi

echo "Smoke test completed: $RUN_DIR"
