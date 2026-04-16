# SG-SLAM Registration FPGA Runbook (From Zero AWS Instances)

This guide gives a practical path to:
- bring up AWS infrastructure from scratch,
- run the same benchmarks you already use,
- move the registration kernel toward real FPGA execution.

Use this runbook in order. If you follow the command blocks exactly, you will get:
- persisted benchmark/profiling artifacts on disk,
- cpu/fpga-proto/fpga-xrt runs with the same command family,
- explicit checks that catch fpga-xrt fallback conditions.

## 0. Fast path checklist (recommended)

1. Launch `f1.2xlarge` with FPGA Developer AMI and an IAM role that can read your S3 bucket.
2. Clone repo and sync KITTI/labels from S3 into `SG-SLAM/data/...`.
3. Build `sg-slam:noetic` and run benchmark/profile commands from Section 5.
4. Put your `.xclbin` in `SG-SLAM/_aws_f1_results/xclbin/registration_accumulate_kernel.xclbin`.
5. Re-run fpga mode with `SGSLAM_REG_XCLBIN` set.
6. Run Section 5.7 checks to confirm no xrt fallback occurred.

## 1. What you can run today

Current branch supports two registration backends in SG-SLAM:
- cpu: existing TBB CPU path
- fpga: two execution modes behind one backend flag
  - fpga-proto: bounded HLS-style accumulation + CPU solve
  - fpga-xrt: real xclbin kernel invocation via XRT host adapter when SGSLAM_REG_XCLBIN is set and build enables XRT

This keeps benchmark parity across cpu, fpga-proto, and fpga-xrt without changing benchmark CLI shape.

## 2. AWS setup from scratch

### 2.1 Request F1 quota (required once)

In AWS Console:
1. Open Service Quotas.
2. EC2 quotas.
3. Find Running On-Demand F instances.
4. Request quota increase for at least f1.2xlarge.

Approval can take 1 to 2 business days.

### 2.2 Launch instances

Use FPGA Developer AMI (includes Vivado/Vitis/XRT setup scripts).

Recommended:
- c5.4xlarge for fast/cheap compile and software validation
- f1.2xlarge for FPGA hardware execution

Attach an IAM instance profile/role with S3 read permissions for your dataset bucket.

Minimum security group inbound:
- SSH (22) from your IP only

Create an EC2 key pair and download the pem file.

### 2.3 Connect

From your local machine:

ssh -i /path/to/key.pem ubuntu@YOUR_INSTANCE_PUBLIC_DNS

## 3. Bootstrap AWS instance

Run on AWS host:

sudo apt update
sudo apt install -y git git-lfs python3-venv python3-pip jq

git lfs install

Clone your repository and checkout branch:

git clone YOUR_REPO_URL fpga-slam
cd fpga-slam
git checkout fpga/registration-kernel-port

### 3.1 Access KITTI data from S3 on the F1 instance

Use this if your KITTI and label files are stored in S3 (for example the bucket shown in your screenshot).

Install AWS CLI if needed:

sudo apt install -y awscli

Verify credentials on the instance:

aws sts get-caller-identity

Set bucket variables (replace values if your key layout differs):

export SGSLAM_S3_BUCKET=sgslam-data-448792657895
export SGSLAM_S3_PREFIX=data
export SGSLAM_SEQ=00

Check bucket visibility:

aws s3 ls s3://$SGSLAM_S3_BUCKET/
aws s3 ls s3://$SGSLAM_S3_BUCKET/$SGSLAM_S3_PREFIX/ --recursive | head

Create local dataset folders expected by SG-SLAM launch files:

mkdir -p ~/fpga-slam/SG-SLAM/data/kitti/sequences/$SGSLAM_SEQ/
mkdir -p ~/fpga-slam/SG-SLAM/data/SegNet4D_predictions/kitti/$SGSLAM_SEQ/predictions/

Sync KITTI velodyne sequence:

aws s3 sync \
  s3://$SGSLAM_S3_BUCKET/$SGSLAM_S3_PREFIX/kitti/sequences/$SGSLAM_SEQ/ \
  ~/fpga-slam/SG-SLAM/data/kitti/sequences/$SGSLAM_SEQ/ \
  --no-progress

Sync SegNet4D prediction labels:

aws s3 sync \
  s3://$SGSLAM_S3_BUCKET/$SGSLAM_S3_PREFIX/SegNet4D_predictions/kitti/$SGSLAM_SEQ/predictions/ \
  ~/fpga-slam/SG-SLAM/data/SegNet4D_predictions/kitti/$SGSLAM_SEQ/predictions/ \
  --no-progress

Quick validation:

ls -lh ~/fpga-slam/SG-SLAM/data/kitti/sequences/$SGSLAM_SEQ/velodyne | head
ls -lh ~/fpga-slam/SG-SLAM/data/SegNet4D_predictions/kitti/$SGSLAM_SEQ/predictions | head

Optional one-command sync helper:

bash ~/fpga-slam/SG-SLAM/scripts/sync_kitti_from_s3.sh \
  --bucket $SGSLAM_S3_BUCKET \
  --prefix $SGSLAM_S3_PREFIX \
  --sequence $SGSLAM_SEQ

If the instance has no IAM role, configure credentials explicitly:

aws configure

Required minimum IAM permissions:
- s3:ListBucket on arn:aws:s3:::YOUR_BUCKET
- s3:GetObject on arn:aws:s3:::YOUR_BUCKET/*

## 4. Verify Xilinx tools on AWS

If using FPGA Developer AMI:

cd ~/aws-fpga
source vitis_setup.sh
which vivado
which vitis_hls
which v++

Also verify FPGA runtime utilities and accelerator visibility:

which xbutil || true
xbutil examine | head -n 40 || true

if command -v fpga-describe-local-image >/dev/null 2>&1; then
  sudo fpga-describe-local-image -S 0 -R -H
fi

If these are missing, confirm you launched the FPGA Developer AMI.

## 5. Build and run SG-SLAM benchmarks (F1-ready, detailed)

All commands below start from:

cd ~/fpga-slam/SG-SLAM

### 5.1 Create persistent output folders on host

mkdir -p _aws_f1_results/{bench_results,profiling,logs,xclbin}

Optional: download xclbin from S3 into the expected path:

aws s3 cp \
  s3://YOUR_BUCKET/YOUR_PREFIX/registration_accumulate_kernel.xclbin \
  _aws_f1_results/xclbin/registration_accumulate_kernel.xclbin

### 5.2 Build Docker image

docker build --build-arg CATKIN_JOBS=4 -t sg-slam:noetic .

### 5.3 Start container (with persistent artifacts)

If you want to try fpga-xrt from inside container, pass FPGA device nodes when present:

FPGA_DEV_FLAGS="$(for d in /dev/xdma* /dev/xclmgmt*; do [ -e "$d" ] && printf -- '--device=%s ' "$d"; done)"

docker run --rm -it \
  $FPGA_DEV_FLAGS \
  -v "$(pwd):/opt/catkin_ws/src/SG-SLAM" \
  -v sg_slam_catkin_build:/opt/catkin_ws/build \
  -v sg_slam_catkin_devel:/opt/catkin_ws/devel \
  -v sg_slam_catkin_logs:/opt/catkin_ws/logs \
  -v "$(pwd)/_aws_f1_results:/opt/catkin_ws/results" \
  -v "$(pwd)/_aws_f1_results/xclbin:/opt/catkin_ws/xclbin" \
  sg-slam:noetic

Inside container, define shared paths first:

source /opt/ros/noetic/setup.bash
cd /opt/catkin_ws
export SGSLAM_RESULTS=/opt/catkin_ws/results
export SGSLAM_XCLBIN=/opt/catkin_ws/xclbin/registration_accumulate_kernel.xclbin

### 5.4 Build SG-SLAM with benchmark + XRT support

catkin config --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_SYSTEM_EIGEN3=ON \
  -DUSE_SYSTEM_TBB=ON \
  -DENABLE_MILESTONE1_BENCHMARKS=ON \
  -DENABLE_XRT_REGISTRATION=ON
catkin build --no-status -j4 -p1 semgraph_slam
source devel/setup.bash

REG_BENCH_BIN="$(find build devel -type f -name benchmark_registration | head -n 1)"

### 5.5 Run registration microbenchmark in cpu, fpga-proto, fpga-xrt

mkdir -p "$SGSLAM_RESULTS/bench_results/registration" "$SGSLAM_RESULTS/logs"

"$REG_BENCH_BIN" --backend cpu --frames 300 --warmup 30 --seed 570 \
  --max-correspondences 12000 --max-iters 500 --kernel 0.333333 --corr-dist 3.0 \
  2>&1 | tee "$SGSLAM_RESULTS/bench_results/registration/cpu.txt"

"$REG_BENCH_BIN" --backend fpga --frames 300 --warmup 30 --seed 570 \
  --max-correspondences 12000 --max-iters 500 --kernel 0.333333 --corr-dist 3.0 \
  2>&1 | tee "$SGSLAM_RESULTS/bench_results/registration/fpga_proto.txt"

if [ ! -f "$SGSLAM_XCLBIN" ]; then
  echo "Missing xclbin at $SGSLAM_XCLBIN"
  echo "Place registration_accumulate_kernel.xclbin there before fpga-xrt run."
else
  SGSLAM_REG_XCLBIN="$SGSLAM_XCLBIN" \
  "$REG_BENCH_BIN" --backend fpga --frames 300 --warmup 30 --seed 570 \
    --max-correspondences 12000 --max-iters 500 --kernel 0.333333 --corr-dist 3.0 \
    2>&1 | tee "$SGSLAM_RESULTS/bench_results/registration/fpga_xrt.txt"
fi

### 5.6 Run full pipeline profiling in cpu, fpga-proto, fpga-xrt

mkdir -p "$SGSLAM_RESULTS/profiling/base_cpu"
SGSLAM_REG_BACKEND=cpu \
SGSLAM_PIPELINE_PROFILE=1 \
SGSLAM_PIPELINE_PROFILE_DATASET=kitti \
SGSLAM_PIPELINE_PROFILE_OUT="$SGSLAM_RESULTS/profiling/base_cpu/slam_frontend_profile.csv" \
SGSLAM_ODOM_PROFILE_OUT="$SGSLAM_RESULTS/profiling/base_cpu/slam_odometry_profile.csv" \
SGSLAM_MAPPING_PROFILE_OUT="$SGSLAM_RESULTS/profiling/base_cpu/slam_mapping_profile.csv" \
roslaunch semgraph_slam semgraph_slam_kitti.launch \
  2>&1 | tee "$SGSLAM_RESULTS/logs/pipeline_cpu.log"

mkdir -p "$SGSLAM_RESULTS/profiling/fpga_proto"
SGSLAM_REG_BACKEND=fpga \
SGSLAM_PIPELINE_PROFILE=1 \
SGSLAM_PIPELINE_PROFILE_DATASET=kitti \
SGSLAM_PIPELINE_PROFILE_OUT="$SGSLAM_RESULTS/profiling/fpga_proto/slam_frontend_profile.csv" \
SGSLAM_ODOM_PROFILE_OUT="$SGSLAM_RESULTS/profiling/fpga_proto/slam_odometry_profile.csv" \
SGSLAM_MAPPING_PROFILE_OUT="$SGSLAM_RESULTS/profiling/fpga_proto/slam_mapping_profile.csv" \
roslaunch semgraph_slam semgraph_slam_kitti.launch \
  2>&1 | tee "$SGSLAM_RESULTS/logs/pipeline_fpga_proto.log"

if [ -f "$SGSLAM_XCLBIN" ]; then
  mkdir -p "$SGSLAM_RESULTS/profiling/fpga_xrt"
  SGSLAM_REG_BACKEND=fpga \
  SGSLAM_REG_XCLBIN="$SGSLAM_XCLBIN" \
  SGSLAM_PIPELINE_PROFILE=1 \
  SGSLAM_PIPELINE_PROFILE_DATASET=kitti \
  SGSLAM_PIPELINE_PROFILE_OUT="$SGSLAM_RESULTS/profiling/fpga_xrt/slam_frontend_profile.csv" \
  SGSLAM_ODOM_PROFILE_OUT="$SGSLAM_RESULTS/profiling/fpga_xrt/slam_odometry_profile.csv" \
  SGSLAM_MAPPING_PROFILE_OUT="$SGSLAM_RESULTS/profiling/fpga_xrt/slam_mapping_profile.csv" \
  roslaunch semgraph_slam semgraph_slam_kitti.launch \
    2>&1 | tee "$SGSLAM_RESULTS/logs/pipeline_fpga_xrt.log"
fi

### 5.7 Compare registration metrics and verify fpga-xrt did not fallback

for f in \
  "$SGSLAM_RESULTS/profiling/base_cpu/slam_frontend_profile.csv" \
  "$SGSLAM_RESULTS/profiling/fpga_proto/slam_frontend_profile.csv" \
  "$SGSLAM_RESULTS/profiling/fpga_xrt/slam_frontend_profile.csv"; do
  [ -f "$f" ] || continue
  awk -F, '
    NR==1 {for(i=1;i<=NF;i++) if($i=="registration_ms") c=i; next}
    $1 !~ /^#/ && c>0 {n++; sum+=$c}
    END {
      if (n>0) printf("%s registration_avg_ms=%.6f\n", FILENAME, sum/n);
      else printf("%s no rows\n", FILENAME);
    }
  ' "$f"
done

if [ -f "$SGSLAM_RESULTS/logs/pipeline_fpga_xrt.log" ]; then
  if grep -q "Falling back to fpga-proto path" "$SGSLAM_RESULTS/logs/pipeline_fpga_xrt.log"; then
    echo "WARNING: fpga-xrt fallback detected. Inspect pipeline_fpga_xrt.log"
  else
    echo "No fallback string detected in pipeline_fpga_xrt.log"
  fi
fi

### 5.8 Generate registration-specific CPU vs FPGA metrics

Run from `SG-SLAM` root (inside container):

python3 profiling/plot_registration_comparison.py \
  --cpu-dir "$SGSLAM_RESULTS/profiling/base_cpu" \
  --fpga-dir "$SGSLAM_RESULTS/profiling/fpga_proto" \
  --fpga-label fpga_proto \
  --output-dir "$SGSLAM_RESULTS/profiling/registration_compare_cpu_vs_fpga_proto" \
  --max-frames 5000

If fpga-xrt data exists, compare cpu vs fpga-xrt:

python3 profiling/plot_registration_comparison.py \
  --cpu-dir "$SGSLAM_RESULTS/profiling/base_cpu" \
  --fpga-dir "$SGSLAM_RESULTS/profiling/fpga_xrt" \
  --fpga-label fpga_xrt \
  --output-dir "$SGSLAM_RESULTS/profiling/registration_compare_cpu_vs_fpga_xrt" \
  --max-frames 5000

Outputs:
- registration_timeline_cpu_vs_fpga.png
- registration_hist_cpu_vs_fpga.png
- registration_scatter_cpu_vs_fpga.png
- registration_speedup_timeline.png
- registration_speedup_hist.png
- registration_compare_summary.json

## 6. HLS synthesis flow for the registration kernel file

Kernel file:
- SG-SLAM/cpp/semgraph_slam/hls/registration_kernel_draft.cpp

### 6.1 Create a minimal HLS TCL script

On AWS host:

cat > /tmp/run_registration_hls.tcl << 'EOF'
open_project registration_hls
set_top registration_accumulate_kernel
add_files ~/fpga-slam/SG-SLAM/cpp/semgraph_slam/hls/registration_kernel_draft.hpp
add_files ~/fpga-slam/SG-SLAM/cpp/semgraph_slam/hls/registration_kernel_draft.cpp -cflags "-DHLS_SYNTHESIS"
open_solution -reset solution1
set_part xcu9p-flgb2104-2L-e
create_clock -period 4
csynth_design
export_design -format ip_catalog
exit
EOF

### 6.2 Run HLS synthesis

vitis_hls -f /tmp/run_registration_hls.tcl

Important notes:
- The kernel now uses synthesis-guarded pragma macros, so normal GCC builds still pass.
- In HLS, pragmas are active when HLS synthesis defines synthesis macros.

## 7. Deploy API endpoints on AWS for kernel send/receive

API files:
- fpga/api/registration_api.py
- fpga/api/requirements.txt

Run service:

cd ~/fpga-slam/fpga/api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn registration_api:app --host 0.0.0.0 --port 8080

Quick test:

curl -sS http://127.0.0.1:8080/healthz
curl -sS http://127.0.0.1:8080/v1/registration/limits

curl -sS http://127.0.0.1:8080/v1/registration/accumulate \
  -H 'content-type: application/json' \
  -d '{
    "backend": "cpu-proto",
    "kernel": 0.333333,
    "correspondence_count": 2,
    "src_xyz": [1.0,2.0,3.0, 2.0,3.0,4.0],
    "tgt_xyz": [1.1,2.0,3.0, 1.9,3.1,4.0],
    "labels": [18, 9]
  }'

## 8. What is still needed for true F1 hardware execution

XRT adapter wiring is implemented in SG-SLAM frontend and activated by:
- build flag: ENABLE_XRT_REGISTRATION=ON
- runtime env: SGSLAM_REG_BACKEND=fpga and SGSLAM_REG_XCLBIN=/path/to/xclbin

Behavior:
1. If XRT is enabled and xclbin is valid, SG-SLAM invokes registration_accumulate_kernel on device.
2. If XRT setup fails at runtime, it falls back to fpga-proto and logs the reason.
3. benchmark_registration and pipeline profiling command shapes remain unchanged, so comparisons stay apples-to-apples.

Use Section 5.7 fallback checks to confirm whether fpga-xrt truly executed or silently downgraded to fpga-proto.

## 9. Recommended sequence

1. Get baseline CPU and fpga-proto benchmark numbers on c5.
2. Run HLS csynth and inspect initiation interval, latency, and DSP/BRAM usage.
3. Build SG-SLAM with ENABLE_XRT_REGISTRATION=ON.
4. Move to f1.2xlarge and validate fpga-xrt mode with SGSLAM_REG_XCLBIN set.
5. Re-run the exact same benchmark commands and profiling extraction commands above.

## 10. Troubleshooting on F1

### 10.1 fpga-xrt run prints fallback messages

Common causes:
- `SGSLAM_REG_XCLBIN` path is wrong or file is unreadable.
- Container cannot see FPGA device nodes.
- OpenCL platform/device is not visible from runtime environment.

Checks:

ls -lh "$SGSLAM_XCLBIN"
xbutil examine | head -n 40
grep -n "Falling back to fpga-proto path" "$SGSLAM_RESULTS/logs/pipeline_fpga_xrt.log"

### 10.2 `benchmark_registration` not found

Rebuild and discover binary path again:

cd /opt/catkin_ws
catkin build --no-status -j4 -p1 semgraph_slam
REG_BENCH_BIN="$(find build devel -type f -name benchmark_registration | head -n 1)"
echo "$REG_BENCH_BIN"

### 10.3 Profiling CSV files are empty or missing

Make sure these are set for each run:
- `SGSLAM_PIPELINE_PROFILE=1`
- `SGSLAM_PIPELINE_PROFILE_OUT=...`
- `SGSLAM_ODOM_PROFILE_OUT=...`
- `SGSLAM_MAPPING_PROFILE_OUT=...`

Then check writable destination:

ls -lah "$SGSLAM_RESULTS/profiling"
