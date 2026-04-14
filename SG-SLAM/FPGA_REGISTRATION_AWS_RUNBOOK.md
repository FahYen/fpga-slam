# SG-SLAM Registration FPGA Runbook (From Zero AWS Instances)

This guide gives a practical path to:
- bring up AWS infrastructure from scratch,
- run the same benchmarks you already use,
- move the registration kernel toward real FPGA execution.

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

If these are missing, confirm you launched the FPGA Developer AMI.

## 5. Build and run SG-SLAM benchmarks (same benchmark family)

All commands below run from SG-SLAM directory:

cd ~/fpga-slam/SG-SLAM

### 5.1 Build Docker image

docker build --build-arg CATKIN_JOBS=4 -t sg-slam:noetic .

### 5.2 Start container

docker run --rm -it \
  -v "$(pwd):/opt/catkin_ws/src/SG-SLAM" \
  -v sg_slam_catkin_build:/opt/catkin_ws/build \
  -v sg_slam_catkin_devel:/opt/catkin_ws/devel \
  -v sg_slam_catkin_logs:/opt/catkin_ws/logs \
  sg-slam:noetic

Inside container:

source /opt/ros/noetic/setup.bash
cd /opt/catkin_ws
catkin config --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_SYSTEM_EIGEN3=ON \
  -DUSE_SYSTEM_TBB=ON \
  -DENABLE_MILESTONE1_BENCHMARKS=ON \
  -DENABLE_XRT_REGISTRATION=ON
catkin build --no-status -j4 -p1 semgraph_slam
source devel/setup.bash

### 5.3 Run registration microbenchmark in both modes

REG_BENCH_BIN="$(find build devel -type f -name benchmark_registration | head -n 1)"

mkdir -p /opt/catkin_ws/bench_results/registration

"$REG_BENCH_BIN" --backend cpu --frames 300 --warmup 30 --seed 570 \
  --max-correspondences 12000 --max-iters 500 --kernel 0.333333 --corr-dist 3.0 \
  | tee /opt/catkin_ws/bench_results/registration/cpu.txt

"$REG_BENCH_BIN" --backend fpga --frames 300 --warmup 30 --seed 570 \
  --max-correspondences 12000 --max-iters 500 --kernel 0.333333 --corr-dist 3.0 \
  | tee /opt/catkin_ws/bench_results/registration/fpga_proto.txt

FPGA-xrt run (same benchmark command, add xclbin env var):

SGSLAM_REG_XCLBIN=/path/to/registration_accumulate_kernel.xclbin \
"$REG_BENCH_BIN" --backend fpga --frames 300 --warmup 30 --seed 570 \
  --max-correspondences 12000 --max-iters 500 --kernel 0.333333 --corr-dist 3.0 \
  | tee /opt/catkin_ws/bench_results/registration/fpga_xrt.txt

### 5.4 Run full pipeline profiling in both modes

CPU run:

mkdir -p /opt/catkin_ws/profiling/base_cpu
SGSLAM_REG_BACKEND=cpu \
SGSLAM_PIPELINE_PROFILE=1 \
SGSLAM_PIPELINE_PROFILE_DATASET=kitti \
SGSLAM_PIPELINE_PROFILE_OUT=/opt/catkin_ws/profiling/base_cpu/slam_frontend_profile.csv \
SGSLAM_ODOM_PROFILE_OUT=/opt/catkin_ws/profiling/base_cpu/slam_odometry_profile.csv \
SGSLAM_MAPPING_PROFILE_OUT=/opt/catkin_ws/profiling/base_cpu/slam_mapping_profile.csv \
roslaunch semgraph_slam semgraph_slam_kitti.launch

FPGA-proto run:

mkdir -p /opt/catkin_ws/profiling/fpga_proto
SGSLAM_REG_BACKEND=fpga \
SGSLAM_PIPELINE_PROFILE=1 \
SGSLAM_PIPELINE_PROFILE_DATASET=kitti \
SGSLAM_PIPELINE_PROFILE_OUT=/opt/catkin_ws/profiling/fpga_proto/slam_frontend_profile.csv \
SGSLAM_ODOM_PROFILE_OUT=/opt/catkin_ws/profiling/fpga_proto/slam_odometry_profile.csv \
SGSLAM_MAPPING_PROFILE_OUT=/opt/catkin_ws/profiling/fpga_proto/slam_mapping_profile.csv \
roslaunch semgraph_slam semgraph_slam_kitti.launch

FPGA-xrt run (same profiling command shape, add xclbin env var):

mkdir -p /opt/catkin_ws/profiling/fpga_xrt
SGSLAM_REG_BACKEND=fpga \
SGSLAM_REG_XCLBIN=/path/to/registration_accumulate_kernel.xclbin \
SGSLAM_PIPELINE_PROFILE=1 \
SGSLAM_PIPELINE_PROFILE_DATASET=kitti \
SGSLAM_PIPELINE_PROFILE_OUT=/opt/catkin_ws/profiling/fpga_xrt/slam_frontend_profile.csv \
SGSLAM_ODOM_PROFILE_OUT=/opt/catkin_ws/profiling/fpga_xrt/slam_odometry_profile.csv \
SGSLAM_MAPPING_PROFILE_OUT=/opt/catkin_ws/profiling/fpga_xrt/slam_mapping_profile.csv \
roslaunch semgraph_slam semgraph_slam_kitti.launch

### 5.5 Compare registration metrics quickly

awk -F, 'BEGIN{n=0;sum=0} $1 !~ /^#/ && $1 != "frame_idx" {n++; sum+=$17} END {if(n>0) printf("cpu_registration_avg_ms=%.6f\n", sum/n); else print "no cpu rows"}' /opt/catkin_ws/profiling/base_cpu/slam_frontend_profile.csv

awk -F, 'BEGIN{n=0;sum=0} $1 !~ /^#/ && $1 != "frame_idx" {n++; sum+=$17} END {if(n>0) printf("fpga_proto_registration_avg_ms=%.6f\n", sum/n); else print "no fpga rows"}' /opt/catkin_ws/profiling/fpga_proto/slam_frontend_profile.csv

awk -F, 'BEGIN{n=0;sum=0} $1 !~ /^#/ && $1 != "frame_idx" {n++; sum+=$17} END {if(n>0) printf("fpga_xrt_registration_avg_ms=%.6f\n", sum/n); else print "no fpga-xrt rows"}' /opt/catkin_ws/profiling/fpga_xrt/slam_frontend_profile.csv

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

## 9. Recommended sequence

1. Get baseline CPU and fpga-proto benchmark numbers on c5.
2. Run HLS csynth and inspect initiation interval, latency, and DSP/BRAM usage.
3. Build SG-SLAM with ENABLE_XRT_REGISTRATION=ON.
4. Move to f1.2xlarge and validate fpga-xrt mode with SGSLAM_REG_XCLBIN set.
5. Re-run the exact same benchmark commands and profiling extraction commands above.
