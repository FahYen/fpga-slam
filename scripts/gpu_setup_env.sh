#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv-rangenet}"
GTSAM_ARCHIVE="${GTSAM_ARCHIVE:-$REPO_ROOT/scripts/gtsam-local-install.tar.gz}"
GTSAM_SOURCE_DIR="${GTSAM_SOURCE_DIR:-$REPO_ROOT/third_party/gtsam}"
GTSAM_BUILD_DIR="${GTSAM_BUILD_DIR:-$REPO_ROOT/third_party/gtsam/build-smoke-install}"
GTSAM_INSTALL_PREFIX="${GTSAM_INSTALL_PREFIX:-/usr/local}"

install_gtsam_from_source() {
  if [[ ! -d "$GTSAM_SOURCE_DIR" ]]; then
    echo "Missing GTSAM source directory: $GTSAM_SOURCE_DIR" >&2
    exit 1
  fi

  cmake -S "$GTSAM_SOURCE_DIR" -B "$GTSAM_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$GTSAM_INSTALL_PREFIX" \
    -DBUILD_SHARED_LIBS=ON \
    -DGTSAM_BUILD_TESTS=OFF \
    -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
    -DGTSAM_BUILD_TIMING_ALWAYS=OFF \
    -DGTSAM_BUILD_UNSTABLE=OFF \
    -DGTSAM_BUILD_PYTHON=OFF \
    -DGTSAM_INSTALL_MATLAB_TOOLBOX=OFF \
    -DGTSAM_WITH_TBB=ON \
    -DGTSAM_USE_SYSTEM_METIS=OFF

  cmake --build "$GTSAM_BUILD_DIR" -j4
  sudo cmake --install "$GTSAM_BUILD_DIR"
  sudo ldconfig
}

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  awscli \
  build-essential \
  cmake \
  git \
  libblas-dev \
  libboost-all-dev \
  libgflags-dev \
  libgoogle-glog-dev \
  liblapack-dev \
  libmetis-dev \
  libsuitesparse-dev \
  libtbb12 \
  libtbb-dev \
  libtbbmalloc2 \
  python3 \
  python3-pip \
  python3-venv

if [[ ! -f /usr/local/include/gtsam/geometry/Pose3.h ]]; then
  if [[ -f "$GTSAM_ARCHIVE" ]]; then
    sudo tar -C /usr/local -xzf "$GTSAM_ARCHIVE"
    sudo ldconfig
  else
    install_gtsam_from_source
  fi
fi

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python3" -m pip install --upgrade pip wheel setuptools
"$VENV_DIR/bin/python3" -m pip install "numpy<1.24" PyYAML scipy
"$VENV_DIR/bin/python3" -m pip install torch --index-url https://download.pytorch.org/whl/cu128

"$VENV_DIR/bin/python3" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

cmake -S "$REPO_ROOT/rangenet_sgslam_ipc" -B "$REPO_ROOT/rangenet_sgslam_ipc/build"
cmake --build "$REPO_ROOT/rangenet_sgslam_ipc/build" -j4

cmake -S "$REPO_ROOT/SG-SLAM/cpp/semgraph_slam" -B "$REPO_ROOT/SG-SLAM/cpp/semgraph_slam/build"
cmake --build "$REPO_ROOT/SG-SLAM/cpp/semgraph_slam/build" --target sgslam_ipc_runner -j4

echo "Environment setup completed."
