#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_BUCKET="${DATA_BUCKET:-s3://sgslam-data-448792657895}"

mkdir -p "$REPO_ROOT/data"

aws s3 sync "$DATA_BUCKET/data/kitti/" "$REPO_ROOT/data/kitti/"
aws s3 sync "$DATA_BUCKET/data/pretrained_darknet53_weights/" \
  "$REPO_ROOT/data/pretrained_darknet53_weights/"

echo "Synced KITTI seq 00 data and pretrained weights into $REPO_ROOT/data"
