#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Sync KITTI + SegNet4D predictions from S3 into SG-SLAM expected local paths.

Usage:
  sync_kitti_from_s3.sh --bucket <bucket> [--prefix data] [--sequence 00] [--root <sg-slam-root>] [--dry-run]

Examples:
  sync_kitti_from_s3.sh --bucket sgslam-data-448792657895 --prefix data --sequence 00
  sync_kitti_from_s3.sh --bucket my-bucket --prefix datasets --sequence 00 --dry-run
EOF
}

BUCKET=""
PREFIX="data"
SEQUENCE="00"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bucket)
      BUCKET="$2"
      shift 2
      ;;
    --prefix)
      PREFIX="$2"
      shift 2
      ;;
    --sequence)
      SEQUENCE="$2"
      shift 2
      ;;
    --root)
      ROOT_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$BUCKET" ]]; then
  echo "Error: --bucket is required" >&2
  usage
  exit 1
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "Error: aws CLI is not installed. Install awscli first." >&2
  exit 1
fi

aws sts get-caller-identity >/dev/null

KITTI_DST="$ROOT_DIR/data/kitti/sequences/$SEQUENCE"
LABEL_DST="$ROOT_DIR/data/SegNet4D_predictions/kitti/$SEQUENCE/predictions"
KITTI_SRC="s3://$BUCKET/$PREFIX/kitti/sequences/$SEQUENCE/"
LABEL_SRC="s3://$BUCKET/$PREFIX/SegNet4D_predictions/kitti/$SEQUENCE/predictions/"

mkdir -p "$KITTI_DST" "$LABEL_DST"

SYNC_ARGS=(--no-progress)
if [[ "$DRY_RUN" -eq 1 ]]; then
  SYNC_ARGS+=(--dryrun)
fi

echo "[sync] KITTI source: $KITTI_SRC"
echo "[sync] KITTI destination: $KITTI_DST"
aws s3 sync "$KITTI_SRC" "$KITTI_DST" "${SYNC_ARGS[@]}"

echo "[sync] Label source: $LABEL_SRC"
echo "[sync] Label destination: $LABEL_DST"
aws s3 sync "$LABEL_SRC" "$LABEL_DST" "${SYNC_ARGS[@]}"

echo "[done] Sync completed for sequence $SEQUENCE"
