#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE_PATH="${1:-$REPO_ROOT/scripts/gtsam-local-install.tar.gz}"

required_paths=(
  "/usr/local/include/gtsam"
  "/usr/local/lib/libgtsam.so"
  "/usr/local/lib/libgtsam.so.4"
  "/usr/local/lib/libgtsam.so.4.2.0"
  "/usr/local/lib/libmetis-gtsam.so"
  "/usr/local/lib/cmake/GTSAM"
)

for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required GTSAM install path: $path" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$ARCHIVE_PATH")"

tar -C /usr/local -czf "$ARCHIVE_PATH" \
  include/gtsam \
  lib/libgtsam.so \
  lib/libgtsam.so.4 \
  lib/libgtsam.so.4.2.0 \
  lib/libmetis-gtsam.so \
  lib/cmake/GTSAM

echo "Wrote $ARCHIVE_PATH"
