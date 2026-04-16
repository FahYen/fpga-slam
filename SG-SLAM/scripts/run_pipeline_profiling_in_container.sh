#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run SG-SLAM CPU and FPGA-proto pipeline profiling and generate plots.

This script is intended to run INSIDE the SG-SLAM Docker container.

Usage:
  run_pipeline_profiling_in_container.sh [options]

Options:
  --catkin-ws <path>        Catkin workspace path (default: /opt/catkin_ws)
  --results-dir <path>      Results output directory (default: /opt/catkin_ws/results)
  --dataset <name>          Dataset tag for profiling env var (default: kitti)
  --max-frames <n>          Max frames for plotting scripts (default: 5000)
  --max-run-frames <n>      Max frames per profiling run (0 = full dataset, default: 0)
  --jobs <n>                Build jobs for catkin build (default: 4)
  --skip-install-pydeps     Skip `python3 -m pip install -U pandas matplotlib`
  -h, --help                Show this help

Outputs:
  <results-dir>/profiling/base_cpu/*
  <results-dir>/profiling/fpga_proto/*
  <results-dir>/profiling/registration_compare_cpu_vs_fpga_proto/*
EOF
}

CATKIN_WS="/opt/catkin_ws"
RESULTS_DIR="/opt/catkin_ws/results"
DATASET="kitti"
MAX_FRAMES="5000"
MAX_RUN_FRAMES="0"
JOBS="4"
INSTALL_PYDEPS="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --catkin-ws)
      CATKIN_WS="$2"
      shift 2
      ;;
    --results-dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --max-frames)
      MAX_FRAMES="$2"
      shift 2
      ;;
    --max_frames)
      MAX_FRAMES="$2"
      shift 2
      ;;
    --max-run-frames)
      MAX_RUN_FRAMES="$2"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --skip-install-pydeps)
      INSTALL_PYDEPS="0"
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

if [[ ! -d "$CATKIN_WS" ]]; then
  echo "Error: catkin workspace not found: $CATKIN_WS" >&2
  exit 1
fi

if [[ ! -f "/opt/ros/noetic/setup.bash" ]]; then
  echo "Error: /opt/ros/noetic/setup.bash not found. Are you inside the noetic container?" >&2
  exit 1
fi

source /opt/ros/noetic/setup.bash

mkdir -p "$RESULTS_DIR"/profiling/base_cpu
mkdir -p "$RESULTS_DIR"/profiling/fpga_proto
mkdir -p "$RESULTS_DIR"/logs

cd "$CATKIN_WS"

catkin config --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_SYSTEM_EIGEN3=ON \
  -DUSE_SYSTEM_TBB=ON \
  -DENABLE_MILESTONE1_BENCHMARKS=ON \
  -DENABLE_XRT_REGISTRATION=OFF

catkin build --no-status -j"$JOBS" -p1 semgraph_slam
source "$CATKIN_WS/devel/setup.bash"

export SGSLAM_RESULTS="$RESULTS_DIR"

CPU_DIR="$SGSLAM_RESULTS/profiling/base_cpu"
FPGA_PROTO_DIR="$SGSLAM_RESULTS/profiling/fpga_proto"

run_profile_mode() {
  local mode="$1"
  local backend="$2"
  local out_dir="$3"
  local log_path="$4"

  echo "[run] Profiling ${mode} backend (SGSLAM_REG_BACKEND=${backend})"
  local frontend_csv="$out_dir/slam_frontend_profile.csv"
  local odom_csv="$out_dir/slam_odometry_profile.csv"
  local mapping_csv="$out_dir/slam_mapping_profile.csv"
  local default_frontend_csv="$CATKIN_WS/src/SG-SLAM/profiling/slam_frontend_profile.csv"
  local default_odom_csv="$CATKIN_WS/src/SG-SLAM/profiling/slam_odometry_profile.csv"
  local default_mapping_csv="$CATKIN_WS/src/SG-SLAM/profiling/slam_mapping_profile.csv"

  # Remove previous outputs to avoid mixing old and new runs.
  rm -f "$frontend_csv" "$odom_csv" "$mapping_csv"
  rm -f "$default_frontend_csv" "$default_odom_csv" "$default_mapping_csv"

  set +e
  SGSLAM_REG_BACKEND="$backend" \
  SGSLAM_PIPELINE_PROFILE=1 \
  SGSLAM_PIPELINE_PROFILE_DATASET="$DATASET" \
  SGSLAM_PIPELINE_PROFILE_OUT="$frontend_csv" \
  SGSLAM_ODOM_PROFILE_OUT="$odom_csv" \
  SGSLAM_MAPPING_PROFILE_OUT="$mapping_csv" \
  roslaunch semgraph_slam semgraph_slam_kitti.launch \
    pipeline_profile_enable:=1 \
    pipeline_profile_dataset:="$DATASET" \
    pipeline_profile_out:="$frontend_csv" \
    odom_profile_out:="$odom_csv" \
    mapping_profile_out:="$mapping_csv" \
    registration_backend:="$backend" >"$log_path" 2>&1 &
  local launch_pid=$!

  local stopped_by_cap=0
  if [[ "$MAX_RUN_FRAMES" -gt 0 ]]; then
    echo "[run] max-run-frames=${MAX_RUN_FRAMES} enabled for ${mode}"
    local last_frame=0
    local last_reported=-1
    local startup_deadline=$((SECONDS + 180))
    while kill -0 "$launch_pid" 2>/dev/null; do
      if [[ -f "$log_path" ]]; then
        local parsed_frame
        parsed_frame=$(awk -F'frame count:' '/frame count:/ {v=$2} END {if (v=="") {print 0; exit} gsub(/^[[:space:]]+/, "", v); sub(/[^0-9].*$/, "", v); if (v=="") print 0; else print v}' "$log_path" 2>/dev/null)
        if [[ "$parsed_frame" =~ ^[0-9]+$ ]]; then
          last_frame="$parsed_frame"
        fi

        if [[ "$last_frame" -ne "$last_reported" && "$last_frame" -gt 0 ]]; then
          echo "[run] ${mode} frame_count=${last_frame}"
          last_reported="$last_frame"
        fi

        if [[ "$last_frame" -ge "$MAX_RUN_FRAMES" ]]; then
          echo "[run] Reached frame cap for ${mode} at frame_count=${last_frame}; stopping roslaunch"
          kill -INT "$launch_pid" 2>/dev/null || true
          stopped_by_cap=1
          break
        fi
      fi

      if [[ "$last_frame" -eq 0 && "$SECONDS" -gt "$startup_deadline" ]]; then
        echo "[warn] ${mode} has not reported frame progress after 180s; check log: ${log_path}"
        echo "[warn] Continuing to wait for roslaunch process to finish..."
        startup_deadline=$((SECONDS + 180))
      fi
      sleep 1
    done
  fi

  wait "$launch_pid"
  local launch_status=$?
  set -e

  if [[ $launch_status -ne 0 ]]; then
    if [[ "$stopped_by_cap" -eq 1 && ( "$launch_status" -eq 130 || "$launch_status" -eq 143 ) ]]; then
      echo "[run] ${mode} stopped at frame cap (exit ${launch_status})"
    fi
    echo "[warn] roslaunch exited with code ${launch_status} for mode '${mode}'."
    echo "[warn] This can happen at end-of-data; validating CSV outputs before continuing."
  fi

  for csv in "$frontend_csv" "$odom_csv" "$mapping_csv"; do
    if [[ ! -f "$csv" ]]; then
      # Fallback for runs where launch defaults were used and CSVs ended up under SG-SLAM/profiling.
      if [[ "$csv" == "$frontend_csv" && -f "$default_frontend_csv" ]]; then
        cp "$default_frontend_csv" "$frontend_csv"
      elif [[ "$csv" == "$odom_csv" && -f "$default_odom_csv" ]]; then
        cp "$default_odom_csv" "$odom_csv"
      elif [[ "$csv" == "$mapping_csv" && -f "$default_mapping_csv" ]]; then
        cp "$default_mapping_csv" "$mapping_csv"
      fi
    fi

    if [[ ! -f "$csv" ]]; then
      echo "Error: expected profiling CSV not found: $csv" >&2
      echo "Error: also checked default location under $CATKIN_WS/src/SG-SLAM/profiling" >&2
      exit 1
    fi
    if [[ $(wc -l < "$csv") -le 1 ]]; then
      echo "Error: profiling CSV has no data rows: $csv" >&2
      exit 1
    fi
  done

  local frontend_rows
  frontend_rows=$(( $(wc -l < "$frontend_csv") - 1 ))
  local odom_rows
  odom_rows=$(( $(wc -l < "$odom_csv") - 1 ))
  local mapping_rows
  mapping_rows=$(( $(wc -l < "$mapping_csv") - 1 ))
  echo "[run] ${mode} rows: frontend=${frontend_rows}, odom=${odom_rows}, mapping=${mapping_rows}"

  if [[ "$MAX_RUN_FRAMES" -gt 0 ]]; then
    local warn_cap=$((MAX_RUN_FRAMES + 50))
    local hard_cap=$((MAX_RUN_FRAMES * 5))

    if [[ "$frontend_rows" -gt "$warn_cap" || "$odom_rows" -gt "$warn_cap" ]]; then
      echo "[warn] ${mode} row count is above requested max-run-frames (${MAX_RUN_FRAMES})."
      echo "[warn] This can happen due to shutdown latency after reaching frame cap."
      echo "[warn] frontend_rows=${frontend_rows}, odom_rows=${odom_rows}"
    fi

    if [[ "$frontend_rows" -gt "$hard_cap" || "$odom_rows" -gt "$hard_cap" ]]; then
      echo "Error: ${mode} row count exceeds max-run-frames cap (${MAX_RUN_FRAMES}) by an extreme margin." >&2
      echo "Error: This usually indicates stale/default-path CSVs were used unexpectedly." >&2
      exit 1
    fi
  fi

  echo "[run] ${mode} profiling CSVs captured successfully"
}

run_profile_mode "base_cpu" "cpu" "$CPU_DIR" "$SGSLAM_RESULTS/logs/pipeline_cpu.log"
run_profile_mode "fpga_proto" "fpga" "$FPGA_PROTO_DIR" "$SGSLAM_RESULTS/logs/pipeline_fpga_proto.log"

cd "$CATKIN_WS/src/SG-SLAM"

if [[ "$INSTALL_PYDEPS" == "1" ]]; then
  python3 -m pip install -U pandas matplotlib
fi

echo "[plots] Generating per-run report and important plots"
for mode in base_cpu fpga_proto; do
  IN_DIR="$SGSLAM_RESULTS/profiling/$mode"
  TMP_IN_DIR="$SGSLAM_RESULTS/profiling/${mode}_filtered_for_report"

  rm -rf "$IN_DIR/report_plots" "$IN_DIR/important_plots" "$TMP_IN_DIR"

  mkdir -p "$TMP_IN_DIR"
  cp "$IN_DIR/slam_frontend_profile.csv" "$TMP_IN_DIR/slam_frontend_profile.csv"
  cp "$IN_DIR/slam_odometry_profile.csv" "$TMP_IN_DIR/slam_odometry_profile.csv"

  # profile_report_plots.py reads mapping CSV fully; filter to had_frame==1 and cap rows for speed.
  if [[ "$MAX_FRAMES" -gt 0 ]]; then
    local_cap=$((MAX_FRAMES + 1))
    awk -F, -v cap="$local_cap" 'NR==1 || $2==1 {print; if (NR > 1) {count++; if (count >= cap-1) exit}}' \
      "$IN_DIR/slam_mapping_profile.csv" > "$TMP_IN_DIR/slam_mapping_profile.csv"
    awk -v cap="$local_cap" 'NR==1 || NR<=cap' "$IN_DIR/slam_frontend_profile.csv" > "$TMP_IN_DIR/slam_frontend_profile.csv"
    awk -v cap="$local_cap" 'NR==1 || NR<=cap' "$IN_DIR/slam_odometry_profile.csv" > "$TMP_IN_DIR/slam_odometry_profile.csv"
  else
    awk -F, 'NR==1 || $2==1' "$IN_DIR/slam_mapping_profile.csv" > "$TMP_IN_DIR/slam_mapping_profile.csv"
  fi

  python3 profile_report_plots.py \
    --input-dir "$TMP_IN_DIR" \
    --output-dir "$IN_DIR/report_plots" \
    --max-frames "$MAX_FRAMES"

  python3 profiling/plot_important_visualizations.py \
    --input-dir "$TMP_IN_DIR" \
    --output-dir "$IN_DIR/important_plots" \
    --max-frames "$MAX_FRAMES"
done

echo "[plots] Generating registration-specific CPU vs FPGA-proto comparison"
python3 profiling/plot_registration_comparison.py \
  --cpu-dir "$CPU_DIR" \
  --fpga-dir "$FPGA_PROTO_DIR" \
  --fpga-label fpga_proto \
  --output-dir "$SGSLAM_RESULTS/profiling/registration_compare_cpu_vs_fpga_proto" \
  --max-frames "$MAX_FRAMES"

echo "[done] Output root: $SGSLAM_RESULTS/profiling"
find "$SGSLAM_RESULTS/profiling" \( -name summary_stats.json -o -name important_summary.json -o -name registration_compare_summary.json \)
