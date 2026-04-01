#!/usr/bin/env python3
"""Generate high-signal profiling visualizations for SG-SLAM.

This script is optimized for SG-SLAM's large mapping profile CSV by reading
it in chunks and filtering to rows where had_frame == 1.

Inputs (defaults to ./profiling):
- slam_frontend_profile.csv
- slam_odometry_profile.csv
- slam_mapping_profile.csv

Outputs (default: ./profiling/important_plots):
- PNG figures with key timelines and bottleneck breakdowns
- important_summary.json with useful aggregates and percentiles

Usage examples:
  python3 profiling/plot_important_visualizations.py
  python3 profiling/plot_important_visualizations.py --max-frames 1000
  python3 profiling/plot_important_visualizations.py --input-dir profiling --output-dir profiling/important_plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


def ensure_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def read_small_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, comment="#")


def read_mapping_csv_chunked(path: Path, max_frames: Optional[int] = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    usecols = [
        "frame_idx",
        "had_frame",
        "run_isam",
        "run_map_update",
        "loop_detected",
        "dequeue_ms",
        "mapping_main_ms",
        "pgo_ms",
        "update_map_ms",
        "publish_map_ms",
        "loop_total_ms",
        "gen_des_ms",
        "search_ms",
    ]

    chunks: List[pd.DataFrame] = []
    for chunk in pd.read_csv(path, comment="#", usecols=usecols, chunksize=200000):
        ensure_numeric(chunk, ["had_frame", "frame_idx"])
        filtered = chunk[chunk["had_frame"] == 1].copy()
        if filtered.empty:
            continue
        chunks.append(filtered)

        if max_frames is not None and max_frames > 0:
            current_rows = sum(len(c) for c in chunks)
            if current_rows >= max_frames:
                break

    if not chunks:
        return pd.DataFrame(columns=usecols)

    mapping = pd.concat(chunks, ignore_index=True)
    mapping = mapping.sort_values("frame_idx").reset_index(drop=True)

    if max_frames is not None and max_frames > 0:
        mapping = mapping.head(max_frames)

    return mapping


def save_fig(output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=220)
    plt.close()


def plot_frontend(frontend: pd.DataFrame, output_dir: Path) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    if frontend.empty:
        return stats

    if "frame_idx" not in frontend.columns:
        frontend = frontend.copy()
        frontend["frame_idx"] = range(len(frontend))
    else:
        frontend = frontend.sort_values("frame_idx").reset_index(drop=True)

    time_cols = [
        "total_ms",
        "registration_ms",
        "buildgraph_ms",
        "cluster_ms",
        "voxelize_ms",
        "preprocess_ms",
        "find_match_ms",
        "local_map_update_ms",
        "local_graph_update_ms",
    ]
    ensure_numeric(frontend, time_cols)

    if "total_ms" in frontend.columns:
        plt.figure(figsize=(11, 4.5))
        plt.plot(frontend["frame_idx"], frontend["total_ms"], linewidth=1.0)
        plt.title("Frontend Total Runtime Per Frame")
        plt.xlabel("Frame Index")
        plt.ylabel("Time (ms)")
        plt.grid(alpha=0.3)
        save_fig(output_dir, "frontend_total_runtime_timeline.png")

        stats["frontend_avg_total_ms"] = float(frontend["total_ms"].mean())
        stats["frontend_p95_total_ms"] = float(frontend["total_ms"].quantile(0.95))

    important_stages = [
        c
        for c in [
            "registration_ms",
            "buildgraph_ms",
            "cluster_ms",
            "voxelize_ms",
            "preprocess_ms",
            "find_match_ms",
            "local_map_update_ms",
            "local_graph_update_ms",
        ]
        if c in frontend.columns
    ]
    if important_stages:
        means = frontend[important_stages].mean().sort_values(ascending=False)

        plt.figure(figsize=(10.5, 5.0))
        means.plot(kind="bar")
        plt.title("Frontend Bottlenecks: Average Stage Time")
        plt.xlabel("Stage")
        plt.ylabel("Average Time (ms)")
        plt.grid(axis="y", alpha=0.3)
        save_fig(output_dir, "frontend_bottleneck_bar.png")

        for k, v in means.items():
            stats[f"frontend_avg_{k}"] = float(v)

    if "buildgraph_ms" in frontend.columns and "registration_ms" in frontend.columns:
        plt.figure(figsize=(11, 4.5))
        plt.plot(frontend["frame_idx"], frontend["buildgraph_ms"], label="buildgraph_ms", linewidth=1.0)
        plt.plot(frontend["frame_idx"], frontend["registration_ms"], label="registration_ms", linewidth=1.0)
        plt.title("Frontend: BuildGraph vs Registration")
        plt.xlabel("Frame Index")
        plt.ylabel("Time (ms)")
        plt.legend()
        plt.grid(alpha=0.3)
        save_fig(output_dir, "frontend_buildgraph_vs_registration.png")

    return stats


def plot_odometry(odom: pd.DataFrame, output_dir: Path) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    if odom.empty:
        return stats

    if "frame_idx" not in odom.columns:
        odom = odom.copy()
        odom["frame_idx"] = range(len(odom))
    else:
        odom = odom.sort_values("frame_idx").reset_index(drop=True)

    ensure_numeric(
        odom,
        [
            "wall_total_ms",
            "compute_total_ms",
            "frontend_ms",
            "load_ms",
            "viz_publish_ms",
            "load_pct",
            "frontend_pct",
            "viz_publish_pct",
        ],
    )

    if "wall_total_ms" in odom.columns and "compute_total_ms" in odom.columns:
        plt.figure(figsize=(11, 4.5))
        plt.plot(odom["frame_idx"], odom["wall_total_ms"], label="wall_total_ms", linewidth=1.0)
        plt.plot(odom["frame_idx"], odom["compute_total_ms"], label="compute_total_ms", linewidth=1.0)
        plt.title("Odometry Loop Runtime: Wall vs Compute")
        plt.xlabel("Frame Index")
        plt.ylabel("Time (ms)")
        plt.legend()
        plt.grid(alpha=0.3)
        save_fig(output_dir, "odometry_wall_vs_compute.png")

        stats["odometry_avg_wall_ms"] = float(odom["wall_total_ms"].mean())
        stats["odometry_p95_wall_ms"] = float(odom["wall_total_ms"].quantile(0.95))

    pct_cols = [
        c
        for c in [
            "frontend_pct",
            "load_pct",
            "viz_publish_pct",
            "queue_push_pct",
            "save_pose_pct",
            "tf_publish_pct",
            "odom_publish_pct",
        ]
        if c in odom.columns
    ]
    if pct_cols:
        means = odom[pct_cols].mean().sort_values(ascending=False)
        plt.figure(figsize=(10.5, 5.0))
        means.plot(kind="bar")
        plt.title("Odometry Cost Share: Average Percentage")
        plt.xlabel("Stage")
        plt.ylabel("Percent of Wall Time")
        plt.grid(axis="y", alpha=0.3)
        save_fig(output_dir, "odometry_cost_share_bar.png")

        for k, v in means.items():
            stats[f"odometry_avg_{k}"] = float(v)

    return stats


def plot_mapping(mapping: pd.DataFrame, output_dir: Path) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    if mapping.empty:
        return stats

    ensure_numeric(
        mapping,
        [
            "run_isam",
            "run_map_update",
            "loop_detected",
            "loop_total_ms",
            "mapping_main_ms",
            "pgo_ms",
            "update_map_ms",
            "publish_map_ms",
            "gen_des_ms",
            "search_ms",
        ],
    )

    if "loop_total_ms" in mapping.columns:
        plt.figure(figsize=(11, 4.5))
        plt.plot(mapping["frame_idx"], mapping["loop_total_ms"], linewidth=1.0)
        plt.title("Mapping Runtime (had_frame == 1)")
        plt.xlabel("Frame Index")
        plt.ylabel("Time (ms)")
        plt.grid(alpha=0.3)
        save_fig(output_dir, "mapping_loop_runtime_timeline.png")

        stats["mapping_avg_loop_ms"] = float(mapping["loop_total_ms"].mean())
        stats["mapping_p95_loop_ms"] = float(mapping["loop_total_ms"].quantile(0.95))

    stage_cols = [
        c
        for c in [
            "mapping_main_ms",
            "pgo_ms",
            "update_map_ms",
            "publish_map_ms",
            "gen_des_ms",
            "search_ms",
        ]
        if c in mapping.columns
    ]
    if stage_cols:
        means = mapping[stage_cols].mean().sort_values(ascending=False)
        plt.figure(figsize=(10.0, 5.0))
        means.plot(kind="bar")
        plt.title("Mapping Bottlenecks: Average Stage Time")
        plt.xlabel("Stage")
        plt.ylabel("Average Time (ms)")
        plt.grid(axis="y", alpha=0.3)
        save_fig(output_dir, "mapping_bottleneck_bar.png")

        for k, v in means.items():
            stats[f"mapping_avg_{k}"] = float(v)

    event_cols = [c for c in ["run_isam", "run_map_update", "loop_detected"] if c in mapping.columns]
    if event_cols:
        rates = (mapping[event_cols].mean() * 100.0).sort_values(ascending=False)
        plt.figure(figsize=(8.0, 4.8))
        rates.plot(kind="bar")
        plt.title("Mapping Event Frequency")
        plt.xlabel("Event")
        plt.ylabel("Frames with Event (%)")
        plt.grid(axis="y", alpha=0.3)
        save_fig(output_dir, "mapping_event_frequency.png")

        for k, v in rates.items():
            stats[f"mapping_event_rate_{k}_pct"] = float(v)

    return stats


def plot_cross(frontend: pd.DataFrame, odom: pd.DataFrame, output_dir: Path) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    if frontend.empty or odom.empty:
        return stats

    required_frontend = {"frame_idx", "total_ms"}
    required_odom = {"frame_idx", "wall_total_ms"}
    if not required_frontend.issubset(frontend.columns) or not required_odom.issubset(odom.columns):
        return stats

    merged = pd.merge(
        frontend[["frame_idx", "total_ms"]],
        odom[["frame_idx", "wall_total_ms"]],
        on="frame_idx",
        how="inner",
    )
    if merged.empty:
        return stats

    ensure_numeric(merged, ["total_ms", "wall_total_ms"])

    plt.figure(figsize=(6.6, 5.5))
    plt.scatter(merged["total_ms"], merged["wall_total_ms"], s=12, alpha=0.6)
    plt.title("Frontend vs Odometry Runtime Correlation")
    plt.xlabel("Frontend total_ms")
    plt.ylabel("Odometry wall_total_ms")
    plt.grid(alpha=0.3)
    save_fig(output_dir, "cross_frontend_vs_odometry_scatter.png")

    corr = merged["total_ms"].corr(merged["wall_total_ms"])
    if pd.notna(corr):
        stats["cross_frontend_odom_corr"] = float(corr)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate important SG-SLAM profiling visualizations")
    parser.add_argument("--input-dir", default="profiling", help="Directory containing profiling CSV files")
    parser.add_argument("--output-dir", default="profiling/important_plots", help="Directory for generated plots")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on frames used in plots")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frontend_path = input_dir / "slam_frontend_profile.csv"
    odom_path = input_dir / "slam_odometry_profile.csv"
    mapping_path = input_dir / "slam_mapping_profile.csv"

    frontend = read_small_csv(frontend_path)
    odom = read_small_csv(odom_path)

    if args.max_frames is not None and args.max_frames > 0:
        if not frontend.empty and "frame_idx" in frontend.columns:
            frontend = frontend.sort_values("frame_idx").head(args.max_frames).reset_index(drop=True)
        if not odom.empty and "frame_idx" in odom.columns:
            odom = odom.sort_values("frame_idx").head(args.max_frames).reset_index(drop=True)

    mapping = read_mapping_csv_chunked(mapping_path, max_frames=args.max_frames)

    stats: Dict[str, float] = {}
    stats.update(plot_frontend(frontend, output_dir))
    stats.update(plot_odometry(odom, output_dir))
    stats.update(plot_mapping(mapping, output_dir))
    stats.update(plot_cross(frontend, odom, output_dir))

    metadata = {
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "frontend_rows": int(len(frontend)),
        "odometry_rows": int(len(odom)),
        "mapping_had_frame_rows": int(len(mapping)),
        "max_frames": args.max_frames,
        "stats": stats,
    }

    summary_path = output_dir / "important_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated important profiling plots in: {output_dir}")
    print(f"Frontend rows: {len(frontend)}, Odometry rows: {len(odom)}, Mapping had_frame rows: {len(mapping)}")
    print(f"Summary file: {summary_path}")


if __name__ == "__main__":
    main()
