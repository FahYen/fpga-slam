#!/usr/bin/env python3
"""
Generate profiling visualizations for SG-SLAM pipeline CSV outputs.

Expected inputs (default: ./profiling):
- slam_frontend_profile.csv
- slam_odometry_profile.csv
- slam_mapping_profile.csv

Output (default: ./profiling/report_plots):
- Multiple PNG charts ready for report inclusion.
- summary_stats.json with quick numeric highlights.

Usage:
  python3 profile_report_plots.py
  python3 profile_report_plots.py --input-dir profiling --output-dir profiling/report_plots
  python3 profile_report_plots.py --max-frames 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def read_profile_csv(path: Path) -> pd.DataFrame:
    """Read CSV while skipping summary lines that start with '#'"""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, comment="#")


def ensure_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def save_fig(output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / name, dpi=200)
    plt.close()


def plot_frontend(frontend: pd.DataFrame, output_dir: Path, max_frames: int | None) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    if frontend.empty:
        return stats

    if "frame_idx" not in frontend.columns:
        frontend = frontend.copy()
        frontend["frame_idx"] = range(len(frontend))
    else:
        frontend = frontend.sort_values("frame_idx").reset_index(drop=True)

    if max_frames is not None and max_frames > 0:
        frontend = frontend.head(max_frames)

    time_cols = [
        "preprocess_ms",
        "voxelize_ms",
        "cluster_ms",
        "buildgraph_ms",
        "find_match_ms",
        "registration_ms",
        "relocalization_ms",
        "local_map_update_ms",
        "local_graph_update_ms",
        "total_ms",
    ]
    ensure_numeric(frontend, time_cols)

    # 1) Frontend total frame time over index
    if "total_ms" in frontend.columns:
        plt.figure(figsize=(11, 4.5))
        plt.plot(frontend["frame_idx"], frontend["total_ms"], linewidth=1.0)
        plt.title("Frontend Total Time Per Frame")
        plt.xlabel("Frame Index")
        plt.ylabel("Time (ms)")
        plt.grid(alpha=0.3)
        save_fig(output_dir, "frontend_total_ms_timeline.png")
        stats["frontend_avg_total_ms"] = float(frontend["total_ms"].mean())
        stats["frontend_p95_total_ms"] = float(frontend["total_ms"].quantile(0.95))

    # 2) Frontend stage percentages stacked area
    pct_cols = [
        "preprocess_pct",
        "voxelize_pct",
        "cluster_pct",
        "buildgraph_pct",
        "find_match_pct",
        "registration_pct",
        "relocalization_pct",
        "local_map_update_pct",
        "local_graph_update_pct",
    ]
    pct_cols = [c for c in pct_cols if c in frontend.columns]
    if pct_cols:
        ensure_numeric(frontend, pct_cols)
        plt.figure(figsize=(12, 5))
        x = frontend["frame_idx"].to_numpy()
        ys = [frontend[c].fillna(0).to_numpy() for c in pct_cols]
        plt.stackplot(x, ys, labels=pct_cols, alpha=0.85)
        plt.title("Frontend Stage Percentage Breakdown (Per Frame)")
        plt.xlabel("Frame Index")
        plt.ylabel("Percent of Frontend Frame Time")
        plt.legend(loc="upper right", fontsize=8, ncol=2)
        plt.grid(alpha=0.2)
        save_fig(output_dir, "frontend_stage_pct_stack.png")

    # 3) Average frontend stage times bar chart
    bar_cols = [
        "preprocess_ms",
        "voxelize_ms",
        "cluster_ms",
        "buildgraph_ms",
        "find_match_ms",
        "registration_ms",
        "relocalization_ms",
        "local_map_update_ms",
        "local_graph_update_ms",
    ]
    bar_cols = [c for c in bar_cols if c in frontend.columns]
    if bar_cols:
        means = frontend[bar_cols].mean().sort_values(ascending=False)
        plt.figure(figsize=(11, 5))
        means.plot(kind="bar")
        plt.title("Average Frontend Stage Time")
        plt.xlabel("Stage")
        plt.ylabel("Average Time (ms)")
        plt.grid(axis="y", alpha=0.3)
        save_fig(output_dir, "frontend_stage_avg_ms_bar.png")

        if "buildgraph_ms" in means.index:
            stats["frontend_buildgraph_avg_ms"] = float(means["buildgraph_ms"])
        if "registration_ms" in means.index:
            stats["frontend_registration_avg_ms"] = float(means["registration_ms"])

    # 4) BuildGraph vs registration timeline
    if "buildgraph_ms" in frontend.columns and "registration_ms" in frontend.columns:
        plt.figure(figsize=(11, 4.5))
        plt.plot(frontend["frame_idx"], frontend["buildgraph_ms"], label="buildgraph_ms", linewidth=1.0)
        plt.plot(frontend["frame_idx"], frontend["registration_ms"], label="registration_ms", linewidth=1.0)
        plt.title("BuildGraph vs Registration Time")
        plt.xlabel("Frame Index")
        plt.ylabel("Time (ms)")
        plt.legend()
        plt.grid(alpha=0.3)
        save_fig(output_dir, "frontend_buildgraph_vs_registration.png")

    return stats


def plot_odometry(odom: pd.DataFrame, output_dir: Path, max_frames: int | None) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    if odom.empty:
        return stats

    if "frame_idx" not in odom.columns:
        odom = odom.copy()
        odom["frame_idx"] = range(len(odom))
    else:
        odom = odom.sort_values("frame_idx").reset_index(drop=True)

    if max_frames is not None and max_frames > 0:
        odom = odom.head(max_frames)

    numeric_cols = [
        "wall_total_ms",
        "compute_total_ms",
        "frontend_ms",
        "load_ms",
        "viz_publish_ms",
        "frontend_pct",
        "load_pct",
        "viz_publish_pct",
    ]
    ensure_numeric(odom, numeric_cols)

    # 5) Odometry wall vs compute
    if "wall_total_ms" in odom.columns and "compute_total_ms" in odom.columns:
        plt.figure(figsize=(11, 4.5))
        plt.plot(odom["frame_idx"], odom["wall_total_ms"], label="wall_total_ms", linewidth=1.0)
        plt.plot(odom["frame_idx"], odom["compute_total_ms"], label="compute_total_ms", linewidth=1.0)
        plt.title("Odometry Loop Time: Wall vs Compute")
        plt.xlabel("Frame Index")
        plt.ylabel("Time (ms)")
        plt.legend()
        plt.grid(alpha=0.3)
        save_fig(output_dir, "odometry_wall_vs_compute.png")
        stats["odom_avg_wall_ms"] = float(odom["wall_total_ms"].mean())
        stats["odom_p95_wall_ms"] = float(odom["wall_total_ms"].quantile(0.95))

    # 6) Odometry average percentage breakdown
    pct_cols = [c for c in ["load_pct", "frontend_pct", "queue_push_pct", "tf_publish_pct", "odom_publish_pct", "save_pose_pct", "viz_publish_pct"] if c in odom.columns]
    if pct_cols:
        means = odom[pct_cols].mean().sort_values(ascending=False)
        plt.figure(figsize=(10, 5))
        means.plot(kind="bar")
        plt.title("Odometry Average Percentage Breakdown")
        plt.xlabel("Stage")
        plt.ylabel("Percent of Odometry Wall Time")
        plt.grid(axis="y", alpha=0.3)
        save_fig(output_dir, "odometry_avg_pct_bar.png")

    return stats


def plot_mapping(mapping: pd.DataFrame, output_dir: Path, max_frames: int | None) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    if mapping.empty:
        return stats

    if "frame_idx" not in mapping.columns:
        mapping = mapping.copy()
        mapping["frame_idx"] = range(len(mapping))
    else:
        mapping = mapping.sort_values("frame_idx").reset_index(drop=True)

    # Mapping logger emits rows even when queue is empty (had_frame=0).
    # Those idle iterations can dominate counts and distort event frequency/averages.
    if "had_frame" in mapping.columns:
        mapping = ensure_numeric(mapping, ["had_frame"])
        mapping = mapping[mapping["had_frame"] == 1].copy()
        mapping = mapping.reset_index(drop=True)

    if mapping.empty:
        return stats

    if max_frames is not None and max_frames > 0:
        mapping = mapping.head(max_frames)

    numeric_cols = [
        "loop_total_ms",
        "mapping_main_ms",
        "pgo_ms",
        "update_map_ms",
        "publish_map_ms",
        "gen_des_ms",
        "search_ms",
        "run_isam",
        "run_map_update",
    ]
    ensure_numeric(mapping, numeric_cols)

    # 7) Mapping loop total timeline
    if "loop_total_ms" in mapping.columns:
        plt.figure(figsize=(11, 4.5))
        plt.plot(mapping["frame_idx"], mapping["loop_total_ms"], linewidth=1.0)
        plt.title("Mapping Loop Total Time")
        plt.xlabel("Frame Index")
        plt.ylabel("Time (ms)")
        plt.grid(alpha=0.3)
        save_fig(output_dir, "mapping_loop_total_ms_timeline.png")
        stats["mapping_avg_loop_ms"] = float(mapping["loop_total_ms"].mean())
        stats["mapping_p95_loop_ms"] = float(mapping["loop_total_ms"].quantile(0.95))

    # 8) Mapping average stage times
    bar_cols = [c for c in ["mapping_main_ms", "pgo_ms", "update_map_ms", "publish_map_ms", "gen_des_ms", "search_ms"] if c in mapping.columns]
    if bar_cols:
        means = mapping[bar_cols].mean().sort_values(ascending=False)
        plt.figure(figsize=(10, 5))
        means.plot(kind="bar")
        plt.title("Mapping Average Stage Time")
        plt.xlabel("Stage")
        plt.ylabel("Average Time (ms)")
        plt.grid(axis="y", alpha=0.3)
        save_fig(output_dir, "mapping_stage_avg_ms_bar.png")

    # 9) Event frequency (iSAM / map update)
    event_cols = [c for c in ["run_isam", "run_map_update"] if c in mapping.columns]
    if event_cols:
        event_rates = mapping[event_cols].mean() * 100.0
        plt.figure(figsize=(7, 4.5))
        event_rates.plot(kind="bar")
        plt.title("Mapping Event Frequency")
        plt.xlabel("Event")
        plt.ylabel("Frames with Event (%)")
        plt.grid(axis="y", alpha=0.3)
        save_fig(output_dir, "mapping_event_frequency.png")

    return stats


def plot_cross(frontend: pd.DataFrame, odom: pd.DataFrame, output_dir: Path) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    if frontend.empty or odom.empty:
        return stats

    if "frame_idx" not in frontend.columns or "frame_idx" not in odom.columns:
        return stats

    if "total_ms" not in frontend.columns or "wall_total_ms" not in odom.columns:
        return stats

    merged = pd.merge(
        frontend[["frame_idx", "total_ms"]],
        odom[["frame_idx", "wall_total_ms"]],
        on="frame_idx",
        how="inner",
    )
    merged = merged.sort_values("frame_idx").reset_index(drop=True)
    if merged.empty:
        return stats

    ensure_numeric(merged, ["total_ms", "wall_total_ms"])

    # 10) Frontend vs odometry wall scatter
    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(merged["total_ms"], merged["wall_total_ms"], s=10, alpha=0.6)
    plt.title("Frontend Time vs Odometry Wall Time")
    plt.xlabel("Frontend total_ms")
    plt.ylabel("Odometry wall_total_ms")
    plt.grid(alpha=0.3)
    save_fig(output_dir, "cross_frontend_vs_odometry_scatter.png")

    corr = merged["total_ms"].corr(merged["wall_total_ms"])
    stats["frontend_odom_corr"] = float(corr) if pd.notna(corr) else float("nan")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SG-SLAM profiling plots")
    parser.add_argument("--input-dir", default="profiling", help="Directory containing profiling CSV files")
    parser.add_argument("--output-dir", default="profiling/report_plots", help="Directory to save generated plots")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on number of rows to plot")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frontend_path = input_dir / "slam_frontend_profile.csv"
    odom_path = input_dir / "slam_odometry_profile.csv"
    mapping_path = input_dir / "slam_mapping_profile.csv"

    frontend = read_profile_csv(frontend_path)
    odom = read_profile_csv(odom_path)
    mapping = read_profile_csv(mapping_path)

    stats: Dict[str, float] = {}
    stats.update(plot_frontend(frontend, output_dir, args.max_frames))
    stats.update(plot_odometry(odom, output_dir, args.max_frames))
    stats.update(plot_mapping(mapping, output_dir, args.max_frames))
    stats.update(plot_cross(frontend, odom, output_dir))

    metadata = {
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "frontend_rows": int(len(frontend)),
        "odometry_rows": int(len(odom)),
        "mapping_rows": int(len(mapping)),
        "stats": stats,
    }

    with open(output_dir / "summary_stats.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Generated report plots in:", output_dir)
    print("Frontend rows:", len(frontend), "Odometry rows:", len(odom), "Mapping rows:", len(mapping))
    print("Summary file:", output_dir / "summary_stats.json")


if __name__ == "__main__":
    main()
