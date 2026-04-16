#!/usr/bin/env python3
"""Generate registration-focused CPU vs FPGA comparison metrics and plots.

Inputs:
- CPU run frontend profile CSV: <cpu-dir>/slam_frontend_profile.csv
- FPGA run frontend profile CSV: <fpga-dir>/slam_frontend_profile.csv

Outputs:
- Registration comparison plots (timeline, histogram, scatter, speedup)
- registration_compare_summary.json with aggregate metrics

Usage examples:
  python3 profiling/plot_registration_comparison.py \
    --cpu-dir profiling/base_cpu \
    --fpga-dir profiling/fpga_proto \
    --output-dir profiling/registration_compare_cpu_vs_fpga_proto

  python3 profiling/plot_registration_comparison.py \
    --cpu-dir profiling/base_cpu \
    --fpga-dir profiling/fpga_xrt \
    --fpga-label fpga_xrt \
    --output-dir profiling/registration_compare_cpu_vs_fpga_xrt \
    --max-frames 5000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def save_fig(output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=220)
    plt.close()


def _load_registration(frontend_csv: Path, value_name: str) -> pd.DataFrame:
    if not frontend_csv.exists():
        raise FileNotFoundError(f"Missing frontend profile CSV: {frontend_csv}")

    df = pd.read_csv(frontend_csv, comment="#")
    required = {"frame_idx", "registration_ms"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{frontend_csv} missing required columns: {sorted(missing)}")

    df["frame_idx"] = pd.to_numeric(df["frame_idx"], errors="coerce")
    df["registration_ms"] = pd.to_numeric(df["registration_ms"], errors="coerce")
    df = df.dropna(subset=["frame_idx", "registration_ms"]).copy()
    df["frame_idx"] = df["frame_idx"].astype(int)
    df = df.sort_values("frame_idx").reset_index(drop=True)

    keep_cols = ["frame_idx", "registration_ms"]
    if "registration_backend" in df.columns:
        keep_cols.append("registration_backend")

    df = df[keep_cols].rename(columns={"registration_ms": value_name})
    return df


def _percentile(s: pd.Series, q: float) -> float:
    if s.empty:
        return float("nan")
    return float(s.quantile(q))


def build_summary(merged: pd.DataFrame,
                  cpu_source_rows: int,
                  fpga_source_rows: int,
                  cpu_dir: Path,
                  fpga_dir: Path,
                  cpu_label: str,
                  fpga_label: str) -> Dict[str, float]:
    cpu_col = f"{cpu_label}_ms"
    fpga_col = f"{fpga_label}_ms"
    speedup_col = "speedup_x"
    delta_col = "delta_ms"

    cpu = merged[cpu_col]
    fpga = merged[fpga_col]
    speedup = merged[speedup_col]
    delta = merged[delta_col]

    summary: Dict[str, float] = {
        "cpu_input_dir": str(cpu_dir.resolve()),
        "fpga_input_dir": str(fpga_dir.resolve()),
        "cpu_source_rows": int(cpu_source_rows),
        "fpga_source_rows": int(fpga_source_rows),
        "aligned_rows": int(len(merged)),
        f"{cpu_label}_avg_ms": float(cpu.mean()),
        f"{cpu_label}_median_ms": float(cpu.median()),
        f"{cpu_label}_p95_ms": _percentile(cpu, 0.95),
        f"{fpga_label}_avg_ms": float(fpga.mean()),
        f"{fpga_label}_median_ms": float(fpga.median()),
        f"{fpga_label}_p95_ms": _percentile(fpga, 0.95),
        "delta_avg_ms": float(delta.mean()),
        "delta_median_ms": float(delta.median()),
        "speedup_avg_x": float(speedup.mean()),
        "speedup_median_x": float(speedup.median()),
        "speedup_p95_x": _percentile(speedup, 0.95),
        "fpga_faster_frame_pct": float((fpga < cpu).mean() * 100.0),
    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare registration performance between CPU and FPGA runs")
    parser.add_argument("--cpu-dir", required=True, help="Directory containing CPU slam_frontend_profile.csv")
    parser.add_argument("--fpga-dir", required=True, help="Directory containing FPGA slam_frontend_profile.csv")
    parser.add_argument("--output-dir", default="profiling/registration_compare", help="Directory for generated plots and summary")
    parser.add_argument("--cpu-label", default="cpu", help="Label for CPU series in plots and summary keys")
    parser.add_argument("--fpga-label", default="fpga", help="Label for FPGA series in plots and summary keys")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on aligned rows to plot")
    parser.add_argument("--skip-initial-frames", type=int, default=0, help="Drop this many initial aligned frames before analysis")
    args = parser.parse_args()

    cpu_dir = Path(args.cpu_dir)
    fpga_dir = Path(args.fpga_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cpu_label = args.cpu_label.strip() or "cpu"
    fpga_label = args.fpga_label.strip() or "fpga"
    cpu_col = f"{cpu_label}_ms"
    fpga_col = f"{fpga_label}_ms"

    cpu_df = _load_registration(cpu_dir / "slam_frontend_profile.csv", cpu_col)
    fpga_df = _load_registration(fpga_dir / "slam_frontend_profile.csv", fpga_col)

    cpu_source_rows = len(cpu_df)
    fpga_source_rows = len(fpga_df)

    merged = pd.merge(cpu_df, fpga_df, on="frame_idx", how="inner")
    merged = merged.sort_values("frame_idx").reset_index(drop=True)
    merged = merged[(merged[cpu_col] > 0.0) & (merged[fpga_col] > 0.0)].copy()

    if args.skip_initial_frames > 0:
        merged = merged.iloc[args.skip_initial_frames :].copy()

    if args.max_frames is not None and args.max_frames > 0:
        merged = merged.head(args.max_frames).copy()

    if merged.empty:
        raise RuntimeError("No aligned, positive registration_ms rows found across CPU and FPGA runs.")

    merged["delta_ms"] = merged[cpu_col] - merged[fpga_col]
    merged["speedup_x"] = merged[cpu_col] / merged[fpga_col]

    # Registration timeline comparison.
    plt.figure(figsize=(12.0, 4.8))
    plt.plot(merged["frame_idx"], merged[cpu_col], linewidth=1.0, label=f"{cpu_label}_registration_ms")
    plt.plot(merged["frame_idx"], merged[fpga_col], linewidth=1.0, label=f"{fpga_label}_registration_ms")
    plt.title("Registration Runtime Per Frame: CPU vs FPGA")
    plt.xlabel("Frame Index")
    plt.ylabel("registration_ms")
    plt.grid(alpha=0.3)
    plt.legend()
    save_fig(output_dir, "registration_timeline_cpu_vs_fpga.png")

    # Registration time histogram.
    plt.figure(figsize=(10.0, 4.8))
    plt.hist(merged[cpu_col], bins=80, alpha=0.55, label=f"{cpu_label}_registration_ms")
    plt.hist(merged[fpga_col], bins=80, alpha=0.55, label=f"{fpga_label}_registration_ms")
    plt.title("Registration Time Distribution: CPU vs FPGA")
    plt.xlabel("registration_ms")
    plt.ylabel("count")
    plt.grid(alpha=0.25)
    plt.legend()
    save_fig(output_dir, "registration_hist_cpu_vs_fpga.png")

    # CPU vs FPGA scatter with y=x baseline.
    lo = min(float(merged[cpu_col].min()), float(merged[fpga_col].min()))
    hi = max(float(merged[cpu_col].max()), float(merged[fpga_col].max()))
    plt.figure(figsize=(6.4, 6.0))
    plt.scatter(merged[cpu_col], merged[fpga_col], s=10, alpha=0.55)
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
    plt.title("Registration Scatter: CPU vs FPGA")
    plt.xlabel(f"{cpu_label}_registration_ms")
    plt.ylabel(f"{fpga_label}_registration_ms")
    plt.grid(alpha=0.3)
    save_fig(output_dir, "registration_scatter_cpu_vs_fpga.png")

    # Speedup timeline.
    plt.figure(figsize=(12.0, 4.8))
    plt.plot(merged["frame_idx"], merged["speedup_x"], linewidth=1.0)
    plt.axhline(1.0, linestyle="--", linewidth=1.0)
    plt.title("Registration Speedup Per Frame (CPU/FPGA)")
    plt.xlabel("Frame Index")
    plt.ylabel("speedup_x")
    plt.grid(alpha=0.3)
    save_fig(output_dir, "registration_speedup_timeline.png")

    # Speedup histogram.
    plt.figure(figsize=(10.0, 4.8))
    plt.hist(merged["speedup_x"], bins=80, alpha=0.85)
    plt.axvline(1.0, linestyle="--", linewidth=1.0)
    plt.title("Registration Speedup Distribution (CPU/FPGA)")
    plt.xlabel("speedup_x")
    plt.ylabel("count")
    plt.grid(alpha=0.25)
    save_fig(output_dir, "registration_speedup_hist.png")

    summary = build_summary(
        merged=merged,
        cpu_source_rows=cpu_source_rows,
        fpga_source_rows=fpga_source_rows,
        cpu_dir=cpu_dir,
        fpga_dir=fpga_dir,
        cpu_label=cpu_label,
        fpga_label=fpga_label,
    )

    summary["max_frames"] = args.max_frames
    summary["skip_initial_frames"] = args.skip_initial_frames

    if "registration_backend_x" in merged.columns:
        summary["cpu_backend_values"] = sorted({str(v) for v in merged["registration_backend_x"].dropna().unique()})
    if "registration_backend_y" in merged.columns:
        summary["fpga_backend_values"] = sorted({str(v) for v in merged["registration_backend_y"].dropna().unique()})

    summary_path = output_dir / "registration_compare_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Generated registration comparison plots in: {output_dir}")
    print(f"Aligned rows: {len(merged)}")
    print(f"Summary file: {summary_path}")


if __name__ == "__main__":
    main()
