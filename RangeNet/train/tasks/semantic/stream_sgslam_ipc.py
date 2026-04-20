#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np

import __init__ as booger  # noqa: F401, sets up TRAIN_PATH for local imports
from export_sgslam_labels import load_model, load_yaml, predict_scan, resolve_device


REPO_ROOT = Path(__file__).resolve().parents[4]
RNSG_PYTHON_ROOT = REPO_ROOT / "rangenet_sgslam_ipc" / "python"
if str(RNSG_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(RNSG_PYTHON_ROOT))

from rnsg_ipc.core import (  # noqa: E402
    DEFAULT_CAPACITY_POINTS,
    DEFAULT_SLOT_COUNT,
    FLAG_RAW_SEMANTICKITTI_LABELS,
    Producer,
    unlink,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stream KITTI LiDAR scans through GPU RangeNet into the RangeNet -> "
            "SG-SLAM shared-memory ring."
        )
    )
    parser.add_argument(
        "--scan-root",
        required=True,
        help="Parent directory containing KITTI sequences such as <root>/00/velodyne.",
    )
    parser.add_argument(
        "--sequence",
        default="00",
        help="Sequence id to stream. Default: 00",
    )
    parser.add_argument(
        "--scan-subdir",
        default="velodyne",
        help="Scan subdirectory under the sequence. Default: velodyne",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model directory containing arch_cfg.yaml, data_cfg.yaml, and weights.",
    )
    parser.add_argument(
        "--ipc-name",
        default="/rnsg_kitti_00",
        help="POSIX shared-memory ring name. Default: /rnsg_kitti_00",
    )
    parser.add_argument(
        "--slot-count",
        type=int,
        default=DEFAULT_SLOT_COUNT,
        help=f"IPC ring slot count. Default: {DEFAULT_SLOT_COUNT}",
    )
    parser.add_argument(
        "--capacity-points",
        type=int,
        default=DEFAULT_CAPACITY_POINTS,
        help=f"Max points per frame in the ring. Default: {DEFAULT_CAPACITY_POINTS}",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=10.0,
        help="Playback rate in Hz. Default: 10.0",
    )
    parser.add_argument(
        "--max-scans",
        type=int,
        default=None,
        help="Optional cap for smoke runs.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda"],
        default="auto",
        help="Execution device. Default: auto",
    )
    parser.add_argument(
        "--trace-path",
        default=None,
        help="JSONL trace path. Default: <sequence_dir>/rangenet_ipc_trace.jsonl",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Run manifest path. Default: <sequence_dir>/rangenet_ipc_manifest.json",
    )
    parser.add_argument(
        "--unlink-existing",
        action="store_true",
        help="Unlink any pre-existing ring before creating a new one.",
    )
    return parser.parse_args()


def append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def monotonic_ns():
    return time.monotonic_ns()


def load_points(scan_path):
    points = np.fromfile(scan_path, dtype=np.float32)
    if points.size % 4 != 0:
        raise RuntimeError(f"Invalid KITTI scan shape in {scan_path}: {points.size} floats")
    return points.reshape((-1, 4))


def main():
    args = parse_args()

    scan_root = Path(args.scan_root).expanduser().resolve()
    model_dir = Path(args.model).expanduser().resolve()
    sequence_dir = scan_root / args.sequence / args.scan_subdir
    if not sequence_dir.is_dir():
        raise FileNotFoundError(f"Missing scan directory: {sequence_dir}")

    arch_cfg_path = model_dir / "arch_cfg.yaml"
    data_cfg_path = model_dir / "data_cfg.yaml"
    if not arch_cfg_path.is_file():
        raise FileNotFoundError(f"Missing arch config: {arch_cfg_path}")
    if not data_cfg_path.is_file():
        raise FileNotFoundError(f"Missing data config: {data_cfg_path}")

    trace_path = (
        Path(args.trace_path).expanduser().resolve()
        if args.trace_path
        else sequence_dir / "rangenet_ipc_trace.jsonl"
    )
    manifest_path = (
        Path(args.manifest_path).expanduser().resolve()
        if args.manifest_path
        else sequence_dir / "rangenet_ipc_manifest.json"
    )
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    arch_cfg = load_yaml(arch_cfg_path)
    data_cfg = load_yaml(data_cfg_path)
    learning_map_inv = data_cfg["learning_map_inv"]
    device = resolve_device(args.device)
    model, post = load_model(model_dir, arch_cfg, data_cfg, device)

    scan_paths = sorted(sequence_dir.glob("*.bin"))
    if args.max_scans is not None:
        scan_paths = scan_paths[: args.max_scans]
    if not scan_paths:
        raise ValueError(f"No .bin scans found in {sequence_dir}")

    if args.slot_count < 2:
        raise ValueError("--slot-count must be >= 2")
    if args.capacity_points < 1:
        raise ValueError("--capacity-points must be >= 1")
    if args.hz <= 0.0:
        raise ValueError("--hz must be > 0")

    if args.unlink_existing:
        unlink(args.ipc_name)

    producer = Producer.create(
        args.ipc_name,
        slot_count=args.slot_count,
        capacity_points=args.capacity_points,
    )

    sensor_cfg = arch_cfg["dataset"]["sensor"]
    period_ns = int(1e9 / args.hz)
    start_ns = monotonic_ns()

    manifest = {
        "created_at_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "scan_root": str(scan_root),
        "sequence": args.sequence,
        "scan_subdir": args.scan_subdir,
        "model_dir": str(model_dir),
        "arch_cfg_path": str(arch_cfg_path),
        "data_cfg_path": str(data_cfg_path),
        "device": str(device),
        "ipc_name": args.ipc_name,
        "slot_count": producer.slot_count,
        "capacity_points": producer.capacity_points,
        "requested_hz": args.hz,
        "scan_count": len(scan_paths),
        "trace_path": str(trace_path),
        "flag_raw_semantickitti_labels": True,
    }
    write_json(manifest_path, manifest)

    print(f"Streaming {len(scan_paths)} scans from {sequence_dir}")
    print(f"Writing trace to {trace_path}")
    print(
        f"IPC {args.ipc_name}: slots={producer.slot_count}, "
        f"capacity_points={producer.capacity_points}"
    )

    try:
        for frame_id, scan_path in enumerate(scan_paths):
            target_capture_ns = start_ns + frame_id * period_ns
            before_sleep_ns = monotonic_ns()
            if target_capture_ns > before_sleep_ns:
                time.sleep((target_capture_ns - before_sleep_ns) / 1e9)
            actual_capture_ns = monotonic_ns()

            load_points_start_ns = monotonic_ns()
            points = load_points(scan_path)
            load_points_ms = (monotonic_ns() - load_points_start_ns) / 1e6
            num_points = int(points.shape[0])
            if num_points > producer.capacity_points:
                raise RuntimeError(
                    f"Scan {scan_path} has {num_points} points, exceeds IPC capacity "
                    f"{producer.capacity_points}"
                )

            infer_start_ns = monotonic_ns()
            predicted_points, raw_labels = predict_scan(
                scan_path=scan_path,
                model=model,
                post=post,
                sensor_cfg=sensor_cfg,
                learning_map_inv=learning_map_inv,
                device=device,
            )
            infer_ms = (monotonic_ns() - infer_start_ns) / 1e6

            if predicted_points != num_points:
                raise RuntimeError(
                    f"Point count mismatch for {scan_path}: file has {num_points}, "
                    f"RangeNet returned {predicted_points}"
                )
            if raw_labels.shape[0] != num_points:
                raise RuntimeError(
                    f"Label count mismatch for {scan_path}: expected {num_points}, "
                    f"got {raw_labels.shape[0]}"
                )

            slot = producer.lease()
            copy_start_ns = monotonic_ns()
            slot.points[:num_points, :] = points
            slot.labels[:num_points] = raw_labels.astype(np.int32, copy=False)
            copy_ms = (monotonic_ns() - copy_start_ns) / 1e6

            publish_start_ns = monotonic_ns()
            producer.publish(
                num_points=num_points,
                capture_ns=actual_capture_ns,
                frame_id=frame_id,
                flags=FLAG_RAW_SEMANTICKITTI_LABELS,
            )
            publish_ms = (monotonic_ns() - publish_start_ns) / 1e6

            raw_unique = [int(value) for value in np.unique(raw_labels)]
            payload = {
                "timestamp_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "status": "published",
                "frame_id": frame_id,
                "scan_path": str(scan_path),
                "scan_name": scan_path.name,
                "num_points": num_points,
                "slot_idx": slot.slot_idx,
                "requested_capture_ns": target_capture_ns,
                "actual_capture_ns": actual_capture_ns,
                "playback_lag_ms": round((actual_capture_ns - target_capture_ns) / 1e6, 3),
                "load_points_ms": round(load_points_ms, 3),
                "infer_ms": round(infer_ms, 3),
                "copy_to_ring_ms": round(copy_ms, 3),
                "publish_ms": round(publish_ms, 3),
                "drop_count_after_publish": producer.drop_count,
                "head_seq_after_publish": producer.head_seq,
                "tail_seq_after_publish": producer.tail_seq,
                "device": str(device),
                "model_dir": str(model_dir),
                "raw_labels_present": raw_unique,
            }
            append_jsonl(trace_path, payload)
            print(
                f"[frame {frame_id:06d}] points={num_points} infer_ms={infer_ms:.2f} "
                f"publish_ms={publish_ms:.3f} drops={producer.drop_count}"
            )
    finally:
        producer.close()


if __name__ == "__main__":
    main()
