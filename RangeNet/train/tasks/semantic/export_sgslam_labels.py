#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import datetime
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml

import __init__ as booger  # noqa: F401, sets up TRAIN_PATH for local imports
from common.laserscan import LaserScan
from tasks.semantic.dataset.kitti.parser import SemanticKitti
from tasks.semantic.modules.segmentator import Segmentator
from tasks.semantic.postproc.KNN import KNN


SG_SLAM_LABEL_MAP = {
    0: 0,
    1: 0,
    10: 1,
    11: 2,
    13: 5,
    15: 3,
    16: 5,
    18: 4,
    20: 5,
    30: 6,
    31: 7,
    32: 8,
    40: 9,
    44: 10,
    48: 11,
    49: 12,
    50: 13,
    51: 14,
    52: 0,
    60: 9,
    70: 15,
    71: 16,
    72: 17,
    80: 18,
    81: 19,
    99: 0,
    252: 20,
    253: 21,
    254: 22,
    255: 23,
    256: 24,
    257: 24,
    258: 25,
    259: 24,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run RangeNet on unlabeled scan folders and export SG-SLAM-compatible "
            ".label files plus a reproducibility manifest."
        )
    )
    parser.add_argument(
        "--scan-root",
        required=True,
        help="Parent directory that contains per-sequence scan folders such as 00/velodyne or 00/Ouster.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Parent directory where per-sequence label folders will be created.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model directory containing arch_cfg.yaml, data_cfg.yaml, and pretrained weights.",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Optional list of sequence IDs such as 00 08. Defaults to all numeric directories under --scan-root.",
    )
    parser.add_argument(
        "--scan-subdir",
        default="velodyne",
        help="Input scan subdirectory name inside each sequence. Default: velodyne",
    )
    parser.add_argument(
        "--label-subdir",
        default="predictions",
        help="Output label subdirectory name inside each sequence. Default: predictions",
    )
    parser.add_argument(
        "--max-scans-per-sequence",
        type=int,
        default=None,
        help="Optional cap for quick smoke runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .label files instead of skipping them.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda"],
        default="auto",
        help="Execution device. Default: auto",
    )
    parser.add_argument(
        "--manifest-name",
        default="sgslam_contract_manifest.json",
        help="Filename for the JSON manifest written under --output-root.",
    )
    parser.add_argument(
        "--trace-name",
        default="sgslam_export_trace.jsonl",
        help="Filename for the per-scan JSONL trace written under --output-root.",
    )
    return parser.parse_args()


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def discover_sequences(scan_root, requested_sequences):
    if requested_sequences:
        return sorted(requested_sequences)

    sequences = sorted(
        path.name for path in scan_root.iterdir() if path.is_dir() and path.name.isdigit()
    )
    if not sequences:
        raise ValueError(f"No numeric sequence folders found under {scan_root}")
    return sequences


def build_lut(mapping):
    max_key = max(int(key) for key in mapping.keys())
    lut = np.zeros(max_key + 100, dtype=np.int32)
    for key, value in mapping.items():
        lut[int(key)] = int(value)
    return lut


def resolve_device(device_flag):
    if device_flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_projection_tensors(scan, sensor_cfg):
    img_means = torch.tensor(sensor_cfg["img_means"], dtype=torch.float32)
    img_stds = torch.tensor(sensor_cfg["img_stds"], dtype=torch.float32)

    proj_range = torch.from_numpy(scan.proj_range).clone()
    proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
    proj_remission = torch.from_numpy(scan.proj_remission).clone()
    proj_mask = torch.from_numpy(scan.proj_mask)

    proj = torch.cat(
        [
            proj_range.unsqueeze(0).clone(),
            proj_xyz.clone().permute(2, 0, 1),
            proj_remission.unsqueeze(0).clone(),
        ]
    )
    proj = (proj - img_means[:, None, None]) / img_stds[:, None, None]
    proj = proj * proj_mask.float()

    proj_x = torch.from_numpy(scan.proj_x).long()
    proj_y = torch.from_numpy(scan.proj_y).long()
    unproj_range = torch.from_numpy(scan.unproj_range).clone()

    return proj, proj_mask, proj_x, proj_y, proj_range, unproj_range


def load_model(model_dir, arch_cfg, data_cfg, device):
    nclasses = len(data_cfg["learning_map_inv"])
    with torch.no_grad():
        model = Segmentator(arch_cfg, nclasses, str(model_dir))

    model = model.to(device)
    model.eval()

    post = None
    if arch_cfg["post"]["KNN"]["use"]:
        post = KNN(arch_cfg["post"]["KNN"]["params"], nclasses)
        if device.type == "cuda":
            post = post.cuda()
        post.eval()

    if device.type == "cuda":
        cudnn.benchmark = True
        cudnn.fastest = True

    return model, post


def predict_scan(scan_path, model, post, sensor_cfg, learning_map_inv, device):
    scan = LaserScan(
        project=True,
        H=sensor_cfg["img_prop"]["height"],
        W=sensor_cfg["img_prop"]["width"],
        fov_up=sensor_cfg["fov_up"],
        fov_down=sensor_cfg["fov_down"],
    )
    scan.open_scan(str(scan_path))

    proj, proj_mask, proj_x, proj_y, proj_range, unproj_range = build_projection_tensors(
        scan, sensor_cfg
    )

    with torch.no_grad():
        proj_in = proj.unsqueeze(0).to(device)
        proj_mask_in = proj_mask.unsqueeze(0).to(device)
        proj_output = model(proj_in, proj_mask_in)
        proj_argmax = proj_output[0].argmax(dim=0)

        proj_x = proj_x.to(device)
        proj_y = proj_y.to(device)
        if post:
            proj_range = proj_range.to(device)
            unproj_range = unproj_range.to(device)
            unproj_argmax = post(proj_range, unproj_range, proj_argmax, proj_x, proj_y)
        else:
            unproj_argmax = proj_argmax[proj_y, proj_x]

        if device.type == "cuda":
            torch.cuda.synchronize()

    pred_np = unproj_argmax.cpu().numpy().reshape((-1)).astype(np.int32)
    pred_np = SemanticKitti.map(pred_np, learning_map_inv).astype(np.int32)
    return scan.size(), pred_np


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def main():
    args = parse_args()

    scan_root = Path(args.scan_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    model_dir = Path(args.model).expanduser().resolve()

    arch_cfg_path = model_dir / "arch_cfg.yaml"
    data_cfg_path = model_dir / "data_cfg.yaml"
    if not arch_cfg_path.is_file():
        raise FileNotFoundError(f"Missing arch config: {arch_cfg_path}")
    if not data_cfg_path.is_file():
        raise FileNotFoundError(f"Missing data config: {data_cfg_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / args.manifest_name
    trace_path = output_root / args.trace_name

    arch_cfg = load_yaml(arch_cfg_path)
    data_cfg = load_yaml(data_cfg_path)
    learning_map_inv = data_cfg["learning_map_inv"]
    sgslam_lut = build_lut(SG_SLAM_LABEL_MAP)
    predicted_raw_labels = sorted(int(label) for label in learning_map_inv.values())

    sequences = discover_sequences(scan_root, args.sequences)
    device = resolve_device(args.device)
    model, post = load_model(model_dir, arch_cfg, data_cfg, device)

    manifest = {
        "created_at_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "device": str(device),
        "model_dir": str(model_dir),
        "arch_cfg_path": str(arch_cfg_path),
        "data_cfg_path": str(data_cfg_path),
        "scan_root": str(scan_root),
        "output_root": str(output_root),
        "scan_subdir": args.scan_subdir,
        "label_subdir": args.label_subdir,
        "layout_contract": {
            "scan_path_pattern": "<scan_root>/<sequence>/<scan_subdir>/<frame>.bin",
            "label_path_pattern": "<output_root>/<sequence>/<label_subdir>/<frame>.label",
            "sgslam_label_path_parameter": "<output_root>/<sequence>/<label_subdir>/",
        },
        "label_file_contract": {
            "dtype": "int32",
            "endianness": "little",
            "semantic_bits": "lower_16_bits",
            "instance_bits": "upper_16_bits_zero",
            "semantic_label_space": "original raw labels from data_cfg learning_map_inv",
        },
        "predicted_raw_labels_from_model": predicted_raw_labels,
        "sgslam_raw_to_reduced_label_map": SG_SLAM_LABEL_MAP,
        "sgslam_reduced_classes_of_interest": {
            "vehicles": [1, 4, 5],
            "trunk": [16],
            "pole": [18],
            "background": [9, 13, 14, 15],
            "dynamic_classes_filtered_later": [20, 21, 22, 23, 24, 25],
        },
        "overwrite_existing": args.overwrite,
        "max_scans_per_sequence": args.max_scans_per_sequence,
        "trace_path": str(trace_path),
        "sequences": {},
        "totals": {
            "processed_scans": 0,
            "skipped_existing": 0,
            "processed_points": 0,
        },
    }

    for sequence in sequences:
        scan_dir = scan_root / sequence / args.scan_subdir
        if not scan_dir.is_dir():
            raise FileNotFoundError(f"Missing scan directory: {scan_dir}")

        output_dir = output_root / sequence / args.label_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        scan_paths = sorted(scan_dir.glob("*.bin"))
        if args.max_scans_per_sequence is not None:
            scan_paths = scan_paths[: args.max_scans_per_sequence]

        seq_stats = {
            "scan_dir": str(scan_dir),
            "output_dir": str(output_dir),
            "processed_scans": 0,
            "skipped_existing": 0,
            "processed_points": 0,
            "observed_raw_labels": set(),
            "observed_sgslam_reduced_labels": set(),
        }

        if not scan_paths:
            raise ValueError(f"No .bin scans found in {scan_dir}")

        for scan_path in scan_paths:
            label_path = output_dir / scan_path.with_suffix(".label").name
            if label_path.exists() and not args.overwrite:
                seq_stats["skipped_existing"] += 1
                manifest["totals"]["skipped_existing"] += 1
                append_jsonl(
                    trace_path,
                    {
                        "timestamp_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        "sequence": sequence,
                        "scan": scan_path.name,
                        "status": "skipped_existing",
                        "label_path": str(label_path),
                    },
                )
                continue

            start_time = time.time()
            num_points, raw_labels = predict_scan(
                scan_path=scan_path,
                model=model,
                post=post,
                sensor_cfg=arch_cfg["dataset"]["sensor"],
                learning_map_inv=learning_map_inv,
                device=device,
            )
            latency_ms = (time.time() - start_time) * 1000.0

            if raw_labels.shape[0] != num_points:
                raise RuntimeError(
                    f"Point count mismatch for {scan_path}: {num_points} points vs {raw_labels.shape[0]} labels"
                )

            reduced_labels = sgslam_lut[raw_labels]
            raw_labels.astype("<i4").tofile(label_path)

            expected_size = int(num_points) * 4
            actual_size = label_path.stat().st_size
            if actual_size != expected_size:
                raise RuntimeError(
                    f"Label file size mismatch for {label_path}: expected {expected_size}, got {actual_size}"
                )

            raw_unique = [int(value) for value in np.unique(raw_labels)]
            reduced_unique = [int(value) for value in np.unique(reduced_labels)]

            seq_stats["processed_scans"] += 1
            seq_stats["processed_points"] += int(num_points)
            seq_stats["observed_raw_labels"].update(raw_unique)
            seq_stats["observed_sgslam_reduced_labels"].update(reduced_unique)

            manifest["totals"]["processed_scans"] += 1
            manifest["totals"]["processed_points"] += int(num_points)

            append_jsonl(
                trace_path,
                {
                    "timestamp_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "sequence": sequence,
                    "scan": scan_path.name,
                    "status": "written",
                    "num_points": int(num_points),
                    "latency_ms": round(latency_ms, 3),
                    "scan_path": str(scan_path),
                    "label_path": str(label_path),
                    "raw_labels_present": raw_unique,
                    "sgslam_reduced_labels_present": reduced_unique,
                },
            )

        seq_stats["observed_raw_labels"] = sorted(seq_stats["observed_raw_labels"])
        seq_stats["observed_sgslam_reduced_labels"] = sorted(
            seq_stats["observed_sgslam_reduced_labels"]
        )
        manifest["sequences"][sequence] = seq_stats

    write_json(manifest_path, manifest)
    print("Wrote manifest to", manifest_path)
    print("Appended per-scan trace to", trace_path)


if __name__ == "__main__":
    main()
