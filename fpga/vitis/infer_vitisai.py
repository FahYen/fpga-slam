#!/usr/bin/env python3
"""Vitis AI Runtime (VART) inference for RangeNet DarkNet53 on DPU.

Runs on the target FPGA board (AWS F1 with DPUCADF8H overlay) where the
VART runtime is installed. Loads a compiled .xmodel and runs inference on
KITTI LiDAR scans, producing .label files compatible with SG-SLAM.

Usage (on F1 instance with DPU overlay loaded):
    python fpga/vitis/infer_vitisai.py \
        --xmodel /workspace/vitis_output/compiled/rangenet_darknet53.xmodel \
        --model /workspace/models/rangenet_darknet53 \
        --scan-root /workspace/data/kitti/sequences \
        --output-root /workspace/data/rangenet_vitisai/kitti \
        --sequence 00

Output:
    .label files in <output-root>/<sequence>/predictions/
    Format matches the RANGENET_SGSLAM_CONTRACT (little-endian int32, raw labels).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Resolve RangeNet import paths for LaserScan
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
FPGA_DIR = SCRIPT_DIR.parent
ROOT_DIR = FPGA_DIR.parent
RANGENET_TRAIN = ROOT_DIR / "RangeNet" / "train"

sys.path.insert(0, str(RANGENET_TRAIN))
sys.path.insert(0, str(RANGENET_TRAIN / "tasks" / "semantic"))

from common.laserscan import LaserScan


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# DPU runner helpers
# ---------------------------------------------------------------------------

def get_dpu_subgraph(graph):
    """Extract the DPU subgraph from a loaded XIR graph."""
    root = graph.get_root_subgraph()
    children = root.toposort_child_subgraph()
    dpu_subgraphs = [
        s for s in children
        if s.has_attr("device") and s.get_attr("device").upper() == "DPU"
    ]
    if not dpu_subgraphs:
        raise RuntimeError(
            "No DPU subgraph found in xmodel. Check that the model was "
            "compiled for the correct DPU target (DPUCADF8H for F1)."
        )
    return dpu_subgraphs[0]


def get_tensor_info(tensors):
    """Extract name, shape, and fixpos from VART tensor descriptors."""
    info = []
    for t in tensors:
        info.append({
            "name": t.name,
            "dims": tuple(t.dims),
            "fixpos": t.get_attr("fix_point") if t.has_attr("fix_point") else 0,
        })
    return info


# ---------------------------------------------------------------------------
# Pre/post processing
# ---------------------------------------------------------------------------

def build_projection(scan_path, sensor_cfg):
    """Load a KITTI scan and produce the normalized 5-channel range image."""
    H = sensor_cfg["img_prop"]["height"]
    W = sensor_cfg["img_prop"]["width"]

    scan = LaserScan(
        project=True, H=H, W=W,
        fov_up=sensor_cfg["fov_up"],
        fov_down=sensor_cfg["fov_down"],
    )
    scan.open_scan(str(scan_path))

    img_means = np.array(sensor_cfg["img_means"], dtype=np.float32)
    img_stds = np.array(sensor_cfg["img_stds"], dtype=np.float32)

    proj_range = scan.proj_range.copy()
    proj_xyz = scan.proj_xyz.copy()
    proj_remission = scan.proj_remission.copy()
    proj_mask = scan.proj_mask.astype(np.float32)

    # Stack into [5, H, W]
    proj = np.concatenate([
        proj_range[np.newaxis, :, :],
        proj_xyz.transpose(2, 0, 1),
        proj_remission[np.newaxis, :, :],
    ], axis=0)

    # Normalize
    proj = (proj - img_means[:, None, None]) / img_stds[:, None, None]
    proj = proj * proj_mask[np.newaxis, :, :]

    return proj, scan.proj_x.copy(), scan.proj_y.copy(), scan.size()


def quantize_input(proj_float, input_fixpos):
    """Convert float input to INT8 using the DPU's input fix-point."""
    scale = 2.0 ** input_fixpos
    proj_int8 = np.clip(np.round(proj_float * scale), -128, 127).astype(np.int8)
    return proj_int8


def dequantize_output(output_int8, output_fixpos):
    """Convert INT8 DPU output back to float for argmax."""
    scale = 2.0 ** (-output_fixpos)
    return output_int8.astype(np.float32) * scale


def output_to_labels(output_float, proj_x, proj_y, learning_map_inv):
    """Convert DPU logits to per-point raw semantic labels."""
    # output_float shape: [1, nclasses, H, W] or [nclasses, H, W]
    if output_float.ndim == 4:
        output_float = output_float[0]

    # Per-pixel argmax on range image
    proj_argmax = np.argmax(output_float, axis=0)  # [H, W]

    # Unproject to original point order
    unproj_argmax = proj_argmax[proj_y, proj_x]

    # Map reduced class IDs back to raw SemanticKITTI labels
    max_class = max(int(k) for k in learning_map_inv.keys())
    inv_lut = np.zeros(max_class + 100, dtype=np.int32)
    for k, v in learning_map_inv.items():
        inv_lut[int(k)] = int(v)

    raw_labels = inv_lut[unproj_argmax]
    return raw_labels.astype(np.int32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="VART inference for RangeNet on DPU"
    )
    parser.add_argument("--xmodel", required=True,
                        help="Path to compiled .xmodel")
    parser.add_argument("--model", required=True,
                        help="Path to model dir with arch_cfg.yaml / data_cfg.yaml")
    parser.add_argument("--scan-root", required=True,
                        help="Path to KITTI sequences root")
    parser.add_argument("--output-root", required=True,
                        help="Where to write .label files")
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--max-scans", type=int, default=None,
                        help="Limit number of scans (None = all)")
    parser.add_argument("--label-subdir", default="predictions")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load configs
    # ------------------------------------------------------------------
    model_dir = Path(args.model).resolve()
    arch_cfg = load_yaml(model_dir / "arch_cfg.yaml")
    data_cfg = load_yaml(model_dir / "data_cfg.yaml")
    sensor_cfg = arch_cfg["dataset"]["sensor"]
    learning_map_inv = data_cfg["learning_map_inv"]

    scan_dir = Path(args.scan_root).resolve() / args.sequence / "velodyne"
    scan_paths = sorted(scan_dir.glob("*.bin"))
    if args.max_scans:
        scan_paths = scan_paths[:args.max_scans]

    if not scan_paths:
        raise FileNotFoundError(f"No .bin files in {scan_dir}")

    out_dir = Path(args.output_root).resolve() / args.sequence / args.label_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"xmodel:      {args.xmodel}")
    print(f"Scans:       {len(scan_paths)} in {scan_dir}")
    print(f"Output:      {out_dir}")

    # ------------------------------------------------------------------
    # Import VART
    # ------------------------------------------------------------------
    try:
        import xir
        import vart
    except ImportError:
        print("\nERROR: xir/vart not found.")
        print("This script must be run on the FPGA target with Vitis AI Runtime installed.")
        print("Install VART packages or run inside the Vitis AI Docker on the target.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load xmodel and create DPU runner
    # ------------------------------------------------------------------
    print("\nLoading xmodel ...")
    graph = xir.Graph.deserialize(args.xmodel)
    dpu_subgraph = get_dpu_subgraph(graph)
    runner = vart.Runner.create_runner(dpu_subgraph, "run")

    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()

    input_info = get_tensor_info(input_tensors)
    output_info = get_tensor_info(output_tensors)

    print(f"  Input:  {input_info[0]['name']}  shape={input_info[0]['dims']}  fixpos={input_info[0]['fixpos']}")
    print(f"  Output: {output_info[0]['name']}  shape={output_info[0]['dims']}  fixpos={output_info[0]['fixpos']}")

    input_fixpos = input_info[0]["fixpos"]
    output_fixpos = output_info[0]["fixpos"]
    input_dims = input_info[0]["dims"]
    output_dims = output_info[0]["dims"]

    # DPU typically uses NHWC layout; detect from dims
    # RangeNet uses NCHW — we may need to transpose
    # input_dims from DPU: (1, H, W, C) if NHWC, or (1, C, H, W) if NCHW
    nhwc_input = len(input_dims) == 4 and input_dims[3] < input_dims[1]

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    print(f"\nRunning inference on {len(scan_paths)} scans ...")
    timings = []
    manifest_entries = []

    for idx, sp in enumerate(scan_paths):
        t0 = time.time()

        # Build normalized float projection [5, H, W]
        proj_float, proj_x, proj_y, npoints = build_projection(sp, sensor_cfg)

        # Prepare input buffer
        if nhwc_input:
            # Convert NCHW -> NHWC
            proj_nhwc = proj_float.transpose(1, 2, 0)  # [H, W, 5]
            input_data = quantize_input(proj_nhwc[np.newaxis], input_fixpos)
        else:
            input_data = quantize_input(proj_float[np.newaxis], input_fixpos)

        # Allocate output buffer
        output_data = np.empty(output_dims, dtype=np.int8)

        # Run DPU
        job_id = runner.execute_async([input_data], [output_data])
        runner.wait(job_id)

        # Dequantize output
        output_float = dequantize_output(output_data, output_fixpos)

        # Convert layout back to NCHW if needed
        if nhwc_input and output_float.ndim == 4:
            output_float = output_float.transpose(0, 3, 1, 2)  # NHWC -> NCHW

        # Post-process: argmax + unproject + label mapping
        raw_labels = output_to_labels(output_float, proj_x, proj_y, learning_map_inv)

        # Write .label file
        label_path = out_dir / sp.with_suffix(".label").name
        raw_labels.tofile(str(label_path))

        dt = time.time() - t0
        timings.append(dt)
        manifest_entries.append({
            "scan": sp.stem,
            "npoints": npoints,
            "label_file": label_path.name,
            "inference_time_s": round(dt, 4),
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == len(scan_paths):
            avg = sum(timings[-50:]) / len(timings[-50:])
            print(f"  {idx + 1}/{len(scan_paths)}  avg={avg:.3f}s/frame")

    # ------------------------------------------------------------------
    # Write manifest
    # ------------------------------------------------------------------
    manifest = {
        "xmodel": str(args.xmodel),
        "sequence": args.sequence,
        "num_scans": len(scan_paths),
        "total_time_s": round(sum(timings), 2),
        "avg_time_s": round(sum(timings) / len(timings), 4),
        "backend": "vitis-ai-dpu",
        "scans": manifest_entries,
    }
    manifest_path = out_dir / "vitisai_inference_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"  Scans processed: {len(scan_paths)}")
    print(f"  Total time:      {sum(timings):.2f}s")
    print(f"  Avg per frame:   {sum(timings)/len(timings):.4f}s")
    print(f"  Labels:          {out_dir}")
    print(f"  Manifest:        {manifest_path}")


if __name__ == "__main__":
    main()