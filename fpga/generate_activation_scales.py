#!/usr/bin/env python3
"""Phase 4a: Collect per-layer activation scales from the BN-folded ONNX.

Runs the fused FP32 model on real KITTI LiDAR frames and records the
absolute max (or a high percentile) of each layer's input and
post-activation tensors. Produces `activation_scales.json` used by the
AIE-ML requantization step.

Design notes:
- Uses the same layer boundary logic as generate_golden.py so layer indices
  and tensor identities line up with weights/manifest.json and the
  int8 weight files.
- Uses real KITTI frames with the same preprocessing the PyTorch model
  expects (5-channel projection, mean/std normalization, mask applied).
  This avoids the domain shift you'd get from random noise.
- Captures tensors in batches to bound RAM (same approach as
  generate_golden.py).
- "Output scale" per layer is measured at the point the *next* layer sees
  (post_residual if a residual add follows, otherwise post_activation,
  otherwise the raw conv output).

Output JSON format:
{
  "method": "max" | "percentile",
  "percentile": 99.99,
  "num_frames": 200,
  "layers": {
      "backbone.conv1": {
          "input_scale":  0.0234,
          "output_scale": 0.1812,
          "input_tensor":  "<onnx tensor name>",
          "output_tensor": "<onnx tensor name>",
          "input_max_abs":  2.97,
          "output_max_abs": 23.0,
          "index": 0
      },
      ...
  }
}

Symmetric INT8 scale convention: scale = max_abs / 127

Usage:
  python fpga/generate_activation_scales.py \
      --scan-root /workspace/data/kitti/sequences \
      --sequence 00 \
      --arch-cfg /workspace/data/pretrained_darknet53_weights/arch_cfg.yaml \
      --num-frames 200 \
      --output fpga/activation_scales.json
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
import yaml

# ---------------------------------------------------------------------------
# Paths and imports (reuse RangeNet's LaserScan for preprocessing)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent            # fpga/
ROOT_DIR = SCRIPT_DIR.parent                            # repo root
RANGENET_TRAIN = ROOT_DIR / "RangeNet" / "train"

sys.path.insert(0, str(RANGENET_TRAIN))
sys.path.insert(0, str(RANGENET_TRAIN / "tasks" / "semantic"))

from common.laserscan import LaserScan  # noqa: E402

DEFAULT_FUSED_ONNX = SCRIPT_DIR / "model_fused.onnx"
DEFAULT_MANIFEST = SCRIPT_DIR / "weights" / "manifest.json"


# ---------------------------------------------------------------------------
# Layer boundary discovery (mirrors generate_golden.py)
# ---------------------------------------------------------------------------

def find_layer_boundaries(model, manifest_layers):
    """For each conv/deconv node, find its input tensor name and the tensor
    that represents the "next layer's input" (post-residual > post-activation
    > conv output)."""
    node_list = list(model.graph.node)
    input_to_consumers = {}
    for node in node_list:
        for inp in node.input:
            input_to_consumers.setdefault(inp, []).append(node)

    conv_nodes = [n for n in node_list if n.op_type in ("Conv", "ConvTranspose")]
    if len(conv_nodes) != len(manifest_layers):
        raise RuntimeError(
            f"ONNX conv count ({len(conv_nodes)}) != manifest layer count "
            f"({len(manifest_layers)}). Is model_fused.onnx stale?"
        )

    boundaries = []
    for conv, layer in zip(conv_nodes, manifest_layers):
        post_act = None
        post_res = None
        for c in input_to_consumers.get(conv.output[0], []):
            if c.op_type == "LeakyRelu":
                post_act = c.output[0]
                for rc in input_to_consumers.get(c.output[0], []):
                    if rc.op_type == "Add":
                        post_res = rc.output[0]
                break

        out_tensor = post_res or post_act or conv.output[0]
        boundaries.append({
            "index": layer["index"],
            "name": layer["name"],
            "op": layer["op"],
            "input_tensor": conv.input[0],
            "output_tensor": out_tensor,
        })
    return boundaries


def make_model_with_extra_outputs(model, extra_names):
    existing = {o.name for o in model.graph.output}
    extra = [
        helper.make_tensor_value_info(n, TensorProto.FLOAT, None)
        for n in extra_names if n not in existing
    ]
    graph = helper.make_graph(
        list(model.graph.node), model.graph.name,
        list(model.graph.input),
        list(model.graph.output) + extra,
        initializer=list(model.graph.initializer),
    )
    m = helper.make_model(graph, opset_imports=model.opset_import)
    m.ir_version = model.ir_version
    return m


# ---------------------------------------------------------------------------
# KITTI preprocessing (must match what the PyTorch model expects)
# ---------------------------------------------------------------------------

def load_arch_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def preprocess_scan(scan_path, sensor_cfg):
    """Open a .bin scan and produce the 5-channel normalized input tensor.

    Matches the preprocessing in fpga/vitis/quantize_vitisai.py.
    Returns a float32 numpy array shaped [1, 5, H, W].
    """
    H = sensor_cfg["img_prop"]["height"]
    W = sensor_cfg["img_prop"]["width"]

    scan = LaserScan(
        project=True, H=H, W=W,
        fov_up=sensor_cfg["fov_up"],
        fov_down=sensor_cfg["fov_down"],
    )
    scan.open_scan(str(scan_path))

    proj_range = scan.proj_range.astype(np.float32)             # [H, W]
    proj_xyz = scan.proj_xyz.astype(np.float32)                 # [H, W, 3]
    proj_rem = scan.proj_remission.astype(np.float32)           # [H, W]
    proj_mask = scan.proj_mask.astype(np.float32)               # [H, W]

    # Stack to [5, H, W]: range, x, y, z, remission
    stacked = np.concatenate([
        proj_range[None, :, :],
        np.transpose(proj_xyz, (2, 0, 1)),
        proj_rem[None, :, :],
    ], axis=0)

    means = np.array(sensor_cfg["img_means"], dtype=np.float32)[:, None, None]
    stds = np.array(sensor_cfg["img_stds"], dtype=np.float32)[:, None, None]
    stacked = (stacked - means) / stds
    stacked = stacked * proj_mask[None, :, :]

    return stacked[None, :, :, :]  # [1, 5, H, W]


# ---------------------------------------------------------------------------
# Statistics accumulators
# ---------------------------------------------------------------------------

class MaxAbsAccumulator:
    """Tracks running max(|x|) per tensor across all calibration frames."""
    def __init__(self, names):
        self.names = set(names)
        self.max_abs = {n: 0.0 for n in names}

    def update(self, name, arr):
        if name not in self.names:
            return
        m = float(np.max(np.abs(arr))) if arr.size else 0.0
        if m > self.max_abs[name]:
            self.max_abs[name] = m

    def finalize(self, _percentile=None):
        return {n: self.max_abs[n] for n in self.max_abs}


class PercentileAccumulator:
    """Tracks a reservoir of abs-values per tensor, then returns the requested
    percentile. Uses reservoir sampling to cap memory."""
    def __init__(self, names, reservoir=200_000, rng_seed=0):
        self.names = set(names)
        self.reservoir_size = reservoir
        self.samples = {n: [] for n in names}
        self.counts = {n: 0 for n in names}
        self.rng = np.random.RandomState(rng_seed)

    def update(self, name, arr):
        if name not in self.names:
            return
        flat = np.abs(arr.reshape(-1))
        n_new = flat.size
        if n_new == 0:
            return

        cur = self.samples[name]
        seen = self.counts[name]
        R = self.reservoir_size

        if len(cur) < R:
            take = min(R - len(cur), n_new)
            cur.extend(flat[:take].tolist())
            remaining = flat[take:]
            seen += take
        else:
            remaining = flat

        # Reservoir sampling for the rest
        for v in remaining:
            seen += 1
            j = self.rng.randint(0, seen)
            if j < R:
                cur[j] = float(v)

        self.samples[name] = cur
        self.counts[name] = seen

    def finalize(self, percentile=99.99):
        out = {}
        for n, s in self.samples.items():
            if not s:
                out[n] = 0.0
            else:
                out[n] = float(np.percentile(np.asarray(s), percentile))
        return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect per-layer activation scales for INT8 AIE-ML pipeline"
    )
    parser.add_argument("--fused-onnx", default=str(DEFAULT_FUSED_ONNX),
                        help="Path to BN-folded ONNX (from export_weights.py)")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                        help="Path to weights/manifest.json")
    parser.add_argument("--scan-root", required=True,
                        help="KITTI sequences root, e.g. /workspace/data/kitti/sequences")
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--arch-cfg", required=True,
                        help="Path to arch_cfg.yaml (defines sensor dims + normalization)")
    parser.add_argument("--num-frames", type=int, default=200,
                        help="Number of KITTI frames for calibration (100-500 typical)")
    parser.add_argument("--method", choices=["max", "percentile"], default="max",
                        help="Scale-setting method (percentile is more robust to outliers)")
    parser.add_argument("--percentile", type=float, default=99.99,
                        help="Percentile to use when --method=percentile")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="How many extra tensors to capture per ORT session (RAM tradeoff)")
    parser.add_argument("--output", default=str(SCRIPT_DIR / "activation_scales.json"))
    parser.add_argument("--tmp-dir", default=str(SCRIPT_DIR / "_tmp_actscale"))
    args = parser.parse_args()

    # ---- Load config / model / manifest
    arch_cfg = load_arch_cfg(args.arch_cfg)
    sensor_cfg = arch_cfg["dataset"]["sensor"]

    with open(args.manifest) as f:
        manifest = json.load(f)

    print(f"Loading ONNX: {args.fused_onnx}")
    model = onnx.load(args.fused_onnx)

    boundaries = find_layer_boundaries(model, manifest["layers"])
    print(f"  {len(boundaries)} conv/deconv layers")

    # All unique tensor names we need to capture
    tensor_names = []
    seen = set()
    for b in boundaries:
        for t in (b["input_tensor"], b["output_tensor"]):
            if t not in seen:
                tensor_names.append(t)
                seen.add(t)
    print(f"  {len(tensor_names)} unique tensors to track")

    # ---- Collect KITTI scan paths
    scan_dir = Path(args.scan_root) / args.sequence / "velodyne"
    scan_paths = sorted(scan_dir.glob("*.bin"))
    if not scan_paths:
        raise FileNotFoundError(f"No .bin files in {scan_dir}")
    scan_paths = scan_paths[: args.num_frames]
    print(f"  {len(scan_paths)} calibration frames from {scan_dir}")

    # ---- Pick accumulator
    if args.method == "max":
        acc = MaxAbsAccumulator(tensor_names)
    else:
        acc = PercentileAccumulator(tensor_names)
    print(f"  Scale method: {args.method}" +
          (f" (p={args.percentile})" if args.method == "percentile" else ""))

    # ---- Pre-build the batched "extra-output" ONNX files once, reuse across frames
    os.makedirs(args.tmp_dir, exist_ok=True)

    existing_outputs = {o.name for o in model.graph.output}
    extras = [n for n in tensor_names if n not in existing_outputs]
    batches = [extras[i:i + args.batch_size] for i in range(0, len(extras), args.batch_size)]
    n_batches = len(batches) + 1  # +1 for the graph's own outputs

    batch_sess = []
    for bi, batch in enumerate(batches):
        m = make_model_with_extra_outputs(model, batch)
        p = os.path.join(args.tmp_dir, f"_b{bi}.onnx")
        onnx.save(m, p)
        del m
        sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        batch_sess.append((sess, batch, p))
    sess_final = ort.InferenceSession(args.fused_onnx, providers=["CPUExecutionProvider"])
    final_names = [o.name for o in sess_final.get_outputs()]

    # ---- Iterate calibration frames
    print("\nRunning calibration ...")
    input_name = batch_sess[0][0].get_inputs()[0].name if batch_sess else sess_final.get_inputs()[0].name

    for fi, scan_path in enumerate(scan_paths):
        x = preprocess_scan(scan_path, sensor_cfg)

        for sess, batch, _ in batch_sess:
            # Only request names the session actually exposes (some may be intermediate outputs only)
            avail = {o.name for o in sess.get_outputs()}
            req = [n for n in batch if n in avail]
            if not req:
                continue
            outs = sess.run(req, {input_name: x})
            for n, arr in zip(req, outs):
                acc.update(n, arr)
            del outs
            gc.collect()

        # Final outputs (present in the graph already)
        outs = sess_final.run(final_names, {input_name: x})
        for n, arr in zip(final_names, outs):
            acc.update(n, arr)  # no-op if name wasn't tracked
        del outs
        gc.collect()

        if (fi + 1) % 10 == 0 or (fi + 1) == len(scan_paths):
            print(f"  Frame {fi + 1}/{len(scan_paths)}")

    # ---- Clean up temp ONNXs
    for _, _, p in batch_sess:
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(args.tmp_dir)
    except OSError:
        pass

    # ---- Compute scales and write JSON
    stats = acc.finalize(args.percentile)

    layers_out = {}
    for b in boundaries:
        in_t = b["input_tensor"]
        out_t = b["output_tensor"]
        in_max = stats.get(in_t, 0.0)
        out_max = stats.get(out_t, 0.0)
        layers_out[b["name"]] = {
            "index": b["index"],
            "op": b["op"],
            "input_tensor": in_t,
            "output_tensor": out_t,
            "input_max_abs": in_max,
            "output_max_abs": out_max,
            "input_scale": (in_max / 127.0) if in_max > 0 else 0.0,
            "output_scale": (out_max / 127.0) if out_max > 0 else 0.0,
        }

    result = {
        "source_onnx": os.path.basename(args.fused_onnx),
        "method": args.method,
        "percentile": args.percentile if args.method == "percentile" else None,
        "num_frames": len(scan_paths),
        "scale_convention": "symmetric_int8: scale = max_abs / 127",
        "layers": layers_out,
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nWrote {args.output}")

    # ---- Sanity summary
    n_zero = sum(1 for v in layers_out.values()
                 if v["input_scale"] == 0.0 or v["output_scale"] == 0.0)
    if n_zero:
        print(f"  WARNING: {n_zero} layer(s) have zero scale — inspect input preprocessing")

    print("\nPer-layer summary (first 10):")
    for name, v in list(layers_out.items())[:10]:
        print(f"  [{v['index']:2d}] {name:<40s} "
              f"s_in={v['input_scale']:.5f}  s_out={v['output_scale']:.5f}")


if __name__ == "__main__":
    main()
