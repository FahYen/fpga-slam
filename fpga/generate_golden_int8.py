#!/usr/bin/env python3
"""Phase 4c: Generate bit-exact INT8 golden vectors for HLS kernel validation.

Unlike generate_golden.py (which produces FP32 reference activations from random
inputs), this script simulates the EXACT integer math your AIE-ML conv kernel
will execute, so each .bin file can be used as a byte-level unit test.

For each layer and each real KITTI frame, writes:
    layer_XX_input.i8.bin             INT8 input to the conv (quantized via s_in)
    layer_XX_bias.i32.bin             Bias converted into accumulator-scale INT32
    layer_XX_acc.i32.bin              Raw INT32 accumulator (MAC + bias, before requant)
    layer_XX_pre_sat.i32.bin          After requant multiply-shift, before saturation
    layer_XX_output.i8.bin            Final INT8 output (after LeakyReLU + saturate)
    layer_XX_ref.f32.bin              FP32 reference activation for sanity comparison
    manifest.json                     Shapes, dtypes, scales, layer ops

Design choice: ISOLATED mode.
  Each layer's INT8 input is derived by quantizing that layer's *true* FP32 input
  captured from ONNX. This means a bug in layer N's kernel won't break layer N+1's
  test. Ideal for unit testing a single HLS kernel.

Integer math details (must match your conv_kernel.cpp exactly):
  1) x_i8 = clip(round(x_fp / s_in), -128, 127)
  2) acc = sum_{ic,kh,kw}  x_i8 * w_i8                        (int32)
  3) b_i32[c] = round(b_fp[c] / (s_in * w_scale[c]))
  4) acc += b_i32[c]
  5) prod = (int64) acc * int_mult[c]
     scaled = (prod + (1 << (shift[c]-1))) >> shift[c]        (round-to-nearest)
  6) If LeakyRelu (alpha=0.1): x < 0 -> (x * 13) >> 7         (matches kernel)
  7) out_i8 = saturate(scaled, -128, 127)

Usage:
  python fpga/generate_golden_int8.py \
      --scan-root data/kitti/sequences --sequence 00 \
      --arch-cfg data/pretrained_darknet53_weights/arch_cfg.yaml \
      --num-frames 3 \
      --output-dir fpga/golden_int8
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto
import yaml

# PyTorch is only needed for the exact conv/conv_transpose dataflow simulation.
# We use float64 (53-bit mantissa) so int32 MAC results are represented exactly.
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths & RangeNet import (for LaserScan preprocessing)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
RANGENET_TRAIN = ROOT_DIR / "RangeNet" / "train"
sys.path.insert(0, str(RANGENET_TRAIN))
sys.path.insert(0, str(RANGENET_TRAIN / "tasks" / "semantic"))
from common.laserscan import LaserScan  # noqa: E402

DEFAULT_FUSED_ONNX = SCRIPT_DIR / "model_fused.onnx"
DEFAULT_FP32_MANIFEST = SCRIPT_DIR / "weights" / "manifest.json"        # has bias files
DEFAULT_INT8_MANIFEST = SCRIPT_DIR / "weights" / "int8" / "manifest.json"
DEFAULT_REQUANT_MANIFEST = SCRIPT_DIR / "weights" / "requant" / "manifest.json"
DEFAULT_ACT_SCALES = SCRIPT_DIR / "activation_scales.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "golden_int8"

# LeakyReLU coefficient (must match kernel exactly)
LEAKY_NUM = 13
LEAKY_SHIFT = 7   # (x * 13) >> 7  ≈  x * 0.1015625


# ---------------------------------------------------------------------------
# ONNX helpers (reused / adapted from generate_golden.py)
# ---------------------------------------------------------------------------

def make_model_with_outputs(model, extra_names):
    existing = {o.name for o in model.graph.output}
    extra = [helper.make_tensor_value_info(n, TensorProto.FLOAT, None)
             for n in extra_names if n not in existing]
    graph = helper.make_graph(
        list(model.graph.node), model.graph.name,
        list(model.graph.input),
        list(model.graph.output) + extra,
        initializer=list(model.graph.initializer),
    )
    m = helper.make_model(graph, opset_imports=model.opset_import)
    m.ir_version = model.ir_version
    return m


def get_conv_attrs(node):
    """Extract Conv / ConvTranspose attributes in a normalized form."""
    a = {attr.name: attr for attr in node.attribute}
    def ints(name, default):
        return list(a[name].ints) if name in a else list(default)
    def i(name, default):
        return a[name].i if name in a else default
    return {
        "strides": ints("strides", [1, 1]),
        "pads": ints("pads", [0, 0, 0, 0]),            # [top, left, bottom, right]
        "dilations": ints("dilations", [1, 1]),
        "output_padding": ints("output_padding", [0, 0]),
        "group": i("group", 1),
    }


def find_layer_boundaries(model, manifest_layers):
    """Return list of boundaries aligned with manifest. Each entry captures
    input/output tensor names, conv attrs, and whether a LeakyReLU follows."""
    nodes = list(model.graph.node)
    input_to_consumers = {}
    for n in nodes:
        for inp in n.input:
            input_to_consumers.setdefault(inp, []).append(n)

    conv_nodes = [n for n in nodes if n.op_type in ("Conv", "ConvTranspose")]
    if len(conv_nodes) != len(manifest_layers):
        raise RuntimeError(
            f"ONNX conv count ({len(conv_nodes)}) != manifest count "
            f"({len(manifest_layers)}). Is model_fused.onnx stale?"
        )

    boundaries = []
    for conv, layer in zip(conv_nodes, manifest_layers):
        has_leaky = False
        for c in input_to_consumers.get(conv.output[0], []):
            if c.op_type == "LeakyRelu":
                has_leaky = True
                break
        boundaries.append({
            "index": layer["index"],
            "name": layer["name"],
            "op": layer["op"],
            "input_tensor": conv.input[0],
            "output_tensor": conv.output[0],
            "has_leaky_relu": has_leaky,
            "attrs": get_conv_attrs(conv),
        })
    return boundaries


# ---------------------------------------------------------------------------
# Preprocessing (same as generate_activation_scales.py)
# ---------------------------------------------------------------------------

def preprocess_scan(scan_path, sensor_cfg):
    H = sensor_cfg["img_prop"]["height"]
    W = sensor_cfg["img_prop"]["width"]
    scan = LaserScan(
        project=True, H=H, W=W,
        fov_up=sensor_cfg["fov_up"], fov_down=sensor_cfg["fov_down"],
    )
    scan.open_scan(str(scan_path))
    proj_range = scan.proj_range.astype(np.float32)
    proj_xyz = scan.proj_xyz.astype(np.float32)
    proj_rem = scan.proj_remission.astype(np.float32)
    proj_mask = scan.proj_mask.astype(np.float32)
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
# FP32 activation capture (memory-friendly: one batch at a time)
# ---------------------------------------------------------------------------

def capture_fp32_activations(model, fused_onnx_path, x, tensor_names,
                             batch_size, tmp_dir):
    """Run ONNX forward pass capturing requested tensor_names. Returns a dict."""
    results = {}
    existing_outputs = {o.name for o in model.graph.output}
    extras = [n for n in tensor_names if n not in existing_outputs]
    batches = [extras[i:i + batch_size] for i in range(0, len(extras), batch_size)]

    for bi, batch in enumerate(batches):
        m = make_model_with_outputs(model, batch)
        p = os.path.join(tmp_dir, f"_cap{bi}.onnx")
        onnx.save(m, p)
        del m
        gc.collect()

        sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        avail = {o.name for o in sess.get_outputs()}
        req = [n for n in batch if n in avail]
        input_name = sess.get_inputs()[0].name
        outs = sess.run(req, {input_name: x})
        for n, arr in zip(req, outs):
            results[n] = arr
        del sess, outs
        gc.collect()
        try:
            os.remove(p)
        except OSError:
            pass

    # Final graph outputs too (some requested tensors might live in original graph)
    sess = ort.InferenceSession(fused_onnx_path, providers=["CPUExecutionProvider"])
    final_names = [o.name for o in sess.get_outputs()]
    input_name = sess.get_inputs()[0].name
    outs = sess.run(final_names, {input_name: x})
    for n, arr in zip(final_names, outs):
        if n in tensor_names and n not in results:
            results[n] = arr
    del sess, outs
    gc.collect()

    return results


# ---------------------------------------------------------------------------
# INT8 simulation primitives (must match conv_kernel.cpp bit-exactly)
# ---------------------------------------------------------------------------

def quantize_int8(x_fp, scale):
    """Symmetric INT8 quantize: x_i8 = clip(round(x / s), -128, 127)."""
    q = np.round(x_fp.astype(np.float64) / float(scale))
    return np.clip(q, -128, 127).astype(np.int8)


def conv_int32(x_i8, w_i8, attrs, is_transpose):
    """Exact INT32 conv / conv_transpose accumulator (no bias, no requant).

    Uses torch in float64 so int-range products are represented exactly.
    """
    pads = attrs["pads"]                # [top, left, bottom, right]
    stride = tuple(attrs["strides"])
    dilation = tuple(attrs["dilations"])
    out_pad = tuple(attrs["output_padding"])
    groups = attrs["group"]

    if is_transpose:
        # ConvTranspose: torch requires symmetric pads. For RangeNet all pads are
        # symmetric, so pad_h = pads[0]=pads[2], pad_w = pads[1]=pads[3].
        assert pads[0] == pads[2] and pads[1] == pads[3], \
            f"Asymmetric transpose pads not supported: {pads}"
        pad_hw = (pads[0], pads[1])
        xt = torch.from_numpy(x_i8.astype(np.float64))
        wt = torch.from_numpy(w_i8.astype(np.float64))
        out = F.conv_transpose2d(
            xt, wt, stride=stride, padding=pad_hw,
            output_padding=out_pad, dilation=dilation, groups=groups,
        )
    else:
        # Pre-pad to support asymmetric pads then call F.conv2d with padding=0.
        xpad = np.pad(
            x_i8,
            ((0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3])),
            mode="constant", constant_values=0,
        )
        xt = torch.from_numpy(xpad.astype(np.float64))
        wt = torch.from_numpy(w_i8.astype(np.float64))
        out = F.conv2d(
            xt, wt, stride=stride, padding=0,
            dilation=dilation, groups=groups,
        )
    return out.round().to(torch.int64).numpy().astype(np.int64)


def bias_to_int32(bias_fp, s_in, w_scales):
    """Convert FP32 (BN-folded) bias to accumulator-scale INT32 per output channel."""
    # bias_i32[c] = round(bias_fp[c] / (s_in * w_scale[c]))
    denom = float(s_in) * w_scales.astype(np.float64)
    b_i32 = np.round(bias_fp.astype(np.float64) / denom)
    # Clamp to int32 bounds just in case
    b_i32 = np.clip(b_i32, -(1 << 31), (1 << 31) - 1)
    return b_i32.astype(np.int32)


def requantize(acc_i64, int_mult, shift):
    """Per-channel (acc * int_mult + round) >> shift with round-to-nearest.

    acc_i64:   [N, C, H, W]  int64 accumulator
    int_mult:  [C]           int32
    shift:     [C]           int8 (>=0 typical)
    returns:   [N, C, H, W]  int32 (pre-saturation)
    """
    N, C, H, W = acc_i64.shape
    out = np.empty_like(acc_i64)
    for c in range(C):
        s = int(shift[c])
        m = int(int_mult[c])
        prod = acc_i64[:, c] * m                # int64
        if s > 0:
            rounded = prod + (1 << (s - 1))     # round-to-nearest-up
            out[:, c] = rounded >> s
        elif s == 0:
            out[:, c] = prod
        else:  # s < 0 (rare, real_m > 1): shift left
            out[:, c] = prod << (-s)
    # Clip to int32 for downstream storage
    out = np.clip(out, -(1 << 31), (1 << 31) - 1).astype(np.int32)
    return out


def leaky_relu_int(x_i32):
    """Kernel-exact LeakyReLU: if x<0, x = (x * 13) >> 7."""
    out = x_i32.copy()
    neg = out < 0
    # numpy's >> on signed int32 is arithmetic shift
    out[neg] = (out[neg].astype(np.int32) * LEAKY_NUM) >> LEAKY_SHIFT
    return out


def saturate_i8(x_i32):
    return np.clip(x_i32, -128, 127).astype(np.int8)


# ---------------------------------------------------------------------------
# Weight / bias file lookup
# ---------------------------------------------------------------------------

def safe_name(name):
    return name.replace(".", "_")


def load_weight_int8(int8_dir, int8_entry):
    path = int8_dir / int8_entry["weight_int8_file"]
    arr = np.fromfile(str(path), dtype=np.int8)
    return arr.reshape(int8_entry["weight_shape"])


def load_weight_scales(int8_dir, int8_entry):
    path = int8_dir / int8_entry["weight_scale_f32_file"]
    return np.fromfile(str(path), dtype=np.float32)


def load_bias_fp32(fp32_weights_dir, layer_name):
    path = fp32_weights_dir / f"{safe_name(layer_name)}.bias.bin"
    return np.fromfile(str(path), dtype=np.float32)


def load_requant(requant_dir, layer_name):
    base = safe_name(layer_name)
    mult = np.fromfile(str(requant_dir / f"{base}.requant_mult.bin"), dtype=np.int32)
    shift = np.fromfile(str(requant_dir / f"{base}.requant_shift.bin"), dtype=np.int8)
    return mult, shift


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Generate INT8 golden vectors for HLS validation")
    p.add_argument("--fused-onnx", default=str(DEFAULT_FUSED_ONNX))
    p.add_argument("--fp32-manifest", default=str(DEFAULT_FP32_MANIFEST))
    p.add_argument("--int8-manifest", default=str(DEFAULT_INT8_MANIFEST))
    p.add_argument("--requant-manifest", default=str(DEFAULT_REQUANT_MANIFEST))
    p.add_argument("--activation-scales", default=str(DEFAULT_ACT_SCALES))
    p.add_argument("--scan-root", required=True)
    p.add_argument("--sequence", default="00")
    p.add_argument("--arch-cfg", required=True)
    p.add_argument("--num-frames", type=int, default=3)
    p.add_argument("--frame-offset", type=int, default=0,
                   help="Start from the Nth scan (useful for picking different frames)")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--tmp-dir", default=str(SCRIPT_DIR / "_tmp_gold_i8"))
    p.add_argument("--capture-batch-size", type=int, default=8)
    p.add_argument("--skip-weight-dump", action="store_true",
                   help="Don't dump weight .bin copies (they're already on disk)")
    args = p.parse_args()

    # ---- Load manifests
    with open(args.fp32_manifest) as f:
        fp32_manifest = json.load(f)
    with open(args.int8_manifest) as f:
        int8_manifest = json.load(f)
    with open(args.requant_manifest) as f:
        _ = json.load(f)  # not strictly needed; we read .bin files directly
    with open(args.activation_scales) as f:
        act = json.load(f)

    int8_dir = Path(args.int8_manifest).parent
    requant_dir = Path(args.requant_manifest).parent
    fp32_weights_dir = Path(args.fp32_manifest).parent
    int8_map = {L["name"]: L for L in int8_manifest["layers"]}
    act_layers = act["layers"]

    # ---- Load ONNX model & boundaries
    print(f"Loading ONNX: {args.fused_onnx}")
    model = onnx.load(args.fused_onnx)
    boundaries = find_layer_boundaries(model, fp32_manifest["layers"])
    print(f"  {len(boundaries)} layers")

    # ---- Sensor config
    with open(args.arch_cfg) as f:
        sensor_cfg = yaml.safe_load(f)["dataset"]["sensor"]

    # ---- Frame list
    scan_dir = Path(args.scan_root) / args.sequence / "velodyne"
    scan_paths = sorted(scan_dir.glob("*.bin"))
    if not scan_paths:
        raise FileNotFoundError(f"No .bin files in {scan_dir}")
    scan_paths = scan_paths[args.frame_offset : args.frame_offset + args.num_frames]
    print(f"  {len(scan_paths)} frames from {scan_dir}")

    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Tensor names we need FP32 captures for (each layer's input)
    needed_tensors = list({b["input_tensor"] for b in boundaries})
    # Also grab each layer's conv_output for FP32 reference comparison
    needed_tensors += list({b["output_tensor"] for b in boundaries})
    needed_tensors = list(dict.fromkeys(needed_tensors))

    # ---- Loop over frames
    for fi, scan_path in enumerate(scan_paths):
        frame_dir = Path(args.output_dir) / f"frame_{fi:04d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nFrame {fi} ({scan_path.name})")

        x_fp = preprocess_scan(scan_path, sensor_cfg)
        x_fp.astype(np.float32).tofile(str(frame_dir / "input.f32.bin"))

        print("  Capturing FP32 activations ...")
        fp32_map = capture_fp32_activations(
            model, args.fused_onnx, x_fp, needed_tensors,
            args.capture_batch_size, args.tmp_dir,
        )
        print(f"  Captured {len(fp32_map)} tensors")

        # Per-layer INT8 simulation
        print("  Running INT8 simulation per layer ...")
        layer_entries = []
        for b in boundaries:
            idx = b["index"]
            name = b["name"]
            op = b["op"]
            is_transpose = (op == "ConvTranspose")

            # Scales
            if name not in act_layers:
                print(f"    [{idx:2d}] {name}: SKIP — not in activation_scales")
                continue
            s_in = float(act_layers[name]["input_scale"])
            s_out = float(act_layers[name]["output_scale"])
            if s_in <= 0 or s_out <= 0:
                print(f"    [{idx:2d}] {name}: SKIP — non-positive scale")
                continue

            # FP32 input & reference output
            if b["input_tensor"] not in fp32_map or b["output_tensor"] not in fp32_map:
                print(f"    [{idx:2d}] {name}: SKIP — missing FP32 capture")
                continue
            x_layer_fp = fp32_map[b["input_tensor"]]      # [1, IC, H, W]
            y_layer_fp = fp32_map[b["output_tensor"]]     # raw conv output (pre-relu)

            # Weights, scales, bias, requant params
            int8_entry = int8_map[name]
            w_i8 = load_weight_int8(int8_dir, int8_entry)         # int8
            w_scales = load_weight_scales(int8_dir, int8_entry)   # float32 [OC]
            bias_fp = load_bias_fp32(fp32_weights_dir, name)      # float32 [OC]
            int_mult, shift = load_requant(requant_dir, name)     # int32 [OC], int8 [OC]

            # 1) Quantize input
            x_i8 = quantize_int8(x_layer_fp, s_in)                # int8 [1, IC, H, W]

            # 2) INT32 MAC
            acc = conv_int32(x_i8, w_i8, b["attrs"], is_transpose)  # int64 [1, OC, H', W']

            # 3) Bias in accumulator space
            b_i32 = bias_to_int32(bias_fp, s_in, w_scales)         # int32 [OC]
            acc = acc + b_i32[None, :, None, None].astype(np.int64)

            # 4) Requant (multiply & shift with round-to-nearest)
            pre_sat = requantize(acc, int_mult, shift)             # int32 [1, OC, H', W']

            # 5) LeakyReLU if applicable
            post_lrelu = leaky_relu_int(pre_sat) if b["has_leaky_relu"] else pre_sat

            # 6) Saturate
            out_i8 = saturate_i8(post_lrelu)                       # int8

            # ---- Dump everything for this layer
            prefix = f"layer_{idx:02d}"
            (frame_dir / f"{prefix}_input.i8.bin").write_bytes(x_i8.tobytes())
            (frame_dir / f"{prefix}_bias.i32.bin").write_bytes(b_i32.tobytes())
            (frame_dir / f"{prefix}_acc.i32.bin").write_bytes(acc.astype(np.int32).tobytes())
            (frame_dir / f"{prefix}_pre_sat.i32.bin").write_bytes(pre_sat.tobytes())
            (frame_dir / f"{prefix}_output.i8.bin").write_bytes(out_i8.tobytes())
            (frame_dir / f"{prefix}_ref.f32.bin").write_bytes(
                y_layer_fp.astype(np.float32).tobytes()
            )
            if not args.skip_weight_dump:
                # Small convenience: copy per-layer weights + requant into frame dir
                # so a unit test only needs this one dir. Skip for speed/space.
                pass

            # FP32 sanity: INT8 output vs FP32 reference (on the post-relu tensor)
            # For comparison we de-quantize out_i8 to FP32 using s_out.
            out_fp32 = out_i8.astype(np.float32) * s_out
            # Reference is the PRE-LeakyReLU conv output from ONNX, so apply LeakyReLU
            # to the reference too for an apples-to-apples diff when applicable.
            if b["has_leaky_relu"]:
                ref = np.where(y_layer_fp < 0, 0.1 * y_layer_fp, y_layer_fp)
            else:
                ref = y_layer_fp
            diff = np.abs(out_fp32 - ref)
            max_abs_err = float(diff.max()) if diff.size else 0.0
            rel_err = float((diff.sum() / np.maximum(np.abs(ref).sum(), 1e-9)))

            layer_entries.append({
                "index": idx,
                "name": name,
                "op": op,
                "has_leaky_relu": b["has_leaky_relu"],
                "input_shape": list(x_i8.shape),
                "output_shape": list(out_i8.shape),
                "input_scale": s_in,
                "output_scale": s_out,
                "files": {
                    "input_i8": f"{prefix}_input.i8.bin",
                    "bias_i32": f"{prefix}_bias.i32.bin",
                    "acc_i32": f"{prefix}_acc.i32.bin",
                    "pre_sat_i32": f"{prefix}_pre_sat.i32.bin",
                    "output_i8": f"{prefix}_output.i8.bin",
                    "ref_f32": f"{prefix}_ref.f32.bin",
                },
                "sanity_max_abs_err_fp32": max_abs_err,
                "sanity_rel_err_fp32": rel_err,
            })

            if (idx + 1) % 10 == 0 or (idx + 1) == len(boundaries):
                print(f"    [{idx:2d}] {name:<40s} max|Δ|={max_abs_err:.3f}  rel={rel_err:.4f}")

            del acc, pre_sat, post_lrelu, x_i8, out_i8
            gc.collect()

        # Write per-frame manifest
        manifest = {
            "frame_index": fi,
            "scan_file": scan_path.name,
            "input_file": "input.f32.bin",
            "input_shape": list(x_fp.shape),
            "source": {
                "onnx": os.path.basename(args.fused_onnx),
                "activation_scales": os.path.basename(args.activation_scales),
                "int8_manifest": os.path.basename(args.int8_manifest),
            },
            "leaky_relu": {"numerator": LEAKY_NUM, "right_shift": LEAKY_SHIFT,
                           "approximates": LEAKY_NUM / (1 << LEAKY_SHIFT)},
            "layers": layer_entries,
        }
        with open(frame_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # Quick summary
        if layer_entries:
            rel_errs = [L["sanity_rel_err_fp32"] for L in layer_entries]
            print(f"  Frame {fi} sanity: mean rel_err={np.mean(rel_errs):.4f}  "
                  f"max rel_err={np.max(rel_errs):.4f}")

        del fp32_map
        gc.collect()

    # Cleanup
    try:
        os.rmdir(args.tmp_dir)
    except OSError:
        pass

    print(f"\nGolden vectors written to {args.output_dir}")
    print("Next: use these .bin files to bit-exact unit-test your conv_kernel.cpp")


if __name__ == "__main__":
    main()
