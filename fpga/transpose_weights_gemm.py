#!/usr/bin/env python3
"""Transpose INT8 weights to GEMM-B layout and compute INT32 bias for AIE-ML.

Reads:
    fpga/weights/int8/manifest.json              — per-layer INT8 weight metadata
    fpga/weights/int8/<layer>.weight.i8.bin      — weights in [OC, IC, kH, kW]
    fpga/weights/manifest.json                    — FP32 fused weight manifest (for bias)
    fpga/weights/<layer>.bias.bin                 — FP32 BN-folded bias
    fpga/weights/requant/manifest.json            — requant params
    fpga/weights/requant/<layer>.requant_mult.bin — per-channel int_mult
    fpga/weights/requant/<layer>.requant_shift.bin— per-channel shift
    fpga/activation_scales.json                   — per-layer activation scales

Writes:
    fpga/weights/gemm/<layer>.weight.gemm.i8.bin  — weights in GEMM-B layout
    fpga/weights/gemm/<layer>.bias.i32.bin        — INT32 bias in accumulator space
    fpga/weights/gemm/<layer>.requant_mult.i32.bin— per-channel requant multiplier (copy)
    fpga/weights/gemm/<layer>.requant_shift.i8.bin— per-channel requant shift (copy)
    fpga/weights/gemm/manifest.json               — complete manifest for AIE kernels

GEMM-B layout (what the AIE kernel expects):
    For Conv (kH×kW > 1):  [IC*kH*kW, OC]  (transposed from [OC, IC*kH*kW])
    For Conv (1×1):         [IC, OC]         (transposed from [OC, IC])
    For ConvTranspose:      [IC*kH*kW, OC]  (from [IC, OC, kH, kW] → reshape → transpose)

This matches conv_kernel.cpp which does:
    A[spatial × K] × B[K × OC] → C[spatial × OC]
where K = IC*kH*kW for 3×3, K = IC for 1×1.

INT32 bias computation (matches generate_golden_int8.py):
    bias_i32[c] = round( bias_fp[c] / (s_in * w_scale[c]) )
    where s_in = activation scale for the layer's input
          w_scale[c] = per-channel weight quantization scale
"""

import os
import json
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input directories / files
INT8_WEIGHT_DIR  = os.path.join(SCRIPT_DIR, "weights", "int8")
INT8_MANIFEST    = os.path.join(INT8_WEIGHT_DIR, "manifest.json")

FP32_WEIGHT_DIR  = os.path.join(SCRIPT_DIR, "weights")
FP32_MANIFEST    = os.path.join(FP32_WEIGHT_DIR, "manifest.json")

REQUANT_DIR      = os.path.join(SCRIPT_DIR, "weights", "requant")
REQUANT_MANIFEST = os.path.join(REQUANT_DIR, "manifest.json")

ACT_SCALES_PATH  = os.path.join(SCRIPT_DIR, "activation_scales.json")

# Output directory
GEMM_DIR = os.path.join(SCRIPT_DIR, "weights", "gemm")


def transpose_conv_weight(w_i8, op_type, kernel_shape):
    """Transpose weight tensor to GEMM-B layout.

    Args:
        w_i8: numpy int8 array in original layout
            Conv:          [OC, IC, kH, kW]
            ConvTranspose: [IC, OC, kH, kW]
        op_type: "Conv" or "ConvTranspose"
        kernel_shape: [kH, kW]

    Returns:
        w_gemm: numpy int8 array in [K, OC] layout
            where K = IC * kH * kW
    """
    kH, kW = kernel_shape

    if op_type == "Conv":
        # w_i8 shape: [OC, IC, kH, kW]
        OC = w_i8.shape[0]
        IC = w_i8.shape[1]

        if kH == 1 and kW == 1:
            # 1×1: [OC, IC, 1, 1] → [OC, IC] → transpose → [IC, OC]
            w_flat = w_i8.reshape(OC, IC)
            w_gemm = w_flat.T.copy()  # [IC, OC]
        else:
            # 3×3 or other: reorder K dimension to match im2col in conv_kernel.cpp
            # im2col produces K = (kh, kw, ic) — spatial outer, channel inner
            # So: [OC, IC, kH, kW] → [OC, kH, kW, IC] → [OC, kH*kW*IC] → T → [K, OC]
            K = IC * kH * kW
            w_reorder = w_i8.transpose(0, 2, 3, 1)  # [OC, kH, kW, IC]
            w_flat = w_reorder.reshape(OC, K)
            w_gemm = w_flat.T.copy()  # [K, OC]

    elif op_type == "ConvTranspose":
        # w_i8 shape: [IC, OC, kH, kW]
        IC = w_i8.shape[0]
        OC = w_i8.shape[1]
        K = IC * kH * kW

        # Reshape to [IC*kH*kW, OC] by reinterpreting the axes
        # Original: w[ic, oc, kh, kw]
        # We want B[k, oc] where k = ic * kH * kW + kh * kW + kw
        # This is just: transpose axes (0,2,3) as the K dim, axis 1 as OC
        w_gemm = w_i8.transpose(0, 2, 3, 1).reshape(K, OC).copy()

    else:
        raise ValueError(f"Unknown op_type: {op_type}")

    return w_gemm


def bias_to_int32(bias_fp, s_in, w_scales):
    """Convert FP32 (BN-folded) bias to accumulator-space INT32 per channel.

    Matches generate_golden_int8.py exactly:
        bias_i32[c] = round( bias_fp[c] / (s_in * w_scale[c]) )
    """
    denom = float(s_in) * w_scales.astype(np.float64)
    b_i32 = np.round(bias_fp.astype(np.float64) / denom)
    b_i32 = np.clip(b_i32, -(1 << 31), (1 << 31) - 1)
    return b_i32.astype(np.int32)


def main():
    # ---- Load manifests ----
    for path, desc in [
        (INT8_MANIFEST, "INT8 weight manifest"),
        (FP32_MANIFEST, "FP32 fused weight manifest"),
        (REQUANT_MANIFEST, "Requant manifest"),
        (ACT_SCALES_PATH, "Activation scales"),
    ]:
        if not os.path.exists(path):
            print(f"ERROR: {desc} not found: {path}")
            sys.exit(1)

    with open(INT8_MANIFEST) as f:
        int8_manifest = json.load(f)
    with open(FP32_MANIFEST) as f:
        fp32_manifest = json.load(f)
    with open(REQUANT_MANIFEST) as f:
        rq_manifest = json.load(f)
    with open(ACT_SCALES_PATH) as f:
        act_scales = json.load(f)

    # Build lookup tables
    fp32_by_idx = {L["index"]: L for L in fp32_manifest["layers"]}
    rq_by_idx   = {L["index"]: L for L in rq_manifest["layers"]}

    # Activation scales: keyed by layer name
    act_by_name = act_scales.get("layers", {})

    os.makedirs(GEMM_DIR, exist_ok=True)

    gemm_layers = []
    total_bytes = 0

    for layer in int8_manifest["layers"]:
        idx  = layer["index"]
        name = layer["name"]
        op   = layer["op"]
        w_shape = layer["weight_shape"]
        kernel = layer.get("kernel",
                           [w_shape[2], w_shape[3]] if len(w_shape) == 4 else [1, 1])
        kH, kW = kernel

        if op == "Conv":
            OC, IC = w_shape[0], w_shape[1]
        else:
            IC, OC = w_shape[0], w_shape[1]
        K = IC * kH * kW

        safe_name = name.replace(".", "_")

        # ---- 1. Transpose weight to GEMM-B layout ----
        w_path = os.path.join(INT8_WEIGHT_DIR, layer["weight_int8_file"])
        w_i8 = np.fromfile(w_path, dtype=np.int8).reshape(w_shape)
        w_gemm = transpose_conv_weight(w_i8, op, kernel)

        gemm_w_file = f"{safe_name}.weight.gemm.i8.bin"
        w_gemm.tofile(os.path.join(GEMM_DIR, gemm_w_file))

        # ---- 2. Compute INT32 bias ----
        gemm_b_file = None
        fp32_layer = fp32_by_idx.get(idx)
        if fp32_layer and fp32_layer.get("bias_file"):
            bias_fp = np.fromfile(
                os.path.join(FP32_WEIGHT_DIR, fp32_layer["bias_file"]),
                dtype=np.float32
            )
            # Need s_in (activation input scale) and w_scales (per-channel weight scale)
            act_info = act_by_name.get(name)
            w_scale_file = layer.get("weight_scale_f32_file")
            if act_info and w_scale_file:
                s_in = act_info["input_scale"]
                w_scales = np.fromfile(
                    os.path.join(INT8_WEIGHT_DIR, w_scale_file), dtype=np.float32
                ).astype(np.float64)
                b_i32 = bias_to_int32(bias_fp, s_in, w_scales)
            else:
                # Fallback: approximate using overall scale
                # bias_i32 ≈ round(bias_fp / overall_scale)
                # Use a safe default; this will be less accurate
                print(f"  WARNING [{idx}] {name}: missing act/weight scales, "
                      f"using zeros for bias")
                b_i32 = np.zeros(OC, dtype=np.int32)

            gemm_b_file = f"{safe_name}.bias.i32.bin"
            b_i32.tofile(os.path.join(GEMM_DIR, gemm_b_file))

        # ---- 3. Copy requant mult & shift ----
        gemm_rq_mult_file = None
        gemm_rq_shift_file = None
        rq_layer = rq_by_idx.get(idx)
        if rq_layer:
            # Requant multiplier
            src_mult = os.path.join(REQUANT_DIR, rq_layer["requant_mult_file"])
            gemm_rq_mult_file = f"{safe_name}.requant_mult.i32.bin"
            rq_mult = np.fromfile(src_mult, dtype=np.int32)
            rq_mult.tofile(os.path.join(GEMM_DIR, gemm_rq_mult_file))

            # Requant shift
            src_shift = os.path.join(REQUANT_DIR, rq_layer["requant_shift_file"])
            gemm_rq_shift_file = f"{safe_name}.requant_shift.i8.bin"
            rq_shift = np.fromfile(src_shift, dtype=np.int8)
            rq_shift.tofile(os.path.join(GEMM_DIR, gemm_rq_shift_file))

        # ---- Record ----
        info = {
            "index": idx,
            "name": name,
            "op": op,
            "kernel": kernel,
            "IC": IC,
            "OC": OC,
            "K_dim": K,
            "gemm_shape": list(w_gemm.shape),
            "weight_file": gemm_w_file,
            "bias_file": gemm_b_file,
            "requant_mult_file": gemm_rq_mult_file,
            "requant_shift_file": gemm_rq_shift_file,
            "original_shape": w_shape,
            "weight_bytes": w_gemm.nbytes,
            "activation": layer.get("activation", "none"),
        }
        gemm_layers.append(info)
        total_bytes += w_gemm.nbytes

        kernel_str = f"{kH}x{kW}"
        bias_tag = "✓" if gemm_b_file else "—"
        rq_tag   = "✓" if gemm_rq_mult_file else "—"
        print(f"  [{idx:2d}] {op:<14s} {name:<40s} "
              f"{str(w_shape):<22s} → [{K},{OC}]  "
              f"bias={bias_tag} rq={rq_tag}")

    # ---- Write manifest ----
    manifest = {
        "description": "Complete AIE-ML kernel data: GEMM-B weights + INT32 bias + requant",
        "source": "transpose_weights_gemm.py",
        "weight_layout": "[K, OC] where K = IC * kH * kW",
        "weight_dtype": "int8",
        "bias_dtype": "int32",
        "requant_mult_dtype": "int32",
        "requant_shift_dtype": "int8",
        "byte_order": "little-endian",
        "total_weight_bytes": total_bytes,
        "num_layers": len(gemm_layers),
        "tile_config": {
            "IC_BLOCK": 32,
            "OC_BLOCK": 32,
            "TILE_H": 8,
            "TILE_W": 32,
        },
        "layers": gemm_layers,
    }

    manifest_path = os.path.join(GEMM_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Total GEMM weight bytes: {total_bytes / 1e6:.2f} MB")
    print(f"  {len(gemm_layers)} layers written to {GEMM_DIR}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    print("Transposing INT8 weights to GEMM-B layout for AIE-ML ...\n")
    main()
