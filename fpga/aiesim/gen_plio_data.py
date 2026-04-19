#!/usr/bin/env python3
"""Generate PLIO text files for AIE x86 functional simulation.

Extracts a single spatial tile from golden INT8 vectors, transposes NCHW → HWC,
and writes text files in the format expected by ADF PLIO (one value per line,
decimal for int8/int32).

Usage:
    python3 gen_plio_data.py \
        --golden-dir ../golden_int8/frame_0000 \
        --gemm-dir   ../weights/gemm \
        --layer-idx  1 \
        --tile-oh    0 \
        --tile-ow    0 \
        --output-dir data

This produces data/ with:
    act_3x3.txt       — input activation tile (HWC, int8)
    wt_3x3.txt        — weight in GEMM-B layout (int8)
    bias_3x3.txt       — bias (int32)
    rq_mult_3x3.txt    — requant multiplier (int32)
    rq_shift_3x3.txt   — requant shift (int8)
    expected_out.txt   — expected output tile (HWC, int8) for verification
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path


# Must match conv_kernel.cpp
TILE_H   = 8
TILE_W   = 32
IC_BLOCK = 32
OC_BLOCK = 32


def write_plio_txt(filepath, data, dtype_str="int", plio_bits=64):
    """Write array to PLIO text file in packed format.

    The x86simulator expects multiple values per line based on PLIO width:
      plio_64_bits + int8  → 8 values per line
      plio_64_bits + int32 → 2 values per line
      plio_32_bits + int8  → 4 values per line
      plio_32_bits + int32 → 1 value per line
    """
    flat = data.flatten()
    elem_bytes = 4 if "32" in dtype_str else 1  # int32=4 bytes, int8=1 byte
    vals_per_line = plio_bits // (8 * elem_bytes)
    with open(filepath, "w") as f:
        for i in range(0, len(flat), vals_per_line):
            chunk = flat[i:i + vals_per_line]
            f.write(" ".join(str(int(v)) for v in chunk) + "\n")
    n_lines = (len(flat) + vals_per_line - 1) // vals_per_line
    print(f"  Wrote {filepath} ({len(flat)} values, {dtype_str}, "
          f"{vals_per_line}/line, {n_lines} lines)")


def write_dummy_plio(filepath, num_elements, dtype_str="int8", plio_bits=64):
    """Write a dummy (all-zeros) PLIO file."""
    if "32" in dtype_str:
        data = np.zeros(num_elements, dtype=np.int32)
    else:
        data = np.zeros(num_elements, dtype=np.int8)
    write_plio_txt(filepath, data, dtype_str, plio_bits)


def extract_tile_3x3(act_nchw, oh_start, ow_start, ic_start, ic_block):
    """Extract input tile for 3×3 conv (with halo=1) and convert to HWC.

    act_nchw: [1, IC, H, W]  int8
    Returns: [IN_H_3x3, IN_W_3x3, ic_block] int8  (HWC layout)
    """
    _, IC, H, W = act_nchw.shape

    # Input tile includes 1-pixel halo on each side for 3×3 conv (pad=1)
    # Output pixel oh maps to input center oh; receptive field is [oh-1 .. oh+1]
    # So the input tile starts 1 row/col before the output tile start
    ih_start = oh_start - 1
    iw_start = ow_start - 1

    in_h = TILE_H + 2  # 10
    in_w = TILE_W + 2  # 34

    # Clip to valid range (zero-pad if needed)
    tile = np.zeros((1, ic_block, in_h, in_w), dtype=np.int8)
    for ic in range(min(ic_block, IC - ic_start)):
        for h in range(in_h):
            for w in range(in_w):
                ih = ih_start + h
                iw = iw_start + w
                if 0 <= ih < H and 0 <= iw < W:
                    tile[0, ic, h, w] = act_nchw[0, ic_start + ic, ih, iw]

    # NCHW → HWC: [in_h, in_w, ic_block]
    tile_hwc = tile[0].transpose(1, 2, 0)  # [C, H, W] → [H, W, C]
    return tile_hwc


def extract_tile_3x3_s2(act_nchw, oh_start, ow_start, ic_start, ic_block):
    """Extract input tile for 3×3 conv with stride=(1,2) and convert to HWC.

    act_nchw: [1, IC, H, W]  int8
    Returns: [IN_H_3x3, IN_W_3x3_S2, ic_block] int8  (HWC layout)
      where IN_W_3x3_S2 = TILE_W * 2 + 2 = 66
    """
    _, IC, H, W = act_nchw.shape

    # pad=1 for both h and w
    ih_start = oh_start - 1
    # stride_w=2: output col ow maps to input center ow*2
    iw_start = ow_start * 2 - 1

    in_h = TILE_H + 2       # 10
    in_w = TILE_W * 2 + 2   # 66

    tile = np.zeros((1, ic_block, in_h, in_w), dtype=np.int8)
    for ic in range(min(ic_block, IC - ic_start)):
        for h in range(in_h):
            for w in range(in_w):
                ih = ih_start + h
                iw = iw_start + w
                if 0 <= ih < H and 0 <= iw < W:
                    tile[0, ic, h, w] = act_nchw[0, ic_start + ic, ih, iw]

    tile_hwc = tile[0].transpose(1, 2, 0)  # [C, H, W] → [H, W, C]
    return tile_hwc


def extract_tile_1x1(act_nchw, oh_start, ow_start, ic_start, ic_block):
    """Extract input tile for 1×1 conv (no halo) and convert to HWC.

    Returns: [TILE_H, TILE_W, ic_block] int8  (HWC layout)
    """
    _, IC, H, W = act_nchw.shape

    tile = np.zeros((1, ic_block, TILE_H, TILE_W), dtype=np.int8)
    for ic in range(min(ic_block, IC - ic_start)):
        for h in range(TILE_H):
            for w in range(TILE_W):
                ih = oh_start + h
                iw = ow_start + w
                if 0 <= ih < H and 0 <= iw < W:
                    tile[0, ic, h, w] = act_nchw[0, ic_start + ic, ih, iw]

    tile_hwc = tile[0].transpose(1, 2, 0)
    return tile_hwc


def extract_output_tile(out_nchw, oh_start, ow_start, oc_start, oc_block):
    """Extract expected output tile and convert to HWC.

    out_nchw: [1, OC, H', W']  int8
    Returns: [TILE_H, TILE_W, oc_block]  int8
    """
    _, OC, H, W = out_nchw.shape

    tile = np.zeros((1, oc_block, TILE_H, TILE_W), dtype=np.int8)
    for oc in range(min(oc_block, OC - oc_start)):
        for h in range(TILE_H):
            for w in range(TILE_W):
                oh = oh_start + h
                ow = ow_start + w
                if 0 <= oh < H and 0 <= ow < W:
                    tile[0, oc, h, w] = out_nchw[0, oc_start + oc, oh, ow]

    tile_hwc = tile[0].transpose(1, 2, 0)
    return tile_hwc


def extract_weight_block(gemm_weight, ic_start, oc_start, K_per_ic, ic_block, oc_block):
    """Extract a [K_block, OC_block] sub-block from full GEMM-B weight.

    gemm_weight: [K_total, OC_total] int8
    K_per_ic: number of K elements per input channel (9 for 3×3, 1 for 1×1)

    Returns: [ic_block * K_per_ic, oc_block] int8
    """
    K_total, OC_total = gemm_weight.shape

    k_start = ic_start * K_per_ic
    k_end = min(k_start + ic_block * K_per_ic, K_total)
    oc_end = min(oc_start + oc_block, OC_total)

    block = np.zeros((ic_block * K_per_ic, oc_block), dtype=np.int8)
    k_len = k_end - k_start
    oc_len = oc_end - oc_start
    block[:k_len, :oc_len] = gemm_weight[k_start:k_end, oc_start:oc_end]

    return block


def main():
    p = argparse.ArgumentParser(description="Generate PLIO data for AIE x86sim")
    p.add_argument("--golden-dir", required=True,
                   help="Path to golden frame dir (e.g., golden_int8/frame_0000)")
    p.add_argument("--gemm-dir", required=True,
                   help="Path to weights/gemm directory")
    p.add_argument("--layer-idx", type=int, default=1,
                   help="Layer index to test (default: 1, a 3×3 conv)")
    p.add_argument("--tile-oh", type=int, default=0,
                   help="Output tile row start (in output spatial coords)")
    p.add_argument("--tile-ow", type=int, default=0,
                   help="Output tile col start (in output spatial coords)")
    p.add_argument("--ic-start", type=int, default=0,
                   help="Input channel block start")
    p.add_argument("--oc-start", type=int, default=0,
                   help="Output channel block start")
    p.add_argument("--output-dir", default="data",
                   help="Output directory for PLIO text files")
    p.add_argument("--num-instances", type=int, default=1,
                   help="Number of parallel instances for multi-tile parallelization")
    args = p.parse_args()

    golden_dir = Path(args.golden_dir)
    gemm_dir = Path(args.gemm_dir)

    # Load golden frame manifest
    manifest_path = golden_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found")
        sys.exit(1)
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Load GEMM manifest
    gemm_manifest_path = gemm_dir / "manifest.json"
    if not gemm_manifest_path.exists():
        print(f"ERROR: {gemm_manifest_path} not found")
        sys.exit(1)
    with open(gemm_manifest_path) as f:
        gemm_manifest = json.load(f)

    # Find the layer
    layer_info = None
    for L in manifest["layers"]:
        if L["index"] == args.layer_idx:
            layer_info = L
            break
    if layer_info is None:
        print(f"ERROR: Layer {args.layer_idx} not found in golden manifest")
        sys.exit(1)

    gemm_layer = None
    for L in gemm_manifest["layers"]:
        if L["index"] == args.layer_idx:
            gemm_layer = L
            break
    if gemm_layer is None:
        print(f"ERROR: Layer {args.layer_idx} not found in GEMM manifest")
        sys.exit(1)

    name = layer_info["name"]
    kernel = gemm_layer["kernel"]
    kH, kW = kernel
    is_3x3 = (kH == 3 and kW == 3)
    IC = gemm_layer["IC"]
    OC = gemm_layer["OC"]

    prefix = f"layer_{args.layer_idx:02d}"
    input_shape = layer_info["input_shape"]   # [1, IC, H, W]
    output_shape = layer_info["output_shape"]  # [1, OC, H', W']

    print(f"Layer {args.layer_idx}: {name}")
    print(f"  Op: {gemm_layer['op']}, Kernel: {kH}×{kW}, IC={IC}, OC={OC}")
    print(f"  Input shape:  {input_shape}")
    print(f"  Output shape: {output_shape}")
    print(f"  Tile position: oh={args.tile_oh}, ow={args.tile_ow}")
    print(f"  Channel block: ic={args.ic_start}, oc={args.oc_start}")

    # Load golden input and output (NCHW, int8)
    input_data = np.fromfile(
        str(golden_dir / layer_info["files"]["input_i8"]),
        dtype=np.int8
    ).reshape(input_shape)

    output_data = np.fromfile(
        str(golden_dir / layer_info["files"]["output_i8"]),
        dtype=np.int8
    ).reshape(output_shape)

    # Extract tiles
    os.makedirs(args.output_dir, exist_ok=True)

    # Detect stride from input/output shapes
    stride_h = input_shape[2] // output_shape[2] if output_shape[2] > 0 else 1
    stride_w = input_shape[3] // output_shape[3] if output_shape[3] > 0 else 1
    is_s2 = (stride_h == 1 and stride_w == 2)
    print(f"  Stride: ({stride_h},{stride_w})")

    if is_3x3 and is_s2:
        kernel_type = "3x3s2"
    elif is_3x3:
        kernel_type = "3x3"
    else:
        kernel_type = "1x1"

    if kernel_type == "3x3s2":
        act_tile = extract_tile_3x3_s2(
            input_data, args.tile_oh, args.tile_ow,
            args.ic_start, IC_BLOCK
        )
        K_per_ic = 9
    elif kernel_type == "3x3":
        act_tile = extract_tile_3x3(
            input_data, args.tile_oh, args.tile_ow,
            args.ic_start, IC_BLOCK
        )
        K_per_ic = 9
    else:
        act_tile = extract_tile_1x1(
            input_data, args.tile_oh, args.tile_ow,
            args.ic_start, IC_BLOCK
        )
        K_per_ic = 1

    # Load GEMM weight and extract block
    w_gemm = np.fromfile(
        str(gemm_dir / gemm_layer["weight_file"]),
        dtype=np.int8
    ).reshape(gemm_layer["gemm_shape"])

    wt_block = extract_weight_block(
        w_gemm, args.ic_start, args.oc_start, K_per_ic, IC_BLOCK, OC_BLOCK
    )

    # Load bias (int32, per output channel block)
    bias_full = np.fromfile(
        str(gemm_dir / gemm_layer["bias_file"]),
        dtype=np.int32
    )
    bias_block = np.zeros(OC_BLOCK, dtype=np.int32)
    oc_len = min(OC_BLOCK, OC - args.oc_start)
    bias_block[:oc_len] = bias_full[args.oc_start:args.oc_start + oc_len]

    # Load requant params (per output channel block)
    rq_mult_full = np.fromfile(
        str(gemm_dir / gemm_layer["requant_mult_file"]),
        dtype=np.int32
    )
    rq_shift_full = np.fromfile(
        str(gemm_dir / gemm_layer["requant_shift_file"]),
        dtype=np.int8
    )
    rq_mult_block = np.zeros(OC_BLOCK, dtype=np.int32)
    rq_shift_block = np.zeros(OC_BLOCK, dtype=np.int8)
    rq_mult_block[:oc_len] = rq_mult_full[args.oc_start:args.oc_start + oc_len]
    rq_shift_block[:oc_len] = rq_shift_full[args.oc_start:args.oc_start + oc_len]

    # Expected output tile
    out_tile = extract_output_tile(
        output_data, args.tile_oh, args.tile_ow,
        args.oc_start, OC_BLOCK
    )

    # Write PLIO text files (format must match plio width in AIE_graph.cpp)
    # 3x3/1x1 act, wt: plio_64_bits, int8 → 8 vals/line
    # bias, rq_mult:    plio_32_bits, int32 → 1 val/line
    # rq_shift:         plio_32_bits, int8 → 4 vals/line
    print(f"\nWriting PLIO files to {args.output_dir}/")
    
    if args.num_instances > 1:
        # Multi-tile parallelization: split data across instances
        print(f"  Multi-tile parallelization: {args.num_instances} instances")
        
        # Split data across instances (for now, duplicate data for simplicity)
        # In a real implementation, you would split the actual spatial tiles
        for i in range(args.num_instances):
            instance_suffix = f"_{i}"
            write_plio_txt(f"{args.output_dir}/act_{kernel_type}{instance_suffix}.txt", act_tile, "int8", plio_bits=64)
            write_plio_txt(f"{args.output_dir}/wt_{kernel_type}{instance_suffix}.txt", wt_block, "int8", plio_bits=64)
            write_plio_txt(f"{args.output_dir}/bias_{kernel_type}{instance_suffix}.txt", bias_block, "int32", plio_bits=32)
            write_plio_txt(f"{args.output_dir}/rq_mult_{kernel_type}{instance_suffix}.txt", rq_mult_block, "int32", plio_bits=32)
            write_plio_txt(f"{args.output_dir}/rq_shift_{kernel_type}{instance_suffix}.txt", rq_shift_block, "int8", plio_bits=32)
            write_plio_txt(f"{args.output_dir}/expected_out{instance_suffix}.txt", out_tile, "int8", plio_bits=64)
    else:
        # Single instance (original behavior)
        write_plio_txt(f"{args.output_dir}/act_{kernel_type}.txt", act_tile, "int8", plio_bits=64)
        write_plio_txt(f"{args.output_dir}/wt_{kernel_type}.txt", wt_block, "int8", plio_bits=64)
        write_plio_txt(f"{args.output_dir}/bias_{kernel_type}.txt", bias_block, "int32", plio_bits=32)
        write_plio_txt(f"{args.output_dir}/rq_mult_{kernel_type}.txt", rq_mult_block, "int32", plio_bits=32)
        write_plio_txt(f"{args.output_dir}/rq_shift_{kernel_type}.txt", rq_shift_block, "int8", plio_bits=32)
        write_plio_txt(f"{args.output_dir}/expected_out.txt", out_tile, "int8", plio_bits=64)

    # Generate dummy PLIO files for all unused kernels
    # (the graph instantiates all 4 kernel types, so all PLIOs need files)
    IN_BUF_3x3    = (TILE_H+2) * (TILE_W+2) * IC_BLOCK       # 10880
    IN_BUF_3x3_S2 = (TILE_H+2) * (TILE_W*2+2) * IC_BLOCK     # 21120
    IN_BUF_1x1    = TILE_H * TILE_W * IC_BLOCK                # 8192
    WT_BUF_3x3_   = IC_BLOCK * 9 * OC_BLOCK                   # 9216
    WT_BUF_1x1_   = IC_BLOCK * OC_BLOCK                       # 1024
    out_buf       = TILE_H * TILE_W * OC_BLOCK                 # 8192

    # Map of kernel_tag -> (act_buf_size, wt_buf_size)
    all_types = {
        "3x3":   (IN_BUF_3x3,    WT_BUF_3x3_),
        "3x3s2": (IN_BUF_3x3_S2, WT_BUF_3x3_),
        "1x1":   (IN_BUF_1x1,    WT_BUF_1x1_),
    }
    unused = [k for k in all_types if k != kernel_type]
    print(f"\n  Writing dummy files for unused kernels: {unused} + elem_add...")
    for ut in unused:
        ab, wb = all_types[ut]
        write_dummy_plio(f"{args.output_dir}/act_{ut}.txt", ab, "int8", 64)
        write_dummy_plio(f"{args.output_dir}/wt_{ut}.txt", wb, "int8", 64)
        write_dummy_plio(f"{args.output_dir}/bias_{ut}.txt", OC_BLOCK, "int32", 32)
        write_dummy_plio(f"{args.output_dir}/rq_mult_{ut}.txt", OC_BLOCK, "int32", 32)
        write_dummy_plio(f"{args.output_dir}/rq_shift_{ut}.txt", OC_BLOCK, "int8", 32)
    write_dummy_plio(f"{args.output_dir}/add_a.txt", out_buf, "int8", 64)
    write_dummy_plio(f"{args.output_dir}/add_b.txt", out_buf, "int8", 64)

    # Also write a small metadata file
    meta = {
        "layer_index": args.layer_idx,
        "layer_name": name,
        "kernel_type": kernel_type,
        "tile_oh": args.tile_oh,
        "tile_ow": args.tile_ow,
        "ic_start": args.ic_start,
        "oc_start": args.oc_start,
        "act_shape": list(act_tile.shape),
        "wt_shape": list(wt_block.shape),
        "out_shape": list(out_tile.shape),
        "IC": IC,
        "OC": OC,
    }
    with open(f"{args.output_dir}/tile_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Wrote {args.output_dir}/tile_meta.json")

    # NOTE: For layers where IC > IC_BLOCK, the full convolution requires
    # accumulating across multiple IC blocks. For a single-IC-block x86sim test,
    # the output will only match the golden if IC == IC_BLOCK (layer 0 or 1).
    if IC > IC_BLOCK:
        print(f"\n  WARNING: IC={IC} > IC_BLOCK={IC_BLOCK}. This tile only covers "
              f"ic_start={args.ic_start}..{args.ic_start + IC_BLOCK}.")
        print(f"  The AIE output for this single block will NOT match the full-layer "
              f"golden output (which accumulates across all IC blocks).")
        print(f"  Use a layer with IC <= {IC_BLOCK} (e.g., layer 0 or 1) for "
              f"single-block validation.")

    if OC > OC_BLOCK:
        print(f"  NOTE: OC={OC} > OC_BLOCK={OC_BLOCK}. Testing oc_start={args.oc_start} "
              f"block only.")

    print("\nDone. Ready for x86 simulation.")


if __name__ == "__main__":
    main()
