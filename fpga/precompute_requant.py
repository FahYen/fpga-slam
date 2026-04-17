#!/usr/bin/env python3
"""Phase 4b: Precompute per-channel requantization parameters (int_mult, shift).

Combines:
  - Per-channel INT8 weight scales  : fpga/weights/int8/<layer>.weight.scale.f32.bin
  - Per-layer activation scales     : fpga/activation_scales.json
Into per-channel fixed-point (int_mult, shift) values that the AIE-ML
conv kernel uses to project INT32 accumulators back to INT8.

Math:
  For symmetric-INT8 quantization:
      real_value = int8_value * scale
  The conv accumulator is:
      acc_i32 = sum(a_i8 * w_i8)   # in per-channel INT8 space
  To convert back to the next layer's INT8 input:
      real_m[c] = (s_in * w_scale[c]) / s_out
      out_i8    = saturate( round(acc_i32 * real_m[c]) )
  We represent real_m[c] as a fixed-point (int_mult[c], shift[c]):
      real_m[c] ~= int_mult[c] / 2^shift[c]
  In hardware:
      scaled = (int64_t)acc * int_mult[c] >> shift[c]

Target int_mult range [16384, 32767] gives ~15 bits of mantissa precision
while leaving 1 bit headroom for rounding.

Outputs per layer:
  fpga/weights/requant/<safe_name>.requant_mult.bin   (int32, [channels])
  fpga/weights/requant/<safe_name>.requant_shift.bin  (int8,  [channels])
  fpga/weights/requant/manifest.json                  (summary + paths)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INT8_MANIFEST = SCRIPT_DIR / "weights" / "int8" / "manifest.json"
DEFAULT_ACT_SCALES = SCRIPT_DIR / "activation_scales.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "weights" / "requant"

# Target range for int_mult: [MIN_MULT, MAX_MULT).
# 16384..32768 gives ~15 bits of mantissa with 1 bit of rounding headroom.
MIN_MULT = 16384
MAX_MULT = 32768
MAX_SHIFT = 31  # int_mult is int32, shift must stay in a sensible range


def real_m_to_fixed(real_m):
    """Convert a positive real multiplier to (int_mult, shift) such that
    real_m ≈ int_mult / 2^shift, with int_mult in [MIN_MULT, MAX_MULT)
    whenever possible.

    Returns (int_mult: int, shift: int).
    """
    if not np.isfinite(real_m) or real_m <= 0.0:
        return 0, 0

    val = float(real_m)
    shift = 0

    # Shift up if val is too small.
    while val < MIN_MULT and shift < MAX_SHIFT:
        val *= 2.0
        shift += 1

    # Shift down if val is too large (happens only if real_m > 1, which is rare).
    while val >= MAX_MULT and shift > -16:
        val *= 0.5
        shift -= 1

    int_mult = int(round(val))
    # Clamp just in case of FP edge cases
    if int_mult >= (1 << 31):
        int_mult = (1 << 31) - 1
    if int_mult < 0:
        int_mult = 0
    return int_mult, shift


def main():
    parser = argparse.ArgumentParser(
        description="Precompute per-channel requantization params for AIE-ML kernels"
    )
    parser.add_argument("--int8-manifest", default=str(DEFAULT_INT8_MANIFEST))
    parser.add_argument("--activation-scales", default=str(DEFAULT_ACT_SCALES))
    parser.add_argument("--weights-dir", default=None,
                        help="Directory holding the INT8 weight/scale .bin files. "
                             "Defaults to the directory of --int8-manifest.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    # ---- Load manifests
    int8_manifest_path = Path(args.int8_manifest)
    with open(int8_manifest_path) as f:
        int8_manifest = json.load(f)

    with open(args.activation_scales) as f:
        act = json.load(f)
    act_layers = act["layers"]

    weights_dir = Path(args.weights_dir) if args.weights_dir else int8_manifest_path.parent
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"INT8 manifest:     {int8_manifest_path}")
    print(f"Activation scales: {args.activation_scales}")
    print(f"Weights dir:       {weights_dir}")
    print(f"Output dir:        {out_dir}")
    print(f"Target int_mult:   [{MIN_MULT}, {MAX_MULT})")

    layers_in = int8_manifest["layers"]
    print(f"\nProcessing {len(layers_in)} layers ...")

    out_layers = []
    missing_act = []
    shift_stats = []     # for global summary
    int_mult_stats = []

    for L in layers_in:
        name = L["name"]
        channels = L["channels"]
        w_scale_file = L["weight_scale_f32_file"]

        if name not in act_layers:
            missing_act.append(name)
            continue

        a = act_layers[name]
        s_in = float(a["input_scale"])
        s_out = float(a["output_scale"])

        if s_in <= 0.0 or s_out <= 0.0:
            print(f"  [{L['index']:2d}] {name}: SKIP — non-positive activation scale "
                  f"(s_in={s_in}, s_out={s_out})")
            continue

        # Load per-channel weight scales: shape [channels], float32
        w_scale_path = weights_dir / w_scale_file
        w_scales = np.fromfile(str(w_scale_path), dtype=np.float32)
        if w_scales.size != channels:
            raise RuntimeError(
                f"Layer {name}: weight_scale file has {w_scales.size} elements "
                f"but manifest says channels={channels} ({w_scale_path})"
            )

        # Compute real multipliers and convert to fixed point
        real_m = (s_in * w_scales) / s_out  # shape [channels], float64-ish
        mults = np.zeros(channels, dtype=np.int32)
        shifts = np.zeros(channels, dtype=np.int8)
        for c in range(channels):
            m, s = real_m_to_fixed(real_m[c])
            mults[c] = m
            shifts[c] = s

        # Write binaries
        safe = name.replace(".", "_")
        mult_file = f"{safe}.requant_mult.bin"
        shift_file = f"{safe}.requant_shift.bin"
        mults.tofile(str(out_dir / mult_file))
        shifts.tofile(str(out_dir / shift_file))

        # Track stats
        shift_stats.append((int(shifts.min()), int(shifts.max())))
        int_mult_stats.append((int(mults.min()), int(mults.max())))

        out_layers.append({
            "index": L["index"],
            "name": name,
            "op": L["op"],
            "channels": channels,
            "per_channel_axis": L.get("per_channel_axis"),
            "input_scale": s_in,
            "output_scale": s_out,
            "weight_int8_file": L["weight_int8_file"],
            "weight_scale_f32_file": w_scale_file,
            "requant_mult_file": mult_file,
            "requant_shift_file": shift_file,
            "requant_mult_bytes": int(mults.nbytes),
            "requant_shift_bytes": int(shifts.nbytes),
            "real_m_min": float(real_m.min()),
            "real_m_max": float(real_m.max()),
            "shift_min": int(shifts.min()),
            "shift_max": int(shifts.max()),
            "int_mult_min": int(mults.min()),
            "int_mult_max": int(mults.max()),
        })

    if missing_act:
        print(f"\n  WARNING: {len(missing_act)} layers missing from activation_scales.json:")
        for n in missing_act:
            print(f"    - {n}")

    # ---- Write summary manifest
    out_manifest = {
        "source_int8_manifest": str(int8_manifest_path.name),
        "source_activation_scales": os.path.basename(args.activation_scales),
        "num_layers": len(out_layers),
        "int_mult_target_range": [MIN_MULT, MAX_MULT],
        "int_mult_dtype": "int32",
        "shift_dtype": "int8",
        "fixed_point_convention":
            "scaled_i32 = (int64)acc_i32 * int_mult[c] >> shift[c]",
        "layers": out_layers,
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(out_manifest, f, indent=2)

    # ---- Global stats for sanity
    if shift_stats:
        all_shift_min = min(s[0] for s in shift_stats)
        all_shift_max = max(s[1] for s in shift_stats)
        all_mult_min = min(m[0] for m in int_mult_stats)
        all_mult_max = max(m[1] for m in int_mult_stats)
        print(f"\n--- Summary ---")
        print(f"  Layers written:   {len(out_layers)}")
        print(f"  Shift range:      [{all_shift_min}, {all_shift_max}]")
        print(f"  int_mult range:   [{all_mult_min}, {all_mult_max}]")
        if all_shift_min < 0:
            print(f"  Note: negative shifts present — real_m > 1 on some channels.")
        if all_mult_min < MIN_MULT // 2:
            print(f"  WARNING: some int_mult < {MIN_MULT // 2}; precision may be low for those channels.")

    # ---- Spot-check first 5 layers
    print("\nFirst 5 layers:")
    for L in out_layers[:5]:
        print(f"  [{L['index']:2d}] {L['name']:<40s} "
              f"C={L['channels']:3d}  "
              f"real_m=[{L['real_m_min']:.2e}, {L['real_m_max']:.2e}]  "
              f"shift=[{L['shift_min']:2d}, {L['shift_max']:2d}]  "
              f"int_mult=[{L['int_mult_min']}, {L['int_mult_max']}]")

    print(f"\nWrote manifest: {manifest_path}")
    print(f"Wrote {len(out_layers) * 2} .bin files to {out_dir}")


if __name__ == "__main__":
    main()
