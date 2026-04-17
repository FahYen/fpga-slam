#!/usr/bin/env python3
"""Drive ./test_layer across all 68 layers of a golden frame.

Reads the golden manifest, the INT8 weight manifest, figures out the per-layer
shapes/attrs, and invokes the C++ binary once per layer.

Usage:
    python3 fpga/test/run_tests.py \
        --frame-dir fpga/golden_int8/frame_0000 \
        --int8-manifest fpga/weights/int8/manifest.json \
        --int8-dir fpga/weights/int8 \
        --requant-dir fpga/weights/requant \
        --binary fpga/test/test_layer

Optional:
    --only IDX        Run just one layer
    --stop-on-fail    Abort after first failure (useful when debugging)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frame-dir", default="fpga/golden_int8/frame_0000")
    p.add_argument("--int8-manifest", default="fpga/weights/int8/manifest.json")
    p.add_argument("--int8-dir", default="fpga/weights/int8")
    p.add_argument("--requant-dir", default="fpga/weights/requant")
    p.add_argument("--binary", default="fpga/test/test_layer")
    p.add_argument("--only", type=int, default=None,
                   help="Only test this layer index")
    p.add_argument("--op-filter", default=None,
                   help="Only test layers of this op type (Conv or ConvTranspose)")
    p.add_argument("--stop-on-fail", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    frame_dir = Path(args.frame_dir)
    int8_dir = Path(args.int8_dir)
    requant_dir = Path(args.requant_dir)
    binary = Path(args.binary)

    if not binary.exists():
        print(f"ERROR: binary not found: {binary}")
        print("Build it first:  cd fpga/test && make")
        sys.exit(2)

    with open(args.int8_manifest) as f:
        int8 = json.load(f)
    int8_map = {L["name"]: L for L in int8["layers"]}

    manifest_path = frame_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: frame manifest not found: {manifest_path}")
        print("Run generate_golden_int8.py first.")
        sys.exit(2)
    with open(manifest_path) as f:
        frame_m = json.load(f)

    results = []
    for L in frame_m["layers"]:
        idx = L["index"]
        name = L["name"]
        op = L["op"]

        if args.only is not None and idx != args.only:
            continue
        if args.op_filter and op != args.op_filter:
            continue

        i8 = int8_map[name]
        safe = name.replace(".", "_")

        kernel = i8["kernel"]          # [KH, KW]
        stride = i8["stride"]          # [sh, sw]
        padding = i8["padding"]        # [top, left, bottom, right] (ONNX)
        weight_shape = i8["weight_shape"]

        # Weight layout depends on op
        if op == "ConvTranspose":
            # [IC, OC, KH, KW]
            IC, OC = weight_shape[0], weight_shape[1]
        else:
            # [OC, IC, KH, KW]
            OC, IC = weight_shape[0], weight_shape[1]

        input_shape = L["input_shape"]   # [1, IC, H, W]
        _, _, H, W = input_shape

        cmd = [
            str(binary),
            "--input",    str(frame_dir / L["files"]["input_i8"]),
            "--weight",   str(int8_dir / i8["weight_int8_file"]),
            "--bias",     str(frame_dir / L["files"]["bias_i32"]),
            "--mult",     str(requant_dir / f"{safe}.requant_mult.bin"),
            "--shift",    str(requant_dir / f"{safe}.requant_shift.bin"),
            "--expected", str(frame_dir / L["files"]["output_i8"]),
            "--IC", str(IC), "--H", str(H), "--W", str(W),
            "--OC", str(OC), "--KH", str(kernel[0]), "--KW", str(kernel[1]),
            "--stride-h", str(stride[0]), "--stride-w", str(stride[1]),
            "--pad-h", str(padding[0]),   "--pad-w", str(padding[1]),
            "--op", op,
        ]
        if L["has_leaky_relu"]:
            cmd.append("--lrelu")
        if args.verbose:
            cmd.append("--verbose")

        label = f"[{idx:2d}] {name:<42s} {op:<15s}"
        print(f"{label}", end=" ", flush=True)

        r = subprocess.run(cmd, capture_output=True, text=True)
        out = (r.stdout or "").strip()
        err = (r.stderr or "").strip()
        ok = (r.returncode == 0)
        print(out if out else f"(no output, rc={r.returncode})")
        if err and (not ok or args.verbose):
            for line in err.splitlines():
                print(f"    | {line}")
        results.append((idx, name, ok, out))

        if not ok and args.stop_on_fail:
            break

    n_ok = sum(1 for _, _, ok, _ in results if ok)
    n_tot = len(results)
    print(f"\n=== Summary: {n_ok}/{n_tot} layers PASS ===")

    if n_ok < n_tot:
        print("\nFailed layers:")
        for idx, name, ok, out in results:
            if not ok:
                print(f"  [{idx:2d}] {name}: {out}")
        sys.exit(1)


if __name__ == "__main__":
    main()
