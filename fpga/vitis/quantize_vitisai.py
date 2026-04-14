#!/usr/bin/env python3
"""Vitis AI post-training quantization (PTQ) for RangeNet DarkNet53.

Runs inside the Vitis AI 3.0 Docker container where pytorch_nndct is available.

Workflow:
  1. Load pretrained FP32 Segmentator (backbone + decoder + head).
  2. Wrap it in a quantizer-friendly forward (no softmax, no .detach()).
  3. Calibrate INT8 per-channel scales on real KITTI LiDAR frames.
  4. Optionally evaluate quantized accuracy vs FP32 baseline.
  5. Export quantized xmodel for the Vitis AI Compiler (vai_c_xir).

Usage (inside Vitis AI Docker):
    # Step 1: Calibrate
    python fpga/vitis/quantize_vitisai.py \
        --model /workspace/models/rangenet_darknet53 \
        --scan-root /workspace/data/kitti/sequences \
        --sequence 00 \
        --num-calib-frames 200 \
        --output-dir /workspace/vitis_output/quantize_result \
        --device cuda \
        --quant-mode calib

    # Step 2: Test + export xmodel
    python fpga/vitis/quantize_vitisai.py \
        --model /workspace/models/rangenet_darknet53 \
        --scan-root /workspace/data/kitti/sequences \
        --sequence 00 \
        --num-test-frames 50 \
        --output-dir /workspace/vitis_output/quantize_result \
        --device cuda \
        --quant-mode test

After calibration + export, compile with:
    bash fpga/vitis/compile_vitisai.sh /workspace/vitis_output/quantize_result
"""

import argparse
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

# ---------------------------------------------------------------------------
# Resolve RangeNet import paths — the repo layout requires this
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # fpga/vitis/
FPGA_DIR = SCRIPT_DIR.parent                          # fpga/
ROOT_DIR = FPGA_DIR.parent                            # repo root
RANGENET_TRAIN = ROOT_DIR / "RangeNet" / "train"

# RangeNet expects TRAIN_PATH on sys.path via its __init__.py
sys.path.insert(0, str(RANGENET_TRAIN))
sys.path.insert(0, str(RANGENET_TRAIN / "tasks" / "semantic"))

from common.laserscan import LaserScan
from tasks.semantic.modules.segmentator import Segmentator


# ---------------------------------------------------------------------------
# Quantizer-friendly wrapper
# ---------------------------------------------------------------------------

class SegmentatorForQuantization(nn.Module):
    """Thin wrapper around Segmentator that makes it Vitis-AI-friendly.

    Changes vs the original forward():
      - Removes F.softmax (DPU doesn't support it; argmax on logits is equivalent).
      - Removes .detach() on skip connections (breaks quantizer graph tracing).
      - Removes CRF post-processing (not relevant for DPU deployment).
      - Removes Dropout (identity at eval, but can confuse the tracer).
      - Accepts a single tensor (no mask arg) since DPU graphs are single-input.
    """

    def __init__(self, segmentator):
        super().__init__()
        self.backbone = segmentator.backbone
        self.decoder = segmentator.decoder
        self.head = segmentator.head

        # Patch backbone and decoder: replace run_layer to remove .detach()
        self._patch_skip_detach()

    def _patch_skip_detach(self):
        """Monkey-patch backbone.run_layer and decoder.run_layer to drop .detach()."""

        def backbone_run_layer(self_bb, x, layer, skips, os):
            y = layer(x)
            if y.shape[2] < x.shape[2] or y.shape[3] < x.shape[3]:
                skips[os] = x  # no .detach()
                os *= 2
            return y, skips, os

        def decoder_run_layer(self_dec, x, layer, skips, os):
            feats = layer(x)
            if feats.shape[-1] > x.shape[-1]:
                os //= 2
                feats = feats + skips[os]  # no .detach()
            return feats, skips, os

        self.backbone.run_layer = types.MethodType(backbone_run_layer, self.backbone)
        self.decoder.run_layer = types.MethodType(decoder_run_layer, self.decoder)

    def forward(self, x):
        y, skips = self.backbone(x)
        y = self.decoder(y, skips)
        y = self.head(y)
        # Return raw logits — no softmax
        return y


# ---------------------------------------------------------------------------
# Calibration data loader
# ---------------------------------------------------------------------------

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_calibration_tensors(scan_paths, sensor_cfg, max_frames):
    """Load KITTI scans and produce normalized 5-channel projected tensors."""
    img_means = torch.tensor(sensor_cfg["img_means"], dtype=torch.float32)
    img_stds = torch.tensor(sensor_cfg["img_stds"], dtype=torch.float32)

    H = sensor_cfg["img_prop"]["height"]
    W = sensor_cfg["img_prop"]["width"]

    tensors = []
    for sp in scan_paths[:max_frames]:
        scan = LaserScan(
            project=True, H=H, W=W,
            fov_up=sensor_cfg["fov_up"],
            fov_down=sensor_cfg["fov_down"],
        )
        scan.open_scan(str(sp))

        proj_range = torch.from_numpy(scan.proj_range).clone()
        proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
        proj_remission = torch.from_numpy(scan.proj_remission).clone()
        proj_mask = torch.from_numpy(scan.proj_mask).float()

        proj = torch.cat([
            proj_range.unsqueeze(0),
            proj_xyz.permute(2, 0, 1),
            proj_remission.unsqueeze(0),
        ])
        proj = (proj - img_means[:, None, None]) / img_stds[:, None, None]
        proj = proj * proj_mask

        tensors.append(proj.unsqueeze(0))  # [1, 5, H, W]

    return tensors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Vitis AI PTQ quantization for RangeNet DarkNet53"
    )
    parser.add_argument("--model", required=True,
                        help="Path to pretrained model dir (arch_cfg.yaml + weights)")
    parser.add_argument("--scan-root", required=True,
                        help="Path to KITTI sequences root")
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--num-calib-frames", type=int, default=200,
                        help="Number of frames for calibration (100-500 recommended)")
    parser.add_argument("--num-test-frames", type=int, default=50,
                        help="Number of frames for post-quant accuracy test")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write quantization results")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quant-mode", default="calib",
                        choices=["calib", "test"],
                        help="'calib' to calibrate, 'test' to evaluate + export xmodel")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------
    model_dir = Path(args.model).resolve()
    arch_cfg = load_yaml(model_dir / "arch_cfg.yaml")
    data_cfg = load_yaml(model_dir / "data_cfg.yaml")
    nclasses = len(data_cfg["learning_map_inv"])
    sensor_cfg = arch_cfg["dataset"]["sensor"]

    scan_dir = Path(args.scan_root).resolve() / args.sequence / "velodyne"
    scan_paths = sorted(scan_dir.glob("*.bin"))
    if not scan_paths:
        raise FileNotFoundError(f"No .bin files in {scan_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else FPGA_DIR / "vitis_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    device_str = args.device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"Device:           {device}")
    print(f"Classes:          {nclasses}")
    print(f"Available scans:  {len(scan_paths)}")
    print(f"Calib frames:     {args.num_calib_frames}")
    print(f"Output dir:       {output_dir}")
    print(f"Quant mode:       {args.quant_mode}")

    # ------------------------------------------------------------------
    # Load FP32 model
    # ------------------------------------------------------------------
    print("\nLoading FP32 Segmentator ...")
    with torch.no_grad():
        segmentator = Segmentator(arch_cfg, nclasses, str(model_dir))

    # Wrap for quantization
    model = SegmentatorForQuantization(segmentator)
    model.to(device).eval()
    print("  Model wrapped for quantization (no softmax, no .detach())")

    # ------------------------------------------------------------------
    # Create dummy input for quantizer graph tracing
    # ------------------------------------------------------------------
    H = sensor_cfg["img_prop"]["height"]
    W = sensor_cfg["img_prop"]["width"]
    input_depth = segmentator.backbone.get_input_depth()
    dummy_input = torch.randn(1, input_depth, H, W).to(device)

    # ------------------------------------------------------------------
    # Import Vitis AI quantizer
    # ------------------------------------------------------------------
    try:
        from pytorch_nndct.apis import torch_quantizer, dump_xmodel
    except ImportError:
        print("\nERROR: pytorch_nndct not found.")
        print("This script must be run inside the Vitis AI 3.0 Docker container:")
        print("  docker pull xilinx/vitis-ai-pytorch-gpu:3.0.0")
        print("  docker run -it --gpus all -v /workspace:/workspace \\")
        print("    xilinx/vitis-ai-pytorch-gpu:3.0.0 bash")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Create quantizer
    # ------------------------------------------------------------------
    print(f"\nCreating Vitis AI quantizer (mode={args.quant_mode}) ...")
    quantizer = torch_quantizer(
        quant_mode=args.quant_mode,
        module=model,
        input_args=(dummy_input,),
        output_dir=str(output_dir),
        bitwidth=8,
        device=device,
    )
    quant_model = quantizer.quant_model

    # ------------------------------------------------------------------
    # Calibration or Test
    # ------------------------------------------------------------------
    if args.quant_mode == "calib":
        print(f"\nLoading {args.num_calib_frames} calibration frames ...")
        calib_tensors = build_calibration_tensors(
            scan_paths, sensor_cfg, args.num_calib_frames
        )
        print(f"  Loaded {len(calib_tensors)} frames")

        print("\nRunning calibration ...")
        quant_model.eval()
        with torch.no_grad():
            for i, x in enumerate(calib_tensors):
                x = x.to(device)
                _ = quant_model(x)
                if (i + 1) % 50 == 0 or (i + 1) == len(calib_tensors):
                    print(f"  Calibrated {i + 1}/{len(calib_tensors)} frames")

        # Export calibration config (quant_info.json / scale factors)
        quantizer.export_quant_config()
        print(f"\nCalibration complete. Results in: {output_dir}")
        print("\nNext step: re-run with --quant-mode test to evaluate + export xmodel")

    elif args.quant_mode == "test":
        print(f"\nLoading {args.num_test_frames} test frames ...")
        test_tensors = build_calibration_tensors(
            scan_paths, sensor_cfg, args.num_test_frames
        )

        # Also run FP32 for comparison
        print("Running FP32 baseline ...")
        fp32_preds = []
        model.eval()
        with torch.no_grad():
            for x in test_tensors:
                x = x.to(device)
                logits = model(x)
                fp32_preds.append(logits.argmax(dim=1).cpu().numpy())

        print("Running INT8 quantized ...")
        int8_preds = []
        quant_model.eval()
        with torch.no_grad():
            for x in test_tensors:
                x = x.to(device)
                logits = quant_model(x)
                int8_preds.append(logits.argmax(dim=1).cpu().numpy())

        # Compare
        total_pixels = 0
        total_mismatch = 0
        for fp32_p, int8_p in zip(fp32_preds, int8_preds):
            total_pixels += fp32_p.size
            total_mismatch += int(np.sum(fp32_p != int8_p))

        mismatch_pct = 100.0 * total_mismatch / max(total_pixels, 1)
        print(f"\n=== INT8 vs FP32 Accuracy ===")
        print(f"  Total pixels:     {total_pixels}")
        print(f"  Mismatched:       {total_mismatch}")
        print(f"  Mismatch rate:    {mismatch_pct:.3f}%")

        if mismatch_pct > 5.0:
            print("  WARNING: >5% mismatch — consider quantization-aware training")
        elif mismatch_pct > 1.0:
            print("  ACCEPTABLE: 1-5% mismatch — typical for PTQ")
        else:
            print("  EXCELLENT: <1% mismatch")

        # Export deployable xmodel
        print(f"\nExporting xmodel to {output_dir} ...")
        quantizer.export_xmodel(output_dir=str(output_dir))
        print("  xmodel exported successfully")
        print(f"\nNext step: compile with vai_c_xir:")
        print(f"  bash fpga/vitis/compile_vitisai.sh {output_dir}")


if __name__ == "__main__":
    main()