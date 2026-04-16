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
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

# ---------------------------------------------------------------------------
# Resolve RangeNet import paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
FPGA_DIR = SCRIPT_DIR.parent
ROOT_DIR = FPGA_DIR.parent
RANGENET_TRAIN = ROOT_DIR / "RangeNet" / "train"

sys.path.insert(0, str(RANGENET_TRAIN))
sys.path.insert(0, str(RANGENET_TRAIN / "tasks" / "semantic"))

from common.laserscan import LaserScan
from tasks.semantic.modules.segmentator import Segmentator

_train_path_abs = str(RANGENET_TRAIN)
for _mod_key in list(sys.modules.keys()):
    mod = sys.modules[_mod_key]
    if hasattr(mod, 'TRAIN_PATH'):
        mod.TRAIN_PATH = _train_path_abs


# ---------------------------------------------------------------------------
# Quantizer-friendly wrapper
# ---------------------------------------------------------------------------

class SegmentatorForQuantization(nn.Module):
    def __init__(self, segmentator):
        super().__init__()
        self.backbone = segmentator.backbone
        self.decoder = segmentator.decoder
        self.head = segmentator.head
        self._patch_skip_detach()

    def _patch_skip_detach(self):
        def backbone_run_layer(self_bb, x, layer, skips, os):
            y = layer(x)
            if y.shape[2] < x.shape[2] or y.shape[3] < x.shape[3]:
                skips[os] = x  
                os *= 2
            return y, skips, os

        def decoder_run_layer(self_dec, x, layer, skips, os):
            feats = layer(x)
            if feats.shape[-1] > x.shape[-1]:
                os //= 2
                feats = feats + skips[os]  
            return feats, skips, os

        self.backbone.run_layer = types.MethodType(backbone_run_layer, self.backbone)
        self.decoder.run_layer = types.MethodType(decoder_run_layer, self.decoder)

    def forward(self, x):
        y, skips = self.backbone(x)
        y = self.decoder(y, skips)
        y = self.head(y)
        return y


# ---------------------------------------------------------------------------
# Calibration data loader
# ---------------------------------------------------------------------------

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_calibration_tensors(scan_paths, sensor_cfg, max_frames, shuffle=False):
    img_means = torch.tensor(sensor_cfg["img_means"], dtype=torch.float32)
    img_stds = torch.tensor(sensor_cfg["img_stds"], dtype=torch.float32)

    H = sensor_cfg["img_prop"]["height"]
    W = sensor_cfg["img_prop"]["width"]

    paths_to_use = list(scan_paths)
    if shuffle:
        random.seed(42) 
        random.shuffle(paths_to_use)

    tensors = []
    for sp in paths_to_use[:max_frames]:
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
        
        tensors.append((proj.unsqueeze(0), proj_mask.unsqueeze(0))) 

    return tensors

class FastFinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, tensors):
        self.tensors = tensors
    def __len__(self):
        return len(self.tensors)
    def __getitem__(self, idx):
        return self.tensors[idx][0].squeeze(0), 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Vitis AI PTQ for RangeNet DarkNet53")
    parser.add_argument("--model", required=True)
    parser.add_argument("--scan-root", required=True)
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--num-calib-frames", type=int, default=200)
    parser.add_argument("--num-test-frames", type=int, default=200)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quant-mode", default="calib",
                        choices=["calib", "fast_finetune", "test"],
                        help="Sequence: calib -> fast_finetune -> test")
    args = parser.parse_args()

    model_dir = Path(args.model).resolve()
    arch_cfg = load_yaml(model_dir / "arch_cfg.yaml")
    data_cfg = load_yaml(model_dir / "data_cfg.yaml")
    nclasses = len(data_cfg["learning_map_inv"])
    sensor_cfg = arch_cfg["dataset"]["sensor"]

    scan_dir = Path(args.scan_root).resolve() / args.sequence / "velodyne"
    scan_paths = sorted(scan_dir.glob("*.bin"))

    output_dir = Path(args.output_dir) if args.output_dir else FPGA_DIR / "vitis_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)

    print("\nLoading FP32 Segmentator ...")
    with torch.no_grad():
        segmentator = Segmentator(arch_cfg, nclasses, str(model_dir))

    model = SegmentatorForQuantization(segmentator).to(device).eval()
    
    dummy_scan = build_calibration_tensors(scan_paths, sensor_cfg, 1, shuffle=False)[0][0]
    dummy_input = dummy_scan.clone().to(device)

    try:
        from pytorch_nndct.apis import torch_quantizer, dump_xmodel
    except ImportError:
        print("\nERROR: pytorch_nndct not found. Run inside Vitis AI Docker.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # PHASE 1 & 3: CALIB AND TEST
    # ------------------------------------------------------------------
    if args.quant_mode in ["calib", "test"]:
        quantizer = torch_quantizer(
            quant_mode=args.quant_mode,
            module=model,
            input_args=(dummy_input,),
            output_dir=str(output_dir),
            bitwidth=8,
            device=device,
        )
        quant_model = quantizer.quant_model

        if args.quant_mode == "calib":
            print(f"\n[PHASE 1] Running Standard Calibration ({args.num_calib_frames} frames)...")
            calib_tensors = build_calibration_tensors(scan_paths, sensor_cfg, args.num_calib_frames, shuffle=True)
            
            quant_model.eval()
            with torch.no_grad():
                for i, (x, _) in enumerate(calib_tensors):
                    _ = quant_model(x.to(device))
            
            quantizer.export_quant_config()
            print("Calibration complete. Next step: run with --quant-mode fast_finetune")

        elif args.quant_mode == "test":
            print(f"\n[PHASE 3] Running Accuracy Test ({args.num_test_frames} frames)...")
            test_tensors = build_calibration_tensors(scan_paths, sensor_cfg, args.num_test_frames, shuffle=False)

            fp32_preds, masks, int8_preds = [], [], []
            
            model.eval()
            with torch.no_grad():
                for x, mask in test_tensors:
                    fp32_preds.append(model(x.to(device)).argmax(dim=1).cpu().numpy())
                    masks.append(mask.cpu().numpy().astype(bool)) 

            quant_model.eval()
            with torch.no_grad():
                for x, _ in test_tensors:
                    int8_preds.append(quant_model(x.to(device)).argmax(dim=1).cpu().numpy())

            total_pixels, total_mismatch = 0, 0
            for fp32_p, int8_p, mask in zip(fp32_preds, int8_preds, masks):
                fp32_valid = fp32_p[mask]
                int8_valid = int8_p[mask]
                total_pixels += fp32_valid.size
                total_mismatch += int(np.sum(fp32_valid != int8_valid))

            print(f"\n=== INT8 vs FP32 Accuracy ===")
            print(f"  Total pixels:  {total_pixels}")
            print(f"  Mismatched:    {total_mismatch}")
            print(f"  Mismatch rate: {(100.0 * total_mismatch / max(total_pixels, 1)):.3f}%")

            print(f"\nExporting xmodel to {output_dir} ...")
            quantizer.export_xmodel(output_dir=str(output_dir))

    # ------------------------------------------------------------------
    # PHASE 2: FAST FINETUNE (AdaQuant)
    # ------------------------------------------------------------------
    elif args.quant_mode == "fast_finetune":
        try:
            from pytorch_nndct.qproc.ada_quant import AdvancedQuantProcessor
        except ImportError:
            print("\nERROR: AdvancedQuantProcessor not found. Ensure you are on Vitis AI 3.0+.")
            sys.exit(1)

        print(f"\n[PHASE 2] Running Fast Finetuning to align DarkNet skip connections...")
        calib_tensors = build_calibration_tensors(scan_paths, sensor_cfg, args.num_calib_frames, shuffle=True)
        
        # Batch size 1 keeps VRAM usage safe for massive LiDAR tensors
        dataset = FastFinetuneDataset(calib_tensors)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

        processor = AdvancedQuantProcessor(
            model,
            dummy_input,
            dataloader=dataloader,
            output_dir=str(output_dir),
            bitwidth=8,
            device=device
        )

        print("  Starting weight adjustment. This uses FP32 outputs as a target to fix INT8 scales...")
        _ = processor.finetune()
        processor.export_quant_config()
        
        print("\nFast Finetuning complete! Config updated.")
        print("Next step: run with --quant-mode test to evaluate and export.")


if __name__ == "__main__":
    main()