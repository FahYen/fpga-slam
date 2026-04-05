#!/usr/bin/env python3
"""
Quantization feasibility study for RangeNet DarkNet53 → FPGA accelerator.

Compares FP32, INT8/INT16 weight-only, and INT8 weight+activation quantized
inference against golden GPU reference labels. Reports per-frame and aggregate
pixel mismatch rates to determine if post-training quantization is viable.

Usage:
    python quantize_feasibility.py \
        --model /path/to/model_dir \
        --scan-root /path/to/kitti/sequences \
        --golden-root /path/to/rangenet_gpu_labels/kitti \
        --sequence 00 --num-frames 10
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

import __init__ as booger  # noqa: F401 – sets TRAIN_PATH for local imports
from common.laserscan import LaserScan
from tasks.semantic.modules.segmentator import Segmentator


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_projection_tensors(scan, sensor_cfg):
    img_means = torch.tensor(sensor_cfg["img_means"], dtype=torch.float32)
    img_stds = torch.tensor(sensor_cfg["img_stds"], dtype=torch.float32)

    proj_range = torch.from_numpy(scan.proj_range).clone()
    proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
    proj_remission = torch.from_numpy(scan.proj_remission).clone()
    proj_mask = torch.from_numpy(scan.proj_mask)

    proj = torch.cat([
        proj_range.unsqueeze(0),
        proj_xyz.permute(2, 0, 1),
        proj_remission.unsqueeze(0),
    ])
    proj = (proj - img_means[:, None, None]) / img_stds[:, None, None]
    proj = proj * proj_mask.float()

    proj_x = torch.from_numpy(scan.proj_x).long()
    proj_y = torch.from_numpy(scan.proj_y).long()
    return proj, proj_mask, proj_x, proj_y


def run_inference(model, proj, proj_mask, proj_x, proj_y, device):
    with torch.no_grad():
        proj_in = proj.unsqueeze(0).to(device)
        proj_mask_in = proj_mask.unsqueeze(0).to(device)
        proj_output = model(proj_in, proj_mask_in)
        proj_argmax = proj_output[0].argmax(dim=0)
        unproj_argmax = proj_argmax[proj_y.to(device), proj_x.to(device)]
        if device.type == "cuda":
            torch.cuda.synchronize()
    return unproj_argmax.cpu().numpy().astype(np.int32)


# ---------------------------------------------------------------------------
# Fake quantization helpers
# ---------------------------------------------------------------------------

def fake_quantize_tensor(x, bits, per_channel_dim=None):
    """Symmetric fake quantization (round-trip through integer grid)."""
    qmax = 2 ** (bits - 1) - 1
    if per_channel_dim is not None:
        dims = [d for d in range(x.ndim) if d != per_channel_dim]
        shape = [1] * x.ndim
        shape[per_channel_dim] = x.shape[per_channel_dim]
        max_val = x.abs().amax(dim=dims).reshape(shape).clamp(min=1e-8)
    else:
        max_val = x.abs().max().clamp(min=1e-8)
    scale = max_val / qmax
    return (x / scale).round().clamp(-qmax - 1, qmax) * scale


def fake_quantize_weights(model, bits=8):
    """In-place per-output-channel symmetric weight quantization."""
    count = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            m.weight.data = fake_quantize_tensor(
                m.weight.data, bits, per_channel_dim=0
            )
            if m.bias is not None:
                m.bias.data = fake_quantize_tensor(m.bias.data, bits)
            count += 1
    return count


class ActivationCollector:
    """Forward hooks that record per-layer min/max activation ranges."""

    def __init__(self):
        self.ranges = {}
        self._hooks = []

    def register(self, model):
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d,
                              nn.BatchNorm2d, nn.LeakyReLU)):
                self._hooks.append(m.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def fn(module, inp, out):
            if isinstance(out, tuple):
                return
            v = out.detach()
            lo, hi = v.min().item(), v.max().item()
            if name not in self.ranges:
                self.ranges[name] = [lo, hi]
            else:
                self.ranges[name][0] = min(self.ranges[name][0], lo)
                self.ranges[name][1] = max(self.ranges[name][1], hi)
        return fn

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class ActivationQuantizer:
    """Forward hooks that fake-quantize activations using pre-calibrated ranges."""

    def __init__(self, ranges, bits=8):
        self.ranges = ranges
        self.bits = bits
        self._hooks = []

    def register(self, model):
        for name, m in model.named_modules():
            if name in self.ranges:
                self._hooks.append(m.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        lo, hi = self.ranges[name]
        max_abs = max(abs(lo), abs(hi), 1e-8)
        qmax = 2 ** (self.bits - 1) - 1
        scale = max_abs / qmax

        def fn(module, inp, out):
            if isinstance(out, tuple):
                return out
            return (out / scale).round().clamp(-qmax - 1, qmax) * scale
        return fn

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class PerChannelActivationCollector:
    """Forward hooks that record per-channel (dim=1) min/max activation ranges."""

    def __init__(self):
        self.ranges = {}
        self._hooks = []

    def register(self, model):
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d,
                              nn.BatchNorm2d, nn.LeakyReLU)):
                self._hooks.append(m.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def fn(module, inp, out):
            if isinstance(out, tuple):
                return
            v = out.detach()
            if v.ndim < 2:
                return
            # per-channel: reduce over N, H, W → shape [C]
            lo = v.amin(dim=[0] + list(range(2, v.ndim)))
            hi = v.amax(dim=[0] + list(range(2, v.ndim)))
            if name not in self.ranges:
                self.ranges[name] = [lo, hi]
            else:
                self.ranges[name][0] = torch.min(self.ranges[name][0], lo)
                self.ranges[name][1] = torch.max(self.ranges[name][1], hi)
        return fn

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class PerChannelActivationQuantizer:
    """Forward hooks that fake-quantize activations per-channel using calibrated ranges."""

    def __init__(self, ranges, bits=8):
        self.ranges = ranges
        self.bits = bits
        self._hooks = []

    def register(self, model):
        for name, m in model.named_modules():
            if name in self.ranges:
                self._hooks.append(m.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        lo, hi = self.ranges[name]
        max_abs = torch.max(lo.abs(), hi.abs()).clamp(min=1e-8)
        qmax = 2 ** (self.bits - 1) - 1
        scale = (max_abs / qmax).reshape(1, -1, *([1] * (4 - 2)))

        def fn(module, inp, out):
            if isinstance(out, tuple):
                return out
            s = scale.to(out.device)
            return (out / s).round().clamp(-qmax - 1, qmax) * s
        return fn

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Golden label loading
# ---------------------------------------------------------------------------

def load_golden_labels(label_path, learning_map):
    """Load .label file (raw label space) and map to reduced 0..N-1 space."""
    raw = np.fromfile(str(label_path), dtype=np.int32)
    semantic = raw & 0xFFFF
    max_key = max(int(k) for k in learning_map.keys())
    lut = np.zeros(max_key + 100, dtype=np.int32)
    for k, v in learning_map.items():
        lut[int(k)] = int(v)
    return lut[semantic]


def compare(pred, golden):
    total = len(golden)
    mismatched = int(np.sum(pred != golden))
    return {
        "total_points": total,
        "mismatched_points": mismatched,
        "mismatch_pct": round(100.0 * mismatched / max(total, 1), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RangeNet INT8/INT16 quantization feasibility for FPGA"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--scan-root", required=True)
    parser.add_argument("--golden-root", required=True)
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--output", default="quantize_feasibility_report.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model_dir = Path(args.model).resolve()
    arch_cfg = load_yaml(model_dir / "arch_cfg.yaml")
    data_cfg = load_yaml(model_dir / "data_cfg.yaml")
    nclasses = len(data_cfg["learning_map_inv"])
    sensor_cfg = arch_cfg["dataset"]["sensor"]
    learning_map = data_cfg["learning_map"]

    scan_dir = Path(args.scan_root).resolve() / args.sequence / "velodyne"
    golden_dir = Path(args.golden_root).resolve() / args.sequence / "predictions"

    scan_paths = sorted(scan_dir.glob("*.bin"))[:args.num_frames]
    if not scan_paths:
        raise FileNotFoundError(f"No .bin files in {scan_dir}")

    device_str = args.device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Device: {device}  |  Classes: {nclasses}  |  Frames: {len(scan_paths)}")

    # Load FP32 model
    with torch.no_grad():
        model_fp32 = Segmentator(arch_cfg, nclasses, str(model_dir))
    model_fp32.to(device).eval()

    # Pre-load scans + golden labels
    scans = []
    for sp in scan_paths:
        scan = LaserScan(
            project=True,
            H=sensor_cfg["img_prop"]["height"],
            W=sensor_cfg["img_prop"]["width"],
            fov_up=sensor_cfg["fov_up"],
            fov_down=sensor_cfg["fov_down"],
        )
        scan.open_scan(str(sp))
        proj, proj_mask, proj_x, proj_y = build_projection_tensors(scan, sensor_cfg)
        gpath = golden_dir / sp.with_suffix(".label").name
        golden = load_golden_labels(gpath, learning_map) if gpath.exists() else None
        scans.append(dict(
            name=sp.stem, num_points=scan.size(),
            proj=proj, proj_mask=proj_mask, proj_x=proj_x, proj_y=proj_y,
            golden=golden,
        ))

    report = {
        "config": {
            "model_dir": str(model_dir),
            "nclasses": nclasses,
            "device": str(device),
            "num_frames": len(scan_paths),
        },
        "frames": [],
        "aggregate": {},
    }

    def infer_all(model):
        preds = []
        for s in scans:
            preds.append(run_inference(
                model, s["proj"], s["proj_mask"], s["proj_x"], s["proj_y"], device
            ))
        return preds

    def ensure_frame(name, num_points):
        for f in report["frames"]:
            if f["frame"] == name:
                return f
        f = {"frame": name, "num_points": num_points}
        report["frames"].append(f)
        return f

    # Phase 1 — FP32 baseline
    t0 = time.time()
    print("Phase 1: FP32 baseline")
    fp32_preds = infer_all(model_fp32)
    print(f"  done in {time.time()-t0:.1f}s")

    for i, s in enumerate(scans):
        f = ensure_frame(s["name"], s["num_points"])
        if s["golden"] is not None:
            f["fp32_vs_golden"] = compare(fp32_preds[i], s["golden"])

    # Phase 2 — weight-only quantization (INT8 and INT16)
    for bits in [8, 16]:
        tag = f"int{bits}_wt"
        t0 = time.time()
        print(f"Phase 2: {tag} weight-only quantization")
        m = copy.deepcopy(model_fp32)
        nlayers = fake_quantize_weights(m, bits=bits)
        print(f"  quantized {nlayers} conv layers")
        preds = infer_all(m)
        print(f"  done in {time.time()-t0:.1f}s")
        del m
        for i, s in enumerate(scans):
            f = ensure_frame(s["name"], s["num_points"])
            f[f"{tag}_vs_fp32"] = compare(preds[i], fp32_preds[i])
            if s["golden"] is not None:
                f[f"{tag}_vs_golden"] = compare(preds[i], s["golden"])

    # Phase 3 — per-layer weight+activation quantization (INT8 and INT16)
    for bits in [8, 16]:
        tag = f"int{bits}_wt_act"
        t0 = time.time()
        print(f"Phase 3: {tag} per-layer — calibrating")
        m = copy.deepcopy(model_fp32)
        fake_quantize_weights(m, bits=bits)
        collector = ActivationCollector()
        collector.register(m)
        _ = infer_all(m)
        collector.remove()

        quantizer = ActivationQuantizer(collector.ranges, bits=bits)
        quantizer.register(m)
        print(f"  {len(collector.ranges)} activation layers quantized — evaluating")
        preds = infer_all(m)
        quantizer.remove()
        del m
        print(f"  done in {time.time()-t0:.1f}s")

        for i, s in enumerate(scans):
            f = ensure_frame(s["name"], s["num_points"])
            f[f"{tag}_vs_fp32"] = compare(preds[i], fp32_preds[i])
            if s["golden"] is not None:
                f[f"{tag}_vs_golden"] = compare(preds[i], s["golden"])

    # Phase 4 — per-channel activation quantization (INT8 only)
    tag = "int8_wt_act_perchan"
    t0 = time.time()
    print(f"Phase 4: {tag} — calibrating per-channel")
    m = copy.deepcopy(model_fp32)
    fake_quantize_weights(m, bits=8)
    pc_collector = PerChannelActivationCollector()
    pc_collector.register(m)
    _ = infer_all(m)
    pc_collector.remove()

    pc_quantizer = PerChannelActivationQuantizer(pc_collector.ranges, bits=8)
    pc_quantizer.register(m)
    print(f"  {len(pc_collector.ranges)} activation layers quantized per-channel — evaluating")
    preds = infer_all(m)
    pc_quantizer.remove()
    del m
    print(f"  done in {time.time()-t0:.1f}s")

    for i, s in enumerate(scans):
        f = ensure_frame(s["name"], s["num_points"])
        f[f"{tag}_vs_fp32"] = compare(preds[i], fp32_preds[i])
        if s["golden"] is not None:
            f[f"{tag}_vs_golden"] = compare(preds[i], s["golden"])

    # Aggregate
    keys = [
        "fp32_vs_golden",
        "int8_wt_vs_fp32", "int8_wt_vs_golden",
        "int16_wt_vs_fp32", "int16_wt_vs_golden",
        "int8_wt_act_vs_fp32", "int8_wt_act_vs_golden",
        "int16_wt_act_vs_fp32", "int16_wt_act_vs_golden",
        "int8_wt_act_perchan_vs_fp32", "int8_wt_act_perchan_vs_golden",
    ]
    for key in keys:
        vals = [f[key]["mismatch_pct"] for f in report["frames"] if key in f]
        if vals:
            report["aggregate"][key] = {
                "mean_mismatch_pct": round(sum(vals) / len(vals), 4),
                "max_mismatch_pct": round(max(vals), 4),
                "min_mismatch_pct": round(min(vals), 4),
            }

    out_path = Path(args.output)
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    print(f"\nReport → {out_path}")
    print("\n=== SUMMARY ===")
    for key, agg in report["aggregate"].items():
        print(f"  {key:30s}  mean={agg['mean_mismatch_pct']:6.2f}%  "
              f"max={agg['max_mismatch_pct']:6.2f}%  "
              f"min={agg['min_mismatch_pct']:6.2f}%")


if __name__ == "__main__":
    main()
