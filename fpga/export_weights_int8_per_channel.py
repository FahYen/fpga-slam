#!/usr/bin/env python3
"""Export full BN-folded per-channel INT8 weights for FPGA use.

Reads:
  data/pretrained_darknet53_weights/model.onnx

Writes:
  fpga/weights/int8_per_channel/<layer>.weight.int8.bin
  fpga/weights/int8_per_channel/<layer>.weight.scale.f32.bin
  fpga/weights/int8_per_channel/manifest.json
  fpga/weights/int8_per_channel/export_receipt.json
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import shape_inference
from onnx import numpy_helper

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
ONNX_PATH = os.path.join(ROOT_DIR, "data", "pretrained_darknet53_weights", "model.onnx")
OUT_DIR = os.path.join(SCRIPT_DIR, "weights", "int8_per_channel")
QMAX = 127.0
EPS = 1e-8


@dataclass
class ExportedLayer:
    index: int
    name: str
    op: str
    kernel: list[int]
    stride: list[int]
    padding: list[int]
    dilation: list[int]
    weight_shape: list[int]
    input_shape: list[int]
    output_shape: list[int]
    per_channel_axis: int
    channels: int
    weight_int8_file: str
    weight_scale_f32_file: str
    weight_int8_bytes: int
    weight_scale_f32_bytes: int
    bn_folded: bool
    scale_min: float
    scale_max: float
    scale_mean: float


def load_initializers(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}


def get_attr(node: onnx.NodeProto, name: str, default=None):
    for attr in node.attribute:
        if attr.name == name:
            if attr.type == onnx.AttributeProto.FLOAT:
                return attr.f
            if attr.type == onnx.AttributeProto.INT:
                return attr.i
            if attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
    return default


def build_maps(model: onnx.ModelProto):
    output_to_node = {}
    for node in model.graph.node:
        for output_name in node.output:
            output_to_node[output_name] = node
    return output_to_node


def identify_bn_pairs(model: onnx.ModelProto):
    output_to_node = build_maps(model)
    pairs = []
    for node in model.graph.node:
        if node.op_type != "BatchNormalization":
            continue
        producer_name = node.input[0]
        if producer_name not in output_to_node:
            continue
        conv = output_to_node[producer_name]
        if conv.op_type in ("Conv", "ConvTranspose"):
            pairs.append((conv, node))
    return pairs


def identify_softmax_tail(model: onnx.ModelProto):
    output_to_node = build_maps(model)
    for node in reversed(list(model.graph.node)):
        if node.op_type != "Div":
            continue
        num_in = node.input[0]
        den_in = node.input[1]
        if num_in not in output_to_node or output_to_node[num_in].op_type != "Exp":
            continue
        if den_in not in output_to_node or output_to_node[den_in].op_type != "ReduceSum":
            continue
        return [output_to_node[num_in], output_to_node[den_in], node]
    return []


def fold_bn(conv_node: onnx.NodeProto, bn_node: onnx.NodeProto, inits: dict[str, np.ndarray]):
    """Fold BatchNorm into Conv/ConvTranspose and return fused float32 params."""
    weight = inits[conv_node.input[1]].copy()
    has_bias = len(conv_node.input) >= 3 and conv_node.input[2] and conv_node.input[2] in inits

    out_channels = weight.shape[0] if conv_node.op_type == "Conv" else weight.shape[1]
    conv_bias = inits[conv_node.input[2]].copy() if has_bias else np.zeros(out_channels, dtype=np.float32)

    gamma = inits[bn_node.input[1]]
    beta = inits[bn_node.input[2]]
    mean = inits[bn_node.input[3]]
    var = inits[bn_node.input[4]]
    eps = get_attr(bn_node, "epsilon", 1e-5)

    factor = gamma / np.sqrt(var + eps)

    if conv_node.op_type == "Conv":
        weight_fused = weight * factor.reshape(-1, 1, 1, 1)
    else:
        weight_fused = weight * factor.reshape(1, -1, 1, 1)

    bias_fused = factor * (conv_bias - mean) + beta
    return weight_fused.astype(np.float32), bias_fused.astype(np.float32)


def per_channel_quantize(weight_f32: np.ndarray, axis: int):
    reduce_dims = tuple(d for d in range(weight_f32.ndim) if d != axis)
    max_abs = np.max(np.abs(weight_f32), axis=reduce_dims).astype(np.float64)
    scales = np.maximum(max_abs / QMAX, EPS).astype(np.float32)

    shape = [1] * weight_f32.ndim
    shape[axis] = scales.shape[0]
    q = np.rint(weight_f32 / scales.reshape(shape))
    q = np.clip(q, -128, 127).astype(np.int8)
    return q, scales


def extract_input_output_shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    shaped = shape_inference.infer_shapes(model)
    vi_map = {}
    for collection in (shaped.graph.value_info, shaped.graph.input, shaped.graph.output):
        for vi in collection:
            vi_map[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    return vi_map


def export() -> dict:
    start_s = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    model = onnx.load(ONNX_PATH)
    inits = load_initializers(model)
    vi_map = extract_input_output_shapes(model)

    softmax_nodes = identify_softmax_tail(model)
    softmax_set = {id(n) for n in softmax_nodes}
    pairs = identify_bn_pairs(model)
    bn_set = {id(bn) for _, bn in pairs}
    conv_to_bn = {id(conv): bn for conv, bn in pairs}

    exported: list[ExportedLayer] = []
    total_weight_int8_bytes = 0
    total_scale_bytes = 0

    for node in model.graph.node:
        node_id = id(node)
        if node_id in bn_set or node_id in softmax_set:
            continue
        if node.op_type not in ("Conv", "ConvTranspose"):
            continue

        if node_id in conv_to_bn:
            bn = conv_to_bn[node_id]
            weight_f32, _ = fold_bn(node, bn, inits)
            out_name = bn.output[0]
            bn_folded = True
        else:
            weight_f32 = inits[node.input[1]].astype(np.float32, copy=True)
            out_name = node.output[0]
            bn_folded = False

        axis = 0 if node.op_type == "Conv" else 1
        weight_int8, scales = per_channel_quantize(weight_f32, axis=axis)

        raw_name = node.input[1].replace(".weight", "")
        safe_name = raw_name.replace(".", "_")
        w_file = f"{safe_name}.weight.int8.bin"
        s_file = f"{safe_name}.weight.scale.f32.bin"

        weight_int8.tofile(os.path.join(OUT_DIR, w_file))
        scales.tofile(os.path.join(OUT_DIR, s_file))

        weight_bytes = int(weight_int8.nbytes)
        scale_bytes = int(scales.nbytes)
        total_weight_int8_bytes += weight_bytes
        total_scale_bytes += scale_bytes

        exported.append(
            ExportedLayer(
                index=len(exported),
                name=raw_name,
                op=node.op_type,
                kernel=get_attr(node, "kernel_shape", list(weight_f32.shape[2:])),
                stride=get_attr(node, "strides", [1, 1]),
                padding=get_attr(node, "pads", [0, 0, 0, 0]),
                dilation=get_attr(node, "dilations", [1, 1]),
                weight_shape=list(weight_f32.shape),
                input_shape=vi_map.get(node.input[0], []),
                output_shape=vi_map.get(out_name, []),
                per_channel_axis=axis,
                channels=int(scales.shape[0]),
                weight_int8_file=w_file,
                weight_scale_f32_file=s_file,
                weight_int8_bytes=weight_bytes,
                weight_scale_f32_bytes=scale_bytes,
                bn_folded=bn_folded,
                scale_min=float(scales.min()),
                scale_max=float(scales.max()),
                scale_mean=float(scales.mean()),
            )
        )

    elapsed_s = time.time() - start_s
    manifest = {
        "schema_version": 1,
        "generated_at_unix_s": start_s,
        "source_onnx": ONNX_PATH,
        "output_dir": OUT_DIR,
        "quantization": {
            "dtype": "int8",
            "scheme": "symmetric_absmax_per_output_channel",
            "range": [-128, 127],
            "qmax_for_scale": 127,
            "epsilon": EPS,
        },
        "bn_folded": True,
        "softmax_stripped_for_export": True,
        "num_layers": len(exported),
        "total_weight_int8_bytes": total_weight_int8_bytes,
        "total_scale_f32_bytes": total_scale_bytes,
        "layers": [layer.__dict__ for layer in exported],
    }

    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    receipt = {
        "status": "ok",
        "manifest": manifest_path,
        "elapsed_seconds": elapsed_s,
        "files_written": len(exported) * 2 + 2,
        "num_layers": len(exported),
    }
    with open(os.path.join(OUT_DIR, "export_receipt.json"), "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
        f.write("\n")

    return receipt


def main():
    print(f"Source ONNX: {ONNX_PATH}")
    print(f"Output dir : {OUT_DIR}")
    receipt = export()
    print(f"Status     : {receipt['status']}")
    print(f"Layers     : {receipt['num_layers']}")
    print(f"Manifest   : {receipt['manifest']}")
    print(f"Elapsed(s) : {receipt['elapsed_seconds']:.2f}")


if __name__ == "__main__":
    main()
