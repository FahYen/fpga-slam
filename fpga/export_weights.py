#!/usr/bin/env python3
"""Phase 0: BN-fold, softmax-strip, weight export for FPGA HLS.

Reads  data/pretrained_darknet53_weights/model.onnx
Writes fpga/model_fused.onnx          (clean ONNX for golden-vector generation)
       fpga/weights/<layer>.weight.bin (little-endian float32)
       fpga/weights/<layer>.bias.bin
       fpga/weights/manifest.json
"""

import os, json, sys
import numpy as np
import onnx
from onnx import numpy_helper, helper, TensorProto, shape_inference

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
ONNX_PATH = os.path.join(ROOT_DIR, "data", "pretrained_darknet53_weights", "model.onnx")
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights")
FUSED_ONNX_PATH = os.path.join(SCRIPT_DIR, "model_fused.onnx")

os.makedirs(WEIGHTS_DIR, exist_ok=True)


def load_initializers(model):
    return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}


def get_attr(node, name, default=None):
    for attr in node.attribute:
        if attr.name == name:
            if attr.type == onnx.AttributeProto.FLOAT:
                return attr.f
            if attr.type == onnx.AttributeProto.INT:
                return attr.i
            if attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
    return default


def fold_bn(conv_node, bn_node, inits):
    """Fold BatchNorm into preceding Conv/ConvTranspose. Returns (W_fused, b_fused)."""
    W = inits[conv_node.input[1]].copy()

    has_bias = (len(conv_node.input) >= 3
                and conv_node.input[2] != ""
                and conv_node.input[2] in inits)

    out_channels = W.shape[0] if conv_node.op_type == "Conv" else W.shape[1]
    b_conv = inits[conv_node.input[2]].copy() if has_bias else np.zeros(out_channels, dtype=np.float32)

    gamma = inits[bn_node.input[1]]
    beta  = inits[bn_node.input[2]]
    mean  = inits[bn_node.input[3]]
    var   = inits[bn_node.input[4]]
    eps   = get_attr(bn_node, "epsilon", 1e-5)

    factor = gamma / np.sqrt(var + eps)

    if conv_node.op_type == "Conv":
        W_fused = W * factor.reshape(-1, 1, 1, 1)
    else:
        W_fused = W * factor.reshape(1, -1, 1, 1)

    b_fused = factor * (b_conv - mean) + beta
    return W_fused.astype(np.float32), b_fused.astype(np.float32)


def build_maps(model):
    output_to_node = {}
    for node in model.graph.node:
        for o in node.output:
            output_to_node[o] = node

    input_to_consumers = {}
    for node in model.graph.node:
        for i in node.input:
            input_to_consumers.setdefault(i, []).append(node)

    return output_to_node, input_to_consumers


def identify_bn_pairs(model):
    """Return list of (conv_node, bn_node) pairs."""
    output_to_node, _ = build_maps(model)
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


def identify_softmax_tail(model):
    """Find Exp->ReduceSum->Div softmax pattern. Return (nodes_to_remove, logit_tensor_name)."""
    output_to_node, _ = build_maps(model)
    for node in reversed(list(model.graph.node)):
        if node.op_type == "Div":
            num_in, den_in = node.input[0], node.input[1]
            if num_in in output_to_node and output_to_node[num_in].op_type == "Exp":
                exp_node = output_to_node[num_in]
                if den_in in output_to_node and output_to_node[den_in].op_type == "ReduceSum":
                    rsum_node = output_to_node[den_in]
                    return [exp_node, rsum_node, node], exp_node.input[0]
    return [], None


def build_fused_onnx(model, inits):
    """Create new ONNX with BN folded into Conv and softmax stripped."""
    pairs = identify_bn_pairs(model)
    softmax_nodes, logit_name = identify_softmax_tail(model)

    conv_to_bn = {id(conv): bn for conv, bn in pairs}
    bn_set = {id(bn) for _, bn in pairs}
    softmax_set = {id(n) for n in softmax_nodes}

    pair_fused = {}
    for conv, bn in pairs:
        pair_fused[id(conv)] = fold_bn(conv, bn, inits)

    new_initializers = []
    init_names_added = set()
    new_nodes = []

    for node in model.graph.node:
        nid = id(node)
        if nid in bn_set or nid in softmax_set:
            continue

        if nid in pair_fused:
            W_fused, b_fused = pair_fused[nid]
            bn = conv_to_bn[nid]

            w_name = node.input[1] + "_fused"
            b_name = w_name.replace(".weight_fused", ".bias_fused")
            if ".weight" not in w_name:
                b_name = w_name + "_bias_fused"

            if w_name not in init_names_added:
                new_initializers.append(numpy_helper.from_array(W_fused, name=w_name))
                new_initializers.append(numpy_helper.from_array(b_fused, name=b_name))
                init_names_added.add(w_name)
                init_names_added.add(b_name)

            new_node = helper.make_node(
                node.op_type,
                inputs=[node.input[0], w_name, b_name],
                outputs=[bn.output[0]],
                name=(node.name or node.input[1]) + "_fused",
            )
            for attr in node.attribute:
                new_node.attribute.append(attr)
            new_nodes.append(new_node)
        else:
            for inp in node.input:
                if inp in inits and inp not in init_names_added:
                    new_initializers.append(
                        numpy_helper.from_array(inits[inp], name=inp))
                    init_names_added.add(inp)
            new_nodes.append(node)

    if logit_name:
        outputs = [helper.make_tensor_value_info(logit_name, TensorProto.FLOAT, None)]
    else:
        outputs = list(model.graph.output)

    graph = helper.make_graph(
        new_nodes,
        model.graph.name + "_fused",
        list(model.graph.input)[:1],
        outputs,
        initializer=new_initializers,
    )
    fused_model = helper.make_model(graph, opset_imports=model.opset_import)
    fused_model.ir_version = model.ir_version
    return fused_model


def extract_layers_and_export(model, inits):
    """Walk graph, fuse weights, export .bin files, return layer metadata list."""
    pairs = identify_bn_pairs(model)
    softmax_nodes, _ = identify_softmax_tail(model)
    softmax_set = {id(n) for n in softmax_nodes}
    bn_set = {id(bn) for _, bn in pairs}
    conv_to_bn = {id(conv): bn for conv, bn in pairs}

    model_shaped = shape_inference.infer_shapes(model)
    vi_map = {}
    for collection in [model_shaped.graph.value_info,
                       model_shaped.graph.input,
                       model_shaped.graph.output]:
        for vi in collection:
            dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            vi_map[vi.name] = dims

    layers = []
    for node in model.graph.node:
        nid = id(node)
        if nid in bn_set or nid in softmax_set:
            continue
        if node.op_type not in ("Conv", "ConvTranspose"):
            continue

        if nid in conv_to_bn:
            bn = conv_to_bn[nid]
            W_fused, b_fused = fold_bn(node, bn, inits)
            out_name = bn.output[0]
        else:
            W_fused = inits[node.input[1]].copy()
            if len(node.input) > 2 and node.input[2] in inits:
                b_fused = inits[node.input[2]].copy()
            else:
                b_fused = None
            out_name = node.output[0]

        raw_name = node.input[1].replace(".weight", "")
        safe_name = raw_name.replace(".", "_")

        w_file = f"{safe_name}.weight.bin"
        W_fused.tofile(os.path.join(WEIGHTS_DIR, w_file))

        b_file = None
        if b_fused is not None:
            b_file = f"{safe_name}.bias.bin"
            b_fused.tofile(os.path.join(WEIGHTS_DIR, b_file))

        next_nodes = [n for n in model.graph.node if out_name in n.input]
        relu_alpha = None
        for nn in next_nodes:
            if nn.op_type == "LeakyRelu":
                relu_alpha = get_attr(nn, "alpha", 0.01)
                break

        info = {
            "index": len(layers),
            "name": raw_name,
            "op": node.op_type,
            "kernel": get_attr(node, "kernel_shape", list(W_fused.shape[2:])),
            "stride": get_attr(node, "strides", [1, 1]),
            "padding": get_attr(node, "pads", [0, 0, 0, 0]),
            "dilation": get_attr(node, "dilations", [1, 1]),
            "weight_shape": list(W_fused.shape),
            "weight_file": w_file,
            "bias_shape": list(b_fused.shape) if b_fused is not None else None,
            "bias_file": b_file,
            "bn_folded": nid in conv_to_bn,
            "activation": "leaky_relu" if relu_alpha is not None else "none",
            "activation_alpha": float(relu_alpha) if relu_alpha is not None else None,
            "input_shape": vi_map.get(node.input[0], []),
            "output_shape": vi_map.get(out_name, []),
            "weight_bytes": W_fused.nbytes,
            "bias_bytes": b_fused.nbytes if b_fused is not None else 0,
        }
        layers.append(info)

    return layers


def verify_fused_model(orig_path, fused_path):
    """Run both models on random input and compare argmax outputs."""
    import onnxruntime as ort

    sess_orig  = ort.InferenceSession(orig_path)
    sess_fused = ort.InferenceSession(fused_path)

    rng = np.random.RandomState(42)
    x = rng.randn(1, 5, 64, 2048).astype(np.float32)

    out_orig  = sess_orig.run(None,  {sess_orig.get_inputs()[0].name:  x})[0]
    out_fused = sess_fused.run(None, {sess_fused.get_inputs()[0].name: x})[0]

    argmax_orig  = np.argmax(out_orig,  axis=1)
    argmax_fused = np.argmax(out_fused, axis=1)
    match_pct = 100.0 * np.mean(argmax_orig == argmax_fused)

    # Compare logits directly: undo softmax from original
    logits_from_orig = np.log(np.clip(out_orig, 1e-30, None))
    # Shift doesn't matter for comparison, but let's look at relative scale
    # Just compare at one sample point
    max_logit_diff = np.max(np.abs(out_fused[:, :, 0, 0] -
                                    (logits_from_orig[:, :, 0, 0] -
                                     logits_from_orig[:, :, 0, 0].max())))

    return match_pct, max_logit_diff


def main():
    print(f"Loading {ONNX_PATH}")
    model = onnx.load(ONNX_PATH)
    inits = load_initializers(model)
    print(f"  {len(model.graph.node)} nodes, {len(inits)} initializers")

    print("\nExtracting layers + exporting fused weights ...")
    layers = extract_layers_and_export(model, inits)
    print(f"  {len(layers)} conv/deconv layers exported")

    total_w = sum(l["weight_bytes"] for l in layers)
    total_b = sum(l["bias_bytes"] for l in layers)
    print(f"  Total weight: {total_w / 1e6:.1f} MB  |  Total bias: {total_b / 1e3:.1f} KB")

    manifest = {
        "source_onnx": os.path.basename(ONNX_PATH),
        "input_shape": [1, 5, 64, 2048],
        "num_classes": 20,
        "bn_folded": True,
        "softmax_stripped": True,
        "dtype": "float32",
        "byte_order": "little-endian",
        "total_weight_bytes": total_w,
        "total_bias_bytes": total_b,
        "layers": layers,
    }
    manifest_path = os.path.join(WEIGHTS_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: {manifest_path}")

    print("\nBuilding fused ONNX (BN folded, softmax stripped) ...")
    fused = build_fused_onnx(model, inits)
    onnx.save(fused, FUSED_ONNX_PATH)
    print(f"  Saved: {FUSED_ONNX_PATH}")

    print("\nVerifying fused model vs original ...")
    try:
        match_pct, logit_diff = verify_fused_model(ONNX_PATH, FUSED_ONNX_PATH)
        print(f"  Argmax agreement: {match_pct:.2f}%")
        if match_pct < 99.9:
            print("  WARNING: >0.1% mismatch — inspect fusion")
        else:
            print("  PASS")
    except Exception as e:
        print(f"  Verification error: {e}")

    print("\nLayer summary:")
    for l in layers:
        tag = ""
        if not l["bn_folded"]:
            tag = " [NO-BN]"
        if l["op"] == "ConvTranspose":
            tag += " [DECONV]"
        print(f"  [{l['index']:2d}] {l['op']:<14s} {l['name']:<50s} "
              f"W={str(l['weight_shape']):<22s} "
              f"in={str(l['input_shape']):<22s} out={str(l['output_shape'])}{tag}")


if __name__ == "__main__":
    main()
