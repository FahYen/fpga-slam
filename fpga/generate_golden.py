#!/usr/bin/env python3
"""Phase 3: Generate layer-level golden vectors from BN-folded ONNX for HLS debugging.

Uses onnxruntime (no PyTorch needed). Captures intermediate tensors in
small batches, writing each to disk immediately to stay within RAM limits.

Writes to fpga/golden_vectors/frame_XXXX/
"""

import os, json, sys, gc
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FUSED_ONNX = os.path.join(SCRIPT_DIR, "model_fused.onnx")
MANIFEST = os.path.join(SCRIPT_DIR, "weights", "manifest.json")
GOLDEN_DIR = os.path.join(SCRIPT_DIR, "golden_vectors")

NUM_FRAMES = 3
SEEDS = [0, 1, 42]
BATCH_SIZE = 6


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


def find_layer_boundaries(model, manifest_layers):
    node_list = list(model.graph.node)
    input_to_consumers = {}
    for node in node_list:
        for inp in node.input:
            input_to_consumers.setdefault(inp, []).append(node)

    conv_nodes = [n for n in node_list if n.op_type in ("Conv", "ConvTranspose")]
    boundaries = []
    for conv, layer in zip(conv_nodes, manifest_layers):
        entry = {
            "layer_index": layer["index"],
            "layer_name": layer["name"],
            "op": layer["op"],
            "conv_input": conv.input[0],
            "conv_output": conv.output[0],
            "post_activation": None,
            "post_residual": None,
        }
        for c in input_to_consumers.get(conv.output[0], []):
            if c.op_type == "LeakyRelu":
                entry["post_activation"] = c.output[0]
                for rc in input_to_consumers.get(c.output[0], []):
                    if rc.op_type == "Add":
                        entry["post_residual"] = rc.output[0]
                break
        boundaries.append(entry)
    return boundaries


def build_tensor_file_map(boundaries):
    """Map tensor_name -> list of (file_path_template, key, layer_index)."""
    tmap = {}
    for b in boundaries:
        prefix = f"layer_{b['layer_index']:02d}"
        for key in ["conv_input", "conv_output", "post_activation", "post_residual"]:
            tname = b[key]
            if tname:
                tmap.setdefault(tname, []).append((prefix, key, b["layer_index"]))
    return tmap


def run_batch(model, x, tensor_names, tmp_path):
    """Run inference with a small set of extra outputs, return dict of name->array."""
    m = make_model_with_outputs(model, tensor_names)
    onnx.save(m, tmp_path)
    del m

    sess = ort.InferenceSession(tmp_path, providers=["CPUExecutionProvider"])
    avail = {o.name for o in sess.get_outputs()}
    request = [n for n in tensor_names if n in avail]
    input_name = sess.get_inputs()[0].name
    out = sess.run(request, {input_name: x})
    result = {name: arr for name, arr in zip(request, out)}
    del sess, out
    gc.collect()
    os.remove(tmp_path)
    return result


def process_frame(model, x, boundaries, tensor_file_map, frame_dir, tmp_dir):
    """Capture all tensors for one frame, writing to disk in batches."""
    all_tnames = sorted(tensor_file_map.keys())
    final_names = [o.name for o in model.graph.output]

    all_tnames_with_final = list(dict.fromkeys(all_tnames + final_names))

    existing_outputs = {o.name for o in model.graph.output}
    extras = [n for n in all_tnames_with_final if n not in existing_outputs]

    batches = [extras[i:i + BATCH_SIZE] for i in range(0, len(extras), BATCH_SIZE)]
    n_batches = len(batches) + 1  # +1 for final outputs batch

    written_files = {}  # (layer_index, key) -> {file, shape, dtype, tensor_name}

    for bi, batch in enumerate(batches):
        tmp_path = os.path.join(tmp_dir, f"_b{bi}.onnx")
        results = run_batch(model, x, batch, tmp_path)

        for tname, arr in results.items():
            if tname in tensor_file_map:
                for prefix, key, lidx in tensor_file_map[tname]:
                    fname = f"{prefix}_{key}.bin"
                    arr.tofile(os.path.join(frame_dir, fname))
                    written_files[(lidx, key)] = {
                        "file": fname,
                        "shape": list(arr.shape),
                        "dtype": "float32",
                        "tensor_name": tname,
                    }

        del results
        gc.collect()
        sys.stdout.write(f"\r    batch {bi + 1}/{n_batches}")
        sys.stdout.flush()

    # Final outputs (graph's own outputs — no extra needed)
    sess = ort.InferenceSession(FUSED_ONNX, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    final_avail = [o.name for o in sess.get_outputs()]
    out = sess.run(final_avail, {input_name: x})
    final_results = {n: a for n, a in zip(final_avail, out)}

    for tname, arr in final_results.items():
        if tname in tensor_file_map:
            for prefix, key, lidx in tensor_file_map[tname]:
                if (lidx, key) not in written_files:
                    fname = f"{prefix}_{key}.bin"
                    arr.tofile(os.path.join(frame_dir, fname))
                    written_files[(lidx, key)] = {
                        "file": fname,
                        "shape": list(arr.shape),
                        "dtype": "float32",
                        "tensor_name": tname,
                    }

    logits = final_results.get(final_avail[0])
    if logits is not None:
        labels = np.argmax(logits, axis=1).astype(np.int32)
        logits.tofile(os.path.join(frame_dir, "logits.bin"))
        labels.tofile(os.path.join(frame_dir, "labels.bin"))

    del sess, out, final_results
    gc.collect()
    sys.stdout.write(f"\r    batch {n_batches}/{n_batches}\n")

    return written_files, logits is not None


def main():
    print(f"Loading fused model: {FUSED_ONNX}")
    model = onnx.load(FUSED_ONNX)

    with open(MANIFEST) as f:
        manifest = json.load(f)
    layers = manifest["layers"]
    print(f"  {len(layers)} layers in manifest")

    boundaries = find_layer_boundaries(model, layers)
    tensor_file_map = build_tensor_file_map(boundaries)
    print(f"  {len(tensor_file_map)} unique tensors, batch size {BATCH_SIZE}")

    tmp_dir = os.path.join(SCRIPT_DIR, "_tmp_golden")
    os.makedirs(tmp_dir, exist_ok=True)

    for frame_idx, seed in enumerate(SEEDS):
        frame_dir = os.path.join(GOLDEN_DIR, f"frame_{frame_idx:04d}")
        os.makedirs(frame_dir, exist_ok=True)

        rng = np.random.RandomState(seed)
        x = rng.randn(1, 5, 64, 2048).astype(np.float32)
        x.tofile(os.path.join(frame_dir, "input.bin"))

        print(f"\n  Frame {frame_idx} (seed={seed}):")
        written, has_logits = process_frame(
            model, x, boundaries, tensor_file_map, frame_dir, tmp_dir)

        frame_manifest = {
            "seed": seed,
            "input_shape": [1, 5, 64, 2048],
            "input_file": "input.bin",
            "layers": [],
        }
        for b in boundaries:
            entry = {
                "layer_index": b["layer_index"],
                "layer_name": b["layer_name"],
                "op": b["op"],
                "files": {},
            }
            for key in ["conv_input", "conv_output", "post_activation", "post_residual"]:
                wk = (b["layer_index"], key)
                if wk in written:
                    entry["files"][key] = written[wk]
            frame_manifest["layers"].append(entry)

        if has_logits:
            frame_manifest["logits_file"] = "logits.bin"
            frame_manifest["logits_shape"] = [1, 20, 64, 2048]
            frame_manifest["labels_file"] = "labels.bin"
            frame_manifest["labels_shape"] = [1, 64, 2048]

        with open(os.path.join(frame_dir, "manifest.json"), "w") as f:
            json.dump(frame_manifest, f, indent=2)

        total_bytes = sum(os.path.getsize(os.path.join(frame_dir, fn))
                          for fn in os.listdir(frame_dir) if fn.endswith(".bin"))
        print(f"    {sum(1 for v in written.values())} tensor files, {total_bytes / 1e6:.1f} MB")

        del written
        gc.collect()

    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    print("\n--- Summary ---")
    for frame_idx in range(NUM_FRAMES):
        frame_dir = os.path.join(GOLDEN_DIR, f"frame_{frame_idx:04d}")
        n_files = len([f for f in os.listdir(frame_dir) if f.endswith(".bin")])
        total = sum(os.path.getsize(os.path.join(frame_dir, f))
                    for f in os.listdir(frame_dir) if f.endswith(".bin"))
        print(f"  frame_{frame_idx:04d}: {n_files} .bin files, {total / 1e6:.1f} MB")

    print(f"\nGolden vectors: {GOLDEN_DIR}")


if __name__ == "__main__":
    main()
