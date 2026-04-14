#!/usr/bin/env bash
# Compile a Vitis AI quantized xmodel for a target DPU.
#
# Usage (inside Vitis AI Docker):
#   bash fpga/vitis/compile_vitisai.sh /workspace/vitis_output/quantize_result
#
# Optional env vars:
#   DPU_ARCH   - path to arch.json (default: DPUCADF8H/U250 for AWS F1)
#   MODEL_NAME - output model name (default: rangenet_darknet53)

set -euo pipefail

QUANT_DIR="${1:?Usage: $0 <quantize_result_dir>}"
MODEL_NAME="${MODEL_NAME:-rangenet_darknet53}"

# ---------------------------------------------------------------------------
# Locate the quantized xmodel
# ---------------------------------------------------------------------------
# pytorch_nndct typically writes the xmodel with the class name
XMODEL=""
for candidate in \
    "$QUANT_DIR/SegmentatorForQuantization_int.xmodel" \
    "$QUANT_DIR/Segmentator_int.xmodel" \
    "$QUANT_DIR/quantize_result/SegmentatorForQuantization_int.xmodel" \
    "$QUANT_DIR"/*.xmodel; do
    if [[ -f "$candidate" ]]; then
        XMODEL="$candidate"
        break
    fi
done

if [[ -z "$XMODEL" ]]; then
    echo "ERROR: No .xmodel found in $QUANT_DIR"
    echo "  Expected: SegmentatorForQuantization_int.xmodel"
    echo "  Run quantize_vitisai.py --quant-mode test first."
    exit 1
fi

echo "Found xmodel: $XMODEL"

# ---------------------------------------------------------------------------
# Locate DPU architecture fingerprint
# ---------------------------------------------------------------------------
# Default: DPUCADF8H for Alveo U250 (AWS F1)
# Override with DPU_ARCH env var for other targets
if [[ -n "${DPU_ARCH:-}" ]]; then
    ARCH_JSON="$DPU_ARCH"
elif [[ -f /opt/vitis_ai/compiler/arch/DPUCADF8H/U250/arch.json ]]; then
    ARCH_JSON="/opt/vitis_ai/compiler/arch/DPUCADF8H/U250/arch.json"
elif [[ -f /opt/vitis_ai/compiler/arch/DPUCADF8H/U200/arch.json ]]; then
    ARCH_JSON="/opt/vitis_ai/compiler/arch/DPUCADF8H/U200/arch.json"
else
    echo "ERROR: Cannot find DPU arch.json"
    echo "  Looked in: /opt/vitis_ai/compiler/arch/DPUCADF8H/{U250,U200}/arch.json"
    echo "  Set DPU_ARCH env var to override."
    echo ""
    echo "  Available arches:"
    find /opt/vitis_ai/compiler/arch/ -name arch.json 2>/dev/null || echo "    (none found)"
    exit 1
fi

echo "DPU arch:    $ARCH_JSON"

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR="$QUANT_DIR/compiled"
mkdir -p "$OUT_DIR"

echo "Output dir:  $OUT_DIR"
echo "Model name:  $MODEL_NAME"
echo ""

# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------
echo "Compiling ..."
vai_c_xir \
    -x "$XMODEL" \
    -a "$ARCH_JSON" \
    -o "$OUT_DIR" \
    -n "$MODEL_NAME"

echo ""
echo "=== Compilation complete ==="
echo "  Compiled xmodel: $OUT_DIR/${MODEL_NAME}.xmodel"
echo ""
echo "Next step: copy to F1 and run inference:"
echo "  python fpga/vitis/infer_vitisai.py \\"
echo "    --xmodel $OUT_DIR/${MODEL_NAME}.xmodel \\"
echo "    --model /workspace/models/rangenet_darknet53 \\"
echo "    --scan-root /workspace/data/kitti/sequences \\"
echo "    --output-root /workspace/data/rangenet_vitisai/kitti \\"
echo "    --sequence 00"
