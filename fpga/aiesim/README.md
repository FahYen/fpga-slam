# AIE x86 Functional Simulation

Verify AIE kernel logic using fast x86 simulation before running the full AIE compiler.
This compiles the ADF graph as native x86 code and runs in seconds.

## Prerequisites

- Vitis 2023.2+ installed on the EC2 instance
- Golden INT8 vectors generated (`generate_golden_int8.py`)
- GEMM-B weights generated (`transpose_weights_gemm.py`)

## Step-by-Step

### 1. Source Vitis environment

```bash
source /tools/Xilinx/Vitis/2023.2/settings64.sh
# or wherever your Vitis install lives
```

### 2. Generate PLIO test data for one tile

Pick a layer where IC ≤ 32 so a single IC_BLOCK covers all input channels.
Layer 0 (IC=5) or layer 1 (IC=32) are good choices.

```bash
cd fpga/aiesim

python3 gen_plio_data.py \
    --golden-dir ../golden_int8/frame_0000 \
    --gemm-dir   ../weights/gemm \
    --layer-idx  1 \
    --output-dir data
```

This creates `data/` with PLIO text files and `expected_out.txt`.

### 3. Compile for x86 simulation

```bash
make x86compile
```

This runs `aiecompiler --target=x86sim` on `AIE_graph.cpp`.

### 4. Run the simulation

```bash
make x86sim
```

This runs `x86simulator` which reads `data/*.txt`, executes the kernels,
and writes output to `build_x86/x86simulator_output/data/out_3x3.txt`.

### 5. Verify against golden

```bash
make verify
```

Or manually:

```bash
python3 verify_x86sim.py \
    --sim-output build_x86/x86simulator_output/data/out_3x3.txt \
    --expected   data/expected_out.txt \
    --tile-meta  data/tile_meta.json
```

## Notes

- **Single IC block limitation**: For layers with IC > 32, a single kernel call
  only computes a partial accumulator. The output won't match the golden (which
  sums across all IC blocks). Use layers with IC ≤ 32 for single-block tests.

- **LeakyReLU**: The current graph always calls the `requant_row<true>` variant
  (LeakyReLU enabled). Pick a layer that uses LeakyReLU (most encoder layers do).

- **x86sim is functionally accurate**: It runs the exact same C++ code as the
  AIE hardware, just compiled for x86. A pass here means the kernel logic is correct.
