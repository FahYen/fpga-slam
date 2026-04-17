# Scalar INT8 kernel unit tests

Compile-and-run unit tests that validate the integer math in
`fpga/conv_kernel.cpp` against bit-exact golden vectors produced by
`fpga/generate_golden_int8.py`. Uses a portable scalar C++ reimplementation
(`conv_scalar.hpp`) — no AIE tooling required.

## Prerequisites

1. INT8 weights   : `fpga/weights/int8/manifest.json` + `.bin` files
2. Requant params : `fpga/weights/requant/manifest.json` + `.bin` files
3. Golden vectors : `fpga/golden_int8/frame_0000/manifest.json` + `.bin` files

## Build

```bash
cd fpga/test
make
```

Produces `./test_layer`.

## Run all 68 layers of frame 0

```bash
# From the repo root (so the default relative paths work):
python3 fpga/test/run_tests.py
```

Expected output (when the scalar logic matches the golden generator):

```
[ 0] backbone.conv1                             Conv            PASS  (4194304 bytes match)
[ 1] backbone.enc1.conv                         Conv            PASS  (2097152 bytes match)
[ 2] backbone.enc1.residual_0.conv1             Conv            PASS  (1048576 bytes match)
...
[67] head.1                                     Conv            PASS  (2621440 bytes match)
=== Summary: 68/68 layers PASS ===
```

## Run one layer (fast iteration)

```bash
python3 fpga/test/run_tests.py --only 2 --verbose
```

## Stop at the first failure

```bash
python3 fpga/test/run_tests.py --stop-on-fail --verbose
```

On mismatch, the binary prints:
- Number of mismatching bytes and percentage
- First mismatch position (flat index and decoded oc/oh/ow)
- `got` vs `expected` values
- Mean and max absolute difference

## Runtime

First two layers (big H×W) take the longest — ~5-20 seconds each on a c5.4xlarge.
Deep layers (more channels, smaller H×W) also heavy. Expect **5-15 minutes
total** for all 68 layers, single-threaded scalar C++.

## What this tests

| Component | Covered |
|---|---|
| INT8 × INT8 → INT32 MAC | ✓ |
| Bias add in accumulator scale | ✓ |
| Per-channel requantize (mult + shift + rounding) | ✓ |
| LeakyReLU via `(x*13) >> 7` | ✓ |
| INT8 saturation | ✓ |
| Conv2D with stride, padding, 1x1 & 3x3 | ✓ |
| ConvTranspose2D | ✓ |

What this does **NOT** test:
- AIE-ML vector intrinsics (`aie::mmul`, `aie::vector`)
- DMA / data mover tile orchestration
- HWC layout conversion
- Multi-tile scheduling

Those come once the scalar layer is bit-exact.

## Debugging workflow

1. If a layer FAILS here → your scalar logic (`conv_scalar.hpp`) is wrong.
   Compare to `fpga/generate_golden_int8.py` which uses the same math in Python.
2. Once all 68 layers PASS here → port the math to AIE-ML intrinsics in
   `conv_kernel.cpp`. Use `x86simulator` or AIE simulator to validate vectors
   against the same golden .i8.bin files.

## Files

- `conv_scalar.hpp`   — portable scalar conv / conv_transpose (header-only)
- `test_layer.cpp`    — CLI binary: takes all params + file paths, runs one layer
- `run_tests.py`      — driver: reads manifests, invokes `test_layer` per layer
- `Makefile`          — `make` builds `test_layer`
