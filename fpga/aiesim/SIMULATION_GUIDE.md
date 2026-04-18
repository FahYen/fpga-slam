# AIE Kernel Simulation Guide

This guide provides step-by-step instructions for simulating AIE kernels for the RangeNet FPGA implementation.

## Overview

The AIE graph currently includes four kernel types:
- **3x3 convolution** (stride 1)
- **3x3 convolution** (stride 2)
- **1x1 convolution**
- **Element-wise addition** (residual connection)

Each kernel is tested on a single tile (8×32 spatial resolution, 32 input/output channels).

## Prerequisites

- Vitis 2024.2 with AIE tools installed
- Access to EC2 instance or local machine with Vitis
- Python 3 with required packages
- Golden data and weights already generated
- tmux or screen installed (for long-running simulations)

### Running Long Simulations Safely

Hardware simulations can take several minutes. If you accidentally close your SSH session, the simulation will be killed. Use one of these methods to prevent this:

**Option 1: Using tmux (Recommended)**
```bash
# Install tmux if not already installed
sudo apt install tmux

# Start a new tmux session
tmux new -s aiesim

# Run your simulation commands inside tmux
cd /home/ubuntu/workspace/slam/fpga/aiesim
# ... run simulation ...

# Detach from tmux session (Ctrl+B, then D)
# You can now safely close SSH session

# Reattach to tmux session later
tmux attach -t aiesim

# List all tmux sessions
tmux ls
```

**Option 2: Using screen**
```bash
# Install screen if not already installed
sudo apt install screen

# Start a new screen session
screen -S aiesim

# Run your simulation commands inside screen
cd /home/ubuntu/workspace/slam/fpga/aiesim
# ... run simulation ...

# Detach from screen (Ctrl+A, then D)
# You can now safely close SSH session

# Reattach to screen session later
screen -r aiesim

# List all screen sessions
screen -ls
```

**Option 3: Using nohup**
```bash
# Run simulation in background with nohup
cd /home/ubuntu/workspace/slam/fpga/aiesim/build_hw
nohup aiesimulator --pkg-dir=Work --profile > ../aiesim.log 2>&1 &

# View process status
jobs -l

# View log file in real-time
tail -f ../aiesim.log

# Kill the job if needed
kill %1
```

## Directory Structure

```
fpga/
├── conv_kernel.cpp          # AIE kernel implementations
├── conv_kernel.h            # Kernel function declarations
├── AIE_graph.cpp            # ADF graph definition
└── aiesim/
    ├── gen_plio_data.py     # PLIO data generation script
    ├── verify_x86sim.py     # Verification script
    ├── data/                # PLIO input/output files
    ├── build_x86/           # x86 simulation build directory
    └── build_hw/            # Hardware simulation build directory
```

## Simulation Workflow

### Step 1: Select Kernel Type

The `AIE_graph.cpp` currently has all kernels enabled. To test a specific kernel, comment out the others in the `RangeNetGraph` class:

**For 3x3 kernel only:**
- Keep `plio_3x3_*` and `g_conv3x3` uncommented
- Comment out all other PLIOs and kernels

**For 3x3s2 kernel only:**
- Keep `plio_3x3s2_*` and `g_conv3x3_s2` uncommented
- Comment out all other PLIOs and kernels

**For 1x1 kernel only:**
- Keep `plio_1x1_*` and `g_conv1x1` uncommented
- Comment out all other PLIOs and kernels

**For elem_add kernel only:**
- Keep `plio_add_*` and `g_elem_add` uncommented
- Comment out all other PLIOs and kernels

### Step 2: Generate PLIO Data

Generate PLIO input data for the selected kernel:

```bash
cd /home/ubuntu/workspace/slam/fpga/aiesim

# For 3x3 kernel (layer 1)
python3 gen_plio_data.py \
    --golden-dir ../../golden_int8/frame_0000 \
    --gemm-dir ../../weights/gemm \
    --layer-idx 1 \
    --tile-oh 0 \
    --tile-ow 0

# For 3x3s2 kernel (layer 1)
python3 gen_plio_data.py \
    --golden-dir ../../golden_int8/frame_0000 \
    --gemm-dir ../../weights/gemm \
    --layer-idx 1 \
    --tile-oh 0 \
    --tile-ow 0

# For 1x1 kernel (layer 2)
python3 gen_plio_data.py \
    --golden-dir ../../golden_int8/frame_0000 \
    --gemm-dir ../../weights/gemm \
    --layer-idx 2 \
    --tile-oh 0 \
    --tile-ow 0

# For elem_add kernel (layer 3)
python3 gen_plio_data.py \
    --golden-dir ../../golden_int8/frame_0000 \
    --gemm-dir ../../weights/gemm \
    --layer-idx 3 \
    --tile-oh 0 \
    --tile-ow 0
```

### Step 3: Run x86 Simulation (Functional Validation)

x86 simulation is fast (seconds) and validates kernel correctness.

```bash
cd /home/ubuntu/workspace/slam/fpga/aiesim

# Compile for x86 simulation
rm -rf build_x86
aiecompiler --target=x86sim \
    --include=.. \
    --workdir=build_x86/Work \
    --verbose \
    --part=xcve2302-sfva784-1LP-e-S \
    ../AIE_graph.cpp

# Run x86 simulation
cd build_x86
ln -s ../data data
x86simulator --pkg-dir=Work

# Verify output (if verification script supports x86 output format)
cd ..
python3 verify_x86sim.py \
    --sim-output build_x86/x86simulator_output/data/out_<kernel>.txt \
    --expected data/expected_out.txt \
    --tile-meta data/tile_meta.json
```

### Step 4: Run Hardware Simulation (Cycle Counts)

Hardware simulation provides cycle-accurate results but is slower.

```bash
cd /home/ubuntu/workspace/slam/fpga/aiesim

# Compile for hardware emulation
rm -rf build_hw
aiecompiler --target=hw \
    --include=.. \
    --workdir=build_hw/Work \
    --part=xcve2302-sfva784-1LP-e-S \
    --stacksize=16384 \
    ../AIE_graph.cpp

# Run hardware simulation (no timeout for faster completion)
cd build_hw
ln -s ../data data
aiesimulator --pkg-dir=Work --profile 2>&1 | tee ../aiesim.log
```

**Important:** Do NOT use `--simulation-cycle-timeout` unless you suspect the kernel will hang. The timeout causes the simulator to run idle cycles after kernel completion, making simulation 46x slower.

### Step 5: View Simulation Results

**Check simulation completion:**
```bash
# View simulation log
cat ../aiesim.log

# Look for:
# - "Simulation Finished, Sim result: 0" (success)
# - Total Simulation time (in ps or ns)
# - Wall clock time (actual runtime)
```

**Calculate cycle count:**
```
Cycles = Total Simulation Time (ns) × AIE Frequency (GHz)
Example: 1,650,620 ns × 1 GHz = 1,650,620 cycles
```

**View performance reports:**
```bash
cd /home/ubuntu/workspace/slam/fpga/aiesim/build_hw

# Throughput information
cat throughput_info.json

# Sample counts (DMA transfers)
cat pl_sample_counts

# Guidance report (performance warnings)
cat Work/reports/guidance.html

# Compiler estimates
cat Work/reports/AIE_graph.xpe
```

**Enable profiling for detailed stats:**
```bash
# Run with profiling enabled
aiesimulator --pkg-dir=Work --profile 2>&1 | tee ../aiesim_profile.log

# Profile specific cores
aiesimulator --pkg-dir=Work --profile="(col,row)" 2>&1 | tee ../aiesim_profile.log
```

## Expected Runtime

| Kernel Type | x86 Simulation | Hardware Simulation |
|-------------|----------------|---------------------|
| elem_add    | ~5 seconds     | ~10 seconds         |
| 1x1         | ~10 seconds    | ~20-30 seconds      |
| 3x3s2       | ~15 seconds    | ~1-2 minutes        |
| 3x3         | ~20 seconds    | ~2-3 minutes        |

Hardware simulation is ~470x slower than real-time but provides cycle-accurate results.

## Common Issues and Solutions

### Issue: "Input file path not found"
**Cause:** PLIO data files not generated for the selected kernel.
**Solution:** Run `gen_plio_data.py` for the correct layer index.

### Issue: Simulation hangs (no output)
**Cause:** Kernel bug or infinite loop.
**Solution:** 
1. Run x86 simulation first to validate kernel logic
2. If x86 also hangs, debug kernel code
3. Use `--hang-detect-time` to auto-terminate stalled simulations

### Issue: Very slow hardware simulation
**Cause:** Using `--simulation-cycle-timeout` flag.
**Solution:** Remove the timeout flag and let simulation complete naturally.

### Issue: "Cannot open DISPLAY" when using vitis_analyzer
**Cause:** Running on remote EC2 instance without GUI.
**Solution:** View reports as text files or download to local machine:
```bash
# View reports as text
cat Work/reports/guidance.html
cat throughput_info.json

# Download to local machine
scp ubuntu@<ec2-ip>:/path/to/report.html .
```

### Issue: Verification script fails on hardware output
**Cause:** Hardware simulator output format differs from x86.
**Solution:** Use x86 simulation for verification, hardware simulation only for cycle counts.

## Performance Optimization Tips

1. **Use x86 simulation for kernel validation** - It's fast and functionally correct
2. **Remove timeout flag** - Let simulation complete naturally for 46x speedup
3. **Profile specific cores** - Use `--profile="(col,row)"` instead of all cores
4. **Use compiler estimates** - Check `AIE_graph.xpe` for quick performance estimates
5. **Test single kernels** - Don't simulate all kernels together unless testing system integration

## Additional Resources

- Vitis AI Engine Documentation: UG1076
- AIE Kernel Programming Guide: UG1520
- Vitis Analyzer: `vitis_analyzer build_hw/Work/AIE_graph.aiecompile_summary`

## Summary

1. Modify `AIE_graph.cpp` to select kernel type
2. Generate PLIO data with `gen_plio_data.py`
3. Run x86 simulation for functional validation
4. Run hardware simulation for cycle counts (no timeout)
5. View results in log files and JSON reports
6. Calculate cycles: `Cycles = Simulation Time (ns) × Frequency (GHz)`
