# AWS F1 FPGA Development Setup

## Target: Xilinx VU9P on AWS F1 Instance

### Key Resources
- VU9P: ~1.2M LUTs, 6840 DSPs, 75.9 MB (BRAM + URAM)
- 4x DDR4 channels (~64 GB/s aggregate)
- PCIe Gen3 x16 to host CPU
- Shell provided by AWS (handles PCIe, DMA, etc.)

---

## Step 1: Request F1 Access

F1 instances require an AWS service limit increase. Submit a request via:
**AWS Console → Service Quotas → EC2 → Running On-Demand F instances**

Request at least 8 vCPUs (f1.2xlarge = 1 FPGA, 8 vCPUs). Approval typically takes 1-2 business days.

---

## Step 2: Launch Development Instance

You have two options:

### Option A: Develop directly on F1 (recommended for testing)
```bash
# Use the FPGA Developer AMI (search "FPGA Developer AMI" in AWS Marketplace)
# Instance type: f1.2xlarge (1 FPGA)
# AMI includes Vitis 2021.2+, Vivado, aws-fpga HDK/SDK pre-installed
```

### Option B: Develop on cheaper instance, deploy to F1
```bash
# Use any large instance (c5.4xlarge recommended) with FPGA Developer AMI
# Vitis synthesis runs on CPU — no FPGA needed until bitstream testing
# Only switch to f1.2xlarge for hardware testing
```

**Cost note**: f1.2xlarge is ~$1.65/hr. Synthesis can take 4-8 hours. Option B saves money during development.

---

## Step 3: Environment Setup

```bash
# Clone AWS FPGA repo
git clone https://github.com/aws/aws-fpga.git
cd aws-fpga

# Source the Vitis development environment
source vitis_setup.sh
# This sets up: Vitis HLS, Vivado, XRT (Xilinx Runtime), platform files

# Verify
which v++ && echo "Vitis OK"
which vitis_hls && echo "Vitis HLS OK"
```

---

## Step 4: Hello World — Verify Toolchain

```bash
cd aws-fpga/Vitis/examples/xilinx_2021.2/hello_world

# Build for software emulation first (minutes, not hours)
make clean
make run TARGET=sw_emu PLATFORM=$AWS_PLATFORM

# If that works, build for hardware emulation (slower but more accurate)
make run TARGET=hw_emu PLATFORM=$AWS_PLATFORM

# Full hardware build (4-8 hours)
make all TARGET=hw PLATFORM=$AWS_PLATFORM
```

---

## Step 5: Understand the F1 Execution Model

```
┌─────────────────────────────────────────────────┐
│  Host (x86 CPU)                                 │
│  ┌───────────────┐                              │
│  │ Your C++ app  │ ← OpenCL/XRT API             │
│  │ - loads .xclbin                              │
│  │ - DMA input   │──────────┐                   │
│  │ - launch kernel          │                   │
│  │ - DMA output  │←─────────┤                   │
│  └───────────────┘          │ PCIe Gen3 x16     │
├─────────────────────────────┤                   │
│  FPGA (VU9P)                │                   │
│  ┌──────────────────────────┤                   │
│  │ AWS Shell (fixed)        │ ← PCIe, DMA, etc. │
│  ├──────────────────────────┤                   │
│  │ Your Custom Logic        │                   │
│  │ ┌──────────┐ ┌─────────┐│                   │
│  │ │ Conv2d   │ │ DDR4    ││                   │
│  │ │ Kernel   │ │ banks   ││                   │
│  │ │ (HLS)    │ │ 0,1,2,3 ││                   │
│  │ └──────────┘ └─────────┘│                   │
│  └──────────────────────────┘                   │
└─────────────────────────────────────────────────┘
```

### Data flow for inference:
1. Host loads range image [1,5,64,2048] into DDR bank 0 via DMA
2. Host loads weights into DDR banks 1-3 (or stream from host)
3. Host launches kernel
4. FPGA reads input from DDR, computes, writes labels to DDR
5. Host reads labels [1,64,2048] back via DMA

---

## Step 6: Your First HLS Kernel (Template)

Create `src/conv2d_kernel.cpp`:

```cpp
#include <hls_stream.h>
#include <ap_fixed.h>

// Fixed-point type (adjust after quantization results)
typedef ap_fixed<16,8> data_t;  // 16-bit, 8 integer bits
typedef ap_fixed<16,8> weight_t;

extern "C" {
void conv2d_kernel(
    const data_t *input,    // [Cin, H, W]
    const weight_t *weight, // [Cout, Cin, Kh, Kw]
    const data_t *bias,     // [Cout]
    data_t *output,         // [Cout, H_out, W_out]
    int Cin, int Cout,
    int H, int W,
    int Kh, int Kw,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    float relu_alpha        // 0.1 for LeakyReLU
) {
    #pragma HLS INTERFACE m_axi port=input  offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=weight offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=bias   offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem0
    #pragma HLS INTERFACE s_axilite port=return

    // TODO: Your implementation here
    // Key optimizations to explore:
    // - Loop tiling for on-chip buffer reuse
    // - #pragma HLS PIPELINE for inner loops
    // - #pragma HLS ARRAY_PARTITION for parallel MAC
    // - #pragma HLS DATAFLOW for layer pipelining
}
}
```

Build with:
```bash
v++ -c -t sw_emu --platform $AWS_PLATFORM -k conv2d_kernel src/conv2d_kernel.cpp -o conv2d_kernel.xo
v++ -l -t sw_emu --platform $AWS_PLATFORM conv2d_kernel.xo -o conv2d_kernel.xclbin
```

---

## Step 7: Create Amazon FPGA Image (AFI)

After hardware build succeeds:
```bash
# Create AFI from .xclbin
$VITIS_DIR/tools/create_vitis_afi.sh \
    -xclbin=conv2d_kernel.xclbin \
    -o=conv2d_kernel \
    -s3_bucket=<your-bucket> \
    -s3_dcp_key=<dcp-folder> \
    -s3_logs_key=<logs-folder>

# This returns an AFI ID (agfi-XXXX). Wait for it to become "available":
aws ec2 describe-fpga-images --fpga-image-ids <afi-id>
```

---

## Key Files from Phase 0 You'll Need

All in `fpga/weights/`:
- `manifest.json` — complete layer spec (shapes, strides, padding per layer)
- `<layer>.weight.bin` — BN-folded float32 weights (little-endian)
- `<layer>.bias.bin` — fused biases

Golden vectors in `fpga/golden_vectors/frame_XXXX/`:
- `layer_XX_conv_input.bin` — input activation for each layer
- `layer_XX_post_activation.bin` — expected output after Conv+Bias+LeakyReLU

### Reading weights in C++ testbench:
```cpp
#include <fstream>
#include <vector>

std::vector<float> load_bin(const std::string& path, size_t num_elements) {
    std::vector<float> data(num_elements);
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(float));
    return data;
}
```

---

## Recommended Development Order

1. `sw_emu` — fast iteration, verify correctness against golden vectors
2. `hw_emu` — cycle-accurate, verify timing/throughput
3. `hw` — full synthesis, deploy on F1
