# EECS 570 Project Proposal: FPGA Accelerator for Semantic SLAM

## 1) Problem Definition and Motivation

Advanced semantic-graph SLAM systems such as **SG-SLAM** provide strong semantic object labeling, topological reasoning, and robust mapping in repetitive or dynamic environments. However, these systems are computationally intensive and often rely on high-end GPUs (e.g., NVIDIA RTX 3090-class hardware) for real-time performance.

There is still a major gap in hardware acceleration approaches that make this level of SLAM performance practical on edge platforms such as drones.  
This project proposes an **FPGA-based accelerator** to address that gap.

Our goal is to enable significantly lower latency and much better power efficiency than CPU/GPU baselines, targeting real-time, low-power autonomy for single-drone indoor tasks such as:

- 3D mapping
- anomaly detection
- real-time scene understanding and navigation

This project aligns directly with EECS 570 themes in parallel architecture, accelerator design, and heterogeneous systems for edge robotics.

## 2) Brief Survey of Related Work

### SG-SLAM (IROS 2025, Wang et al.)

**Leveraging Semantic Graphs for Efficient and Robust LiDAR SLAM** develops SG-SLAM, a semantic graph-enhanced framework integrating object-level semantics, topological reasoning, and covisibility graphs. It achieves state-of-the-art robustness and map consistency in dynamic/repetitive environments, but remains expensive for edge deployment.

### SuperNoVA (ASPLOS 2025, Shao et al.)

**SuperNoVA: Algorithm-Hardware Co-Design for Resource-Aware SLAM** proposes a co-design framework for accurate, real-time SLAM on constrained platforms (e.g., drones/SoCs), demonstrating improved energy efficiency via custom VIO and LBA accelerators.

### ORIANNA (ASPLOS 2024, Gan et al.)

**ORIANNA: An Accelerator Generation Framework for Optimization-based Robotic Applications** presents a method to generate FPGA accelerators for robotic optimization workloads, showing substantial performance gains while respecting hardware constraints.

### Archytas (MICRO 2021, Gan et al.)

**Archytas: A Framework for Synthesizing and Dynamically Optimizing Accelerators for Robotic Localization** introduces a synthesis framework for FPGA-based localization acceleration and demonstrates significant design-space exploration benefits and performance improvements.

## 3) Detailed Description of Experimental Setup

We propose a **hybrid hardware-software system** built on SG-SLAM:

1. **Software baseline strengthening** through parallelization in C++.
2. **FPGA acceleration** for remaining compute-intensive kernels.

### 3.1 Software-Level Parallelization (Baseline and Pre-Acceleration)

Before hardware offload, we will profile and optimize SG-SLAM on CPU by parallelizing selected kernels (e.g., via OpenMP and/or `std::thread`), including:

- voxel hashing paths
- semantic graph traversal operations
- selected g2o optimization stages

This establishes a stronger CPU baseline and identifies residual bottlenecks that are best suited for FPGA offloading.

### 3.2 FPGA Target Kernels

The accelerator phase will focus on compute-heavy kernels likely to benefit from custom dataflow and operator replication, including:

- semantic graph operations
- loop closure matching
- additional hotspots identified by profiling

We will use **Vivado HLS** to generate hardware from C++ kernels, then synthesize and evaluate on AWS EC2 F1 FPGA instances.

### 3.3 Infrastructure and Tools

- **Simulation:** gem5 and/or SniperSim for initial cycle-level multiprocessor evaluation.
- **Accelerator development:** Vivado HLS + Verilog generation/synthesis.
- **FPGA platform:** AWS EC2 F1.
- **Datasets:** MulRAN and KITTI (LiDAR + camera data for realistic scenarios).
- **Benchmark support:** PARSEC-style traces/workloads adapted for robotics-relevant behavior where applicable.

### 3.4 Evaluation Metrics

We will compare CPU-only, GPU-assisted, and CPU+FPGA configurations using:

- end-to-end latency / throughput
- kernel-level speedup
- power and energy efficiency
- SLAM quality metrics (trajectory error, loop closure quality, map consistency)

## 4) Project Milestones and Schedule

### Milestone 1 (Slides due **March 18, 2026**)

**Goals**

- Complete related work review.
- Establish SG-SLAM baseline performance on selected datasets.
- Identify software parallelization opportunities and prototype them.
- Run initial gem5/SniperSim experiments.

**Expected outcomes**

- Clear bottleneck characterization on CPU/GPU baselines.
- Quantified gains from software-level parallelization.
- Pivot decision:
  - If memory bottlenecks dominate, prioritize FPGA memory hierarchy design.
  - Otherwise, prioritize compute-kernel hardware design and integration.

### Milestone 2 (Poster due **April 20, 2026**)

**Goals**

- Synthesize and deploy initial accelerator on AWS FPGA.
- Integrate accelerator with modified SG-SLAM pipeline.
- Run dataset-driven mapping experiments.

**Expected outcomes**

- Significant kernel-level efficiency improvements.
- Preliminary real-time mapping demonstration.
- Refinements based on measured bottlenecks (e.g., precision tuning, dataflow optimization, semantic fusion adjustments).

### Final Report (Due **April 24, 2026**)

**Goals**

- Full end-to-end evaluation across latency, power, and SLAM quality.
- Compare against CPU/GPU baselines in dynamic/repetitive scenes.
- Deliver implementation artifacts and publishable analysis.

**Expected outcomes**

- Reproducible FPGA acceleration results on AWS F1.
- Resource/performance characterization (e.g., utilization, timing, speedup, energy).
- Discussion of transferability toward embedded UltraScale+ class platforms (e.g., Zynq UltraScale+), including constraints and scaling considerations.
- Stretch goal: conference-quality results suitable for submission (e.g., MICRO/HPCA).

## 5) Division of Labor

### Hardware Accelerator Design (Verilog/HLS)

Develop accelerator modules for selected SLAM kernels, emphasizing parallel dataflow and resource-constrained optimization.

- **Members:** Fa, JT, Sid, Nandan

### Software Modifications and Integration

Integrate hardware with SG-SLAM, profile and optimize C++ paths, and implement CPU-FPGA communication mechanisms.

- **Members:** Fa, JT, Sid, Nandan

### Experimental Setup and Simulation

Configure gem5/SniperSim and AWS FPGA workflows; adapt and run benchmark traces and experiments.

- **Members:** Fa, JT, Nandan

### Related Work and Analysis

Lead literature review and analysis methodology (including performance modeling such as roofline analysis where applicable).

- **Members:** Fa, Sid

### Milestones, Scheduling, and Presentation

Coordinate project timeline, testing checkpoints, demos, and preparation of slides/poster.

- **Members:** Fa, Nandan

## 6) Risks and Contingency Plans

- **Risk:** FPGA integration overhead limits end-to-end gains.  
**Mitigation:** prioritize kernels with high arithmetic intensity and low host-device synchronization cost.
- **Risk:** Memory bandwidth or irregular access patterns bottleneck acceleration.  
**Mitigation:** redesign on-chip buffering/tiling and evaluate alternate data layouts.
- **Risk:** Accuracy drop due to optimization or precision changes.  
**Mitigation:** keep accuracy-aware validation in the loop and gate optimizations on SLAM quality metrics.
- **Risk:** Toolflow/runtime instability (HLS, synthesis, AWS workflow).  
**Mitigation:** maintain a staged fallback path with stronger CPU parallel baseline and partial kernel offload.

