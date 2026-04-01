# Milestone 1 FPGA Notes (Draft)

## Kernel selected for first acceleration pass

- Target function: `BuildGraph(...)` in `frontend/SemGraph.cpp`
- Focused sub-kernel: O(N^2) edge construction and adjacency/edge matrix creation.

## Why this kernel first

- Data-parallel nested loops over node pairs.
- Clear arithmetic pattern (distance + threshold + histogram updates).
- Used in both `BuildGraph` and `ReBuildGraph`, so speedups impact multiple stages.

## Current blockers / open issues

- `MatrixDecomposing` uses Eigen eigensolver with dynamic internals; not directly HLS-friendly.
- Existing code uses dynamic containers (`std::vector`, `Eigen::MatrixXd`) that need fixed-size buffers for HLS top functions.
- Need realistic node-count distribution from KITTI/MulRAN runs to choose `MAX_NODES` safely.
- Need clear CPU/FPGA partition decision for Hungarian + outlier pruning path in loop closure.
- Need transfer-overhead measurement for host<->FPGA path before claiming net speedup.

## Near-term mitigation plan

- Keep eigendecomposition and Hungarian on CPU in Milestone 1.
- Offload only edge-construction kernel first, then re-profile end-to-end.
- Add per-stage CSV timing in `SemGraphSLAM::mainProcess` and benchmark harness.
- Validate numerical equivalence on sampled frames before reporting speedup.
