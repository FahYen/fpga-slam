# BuildGraph Benchmark Summary

This directory captures the first-pass `BuildGraph()` acceleration results for SG-SLAM.

## Benchmark configuration

- Binary: `/workspace/build/semgraph_slam/benchmarks/benchmark_buildgraph`
- Build: `Release`, `g++`, `-DENABLE_MILESTONE1_BENCHMARKS=ON`, `-DMINIGLOG=ON`
- Timed frames: `150`
- Warmup frames: `20`
- Seed: `570`
- Parameters: `--edge-th 40 --subinterval 40 --node-dim 8 --subgraph-edge-th 20`

## Before/after microbenchmark results

| Nodes/frame | Baseline avg ms | Optimized avg ms | Speedup | Optimized p95 ms | Optimized us/node | Optimized ns/pair |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.4858 | 0.3510 | 1.38x | 0.3776 | 5.4845 | 85.6959 |
| 128 | 2.7769 | 2.0866 | 1.33x | 2.1896 | 16.3017 | 127.3574 |
| 256 | 19.5040 | 12.5655 | 1.55x | 12.9526 | 49.0838 | 191.7336 |
| 512 | 147.0508 | 93.9907 | 1.56x | 97.8928 | 183.5756 | 358.5461 |

## Interpretation

- The optimization produces a measurable win across all graph sizes, with the gain increasing as the graph gets denser.
- The largest tested case (`512` nodes/frame) dropped from `147.05 ms` to `93.99 ms`, improving throughput from `6.80 FPS` to `10.64 FPS`.
- The speedup trend strengthens with node count, which is exactly what we want from the first accelerator-oriented kernel.
- `BuildGraph()` remains expensive at high node counts, so it is still the correct first hardware target even after the software optimization.

## Runtime timing hook commands

The microbenchmark above is reproducible in this workspace today. For dataset-backed timing inside the actual SLAM pipeline, use the existing runtime logger in `SemGraphSLAM.cpp`.

### KITTI

```bash
SGSLAM_BUILDGRAPH_TIMING=1 \
SGSLAM_BUILDGRAPH_TIMING_DATASET=kitti \
SGSLAM_BUILDGRAPH_TIMING_OUT="$PWD/buildgraph_kitti_timing.csv" \
roslaunch semgraph_slam semgraph_slam_kitti.launch
```

### MulRAN

```bash
SGSLAM_BUILDGRAPH_TIMING=1 \
SGSLAM_BUILDGRAPH_TIMING_DATASET=mulran \
SGSLAM_BUILDGRAPH_TIMING_OUT="$PWD/buildgraph_mulran_timing.csv" \
roslaunch semgraph_slam semgraph_slam_mulran.launch
```

## Dataset-backed timing status in this workspace

- KITTI point clouds are present under `data/kitti/sequences/00/velodyne/`.
- The semantic label files required by the default launch flow are not present in this workspace.
- MulRAN data is also not present here.

Because of that, the runtime hook could not be executed end-to-end in this container, but the hook is already wired and the exact launch commands above are ready once the labels/datasets are available.
