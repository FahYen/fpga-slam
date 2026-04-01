// BuildGraph micro-benchmark for SG-SLAM.
//
// What this program measures:
// - Runtime of graph_slam::BuildGraph() only.
// - Synthetic per-frame inputs are generated once, then reused during timing.
// - Reported metrics focus on latency (avg/p50/p95) and approximate throughput.
//
// What this program does NOT measure:
// - ROS message passing, roslaunch overhead, bag/dataset I/O, or visualization.
// - Full SLAM pipeline latency.
//
// Why this benchmark exists:
// - Provide a stable baseline before algorithm changes or hardware acceleration work.
// - Enable apples-to-apples comparison by controlling input size and random seed.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "semgraph_slam/frontend/SemGraph.hpp"

namespace {

using graph_slam::Bbox;
using graph_slam::BuildGraph;

// Runtime-configurable benchmark parameters.
// Defaults are intentionally conservative and produce a short but meaningful run.
struct BenchmarkConfig {
    // Number of semantic boxes in each synthetic frame.
    // Higher values increase graph density and usually increase runtime.
    std::size_t nodes = 256;

    // Number of measured frames used for metrics.
    // Larger values reduce noise and improve statistical stability.
    std::size_t frames = 200;

    // Number of initial frames run but excluded from timing.
    // This reduces one-time effects (cold cache, first-call setup).
    std::size_t warmup = 20;

    // Seed used to make synthetic data generation deterministic.
    // Keep fixed when comparing code versions.
    int seed = 570;

    // Parameters forwarded directly to BuildGraph().
    double edge_distance_threshold = 40.0;
    int subinterval = 40;
    int graph_node_dimension = 8;
    double subgraph_edge_threshold = 20.0;
};

// Parse helper for unsigned integer-like CLI values.
std::size_t ParseSize(const std::string &value) {
    return static_cast<std::size_t>(std::stoull(value));
}

// Parse helper for signed integer CLI values.
int ParseInt(const std::string &value) {
    return std::stoi(value);
}

// Parse helper for floating-point CLI values.
double ParseDouble(const std::string &value) {
    return std::stod(value);
}

// Minimal argument parser for this benchmark.
// Unknown arguments are ignored; known flags update BenchmarkConfig.
BenchmarkConfig ParseArgs(int argc, char **argv) {
    BenchmarkConfig config;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--nodes" && i + 1 < argc) {
            config.nodes = ParseSize(argv[++i]);
        } else if (arg == "--frames" && i + 1 < argc) {
            config.frames = ParseSize(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            config.warmup = ParseSize(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            config.seed = ParseInt(argv[++i]);
        } else if (arg == "--edge-th" && i + 1 < argc) {
            config.edge_distance_threshold = ParseDouble(argv[++i]);
        } else if (arg == "--subinterval" && i + 1 < argc) {
            config.subinterval = ParseInt(argv[++i]);
        } else if (arg == "--node-dim" && i + 1 < argc) {
            config.graph_node_dimension = ParseInt(argv[++i]);
        } else if (arg == "--subgraph-edge-th" && i + 1 < argc) {
            config.subgraph_edge_threshold = ParseDouble(argv[++i]);
        } else if (arg == "--help") {
            // Keep help compact and practical for command-line users.
            std::cout << "Usage: benchmark_buildgraph [options]\n"
                      << "  --nodes <N>              Number of clustered boxes per frame\n"
                      << "  --frames <N>             Number of timed frames\n"
                      << "  --warmup <N>             Warmup frame count\n"
                      << "  --seed <N>               RNG seed\n"
                      << "  --edge-th <float>        edge_dis_th for BuildGraph\n"
                      << "  --subinterval <N>        subinterval for BuildGraph\n"
                      << "  --node-dim <N>           graph_node_dimension for BuildGraph\n"
                      << "  --subgraph-edge-th <f>   subgraph_edge_th for BuildGraph\n";
            std::exit(0);
        }
    }
    return config;
}

// Generate synthetic semantic boxes for one frame.
// Distribution choices are simple and broad enough to vary graph connectivity.
std::vector<Bbox> GenerateClusterBoxes(std::size_t count, std::mt19937 &rng) {
    std::vector<Bbox> boxes;
    boxes.reserve(count);

    // Spatial distributions (meters) for center sampling.
    std::uniform_real_distribution<double> xy_dist(-40.0, 40.0);
    std::uniform_real_distribution<double> z_dist(-2.0, 3.0);

    // Bounding-box size distribution (meters).
    std::uniform_real_distribution<double> dim_dist(0.2, 4.5);

    // Semantic label selection and point-count population.
    std::uniform_int_distribution<int> label_dist(0, 2);
    std::uniform_int_distribution<int> points_dist(20, 300);

    for (std::size_t i = 0; i < count; ++i) {
        Bbox box;
        box.center = Eigen::Vector3d(xy_dist(rng), xy_dist(rng), z_dist(rng));
        box.dimension = Eigen::Vector3d(dim_dist(rng), dim_dist(rng), dim_dist(rng));
        box.theta = 0.0;
        box.score = 1.0;
        box.points_num = points_dist(rng);

        // Map synthetic class ID (0/1/2) to SG-SLAM labels (1/2/3).
        const int label_id = label_dist(rng);
        if (label_id == 0) {
            box.label = 1;
        } else if (label_id == 1) {
            box.label = 2;
        } else {
            box.label = 3;
        }
        boxes.emplace_back(box);
    }

    return boxes;
}

// Compute a percentile from a copy of latency samples in milliseconds.
// Example: percentile=0.95 gives p95.
// Uses floor-based index on sorted samples.
double PercentileMs(std::vector<double> samples_ms, double percentile) {
    if (samples_ms.empty()) {
        return 0.0;
    }
    std::sort(samples_ms.begin(), samples_ms.end());
    const std::size_t idx = static_cast<std::size_t>(
        std::floor(percentile * static_cast<double>(samples_ms.size() - 1U)));
    return samples_ms[idx];
}

}  // namespace

int main(int argc, char **argv) {
    // 1) Read benchmark parameters.
    const BenchmarkConfig cfg = ParseArgs(argc, argv);

    // Total frames = warmup (discarded) + timed (reported).
    const std::size_t total_frames = cfg.frames + cfg.warmup;

    // 2) Pre-generate all frame inputs so data generation is not included in timing.
    std::mt19937 rng(static_cast<std::mt19937::result_type>(cfg.seed));
    std::vector<std::vector<Bbox>> frames;
    frames.reserve(total_frames);
    for (std::size_t i = 0; i < total_frames; ++i) {
        frames.emplace_back(GenerateClusterBoxes(cfg.nodes, rng));
    }

    // Stores one elapsed time per timed frame.
    std::vector<double> elapsed_ms;
    elapsed_ms.reserve(cfg.frames);

    // Keep a small sanity snapshot from the last graph output.
    std::size_t output_edges = 0;
    std::size_t output_nodes = 0;

    // 3) Execute benchmark loop.
    for (std::size_t i = 0; i < total_frames; ++i) {
        // Time only BuildGraph invocation.
        const auto t0 = std::chrono::steady_clock::now();
        const graph_slam::Graph graph = BuildGraph(frames[i],
                                                   cfg.edge_distance_threshold,
                                                   cfg.subinterval,
                                                   cfg.graph_node_dimension,
                                                   cfg.subgraph_edge_threshold);
        const auto t1 = std::chrono::steady_clock::now();

        // Ignore warmup iterations in statistics.
        if (i >= cfg.warmup) {
            const double elapsed = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
            elapsed_ms.emplace_back(elapsed);
        }

        output_edges = graph.edges.size();
        output_nodes = graph.node_labels.size();
    }

    // 4) Aggregate timing metrics.
    // avg_ms: arithmetic mean latency across timed frames.
    const double total_ms = std::accumulate(elapsed_ms.begin(), elapsed_ms.end(), 0.0);
    const double avg_ms = elapsed_ms.empty() ? 0.0 : total_ms / static_cast<double>(elapsed_ms.size());

    // p50: median latency. p95: tail-latency indicator.
    const double p50_ms = PercentileMs(elapsed_ms, 0.50);
    const double p95_ms = PercentileMs(elapsed_ms, 0.95);

    // Approximate throughput (frames/sec) from average latency.
    // 1000 ms / avg_ms gives frames processed per second.
    const double fps = avg_ms > 0.0 ? 1000.0 / avg_ms : 0.0;

    // 5) Emit machine-readable key=value output for easy parsing in scripts.
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "buildgraph_benchmark_result\n";
    std::cout << "nodes_per_frame=" << cfg.nodes << "\n";
    std::cout << "timed_frames=" << cfg.frames << "\n";
    std::cout << "warmup_frames=" << cfg.warmup << "\n";
    std::cout << "avg_ms=" << avg_ms << "\n";
    std::cout << "p50_ms=" << p50_ms << "\n";
    std::cout << "p95_ms=" << p95_ms << "\n";
    std::cout << "approx_fps=" << fps << "\n";
    std::cout << "last_graph_nodes=" << output_nodes << "\n";
    std::cout << "last_graph_edges=" << output_edges << "\n";

    return 0;
}
