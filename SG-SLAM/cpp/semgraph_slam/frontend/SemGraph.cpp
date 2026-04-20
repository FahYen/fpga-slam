// This file is covered by the LICENSE file in the root of this project.
// contact: Neng Wang, <neng.wang@hotmail.com>

#include "SemGraph.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <tbb/parallel_for.h>
#include <utility>
#include <vector>

namespace graph_slam {

namespace {

constexpr int kVehicleLabel = 1;
constexpr int kTrunkLabel = 2;
constexpr int kPoleLikeLabel = 3;

struct BuildGraphNodeInput {
    Eigen::Vector3d center;
    Eigen::Vector3d dimension;
    int label = -1;
    int points_num = 0;
};

int HistogramOffsetForLabel(int label, int subinterval) {
    if (label == kVehicleLabel) return 0;
    if (label == kTrunkLabel) return subinterval;
    if (label == kPoleLikeLabel) return subinterval * 2;
    return -1;
}

int HistogramBin(double edge, int sub_interval_value, int subinterval) {
    const int safe_subinterval = std::max(1, subinterval);
    const int safe_sub_interval_value = std::max(1, sub_interval_value);
    const int raw_bin = static_cast<int>(edge / static_cast<double>(safe_sub_interval_value));
    return std::min(safe_subinterval - 1, std::max(0, raw_bin));
}

void AccumulateLocalEmbedding(Eigen::MatrixXd &node_embeddings_local,
                              std::size_t row,
                              int label,
                              double edge,
                              int sub_interval_value,
                              int subinterval) {
    const int offset = HistogramOffsetForLabel(label, subinterval);
    if (offset < 0 || node_embeddings_local.cols() == 0) return;

    const int bin = HistogramBin(edge, sub_interval_value, subinterval);
    node_embeddings_local(static_cast<Eigen::Index>(row),
                          static_cast<Eigen::Index>(offset + bin)) += 1.0;
}

Graph BuildGraphFromNodes(const std::vector<BuildGraphNodeInput> &nodes,
                          double edge_dis_th,
                          int subinterval,
                          int graph_node_dimension,
                          double subgraph_edge_th) {
    Graph frame_graph;
    const std::size_t N = nodes.size();
    if (N == 0) return frame_graph;

    const int safe_subinterval = std::max(1, subinterval);
    const int local_descriptor_dim = safe_subinterval * 3;
    const int sub_interval_value = std::max(1, static_cast<int>(edge_dis_th / safe_subinterval));
    const double edge_dis_th2 = edge_dis_th * edge_dis_th;

    // TODO(M1-FPGA): Baseline this function with benchmarks/benchmark_buildgraph.cpp and track
    // avg/p95 latency vs node count for KITTI and MulRAN scenes.
    Eigen::MatrixXd AdjacencyMatrix = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N),
                                                            static_cast<Eigen::Index>(N));
    Eigen::MatrixXd EdgeMatrix = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N),
                                                       static_cast<Eigen::Index>(N));
    Eigen::MatrixXd NodeEmbeddings = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N),
                                                           local_descriptor_dim + graph_node_dimension);
    Eigen::MatrixXd NodeEmbeddings_Local = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(N),
                                                                 local_descriptor_dim);
    std::vector<std::vector<int>> subgraph_neighbors(N);

    frame_graph.node_labels.resize(N);
    frame_graph.node_centers.resize(N);
    frame_graph.node_dimensions.resize(N);
    frame_graph.points_num.resize(N);

    for (std::size_t i = 0; i < N; ++i) {
        frame_graph.node_labels[i] = nodes[i].label;
        frame_graph.node_centers[i] = nodes[i].center;
        frame_graph.node_dimensions[i] = nodes[i].dimension;
        frame_graph.points_num[i] = nodes[i].points_num;

        if (nodes[i].label == kVehicleLabel) {
            ++frame_graph.vehicle_num;
        } else if (nodes[i].label == kTrunkLabel) {
            ++frame_graph.trunk_num;
        } else if (nodes[i].label == kPoleLikeLabel) {
            ++frame_graph.pole_like_num;
        }
    }

    // TODO(M1-FPGA): This O(N^2) distance loop is the primary kernel candidate.
    // Capture tripcounts and memory access pattern before translating to HLS.
    tbb::parallel_for(std::size_t(0), N, [&](std::size_t i) {
        const auto i_idx = static_cast<Eigen::Index>(i);
        const auto &center_i = frame_graph.node_centers[i];
        auto &neighbors = subgraph_neighbors[i];
        if (N > 1) {
            neighbors.reserve(std::min<std::size_t>(N - 1, 32));
        }

        AdjacencyMatrix(i_idx, i_idx) = 1.0;
        AccumulateLocalEmbedding(NodeEmbeddings_Local,
                                 i,
                                 frame_graph.node_labels[i],
                                 0.0,
                                 sub_interval_value,
                                 safe_subinterval);

        for (std::size_t j = 0; j < N; ++j) {
            if (i == j) continue;

            const double edge2 = (center_i - frame_graph.node_centers[j]).squaredNorm();
            if (edge2 >= edge_dis_th2) continue;

            const double edge = std::sqrt(edge2);
            AdjacencyMatrix(i_idx, static_cast<Eigen::Index>(j)) = 1.0;
            EdgeMatrix(i_idx, static_cast<Eigen::Index>(j)) = edge;

            AccumulateLocalEmbedding(NodeEmbeddings_Local,
                                     i,
                                     frame_graph.node_labels[j],
                                     edge,
                                     sub_interval_value,
                                     safe_subinterval);
            if (edge < subgraph_edge_th) {
                neighbors.emplace_back(static_cast<int>(j));
            }
        }
    });

    const double inv_edge_dis_th = edge_dis_th > 0.0 ? 1.0 / edge_dis_th : 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i + 1; j < N; ++j) {
            if (AdjacencyMatrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) == 0.0) continue;

            const double edge = EdgeMatrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
            frame_graph.edges.emplace_back(std::make_pair(i, j));
            frame_graph.edge_value.emplace_back(edge);
            frame_graph.edge_weights.emplace_back((edge_dis_th - edge) * inv_edge_dis_th);
        }
    }

    // TODO(M1-FPGA): Eigen eigendecomposition is difficult to synthesize directly.
    // For accelerator draft, replace with a fixed-point approximation or offload only
    // adjacency/edge histogram construction while keeping decomposition on CPU.
    Eigen::MatrixXd NodeEmbeddings_Global = MatrixDecomposing(AdjacencyMatrix, graph_node_dimension);
    NodeEmbeddings_Local =
        NodeEmbeddings_Local.array().colwise() / NodeEmbeddings_Local.rowwise().norm().array();
    NodeEmbeddings_Global =
        NodeEmbeddings_Global.array().colwise() / NodeEmbeddings_Global.rowwise().sum().array();
    NodeEmbeddings.leftCols(NodeEmbeddings_Local.cols()) = NodeEmbeddings_Local;
    if (NodeEmbeddings_Global.cols() > 0) {
        NodeEmbeddings.rightCols(NodeEmbeddings_Global.cols()) = NodeEmbeddings_Global;
    }

    frame_graph.node_desc.resize(N);
    tbb::parallel_for(std::size_t(0), N, [&](std::size_t i) {
        std::vector<float> node_descriptor(NodeEmbeddings.cols());
        for (Eigen::Index col = 0; col < NodeEmbeddings.cols(); ++col) {
            node_descriptor[static_cast<std::size_t>(col)] =
                static_cast<float>(NodeEmbeddings(static_cast<Eigen::Index>(i), col));
        }
        frame_graph.node_desc[i] = std::move(node_descriptor);
    });

    // TODO(M1-FPGA): Consider moving this triangle enumeration to a separate kernel only
    // if profiling shows non-trivial cost after accelerating edge construction.
    frame_graph.node_sub_triangles.resize(N);
    tbb::parallel_for(std::size_t(0), N, [&](std::size_t i) {
        const auto &indices = subgraph_neighbors[i];
        std::vector<Eigen::Vector3d> node_subgraph_triangle;
        if (indices.size() > 2) {
            node_subgraph_triangle.reserve(indices.size() * (indices.size() - 1) / 2);
            for (std::size_t m = 0; m + 1 < indices.size(); ++m) {
                for (std::size_t n = m + 1; n < indices.size(); ++n) {
                    std::array<float, 3> sub_triangle = {
                        static_cast<float>(
                            (frame_graph.node_centers[i] - frame_graph.node_centers[indices[m]]).norm()),
                        static_cast<float>(
                            (frame_graph.node_centers[i] - frame_graph.node_centers[indices[n]]).norm()),
                        static_cast<float>((frame_graph.node_centers[indices[m]] -
                                            frame_graph.node_centers[indices[n]])
                                               .norm())};
                    std::sort(sub_triangle.begin(), sub_triangle.end());
                    node_subgraph_triangle.emplace_back(sub_triangle[0], sub_triangle[1], sub_triangle[2]);
                }
            }
        }
        frame_graph.node_sub_triangles[i] = std::move(node_subgraph_triangle);
    });

    frame_graph.edge_matrix = std::move(EdgeMatrix);  // for subsequent correspondences pruning
    return frame_graph;
}

}  // namespace

/*
    Building semantic graph from clustered bounding boxes
*/
Graph BuildGraph(const std::vector<Bbox> &cluster_boxes,
                 double edge_dis_th,
                 int subinterval,
                 int graph_node_dimension,
                 double subgraph_edge_th) {
    std::vector<BuildGraphNodeInput> nodes;
    nodes.reserve(cluster_boxes.size());
    for (const auto &cluster_box : cluster_boxes) {
        nodes.emplace_back(BuildGraphNodeInput{cluster_box.center,
                                               cluster_box.dimension,
                                               cluster_box.label,
                                               cluster_box.points_num});
    }
    return BuildGraphFromNodes(nodes,
                               edge_dis_th,
                               subinterval,
                               graph_node_dimension,
                               subgraph_edge_th);
}

/*
    Rebuilding graph for local graph map
*/
Graph ReBuildGraph(const std::vector<InsNode> &cluster_boxes,
                   double edge_dis_th,
                   int subinterval,
                   int graph_node_dimension,
                   double subgraph_edge_th) {
    std::vector<BuildGraphNodeInput> nodes;
    nodes.reserve(cluster_boxes.size());
    for (const auto &cluster_box : cluster_boxes) {
        nodes.emplace_back(BuildGraphNodeInput{cluster_box.pose,
                                               cluster_box.dimension,
                                               cluster_box.label,
                                               cluster_box.points_num});
    }
    return BuildGraphFromNodes(nodes,
                               edge_dis_th,
                               subinterval,
                               graph_node_dimension,
                               subgraph_edge_th);
}

/*
    Decomposing adjacency matrix to get node vector
*/
Eigen::MatrixXd MatrixDecomposing(const Eigen::MatrixXd &MatrixInput, int Dimension) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    es.compute(MatrixInput);

    const Eigen::MatrixXd evecs_abs = es.eigenvectors().cwiseAbs();
    const Eigen::VectorXd abs_evals = es.eigenvalues().cwiseAbs();

    std::vector<int> indices(static_cast<std::size_t>(abs_evals.size()));
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int i, int j) {
        return abs_evals[i] > abs_evals[j];
    });

    Eigen::MatrixXd evecs_sort =
        Eigen::MatrixXd::Zero(abs_evals.size(), std::max(0, Dimension));
    const std::size_t iter =
        std::min<std::size_t>(std::max(0, Dimension), indices.size());
    for (std::size_t i = 0; i < iter; ++i) {
        evecs_sort.col(static_cast<Eigen::Index>(i)) = evecs_abs.col(indices[i]);
    }

    return evecs_sort;
}

}  // namespace graph_slam

