#include <cmath>
#include <cstddef>

namespace graph_slam_hls_draft {

constexpr int MAX_NODES = 512;

// Draft kernel for the O(N^2) edge construction phase in BuildGraph.
// This keeps a fixed-size interface and avoids dynamic allocation to stay HLS-friendly.
extern "C" void buildgraph_edge_kernel(const float centers_x[MAX_NODES],
                                        const float centers_y[MAX_NODES],
                                        const float centers_z[MAX_NODES],
                                        const int labels[MAX_NODES],
                                        int node_count,
                                        float edge_threshold,
                                        float adjacency[MAX_NODES * MAX_NODES],
                                        float edge_distance[MAX_NODES * MAX_NODES],
                                        int edge_type_histogram[MAX_NODES * 3]) {
    // TODO(M1-FPGA): Add HLS interface and pipeline pragmas when importing into Vivado HLS.
    // Example (tool-specific):
    // #pragma HLS INTERFACE m_axi port=centers_x offset=slave bundle=gmem0
    // #pragma HLS PIPELINE II=1

    const int bounded_count = (node_count > MAX_NODES) ? MAX_NODES : node_count;

    for (int i = 0; i < bounded_count; ++i) {
        for (int b = 0; b < 3; ++b) {
            edge_type_histogram[i * 3 + b] = 0;
        }
    }

    for (int i = 0; i < bounded_count; ++i) {
        const float xi = centers_x[i];
        const float yi = centers_y[i];
        const float zi = centers_z[i];

        for (int j = 0; j < bounded_count; ++j) {
            const float dx = xi - centers_x[j];
            const float dy = yi - centers_y[j];
            const float dz = zi - centers_z[j];
            const float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
            const int idx = i * MAX_NODES + j;

            if (dist < edge_threshold) {
                adjacency[idx] = 1.0f;
                edge_distance[idx] = dist;

                // Maintain a compact per-node histogram for labels 1/2/3.
                const int label = labels[j];
                if (label == 1) {
                    edge_type_histogram[i * 3 + 0] += 1;
                } else if (label == 2) {
                    edge_type_histogram[i * 3 + 1] += 1;
                } else if (label == 3) {
                    edge_type_histogram[i * 3 + 2] += 1;
                }
            } else {
                adjacency[idx] = 0.0f;
                edge_distance[idx] = 0.0f;
            }
        }
    }
}

}  // namespace graph_slam_hls_draft
