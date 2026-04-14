#include "registration_kernel_draft.hpp"
#include <algorithm>

namespace graph_slam_hls_draft {

namespace {

#if defined(__SYNTHESIS__) || defined(HLS_SYNTHESIS)
#define SGSLAM_HLS_PRAGMA(x) _Pragma(#x)
#else
#define SGSLAM_HLS_PRAGMA(x)
#endif

inline float sqr(float x) { return x * x; }

inline float SemanticWeight(int label) {
    switch (label) {
        case 16: case 18: case 19: return 1.2f;
        case 20: case 21: case 22: case 23: case 24: case 25: return 0.0f;
        default: return 1.0f;
    }
}

}  // namespace

extern "C" void registration_accumulate_kernel(const float src_xyz[MAX_REG_CORRESPONDENCES * 3],
                                                const float tgt_xyz[MAX_REG_CORRESPONDENCES * 3],
                                                const int labels[MAX_REG_CORRESPONDENCES],
                                                int correspondence_count,
                                                float kernel,
                                                double jtj_out[36],
                                                double jtr_out[6],
                                                int *used_count,
                                                int *dropped_count) {
    // Interface mappings
    SGSLAM_HLS_PRAGMA(HLS INTERFACE m_axi port=src_xyz offset=slave bundle=gmem0)
    SGSLAM_HLS_PRAGMA(HLS INTERFACE m_axi port=tgt_xyz offset=slave bundle=gmem1)
    SGSLAM_HLS_PRAGMA(HLS INTERFACE m_axi port=labels offset=slave bundle=gmem2)
    SGSLAM_HLS_PRAGMA(HLS INTERFACE m_axi port=jtj_out offset=slave bundle=gmem3)
    SGSLAM_HLS_PRAGMA(HLS INTERFACE m_axi port=jtr_out offset=slave bundle=gmem3)
    SGSLAM_HLS_PRAGMA(HLS INTERFACE m_axi port=used_count offset=slave bundle=gmem4)
    SGSLAM_HLS_PRAGMA(HLS INTERFACE m_axi port=dropped_count offset=slave bundle=gmem4)

    SGSLAM_HLS_PRAGMA(HLS INTERFACE s_axilite port=correspondence_count bundle=control)
    SGSLAM_HLS_PRAGMA(HLS INTERFACE s_axilite port=kernel bundle=control)
    SGSLAM_HLS_PRAGMA(HLS INTERFACE s_axilite port=return bundle=control)

    // LOCAL MEMORY: Eliminate AXI latency from the inner loop
    // 21 elements for the upper triangular portion of the 6x6 symmetric matrix
    double local_jtj[21]; 
    double local_jtr[6];
    
    // Completely partition the local arrays so all elements can be accessed simultaneously
    SGSLAM_HLS_PRAGMA(HLS ARRAY_PARTITION variable=local_jtj complete dim=1)
    SGSLAM_HLS_PRAGMA(HLS ARRAY_PARTITION variable=local_jtr complete dim=1)

    // Initialization
    for (int i = 0; i < 21; ++i) { SGSLAM_HLS_PRAGMA(HLS UNROLL) local_jtj[i] = 0.0; }
    for (int i = 0; i < 6; ++i) { SGSLAM_HLS_PRAGMA(HLS UNROLL) local_jtr[i] = 0.0; }

    const int requested = correspondence_count < 0 ? 0 : correspondence_count;
    const int bounded_count = std::min(requested, (int)MAX_REG_CORRESPONDENCES);

    if (used_count != nullptr) *used_count = bounded_count;
    if (dropped_count != nullptr) *dropped_count = requested - bounded_count;

    const float kernel_sq = sqr(kernel);

    // --- MAIN ACCUMULATION LOOP ---
    for (int i = 0; i < bounded_count; ++i) {
        // Vitis will attempt to pipeline this. Because of double-precision 
        // accumulation, it may achieve II=4 to II=8. This is still immensely faster 
        // than an AXI-bound loop.
        SGSLAM_HLS_PRAGMA(HLS PIPELINE)
        SGSLAM_HLS_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=16384 avg=4096)

        const int base = i * 3;

        // Keep local math in float as long as possible to save DSP slices
        const float sx = src_xyz[base + 0];
        const float sy = src_xyz[base + 1];
        const float sz = src_xyz[base + 2];

        const float tx = tgt_xyz[base + 0];
        const float ty = tgt_xyz[base + 1];
        const float tz = tgt_xyz[base + 2];

        const float rx = sx - tx;
        const float ry = sy - ty;
        const float rz = sz - tz;

        const float residual2 = rx * rx + ry * ry + rz * rz;
        const float w_float = kernel_sq / (sqr(kernel) + residual2);
        const float semantic_w_float = SemanticWeight(labels[i]);
        
        // Cast combined weight to double only at the accumulation stage
        const double w = static_cast<double>(w_float * semantic_w_float);

        double J[3][6];
        SGSLAM_HLS_PRAGMA(HLS ARRAY_PARTITION variable=J complete dim=0)
        
        // Populate Jacobians (cast geometry to double here for matrix math)
        J[0][0] = 1.0; J[0][1] = 0.0; J[0][2] = 0.0; J[0][3] = 0.0; J[0][4] = sz; J[0][5] = -sy;
        J[1][0] = 0.0; J[1][1] = 1.0; J[1][2] = 0.0; J[1][3] = -sz; J[1][4] = 0.0; J[1][5] = sx;
        J[2][0] = 0.0; J[2][1] = 0.0; J[2][2] = 1.0; J[2][3] = sy; J[2][4] = -sx; J[2][5] = 0.0;

        const double r[3] = {rx, ry, rz};
        SGSLAM_HLS_PRAGMA(HLS ARRAY_PARTITION variable=r complete dim=0)

        // Compute JTr
        for (int c = 0; c < 6; ++c) {
            SGSLAM_HLS_PRAGMA(HLS UNROLL)
            double jtr_acc = 0.0;
            for (int rr = 0; rr < 3; ++rr) {
                SGSLAM_HLS_PRAGMA(HLS UNROLL)
                jtr_acc += J[rr][c] * r[rr];
            }
            local_jtr[c] += w * jtr_acc;
        }

        // Compute JTJ (Upper Triangular only)
        int idx = 0;
        for (int c0 = 0; c0 < 6; ++c0) {
            SGSLAM_HLS_PRAGMA(HLS UNROLL)
            for (int c1 = c0; c1 < 6; ++c1) { // Note: c1 starts at c0
                SGSLAM_HLS_PRAGMA(HLS UNROLL)
                double jtj_acc = 0.0;
                for (int rr = 0; rr < 3; ++rr) {
                    SGSLAM_HLS_PRAGMA(HLS UNROLL)
                    jtj_acc += J[rr][c0] * J[rr][c1];
                }
                local_jtj[idx++] += w * jtj_acc;
            }
        }
    }

    // --- WRITEBACK LOOP ---
    // Expand the 21 upper triangular elements back to the 36-element global array
    int read_idx = 0;
    for (int r = 0; r < 6; ++r) {
        for (int c = r; c < 6; ++c) {
            SGSLAM_HLS_PRAGMA(HLS PIPELINE II=1)
            double val = local_jtj[read_idx++];
            jtj_out[r * 6 + c] = val;
            if (r != c) {
                jtj_out[c * 6 + r] = val; // Mirror across diagonal
            }
        }
    }

    for (int i = 0; i < 6; ++i) {
        SGSLAM_HLS_PRAGMA(HLS PIPELINE II=1)
        jtr_out[i] = local_jtr[i];
    }
}

}  // namespace graph_slam_hls_draft