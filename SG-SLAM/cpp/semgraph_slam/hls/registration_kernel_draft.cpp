#include "registration_kernel_draft.hpp"

#include <algorithm>

namespace graph_slam_hls_draft {

namespace {

#if defined(__SYNTHESIS__) || defined(HLS_SYNTHESIS)
#define SGSLAM_HLS_PRAGMA(x) _Pragma(#x)
#else
#define SGSLAM_HLS_PRAGMA(x)
#endif

inline double sqr(double x) { return x * x; }

inline double SemanticWeight(int label) {
    // Match current CPU weighting policy for moving classes and selected static classes.
    switch (label) {
        case 16:  // trunk
        case 18:  // pole
        case 19:  // traffic-sign
            return 1.2;
        case 20:  // moving-car
        case 21:  // moving-bicyclist
        case 22:  // moving-person
        case 23:  // moving-motorcyclist
        case 24:  // moving-other-vehicle
        case 25:  // moving-truck
            return 0.0;
        default:
            return 1.0;
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
    // Vivado/Vitis HLS interface pragmas. They are emitted only in synthesis builds.
    SGSLAM_HLS_PRAGMA(HLS INTERFACE m_axi port=src_xyz offset=slave bundle=gmem0 depth=49152)
    SGSLAM_HLS_PRAGMA(HLS INTERFACE m_axi port=tgt_xyz offset=slave bundle=gmem1 depth=49152)
    SGSLAM_HLS_PRAGMA(HLS INTERFACE m_axi port=labels offset=slave bundle=gmem2 depth=16384)
    SGSLAM_HLS_PRAGMA(HLS INTERFACE m_axi port=jtj_out offset=slave bundle=gmem3 depth=36)
    SGSLAM_HLS_PRAGMA(HLS INTERFACE m_axi port=jtr_out offset=slave bundle=gmem3 depth=6)
    SGSLAM_HLS_PRAGMA(HLS INTERFACE m_axi port=used_count offset=slave bundle=gmem4 depth=1)
    SGSLAM_HLS_PRAGMA(HLS INTERFACE m_axi port=dropped_count offset=slave bundle=gmem4 depth=1)

    SGSLAM_HLS_PRAGMA(HLS INTERFACE s_axilite port=correspondence_count bundle=control)
    SGSLAM_HLS_PRAGMA(HLS INTERFACE s_axilite port=kernel bundle=control)
    SGSLAM_HLS_PRAGMA(HLS INTERFACE s_axilite port=return bundle=control)

    for (int i = 0; i < 36; ++i) {
        SGSLAM_HLS_PRAGMA(HLS PIPELINE II=1)
        jtj_out[i] = 0.0;
    }

    for (int i = 0; i < 6; ++i) {
        SGSLAM_HLS_PRAGMA(HLS PIPELINE II=1)
        jtr_out[i] = 0.0;
    }

    const int requested = correspondence_count < 0 ? 0 : correspondence_count;
    const int bounded_count = std::min(requested, MAX_REG_CORRESPONDENCES);

    if (used_count != nullptr) *used_count = bounded_count;
    if (dropped_count != nullptr) *dropped_count = requested - bounded_count;

    const double kernel_sq = sqr(static_cast<double>(kernel));

    for (int i = 0; i < bounded_count; ++i) {
        SGSLAM_HLS_PRAGMA(HLS PIPELINE II=1)
        SGSLAM_HLS_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=16384 avg=4096)

        const int base = i * 3;

        const double sx = static_cast<double>(src_xyz[base + 0]);
        const double sy = static_cast<double>(src_xyz[base + 1]);
        const double sz = static_cast<double>(src_xyz[base + 2]);

        const double tx = static_cast<double>(tgt_xyz[base + 0]);
        const double ty = static_cast<double>(tgt_xyz[base + 1]);
        const double tz = static_cast<double>(tgt_xyz[base + 2]);

        const double rx = sx - tx;
        const double ry = sy - ty;
        const double rz = sz - tz;

        const double residual2 = rx * rx + ry * ry + rz * rz;
        const double w = kernel_sq / sqr(static_cast<double>(kernel) + residual2);
        const double semantic_w = SemanticWeight(labels[i]);

        // J = [I | -hat(s)] where s = source point in local map frame.
        // Row-major layout: J[row][col].
        double J[3][6];
        SGSLAM_HLS_PRAGMA(HLS ARRAY_PARTITION variable=J complete dim=0)
        J[0][0] = 1.0;
        J[0][1] = 0.0;
        J[0][2] = 0.0;
        J[0][3] = 0.0;
        J[0][4] = sz;
        J[0][5] = -sy;

        J[1][0] = 0.0;
        J[1][1] = 1.0;
        J[1][2] = 0.0;
        J[1][3] = -sz;
        J[1][4] = 0.0;
        J[1][5] = sx;

        J[2][0] = 0.0;
        J[2][1] = 0.0;
        J[2][2] = 1.0;
        J[2][3] = sy;
        J[2][4] = -sx;
        J[2][5] = 0.0;

        const double r[3] = {rx, ry, rz};
        SGSLAM_HLS_PRAGMA(HLS ARRAY_PARTITION variable=r complete dim=0)

        for (int c = 0; c < 6; ++c) {
            SGSLAM_HLS_PRAGMA(HLS UNROLL)
            double jtr_acc = 0.0;
            for (int rr = 0; rr < 3; ++rr) {
                SGSLAM_HLS_PRAGMA(HLS UNROLL)
                jtr_acc += J[rr][c] * r[rr];
            }
            jtr_out[c] += w * semantic_w * jtr_acc;
        }

        for (int c0 = 0; c0 < 6; ++c0) {
            SGSLAM_HLS_PRAGMA(HLS UNROLL)
            for (int c1 = 0; c1 < 6; ++c1) {
                SGSLAM_HLS_PRAGMA(HLS UNROLL)
                double jtj_acc = 0.0;
                for (int rr = 0; rr < 3; ++rr) {
                    SGSLAM_HLS_PRAGMA(HLS UNROLL)
                    jtj_acc += J[rr][c0] * J[rr][c1];
                }
                jtj_out[c0 * 6 + c1] += w * jtj_acc;
            }
        }
    }
}

#undef SGSLAM_HLS_PRAGMA

}  // namespace graph_slam_hls_draft
