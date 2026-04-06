#pragma once

namespace graph_slam_hls_draft {

constexpr int MAX_REG_CORRESPONDENCES = 16384;

// HLS-friendly accumulation kernel draft for registration normal equations.
// Inputs are flattened XYZ arrays and integer semantic labels.
extern "C" void registration_accumulate_kernel(const float src_xyz[MAX_REG_CORRESPONDENCES * 3],
                                                const float tgt_xyz[MAX_REG_CORRESPONDENCES * 3],
                                                const int labels[MAX_REG_CORRESPONDENCES],
                                                int correspondence_count,
                                                float kernel,
                                                double jtj_out[36],
                                                double jtr_out[6],
                                                int *used_count,
                                                int *dropped_count);

}  // namespace graph_slam_hls_draft
