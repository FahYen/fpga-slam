#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace graph_slam {

int RemapSemanticKittiLabel(std::int32_t raw_label);

std::vector<int> RemapSemanticKittiLabels(const std::int32_t *raw_labels,
                                         std::size_t count);

}  // namespace graph_slam
