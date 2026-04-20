#include "SemanticKitti.hpp"

#include <unordered_map>

namespace graph_slam {
namespace {

const std::unordered_map<int, int> kSemanticKittiToSgslam = {
    {0, 0},   {1, 0},   {10, 1},  {11, 2},  {13, 5},  {15, 3},  {16, 5},
    {18, 4},  {20, 5},  {30, 6},  {31, 7},  {32, 8},  {40, 9},  {44, 10},
    {48, 11}, {49, 12}, {50, 13}, {51, 14}, {52, 0},  {60, 9},  {70, 15},
    {71, 16}, {72, 17}, {80, 18}, {81, 19}, {99, 0},  {252, 20}, {253, 21},
    {254, 22}, {255, 23}, {256, 24}, {257, 24}, {258, 25}, {259, 24},
};

}  // namespace

int RemapSemanticKittiLabel(std::int32_t raw_label) {
    const int semantic_label = static_cast<int>(raw_label & 0x0000ffff);
    const auto it = kSemanticKittiToSgslam.find(semantic_label);
    if (it == kSemanticKittiToSgslam.end()) {
        return 0;
    }
    return it->second;
}

std::vector<int> RemapSemanticKittiLabels(const std::int32_t *raw_labels,
                                         std::size_t count) {
    std::vector<int> remapped;
    remapped.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        remapped.emplace_back(RemapSemanticKittiLabel(raw_labels[i]));
    }
    return remapped;
}

}  // namespace graph_slam
