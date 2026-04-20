#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "rnsg_ipc.h"
#include "semgraph_slam/core/Coreutils.h"

namespace semgraph_slam_app {

enum class FrameFetchStatus {
    kOk,
    kTimeout,
    kEnd,
};

struct FrameLease {
    FrameLease() = default;
    FrameLease(const FrameLease &) = delete;
    FrameLease &operator=(const FrameLease &) = delete;
    FrameLease(FrameLease &&other) noexcept;
    FrameLease &operator=(FrameLease &&other) noexcept;
    ~FrameLease();

    void Reset();

    graph_slam::BorrowedFrameView view{};
    std::uint64_t frame_id = 0;
    std::uint64_t consumed_index = 0;
    std::uint64_t skipped_before = 0;
    std::uint64_t capture_ns = 0;
    std::uint64_t publish_ns = 0;
    double acquire_wait_ms = 0.0;
    std::string source_name;

    std::vector<float> owned_points;
    std::vector<std::int32_t> owned_labels;
    std::function<void()> release_fn;
};

class FrameSource {
public:
    virtual ~FrameSource() = default;
    virtual FrameFetchStatus Next(FrameLease &lease) = 0;
};

class FileFrameSource final : public FrameSource {
public:
    FileFrameSource(std::filesystem::path lidar_dir,
                    std::filesystem::path label_dir,
                    std::size_t max_frames = 0);

    FrameFetchStatus Next(FrameLease &lease) override;

private:
    std::filesystem::path lidar_dir_;
    std::filesystem::path label_dir_;
    std::vector<std::filesystem::path> scan_paths_;
    std::size_t next_index_ = 0;
};

class IpcFrameSource final : public FrameSource {
public:
    IpcFrameSource(std::string ring_name, double timeout_s);
    ~IpcFrameSource() override;

    FrameFetchStatus Next(FrameLease &lease) override;

private:
    rnsg_ring *ring_ = nullptr;
    std::string ring_name_;
    double timeout_s_ = -1.0;
};

}  // namespace semgraph_slam_app
