#include "FrameSource.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

#include "rnsg_ipc.h"

namespace semgraph_slam_app {
namespace {

std::int64_t TimeoutNs(double timeout_s) {
    if (timeout_s < 0.0) {
        return -1;
    }
    if (timeout_s == 0.0) {
        return 0;
    }
    const double timeout_ns = timeout_s * 1e9;
    if (timeout_ns > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
        return std::numeric_limits<std::int64_t>::max();
    }
    return static_cast<std::int64_t>(timeout_ns);
}

void CheckRc(rnsg_status rc, const std::string &what) {
    if (rc == RNSG_OK) {
        return;
    }
    throw std::runtime_error(what + " failed with rc=" + std::to_string(rc));
}

std::vector<float> ReadFloatFile(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open " + path.string());
    }
    input.seekg(0, std::ios::end);
    const std::streamsize bytes = input.tellg();
    input.seekg(0, std::ios::beg);
    if (bytes < 0 || (bytes % static_cast<std::streamsize>(sizeof(float))) != 0) {
        throw std::runtime_error("Invalid float file: " + path.string());
    }
    std::vector<float> values(static_cast<std::size_t>(bytes / static_cast<std::streamsize>(sizeof(float))));
    input.read(reinterpret_cast<char *>(values.data()), bytes);
    if (!input) {
        throw std::runtime_error("Failed to read " + path.string());
    }
    return values;
}

std::vector<std::int32_t> ReadLabelFile(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open " + path.string());
    }
    input.seekg(0, std::ios::end);
    const std::streamsize bytes = input.tellg();
    input.seekg(0, std::ios::beg);
    if (bytes < 0 || (bytes % static_cast<std::streamsize>(sizeof(std::int32_t))) != 0) {
        throw std::runtime_error("Invalid label file: " + path.string());
    }
    std::vector<std::int32_t> values(static_cast<std::size_t>(bytes / static_cast<std::streamsize>(sizeof(std::int32_t))));
    input.read(reinterpret_cast<char *>(values.data()), bytes);
    if (!input) {
        throw std::runtime_error("Failed to read " + path.string());
    }
    return values;
}

}  // namespace

FrameLease::FrameLease(FrameLease &&other) noexcept {
    *this = std::move(other);
}

FrameLease &FrameLease::operator=(FrameLease &&other) noexcept {
    if (this != &other) {
        Reset();
        view = other.view;
        frame_id = other.frame_id;
        consumed_index = other.consumed_index;
        skipped_before = other.skipped_before;
        capture_ns = other.capture_ns;
        publish_ns = other.publish_ns;
        acquire_wait_ms = other.acquire_wait_ms;
        source_name = std::move(other.source_name);
        owned_points = std::move(other.owned_points);
        owned_labels = std::move(other.owned_labels);
        release_fn = std::move(other.release_fn);
        other.view = {};
        other.frame_id = 0;
        other.consumed_index = 0;
        other.skipped_before = 0;
        other.capture_ns = 0;
        other.publish_ns = 0;
        other.acquire_wait_ms = 0.0;
    }
    return *this;
}

FrameLease::~FrameLease() {
    Reset();
}

void FrameLease::Reset() {
    if (release_fn) {
        try {
            release_fn();
        } catch (const std::exception &e) {
            std::cerr << "Frame release warning: " << e.what() << std::endl;
        }
        release_fn = {};
    }
    view = {};
    frame_id = 0;
    consumed_index = 0;
    skipped_before = 0;
    capture_ns = 0;
    publish_ns = 0;
    acquire_wait_ms = 0.0;
    source_name.clear();
    owned_points.clear();
    owned_labels.clear();
}

FileFrameSource::FileFrameSource(std::filesystem::path lidar_dir,
                                 std::filesystem::path label_dir,
                                 std::size_t max_frames)
    : lidar_dir_(std::move(lidar_dir)), label_dir_(std::move(label_dir)) {
    if (!std::filesystem::is_directory(lidar_dir_)) {
        throw std::runtime_error("Missing lidar directory: " + lidar_dir_.string());
    }
    if (!std::filesystem::is_directory(label_dir_)) {
        throw std::runtime_error("Missing label directory: " + label_dir_.string());
    }

    for (const auto &entry : std::filesystem::directory_iterator(lidar_dir_)) {
        if (entry.is_regular_file() && entry.path().extension() == ".bin") {
            scan_paths_.push_back(entry.path());
        }
    }
    std::sort(scan_paths_.begin(), scan_paths_.end());
    if (max_frames > 0 && scan_paths_.size() > max_frames) {
        scan_paths_.resize(max_frames);
    }
}

FrameFetchStatus FileFrameSource::Next(FrameLease &lease) {
    lease.Reset();
    if (next_index_ >= scan_paths_.size()) {
        return FrameFetchStatus::kEnd;
    }

    const auto &scan_path = scan_paths_[next_index_];
    const auto label_path = label_dir_ / scan_path.filename().replace_extension(".label");
    lease.owned_points = ReadFloatFile(scan_path);
    lease.owned_labels = ReadLabelFile(label_path);
    if ((lease.owned_points.size() % 4U) != 0U) {
        throw std::runtime_error("Invalid point count in " + scan_path.string());
    }
    if ((lease.owned_points.size() / 4U) != lease.owned_labels.size()) {
        throw std::runtime_error("Point/label count mismatch for " + scan_path.string());
    }

    lease.view.points_xyzi = lease.owned_points.data();
    lease.view.raw_labels = lease.owned_labels.data();
    lease.view.num_points = lease.owned_labels.size();
    lease.frame_id = next_index_;
    lease.consumed_index = next_index_;
    lease.source_name = scan_path.string();
    ++next_index_;
    return FrameFetchStatus::kOk;
}

IpcFrameSource::IpcFrameSource(std::string ring_name, double timeout_s)
    : ring_name_(std::move(ring_name)), timeout_s_(timeout_s) {
    rnsg_ring *ring = nullptr;
    CheckRc(rnsg_open(ring_name_.c_str(), &ring), "rnsg_open(" + ring_name_ + ")");
    ring_ = ring;
}

IpcFrameSource::~IpcFrameSource() {
    if (ring_ != nullptr) {
        rnsg_close(ring_);
        ring_ = nullptr;
    }
}

FrameFetchStatus IpcFrameSource::Next(FrameLease &lease) {
    lease.Reset();
    rnsg_frame_view view{};
    const auto t0 = std::chrono::steady_clock::now();
    const rnsg_status rc = rnsg_consumer_acquire(ring_, TimeoutNs(timeout_s_), &view);
    const auto t1 = std::chrono::steady_clock::now();
    lease.acquire_wait_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();

    if (rc == RNSG_TIMEOUT) {
        return FrameFetchStatus::kTimeout;
    }
    CheckRc(rc, "rnsg_consumer_acquire(" + ring_name_ + ")");

    lease.view.points_xyzi = view.points;
    lease.view.raw_labels = view.labels;
    lease.view.num_points = view.num_points;
    lease.frame_id = view.frame_id;
    lease.consumed_index = view.consumed_index;
    lease.skipped_before = view.skipped_before;
    lease.capture_ns = view.capture_ns;
    lease.publish_ns = view.publish_ns;
    lease.source_name = ring_name_;
    lease.release_fn = [ring = ring_]() {
        const rnsg_status release_rc = rnsg_consumer_release(ring);
        if (release_rc != RNSG_OK) {
            std::cerr << "rnsg_consumer_release failed with rc=" << release_rc << std::endl;
        }
    };
    return FrameFetchStatus::kOk;
}

}  // namespace semgraph_slam_app
