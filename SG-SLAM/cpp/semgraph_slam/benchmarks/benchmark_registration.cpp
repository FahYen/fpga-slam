// Registration micro-benchmark for SG-SLAM.
//
// What this program measures:
// - Runtime of registration using the same RegisterFrameSemantic path used in the pipeline.
// - CPU backend today, with an FPGA backend selector placeholder for the upcoming offload.
// - Pose error against a known synthetic ground-truth pose.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "semgraph_slam/frontend/Registration.hpp"

namespace {

using graph_slam::RegistrationBackend;
using graph_slam::RegisterFrameSemanticWithBackend;
using graph_slam::ParseRegistrationBackend;
using graph_slam::RegistrationBackendName;
using graph_slam::VoxelHashMap;

struct BenchmarkConfig {
    std::size_t max_correspondences = 12000;
    std::size_t frames = 200;
    std::size_t warmup = 20;
    int seed = 570;
    int max_iters = 500;  // informational for now
    double max_correspondence_distance = 3.0;
    double kernel = 1.0 / 3.0;
    double point_noise_std = 0.02;
    RegistrationBackend backend = RegistrationBackend::kCpu;
};

std::size_t ParseSize(const std::string &value) { return static_cast<std::size_t>(std::stoull(value)); }
int ParseInt(const std::string &value) { return std::stoi(value); }
double ParseDouble(const std::string &value) { return std::stod(value); }

BenchmarkConfig ParseArgs(int argc, char **argv) {
    BenchmarkConfig config;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--max-correspondences" && i + 1 < argc) {
            config.max_correspondences = ParseSize(argv[++i]);
        } else if (arg == "--frames" && i + 1 < argc) {
            config.frames = ParseSize(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            config.warmup = ParseSize(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            config.seed = ParseInt(argv[++i]);
        } else if (arg == "--max-iters" && i + 1 < argc) {
            config.max_iters = ParseInt(argv[++i]);
        } else if (arg == "--corr-dist" && i + 1 < argc) {
            config.max_correspondence_distance = ParseDouble(argv[++i]);
        } else if (arg == "--kernel" && i + 1 < argc) {
            config.kernel = ParseDouble(argv[++i]);
        } else if (arg == "--point-noise-std" && i + 1 < argc) {
            config.point_noise_std = ParseDouble(argv[++i]);
        } else if (arg == "--backend" && i + 1 < argc) {
            config.backend = ParseRegistrationBackend(argv[++i]);
        } else if (arg == "--help") {
            std::cout
                << "Usage: benchmark_registration [options]\n"
                << "  --backend <cpu|fpga>       Registration backend\n"
                << "  --max-correspondences <N>  Number of synthetic points\n"
                << "  --frames <N>               Number of timed frames\n"
                << "  --warmup <N>               Warmup frame count\n"
                << "  --seed <N>                 RNG seed\n"
                << "  --max-iters <N>            Informational (registration currently fixed at 500)\n"
                << "  --corr-dist <float>        Max correspondence distance\n"
                << "  --kernel <float>           Robust kernel width\n"
                << "  --point-noise-std <float>  Gaussian xyz noise std (meters)\n";
            std::exit(0);
        }
    }
    return config;
}

double PercentileMs(std::vector<double> samples_ms, double percentile) {
    if (samples_ms.empty()) return 0.0;
    std::sort(samples_ms.begin(), samples_ms.end());
    const std::size_t idx = static_cast<std::size_t>(
        std::floor(percentile * static_cast<double>(samples_ms.size() - 1U)));
    return samples_ms[idx];
}

std::vector<Eigen::Vector4d> Transform4D(const Sophus::SE3d &pose,
                                         const std::vector<Eigen::Vector4d> &points) {
    std::vector<Eigen::Vector4d> out(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        out[i].head<3>() = pose * points[i].head<3>();
        out[i](3) = points[i](3);
    }
    return out;
}

std::vector<Eigen::Vector4d> GenerateMapPoints(std::size_t count, std::mt19937 &rng) {
    std::vector<Eigen::Vector4d> points;
    points.reserve(count);

    std::uniform_real_distribution<double> x_dist(-40.0, 40.0);
    std::uniform_real_distribution<double> y_dist(-40.0, 40.0);
    std::uniform_real_distribution<double> z_dist(-2.5, 3.5);
    const std::array<int, 5> labels = {9, 13, 15, 18, 19};
    std::uniform_int_distribution<int> label_idx(0, static_cast<int>(labels.size() - 1));

    for (std::size_t i = 0; i < count; ++i) {
        Eigen::Vector4d point;
        point << x_dist(rng), y_dist(rng), z_dist(rng), static_cast<double>(labels[label_idx(rng)]);
        points.emplace_back(point);
    }
    return points;
}

Sophus::SE3d SampleTruePose(std::mt19937 &rng) {
    std::uniform_real_distribution<double> t_dist(-1.5, 1.5);
    std::normal_distribution<double> r_dist(0.0, 0.05);

    const Eigen::Vector3d t(t_dist(rng), t_dist(rng), t_dist(rng) * 0.25);
    const Eigen::Vector3d r(r_dist(rng), r_dist(rng), r_dist(rng));
    return Sophus::SE3d(Sophus::SO3d::exp(r), t);
}

Sophus::SE3d SampleInitialGuess(const Sophus::SE3d &true_pose, std::mt19937 &rng) {
    std::normal_distribution<double> t_dist(0.0, 0.15);
    std::normal_distribution<double> r_dist(0.0, 0.02);

    const Eigen::Vector3d dt(t_dist(rng), t_dist(rng), t_dist(rng));
    const Eigen::Vector3d dr(r_dist(rng), r_dist(rng), r_dist(rng));
    const Sophus::SE3d perturb(Sophus::SO3d::exp(dr), dt);
    return true_pose * perturb;
}

void AddNoise(std::vector<Eigen::Vector4d> &points, double stddev, std::mt19937 &rng) {
    if (stddev <= 0.0) return;
    std::normal_distribution<double> n(0.0, stddev);
    for (auto &point : points) {
        point(0) += n(rng);
        point(1) += n(rng);
        point(2) += n(rng);
    }
}

}  // namespace

int main(int argc, char **argv) {
    const BenchmarkConfig cfg = ParseArgs(argc, argv);
    if (cfg.max_iters != 500) {
        std::cerr << "[ benchmark_registration ] --max-iters is currently informational. "
                  << "Registration loop remains fixed at 500 iterations in current implementation." << std::endl;
    }

    std::mt19937 rng(static_cast<std::mt19937::result_type>(cfg.seed));

    const std::vector<Eigen::Vector4d> map_points = GenerateMapPoints(cfg.max_correspondences, rng);
    VoxelHashMap map(/*voxel_size=*/1.0, /*max_distance=*/120.0, /*max_points_per_voxel=*/20);
    map.AddPoints(map_points);

    if (map.Empty()) {
        std::cerr << "benchmark_registration_error=voxel_map_empty" << std::endl;
        return 1;
    }

    const std::size_t total_frames = cfg.frames + cfg.warmup;
    std::vector<double> latency_ms;
    std::vector<double> translation_error_m;
    std::vector<double> rotation_error_rad;
    latency_ms.reserve(cfg.frames);
    translation_error_m.reserve(cfg.frames);
    rotation_error_rad.reserve(cfg.frames);

    for (std::size_t i = 0; i < total_frames; ++i) {
        const Sophus::SE3d true_pose = SampleTruePose(rng);
        const Sophus::SE3d initial_guess = SampleInitialGuess(true_pose, rng);

        // Build a sensor-frame source cloud by applying inverse true pose to map points.
        std::vector<Eigen::Vector4d> frame = Transform4D(true_pose.inverse(), map_points);
        AddNoise(frame, cfg.point_noise_std, rng);

        const auto t0 = std::chrono::steady_clock::now();
        const Sophus::SE3d estimated_pose = RegisterFrameSemanticWithBackend(frame,
                                                                             map,
                                                                             initial_guess,
                                                                             cfg.max_correspondence_distance,
                                                                             cfg.kernel,
                                                                             cfg.backend);
        const auto t1 = std::chrono::steady_clock::now();

        if (i >= cfg.warmup) {
            const double elapsed =
                std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
            latency_ms.emplace_back(elapsed);

            const Sophus::SE3d error_pose = true_pose.inverse() * estimated_pose;
            translation_error_m.emplace_back(error_pose.translation().norm());
            rotation_error_rad.emplace_back(error_pose.so3().log().norm());
        }
    }

    const double total_latency_ms = std::accumulate(latency_ms.begin(), latency_ms.end(), 0.0);
    const double avg_ms = latency_ms.empty() ? 0.0 : total_latency_ms / static_cast<double>(latency_ms.size());
    const double p50_ms = PercentileMs(latency_ms, 0.50);
    const double p95_ms = PercentileMs(latency_ms, 0.95);
    const double fps = avg_ms > 0.0 ? 1000.0 / avg_ms : 0.0;

    const double avg_t_error = translation_error_m.empty()
                                   ? 0.0
                                   : std::accumulate(translation_error_m.begin(), translation_error_m.end(), 0.0) /
                                         static_cast<double>(translation_error_m.size());
    const double avg_r_error = rotation_error_rad.empty()
                                   ? 0.0
                                   : std::accumulate(rotation_error_rad.begin(), rotation_error_rad.end(), 0.0) /
                                         static_cast<double>(rotation_error_rad.size());
    const double max_t_error =
        translation_error_m.empty() ? 0.0 : *std::max_element(translation_error_m.begin(), translation_error_m.end());
    const double max_r_error =
        rotation_error_rad.empty() ? 0.0 : *std::max_element(rotation_error_rad.begin(), rotation_error_rad.end());

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "registration_benchmark_result\n";
    std::cout << "backend=" << RegistrationBackendName(cfg.backend) << "\n";
    std::cout << "timed_frames=" << cfg.frames << "\n";
    std::cout << "warmup_frames=" << cfg.warmup << "\n";
    std::cout << "max_correspondences=" << cfg.max_correspondences << "\n";
    std::cout << "avg_ms=" << avg_ms << "\n";
    std::cout << "p50_ms=" << p50_ms << "\n";
    std::cout << "p95_ms=" << p95_ms << "\n";
    std::cout << "approx_fps=" << fps << "\n";
    std::cout << "avg_translation_error_m=" << avg_t_error << "\n";
    std::cout << "avg_rotation_error_rad=" << avg_r_error << "\n";
    std::cout << "max_translation_error_m=" << max_t_error << "\n";
    std::cout << "max_rotation_error_rad=" << max_r_error << "\n";

    return 0;
}
