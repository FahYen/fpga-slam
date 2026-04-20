#include "FrameSource.hpp"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "semgraph_slam/pipeline/SemGraphMapping.hpp"
#include "semgraph_slam/pipeline/SemGraphSLAM.hpp"

namespace {

using semgraph_slam_app::FileFrameSource;
using semgraph_slam_app::FrameFetchStatus;
using semgraph_slam_app::FrameLease;
using semgraph_slam_app::FrameSource;
using semgraph_slam_app::IpcFrameSource;

struct Options {
    std::string input_mode = "ipc";
    std::string dataset = "kitti";
    std::string ipc_name = "/rnsg_kitti_00";
    double acquire_timeout_s = 1.0;
    int max_idle_timeouts = 10;
    std::size_t max_frames = 0;

    std::filesystem::path lidar_path;
    std::filesystem::path label_path;
    std::filesystem::path result_path = "kitti_odometry_00.txt";
    std::filesystem::path pgo_result_path = "kitti_slam_00.txt";
    std::filesystem::path graph_map_path = "graph_map_00.txt";
    std::filesystem::path graph_edge_path = "graph_edge_00.txt";
    std::filesystem::path trace_path = "sgslam_consumer_trace.csv";

    bool loop_closure_enable = true;
    bool relocalization_enable = false;
    int frame_acc_pgo = 20;
};

struct ScanData {
    int cloud_id = 0;
    Sophus::SE3d pose;
    graph_slam::Graph graph;
};

class TraceWriter {
public:
    explicit TraceWriter(const std::filesystem::path &path) {
        if (path.empty()) {
            return;
        }
        std::filesystem::create_directories(path.parent_path().empty() ? std::filesystem::path(".")
                                                                       : path.parent_path());
        out_.open(path, std::ios::out | std::ios::trunc);
        if (!out_.is_open()) {
            throw std::runtime_error("Failed to open trace file: " + path.string());
        }
        out_ << "status,source,frame_id,consumed_index,skipped_before,num_points,acquire_wait_ms,"
                "frontend_ms,capture_to_consume_ms,publish_to_consume_ms\n";
    }

    void WriteRow(const std::string &status,
                  const FrameLease &lease,
                  double frontend_ms,
                  double capture_to_consume_ms,
                  double publish_to_consume_ms) {
        if (!out_.is_open()) {
            return;
        }
        out_ << status << ','
             << '"' << lease.source_name << '"' << ','
             << lease.frame_id << ','
             << lease.consumed_index << ','
             << lease.skipped_before << ','
             << lease.view.num_points << ','
             << std::fixed << std::setprecision(3)
             << lease.acquire_wait_ms << ','
             << frontend_ms << ','
             << capture_to_consume_ms << ','
             << publish_to_consume_ms << '\n';
        out_.flush();
    }

private:
    std::ofstream out_;
};

void EnsureParentDirectory(const std::filesystem::path &path) {
    const auto parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

bool ParseBool(const std::string &value) {
    if (value == "1" || value == "true" || value == "TRUE") {
        return true;
    }
    if (value == "0" || value == "false" || value == "FALSE") {
        return false;
    }
    throw std::runtime_error("Expected boolean value, got: " + value);
}

std::string RequireValue(int argc, char **argv, int &index) {
    if (index + 1 >= argc) {
        throw std::runtime_error(std::string("Missing value for ") + argv[index]);
    }
    ++index;
    return argv[index];
}

Options ParseArgs(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input-mode") {
            options.input_mode = RequireValue(argc, argv, i);
        } else if (arg == "--dataset") {
            options.dataset = RequireValue(argc, argv, i);
        } else if (arg == "--ipc-name") {
            options.ipc_name = RequireValue(argc, argv, i);
        } else if (arg == "--acquire-timeout-s") {
            options.acquire_timeout_s = std::stod(RequireValue(argc, argv, i));
        } else if (arg == "--max-idle-timeouts") {
            options.max_idle_timeouts = std::stoi(RequireValue(argc, argv, i));
        } else if (arg == "--max-frames") {
            options.max_frames = static_cast<std::size_t>(std::stoull(RequireValue(argc, argv, i)));
        } else if (arg == "--lidar-path") {
            options.lidar_path = RequireValue(argc, argv, i);
        } else if (arg == "--label-path") {
            options.label_path = RequireValue(argc, argv, i);
        } else if (arg == "--result-path") {
            options.result_path = RequireValue(argc, argv, i);
        } else if (arg == "--pgo-result-path") {
            options.pgo_result_path = RequireValue(argc, argv, i);
        } else if (arg == "--graph-map-path") {
            options.graph_map_path = RequireValue(argc, argv, i);
        } else if (arg == "--graph-edge-path") {
            options.graph_edge_path = RequireValue(argc, argv, i);
        } else if (arg == "--trace-path") {
            options.trace_path = RequireValue(argc, argv, i);
        } else if (arg == "--loop-closure-enable") {
            options.loop_closure_enable = ParseBool(RequireValue(argc, argv, i));
        } else if (arg == "--relocalization-enable") {
            options.relocalization_enable = ParseBool(RequireValue(argc, argv, i));
        } else if (arg == "--frame-acc-pgo") {
            options.frame_acc_pgo = std::stoi(RequireValue(argc, argv, i));
        } else if (arg == "--help") {
            std::cout
                << "Usage: sgslam_ipc_runner [options]\n"
                << "  --input-mode ipc|files\n"
                << "  --ipc-name /rnsg_kitti_00\n"
                << "  --lidar-path <dir> --label-path <dir>    (for file mode)\n"
                << "  --result-path <file>\n"
                << "  --pgo-result-path <file>\n"
                << "  --graph-map-path <file>\n"
                << "  --graph-edge-path <file>\n"
                << "  --trace-path <file>\n"
                << "  --acquire-timeout-s <seconds>\n"
                << "  --max-idle-timeouts <count>\n"
                << "  --max-frames <count>\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (options.input_mode != "ipc" && options.input_mode != "files") {
        throw std::runtime_error("--input-mode must be 'ipc' or 'files'");
    }
    if (options.input_mode == "files" &&
        (options.lidar_path.empty() || options.label_path.empty())) {
        throw std::runtime_error("File mode requires --lidar-path and --label-path");
    }
    return options;
}

graph_slam::SemGraphSLAMConfig DefaultOdomConfig() {
    graph_slam::SemGraphSLAMConfig cfg;
    cfg.max_range = 100.0;
    cfg.min_range = 5.0;
    cfg.deskew = false;
    cfg.relocalization_enable = false;
    cfg.voxel_size = 1.0;
    cfg.max_points_per_voxel = 20;
    cfg.initial_threshold = 2.0;
    cfg.min_motion_th = 0.1;
    cfg.deltaA = 2.0;
    cfg.deltaR = 0.5;
    cfg.deltaP = 2.0;
    cfg.edge_dis_th = 60.0;
    cfg.subgraph_edge_th = 20.0;
    cfg.subinterval = 30;
    cfg.graph_node_dimension = 30;
    cfg.nearest_neighbor_vehicle_disth = 2.0;
    cfg.nearest_neighbor_pole_disth = 2.0;
    cfg.max_local_graph_map_range = 100.0;
    cfg.model_deviation_trans = 0.12;
    cfg.model_deviation_rot = 0.01;
    cfg.inlier_rate_th = 0.43;
    return cfg;
}

graph_slam::SemGraphMappingConfig DefaultMappingConfig() {
    graph_slam::SemGraphMappingConfig cfg;
    cfg.global_des_dim = 231;
    cfg.loop_candidate = 5;
    cfg.edge_dis_th = 60.0;
    cfg.subinterval = 30;
    cfg.keyframe_interval = 5;
    cfg.search_results_num = 5;
    cfg.max_distance_for_loop = 0.1F;
    cfg.graph_sim_th = 0.5;
    cfg.back_sim_th = 0.58;
    cfg.map_voxel_size_loop = 1.0;
    cfg.frame_acc_pgo = 20;
    cfg.loop_closure_enable = true;
    return cfg;
}

std::vector<double> BuildTimestamps(const std::string &dataset, std::size_t point_count) {
    if (dataset != "mulran") {
        return {};
    }
    constexpr int kRows = 64;
    constexpr int kCols = 1024;
    if (point_count == static_cast<std::size_t>(kRows * kCols)) {
        std::vector<double> times(point_count);
        for (int i = 0; i < kRows * kCols; ++i) {
            times[static_cast<std::size_t>(i)] =
                static_cast<double>(std::floor(i / static_cast<double>(kRows))) /
                static_cast<double>(kCols);
        }
        return times;
    }
    return std::vector<double>(point_count, 1.0);
}

std::uint64_t MonotonicNowNs() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

void WritePoseLine(std::ofstream &out, const Sophus::SE3d &pose) {
    const Eigen::Matrix4d pose_mat = pose.matrix();
    out << pose_mat(0,0) << " " << pose_mat(0,1) << " " << pose_mat(0,2) << " " << pose_mat(0,3) << " "
        << pose_mat(1,0) << " " << pose_mat(1,1) << " " << pose_mat(1,2) << " " << pose_mat(1,3) << " "
        << pose_mat(2,0) << " " << pose_mat(2,1) << " " << pose_mat(2,2) << " " << pose_mat(2,3) << '\n';
}

void WriteGraphArtifacts(const graph_slam::SemGraphMapping &mapping,
                         const std::filesystem::path &graph_map_path,
                         const std::filesystem::path &graph_edge_path) {
    if (!graph_map_path.empty()) {
        EnsureParentDirectory(graph_map_path);
        std::ofstream graph_map_out(graph_map_path, std::ios::out | std::ios::trunc);
        for (const auto &point : mapping.global_graph_map) {
            graph_map_out << point[0] << " " << point[1] << " " << point[2] << " " << point[3] << '\n';
        }
    }
    if (!graph_edge_path.empty()) {
        EnsureParentDirectory(graph_edge_path);
        std::ofstream graph_edge_out(graph_edge_path, std::ios::out | std::ios::trunc);
        for (const auto &edge : mapping.global_graph_edge) {
            graph_edge_out << edge.first << " " << edge.second << '\n';
        }
    }
}

void RecordThreadError(std::exception_ptr eptr,
                       std::exception_ptr &shared_error,
                       std::mutex &error_mutex) {
    std::lock_guard<std::mutex> lock(error_mutex);
    if (!shared_error) {
        shared_error = eptr;
    }
}

}  // namespace

int main(int argc, char **argv) {
    try {
        const Options options = ParseArgs(argc, argv);
        graph_slam::SemGraphSLAMConfig odom_config = DefaultOdomConfig();
        graph_slam::SemGraphMappingConfig mapping_config = DefaultMappingConfig();
        odom_config.relocalization_enable = options.relocalization_enable;
        mapping_config.loop_closure_enable = options.loop_closure_enable;
        mapping_config.frame_acc_pgo = options.frame_acc_pgo;

        std::unique_ptr<FrameSource> source;
        if (options.input_mode == "ipc") {
            source = std::make_unique<IpcFrameSource>(options.ipc_name, options.acquire_timeout_s);
        } else {
            source = std::make_unique<FileFrameSource>(options.lidar_path, options.label_path, options.max_frames);
        }

        EnsureParentDirectory(options.result_path);
        EnsureParentDirectory(options.pgo_result_path);
        std::ofstream odom_out(options.result_path, std::ios::out | std::ios::trunc);
        if (!odom_out.is_open()) {
            throw std::runtime_error("Failed to open odometry output: " + options.result_path.string());
        }
        odom_out.setf(std::ios::fixed, std::ios::floatfield);
        odom_out.precision(16);

        TraceWriter trace_writer(options.trace_path);

        graph_slam::SemGraphSLAM slam(odom_config);
        graph_slam::SemGraphMapping mapping(mapping_config);

        std::queue<ScanData> scan_queue;
        std::mutex scan_mutex;
        std::condition_variable scan_cv;
        bool frontend_done = false;

        std::exception_ptr shared_error;
        std::mutex error_mutex;

        std::thread frontend_thread([&]() {
            try {
                std::size_t processed_frames = 0;
                int idle_timeouts = 0;
                while (options.max_frames == 0 || processed_frames < options.max_frames) {
                    FrameLease lease;
                    const FrameFetchStatus status = source->Next(lease);
                    if (status == FrameFetchStatus::kTimeout) {
                        trace_writer.WriteRow("timeout", lease, 0.0, -1.0, -1.0);
                        ++idle_timeouts;
                        if (options.max_idle_timeouts >= 0 &&
                            idle_timeouts >= options.max_idle_timeouts) {
                            std::cout << "[ SG-SLAM ] stopping after "
                                      << idle_timeouts << " consecutive IPC timeouts" << std::endl;
                            break;
                        }
                        continue;
                    }
                    if (status == FrameFetchStatus::kEnd) {
                        break;
                    }

                    idle_timeouts = 0;
                    const auto frontend_t0 = std::chrono::steady_clock::now();
                    const auto timestamps = BuildTimestamps(options.dataset, lease.view.num_points);
                    const auto &[frame, keypoints, graph] =
                        slam.mainProcess(lease.view, timestamps, options.dataset);
                    (void)frame;
                    (void)keypoints;
                    const Sophus::SE3d pose = slam.poses().back();
                    WritePoseLine(odom_out, pose);
                    odom_out.flush();

                    {
                        std::lock_guard<std::mutex> lock(scan_mutex);
                        scan_queue.push(ScanData{static_cast<int>(lease.consumed_index), pose, graph});
                    }
                    scan_cv.notify_one();

                    const auto frontend_t1 = std::chrono::steady_clock::now();
                    const double frontend_ms =
                        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(frontend_t1 - frontend_t0).count();
                    const std::uint64_t now_ns = MonotonicNowNs();
                    const double capture_to_consume_ms =
                        (lease.capture_ns == 0) ? -1.0
                                                : static_cast<double>(now_ns - lease.capture_ns) / 1e6;
                    const double publish_to_consume_ms =
                        (lease.publish_ns == 0) ? -1.0
                                                : static_cast<double>(now_ns - lease.publish_ns) / 1e6;
                    trace_writer.WriteRow("consumed",
                                          lease,
                                          frontend_ms,
                                          capture_to_consume_ms,
                                          publish_to_consume_ms);
                    std::cout << "[ SG-SLAM ] frame_id=" << lease.frame_id
                              << " consumed_index=" << lease.consumed_index
                              << " skipped_before=" << lease.skipped_before
                              << " frontend_ms=" << std::fixed << std::setprecision(2)
                              << frontend_ms << std::endl;
                    ++processed_frames;
                }
            } catch (...) {
                RecordThreadError(std::current_exception(), shared_error, error_mutex);
            }

            {
                std::lock_guard<std::mutex> lock(scan_mutex);
                frontend_done = true;
            }
            scan_cv.notify_all();
        });

        std::thread mapping_thread([&]() {
            try {
                int run_isam_count = 0;
                int run_map_update_count = 0;
                int recent_update_idx = 0;
                std::vector<Sophus::SE3d> poses_vec;
                std::vector<Sophus::SE3d> poses_pgo_vec;

                gtsam::Values initial;
                gtsam::NonlinearFactorGraph gtSAMgraph;
                gtsam::ISAM2Params parameters;
                parameters.relinearizeThreshold = 0.01;
                parameters.relinearizeSkip = 1;
                gtsam::ISAM2 isam(parameters);
                gtsam::Values results_isam;

                gtsam::Vector vector6(6);
                vector6 << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
                const auto odometry_noise = gtsam::noiseModel::Diagonal::Variances(vector6);

                gtsam::Vector robust_noise_vector6(6);
                robust_noise_vector6 << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1;
                const auto robust_loop_noise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Cauchy::Create(1),
                    gtsam::noiseModel::Diagonal::Variances(robust_noise_vector6));

                auto write_pgo_results = [&]() {
                    if (options.pgo_result_path.empty() || poses_pgo_vec.empty()) {
                        return;
                    }
                    EnsureParentDirectory(options.pgo_result_path);
                    std::ofstream pgo_out(options.pgo_result_path, std::ios::out | std::ios::trunc);
                    pgo_out.setf(std::ios::fixed, std::ios::floatfield);
                    pgo_out.precision(16);
                    for (const auto &pose : poses_pgo_vec) {
                        WritePoseLine(pgo_out, pose);
                    }
                };

                auto flush_isam = [&](bool final_flush) {
                    if (run_isam_count == 0 && !final_flush) {
                        return;
                    }
                    run_isam_count = 0;
                    isam.update(gtSAMgraph, initial);
                    isam.update();
                    gtSAMgraph.resize(0);
                    initial.clear();
                    results_isam = isam.calculateEstimate();
                    if (results_isam.size() > 0) {
                        recent_update_idx = static_cast<int>(results_isam.size()) - 1;
                        poses_pgo_vec.resize(results_isam.size());
                        for (int node_idx = 0; node_idx < static_cast<int>(results_isam.size()); ++node_idx) {
                            const auto pose_pgo_scan = results_isam.at(node_idx).cast<gtsam::Pose3>();
                            poses_pgo_vec[static_cast<std::size_t>(node_idx)] = Sophus::SE3d(
                                pose_pgo_scan.rotation().matrix(), pose_pgo_scan.translation());
                        }
                    }
                    if (final_flush) {
                        write_pgo_results();
                    }
                };

                auto flush_map = [&](bool final_flush) {
                    if (poses_pgo_vec.empty()) {
                        return;
                    }
                    run_map_update_count = 0;
                    mapping.UpdateMapping(poses_pgo_vec, recent_update_idx);
                    if (final_flush) {
                        WriteGraphArtifacts(mapping, options.graph_map_path, options.graph_edge_path);
                    }
                };

                bool final_outputs_written = false;
                auto finalize_mapping_outputs = [&]() {
                    if (final_outputs_written) {
                        return;
                    }
                    flush_isam(true);
                    flush_map(true);
                    final_outputs_written = true;
                };

                while (true) {
                    ScanData frame_data;
                    bool finishing = false;
                    bool should_finalize = false;
                    {
                        std::unique_lock<std::mutex> lock(scan_mutex);
                        scan_cv.wait(lock, [&]() { return frontend_done || !scan_queue.empty(); });
                        if (scan_queue.empty()) {
                            should_finalize = frontend_done;
                        } else {
                            frame_data = std::move(scan_queue.front());
                            scan_queue.pop();
                            finishing = frontend_done && scan_queue.empty();
                        }
                    }
                    if (should_finalize) {
                        finalize_mapping_outputs();
                        break;
                    }

                    run_isam_count++;
                    run_map_update_count++;
                    poses_vec.emplace_back(frame_data.pose);
                    poses_pgo_vec.emplace_back(frame_data.pose);

                    mapping.mainProcess(frame_data.cloud_id, frame_data.graph);

                    if (frame_data.cloud_id < static_cast<int>(mapping.is_keyframe_vec.size()) &&
                        mapping.is_keyframe_vec[static_cast<std::size_t>(frame_data.cloud_id)] &&
                        mapping.loop_flag) {
                        const auto &loop_pair = mapping.loop_pair_vec.back();
                        const auto &loop_trans = mapping.loop_trans_vec.back();
                        gtsam::Point3 ttem(loop_trans.block<3,1>(0,3));
                        gtsam::Rot3 rtem(loop_trans.block<3,3>(0,0));
                        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(
                            loop_pair.second, loop_pair.first, gtsam::Pose3(rtem, ttem), robust_loop_noise));
                    }

                    const Eigen::Matrix3d poses_r = frame_data.pose.rotationMatrix();
                    const Eigen::Vector3d poses_t = frame_data.pose.translation();
                    if (frame_data.cloud_id == 0) {
                        gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(
                            0, gtsam::Pose3(gtsam::Rot3(poses_r), gtsam::Point3(poses_t)), odometry_noise));
                        initial.insert(0, gtsam::Pose3(gtsam::Rot3(poses_r), gtsam::Point3(poses_t)));
                    } else {
                        Eigen::Vector3d t_ab = poses_vec[static_cast<std::size_t>(frame_data.cloud_id - 1)].translation();
                        Eigen::Matrix3d r_ab = poses_vec[static_cast<std::size_t>(frame_data.cloud_id - 1)].rotationMatrix();
                        t_ab = r_ab.transpose() * (poses_t - t_ab);
                        r_ab = r_ab.transpose() * poses_r;
                        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(
                            frame_data.cloud_id - 1,
                            frame_data.cloud_id,
                            gtsam::Pose3(gtsam::Rot3(r_ab), gtsam::Point3(t_ab)),
                            odometry_noise));
                        initial.insert(frame_data.cloud_id,
                                       gtsam::Pose3(gtsam::Rot3(poses_r), gtsam::Point3(poses_t)));
                    }

                    if (run_isam_count >= mapping_config.frame_acc_pgo) {
                        flush_isam(false);
                    }
                    if (run_map_update_count >= mapping_config.frame_acc_pgo) {
                        flush_map(false);
                    }
                    if (finishing) {
                        finalize_mapping_outputs();
                        break;
                    }
                }
            } catch (...) {
                RecordThreadError(std::current_exception(), shared_error, error_mutex);
            }
        });

        frontend_thread.join();
        mapping_thread.join();

        if (shared_error) {
            std::rethrow_exception(shared_error);
        }

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "sgslam_ipc_runner error: " << e.what() << std::endl;
        return 1;
    }
}
