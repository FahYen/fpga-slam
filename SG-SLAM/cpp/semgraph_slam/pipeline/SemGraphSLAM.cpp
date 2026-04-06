// This file is covered by the LICENSE file in the root of this project.
// contact: Neng Wang, <neng.wang@hotmail.com>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

#include <tbb/parallel_for.h>

#include "SemGraphSLAM.hpp"

#include "semgraph_slam/frontend/Deskew.hpp"
#include "semgraph_slam/frontend/Preprocessing.hpp"
#include "semgraph_slam/frontend/Registration.hpp"
#include "semgraph_slam/frontend/VoxelHashMap.hpp"
#include "semgraph_slam/loopclosure/LoopClosure.hpp"

namespace graph_slam{

namespace {

struct FrontendStageTimes {
    double deskew_ms = 0.0;
    double kitti_correct_ms = 0.0;
    double preprocess_ms = 0.0;
    double voxelize_ms = 0.0;
    double cluster_ms = 0.0;
    double buildgraph_ms = 0.0;
    double find_match_ms = 0.0;
    double threshold_ms = 0.0;
    double prediction_ms = 0.0;
    double fuse_ms = 0.0;
    double registration_ms = 0.0;
    double relocalization_ms = 0.0;
    double model_update_ms = 0.0;
    double local_map_update_ms = 0.0;
    double local_graph_update_ms = 0.0;
    double push_pose_ms = 0.0;
    double total_ms = 0.0;
};

double ComputePercentileMs(std::vector<double> values, double percentile) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t idx = static_cast<std::size_t>(
        std::floor(percentile * static_cast<double>(values.size() - 1U)));
    return values[idx];
}

class BuildGraphTimingLogger {
public:
    static BuildGraphTimingLogger &Instance() {
        static BuildGraphTimingLogger logger;
        return logger;
    }

    void Record(double elapsed_ms,
                std::size_t cluster_boxes,
                std::size_t graph_nodes,
                std::size_t graph_edges,
                const std::string &dataset) {
        if (!enabled_) return;
        if (!dataset_filter_.empty() && dataset != dataset_filter_) return;

        std::lock_guard<std::mutex> lock(mutex_);
        if (!file_.is_open()) return;

        file_ << frame_idx_++ << ','
              << dataset << ','
              << cluster_boxes << ','
              << graph_nodes << ','
              << graph_edges << ','
              << elapsed_ms << '\n';
        latencies_ms_.emplace_back(elapsed_ms);
    }

    ~BuildGraphTimingLogger() {
        if (!enabled_ || !file_.is_open()) return;
        const std::size_t n = latencies_ms_.size();
        if (n == 0) return;

        const double total_ms = std::accumulate(latencies_ms_.begin(), latencies_ms_.end(), 0.0);
        const double avg_ms = total_ms / static_cast<double>(n);
        const double p50_ms = ComputePercentileMs(latencies_ms_, 0.50);
        const double p95_ms = ComputePercentileMs(latencies_ms_, 0.95);
        const double fps = avg_ms > 0.0 ? 1000.0 / avg_ms : 0.0;

        file_ << "# summary_samples=" << n << '\n';
        file_ << "# summary_avg_ms=" << avg_ms << '\n';
        file_ << "# summary_p50_ms=" << p50_ms << '\n';
        file_ << "# summary_p95_ms=" << p95_ms << '\n';
        file_ << "# summary_approx_fps=" << fps << '\n';
        file_.flush();

        std::cout << "[ BuildGraphTiming ] samples=" << n
                  << " avg_ms=" << avg_ms
                  << " p50_ms=" << p50_ms
                  << " p95_ms=" << p95_ms
                  << " approx_fps=" << fps
                  << std::endl;
    }

private:
    BuildGraphTimingLogger() {
        const char *enabled_env = std::getenv("SGSLAM_BUILDGRAPH_TIMING");
        enabled_ = enabled_env != nullptr && std::string(enabled_env) != "0";
        if (!enabled_) return;

        const char *dataset_env = std::getenv("SGSLAM_BUILDGRAPH_TIMING_DATASET");
        if (dataset_env != nullptr) {
            dataset_filter_ = dataset_env;
        }

        const char *out_env = std::getenv("SGSLAM_BUILDGRAPH_TIMING_OUT");
        output_path_ = (out_env != nullptr) ? std::string(out_env) : std::string("buildgraph_timing.csv");

        file_.open(output_path_, std::ios::out | std::ios::trunc);
        if (!file_.is_open()) {
            enabled_ = false;
            std::cerr << "[ BuildGraphTiming ] failed to open output file: " << output_path_ << std::endl;
            return;
        }

        file_ << "frame_idx,dataset,cluster_boxes,graph_nodes,graph_edges,buildgraph_ms\n";
        file_.flush();

        std::cout << "[ BuildGraphTiming ] enabled, writing to " << output_path_;
        if (!dataset_filter_.empty()) {
            std::cout << " (dataset filter: " << dataset_filter_ << ")";
        }
        std::cout << std::endl;
    }

    bool enabled_ = false;
    std::ofstream file_;
    std::string output_path_;
    std::string dataset_filter_;
    std::size_t frame_idx_ = 0;
    std::vector<double> latencies_ms_;
    std::mutex mutex_;
};

class FrontendPipelineLogger {
public:
    static FrontendPipelineLogger &Instance() {
        static FrontendPipelineLogger logger;
        return logger;
    }

    void Record(const std::string &dataset,
                std::size_t frame_idx,
                std::size_t input_points,
                std::size_t cluster_boxes,
                std::size_t graph_nodes,
                std::size_t graph_edges,
                const FrontendStageTimes &times,
                const char *registration_backend) {
        if (!enabled_) return;
        if (!dataset_filter_.empty() && dataset != dataset_filter_) return;

        std::lock_guard<std::mutex> lock(mutex_);
        if (!file_.is_open()) return;

        const double denom = times.total_ms > 0.0 ? times.total_ms : 1.0;
        file_ << frame_idx << ','
              << dataset << ','
              << input_points << ','
              << cluster_boxes << ','
              << graph_nodes << ','
              << graph_edges << ','
              << times.deskew_ms << ','
              << times.kitti_correct_ms << ','
              << times.preprocess_ms << ','
              << times.voxelize_ms << ','
              << times.cluster_ms << ','
              << times.buildgraph_ms << ','
              << times.find_match_ms << ','
              << times.threshold_ms << ','
              << times.prediction_ms << ','
              << times.fuse_ms << ','
              << times.registration_ms << ','
              << times.relocalization_ms << ','
              << times.model_update_ms << ','
              << times.local_map_update_ms << ','
              << times.local_graph_update_ms << ','
              << times.push_pose_ms << ','
              << times.total_ms << ','
              << (100.0 * times.deskew_ms / denom) << ','
              << (100.0 * times.kitti_correct_ms / denom) << ','
              << (100.0 * times.preprocess_ms / denom) << ','
              << (100.0 * times.voxelize_ms / denom) << ','
              << (100.0 * times.cluster_ms / denom) << ','
              << (100.0 * times.buildgraph_ms / denom) << ','
              << (100.0 * times.find_match_ms / denom) << ','
              << (100.0 * times.threshold_ms / denom) << ','
              << (100.0 * times.prediction_ms / denom) << ','
              << (100.0 * times.fuse_ms / denom) << ','
              << (100.0 * times.registration_ms / denom) << ','
              << (100.0 * times.relocalization_ms / denom) << ','
              << (100.0 * times.model_update_ms / denom) << ','
              << (100.0 * times.local_map_update_ms / denom) << ','
              << (100.0 * times.local_graph_update_ms / denom) << ','
              << (100.0 * times.push_pose_ms / denom) << ','
              << registration_backend << '\n';

        samples_++;
        total_ms_sum_ += times.total_ms;
        deskew_sum_ += times.deskew_ms;
        kitti_correct_sum_ += times.kitti_correct_ms;
        preprocess_sum_ += times.preprocess_ms;
        voxelize_sum_ += times.voxelize_ms;
        cluster_sum_ += times.cluster_ms;
        buildgraph_sum_ += times.buildgraph_ms;
        find_match_sum_ += times.find_match_ms;
        threshold_sum_ += times.threshold_ms;
        prediction_sum_ += times.prediction_ms;
        fuse_sum_ += times.fuse_ms;
        registration_sum_ += times.registration_ms;
        relocalization_sum_ += times.relocalization_ms;
        model_update_sum_ += times.model_update_ms;
        local_map_update_sum_ += times.local_map_update_ms;
        local_graph_update_sum_ += times.local_graph_update_ms;
        push_pose_sum_ += times.push_pose_ms;
    }

    ~FrontendPipelineLogger() {
        if (!enabled_ || !file_.is_open() || samples_ == 0) return;

        const double inv_n = 1.0 / static_cast<double>(samples_);
        const double avg_total = total_ms_sum_ * inv_n;
        const double fps = avg_total > 0.0 ? 1000.0 / avg_total : 0.0;
        const double denom = total_ms_sum_ > 0.0 ? total_ms_sum_ : 1.0;

        file_ << "# summary_samples=" << samples_ << '\n';
        file_ << "# summary_avg_total_ms=" << avg_total << '\n';
        file_ << "# summary_approx_fps=" << fps << '\n';
        file_ << "# summary_buildgraph_pct=" << (100.0 * buildgraph_sum_ / denom) << '\n';
        file_ << "# summary_registration_pct=" << (100.0 * registration_sum_ / denom) << '\n';
        file_ << "# summary_cluster_pct=" << (100.0 * cluster_sum_ / denom) << '\n';
        file_ << "# summary_preprocess_pct=" << (100.0 * preprocess_sum_ / denom) << '\n';
        file_ << "# summary_voxelize_pct=" << (100.0 * voxelize_sum_ / denom) << '\n';
        file_ << "# summary_find_match_pct=" << (100.0 * find_match_sum_ / denom) << '\n';
        file_.flush();

        std::cout << "[ FrontendProfile ] samples=" << samples_
                  << " avg_total_ms=" << avg_total
                  << " approx_fps=" << fps
                  << " buildgraph_pct=" << (100.0 * buildgraph_sum_ / denom)
                  << " registration_pct=" << (100.0 * registration_sum_ / denom)
                  << std::endl;
    }

private:
    FrontendPipelineLogger() {
        const char *enabled_env = std::getenv("SGSLAM_PIPELINE_PROFILE");
        enabled_ = enabled_env != nullptr && std::string(enabled_env) != "0";
        if (!enabled_) return;

        const char *dataset_env = std::getenv("SGSLAM_PIPELINE_PROFILE_DATASET");
        if (dataset_env != nullptr) dataset_filter_ = dataset_env;

        const char *out_env = std::getenv("SGSLAM_PIPELINE_PROFILE_OUT");
        output_path_ = (out_env != nullptr) ? std::string(out_env) : std::string("slam_frontend_profile.csv");

        file_.open(output_path_, std::ios::out | std::ios::trunc);
        if (!file_.is_open()) {
            enabled_ = false;
            std::cerr << "[ FrontendProfile ] failed to open output file: " << output_path_ << std::endl;
            return;
        }

        file_ << "frame_idx,dataset,input_points,cluster_boxes,graph_nodes,graph_edges,"
              << "deskew_ms,kitti_correct_ms,preprocess_ms,voxelize_ms,cluster_ms,buildgraph_ms,"
              << "find_match_ms,threshold_ms,prediction_ms,fuse_ms,registration_ms,relocalization_ms,"
              << "model_update_ms,local_map_update_ms,local_graph_update_ms,push_pose_ms,total_ms,"
              << "deskew_pct,kitti_correct_pct,preprocess_pct,voxelize_pct,cluster_pct,buildgraph_pct,"
              << "find_match_pct,threshold_pct,prediction_pct,fuse_pct,registration_pct,relocalization_pct,"
              << "model_update_pct,local_map_update_pct,local_graph_update_pct,push_pose_pct,registration_backend\n";
        file_.flush();

        std::cout << "[ FrontendProfile ] enabled, writing to " << output_path_;
        if (!dataset_filter_.empty()) {
            std::cout << " (dataset filter: " << dataset_filter_ << ")";
        }
        std::cout << std::endl;
    }

    bool enabled_ = false;
    std::ofstream file_;
    std::string output_path_;
    std::string dataset_filter_;
    std::mutex mutex_;

    std::size_t samples_ = 0;
    double total_ms_sum_ = 0.0;
    double deskew_sum_ = 0.0;
    double kitti_correct_sum_ = 0.0;
    double preprocess_sum_ = 0.0;
    double voxelize_sum_ = 0.0;
    double cluster_sum_ = 0.0;
    double buildgraph_sum_ = 0.0;
    double find_match_sum_ = 0.0;
    double threshold_sum_ = 0.0;
    double prediction_sum_ = 0.0;
    double fuse_sum_ = 0.0;
    double registration_sum_ = 0.0;
    double relocalization_sum_ = 0.0;
    double model_update_sum_ = 0.0;
    double local_map_update_sum_ = 0.0;
    double local_graph_update_sum_ = 0.0;
    double push_pose_sum_ = 0.0;
};

}  // namespace

SemGraphSLAM::V3d_i_pair_graph SemGraphSLAM::mainProcess(const V3d &frame, const std::vector<int> &frame_label, const std::vector<double> &timestamps,std::string dataset){

    // TODO(M1-FPGA): Add per-stage timing (preprocess, voxelize, cluster, BuildGraph,
    // registration, relocalization) and dump CSV so hardware speedups can be normalized
    // against stable software baselines.

    FrontendStageTimes stage_times;
    const RegistrationBackend registration_backend = GetRegistrationBackendFromEnv();
    const char *registration_backend_name = RegistrationBackendName(registration_backend);
    const auto total_t0 = std::chrono::steady_clock::now();

    V3d deskew_frame = frame;

    // Deskew
    const auto deskew_t0 = std::chrono::steady_clock::now();
    if(config_.deskew&&!timestamps.empty()){
        const size_t N = poses().size();
        if(N>2){
            const auto &start_pose = poses_[N - 2];
            const auto &finish_pose = poses_[N - 1];
            deskew_frame = DeSkewScan(frame, timestamps, start_pose, finish_pose);
        }
    }
    const auto deskew_t1 = std::chrono::steady_clock::now();
    stage_times.deskew_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(deskew_t1 - deskew_t0).count();

    const auto kitti_t0 = std::chrono::steady_clock::now();
    if(dataset=="kitti"){
        // Correct KITTI scan
        deskew_frame = CorrectKITTIScan(frame);
    }
    const auto kitti_t1 = std::chrono::steady_clock::now();
    stage_times.kitti_correct_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(kitti_t1 - kitti_t0).count();
        
    // Preprocess frame
    const auto preprocess_t0 = std::chrono::steady_clock::now();
    const auto &cropped_frame_label = PreprocessSemantic(deskew_frame,frame_label, config_.max_range, config_.min_range);
    const auto preprocess_t1 = std::chrono::steady_clock::now();
    stage_times.preprocess_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(preprocess_t1 - preprocess_t0).count();

    // Voxel downsample
    const auto voxelize_t0 = std::chrono::steady_clock::now();
    const auto &[source, frame_downsample, frame_downsample_cluster] = VoxelizeSemantic(cropped_frame_label);
    const auto voxelize_t1 = std::chrono::steady_clock::now();
    stage_times.voxelize_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(voxelize_t1 - voxelize_t0).count();

    // Cluster
    V3d_i foreground_points;
    V3d_i background_points;
    const auto cluster_t0 = std::chrono::steady_clock::now();
    auto cluster_box = ClusterPoints(frame_downsample_cluster.first, frame_downsample_cluster.second,
                                     background_points, foreground_points,
                                     config_.deltaA, config_.deltaR, config_.deltaP);
    const auto cluster_t1 = std::chrono::steady_clock::now();
    stage_times.cluster_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(cluster_t1 - cluster_t0).count();

    // TODO(M1-FPGA): Use this callsite as the end-to-end integration point for
    // accelerated BuildGraph variants (CPU reference vs HLS kernel).
    // Build frame graph
    const auto buildgraph_t0 = std::chrono::steady_clock::now();
    auto graph = BuildGraph(cluster_box,config_.edge_dis_th,config_.subinterval,config_.graph_node_dimension,config_.subgraph_edge_th);
    const auto buildgraph_t1 = std::chrono::steady_clock::now();
    const double buildgraph_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(buildgraph_t1 - buildgraph_t0).count();
    stage_times.buildgraph_ms = buildgraph_ms;
    BuildGraphTimingLogger::Instance().Record(buildgraph_ms,
                                               cluster_box.size(),
                                               graph.node_labels.size(),
                                               graph.edges.size(),
                                               dataset);
    graph.back_points = background_points;
    graph.front_points = foreground_points;

    // Node tracking: find instance node match
    const auto find_match_t0 = std::chrono::steady_clock::now();
    const auto frame2map_match =  local_graph_map_.FindInsMatch(graph);
    const auto find_match_t1 = std::chrono::steady_clock::now();
    stage_times.find_match_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(find_match_t1 - find_match_t0).count();
    
    
    // Get motion prediction and adaptive_threshold
    const auto threshold_t0 = std::chrono::steady_clock::now();
    const double sigma = GetAdaptiveThreshold();
    const auto threshold_t1 = std::chrono::steady_clock::now();
    stage_times.threshold_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(threshold_t1 - threshold_t0).count();

    // Compute initial_guess for ICP
    const auto prediction_t0 = std::chrono::steady_clock::now();
    const auto prediction = GetPredictionModel();
    const auto last_pose = !poses_.empty() ? poses_.back() : Sophus::SE3d();
    const auto initial_guess = last_pose * prediction;
    const auto prediction_t1 = std::chrono::steady_clock::now();
    stage_times.prediction_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(prediction_t1 - prediction_t0).count();

    initial_guess_for_relocalization = initial_guess;

    // Fuse (vector3d point, label) -> vector 4d point4 for subsequent processing
    const auto fuse_t0 = std::chrono::steady_clock::now();
    const auto source_4d = FusePointsAndLabels(source);
    const auto frame_downsample_4d = FusePointsAndLabels(frame_downsample);
    const auto fuse_t1 = std::chrono::steady_clock::now();
    stage_times.fuse_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(fuse_t1 - fuse_t0).count();

    // Registration
    const auto registration_t0 = std::chrono::steady_clock::now();
    Sophus::SE3d new_pose = RegisterFrameSemanticWithBackend(source_4d,         // the current point cloud
                                                             local_map_,        // the local pc map
                                                             initial_guess,     // initial guess
                                                             3.0 * sigma,       // max_correspondence_distance
                                                             sigma / 3.0,       // kernel
                                                             registration_backend);
    const auto registration_t1 = std::chrono::steady_clock::now();
    stage_times.registration_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(registration_t1 - registration_t0).count();

    // The deviation between the initial guess and the final pose
    auto model_deviation = initial_guess.inverse() * new_pose;

    // Relocalization
    const auto relocalization_t0 = std::chrono::steady_clock::now();
    relocalization_corr = std::make_pair(V3d(),V3d());
    if(poses_.size()>2 && config_.relocalization_enable){ // relocalization enable

        // Check model deviation
        if(model_deviation.translation().norm()>config_.model_deviation_trans || model_deviation.so3().log().norm()>config_.model_deviation_rot){

            std::cout<<YELLOW<<"[ Relo. ] relocalization"<<std::endl;
            const auto [initial_guess_graph, estimate_poses_flag] = local_graph_map_.Relocalization(graph, frame2map_match,config_.inlier_rate_th);

            std::cout<<YELLOW<<"[ Relo. ] estimate_poses_flag:"<<estimate_poses_flag<<std::endl;
            if(estimate_poses_flag){ // successful relocalization
                // Regisration with relocalized poses
                new_pose = RegisterFrameSemanticWithBackend(source_4d,          //
                                                            local_map_,          //
                                                            initial_guess_graph, // the relocalized poses
                                                            3.0 * sigma,         //
                                                            sigma / 3.0,
                                                            registration_backend);
                model_deviation = Sophus::SE3d();
                relocalization_corr = local_graph_map_.relo_corr;
            }
            else{
                relocalization_corr = std::make_pair(V3d(),V3d());
            }
        }

    }
    const auto relocalization_t1 = std::chrono::steady_clock::now();
    stage_times.relocalization_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(relocalization_t1 - relocalization_t0).count();
     
    // Update constant motion model
    const auto model_update_t0 = std::chrono::steady_clock::now();
    adaptive_threshold_.UpdateModelDeviation(model_deviation);
    const auto model_update_t1 = std::chrono::steady_clock::now();
    stage_times.model_update_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(model_update_t1 - model_update_t0).count();

    // Update local point cloud map
    const auto local_map_update_t0 = std::chrono::steady_clock::now();
    local_map_.Update(frame_downsample_4d, new_pose);
    const auto local_map_update_t1 = std::chrono::steady_clock::now();
    stage_times.local_map_update_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(local_map_update_t1 - local_map_update_t0).count();

    // Update local graph map
    const auto local_graph_update_t0 = std::chrono::steady_clock::now();
    local_graph_map_.Update(graph, frame2map_match, new_pose);
    const auto local_graph_update_t1 = std::chrono::steady_clock::now();
    stage_times.local_graph_update_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(local_graph_update_t1 - local_graph_update_t0).count();

    // Push new pose to global poses
    const auto push_pose_t0 = std::chrono::steady_clock::now();
    poses_.push_back(new_pose);
    const auto push_pose_t1 = std::chrono::steady_clock::now();
    stage_times.push_pose_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(push_pose_t1 - push_pose_t0).count();

    const auto total_t1 = std::chrono::steady_clock::now();
    stage_times.total_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(total_t1 - total_t0).count();
    FrontendPipelineLogger::Instance().Record(dataset,
                                              poses_.size(),
                                              frame.size(),
                                              cluster_box.size(),
                                              graph.node_labels.size(),
                                              graph.edges.size(),
                                              stage_times,
                                              registration_backend_name);

    return {frame_downsample_cluster, source, graph};
}

// Voxel downsample the semantic frame
SemGraphSLAM::V3d_i_tuple SemGraphSLAM::VoxelizeSemantic(const V3d_i &frame) const {
    const auto voxel_size = config_.voxel_size;
    const auto voxel_size_cluster = config_.voxel_size_cluster;

    const auto frame_downsample_cluster = VoxelDownsampleSemantic(frame, voxel_size_cluster);
    const auto frame_downsample = VoxelDownsampleSemantic(frame, voxel_size * 0.5);
    const auto source = VoxelDownsampleSemantic(frame_downsample, voxel_size * 1.5);
    return {source, frame_downsample,frame_downsample_cluster};
}

//Fusing the points and labels, time comsumption: ~0.02ms
V4d  SemGraphSLAM::FusePointsAndLabels(const V3d_i &frame){
    assert(frame.first.size()==frame.second.size());
    V4d points(frame.first.size());

    tbb::parallel_for(size_t(0),frame.first.size(), [&](size_t i){
        points[i].head<3>() =  frame.first[i];
        points[i](3) = frame.second[i];
    });
    return points;
}


double SemGraphSLAM::GetAdaptiveThreshold() {
    if (!HasMoved()) {
        return config_.initial_threshold;
    }
    return adaptive_threshold_.ComputeThreshold();
    // return config_.initial_threshold;
}

bool SemGraphSLAM::HasMoved() {
    if (poses_.empty()) return false;
    const double motion = (poses_.front().inverse() * poses_.back()).translation().norm();
    return motion > 5.0 * config_.min_motion_th;
}

Sophus::SE3d SemGraphSLAM::GetPredictionModel() const {
    Sophus::SE3d pred = Sophus::SE3d();
    const size_t N = poses_.size();
    if (N < 2) return pred;
    return poses_[N - 2].inverse() * poses_[N - 1];
}

}
