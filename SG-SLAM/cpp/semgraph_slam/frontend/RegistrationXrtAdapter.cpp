#include "RegistrationXrtAdapter.hpp"

#include "semgraph_slam/hls/registration_kernel_draft.hpp"

#include <algorithm>
#include <fstream>
#include <mutex>
#include <sstream>
#include <vector>

#if defined(SGSLAM_ENABLE_XRT)
#include <CL/cl.h>
#endif

namespace graph_slam {

namespace {

#if defined(SGSLAM_ENABLE_XRT)
std::string MakeError(const std::string &msg, int status) {
    std::ostringstream os;
    os << msg << " (status=" << status << ")";
    return os.str();
}
#endif

void SetError(std::string *error_message, const std::string &msg) {
    if (error_message != nullptr) *error_message = msg;
}

#if defined(SGSLAM_ENABLE_XRT)

struct GlobalMutex {
    static std::mutex &Instance() {
        static std::mutex mutex;
        return mutex;
    }
};

#endif

}  // namespace

#if defined(SGSLAM_ENABLE_XRT)
struct RegistrationXrtAdapter::XrtHandles {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
};
#endif

RegistrationXrtAdapter &RegistrationXrtAdapter::Instance() {
    static RegistrationXrtAdapter instance;
    return instance;
}

RegistrationXrtAdapter::RegistrationXrtAdapter() = default;

RegistrationXrtAdapter::~RegistrationXrtAdapter() { Cleanup(); }

bool RegistrationXrtAdapter::IsCompiledWithXrt() const {
#if defined(SGSLAM_ENABLE_XRT)
    return true;
#else
    return false;
#endif
}

void RegistrationXrtAdapter::Cleanup() {
#if defined(SGSLAM_ENABLE_XRT)
    if (handles_ != nullptr) {
        if (handles_->kernel != nullptr) clReleaseKernel(handles_->kernel);
        if (handles_->program != nullptr) clReleaseProgram(handles_->program);
        if (handles_->queue != nullptr) clReleaseCommandQueue(handles_->queue);
        if (handles_->context != nullptr) clReleaseContext(handles_->context);
        delete handles_;
        handles_ = nullptr;
    }
#endif
    initialized_ = false;
    xclbin_path_.clear();
}

bool RegistrationXrtAdapter::Configure(const std::string &xclbin_path, std::string *error_message) {
#if !defined(SGSLAM_ENABLE_XRT)
    (void)xclbin_path;
    SetError(error_message,
             "XRT support is not compiled in. Rebuild with -DENABLE_XRT_REGISTRATION=ON.");
    return false;
#else
    std::lock_guard<std::mutex> lock(GlobalMutex::Instance());

    if (initialized_ && xclbin_path == xclbin_path_) return true;

    Cleanup();

    std::ifstream xclbin_file(xclbin_path, std::ios::binary | std::ios::ate);
    if (!xclbin_file.is_open()) {
        SetError(error_message, "Failed to open xclbin: " + xclbin_path);
        return false;
    }

    const auto file_size = xclbin_file.tellg();
    if (file_size <= 0) {
        SetError(error_message, "xclbin file is empty: " + xclbin_path);
        return false;
    }
    xclbin_file.seekg(0, std::ios::beg);

    std::vector<unsigned char> binary(static_cast<std::size_t>(file_size));
    xclbin_file.read(reinterpret_cast<char *>(binary.data()), file_size);
    if (!xclbin_file) {
        SetError(error_message, "Failed to read xclbin file: " + xclbin_path);
        return false;
    }

    auto *new_handles = new XrtHandles();

    cl_int status = CL_SUCCESS;

    cl_uint platform_count = 0;
    status = clGetPlatformIDs(0, nullptr, &platform_count);
    if (status != CL_SUCCESS || platform_count == 0) {
        delete new_handles;
        SetError(error_message, MakeError("No OpenCL platform found", status));
        return false;
    }

    std::vector<cl_platform_id> platforms(platform_count);
    status = clGetPlatformIDs(platform_count, platforms.data(), nullptr);
    if (status != CL_SUCCESS) {
        delete new_handles;
        SetError(error_message, MakeError("Failed to enumerate OpenCL platforms", status));
        return false;
    }

    bool device_found = false;
    for (const auto &platform : platforms) {
        cl_uint device_count = 0;
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 0, nullptr, &device_count);
        if (status != CL_SUCCESS || device_count == 0) continue;

        std::vector<cl_device_id> devices(device_count);
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, device_count, devices.data(), nullptr);
        if (status != CL_SUCCESS || devices.empty()) continue;

        new_handles->platform = platform;
        new_handles->device = devices.front();
        device_found = true;
        break;
    }

    if (!device_found) {
        delete new_handles;
        SetError(error_message, "No accelerator OpenCL device found (is F1/XRT ready?).");
        return false;
    }

    new_handles->context = clCreateContext(nullptr, 1, &new_handles->device, nullptr, nullptr, &status);
    if (status != CL_SUCCESS || new_handles->context == nullptr) {
        delete new_handles;
        SetError(error_message, MakeError("Failed to create OpenCL context", status));
        return false;
    }

    new_handles->queue = clCreateCommandQueue(new_handles->context, new_handles->device, 0, &status);
    if (status != CL_SUCCESS || new_handles->queue == nullptr) {
        Cleanup();
        delete new_handles;
        SetError(error_message, MakeError("Failed to create OpenCL command queue", status));
        return false;
    }

    const unsigned char *binary_ptr = binary.data();
    const size_t binary_size = binary.size();
    cl_int binary_status = CL_SUCCESS;
    new_handles->program = clCreateProgramWithBinary(new_handles->context,
                                                      1,
                                                      &new_handles->device,
                                                      &binary_size,
                                                      &binary_ptr,
                                                      &binary_status,
                                                      &status);
    if (status != CL_SUCCESS || binary_status != CL_SUCCESS || new_handles->program == nullptr) {
        if (new_handles->queue != nullptr) clReleaseCommandQueue(new_handles->queue);
        if (new_handles->context != nullptr) clReleaseContext(new_handles->context);
        delete new_handles;
        SetError(error_message, MakeError("Failed to create program with binary", status));
        return false;
    }

    status = clBuildProgram(new_handles->program, 1, &new_handles->device, nullptr, nullptr, nullptr);
    if (status != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(new_handles->program,
                              new_handles->device,
                              CL_PROGRAM_BUILD_LOG,
                              0,
                              nullptr,
                              &log_size);
        std::string build_log(log_size, '\0');
        clGetProgramBuildInfo(new_handles->program,
                              new_handles->device,
                              CL_PROGRAM_BUILD_LOG,
                              log_size,
                              build_log.data(),
                              nullptr);

        if (new_handles->program != nullptr) clReleaseProgram(new_handles->program);
        if (new_handles->queue != nullptr) clReleaseCommandQueue(new_handles->queue);
        if (new_handles->context != nullptr) clReleaseContext(new_handles->context);
        delete new_handles;
        SetError(error_message, "Failed to build xclbin program: " + build_log);
        return false;
    }

    new_handles->kernel = clCreateKernel(new_handles->program, "registration_accumulate_kernel", &status);
    if (status != CL_SUCCESS || new_handles->kernel == nullptr) {
        if (new_handles->program != nullptr) clReleaseProgram(new_handles->program);
        if (new_handles->queue != nullptr) clReleaseCommandQueue(new_handles->queue);
        if (new_handles->context != nullptr) clReleaseContext(new_handles->context);
        delete new_handles;
        SetError(error_message,
                 MakeError("Failed to create kernel registration_accumulate_kernel", status));
        return false;
    }

    handles_ = new_handles;
    initialized_ = true;
    xclbin_path_ = xclbin_path;
    return true;
#endif
}

bool RegistrationXrtAdapter::Accumulate(const float *src_xyz,
                                        const float *tgt_xyz,
                                        const int *labels,
                                        int correspondence_count,
                                        float kernel,
                                        std::array<double, 36> &jtj_out,
                                        std::array<double, 6> &jtr_out,
                                        int &used_count,
                                        int &dropped_count,
                                        std::string *error_message) {
#if !defined(SGSLAM_ENABLE_XRT)
    (void)src_xyz;
    (void)tgt_xyz;
    (void)labels;
    (void)correspondence_count;
    (void)kernel;
    (void)jtj_out;
    (void)jtr_out;
    used_count = 0;
    dropped_count = 0;
    SetError(error_message,
             "XRT support is not compiled in. Rebuild with -DENABLE_XRT_REGISTRATION=ON.");
    return false;
#else
    std::lock_guard<std::mutex> lock(GlobalMutex::Instance());

    if (!initialized_ || handles_ == nullptr) {
        used_count = 0;
        dropped_count = 0;
        SetError(error_message, "XRT adapter is not configured. Set SGSLAM_REG_XCLBIN first.");
        return false;
    }

    if (src_xyz == nullptr || tgt_xyz == nullptr || labels == nullptr) {
        used_count = 0;
        dropped_count = 0;
        SetError(error_message, "Null pointer passed to XRT accumulate inputs.");
        return false;
    }

    const int requested = correspondence_count < 0 ? 0 : correspondence_count;
    const int bounded = std::min(requested, graph_slam_hls_draft::MAX_REG_CORRESPONDENCES);

    std::fill(jtj_out.begin(), jtj_out.end(), 0.0);
    std::fill(jtr_out.begin(), jtr_out.end(), 0.0);
    used_count = bounded;
    dropped_count = requested - bounded;

    if (bounded == 0) {
        return true;
    }

    const size_t xyz_bytes = static_cast<size_t>(bounded) * 3U * sizeof(float);
    const size_t label_bytes = static_cast<size_t>(bounded) * sizeof(int);

    const size_t jtj_bytes = 36U * sizeof(double);
    const size_t jtr_bytes = 6U * sizeof(double);
    const size_t scalar_bytes = sizeof(int);

    cl_int status = CL_SUCCESS;
    cl_mem src_buf = clCreateBuffer(handles_->context, CL_MEM_READ_ONLY, xyz_bytes, nullptr, &status);
    if (status != CL_SUCCESS || src_buf == nullptr) {
        SetError(error_message, MakeError("Failed to create src buffer", status));
        return false;
    }

    cl_mem tgt_buf = clCreateBuffer(handles_->context, CL_MEM_READ_ONLY, xyz_bytes, nullptr, &status);
    if (status != CL_SUCCESS || tgt_buf == nullptr) {
        clReleaseMemObject(src_buf);
        SetError(error_message, MakeError("Failed to create tgt buffer", status));
        return false;
    }

    cl_mem labels_buf = clCreateBuffer(handles_->context, CL_MEM_READ_ONLY, label_bytes, nullptr, &status);
    if (status != CL_SUCCESS || labels_buf == nullptr) {
        clReleaseMemObject(tgt_buf);
        clReleaseMemObject(src_buf);
        SetError(error_message, MakeError("Failed to create labels buffer", status));
        return false;
    }

    cl_mem jtj_buf = clCreateBuffer(handles_->context, CL_MEM_WRITE_ONLY, jtj_bytes, nullptr, &status);
    cl_mem jtr_buf = clCreateBuffer(handles_->context, CL_MEM_WRITE_ONLY, jtr_bytes, nullptr, &status);
    cl_mem used_buf = clCreateBuffer(handles_->context, CL_MEM_WRITE_ONLY, scalar_bytes, nullptr, &status);
    cl_mem dropped_buf = clCreateBuffer(handles_->context, CL_MEM_WRITE_ONLY, scalar_bytes, nullptr, &status);

    if (jtj_buf == nullptr || jtr_buf == nullptr || used_buf == nullptr || dropped_buf == nullptr) {
        if (dropped_buf != nullptr) clReleaseMemObject(dropped_buf);
        if (used_buf != nullptr) clReleaseMemObject(used_buf);
        if (jtr_buf != nullptr) clReleaseMemObject(jtr_buf);
        if (jtj_buf != nullptr) clReleaseMemObject(jtj_buf);
        clReleaseMemObject(labels_buf);
        clReleaseMemObject(tgt_buf);
        clReleaseMemObject(src_buf);
        SetError(error_message, "Failed to create one or more output buffers.");
        return false;
    }

    status = clEnqueueWriteBuffer(handles_->queue, src_buf, CL_TRUE, 0, xyz_bytes, src_xyz, 0, nullptr, nullptr);
    status |= clEnqueueWriteBuffer(handles_->queue, tgt_buf, CL_TRUE, 0, xyz_bytes, tgt_xyz, 0, nullptr, nullptr);
    status |= clEnqueueWriteBuffer(
        handles_->queue, labels_buf, CL_TRUE, 0, label_bytes, labels, 0, nullptr, nullptr);

    if (status != CL_SUCCESS) {
        SetError(error_message, MakeError("Failed to enqueue input buffer writes", status));
        clReleaseMemObject(dropped_buf);
        clReleaseMemObject(used_buf);
        clReleaseMemObject(jtr_buf);
        clReleaseMemObject(jtj_buf);
        clReleaseMemObject(labels_buf);
        clReleaseMemObject(tgt_buf);
        clReleaseMemObject(src_buf);
        return false;
    }

    status = clSetKernelArg(handles_->kernel, 0, sizeof(cl_mem), &src_buf);
    status |= clSetKernelArg(handles_->kernel, 1, sizeof(cl_mem), &tgt_buf);
    status |= clSetKernelArg(handles_->kernel, 2, sizeof(cl_mem), &labels_buf);
    status |= clSetKernelArg(handles_->kernel, 3, sizeof(int), &requested);
    status |= clSetKernelArg(handles_->kernel, 4, sizeof(float), &kernel);
    status |= clSetKernelArg(handles_->kernel, 5, sizeof(cl_mem), &jtj_buf);
    status |= clSetKernelArg(handles_->kernel, 6, sizeof(cl_mem), &jtr_buf);
    status |= clSetKernelArg(handles_->kernel, 7, sizeof(cl_mem), &used_buf);
    status |= clSetKernelArg(handles_->kernel, 8, sizeof(cl_mem), &dropped_buf);

    if (status != CL_SUCCESS) {
        SetError(error_message, MakeError("Failed to set kernel arguments", status));
        clReleaseMemObject(dropped_buf);
        clReleaseMemObject(used_buf);
        clReleaseMemObject(jtr_buf);
        clReleaseMemObject(jtj_buf);
        clReleaseMemObject(labels_buf);
        clReleaseMemObject(tgt_buf);
        clReleaseMemObject(src_buf);
        return false;
    }

    status = clEnqueueTask(handles_->queue, handles_->kernel, 0, nullptr, nullptr);
    if (status != CL_SUCCESS) {
        SetError(error_message, MakeError("Failed to enqueue kernel task", status));
        clReleaseMemObject(dropped_buf);
        clReleaseMemObject(used_buf);
        clReleaseMemObject(jtr_buf);
        clReleaseMemObject(jtj_buf);
        clReleaseMemObject(labels_buf);
        clReleaseMemObject(tgt_buf);
        clReleaseMemObject(src_buf);
        return false;
    }

    status = clFinish(handles_->queue);
    if (status != CL_SUCCESS) {
        SetError(error_message, MakeError("Kernel execution did not finish successfully", status));
        clReleaseMemObject(dropped_buf);
        clReleaseMemObject(used_buf);
        clReleaseMemObject(jtr_buf);
        clReleaseMemObject(jtj_buf);
        clReleaseMemObject(labels_buf);
        clReleaseMemObject(tgt_buf);
        clReleaseMemObject(src_buf);
        return false;
    }

    status = clEnqueueReadBuffer(
        handles_->queue, jtj_buf, CL_TRUE, 0, jtj_bytes, jtj_out.data(), 0, nullptr, nullptr);
    status |= clEnqueueReadBuffer(
        handles_->queue, jtr_buf, CL_TRUE, 0, jtr_bytes, jtr_out.data(), 0, nullptr, nullptr);
    status |= clEnqueueReadBuffer(
        handles_->queue, used_buf, CL_TRUE, 0, scalar_bytes, &used_count, 0, nullptr, nullptr);
    status |= clEnqueueReadBuffer(
        handles_->queue, dropped_buf, CL_TRUE, 0, scalar_bytes, &dropped_count, 0, nullptr, nullptr);

    clReleaseMemObject(dropped_buf);
    clReleaseMemObject(used_buf);
    clReleaseMemObject(jtr_buf);
    clReleaseMemObject(jtj_buf);
    clReleaseMemObject(labels_buf);
    clReleaseMemObject(tgt_buf);
    clReleaseMemObject(src_buf);

    if (status != CL_SUCCESS) {
        SetError(error_message, MakeError("Failed to read kernel outputs", status));
        return false;
    }

    return true;
#endif
}

}  // namespace graph_slam
