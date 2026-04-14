#pragma once

#include <array>
#include <string>

namespace graph_slam {

class RegistrationXrtAdapter {
public:
    static RegistrationXrtAdapter &Instance();

    bool IsCompiledWithXrt() const;

    bool Configure(const std::string &xclbin_path, std::string *error_message = nullptr);

    bool Accumulate(const float *src_xyz,
                    const float *tgt_xyz,
                    const int *labels,
                    int correspondence_count,
                    float kernel,
                    std::array<double, 36> &jtj_out,
                    std::array<double, 6> &jtr_out,
                    int &used_count,
                    int &dropped_count,
                    std::string *error_message = nullptr);

private:
    RegistrationXrtAdapter();
    ~RegistrationXrtAdapter();

    RegistrationXrtAdapter(const RegistrationXrtAdapter &) = delete;
    RegistrationXrtAdapter &operator=(const RegistrationXrtAdapter &) = delete;

    void Cleanup();

    bool initialized_ = false;
    std::string xclbin_path_;

#if defined(SGSLAM_ENABLE_XRT)
    struct XrtHandles;
    XrtHandles *handles_ = nullptr;
#endif
};

}  // namespace graph_slam
