set(RNSG_IPC_ROOT
    "${PROJECT_SOURCE_DIR}/../../../rangenet_sgslam_ipc"
    CACHE PATH "Path to the rangenet_sgslam_ipc project root")
set(GTSAM_EIGEN_COMPAT_DIR
    "${PROJECT_SOURCE_DIR}/../../../third_party/gtsam/gtsam/3rdparty/Eigen"
    CACHE PATH "Path to Eigen headers compatible with the installed GTSAM")

find_path(RNSG_IPC_INCLUDE_DIR
          NAMES rnsg_ipc.h
          PATHS
            "${RNSG_IPC_ROOT}/include"
            "${RNSG_IPC_ROOT}"
          REQUIRED)

find_library(RNSG_IPC_LIBRARY
             NAMES rnsg_ipc librnsg_ipc
             PATHS
               "${RNSG_IPC_ROOT}/build"
               "${RNSG_IPC_ROOT}/lib"
             REQUIRED)

add_executable(sgslam_ipc_runner
    FrameSource.cpp
    sgslam_ipc_runner.cpp)

target_include_directories(
    sgslam_ipc_runner
    BEFORE PRIVATE
        ${GTSAM_EIGEN_COMPAT_DIR}
        ${RNSG_IPC_INCLUDE_DIR})
target_link_libraries(sgslam_ipc_runner PRIVATE graph_slam::pipeline ${RNSG_IPC_LIBRARY} gtsam)
target_compile_options(sgslam_ipc_runner PRIVATE -Wno-deprecated-copy)
set_global_target_properties(sgslam_ipc_runner)
