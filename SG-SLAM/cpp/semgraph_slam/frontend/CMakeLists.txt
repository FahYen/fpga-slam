#This file is covered by the LICENSE file in the root of this project.

add_library(frontend STATIC)
add_library(graph_slam::frontend ALIAS frontend)
target_sources(frontend PRIVATE Registration.cpp Deskew.cpp VoxelHashMap.cpp 
                            Preprocessing.cpp Threshold.cpp 
                            SemanticCluster.cpp SemGraph.cpp GraphMap.cpp
                            RegistrationXrtAdapter.cpp
                            ${PROJECT_SOURCE_DIR}/hls/registration_kernel_draft.cpp)
target_link_libraries(frontend PUBLIC Eigen3::Eigen tsl::robin_map TBB::tbb Sophus::Sophus Ceres::ceres)

if(ENABLE_XRT_REGISTRATION)
    find_package(OpenCL REQUIRED)
    target_compile_definitions(frontend PUBLIC SGSLAM_ENABLE_XRT=1)
    target_link_libraries(frontend PUBLIC OpenCL::OpenCL)
endif()
#OpenMP::OpenMP_CXX
set_global_target_properties(frontend)
