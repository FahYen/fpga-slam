# Milestone 1 benchmarking utilities for SG-SLAM FPGA exploration.

add_executable(benchmark_buildgraph
    benchmark_buildgraph.cpp
    ${PROJECT_SOURCE_DIR}/frontend/SemGraph.cpp
)

target_link_libraries(benchmark_buildgraph PRIVATE Eigen3::Eigen)
set_global_target_properties(benchmark_buildgraph)

add_executable(benchmark_registration
    benchmark_registration.cpp
)

target_link_libraries(benchmark_registration PRIVATE graph_slam::frontend)
set_global_target_properties(benchmark_registration)
