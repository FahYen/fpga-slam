cmake_minimum_required(VERSION 3.16...3.28)
project(rnsg_ipc
    VERSION 0.1.0
    DESCRIPTION "Shared-memory FIFO between RangeNet and SG-SLAM"
    LANGUAGES C)

option(RNSG_BUILD_TESTS "Build rnsg_ipc tests" ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(rnsg_ipc SHARED
    src/rnsg_ring.c)
target_include_directories(rnsg_ipc
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_compile_options(rnsg_ipc PRIVATE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion
    -fvisibility=hidden)
# Export everything declared in the header. The header has no visibility macros
# so we re-enable default visibility for symbols matching the public prefix.
target_compile_options(rnsg_ipc PRIVATE -fvisibility=default)

# POSIX shm + threads + realtime (sem_timedwait, shm_open).
find_package(Threads REQUIRED)
target_link_libraries(rnsg_ipc PUBLIC Threads::Threads)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(rnsg_ipc PUBLIC rt)
endif()

set_target_properties(rnsg_ipc PROPERTIES
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION   ${PROJECT_VERSION})

install(TARGETS rnsg_ipc
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

if(RNSG_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
