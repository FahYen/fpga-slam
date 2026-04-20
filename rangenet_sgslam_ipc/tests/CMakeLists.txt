add_executable(test_roundtrip test_roundtrip.c)
target_link_libraries(test_roundtrip PRIVATE rnsg_ipc)
target_compile_options(test_roundtrip PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME rnsg_roundtrip COMMAND test_roundtrip)
set_tests_properties(rnsg_roundtrip PROPERTIES TIMEOUT 30)

add_executable(test_overrun test_overrun.c)
target_link_libraries(test_overrun PRIVATE rnsg_ipc)
target_compile_options(test_overrun PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME rnsg_overrun COMMAND test_overrun)
set_tests_properties(rnsg_overrun PROPERTIES TIMEOUT 30)

add_executable(test_held_slot test_held_slot.c)
target_link_libraries(test_held_slot PRIVATE rnsg_ipc)
target_compile_options(test_held_slot PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME rnsg_held_slot COMMAND test_held_slot)
set_tests_properties(rnsg_held_slot PROPERTIES TIMEOUT 30)

add_executable(test_reuse_released_slot test_reuse_released_slot.c)
target_link_libraries(test_reuse_released_slot PRIVATE rnsg_ipc)
target_compile_options(test_reuse_released_slot PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME rnsg_reuse_released_slot COMMAND test_reuse_released_slot)
set_tests_properties(rnsg_reuse_released_slot PROPERTIES TIMEOUT 30)

# Python two-process round-trip. Requires the just-built shared library on path.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(NAME rnsg_python_roundtrip
        COMMAND ${Python3_EXECUTABLE} -m unittest
                discover -s ${CMAKE_CURRENT_SOURCE_DIR} -p "test_python_*.py" -v
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set_tests_properties(rnsg_python_roundtrip PROPERTIES
        TIMEOUT 60
        ENVIRONMENT "RNSG_IPC_LIB=$<TARGET_FILE:rnsg_ipc>;PYTHONPATH=${CMAKE_SOURCE_DIR}/python")
endif()
