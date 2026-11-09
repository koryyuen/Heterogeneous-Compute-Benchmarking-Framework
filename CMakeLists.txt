cmake_minimum_required(VERSION 3.20)
project(HeteroBench LANGUAGES CXX)

option(HETEROBENCH_ENABLE_METAL "Build Apple Metal and MPS backends" ${APPLE})
option(HETEROBENCH_ENABLE_ACCELERATE "Build the Apple Accelerate reference" ${APPLE})
set(CMAKE_CXX_EXTENSIONS OFF)
include(CTest)

execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE source_revision OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
execute_process(COMMAND git status --porcelain WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE source_changes OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(source_changes)
    set(source_state "dirty")
else()
    set(source_state "clean")
endif()
if(NOT source_revision)
    set(source_revision "unavailable")
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" build_type_upper)

add_library(heterobench_core STATIC src/cpu_backend.cpp src/benchmark.cpp src/results.cpp src/system_info.cpp)
target_compile_features(heterobench_core PUBLIC cxx_std_20)
target_include_directories(heterobench_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_definitions(heterobench_core PRIVATE
    HETEROBENCH_COMPILER="${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}"
    HETEROBENCH_BUILD_TYPE="$<CONFIG>"
    HETEROBENCH_GIT_REVISION="${source_revision}"
    HETEROBENCH_SOURCE_STATE="${source_state}"
    HETEROBENCH_BUILD_FLAGS="${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${build_type_upper}}"
)
add_executable(heterobench src/main.cpp)
target_link_libraries(heterobench PRIVATE heterobench_core)

if(BUILD_TESTING)
    add_executable(cpu_backend_test tests/cpu_backend_test.cpp)
    target_link_libraries(cpu_backend_test PRIVATE heterobench_core)
    add_test(NAME cpu_backend_validation COMMAND cpu_backend_test)
    add_executable(benchmark_test tests/benchmark_test.cpp)
    target_link_libraries(benchmark_test PRIVATE heterobench_core)
    add_test(NAME benchmark_validation COMMAND benchmark_test)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME export_and_cli_validation COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/export_test.py" "$<TARGET_FILE:heterobench>")
    endif()
endif()

if(HETEROBENCH_ENABLE_ACCELERATE)
    if(NOT APPLE)
        message(FATAL_ERROR "Accelerate requires Apple platforms")
    endif()
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    add_library(heterobench_accelerate STATIC src/accelerate_backend.cpp)
    target_compile_definitions(heterobench_accelerate PRIVATE ACCELERATE_NEW_LAPACK)
    target_compile_features(heterobench_accelerate PUBLIC cxx_std_20)
    target_include_directories(heterobench_accelerate PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(heterobench_accelerate PUBLIC "${ACCELERATE_FRAMEWORK}")
    target_link_libraries(heterobench PRIVATE heterobench_accelerate)
    target_compile_definitions(heterobench PRIVATE HETEROBENCH_HAS_ACCELERATE=1)
    if(BUILD_TESTING)
        add_executable(accelerate_backend_test tests/accelerate_backend_test.cpp)
        target_link_libraries(accelerate_backend_test PRIVATE heterobench_accelerate heterobench_core)
        add_test(NAME accelerate_backend_validation COMMAND accelerate_backend_test)
    endif()
endif()

if(HETEROBENCH_ENABLE_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "Metal requires Apple platforms")
    endif()
    enable_language(OBJCXX)
    find_program(XCRUN_EXECUTABLE xcrun REQUIRED)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    set(shader_source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/matmul.metal")
    set(shader_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders")
    set(shader_air "${shader_directory}/matmul.air")
    set(shader_library "${shader_directory}/matmul.metallib")
    add_custom_command(
        OUTPUT "${shader_library}"
        BYPRODUCTS "${shader_air}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${shader_directory}"
        COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal -c "${shader_source}" -o "${shader_air}"
        COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metallib "${shader_air}" -o "${shader_library}"
        DEPENDS "${shader_source}"
        COMMENT "Building naive and tiled Metal shaders"
        VERBATIM
    )
    add_custom_target(metal_shaders DEPENDS "${shader_library}")
    add_library(heterobench_metal STATIC src/metal_backend.mm)
    set_target_properties(heterobench_metal PROPERTIES
        OBJCXX_STANDARD 20 OBJCXX_STANDARD_REQUIRED YES OBJCXX_EXTENSIONS OFF)
    target_compile_options(heterobench_metal PRIVATE -fobjc-arc)
    target_compile_definitions(heterobench_metal PRIVATE HETEROBENCH_METALLIB_PATH="${shader_library}")
    target_compile_features(heterobench_metal INTERFACE cxx_std_20)
    target_include_directories(heterobench_metal PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(heterobench_metal PUBLIC
        "${METAL_FRAMEWORK}" "${MPS_FRAMEWORK}" "${FOUNDATION_FRAMEWORK}")
    add_dependencies(heterobench_metal metal_shaders)
    target_link_libraries(heterobench PRIVATE heterobench_metal)
    target_compile_definitions(heterobench PRIVATE HETEROBENCH_HAS_METAL=1)
    add_executable(metal_smoke tests/metal_smoke.mm)
    set_target_properties(metal_smoke PROPERTIES
        OBJCXX_STANDARD 20 OBJCXX_STANDARD_REQUIRED YES OBJCXX_EXTENSIONS OFF)
    target_compile_options(metal_smoke PRIVATE -fobjc-arc)
    target_link_libraries(metal_smoke PRIVATE heterobench_metal)
    if(BUILD_TESTING)
        add_executable(metal_backend_test tests/metal_backend_test.cpp)
        target_link_libraries(metal_backend_test PRIVATE heterobench_metal heterobench_core)
        add_test(NAME metal_backend_validation COMMAND metal_backend_test)
        set_tests_properties(metal_backend_validation PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
    endif()
endif()
