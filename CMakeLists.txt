cmake_minimum_required(VERSION 3.22)
project(cpu_gpu_bench LANGUAGES CXX)

# ---- Build settings ------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Release with debug info: accurate timings + readable stack traces.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE         "-O3 -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO  "-O3 -march=native -g")

# ---- Configurable block size for cache-tiled matmul ----------------
# Override at configure time: cmake -B build -DMATMUL_BLOCK_SIZE=32
set(MATMUL_BLOCK_SIZE "64" CACHE STRING
    "Tile/block size for the cache-optimised CPU matmul (default 64)")
message(STATUS "Cache-tiled matmul block size: ${MATMUL_BLOCK_SIZE}")

# ---- Dependencies --------------------------------------------------
find_package(OpenMP REQUIRED)
find_package(OpenCL REQUIRED)

# ---- Sources -------------------------------------------------------
add_executable(matmul_bench
    src/main.cpp
    src/utils.cpp
    src/cpu/matmul_seq.cpp
    src/cpu/matmul_tiled.cpp
    src/cpu/matmul_omp.cpp
    src/cpu/matmul_tiled_omp.cpp
    src/gpu/matmul_gpu.cpp
    src/gpu/opencl_utils.cpp
)

# Target OpenCL 2.0 host API (for clCreateCommandQueueWithProperties etc.)
# The kernel itself is compiled at the OpenCL C 1.2 language level (-cl-std=CL1.2).
target_compile_definitions(matmul_bench PRIVATE
    CL_TARGET_OPENCL_VERSION=200
    MATMUL_BLOCK_SIZE=${MATMUL_BLOCK_SIZE}
)
target_include_directories(matmul_bench PRIVATE include)
target_link_libraries(matmul_bench PRIVATE
    OpenMP::OpenMP_CXX
    OpenCL::OpenCL
)

# ---- Ensure results/ exists at build time --------------------------
add_custom_command(TARGET matmul_bench POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
            "${CMAKE_SOURCE_DIR}/results"
)

# ---- Runtime note --------------------------------------------------
# The benchmark resolves matmul.cl relative to its working directory.
# Run the executable from the project root so "src/gpu/matmul.cl" resolves.
message(STATUS "Kernel source will be resolved at runtime from the project root.")
