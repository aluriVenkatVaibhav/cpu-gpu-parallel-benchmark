// ============================================================
// matmul_gpu.cpp – OpenCL GPU matrix multiplication benchmark.
//
// Measurement definitions
// -----------------------
//  gpu_kernel_ms : Time between clEnqueueNDRangeKernel submission
//                  and completion (via cl_event profiling).
//                  Does NOT include host→device transfers or
//                  device→host readback.
//
//  gpu_e2e_ms    : Full end-to-end wall-clock time including:
//                    1. clEnqueueWriteBuffer A (host → device)
//                    2. clEnqueueWriteBuffer B (host → device)
//                    3. clEnqueueNDRangeKernel
//                    4. clEnqueueReadBuffer  C (device → host)
//                  Context creation, program compilation, and
//                  buffer allocation are performed once before
//                  the timed loop and are NOT included.
// ============================================================

#include "matmul.hpp"
#include "opencl_utils.hpp"
#include <CL/cl.h>
#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <cassert>
#include <cstring>      // memcpy
#include <filesystem>

// Path to the kernel source (resolved at runtime relative to executable).
static std::string kernel_path() {
    // Try relative paths from common working directories.
    std::vector<std::string> candidates = {
        "src/gpu/matmul.cl",
        "../src/gpu/matmul.cl",
        "matmul.cl"
    };
    for (auto& p : candidates) {
        if (std::filesystem::exists(p)) return p;
    }
    throw std::runtime_error("Cannot locate matmul.cl kernel source.");
}

// ---- Public benchmark function ------------------------------------

// Runs the GPU matmul benchmark for a single N.
// Returns { kernel_median_ms, e2e_median_ms, correct }.
struct GpuResult {
    double kernel_ms;
    double e2e_ms;
    bool   correct;
};

GpuResult gpu_matmul_bench(const Matrix& A, const Matrix& B,
                            const Matrix& ref_C,
                            int warmup_runs, int bench_runs,
                            float tolerance)
{
    const int N = static_cast<int>(A.rows);
    const std::size_t bytes = (std::size_t)N * N * sizeof(float);

    // ---------- One-time setup (not timed) -------------------------
    cl_platform_id platform{};
    cl_device_id   device = ocl::pick_device(&platform);

    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    cl_check(err, "clCreateContext");

    // Command queue with profiling enabled (for kernel timing).
    // Use the OpenCL 2.0 API (clCreateCommandQueueWithProperties) to avoid
    // the deprecation warning for clCreateCommandQueue.
    cl_queue_properties props[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0
    };
    cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, device,
                                props, &err);
    cl_check(err, "clCreateCommandQueueWithProperties");

    std::string src = ocl::read_file(kernel_path());
    cl_program  prog = ocl::build_program(ctx, device, src,
                                          "-cl-std=CL1.2 -cl-mad-enable");

    cl_kernel kernel = clCreateKernel(prog, "matmul", &err);
    cl_check(err, "clCreateKernel");

    // Allocate device buffers once
    cl_mem d_A = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  bytes, nullptr, &err);
    cl_check(err, "clCreateBuffer A");
    cl_mem d_B = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  bytes, nullptr, &err);
    cl_check(err, "clCreateBuffer B");
    cl_mem d_C = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
    cl_check(err, "clCreateBuffer C");

    // Set static kernel arguments (buffers and N)
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int),    &N));

    constexpr int TILE = 16;
    // Round up global size to multiple of TILE
    std::size_t g = ((std::size_t)N + TILE - 1) / TILE * TILE;
    const std::size_t global[2] = {g, g};
    const std::size_t local[2]  = {TILE, TILE};

    std::vector<float> h_C((std::size_t)N * N, 0.f);

    auto run_once = [&](double* out_kernel_ms, double* out_e2e_ms) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // Write A and B to device
        CL_CHECK(clEnqueueWriteBuffer(queue, d_A, CL_FALSE, 0, bytes,
                                      A.data.data(), 0, nullptr, nullptr));
        CL_CHECK(clEnqueueWriteBuffer(queue, d_B, CL_FALSE, 0, bytes,
                                      B.data.data(), 0, nullptr, nullptr));

        // Launch kernel (with event for profiling)
        cl_event kern_event;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2,
                                        nullptr, global, local,
                                        0, nullptr, &kern_event));

        // Read C back (blocking – waits for kernel)
        CL_CHECK(clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, bytes,
                                     h_C.data(), 1, &kern_event, nullptr));

        auto t1 = std::chrono::high_resolution_clock::now();
        *out_e2e_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Extract kernel-only time from event profiling
        cl_ulong t_start = 0, t_end = 0;
        clGetEventProfilingInfo(kern_event, CL_PROFILING_COMMAND_START,
                                sizeof(t_start), &t_start, nullptr);
        clGetEventProfilingInfo(kern_event, CL_PROFILING_COMMAND_END,
                                sizeof(t_end),   &t_end,   nullptr);
        *out_kernel_ms = (t_end - t_start) * 1e-6; // ns → ms

        clReleaseEvent(kern_event);
    };

    // ---------- Warm-up (results discarded) ------------------------
    double dummy_k, dummy_e;
    for (int w = 0; w < warmup_runs; ++w)
        run_once(&dummy_k, &dummy_e);

    // ---------- Benchmark loop ------------------------------------
    std::vector<double> kernel_times, e2e_times;
    kernel_times.reserve(bench_runs);
    e2e_times.reserve(bench_runs);

    for (int r = 0; r < bench_runs; ++r) {
        double km, em;
        run_once(&km, &em);
        kernel_times.push_back(km);
        e2e_times.push_back(em);
    }

    // ---------- Validation (one extra run with result check) -------
    double dummy2k, dummy2e;
    run_once(&dummy2k, &dummy2e);

    float max_diff = 0.f;
    for (std::size_t i = 0; i < (std::size_t)N * N; ++i) {
        float d = std::abs(h_C[i] - ref_C.data[i]);
        if (d > max_diff) max_diff = d;
    }
    bool correct = (max_diff <= tolerance);
    if (!correct) {
        std::cerr << "[GPU] Validation FAILED at N=" << N
                  << "  max_diff=" << max_diff
                  << "  tolerance=" << tolerance << "\n";
    }

    // ---------- Cleanup --------------------------------------------
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return {
        util::median(kernel_times),
        util::median(e2e_times),
        correct
    };
}
