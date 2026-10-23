// ============================================================
// opencl_utils.cpp – OpenCL helper implementations.
// ============================================================

#include "opencl_utils.hpp"
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

namespace ocl {

cl_device_id pick_device(cl_platform_id* out_platform) {
    cl_uint num_platforms = 0;
    CL_CHECK(clGetPlatformIDs(0, nullptr, &num_platforms));
    if (num_platforms == 0)
        throw std::runtime_error("No OpenCL platforms found.");

    std::vector<cl_platform_id> platforms(num_platforms);
    CL_CHECK(clGetPlatformIDs(num_platforms, platforms.data(), nullptr));

    // Prefer a GPU device.
    for (auto& plat : platforms) {
        cl_uint num_devs = 0;
        if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devs) != CL_SUCCESS)
            continue;
        if (num_devs == 0) continue;

        std::vector<cl_device_id> devs(num_devs);
        CL_CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, num_devs, devs.data(), nullptr));
        if (out_platform) *out_platform = plat;

        // Print device name for transparency
        char name[256] = {};
        clGetDeviceInfo(devs[0], CL_DEVICE_NAME, sizeof(name), name, nullptr);
        std::cout << "[OpenCL] Using device: " << name << "\n";
        return devs[0];
    }

    // Fallback: any device
    for (auto& plat : platforms) {
        cl_uint num_devs = 0;
        if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devs) != CL_SUCCESS)
            continue;
        if (num_devs == 0) continue;

        std::vector<cl_device_id> devs(num_devs);
        CL_CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, num_devs, devs.data(), nullptr));
        if (out_platform) *out_platform = plat;

        char name[256] = {};
        clGetDeviceInfo(devs[0], CL_DEVICE_NAME, sizeof(name), name, nullptr);
        std::cout << "[OpenCL] Fallback device: " << name << "\n";
        return devs[0];
    }
    throw std::runtime_error("No OpenCL devices found.");
}

cl_program build_program(cl_context ctx, cl_device_id dev,
                         const std::string& src,
                         const std::string& options) {
    const char* src_ptr = src.c_str();
    std::size_t src_len = src.size();

    cl_int err;
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src_ptr, &src_len, &err);
    cl_check(err, "clCreateProgramWithSource");

    cl_int build_err = clBuildProgram(prog, 1, &dev, options.c_str(), nullptr, nullptr);
    if (build_err != CL_SUCCESS) {
        std::size_t log_size = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        clReleaseProgram(prog);
        throw std::runtime_error("OpenCL build failed:\n" + log);
    }
    return prog;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open file: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

} // namespace ocl
