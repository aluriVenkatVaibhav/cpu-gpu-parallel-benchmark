#pragma once
// ============================================================
// opencl_utils.hpp – lightweight OpenCL helper utilities.
// ============================================================

#include <CL/cl.h>
#include <string>
#include <stdexcept>

// ---- Error checking ------------------------------------------------
static inline void cl_check(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string(msg) + " [CL error " + std::to_string(err) + "]");
    }
}

#define CL_CHECK(expr)   cl_check((expr), #expr)

// ---- Helpers -------------------------------------------------------
namespace ocl {
    // Select the first GPU device; falls back to any device if no GPU found.
    cl_device_id pick_device(cl_platform_id* out_platform = nullptr);

    // Build a program from source string; throws on compile error (with log).
    cl_program build_program(cl_context ctx, cl_device_id dev,
                             const std::string& src,
                             const std::string& options = "");

    // Read a text file into a string.
    std::string read_file(const std::string& path);
}
