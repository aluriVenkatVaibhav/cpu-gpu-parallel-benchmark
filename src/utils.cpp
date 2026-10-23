// ============================================================
// utils.cpp – shared utility implementations.
// ============================================================

#include "matmul.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>

namespace util {

void fill_deterministic(Matrix& M, unsigned seed) {
    // Linear congruential generator – same values every run.
    std::uint32_t state = seed;
    for (auto& v : M.data) {
        state = state * 1664525u + 1013904223u;
        // Map to [-1, 1]
        v = static_cast<float>(static_cast<int32_t>(state)) / 2147483648.0f;
    }
}

float max_abs_diff(const Matrix& A, const Matrix& B) {
    float mx = 0.f;
    for (std::size_t i = 0; i < A.data.size(); ++i) {
        float d = std::abs(A.data[i] - B.data[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    std::size_t n = v.size();
    if (n % 2 == 0)
        return (v[n / 2 - 1] + v[n / 2]) * 0.5;
    return v[n / 2];
}

} // namespace util
