// ============================================================
// matmul_omp.cpp – OpenMP-parallel CPU matrix multiplication.
//
// Parallelises the outermost i-loop across all available
// hardware threads.  Uses the same triple-loop algorithm as
// the sequential version so speedup reflects parallelism only.
// ============================================================

#include "matmul.hpp"
#include <cassert>
#include <omp.h>

namespace cpu {

void matmul_omp(const Matrix& A, const Matrix& B, Matrix& C) {
    const std::size_t N = A.rows;
    assert(A.cols == B.rows && B.cols == N && C.rows == N && C.cols == N);

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            float acc = 0.f;
            for (std::size_t k = 0; k < N; ++k) {
                acc += A.data[i * N + k] * B.data[k * N + j];
            }
            C.data[i * N + j] = acc;
        }
    }
}

} // namespace cpu
