// ============================================================
// matmul_seq.cpp – Sequential (single-threaded) CPU matrix
//                  multiplication.
//
// Algorithm: naïve triple-loop, row-major traversal.
// No BLAS, no tiling (intentionally simple as a baseline).
// ============================================================

#include "matmul.hpp"
#include <cassert>

namespace cpu {

void matmul_seq(const Matrix& A, const Matrix& B, Matrix& C) {
    const std::size_t N = A.rows;
    assert(A.cols == B.rows && B.cols == N && C.rows == N && C.cols == N);

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
