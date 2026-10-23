// ============================================================
// matmul_tiled.cpp – Cache-optimised, single-threaded CPU
//                    matrix multiplication using loop blocking.
//
// Algorithm: i-k-j loop ordering with BLOCK×BLOCK tiles.
//
// Why tiling?
//   The naïve i-j-k loop walks column-wise through B, causing
//   one cache miss per inner-loop iteration for large N.
//   Tiling works on a BLOCK×BLOCK sub-matrix that fits in L1/L2
//   cache, reusing loaded data before eviction.
//
// Loop ordering (i-k-j within tile):
//   For each (I,K) tile pair, the inner j-loop strides through
//   the same row of A (already cached) and a contiguous row of B
//   (cache-friendly), writing a contiguous slice of C.
//
// Block size: MATMUL_BLOCK_SIZE (default 64).
//   - 64×64 floats = 16 KB ≈ half of a typical 32 KB L1 data cache.
//   - Three tiles (A, B, C) fit comfortably in L1.
//   - Can be overridden at compile time:
//       cmake ... -DMATMUL_BLOCK_SIZE=32
//
// Thread safety: single-threaded; no OpenMP or atomics.
// Memory:        no auxiliary allocations inside the timed call.
// ============================================================

#include "matmul.hpp"
#include <cassert>
#include <cstring>   // memset / memcpy not used, but for documentation

// Allow compile-time override via -DMATMUL_BLOCK_SIZE=N
#ifndef MATMUL_BLOCK_SIZE
#define MATMUL_BLOCK_SIZE 64
#endif

namespace cpu {

void matmul_tiled(const Matrix& A, const Matrix& B, Matrix& C) {
    const std::size_t N = A.rows;
    assert(A.cols == B.rows && B.cols == N && C.rows == N && C.cols == N);

    constexpr std::size_t BS = MATMUL_BLOCK_SIZE;

    // Zero the output (required: partial tiles do not always write every cell)
    std::fill(C.data.begin(), C.data.end(), 0.f);

    const float* __restrict__ pA = A.data.data();
    const float* __restrict__ pB = B.data.data();
    float*       __restrict__ pC = C.data.data();

    // Outer tile loops: iterate over (I, K, J) tile triplets.
    // i-k-j ordering maximises spatial locality in B and C within each tile.
    for (std::size_t ii = 0; ii < N; ii += BS) {
        const std::size_t i_end = std::min(ii + BS, N);

        for (std::size_t kk = 0; kk < N; kk += BS) {
            const std::size_t k_end = std::min(kk + BS, N);

            for (std::size_t jj = 0; jj < N; jj += BS) {
                const std::size_t j_end = std::min(jj + BS, N);

                // Inner micro-kernel: accumulate C[i][j] += A[i][k] * B[k][j]
                for (std::size_t i = ii; i < i_end; ++i) {
                    for (std::size_t k = kk; k < k_end; ++k) {
                        const float a_ik = pA[i * N + k];
                        for (std::size_t j = jj; j < j_end; ++j) {
                            pC[i * N + j] += a_ik * pB[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

} // namespace cpu
