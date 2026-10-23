// ============================================================
// matmul_tiled_omp.cpp – Cache-optimised, OpenMP-parallel CPU
//                        matrix multiplication.
//
// Algorithm: Loop blocking (tiling) combined with OpenMP
//            multi-threading across independent output tiles.
//
// Thread Safety & Race Freedom:
//   The outer loops over (ii, jj) partition output matrix C into
//   disjoint BS×BS blocks: C[ii..min(ii+BS, N), jj..min(jj+BS, N)].
//   Each thread is exclusively assigned a set of (ii, jj) tiles.
//   Because no two threads ever write to or accumulate into the
//   same element of C, execution is completely race-free without
//   requiring mutexes, atomic operations, or reduction buffers.
//
// Memory Locality:
//   Within each tile, the loop ordering is i-k-j. The inner j-loop
//   accesses B and C contiguously with unit stride (row-major).
//   The working set of three BS×BS tiles fits comfortably within
//   each core's L1 data cache.
//
// Block size: MATMUL_BLOCK_SIZE (default 64, configurable at CMake time).
// ============================================================

#include "matmul.hpp"
#include <cassert>
#include <algorithm>
#include <omp.h>

#ifndef MATMUL_BLOCK_SIZE
#define MATMUL_BLOCK_SIZE 64
#endif

namespace cpu {

void matmul_tiled_omp(const Matrix& A, const Matrix& B, Matrix& C) {
    const std::size_t N = A.rows;
    assert(A.cols == B.rows && B.cols == N && C.rows == N && C.cols == N);

    constexpr std::size_t BS = MATMUL_BLOCK_SIZE;

    const float* __restrict__ pA = A.data.data();
    const float* __restrict__ pB = B.data.data();
    float*       __restrict__ pC = C.data.data();

    // Parallelize over the 2D grid of output tiles (ii, jj).
    // Each thread independently zeroes its assigned tile and accumulates
    // dot products across all k tiles into its private output block.
    #pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t ii = 0; ii < N; ii += BS) {
        for (std::size_t jj = 0; jj < N; jj += BS) {
            const std::size_t i_end = std::min(ii + BS, N);
            const std::size_t j_end = std::min(jj + BS, N);

            // Initialize assigned output tile to zero in local cache
            for (std::size_t i = ii; i < i_end; ++i) {
                for (std::size_t j = jj; j < j_end; ++j) {
                    pC[i * N + j] = 0.f;
                }
            }

            // Stride over tiles along the k dimension
            for (std::size_t kk = 0; kk < N; kk += BS) {
                const std::size_t k_end = std::min(kk + BS, N);

                // Micro-kernel with contiguous unit stride in B and C
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
