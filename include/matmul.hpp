#pragma once
// ============================================================
// matmul.hpp – shared types and interfaces for all matmul
//              implementations in the CPU/GPU benchmark suite.
// ============================================================

#include <cstddef>
#include <vector>
#include <string>

// ---- Matrix type ---------------------------------------------------
// Row-major, single-precision float matrix stored as a flat vector.
struct Matrix {
    std::size_t rows, cols;
    std::vector<float> data;

    Matrix() : rows(0), cols(0) {}
    Matrix(std::size_t r, std::size_t c) : rows(r), cols(c), data(r * c, 0.f) {}

    float&       at(std::size_t r, std::size_t c)       { return data[r * cols + c]; }
    const float& at(std::size_t r, std::size_t c) const { return data[r * cols + c]; }
};

// ---- Result structure ----------------------------------------------
struct MatmulResult {
    std::size_t N;                  // matrix dimension (N×N)
    double      seq_ms;             // sequential CPU        – median wall-clock (ms)
    double      tiled_ms;           // cache-tiled CPU (1T)  – median wall-clock (ms)
    double      omp_ms;             // OpenMP CPU (naïve)    – median wall-clock (ms)
    double      tiled_omp_ms;       // OpenMP tiled CPU      – median wall-clock (ms)
    double      gpu_kernel_ms;      // GPU kernel only       – median (ms)  [no transfer]
    double      gpu_e2e_ms;         // GPU end-to-end        – median (ms)  [transfer+kernel+readback]
    double      tiled_speedup;      // baseline / tiled
    double      omp_speedup;        // baseline / omp
    double      tiled_omp_speedup;  // baseline / tiled_omp
    double      gpu_kernel_speedup; // baseline / gpu_kernel
    double      gpu_e2e_speedup;    // baseline / gpu_e2e
    bool        tiled_correct;
    bool        omp_correct;
    bool        tiled_omp_correct;
    bool        gpu_correct;
};

// ---- CPU implementations -------------------------------------------
namespace cpu {
    // C = A × B  (sequential naïve triple-loop, O(N³))
    // Intentionally un-optimised; serves as the measurement baseline.
    void matmul_seq(const Matrix& A, const Matrix& B, Matrix& C);

    // C = A × B  (cache-optimised, single-threaded, i-k-j tiled loop)
    // Block size configured by MATMUL_BLOCK_SIZE (default 64).
    void matmul_tiled(const Matrix& A, const Matrix& B, Matrix& C);

    // C = A × B  (OpenMP-parallelised naïve triple-loop)
    // Uses same algorithm as matmul_seq; speedup reflects parallelism only.
    void matmul_omp(const Matrix& A, const Matrix& B, Matrix& C);

    // C = A × B  (cache-optimised, OpenMP-parallelised, i-k-j tiled loop)
    // Combines cache blocking with multi-threading across independent output tiles.
    void matmul_tiled_omp(const Matrix& A, const Matrix& B, Matrix& C);
}

// ---- Utility -------------------------------------------------------
namespace util {
    // Fill matrix with deterministic pseudo-random values (reproducible seed).
    void fill_deterministic(Matrix& M, unsigned seed = 42u);

    // Element-wise max absolute difference.
    float max_abs_diff(const Matrix& A, const Matrix& B);

    // Return median of a vector (copy).
    double median(std::vector<double> v);
}
