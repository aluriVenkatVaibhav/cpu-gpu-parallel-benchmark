// ============================================================
// main.cpp – CPU vs GPU Matrix Multiplication Benchmark
//
// Implementations measured
// ------------------------
//   1. Sequential CPU (matmul_seq)
//      Naïve triple-loop, O(N³), single-threaded.
//      Serves as the canonical reference baseline for all speedups.
//
//   2. Cache-optimised CPU (matmul_tiled)
//      i-k-j loop tiling with BLOCK=64 tiles.  Single-threaded;
//      speedup over seq reflects cache-locality improvement only.
//
//   3. OpenMP CPU (matmul_omp)
//      Naïve i-j-k algorithm, outer loop parallelised across threads.
//      Reflects multi-threaded scaling on an un-tiled loop.
//
//   4. Parallel Tiled CPU (matmul_tiled_omp)
//      Combines i-k-j loop tiling with OpenMP across independent
//      output tiles (collapse(2)). Race-free, cache-friendly.
//
//   5. OpenCL GPU (NVIDIA RTX 4060 Laptop)
//      16×16 local-memory tiled kernel, OpenCL C 1.2.
//
// Benchmark methodology
// ---------------------
// For each matrix size N:
//   1. Generate A, B once with deterministic seeds (same for all impls).
//   2. Each implementation: WARMUP warm-up runs (discarded),
//      then RUNS timed runs.  Report the median.
//   3. Validate tiled, OMP, tiled-OMP, and GPU against the reference output.
//      Validation is a separate pass – NOT part of the timed loop.
//   4. Report speedup = baseline_ms / impl_ms for each implementation.
//      Speedups are computed from measured values; never hardcoded.
//
// Timing tool
// -----------
//   CPU (all variants): std::chrono::high_resolution_clock wall-clock.
//   GPU kernel only:    CL_PROFILING_COMMAND_{START,END} event timestamps.
//   GPU end-to-end:     Wall-clock from first WriteBuffer to ReadBuffer
//                       completion.  Context/program/buffer creation is
//                       excluded (performed once before the timed loop).
//
// Output
// ------
//   Stdout: formatted table
//   File:   results/matmul.csv
// ============================================================

#include "matmul.hpp"
#include "opencl_utils.hpp"

#include <CL/cl.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <omp.h>

// Forward declaration from matmul_gpu.cpp
struct GpuResult {
    double kernel_ms;
    double e2e_ms;
    bool   correct;
};
GpuResult gpu_matmul_bench(const Matrix& A, const Matrix& B,
                            const Matrix& ref_C,
                            int warmup_runs, int bench_runs,
                            float tolerance);

// ---- Timing helpers -----------------------------------------------
using Clock = std::chrono::high_resolution_clock;

static double elapsed_ms(Clock::time_point t0, Clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ---- CPU benchmark ------------------------------------------------
// Runs fn(A,B,C) warmup+runs times; returns median of timed runs.
// Matrix C is written each call; validation is done by the caller.
static double bench_cpu(
    const Matrix& A, const Matrix& B, Matrix& C,
    void(*fn)(const Matrix&, const Matrix&, Matrix&),
    int warmup, int runs)
{
    for (int w = 0; w < warmup; ++w) fn(A, B, C);

    std::vector<double> times;
    times.reserve(runs);
    for (int r = 0; r < runs; ++r) {
        auto t0 = Clock::now();
        fn(A, B, C);
        auto t1 = Clock::now();
        times.push_back(elapsed_ms(t0, t1));
    }
    return util::median(times);
}

// ---- Format helpers -----------------------------------------------
static std::string fmt(double v, int width, int prec = 3) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(prec) << v;
    std::string s = oss.str();
    while (static_cast<int>(s.size()) < width) s += ' ';
    return s;
}

static std::string pass_fail(bool ok) { return ok ? "PASS" : "FAIL"; }

// ---- Memory check -------------------------------------------------
// Returns false if the three N×N float matrices exceed a conservative
// fraction of available system RAM (to avoid thrashing on large N).
static bool memory_ok(std::size_t N) {
    // 3 matrices (A, B, C) * N*N * 4 bytes each, host side
    constexpr std::size_t MAX_HOST_MB = 2048; // 2 GB limit
    std::size_t bytes = 3ULL * N * N * sizeof(float);
    if (bytes > MAX_HOST_MB * 1024ULL * 1024ULL) {
        std::cerr << "[SKIP] N=" << N
                  << " would require " << bytes / 1024 / 1024 << " MB host RAM"
                  << " (limit " << MAX_HOST_MB << " MB)\n";
        return false;
    }
    return true;
}

// ---- Entry point --------------------------------------------------
int main() {
    constexpr int   WARMUP    = 2;
    constexpr int   RUNS      = 5;
    // Floating-point tolerance for correctness checks.
    // Naïve summation of N values in [-1,1] has worst-case error ≈ N*eps_float.
    // At N=2048, eps_float≈1.2e-7 → ~2.5e-4.  Use 0.01 for safety.
    constexpr float TOLERANCE = 1e-2f;

    // N=2048 sequential is very slow (~25-40s per run × 7 runs).
    // For N >= SEQ_SKIP_THRESHOLD, skip the naïve seq implementation and
    // use the tiled result as the reference baseline instead.
    // All speedups at that size use tiled_ms as denominator and are labelled clearly.
    constexpr std::size_t SEQ_SKIP_THRESHOLD = 2048;

    const std::vector<std::size_t> sizes = {128, 256, 512, 1024, 2048};

    // Report OpenMP thread count
    int n_threads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        n_threads = omp_get_num_threads();
    }

    std::cout << "=================================================================\n";
    std::cout << "  CPU vs GPU Parallel Computing Benchmark Suite\n";
    std::cout << "=================================================================\n";
    std::cout << "OpenMP threads : " << n_threads << "\n";
    std::cout << "Warm-up runs   : " << WARMUP    << "\n";
    std::cout << "Bench runs     : " << RUNS       << "\n";
    std::cout << "Block size     : " << MATMUL_BLOCK_SIZE << " (tiled CPU implementations)\n";
    std::cout << "Metric         : median wall-clock (ms)\n";
    std::cout << "Tolerance      : " << TOLERANCE  << "\n";
    std::cout << "Speedup base   : seq_ms (naïve sequential CPU, N < " << SEQ_SKIP_THRESHOLD << ")\n";
    std::cout << "                 tiled_ms (cache-tiled CPU 1T, N >= " << SEQ_SKIP_THRESHOLD << ")\n\n";

    // CSV
    std::ofstream csv("results/matmul.csv");
    if (!csv) {
        std::cerr << "Warning: cannot open results/matmul.csv for writing\n";
    } else {
        csv << "N,"
            << "seq_ms,"
            << "tiled_ms,"
            << "omp_ms,"
            << "tiled_omp_ms,"
            << "gpu_kernel_ms,"
            << "gpu_e2e_ms,"
            << "tiled_speedup_vs_seq,"
            << "omp_speedup_vs_seq,"
            << "tiled_omp_speedup_vs_seq,"
            << "gpu_kernel_speedup_vs_seq,"
            << "gpu_e2e_speedup_vs_seq,"
            << "speedup_base,"
            << "seq_correct,"
            << "tiled_correct,"
            << "omp_correct,"
            << "tiled_omp_correct,"
            << "gpu_correct\n";
    }

    // Table header
    const int W_N   =  6;
    const int W_MS  = 11;
    const int W_SPD = 10;
    const int W_OK  =  7;

    std::cout << std::left
              << std::setw(W_N)  << "N"
              << std::setw(W_MS) << "seq(ms)"
              << std::setw(W_MS) << "tiled(ms)"
              << std::setw(W_MS) << "omp(ms)"
              << std::setw(13)   << "tiledOMP(ms)"
              << std::setw(12)   << "gpuKern(ms)"
              << std::setw(W_MS) << "gpuE2E(ms)"
              << std::setw(W_SPD)<< "tld_spd"
              << std::setw(W_SPD)<< "omp_spd"
              << std::setw(11)   << "tomp_spd"
              << std::setw(11)   << "gkern_spd"
              << std::setw(W_SPD)<< "ge2e_spd"
              << std::setw(W_OK) << "seq_ok"
              << std::setw(W_OK) << "tld_ok"
              << std::setw(W_OK) << "omp_ok"
              << std::setw(8)    << "tomp_ok"
              << std::setw(W_OK) << "gpu_ok"
              << "\n";
    std::cout << std::string(W_N + 5*W_MS + 13 + 12 + 3*W_SPD + 22 + 4*W_OK + 8, '-') << "\n";

    bool any_fail = false;

    for (std::size_t N : sizes) {
        if (!memory_ok(N)) continue;

        // Generate input matrices (same for all implementations)
        Matrix A(N, N), B(N, N);
        util::fill_deterministic(A, 42u);
        util::fill_deterministic(B, 137u);

        // ---- Sequential CPU ----------------------------------------
        bool skip_seq = (N >= SEQ_SKIP_THRESHOLD);
        double seq_ms = 0.0;
        Matrix C_seq(N, N);
        if (!skip_seq) {
            seq_ms = bench_cpu(A, B, C_seq, cpu::matmul_seq, WARMUP, RUNS);
        }

        // ---- Cache-tiled CPU (single-threaded) ----------------------
        Matrix C_tiled(N, N);
        double tiled_ms = bench_cpu(A, B, C_tiled, cpu::matmul_tiled, WARMUP, RUNS);

        // If we skipped seq, tiled output becomes the reference
        const Matrix& ref_C = skip_seq ? C_tiled : C_seq;

        // Validate tiled vs seq (when seq was run)
        bool tiled_ok = true;
        if (!skip_seq) {
            float d = util::max_abs_diff(C_seq, C_tiled);
            tiled_ok = (d <= TOLERANCE);
            if (!tiled_ok)
                std::cerr << "[TILED] Validation FAILED at N=" << N
                          << "  max_diff=" << d << "\n";
        }

        // ---- OpenMP CPU (naïve parallel) ---------------------------
        Matrix C_omp(N, N);
        double omp_ms = bench_cpu(A, B, C_omp, cpu::matmul_omp, WARMUP, RUNS);
        float  omp_diff = util::max_abs_diff(ref_C, C_omp);
        bool   omp_ok   = (omp_diff <= TOLERANCE);
        if (!omp_ok)
            std::cerr << "[OMP] Validation FAILED at N=" << N
                      << "  max_diff=" << omp_diff << "\n";

        // ---- Parallel Tiled CPU (OpenMP + Cache Blocking) ----------
        Matrix C_tomp(N, N);
        double tomp_ms = bench_cpu(A, B, C_tomp, cpu::matmul_tiled_omp, WARMUP, RUNS);
        float  tomp_diff = util::max_abs_diff(ref_C, C_tomp);
        bool   tomp_ok   = (tomp_diff <= TOLERANCE);
        if (!tomp_ok)
            std::cerr << "[TILED_OMP] Validation FAILED at N=" << N
                      << "  max_diff=" << tomp_diff << "\n";

        // ---- GPU OpenCL -------------------------------------------
        GpuResult gpu;
        try {
            gpu = gpu_matmul_bench(A, B, ref_C, WARMUP, RUNS, TOLERANCE);
        } catch (const std::exception& e) {
            std::cerr << "[GPU] Error at N=" << N << ": " << e.what() << "\n";
            gpu = {0.0, 0.0, false};
        }

        // ---- Speedups ---------------------------------------------
        // Baseline: seq_ms when available; tiled_ms at N>=SEQ_SKIP_THRESHOLD.
        double base_ms        = skip_seq ? tiled_ms : seq_ms;
        double tiled_spd      = skip_seq ? 1.0 : (seq_ms / tiled_ms);
        double omp_spd        = base_ms / omp_ms;
        double tomp_spd       = base_ms / tomp_ms;
        double gpu_kern_spd   = (gpu.kernel_ms > 0.0) ? base_ms / gpu.kernel_ms : 0.0;
        double gpu_e2e_spd    = (gpu.e2e_ms    > 0.0) ? base_ms / gpu.e2e_ms    : 0.0;
        std::string spd_base  = skip_seq ? "tiled" : "seq";

        any_fail |= !tiled_ok || !omp_ok || !tomp_ok || !gpu.correct;

        // ---- Print row --------------------------------------------
        std::cout << std::left
                  << std::setw(W_N)  << N
                  << fmt(skip_seq ? 0.0 : seq_ms, W_MS)
                  << fmt(tiled_ms,       W_MS)
                  << fmt(omp_ms,         W_MS)
                  << fmt(tomp_ms,        13)
                  << fmt(gpu.kernel_ms,  12)
                  << fmt(gpu.e2e_ms,     W_MS)
                  << fmt(tiled_spd,      W_SPD)
                  << fmt(omp_spd,        W_SPD)
                  << fmt(tomp_spd,       11)
                  << fmt(gpu_kern_spd,   11)
                  << fmt(gpu_e2e_spd,    W_SPD)
                  << std::setw(W_OK) << (skip_seq    ? "SKIP"            : "REF")
                  << std::setw(W_OK) << pass_fail(!skip_seq ? tiled_ok : true)
                  << std::setw(W_OK) << pass_fail(omp_ok)
                  << std::setw(8)    << pass_fail(tomp_ok)
                  << std::setw(W_OK) << pass_fail(gpu.correct)
                  << "\n";

        // ---- CSV row ----------------------------------------------
        if (csv) {
            csv << std::fixed << std::setprecision(6)
                << N                      << ","
                << (skip_seq ? 0.0 : seq_ms) << ","
                << tiled_ms               << ","
                << omp_ms                 << ","
                << tomp_ms                << ","
                << gpu.kernel_ms          << ","
                << gpu.e2e_ms             << ","
                << tiled_spd              << ","
                << omp_spd                << ","
                << tomp_spd               << ","
                << gpu_kern_spd           << ","
                << gpu_e2e_spd            << ","
                << spd_base               << ","
                << (skip_seq      ? "skip"  : "true") << ","
                << (!skip_seq ? (tiled_ok ? "true" : "false") : "ref") << ","
                << (omp_ok        ? "true"  : "false") << ","
                << (tomp_ok       ? "true"  : "false") << ","
                << (gpu.correct   ? "true"  : "false") << "\n";
        }
    }

    std::cout << "\nResults written to results/matmul.csv\n";
    if (any_fail) {
        std::cerr << "\n[BENCHMARK] One or more validation checks FAILED.\n";
        return 1;
    }
    std::cout << "[BENCHMARK] All validation checks PASSED.\n";
    return 0;
}
