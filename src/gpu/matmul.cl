/* ================================================================
 * matmul.cl – OpenCL C 1.2 kernel for square matrix multiplication.
 *
 * Uses shared local memory (tiling) to improve memory access
 * patterns on GPU hardware.
 *
 * Grid:  global = (N, N),  local = (TILE, TILE)
 * Each work-item computes one element C[row][col].
 * ================================================================ */

#define TILE 16

__kernel void matmul(
    __global const float* A,   /* N×N row-major */
    __global const float* B,   /* N×N row-major */
    __global       float* C,   /* N×N row-major (output) */
    const int N)
{
    /* Local tile storage */
    __local float tileA[TILE][TILE];
    __local float tileB[TILE][TILE];

    int row = get_global_id(1);   /* output row */
    int col = get_global_id(0);   /* output col */

    int lrow = get_local_id(1);
    int lcol = get_local_id(0);

    float acc = 0.0f;

    /* Stride over tiles along the k-dimension */
    int num_tiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {
        int kA = t * TILE + lcol;   /* column of A for this thread */
        int kB = t * TILE + lrow;   /* row    of B for this thread */

        /* Load tiles cooperatively */
        tileA[lrow][lcol] = (row < N && kA < N) ? A[row * N + kA] : 0.0f;
        tileB[lrow][lcol] = (kB  < N && col < N) ? B[kB  * N + col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        /* Compute partial dot product for this tile */
        for (int k = 0; k < TILE; ++k) {
            acc += tileA[lrow][k] * tileB[k][lcol];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < N && col < N) {
        C[row * N + col] = acc;
    }
}
