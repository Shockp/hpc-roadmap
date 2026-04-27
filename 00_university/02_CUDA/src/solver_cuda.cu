#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>

#include "solver_cuda.cuh"

namespace heat_sim {

// -----------------------------------------------------------------------------
// 1. DEVICE CODE (GPU KERNEL)
// -----------------------------------------------------------------------------

// Thread block dimensions optimized for warp scheduling (256 threads total)
constexpr int kBlockDimX = 16;
constexpr int kBlockDimY = 16;

__global__ void StencilKernel(const double* __restrict__ d_t_old,
                              double* __restrict__ d_t_new, int n) {
  // Global thread indices mapping to the 2D grid
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  // Shared memory tile with a 1-cell pading (halo) on all sides.
  // This allows the block to compute the stencil without fetching neighbors
  // from global memory during the math phase.
  __shared__ double tile[kBlockDimY + 2][kBlockDimX + 2];

  // Local thread indices within the block
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Map local thread to shared memory coordinates (offset by 1 for padding)
  int s_col = tx + 1;
  int s_row = ty + 1;

  // Initialize the specific shared memory cell for this thread
  tile[s_row][s_col] = 0.0;

  // --- COLLABORATIVE LOAD TO SHARED MEMORY ---
  if (row < n && col < n) {
    // 1. Every active thread loads its primary cell
    tile[s_row][s_col] = d_t_old[row * n + col];

    // 2. Threads on the boundary of the block load the extra halo cells
    if (tx == 0 && col > 0) {
      tile[s_row][0] = d_t_old[row * n + (col - 1)];  // Left halo
    }
    if (tx == kBlockDimX - 1 && col < n - 1) {
      tile[s_row][s_col + 1] = d_t_old[row * n + (col + 1)];  // Right halo
    }
    if (ty == 0 && row > 0) {
      tile[0][s_col] = d_t_old[(row - 1) * n + col];  // Top halo
    }
    if (ty == kBlockDimY - 1 && row < n - 1) {
      tile[s_row + 1][s_col] = d_t_old[(row + 1) * n + col];  // Bottom halo
    }

    // Barrier : Ensure the entire tile(including all halos) is loaded
    // before any thread begins computation.
    __syncthreads();

    // --- STENCIL COMPUTATION ---
    // Only apply the stencil to internal points (excluding global edges).
    // All reads are now happening from the ultra-fast 'tile' array.
    if (row > 0 && row < n - 1 && col > 0 && col < n - 1) {
      d_t_new[row * n + col] =
          0.25 * (tile[s_row - 1][s_col] + tile[s_row + 1][s_col] +
                  tile[s_row][s_col - 1] + tile[s_row][s_col + 1]);
    }
  }
}

}  // namespace heat_sim