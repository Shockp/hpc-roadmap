#include "solver_cuda.cuh"

#include <cuda_runtime.h>

#include <chrono>

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

// -----------------------------------------------------------------------------
// 2. HOST CODE (CPU WRAPPER)
// -----------------------------------------------------------------------------

ProfilerResult SolverCuda::Run(Grid& host_grid, int iterations) {
  ProfilerResult res;
  const int n = host_grid.cols();
  const size_t bytes = n * n * sizeof(double);

  auto start_comm1 = std::chrono::high_resolution_clock::now();
  // 1. Allocate Device Memory
  double* d_t_old = nullptr;
  double* d_t_new = nullptr;

  // cudaMalloc allocates memory directly on the GPU's global memory (VRAM)
  cudaMalloc(&d_t_old, bytes);
  cudaMalloc(&d_t_new, bytes);

  // 2. Transfer Initial State (Host -> Device)
  // We only transfer across the slow PCIe bus ONCE before the loop begins.
  cudaMemcpy(d_t_old, host_grid.t_old_ptr(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_t_new, host_grid.t_new_ptr(), bytes, cudaMemcpyHostToDevice);
  auto end_comm1 = std::chrono::high_resolution_clock::now();
  res.comm_time +=
      std::chrono::duration<double>(end_comm1 - start_comm1).count();

  // 3. Configure Execution Grid
  dim3 threads(kBlockDimX, kBlockDimY);

  // Ceiling division ensures we spawn enough thread blocks to cover
  // the entire N x N grid, even if N is not perfectly divisible by 16.
  dim3 blocks((n + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);

  auto start_compute = std::chrono::high_resolution_clock::now();
  // 4. Time-Stepping Loop (Entirely on Device)
  for (int t = 0; t < iterations; ++t) {
    // Launch the kernel asynchronously
    StencilKernel<<<blocks, threads>>>(d_t_old, d_t_new, n);

    // Explicit synchronization is required here. We must wait for all
    // GPU thread blocks to finish writing to d_t_new before we swap.
    cudaDeviceSynchronize();

    // Swap device pointers in O(1) time.
    // No data is moved; we just swap the memory addresses.
    double* temp = d_t_old;
    d_t_old = d_t_new;
    d_t_new = temp;
  }
  auto end_compute = std::chrono::high_resolution_clock::now();
  res.compute_time +=
      std::chrono::duration<double>(end_compute - start_compute).count();

  auto start_comm2 = std::chrono::high_resolution_clock::now();
  // 5. Transfer Final State Back (Device -> Host)
  // Because we swapped pointers at the end of every loop iteration,
  // the final, correct state is always held by d_t_old.
  cudaMemcpy(const_cast<double*>(host_grid.t_old_ptr()), d_t_old, bytes,
             cudaMemcpyDeviceToHost);

  // 6. Cleanup Device Memory to prevent memory leaks
  cudaFree(d_t_old);
  cudaFree(d_t_new);
  auto end_comm2 = std::chrono::high_resolution_clock::now();
  res.comm_time +=
      std::chrono::duration<double>(end_comm2 - start_comm2).count();

  return res;
}

}  // namespace heat_sim