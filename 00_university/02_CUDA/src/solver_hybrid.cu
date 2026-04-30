#include "solver_hybrid.cuh"

#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <iostream>

namespace heat_sim {

constexpr int kBlockDimX = 16;
constexpr int kBlockDimY = 16;

// Device Kernel (Identical logic to CUDA, but acts on local sub-grids)
__global__ void HybridStencilKernel(const double* __restrict__ d_t_old,
                                    double* __restrict__ d_t_new, int rows,
                                    int cols) {
  int col = blockIdx.x * kBlockDimX + threadIdx.x;
  int row = blockIdx.y * kBlockDimY + threadIdx.y;

  __shared__ double tile[kBlockDimY + 2][kBlockDimX + 2];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int s_col = tx + 1;
  int s_row = ty + 1;

  tile[s_row][s_col] = 0.0;

  if (row < rows && col < cols) {
    tile[s_row][s_col] = d_t_old[row * cols + col];
    if (tx == 0 && col > 0) tile[s_row][0] = d_t_old[row * cols + (col - 1)];
    if (tx == kBlockDimX - 1 && col < cols - 1)
      tile[s_row][s_col + 1] = d_t_old[row * cols + (col + 1)];
    if (ty == 0 && row > 0) tile[0][s_col] = d_t_old[(row - 1) * cols + col];
    if (ty == kBlockDimY - 1 && row < rows - 1)
      tile[s_row + 1][s_col] = d_t_old[(row + 1) * cols + col];
  }

  __syncthreads();

  // Compute only for internal compute rows (skipping halos and global edges)
  if (row > 0 && row < rows - 1 && col > 0 && col < cols - 1) {
    d_t_new[row * cols + col] =
        0.25 * (tile[s_row - 1][s_col] + tile[s_row + 1][s_col] +
                tile[s_row][s_col - 1] + tile[s_row][s_col + 1]);
  }
}

ProfilerResult SolverHybrid::Run(Grid& host_grid, int iterations) {
  ProfilerResult res;
  int num_gpus = 0;
  cudaGetDeviceCount(&num_gpus);

  if (num_gpus < 1) {
    std::cerr << "Error: No GPUs detected for Hybrid Execution.\n";
  }

  const int global_n = host_grid.cols();
  const int cols = global_n;

  // We need to pass data through the host to exchange halos between GPUs.
  double* h_t_old = const_cast<double*>(host_grid.t_old_ptr());
  double* h_t_new = const_cast<double*>(host_grid.t_new_ptr());

// OpenMP Region: Spawn one thread per GPU
#pragma omp parallel num_threads(num_gpus)
  {
    int tid = omp_get_thread_num();
    int num_threads = omp_get_num_threads();

    double local_compute_time = 0.0;
    double local_comm_time = 0.0;

    // 1. Device Binding
    cudaSetDevice(tid);

    // 2. Domain Decomposition (Partitioning ONLY the internal compute rows)
    int num_compute_rows = global_n - 2;
    int base_rows = num_compute_rows / num_threads;
    int remainder = num_compute_rows % num_threads;
    int local_rows = base_rows + (tid < remainder ? 1 : 0);
    int global_start_row =
        1 + tid * base_rows + (tid < remainder ? tid : remainder);

    int total_rows = local_rows + 2;  // +2 for Halo rows
    size_t local_bytes = total_rows * cols * sizeof(double);

    // 3. Device Memory Allocation
    double *d_t_old, *d_t_new;
    cudaMalloc(&d_t_old, local_bytes);
    cudaMalloc(&d_t_new, local_bytes);

    // Initial transfer: Copy the sub-grid + halos from Host to this specific
    // Device
    double start_comm1 = omp_get_wtime();
    int host_offset = (global_start_row - 1) * cols;
    cudaMemcpy(d_t_old, &h_t_old[host_offset], local_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_t_new, &h_t_new[host_offset], local_bytes,
               cudaMemcpyHostToDevice);
    local_comm_time += omp_get_wtime() - start_comm1;

    dim3 threads(kBlockDimX, kBlockDimY);
    dim3 blocks((cols + threads.x - 1) / threads.x,
                (total_rows + threads.y - 1) / threads.y);

    // 4. Multi-GPU Time-Stepping Loop
    for (int t = 0; t < iterations; ++t) {
      double start_compute = omp_get_wtime();
      HybridStencilKernel<<<blocks, threads>>>(d_t_old, d_t_new, total_rows,
                                               cols);
      cudaDeviceSynchronize();
      local_compute_time += omp_get_wtime() - start_compute;

      double start_comm = omp_get_wtime();
      // --- MULTI-GPU HALO EXCHANGE VIA HOST MEMORY ---
      // Copy computed boundaries back to host
      if (tid > 0) {
        // Send top compute row to host
        cudaMemcpy(&h_t_new[global_start_row * cols], &d_t_new[1 * cols],
                   cols * sizeof(double), cudaMemcpyDeviceToHost);
      }
      if (tid < num_threads - 1) {
        // Send bottom compute row to host
        cudaMemcpy(&h_t_new[(global_start_row + local_rows - 1) * cols],
                   &d_t_new[local_rows * cols], cols * sizeof(double),
                   cudaMemcpyDeviceToHost);
      }

// Synchronize all OpenMP threads to ensure Host RAM has the boundary data
#pragma omp barrier

      // Read neighbor's boundaries from host into local halos
      if (tid > 0) {
        cudaMemcpy(&d_t_new[0], &h_t_new[(global_start_row - 1) * cols],
                   cols * sizeof(double), cudaMemcpyHostToDevice);
      }
      if (tid < num_threads - 1) {
        cudaMemcpy(&d_t_new[(local_rows + 1) * cols],
                   &h_t_new[(global_start_row + local_rows) * cols],
                   cols * sizeof(double), cudaMemcpyHostToDevice);
      }
      local_comm_time += omp_get_wtime() - start_comm;

      // Swap pointers
      double* temp = d_t_old;
      d_t_old = d_t_new;
      d_t_new = temp;

// Keep host pointers in sync for the next loop's exchange
#pragma omp single
      {
        host_grid.SwapBuffers();
        h_t_old = const_cast<double*>(host_grid.t_old_ptr());
        h_t_new = host_grid.t_new_ptr();
      }
    }

    // 5. Final Extraction
    // Copy the internal compute rows back to the global host grid
    double start_comm2 = omp_get_wtime();
    cudaMemcpy(&h_t_old[global_start_row * cols], &d_t_old[1 * cols],
               local_rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
    local_comm_time += omp_get_wtime() - start_comm2;

    cudaFree(d_t_old);
    cudaFree(d_t_new);

    if (tid == 0) {
      res.compute_time = local_compute_time;
      res.comm_time = local_comm_time;
    }
  }  // End of OpenMP Parallel Region
  return res;
}

}  // namespace heat_sim