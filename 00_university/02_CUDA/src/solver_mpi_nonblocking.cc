#include <mpi.h>

#include <chrono>

#include "grid.h"
#include "solver_mpi.h"

namespace heat_sim {

ProfilerResult SolverMpi::RunNonBlocking(int global_n, int iterations) {
  ProfilerResult res;
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto start_setup = std::chrono::high_resolution_clock::now();
  // 1. Setup Distributed Domain
  DomainDecomposition domain(global_n, rank, size);

  // Allocate the local grid including the extra 2 rows for the halos.
  Grid grid(domain.total_rows, global_n, domain.is_top_rank,
            domain.is_bottom_rank);
  auto end_setup = std::chrono::high_resolution_clock::now();
  res.setup_time +=
      std::chrono::duration<double>(end_setup - start_setup).count();

  const int cols = global_n;
  constexpr double kInvFour = 0.25;

  // 2. Main Simulation loop
  for (int t = 0; t < iterations; ++t) {
    // We cast away const for the receive buffers, MPI needs to write the halo
    // data directly into the t_old buffer before we compute the stencil.
    double* t_old = const_cast<double*>(grid.t_old_ptr());
    double* t_new = grid.t_new_ptr();

    // --- HALO EXCHANGE (BLOCKING) ---
    // Memory mapping:
    // Row 0: Top Halo (Receive from rank - 1)
    // Row 1: Top Compute Row (Send to rank - 1)
    // Row local_rows: Bottom Compute Row (Send to rank + 1)
    // Row local_rows + 1: Bottom Halo (Receive from rank + 1)

    double* recv_top = &t_old[0 * cols];
    double* send_top = &t_old[1 * cols];
    double* send_bottom = &t_old[domain.local_rows * cols];
    double* recv_bottom = &t_old[(domain.local_rows + 1) * cols];

    // Array to hold the asynchronous network request handles
    MPI_Request requests[4];
    int req_count = 0;

    // --- 1. INITIATE NON-BLOCKING COMMUNICATIONS ---
    auto start_comm1 = std::chrono::high_resolution_clock::now();
    // Exchange with the Top Neighbor (Tag 0 for upward, 1 for downward)
    if (!domain.is_top_rank) {
      MPI_Irecv(recv_top, cols, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD,
                &requests[req_count++]);
      MPI_Isend(send_top, cols, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD,
                &requests[req_count++]);
    }

    // Exchange with the Bottom Neighbor (Tag 1 for downward, Tag 0 for upward)
    if (!domain.is_bottom_rank) {
      MPI_Irecv(recv_bottom, cols, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD,
                &requests[req_count++]);
      MPI_Isend(send_bottom, cols, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD,
                &requests[req_count++]);
    }
    auto end_comm1 = std::chrono::high_resolution_clock::now();
    res.comm_time +=
        std::chrono::duration<double>(end_comm1 - start_comm1).count();

    // --- 2. ASYNCHRONOUS COMPUTATION (LATENCY HIDING) ---
    // Compute the INNER grid. These rows (2 to local_rows - 1) only depend on
    // local data, so we can calculate them while the network is busy.
    auto start_compute1 = std::chrono::high_resolution_clock::now();
    for (int i = 2; i < domain.local_rows; ++i) {
      for (int j = 1; j < cols - 1; ++j) {
        const int center = i * cols + j;
        const int top = (i - 1) * cols + j;
        const int bottom = (i + 1) * cols + j;
        const int left = i * cols + (j - 1);
        const int right = i * cols + (j + 1);

        t_new[center] =
            (t_old[top] + t_old[bottom] + t_old[left] + t_old[right]) *
            kInvFour;
      }
    }
    auto end_compute1 = std::chrono::high_resolution_clock::now();
    res.compute_time +=
        std::chrono::duration<double>(end_compute1 - start_compute1).count();

    // --- 3. SYNCHRONIZATION ---
    // The CPU has finished the inner grid. Now we MUST wait for the network to
    // finish delivering the halo rows before we computing our local boundaries.
    auto start_comm2 = std::chrono::high_resolution_clock::now();
    if (req_count > 0) {
      MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    }
    auto end_comm2 = std::chrono::high_resolution_clock::now();
    res.comm_time +=
        std::chrono::duration<double>(end_comm2 - start_comm2).count();

    // --- 4. BOUNDARY COMPUTATION ---
    // Now that halos are safely in memory, compute the To compute row (i = 1)
    auto start_compute2 = std::chrono::high_resolution_clock::now();
    if (domain.local_rows >= 1) {
      int i = 1;
      for (int j = 1; j < cols - 1; ++j) {
        const int center = i * cols + j;
        const int top = (i - 1) * cols + j;
        const int bottom = (i + 1) * cols + j;
        const int left = i * cols + (j - 1);
        const int right = i * cols + (j + 1);

        t_new[center] =
            (t_old[top] + t_old[bottom] + t_old[left] + t_old[right]) *
            kInvFour;
      }
    }

    // Compute the Bottom compute row (i = local_rows)
    // We check > 1 to ensure we don't calculate the same row twice
    // if local_rows == 1.
    if (domain.local_rows > 1) {
      int i = domain.local_rows;
      for (int j = 1; j < cols - 1; ++j) {
        const int center = i * cols + j;
        const int top = (i - 1) * cols + j;
        const int bottom = (i + 1) * cols + j;
        const int left = i * cols + (j - 1);
        const int right = i * cols + (j + 1);

        t_new[center] =
            (t_old[top] + t_old[bottom] + t_old[left] + t_old[right]) *
            kInvFour;
      }
    }
    auto end_compute2 = std::chrono::high_resolution_clock::now();
    res.compute_time +=
        std::chrono::duration<double>(end_compute2 - start_compute2).count();

    // --- SWAP BUFFERS ---
    grid.SwapBuffers();
  }
  return res;
}

}  // namespace heat_sim