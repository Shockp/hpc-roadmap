#include "solver_mpi.h"

#include <mpi.h>

#include "grid.h"

namespace heat_sim {

DomainDecomposition::DomainDecomposition(int n, int mpi_rank, int mpi_size)
    : global_n(n), rank(mpi_rank), size(mpi_size) {
  // Calculate base rows per process and the remainder
  int base_rows = n / size;
  int remainder = n % size;

  // Distribute the remainder across the first 'remainder' ranks
  if (rank < remainder) {
    local_rows = base_rows + 1;
    global_start_row = rank * local_rows;
  } else {
    local_rows = base_rows;
    global_start_row = rank * local_rows + remainder;
  }

  // Add 2 extra rows for the Top and Bottom Halos (Ghost Cells)
  total_rows = local_rows + 2;

  // Indentify if this rank touches the global boundaries
  is_top_rank = (rank == 0);
  is_bottom_rank = (rank == size - 1);
}

void SolverMpi::Run(int global_n, int iterations) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // 1. Setup Distributed Domain
  DomainDecomposition domain(global_n, rank, size);

  // Allocate the local grid including the extra 2 rows for the halos.
  Grid grid(domain.total_rows, global_n, domain.is_top_rank,
            domain.is_bottom_rank);

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

    // Exchange with the Top Neighbor (Tag 0 for upward, 1 for downward)
    if (!domain.is_top_rank) {
      MPI_Sendrecv(send_top, cols, MPI_DOUBLE, rank - 1, 0, recv_top, cols,
                   MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Exchange with the Bottom Neighbor (Tag 1 for downward, Tag 0 for upward)
    if (!domain.is_bottom_rank) {
      MPI_Sendrecv(send_bottom, cols, MPI_DOUBLE, rank + 1, 1, recv_bottom,
                   cols, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
    }

    // --- COMPUTATION ---
    // Compute the stencil strictly for the internal compute rows.
    // i starts at 1 (skipping top halo) and ends at local_rows
    // (skipping bottom halo).
    for (int i = 1; i <= domain.local_rows; ++i) {
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

    // --- SWAP BUFFERS ---
    grid.SwapBuffers();
  }
}

}  // namespace heat_sim