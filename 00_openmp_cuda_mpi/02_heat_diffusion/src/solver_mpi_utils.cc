#include "solver_mpi.h"

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

}  // namespace heat_sim
