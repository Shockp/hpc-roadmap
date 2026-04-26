#ifndef HEAT_DIFFUSION_SOLVER_MPI_H_
#define HEAT_DIFFUSION_SOLVER_MPI_H_

#include "profiler.h"

namespace heat_sim {

// Encapsulates the math for splitting the global grid among MPI processes.
struct DomainDecomposition {
  int global_n;
  int rank;
  int size;

  int local_rows;        // Number of compute rows for this process
  int total_rows;        // Compute rows + 2 halo rows
  int global_start_row;  // The global Y-coordinate this sub-grid starts at

  bool is_top_rank;
  bool is_bottom_rank;

  // Calculates the distribution layout for the current MPI process.
  DomainDecomposition(int n, int mpi_rank, int mpi_size);
};

class SolverMpi {
 public:
  // Executes the simulation using blocking MPI communications.
  static ProfilerResult RunBlocking(int global_n, int iteration);

  // Executes the simulation using non-blocking MPI communications.
  static ProfilerResult RunNonBlocking(int global_n, int iteration);
};

}  // namespace heat_sim

#endif  // HEAT_DIFFUSION_SOLVER_MPI_H_