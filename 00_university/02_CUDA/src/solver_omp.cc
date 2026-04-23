#include "solver_omp.h"

#include <omp.h>

namespace heat_sim {

void SolverOmp::Run(Grid& grid, int iterations) {
  const int n = grid.n();
  constexpr double kInvFour = 0.25;

// OPTIMIZATION: Open the parallel region ONCE outside the time loop.
// This eliminates the overhead of thread creation/destruction at every step.
#pragma omp parallel shared(grid, n, iterations)
  {
    // Thread-local pointers to avoid continuous getter calls
    const double* local_t_old = grid.t_old_ptr();
    double* local_t_new = grid.t_new_ptr();

    for (int t = 0; t < iterations; ++t) {
// Distribute the row iterations across threads.
// We use schedule(static) as the workload per cell is uniform, preventing
// the runtime overhead asociated with schefule(dynamic).
#pragma omp for schedule(static)
      for (int i = 1; i < n - 1; ++i) {
        for (int j = 1; j < n - 1; ++j) {
          const int center = i * n + j;
          const int top = (i - 1) * n + j;
          const int bottom = (i + 1) * n + j;
          const int left = i * n + (j - 1);
          const int right = i * n + (j + 1);

          local_t_new[center] = (local_t_old[top] + local_t_old[bottom] +
                                 local_t_old[left] + local_t_old[right]) *
                                kInvFour;
        }
      }

// The #pragma omp single block ensures only ONE thread swaps the underlying
// double buffer pointers. Crucially, it contains an implicit barrier at the
// end, ensuring all thread have finished computing the current strep before
// moving on.
#pragma omp single
      {
        grid.SwapBuffers();
      }

      // Update thread-local pointers for the next iteration step.
      local_t_old = grid.t_old_ptr();
      local_t_new = grid.t_new_ptr();
    }
  }
}

}  // namespace heat_sim