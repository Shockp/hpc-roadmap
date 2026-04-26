#include "solver_seq.h"

namespace heat_sim {

void SolverSeq::Run(Grid& grid, int iterations) {
  for (int t = 0; t < iterations; ++t) {
    ComputeNextState(grid);
    // Buffer swap happens after the entire grid is updated for the current step
    grid.SwapBuffers();
  }
}

void SolverSeq::ComputeNextState(Grid& grid) {
  const int n = grid.rows();

  // Extracting raw pointers prevents the overhead of calling t_old_ptr()
  // and vector::operator[] millions of times inside the tight loop.
  const double* t_old = grid.t_old_ptr();
  double* t_new = grid.t_new_ptr();

  // Optimization: CPU floating-point multiplication is significantly faster
  // than floating-point division. We multiply by 0.25 instead of dividing by 4.
  constexpr double kInvFour = 0.25;

  // The stencil is only applied to internal points.
  // The boundary rows (0 and n-1) and columns (0 and n-1)
  // to preserve their constant initialization conditions.
  for (int i = 1; i < n - 1; ++i) {
    for (int j = 1; j < n - 1; ++j) {
      // Pre-calculate indices to avoid redundant math in the innermost loop
      const int center = grid.Index(i, j);
      const int top = grid.Index(i - 1, j);
      const int bottom = grid.Index(i + 1, j);
      const int left = grid.Index(i, j - 1);
      const int right = grid.Index(i, j + 1);

      // Core 5-point stencil arithmetic
      t_new[center] =
          (t_old[top] + t_old[bottom] + t_old[left] + t_old[right]) * kInvFour;
    }
  }
}

}  // namespace heat_sim