#include "grid.h"

#include "config.h"

namespace heat_sim {

// Sequential and OpenMP compatibility
Grid::Grid(int n) : Grid(n, n, true, true) {}

// MPI compatibility
Grid::Grid(int rows, int cols, bool is_global_top, bool is_global_bottom)
    : rows_(rows),
      cols_(cols),
      is_global_top_(is_global_top),
      is_global_bottom_(is_global_bottom),
      t_old_(rows * cols, kDefaultBoundaryTemp),
      t_new_(rows * cols, kDefaultBoundaryTemp) {
  InitializeBoundaries();
}

void Grid::SwapBuffers() {
  // Exchanges the internal pointers of the vectors. No data is copied.
  t_old_.swap(t_new_);
}

void Grid::InitializeBoundaries() {
  // Only apply the 100.0 top boundary if this grid contains the global top
  // edge. Otherwise, the top row is just a halo row initialized to 0.0.
  if (is_global_top_) {
    for (int j = 0; j < cols_; ++j) {
      t_old_[Index(0, j)] = kTopBoundaryTemp;
      t_new_[Index(0, j)] = kTopBoundaryTemp;
    }
  }
}

}  // namespace heat_sim