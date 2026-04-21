#include "grid.h"

namespace heat_sim {

Grid::Grid(int n)
    : n_(n),
      // Initialize both buffers to the default sink temperature (0.0)
      t_old_(n * n, kDefaultBoundaryTemp),
      t_new_(n * n, kDefaultBoundaryTemp) {
  InitializeBoundaries();
}

void Grid::SwapBuffers() {
  // Exchanges the internal pointers of the vectors. No data is copied.
  t_old_.swap(t_new_);
}

void Grid::InitializeBoundaries() {
  // Set the top row (i = 0) to 100.0 for both buffers.
  // The bottom, left, and right edges remain 0.0 as initialized by the
  // constructor.
  for (int j = 0; j < n_; ++j) {
    t_old_[Index(0, j)] = kTopBoundaryTemp;
    t_new_[Index(0, j)] = kTopBoundaryTemp;
  }
}

}  // namespace heat_sim