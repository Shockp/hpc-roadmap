#ifndef HEAT_DIFFUSION_GRID_H_
#define HEAT_DIFFUSION_GRID_H_

#include <vector>

#include "config.h"

namespace heat_sim {

class Grid {
 public:
  // Constructor allocates the 1D vectors and applies boundary conditions.
  explicit Grid(int n);

  // Constructor for MPI Sub-grids with halo rows.
  Grid(int rows, int cols, bool is_global_top, bool is_global_bottom);

  // Swaps the underlying pointers of the double buffers in O(1) time.
  // This satisfies the requirement to avoid deep copies during time steps.
  void SwapBuffers();

  // Inline helper for 2D to 1D index mapping.
  // Marked inline to avoid function call overhead in tight loops.
  inline int Index(int i, int j) const { return i * cols_ + j; }

  // Accessors
  int rows() const { return rows_; }
  int cols() const { return cols_; }
  bool is_global_top() const { return is_global_top_; }
  bool is_global_bottom() const { return is_global_bottom_; }

  const std::vector<double>& t_old() const { return t_old_; }
  std::vector<double>& t_new() { return t_new_; }

  // Expose raw pointers
  const double* t_old_ptr() const { return t_old_.data(); }
  double* t_new_ptr() { return t_new_.data(); }

 private:
  int rows_;
  int cols_;
  bool is_global_top_;
  bool is_global_bottom_;
  std::vector<double> t_old_;
  std::vector<double> t_new_;

  // Helper to apply the initial heat source and sinks
  void InitializeBoundaries();
};

}  // namespace heat_sim

#endif  // HEAT_DIFFUSION_GRID_H_