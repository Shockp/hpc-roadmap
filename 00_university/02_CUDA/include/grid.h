#ifndef HEAT_DIFFUSION_GRID_H_
#define HEAT_DIFFUSION_GRID_H_

#include <vector>

#include "config.h"

namespace heat_sim {

class Grid {
 public:
  // Constructor allocates the 1D vectors and applies boundary conditions.
  explicit Grid(int n);

  // Swaps the underlying pointers of the double buffers in O(1) time.
  // This satisfies the requirement to avoid deep copies during time steps.
  void SwapBuffers();

  // Inline helper for 2D to 1D index mapping.
  // Marked inline to avoid function call overhead in tight loops.
  inline int Index(int i, int j) const { return i * n_ + j; }

  // Accessors
  int n() const { return n_; }
  const std::vector<double>& t_old() const { return t_old_; }
  std::vector<double>& t_new() { return t_new_; }

  // Expose raw pointers
  const double* t_old_ptr() const { return t_old_.data(); }
  double* t_new_ptr() { return t_new_.data(); }

 private:
  int n_;
  std::vector<double> t_old_;
  std::vector<double> t_new_;

  // Helper to apply the initial heat source and sinks
  void InitializeBoundaries();
};

}  // namespace heat_sim

#endif  // HEAT_DIFFUSION_GRID_H_