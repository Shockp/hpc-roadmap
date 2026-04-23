#ifndef HEAT_DIFFUSION_SOLVER_OMP_H_
#define HEAT_DIFFUSION_SOLVER_OMP_H_

#include "grid.h"

namespace heat_sim {

class SolverOmp {
 public:
  // Executes the multicore simulation using OpenMP.
  static void Run(Grid& grid, int iterations);
};

}  // namespace heat_sim

#endif  // HEAT_DIFFUSION_SOLVER_OMP_H_