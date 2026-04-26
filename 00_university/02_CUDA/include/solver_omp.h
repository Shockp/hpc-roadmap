#ifndef HEAT_DIFFUSION_SOLVER_OMP_H_
#define HEAT_DIFFUSION_SOLVER_OMP_H_

#include "grid.h"
#include "profiler.h"

namespace heat_sim {

class SolverOmp {
 public:
  // Executes the multicore simulation using OpenMP.
  static ProfilerResult Run(Grid& grid, int iterations);
};

}  // namespace heat_sim

#endif  // HEAT_DIFFUSION_SOLVER_OMP_H_