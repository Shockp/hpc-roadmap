#ifndef HEAT_DIFFUSION_SOLVER_HYBRID_H_
#define HEAT_DIFFUSION_SOLVER_HYBRID_H_

#include "grid.h"
#include "profiler.h"

namespace heat_sim {

class SolverHybrid {
 public:
  // Executes the multi-GPU simulation using OpenMP to manage devices.
  static ProfilerResult Run(Grid& host_grid, int iterations);
};

}  // namespace heat_sim

#endif  // HEAT_DIFFUSION_SOLVER_HYBRID_H_