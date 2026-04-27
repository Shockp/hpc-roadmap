#ifndef HEAT_DIFFUSION_SOLVER_CUDA_H_
#define HEAT_DIFFUSION_SOLVER_CUDA_H_

#include "grid.h"

namespace heat_sim {

class SolverCuda {
 public:
  // Executes the simulation on the GPU.
  // The host grid is passed in, but the solver handles all
  // Host-to-Device and Device-to-Host transfers internally.
  static void Run(Grid& host_grid, int iteration);
};

}  // namespace heat_sim

#endif