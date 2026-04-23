#ifndef HEAT_DIFFUSION_SOLVER_SEQ_H_
#define HEAT_DIFFUSION_SOLVER_SEQ_H_

#include "grid.h"

namespace heat_sim {

class SolverSeq {
 public:
  // Executes the sequential simulation for a given number of iterations.
  // Passing the grid by reference to mutate its state directly.
  static void Run(Grid& grid, int iterations);

 private:
  // Performs a single time step update over the entire matrix using
  // the 5-point stencil formulation.
  static void ComputeNextState(Grid& grid);
};

}  // namespace heat_sim

#endif  // HEAT_DIFFUSION_SOLVER_SEQ_H_