#ifndef HEAT_DIFFUSION_CONFIG_H_
#define HEAT_DIFFUSION_CONFIG_H_

namespace heat_sim {

// -----------------------------------------------------------------------------
// Simulation Constants
// -----------------------------------------------------------------------------
// Top edge
constexpr double kTopBoundaryTemp = 100.0;

// Remaining edges (bottom, left, right)
constexpr double kDefaultBoundaryTemp = 0.0;

// -----------------------------------------------------------------------------
// Execution Parameters
// -----------------------------------------------------------------------------
// Structure to hold runtime parameters passed via command line arguments.
struct SimConfig {
  int grid_size;   // N (dimension of the N x N matrix)
  int iterations;  // Total number of time steps to simulate
};

}  // namespace heat_sim

#endif  // HEAT_DIFFUSION_CONFIG_H_