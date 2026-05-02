#ifndef HEAT_DIFFUSION_PROFILER_H_
#define HEAT_DIFFUSION_PROFILER_H_

namespace heat_sim {

struct ProfilerResult {
  double setup_time = 0.0;
  double compute_time = 0.0;
  double comm_time = 0.0;
  double total_time = 0.0;
};

}  // namespace heat_sim

#endif  // HEAT_DIFFUSION_PROFILER_H_
