#ifndef PHYSICS_ENGINE_SIMULATION_H_
#define PHYSICS_ENGINE_SIMULATION_H_

#include <vector>

#include "particle.h"

namespace physics {

void InitializeParticle(std::vector<Particle> *particles);
double CalculateTotalEnergy(const std::vector<Particle> &particles);
void RunSimulation(std::vector<Particle> *particles);

}  // namespace physics

#endif