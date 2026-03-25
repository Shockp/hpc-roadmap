#include <vector>

#include "particle.h"
#include "simulation.h"

int main() {
  std::vector<physics::Particle> particles;

  physics::InitializeParticle(&particles);

  physics::RunSimulation(&particles);

  return 0;
}