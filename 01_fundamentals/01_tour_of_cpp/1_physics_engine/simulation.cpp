#include "simulation.h"

#include <iostream>
#include <random>

#include "constants.h"
#include "particle.h"

namespace physics {

void InitializeParticle(std::vector<Particle> *particles) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> pos_dist(0.0, kBoxSize);
  std::uniform_real_distribution<> vel_dist(-5.0, 5.0);
  std::uniform_real_distribution<> mass_dist(1.0, 5.0);

  particles->reserve(kNumParticles);

  for (int i = 0; i < kNumParticles; ++i) {
    particles->push_back({pos_dist(gen), pos_dist(gen), vel_dist(gen),
                          vel_dist(gen), mass_dist(gen)});
  }
}

double CalculateTotalEnergy(const std::vector<Particle> &particles) {
  double total_energy = 0.0;
  for (const auto &particle : particles) {
    double v_squared = particle.vx * particle.vx * particle.vy * particle.vy;
    total_energy += 0.5 * particle.mass * v_squared;
  }

  return total_energy;
}

void RunSimulation(std::vector<Particle> *particles) {
  std::cout << "Starting Simulation..." << std::endl;

  for (int step = 1; step <= kNumSteps; ++step) {
    for (auto &particle : *particles) {
      particle.x += particle.vx * kDt;
      particle.y += particle.vy * kDt;

      if (particle.x < 0.0) {
        particle.x = 0;
        particle.vx = -particle.vx;
      } else if (particle.x > kBoxSize) {
        particle.x = kBoxSize;
        particle.vx = -particle.vx;
      }

      if (particle.y < 0.0) {
        particle.y = 0;
        particle.vy = -particle.vy;
      } else if (particle.y > kBoxSize) {
        particle.y = kBoxSize;
        particle.vy = -particle.vy;
      }
    }

    if (step % kPrintInterval == 0) {
      double energy = CalculateTotalEnergy(*particles);
      std::cout << "Step " << step << " | Total Energy: " << energy
                << std::endl;
    }
  }

  std::cout << "Simulation Complete." << std::endl;
}

}  // namespace physics