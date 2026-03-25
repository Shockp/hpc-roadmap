#ifndef PHYSICS_ENGINE_PARTICLE_H_
#define PHYSICS_ENGINE_PARTICLE_H_

namespace physics {

struct Particle {
  double x, y;    // Position
  double vx, vy;  // Velocity
  double mass;
};

}  // namespace physics

#endif