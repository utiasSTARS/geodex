// Plans a path on the plane through the installed geodex package.

#include <cstdio>

#include <Eigen/Core>

#include "geodex/manifold/euclidean.hpp"
#include "geodex/planning/plan.hpp"

int main() {
  namespace gp = geodex::planning;
  geodex::Euclidean<2> manifold;
  gp::PlanSettings settings;
  settings.time = 0.5;
  settings.seed = 1;
  const auto result =
      gp::plan(manifold, Eigen::Vector2d(-0.8, -0.8), Eigen::Vector2d(0.8, 0.8), {}, settings);
  if (!result.solved) {
    std::fprintf(stderr, "plan failed\n");
    return 1;
  }
  std::printf("planned %zu waypoints, cost %.4f\n", result.path.size(), result.cost);
  return 0;
}
