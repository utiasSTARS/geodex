// Plans with the installed geodex package and checks that it links the geodex OMPL fork.
// When the package has them, it builds a VAMP checker for every robot and counts the robot
// dynamics.

#include <cmath>
#include <cstdio>
#include <memory>

#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/planners/rrt/GreedyRRTstar.h>

#include "geodex/manifold/euclidean.hpp"
#include "geodex/planning/plan.hpp"

#ifdef CHECK_HAS_VAMP
#include "geodex/integration/vamp/registry.hpp"
#endif
#ifdef CHECK_HAS_ROBOTS
#include "geodex/robots/mass_matrix.hpp"
#endif

int main() {
  namespace gp = geodex::planning;
  geodex::Euclidean<2> manifold;
  gp::PlanSettings settings;
  settings.time = 0.5;
  settings.seed = 1;
  settings.planner = gp::planners::GreedyRRTstar{};
  const auto result =
      gp::plan(manifold, Eigen::Vector2d(-0.8, -0.8), Eigen::Vector2d(0.8, 0.8), {}, settings);
  if (!result.solved) {
    std::fprintf(stderr, "plan failed\n");
    return 1;
  }
  std::printf("planned %zu waypoints, cost %.4f\n", result.path.size(), result.cost);

  // The installed Eigen includes the unsupported modules.
  const Eigen::Matrix2d rotation = (Eigen::Matrix2d() << 0.0, -1.0, 1.0, 0.0).finished().exp();
  if (std::abs(rotation(0, 0) - std::cos(1.0)) > 1e-12) {
    std::fprintf(stderr, "unsupported/Eigen/MatrixFunctions gave a wrong exponential\n");
    return 1;
  }
  std::printf("installed Eigen includes the unsupported modules\n");

  auto space = std::make_shared<ompl::base::RealVectorStateSpace>(2);
  space->setBounds(0.0, 1.0);
  auto si = std::make_shared<ompl::base::SpaceInformation>(space);
  si->setStateValidityChecker([](const ompl::base::State*) { return true; });
  ompl::geometric::GreedyRRTstar planner(si);
  if (!planner.params().hasParam("max_neighbors")) {
    std::fprintf(stderr, "linked OMPL is not the geodex OMPL fork\n");
    return 1;
  }
  std::printf("linked OMPL is the geodex OMPL fork\n");

#ifdef CHECK_HAS_VAMP
  namespace gv = geodex::integration::vamp;
  const auto env = gv::build_scene(gv::make_scene_builder());
  for (const auto& name : gv::registered_robots()) {
    if (!gv::make_vamp_checker(name, env)) {
      std::fprintf(stderr, "no VAMP checker for %s\n", name.c_str());
      return 1;
    }
  }
  std::printf("VAMP checkers: %zu robots\n", gv::registered_robots().size());
#endif
#ifdef CHECK_HAS_ROBOTS
  std::printf("robot dynamics: %zu robots\n", geodex::robots::registered_robots().size());
#endif
  return 0;
}
