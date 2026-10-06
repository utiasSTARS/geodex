/// @file test_smoothing_vamp.cpp
/// @brief plan() on a kinetic-energy Panda returns a path whose waypoints and chords all
/// pass the VAMP collision checker.

#include <cmath>

#include <functional>
#include <string>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "geodex/heuristics/matrix_lower_bound.hpp"
#include "geodex/integration/vamp/registry.hpp"
#include "geodex/manifold/configuration_space.hpp"
#include "geodex/manifold/euclidean.hpp"
#include "geodex/metrics/kinetic_energy.hpp"
#include "geodex/planning/plan.hpp"
#include "geodex/robots/mass_lower_bound.hpp"
#include "geodex/robots/mass_matrix.hpp"

namespace gp = geodex::planning;
namespace vi = geodex::integration::vamp;
using MM = geodex::robots::MassMatrix<geodex::robots::Robot::Panda>;
using Vec7 = Eigen::Matrix<double, 7, 1>;

// The plan checks the scene grown by half the sphere travel between two checks and by the
// travel of the corner tolerance. The returned path then clears the scene itself.
TEST(SmoothingVamp, KineticEnergyArmPathHoldsAlongChords) {
  constexpr double kRes = 0.01;
  auto env =
      vi::load_scene(std::string(GEODEX_TEST_FIXTURES_DIR) + "/smoothing/shelf_post.scene.yaml");
  auto checker = vi::make_vamp_checker("panda", env);
  const double speed = vi::sphere_speed("panda", env);
  auto grown = vi::make_vamp_checker("panda", vi::pad_scene(env, speed * (0.5 * kRes + 1e-4)));
  geodex::Euclidean<7> base;
  const auto [lo, hi] = MM::joint_limits();
  base.set_sampling_bounds(lo, hi);
  const geodex::ConfigurationSpace<geodex::Euclidean<7>, geodex::KineticEnergyMetric<MM>> arm{
      base, geodex::KineticEnergyMetric<MM>{MM{}}};
  std::function<bool(const Vec7&)> valid = [&](const Vec7& q) {
    return checker->is_valid(q.data(), 7);
  };
  std::function<bool(const Vec7&)> padded = [&](const Vec7& q) {
    return grown->is_valid(q.data(), 7);
  };
  const geodex::heuristics::MatrixLowerBound<7> heuristic{
      geodex::robots::MassLowerBound<geodex::robots::Robot::Panda>::matrix()};
  Vec7 start, goal;
  start << 0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785;
  goal << 1.2, -0.4, -0.3, -1.9, 0.2, 1.9, 0.5;
  for (const std::uint64_t seed : {1u, 2u, 3u, 4u}) {
    gp::PlanSettings settings;
    settings.iterations = 1000;
    settings.seed = seed;
    settings.collision_check_resolution = kRes;
    const auto r = gp::plan(arm, start, goal, padded, settings, heuristic);
    ASSERT_TRUE(r.solved) << "seed " << seed;
    EXPECT_TRUE(r.smoothed) << "seed " << seed;
    int bad = 0;
    for (std::size_t k = 0; k < r.path.size(); ++k) {
      if (!valid(r.path[k])) ++bad;
      if (k + 1 == r.path.size()) break;
      const int n = std::max(
          1, static_cast<int>(std::ceil((r.path[k + 1] - r.path[k]).norm() / (0.1 * kRes))));
      for (int j = 1; j < n; ++j) {
        if (!valid(arm.geodesic(r.path[k], r.path[k + 1], static_cast<double>(j) / n))) ++bad;
      }
    }
    EXPECT_EQ(bad, 0) << "seed " << seed;
  }
}
