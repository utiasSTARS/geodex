// Tests for the geodex::planning::plan facade over concrete manifolds.

#include <cmath>
#include <cstdint>
#include <cstdlib>

#include <algorithm>
#include <atomic>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <ompl/base/DiscreteMotionValidator.h>
#include <ompl/util/Console.h>
#include <ompl/util/RandomNumbers.h>

#include "geodex/core/sampler.hpp"
#include "geodex/heuristics/zero.hpp"
#include "geodex/manifold/configuration_space.hpp"
#include "geodex/manifold/euclidean.hpp"
#include "geodex/manifold/se2.hpp"
#include "geodex/manifold/se3.hpp"
#include "geodex/manifold/so2.hpp"
#include "geodex/manifold/so3.hpp"
#include "geodex/manifold/sphere.hpp"
#include "geodex/manifold/torus.hpp"
#include "geodex/metrics/constant_spd.hpp"
#include "geodex/metrics/kinetic_energy.hpp"
#include "geodex/planning/plan.hpp"

namespace gp = geodex::planning;
namespace ob = ompl::base;

namespace {

// Straight-line distance in the ambient coordinates.
double chord(const Eigen::VectorXd& a, const Eigen::VectorXd& b) { return (a - b).norm(); }

}  // namespace

TEST(Planning, EuclideanGreedyRRTstarOptimal) {
  geodex::Euclidean<2> manifold;
  Eigen::Vector2d start(-0.8, -0.8);
  Eigen::Vector2d goal(0.8, 0.8);

  gp::PlanSettings settings;
  settings.time = 1.0;
  settings.seed = 1;
  settings.planner = gp::planners::GreedyRRTstar{};

  auto result = gp::plan(manifold, start, goal, {}, settings);

  ASSERT_TRUE(result.solved);
  const double straight = (goal - start).norm();
  EXPECT_GE(result.cost, straight - 1e-6);
  EXPECT_LT(result.cost, 1.2 * straight);
}

TEST(Planning, SphereAvoidsObstacle) {
  geodex::Sphere<> manifold;
  Eigen::Vector3d start(1.0, 0.0, 0.0);
  Eigen::Vector3d goal(0.0, 0.0, 1.0);

  // Block the direct great-circle route through the +x/+z quadrant. The plan checks the
  // obstacle grown by half the check spacing and the corner tolerance.
  std::function<bool(const Eigen::Vector3d&)> is_valid = [](const Eigen::Vector3d& q) {
    return !(q[0] > 0.3 && q[2] > 0.3);
  };
  std::function<bool(const Eigen::Vector3d&)> padded = [](const Eigen::Vector3d& q) {
    const double grown = 0.3 - (0.5 * 0.02 + 1e-4);
    return !(q[0] > grown && q[2] > grown);
  };

  gp::PlanSettings settings;
  settings.time = 3.0;
  settings.seed = 1;
  settings.collision_check_resolution = 0.02;

  auto result = gp::plan(manifold, start, goal, padded, settings);

  ASSERT_TRUE(result.solved);
  EXPECT_LT((result.path.front() - start).norm(), 1e-6);
  EXPECT_LT((result.path.back() - goal).norm(), 1e-6);
  for (const auto& q : result.path) {
    EXPECT_NEAR(q.norm(), 1.0, 1e-6);
    EXPECT_TRUE(is_valid(q));
  }
}

TEST(Planning, SE2FreeSpace) {
  geodex::SE2<> manifold;
  Eigen::Vector3d start(1.0, 1.0, 0.0);
  Eigen::Vector3d goal(8.0, 8.0, 0.0);

  gp::PlanSettings settings;
  settings.time = 2.0;
  settings.seed = 1;

  auto result = gp::plan(manifold, start, goal, {}, settings);

  ASSERT_TRUE(result.solved);
  EXPECT_LT((result.path.front() - start).norm(), 1e-6);
  EXPECT_LT((result.path.back() - goal).norm(), 1e-6);
}

TEST(Planning, EuclideanDynamicPoint) {
  geodex::Euclidean<Eigen::Dynamic> manifold(3);
  Eigen::VectorXd start(3);
  start << -0.5, -0.5, -0.5;
  Eigen::VectorXd goal(3);
  goal << 0.5, 0.5, 0.5;

  gp::PlanSettings settings;
  settings.time = 1.0;
  settings.seed = 1;
  settings.planner = gp::planners::GreedyRRTstar{};

  auto result = gp::plan(manifold, start, goal, {}, settings);

  ASSERT_TRUE(result.solved);
  ASSERT_GE(result.path.size(), 2u);
  EXPECT_LT(chord(result.path.front(), start), 1e-6);
  EXPECT_LT(chord(result.path.back(), goal), 1e-6);
  const double straight = (goal - start).norm();
  EXPECT_LT(result.cost, 1.3 * straight);
}

TEST(Planning, SeedMakesTheRunReproducible) {
  geodex::Euclidean<4> manifold;
  Eigen::Vector4d start = Eigen::Vector4d::Constant(-0.9);
  Eigen::Vector4d goal = Eigen::Vector4d::Constant(0.9);
  auto is_valid = [](const Eigen::Vector4d& q) {
    return !(std::abs(q[0]) < 0.1 && std::abs(q[1] - 0.45) > 0.2);
  };

  gp::PlanSettings settings;
  settings.iterations = 800;
  settings.seed = 7;

  auto run = [&] { return gp::plan(manifold, start, goal, is_valid, settings); };

  const auto a = run();
  const auto b = run();
  ASSERT_TRUE(a.solved);
  ASSERT_EQ(a.path.size(), b.path.size());
  EXPECT_DOUBLE_EQ(a.cost, b.cost);
  for (std::size_t i = 0; i < a.path.size(); ++i) {
    EXPECT_LT((a.path[i] - b.path[i]).norm(), 1e-15) << "waypoint " << i;
  }

  // A different seed explores differently.
  settings.seed = 99;
  const auto c = run();
  ASSERT_TRUE(c.solved);
  EXPECT_NE(a.cost, c.cost);
}

namespace {

// Two-link planar arm mass matrix, anisotropic in the elbow angle.
struct PlanarArmMass {
  Eigen::Matrix2d operator()(const Eigen::Vector2d& q) const {
    const double h = 0.5 * std::cos(q[1]);
    const double m00 = 1.0 / 12 + 1.0 / 12 + 0.25 + (1.0 + 0.25 + 2 * h);
    const double m01 = 1.0 / 12 + (0.25 + h);
    Eigen::Matrix2d m;
    m << m00, m01, m01, 1.0 / 12 + 0.25;
    return m;
  }
};

using ArmSpace =
    geodex::ConfigurationSpace<geodex::Euclidean<2>, geodex::KineticEnergyMetric<PlanarArmMass>>;

ArmSpace arm_space() {
  geodex::Euclidean<2> base;
  base.set_sampling_bounds(Eigen::Vector2d(-3.0, -3.0), Eigen::Vector2d(3.0, 3.0));
  return ArmSpace{base, geodex::KineticEnergyMetric<PlanarArmMass>{PlanarArmMass{}}};
}

// Checks that every waypoint is valid and every edge is valid along manifold.geodesic at
// `res`.
template <typename M, typename V>
int chord_violations(const M& m, const std::vector<typename M::Point>& path, const V& valid,
                     const double res) {
  int bad = 0;
  for (std::size_t k = 0; k < path.size(); ++k) {
    if (!valid(path[k])) ++bad;
    if (k + 1 == path.size()) break;
    const int n = std::max(1, static_cast<int>(std::ceil((path[k + 1] - path[k]).norm() / res)));
    for (int j = 1; j < n; ++j) {
      if (!valid(m.geodesic(path[k], path[k + 1], static_cast<double>(j) / n))) ++bad;
    }
  }
  return bad;
}

}  // namespace

// Under a custom metric, OMPL validates discrete-geodesic edges. The path that plan()
// returns must be valid along its chords. The plan checks the obstacles grown by half the
// check spacing and the corner tolerance, and its path then clears the obstacles themselves.
TEST(Planning, CurvedPlannerEdgesStillGiveAValidPath) {
  const ArmSpace space = arm_space();
  // Two walls leave a narrow gap. Chords of curved edges clip the walls.
  auto clear_by = [](const double margin) {
    return std::function<bool(const Eigen::Vector2d&)>([margin](const Eigen::Vector2d& q) {
      const bool wall = std::abs(q[0]) < 0.15 + margin && std::abs(q[1]) > 0.5 - margin;
      const bool post = (q - Eigen::Vector2d(1.2, 0.8)).norm() < 0.5 + margin;
      return !wall && !post;
    });
  };
  const auto valid = clear_by(0.0);
  const auto padded = clear_by(0.5 * 0.01 + 1e-4);
  const Eigen::Vector2d start(-2.0, 1.5), goal(2.0, -1.5);
  for (const std::uint64_t seed : {1u, 2u, 3u, 4u, 5u}) {
    gp::PlanSettings settings;
    settings.iterations = 4000;
    settings.seed = seed;
    settings.collision_check_resolution = 0.01;
    const auto r = gp::plan(space, start, goal, padded, settings, geodex::heuristics::Zero{});
    ASSERT_TRUE(r.solved) << "seed " << seed;
    EXPECT_EQ(chord_violations(space, r.path, valid, 0.001), 0) << "seed " << seed;
    EXPECT_LT((r.path.front() - start).norm(), 1e-12);
    EXPECT_LT((r.path.back() - goal).norm(), 1e-12);
  }
}

TEST(Planning, SmoothingShortensAndReportsIt) {
  const ArmSpace space = arm_space();
  std::function<bool(const Eigen::Vector2d&)> valid = [](const Eigen::Vector2d& q) {
    return (q - Eigen::Vector2d(0.0, 0.0)).norm() > 0.8;
  };
  gp::PlanSettings settings;
  settings.iterations = 1000;
  settings.seed = 3;
  settings.collision_check_resolution = 0.01;
  settings.smooth = false;
  const auto raw = gp::plan(space, Eigen::Vector2d(-2.0, 0.1), Eigen::Vector2d(2.0, -0.1), valid,
                            settings, geodex::heuristics::Zero{});
  settings.smooth = true;
  const auto smooth = gp::plan(space, Eigen::Vector2d(-2.0, 0.1), Eigen::Vector2d(2.0, -0.1), valid,
                               settings, geodex::heuristics::Zero{});
  ASSERT_TRUE(raw.solved);
  ASSERT_TRUE(smooth.solved);
  EXPECT_FALSE(raw.smoothed);
  EXPECT_TRUE(smooth.smoothed);
  EXPECT_GT(smooth.smooth_ms, 0.0);
  EXPECT_LT(smooth.cost, raw.cost);
  EXPECT_EQ(raw.raw_path.size(), smooth.raw_path.size());
}

// An explicit interpolation mode holds with a motion validator installed, and Auto
// resolves to the base geodesic.
TEST(Planning, InterpolationModeIsHonoredWithAMotionValidator) {
  using SE2 = geodex::SE2<>;
  using Space = geodex::integration::ompl::GeodexStateSpace<SE2>;
  const SE2 manifold{geodex::SE2LeftInvariantMetric{1.0, 4.0, 0.5}, Eigen::Vector3d(0, 0, -3.2),
                     Eigen::Vector3d(10, 10, 3.2)};
  std::vector<geodex::integration::ompl::InterpolationMode> seen;
  auto factory = [&](const ob::SpaceInformationPtr& si) {
    seen.push_back(si->getStateSpace()->as<Space>()->getInterpolationMode());
    return std::make_shared<ob::DiscreteMotionValidator>(si.get());
  };
  gp::PlanSettings settings;
  settings.iterations = 200;
  settings.seed = 1;
  settings.smooth = false;
  const Eigen::Vector3d start(1, 1, 0), goal(8, 8, 0);
  settings.interp = geodex::integration::ompl::InterpolationMode::RiemannianGeodesic;
  (void)gp::plan(manifold, start, goal, {}, settings, geodex::heuristics::Euclidean{}, factory);
  settings.interp = geodex::integration::ompl::InterpolationMode::Auto;
  (void)gp::plan(manifold, start, goal, {}, settings, geodex::heuristics::Euclidean{}, factory);
  ASSERT_EQ(seen.size(), 2u);
  EXPECT_EQ(seen[0], geodex::integration::ompl::InterpolationMode::RiemannianGeodesic);
  EXPECT_EQ(seen[1], geodex::integration::ompl::InterpolationMode::BaseGeodesic);
}

TEST(Planning, RefineTimeStopsSoonAfterTheFirstSolution) {
  geodex::Euclidean<2> manifold;
  gp::PlanSettings settings;
  settings.time = 5.0;
  settings.refine_time = 0.02;
  settings.seed = 4;
  settings.smooth = false;
  const auto r =
      gp::plan(manifold, Eigen::Vector2d(-0.8, -0.8), Eigen::Vector2d(0.8, 0.8), {}, settings);
  ASSERT_TRUE(r.solved);
  // Free space solves at once. The call lasts about the refinement time.
  EXPECT_LT(r.time_ms, 1000.0);
}

// ---------------------------------------------------------------------------
// Samplers and seeds
// ---------------------------------------------------------------------------

namespace {

using Scrambled = geodex::Euclidean<2>;
using Pseudo =
    geodex::Euclidean<2, geodex::EuclideanStandardMetric<2>, geodex::PseudoRandomSampler>;
using Halton = geodex::Euclidean<2, geodex::EuclideanStandardMetric<2>, geodex::HaltonSampler>;
using Runtime = geodex::Euclidean<2, geodex::EuclideanStandardMetric<2>, geodex::DynamicSampler>;

// Raw GreedyRRTstar path around a wall under a fixed iteration budget.
template <typename M>
gp::PlanResult<Eigen::Vector2d> plan_around_wall(const M& manifold, const std::uint64_t seed) {
  std::function<bool(const Eigen::Vector2d&)> valid = [](const Eigen::Vector2d& q) {
    return !(std::abs(q[0]) < 0.1 && q[1] > -0.5);
  };
  gp::PlanSettings settings;
  settings.iterations = 500;
  settings.seed = seed;
  settings.smooth = false;
  return gp::plan(manifold, Eigen::Vector2d(-0.8, -0.8), Eigen::Vector2d(0.8, 0.8), valid,
                  settings);
}

template <typename P>
bool same_path(const std::vector<P>& a, const std::vector<P>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

template <typename M>
M seeded(const std::uint64_t s) {
  M m;
  m.seed(s);
  return m;
}

}  // namespace

TEST(PlanningSampler, SamplerKindReachesThePlanner) {
  const auto scrambled = plan_around_wall(Scrambled{}, 7);
  const auto pseudo = plan_around_wall(Pseudo{}, 7);
  const auto halton = plan_around_wall(Halton{}, 7);
  ASSERT_TRUE(scrambled.solved);
  ASSERT_TRUE(pseudo.solved);
  ASSERT_TRUE(halton.solved);
  EXPECT_FALSE(same_path(scrambled.raw_path, pseudo.raw_path));
  EXPECT_FALSE(same_path(scrambled.raw_path, halton.raw_path));
  EXPECT_FALSE(same_path(pseudo.raw_path, halton.raw_path));
  // Every kind reproduces under the same seed and changes with it.
  EXPECT_TRUE(same_path(pseudo.raw_path, plan_around_wall(Pseudo{}, 7).raw_path));
  EXPECT_TRUE(same_path(halton.raw_path, plan_around_wall(Halton{}, 7).raw_path));
  EXPECT_TRUE(same_path(scrambled.raw_path, plan_around_wall(Scrambled{}, 7).raw_path));
  EXPECT_FALSE(same_path(scrambled.raw_path, plan_around_wall(Scrambled{}, 8).raw_path));
}

// The Python bindings hold a runtime sampler. It plans exactly as the compile-time
// choice of the same kind.
TEST(PlanningSampler, RuntimeSamplerInstanceMatchesTheCompileTimeKind) {
  Runtime pseudo;
  pseudo.set_sampler(geodex::DynamicSampler{geodex::PseudoRandomSampler{}});
  const auto a = plan_around_wall(Runtime{}, 7);
  const auto b = plan_around_wall(pseudo, 7);
  ASSERT_TRUE(a.solved);
  ASSERT_TRUE(b.solved);
  EXPECT_TRUE(same_path(a.raw_path, plan_around_wall(Scrambled{}, 7).raw_path));
  EXPECT_TRUE(same_path(b.raw_path, plan_around_wall(Pseudo{}, 7).raw_path));
  EXPECT_FALSE(same_path(a.raw_path, b.raw_path));
}

// Without a plan seed, each plan takes a child seed from the manifold's own sampler, one
// random_point() worth. Repeated plans are independent trials.
TEST(PlanningSampler, UnseededPlansOnOneManifoldAreIndependent) {
  const Scrambled scrambled;
  EXPECT_FALSE(
      same_path(plan_around_wall(scrambled, 0).raw_path, plan_around_wall(scrambled, 0).raw_path));
  // Even the deterministic Halton kind moves on.
  const Halton halton;
  EXPECT_FALSE(
      same_path(plan_around_wall(halton, 0).raw_path, plan_around_wall(halton, 0).raw_path));

  // Each unseeded plan advances the manifold by exactly one random_point().
  const Scrambled advanced = seeded<Scrambled>(11);
  Scrambled reference = seeded<Scrambled>(11);
  ASSERT_TRUE(plan_around_wall(advanced, 0).solved);
  (void)reference.random_point();
  EXPECT_EQ(advanced.random_point(), reference.random_point());
}

// A manifold seeded alike repeats the same sequence of distinct plans.
TEST(PlanningSampler, SeededManifoldRepeatsItsSequenceOfPlans) {
  const Scrambled a = seeded<Scrambled>(11);
  const Scrambled b = seeded<Scrambled>(11);
  const auto a1 = plan_around_wall(a, 0);
  const auto a2 = plan_around_wall(a, 0);
  const auto b1 = plan_around_wall(b, 0);
  const auto b2 = plan_around_wall(b, 0);
  ASSERT_TRUE(a1.solved);
  ASSERT_TRUE(a2.solved);
  EXPECT_TRUE(same_path(a1.raw_path, b1.raw_path));
  EXPECT_TRUE(same_path(a2.raw_path, b2.raw_path));
  EXPECT_FALSE(same_path(a1.raw_path, a2.raw_path));
  EXPECT_FALSE(same_path(a1.raw_path, plan_around_wall(seeded<Scrambled>(12), 0).raw_path));

  // A sampler instance reaches the planner with its state.
  Halton late;
  late.set_sampler(geodex::HaltonSampler{100});
  EXPECT_FALSE(
      same_path(plan_around_wall(Halton{}, 0).raw_path, plan_around_wall(late, 0).raw_path));
}

// A nonzero plan seed gives the same plan whatever state the manifold is in, keeps
// the manifold's sampler kind, and leaves the manifold untouched.
TEST(PlanningSampler, PlanSeedIgnoresTheManifoldState) {
  const Scrambled a = seeded<Scrambled>(11);
  const Scrambled b = seeded<Scrambled>(12);
  const Scrambled c = seeded<Scrambled>(11);
  for (int i = 0; i < 3; ++i) (void)c.random_point();
  const auto ra = plan_around_wall(a, 5);
  EXPECT_TRUE(same_path(ra.raw_path, plan_around_wall(b, 5).raw_path));
  EXPECT_TRUE(same_path(ra.raw_path, plan_around_wall(c, 5).raw_path));
  EXPECT_TRUE(same_path(ra.raw_path, plan_around_wall(a, 5).raw_path));
  EXPECT_EQ(a.random_point(), seeded<Scrambled>(11).random_point());
}

TEST(PlanningSampler, PlanSeedLeavesTheGlobalSeedSourceAlone) {
  const Scrambled manifold;
  geodex::set_default_seed(3);
  const std::uint64_t first = geodex::detail::seed_source()();
  geodex::set_default_seed(3);
  ASSERT_TRUE(plan_around_wall(manifold, 7).solved);
  EXPECT_EQ(geodex::detail::seed_source()(), first);
}

TEST(PlanningSampler, InformedSamplingFocusesThroughPlan) {
  const geodex::Euclidean<2> manifold;
  const Eigen::Vector2d start(-0.8, -0.8), goal(0.8, 0.8);
  gp::PlanSettings settings;
  settings.iterations = 2000;
  settings.seed = 1;
  settings.smooth = false;

  const auto greedy = gp::plan(manifold, start, goal, {}, settings);
  ASSERT_TRUE(greedy.solved);
  EXPECT_GT(greedy.informed_samples, 10 * greedy.uniform_samples);
  EXPECT_GT(greedy.focused_samples, greedy.informed_samples / 2);

  gp::planners::GreedyRRTstar plain;
  plain.greedy_ratio = 0.0;
  settings.planner = plain;
  const auto informed = gp::plan(manifold, start, goal, {}, settings);
  ASSERT_TRUE(informed.solved);
  EXPECT_GT(informed.informed_samples, 10 * informed.uniform_samples);
  EXPECT_EQ(informed.focused_samples, 0u);
}

namespace {

// A validity with a batched form that counts its block calls.
struct CountingBatchValidity {
  std::function<bool(const Eigen::Vector2d&)> valid;
  std::shared_ptr<int> blocks = std::make_shared<int>(0);

  bool operator()(const Eigen::Vector2d& q) const { return valid(q); }
  bool batch(const Eigen::Vector2d* q, const std::size_t n) const {
    ++*blocks;
    for (std::size_t i = 0; i < n; ++i) {
      if (!valid(q[i])) return false;
    }
    return true;
  }
  std::size_t batch_size() const { return 8; }
};

}  // namespace

// plan() keeps a batched validity intact. The smoother checks edge samples in blocks
// with the same result as a plain predicate, also when a motion validator decides the
// planner's edges.
TEST(PlanningSampler, BatchedValidityReachesTheSmoother) {
  const geodex::Euclidean<2> manifold;
  const std::function<bool(const Eigen::Vector2d&)> wall = [](const Eigen::Vector2d& q) {
    return !(std::abs(q[0]) < 0.1 && q[1] > -0.5);
  };
  gp::PlanSettings settings;
  settings.iterations = 500;
  settings.seed = 7;
  settings.collision_check_resolution = 0.01;
  const Eigen::Vector2d start(-0.8, -0.8), goal(0.8, 0.8);

  const CountingBatchValidity batched{wall};
  const auto a = gp::plan(manifold, start, goal, batched, settings);
  const auto b = gp::plan(manifold, start, goal, wall, settings);
  ASSERT_TRUE(a.solved);
  ASSERT_TRUE(a.smoothed);
  EXPECT_GT(*batched.blocks, 0);
  EXPECT_TRUE(same_path(a.path, b.path));

  const CountingBatchValidity with_validator{wall};
  auto factory = [](const ob::SpaceInformationPtr& si) {
    return std::make_shared<ob::DiscreteMotionValidator>(si.get());
  };
  const auto c = gp::plan(manifold, start, goal, with_validator, settings,
                          geodex::heuristics::Euclidean{}, factory);
  ASSERT_TRUE(c.solved);
  EXPECT_GT(*with_validator.blocks, 0);
}

// ---------------------------------------------------------------------------
// Lie groups and periodic manifolds
// ---------------------------------------------------------------------------

// A raw chord across the cut overestimates the wrapped distance. SO(2) and the torus
// certify a periodic bound, and informed sampling works across the cut.
TEST(PlanningLieGroups, SO2PlansThroughTheCut) {
  const geodex::SO2<> manifold;
  using P = geodex::SO2<>::Point;
  const P start{-2.5}, goal{2.5};
  const double through_cut = 2.0 * std::numbers::pi - 5.0;
  for (const std::uint64_t seed : {1u, 2u, 3u}) {
    gp::PlanSettings settings;
    settings.iterations = 1000;
    settings.seed = seed;
    settings.smooth = false;
    const auto r = gp::plan(manifold, start, goal, {}, settings);
    ASSERT_TRUE(r.solved) << "seed " << seed;
    EXPECT_GT(r.informed_samples, 0u) << "seed " << seed;
    EXPECT_LT(r.cost, through_cut + 0.01) << "seed " << seed;
    EXPECT_NEAR(r.path.front()[0], start[0], 1e-12);
    EXPECT_NEAR(r.path.back()[0], goal[0], 1e-12);
  }
}

TEST(PlanningLieGroups, TorusPlansThroughTheCut) {
  const geodex::Torus<2> manifold;
  const Eigen::Vector2d start(0.5, 3.0), goal(5.8, 3.0);
  const double through_cut = 2.0 * std::numbers::pi - 5.3;
  gp::PlanSettings settings;
  settings.iterations = 2000;
  settings.seed = 1;
  settings.smooth = false;
  const auto r = gp::plan(manifold, start, goal, {}, settings);
  ASSERT_TRUE(r.solved);
  EXPECT_GT(r.informed_samples, 0u);
  EXPECT_LT(r.cost, 1.1 * through_cut);
}

// A metric that couples the periodic axes does not wrap axis by axis. plan() shrinks
// the certified bound to an isotropic one and still plans.
TEST(PlanningLieGroups, CoupledTorusMetricStillPlansThroughTheCut) {
  Eigen::Matrix2d A;
  A << 2.0, 0.6, 0.6, 1.0;
  const geodex::Torus<2, geodex::ConstantSPDMetric<2>> manifold{geodex::ConstantSPDMetric<2>{A}};
  gp::PlanSettings settings;
  settings.iterations = 1000;
  settings.seed = 1;
  settings.smooth = false;
  const Eigen::Vector2d start(0.5, 3.0), goal(5.8, 3.0);
  gp::PlanResult<Eigen::Vector2d> r;
  ASSERT_NO_THROW(r = gp::plan(manifold, start, goal, {}, settings));
  ASSERT_TRUE(r.solved);
  EXPECT_GT(r.informed_samples, 0u);
  // Through the cut, the x offset is 2 pi - 5.3. The long way round, it is 5.3.
  EXPECT_LT(r.cost, std::sqrt(2.0) * 2.0);
}

namespace {

// Rotation by `angle` about z as a unit quaternion [x, y, z, w].
Eigen::Vector4d rot_z(const double angle) {
  return Eigen::Vector4d(0.0, 0.0, std::sin(0.5 * angle), std::cos(0.5 * angle));
}

}  // namespace

// SO(3) and SE(3) embed in more coordinates than they have dimensions. Every state the
// planner returns stays on the manifold, also around an obstacle. The plan checks the
// obstacle grown by half the check spacing and the corner tolerance.
TEST(PlanningLieGroups, SO3PlansOnTheManifoldAroundAnObstacle) {
  const geodex::SO3<> manifold;
  const Eigen::Vector4d start = rot_z(0.0), goal = rot_z(2.5), middle = rot_z(1.25);
  std::function<bool(const Eigen::Vector4d&)> valid = [&](const Eigen::Vector4d& q) {
    return manifold.distance(q, middle) > 0.4;
  };
  std::function<bool(const Eigen::Vector4d&)> padded = [&](const Eigen::Vector4d& q) {
    return manifold.distance(q, middle) > 0.4 + 0.5 * 0.02 + 1e-4;
  };
  gp::PlanSettings settings;
  settings.iterations = 1500;
  settings.seed = 2;
  settings.collision_check_resolution = 0.02;
  const auto r = gp::plan(manifold, start, goal, padded, settings);
  ASSERT_TRUE(r.solved);
  for (const auto& path : {r.raw_path, r.path}) {
    for (const auto& q : path) {
      EXPECT_NEAR(q.norm(), 1.0, 1e-9);
      EXPECT_TRUE(valid(q));
    }
  }
  EXPECT_LT(manifold.distance(r.path.front(), start), 1e-9);
  EXPECT_LT(manifold.distance(r.path.back(), goal), 1e-9);
  EXPECT_GT(r.cost, manifold.distance(start, goal));
}

TEST(PlanningLieGroups, SE3PlansOnTheManifoldAroundAnObstacle) {
  const geodex::SE3<> manifold{Eigen::Vector3d::Zero(), Eigen::Vector3d::Constant(4.0)};
  using P = geodex::SE3<>::Point;
  P start, goal;
  start << 0.5, 0.5, 0.5, rot_z(0.0);
  goal << 3.5, 3.5, 3.5, rot_z(2.0);
  // A ball of positions around the straight segment's midpoint.
  std::function<bool(const P&)> valid = [](const P& q) {
    return (q.head<3>() - Eigen::Vector3d::Constant(2.0)).norm() > 0.8;
  };
  std::function<bool(const P&)> padded = [](const P& q) {
    return (q.head<3>() - Eigen::Vector3d::Constant(2.0)).norm() > 0.8 + 0.02 + 1e-4;
  };
  gp::PlanSettings settings;
  settings.iterations = 2000;
  settings.seed = 3;
  settings.collision_check_resolution = 0.02;
  const auto r = gp::plan(manifold, start, goal, padded, settings);
  ASSERT_TRUE(r.solved);
  for (const auto& path : {r.raw_path, r.path}) {
    for (const auto& q : path) {
      EXPECT_NEAR(q.tail<4>().norm(), 1.0, 1e-9);
      EXPECT_TRUE(valid(q));
    }
  }
  EXPECT_LT(manifold.distance(r.path.front(), start), 1e-9);
  EXPECT_LT(manifold.distance(r.path.back(), goal), 1e-9);
}

// ---------------------------------------------------------------------------
// DirectionalMotionValidator
// ---------------------------------------------------------------------------

namespace {

using DiffDrive = geodex::SE2<geodex::SE2LeftInvariantMetric, geodex::SE2LeftExponentialMap>;
using DiffDriveSpace = geodex::integration::ompl::GeodexStateSpace<DiffDrive>;

DiffDrive diff_drive() {
  return DiffDrive{geodex::SE2LeftInvariantMetric{1.0, 20.0, 0.5}, geodex::SE2LeftExponentialMap{},
                   Eigen::Vector3d(0.0, 0.0, -std::numbers::pi),
                   Eigen::Vector3d(10.0, 10.0, std::numbers::pi)};
}

}  // namespace

TEST(DirectionalMotionValidator, AcceptsForwardAndRejectsReverseEdges) {
  namespace gio = geodex::integration::ompl;
  const DiffDrive manifold = diff_drive();
  ob::RealVectorBounds bounds(3);
  bounds.setLow(0, 0.0);
  bounds.setHigh(0, 10.0);
  bounds.setLow(1, 0.0);
  bounds.setHigh(1, 10.0);
  bounds.setLow(2, -std::numbers::pi);
  bounds.setHigh(2, std::numbers::pi);
  auto space = std::make_shared<DiffDriveSpace>(manifold, bounds);
  auto si = std::make_shared<ob::SpaceInformation>(space);
  si->setStateValidityChecker([](const ob::State*) { return true; });
  si->setup();
  const auto inner = std::make_shared<ob::DiscreteMotionValidator>(si.get());
  const gio::DirectionalMotionValidator<DiffDrive> strict(si.get(), manifold, inner, 0.0);
  const gio::DirectionalMotionValidator<DiffDrive> budget(si.get(), manifold, inner, 0.5);

  ob::ScopedState<DiffDriveSpace> a(space), ahead(space), behind(space), far_behind(space),
      turned(space);
  auto set = [](auto& s, double x, double y, double t) {
    s->values[0] = x;
    s->values[1] = y;
    s->values[2] = t;
  };
  // Heading pi/2 faces +y. The body-forward axis is world +y.
  set(a, 5.0, 5.0, std::numbers::pi / 2);
  set(ahead, 5.0, 6.0, std::numbers::pi / 2);
  set(behind, 5.0, 4.7, std::numbers::pi / 2);
  set(far_behind, 5.0, 4.0, std::numbers::pi / 2);
  set(turned, 5.0, 5.0, 0.0);

  EXPECT_TRUE(strict.checkMotion(a.get(), ahead.get()));
  EXPECT_FALSE(strict.checkMotion(a.get(), behind.get()));
  EXPECT_TRUE(strict.checkMotion(a.get(), turned.get()));  // pure rotation
  EXPECT_TRUE(budget.checkMotion(a.get(), behind.get()));
  EXPECT_FALSE(budget.checkMotion(a.get(), far_behind.get()));

  std::pair<ob::State*, double> last{space->allocState(), -1.0};
  EXPECT_FALSE(strict.checkMotion(a.get(), behind.get(), last));
  EXPECT_DOUBLE_EQ(last.second, 0.0);
  EXPECT_TRUE(space->equalStates(last.first, a.get()));
  space->freeState(last.first);
}

TEST(DirectionalMotionValidator, PlanDrivesEveryEdgeForward) {
  namespace gio = geodex::integration::ompl;
  const DiffDrive manifold = diff_drive();
  auto factory = [&](const ob::SpaceInformationPtr& si) {
    return std::make_shared<gio::DirectionalMotionValidator<DiffDrive>>(
        si.get(), manifold, std::make_shared<ob::DiscreteMotionValidator>(si.get()), 0.0);
  };
  // The start faces away from the goal, and a direct reverse is shorter. Every
  // interpolation mode keeps the validator on the smoother's edges too.
  const Eigen::Vector3d start(5.0, 5.0, 0.0), goal(2.0, 5.0, 0.0);
  for (const auto mode : {gio::InterpolationMode::Auto, gio::InterpolationMode::BaseGeodesic,
                          gio::InterpolationMode::RiemannianGeodesic}) {
    gp::PlanSettings settings;
    settings.iterations = 3000;
    settings.seed = 4;
    settings.interp = mode;
    const auto r =
        gp::plan(manifold, start, goal, {}, settings, geodex::heuristics::Euclidean{}, factory);
    ASSERT_TRUE(r.solved) << "mode " << static_cast<int>(mode);
    for (const auto& path : {r.raw_path, r.path}) {
      for (std::size_t i = 0; i + 1 < path.size(); ++i) {
        EXPECT_GE(manifold.log(path[i], path[i + 1])[0], -1e-9)
            << "mode " << static_cast<int>(mode) << " edge " << i;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Limits and log level
// ---------------------------------------------------------------------------

// Limits outside the manifold's own sampling box set where the planner samples. A wall
// between start and goal forces a detour that the default box [-1, 1]^2 cannot reach.
TEST(PlanningLimits, SamplesInsideLimitsOutsideTheDefaultBox) {
  const geodex::Euclidean<2> manifold;  // samples [-1, 1]^2 by default
  std::function<bool(const Eigen::Vector2d&)> valid = [](const Eigen::Vector2d& q) {
    return !(q[0] > 2.9 && q[0] < 3.1 && q[1] < 3.6);
  };
  gp::PlanSettings settings;
  settings.iterations = 1500;
  settings.seed = 3;
  settings.collision_check_resolution = 0.01;
  settings.limits.emplace(Eigen::Vector2d(2.0, 2.0), Eigen::Vector2d(4.0, 4.0));
  const auto r =
      gp::plan(manifold, Eigen::Vector2d(2.2, 2.2), Eigen::Vector2d(3.8, 2.2), valid, settings);
  ASSERT_TRUE(r.solved);
  // A sampled check keeps every point of a checked edge out of the wall shrunk by half the
  // check spacing.
  const auto clear = [](const Eigen::Vector2d& q) {
    return !(q[0] > 2.905 && q[0] < 3.095 && q[1] < 3.595);
  };
  for (const auto& path : {r.raw_path, r.path}) {
    for (const auto& q : path) {
      EXPECT_GE(q.minCoeff(), 2.0 - 1e-12);
      EXPECT_LE(q.maxCoeff(), 4.0 + 1e-12);
      EXPECT_TRUE(clear(q));
    }
  }
}

// plan() rejects limits that are not finite.
TEST(PlanningLimits, RejectsLimitsThatAreNotFinite) {
  const geodex::Euclidean<2> manifold;
  gp::PlanSettings settings;
  settings.iterations = 100;
  const double inf = std::numeric_limits<double>::infinity();
  settings.limits.emplace(Eigen::Vector2d(-inf, -1.0), Eigen::Vector2d(inf, 1.0));
  EXPECT_THROW((void)gp::plan(manifold, Eigen::Vector2d(-0.5, 0.0), Eigen::Vector2d(0.5, 0.0), {},
                              settings),
               std::invalid_argument);
}

// plan() spaces the returned waypoints evenly along the smoothed path, at most one hundredth of
// its coordinate length apart. The problem is the first plan of the docs.
TEST(Planning, ReturnsEvenlySpacedWaypoints) {
  const double pi = std::numbers::pi;
  const geodex::SE2<> space{geodex::SE2LeftInvariantMetric{1.0, 100.0, 1.0},
                            geodex::SE2LeftExponentialMap{}, Eigen::Vector3d(0.0, 0.0, -pi),
                            Eigen::Vector3d(4.0, 4.0, pi)};
  const std::function<bool(const Eigen::Vector3d&)> valid = [](const Eigen::Vector3d& q) {
    return std::hypot(q[0] - 2.0, q[1] - 2.0) > 0.8;
  };
  const Eigen::Vector3d start(0.5, 2.0, 0.0), goal(3.5, 2.0, 0.0);
  gp::PlanSettings settings;
  settings.iterations = 2000;
  settings.seed = 1;
  const auto r = gp::plan(space, start, goal, valid, settings);
  ASSERT_TRUE(r.solved);
  ASSERT_TRUE(r.smoothed);
  double length = 0.0;
  for (std::size_t k = 1; k < r.path.size(); ++k) {
    length += space.log(r.path[k - 1], r.path[k]).norm();
  }
  double lo = std::numeric_limits<double>::infinity(), hi = 0.0;
  for (std::size_t k = 1; k < r.path.size(); ++k) {
    const double step = space.log(r.path[k - 1], r.path[k]).norm();
    lo = std::min(lo, step);
    hi = std::max(hi, step);
  }
  EXPECT_GE(r.path.size(), 10u);
  EXPECT_LT(hi - lo, 1e-2 * hi);
  EXPECT_LT((r.path.front() - start).norm(), 1e-12);
  EXPECT_LT((r.path.back() - goal).norm(), 1e-12);
}

// The result reports when the first exact solution arrived, within the search time.
TEST(PlanningTiming, ReportsTheFirstSolution) {
  const geodex::Euclidean<2> manifold;
  std::function<bool(const Eigen::Vector2d&)> valid = [](const Eigen::Vector2d& q) {
    return !(std::abs(q[0]) < 0.1 && q[1] < 0.3);
  };
  gp::PlanSettings settings;
  settings.iterations = 800;
  settings.seed = 5;
  const auto r =
      gp::plan(manifold, Eigen::Vector2d(-0.8, -0.8), Eigen::Vector2d(0.8, -0.8), valid, settings);
  ASSERT_TRUE(r.solved);
  EXPECT_GE(r.first_solution_ms, 0.0);
  EXPECT_LE(r.first_solution_ms, r.time_ms);
  EXPECT_GE(r.first_solution_iterations, 1u);
  EXPECT_LE(r.first_solution_iterations, settings.iterations);

  const auto none = gp::plan(manifold, Eigen::Vector2d(-0.8, -0.8), Eigen::Vector2d(0.8, -0.8),
                             std::function<bool(const Eigen::Vector2d&)>(
                                 [](const Eigen::Vector2d& q) { return std::abs(q[0]) > 0.5; }),
                             settings);
  EXPECT_FALSE(none.solved);
  EXPECT_EQ(none.first_solution_ms, -1.0);
  EXPECT_EQ(none.first_solution_iterations, 0u);
}

// Physical limits bound the search, and the smoother keeps the path inside them.
TEST(PlanningLimits, PathStaysInsideTheLimits) {
  const geodex::Euclidean<2> manifold;  // samples [-1, 1]^2
  std::function<bool(const Eigen::Vector2d&)> valid = [](const Eigen::Vector2d& q) {
    return !(std::abs(q[0]) < 0.1 && q[1] < 0.3);
  };
  gp::PlanSettings settings;
  settings.iterations = 1500;
  settings.seed = 3;
  settings.collision_check_resolution = 0.01;
  settings.limits.emplace(Eigen::Vector2d(-1.0, -1.0), Eigen::Vector2d(1.0, 0.5));
  const auto r =
      gp::plan(manifold, Eigen::Vector2d(-0.8, -0.8), Eigen::Vector2d(0.8, -0.8), valid, settings);
  ASSERT_TRUE(r.solved);
  for (const auto& path : {r.raw_path, r.path}) {
    for (const auto& q : path) EXPECT_LE(q[1], 0.5 + 1e-12);
  }
  settings.limits.emplace(Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones());
  EXPECT_THROW(
      gp::plan(manifold, Eigen::Vector2d(-0.8, -0.8), Eigen::Vector2d(0.8, -0.8), valid, settings),
      std::invalid_argument);
}

namespace {

// Records OMPL's messages by level.
class RecordingOutput : public ompl::msg::OutputHandler {
 public:
  void log(const std::string& text, ompl::msg::LogLevel level, const char* /*filename*/,
           int /*line*/) override {
    ++count[level];
    if (level == ompl::msg::LOG_INFO) info.push_back(text);
  }
  std::map<ompl::msg::LogLevel, int> count;
  std::vector<std::string> info;
};

}  // namespace

// plan() prints only warnings and errors by default, lets a caller ask for more, and
// leaves OMPL's own level as it found it.
TEST(PlanningLogLevel, QuietByDefaultAndAdjustable) {
  RecordingOutput out;
  ompl::msg::useOutputHandler(&out);
  ompl::msg::setLogLevel(ompl::msg::LOG_DEBUG);
  const geodex::Euclidean<2> manifold;
  gp::PlanSettings settings;
  settings.iterations = 200;
  settings.seed = 1;
  auto run = [&] {
    (void)gp::plan(manifold, Eigen::Vector2d(-0.8, -0.8), Eigen::Vector2d(0.8, 0.8), {}, settings);
  };
  // GEODEX_LOG_LEVEL sets the starting level, Warn without it.
  if (std::getenv("GEODEX_LOG_LEVEL") == nullptr) EXPECT_EQ(gp::log_level(), gp::LogLevel::Warn);
  gp::set_log_level(gp::LogLevel::Warn);
  run();
  EXPECT_EQ(out.count[ompl::msg::LOG_INFO] + out.count[ompl::msg::LOG_DEBUG], 0);
  EXPECT_EQ(ompl::msg::getLogLevel(), ompl::msg::LOG_DEBUG);

  gp::set_log_level(gp::LogLevel::Info);
  run();
  EXPECT_GT(out.count[ompl::msg::LOG_INFO], 0);
  EXPECT_EQ(out.count[ompl::msg::LOG_DEBUG], 0);

  gp::set_log_level(gp::LogLevel::Warn);
  ompl::msg::restorePreviousOutputHandler();
  ompl::msg::setLogLevel(ompl::msg::LOG_INFO);
}

// At Info, a plan prints its first-solution, refinement and smoothing times.
TEST(PlanningLogLevel, InfoPrintsThePlanTimes) {
  RecordingOutput out;
  ompl::msg::useOutputHandler(&out);
  const geodex::Euclidean<2> manifold;
  gp::PlanSettings settings;
  settings.iterations = 200;
  settings.seed = 1;
  gp::set_log_level(gp::LogLevel::Info);
  const auto r =
      gp::plan(manifold, Eigen::Vector2d(-0.8, -0.8), Eigen::Vector2d(0.8, 0.8), {}, settings);
  gp::set_log_level(gp::LogLevel::Warn);
  ompl::msg::restorePreviousOutputHandler();
  ompl::msg::setLogLevel(ompl::msg::LOG_INFO);
  ASSERT_TRUE(r.solved);
  const auto line = std::find_if(out.info.begin(), out.info.end(), [](const std::string& t) {
    return t.find("geodex plan: first solution") != std::string::npos;
  });
  ASSERT_NE(line, out.info.end());
  EXPECT_NE(line->find("refinement"), std::string::npos);
  EXPECT_NE(line->find("smoothing"), std::string::npos);
}

#ifndef _WIN32
// GEODEX_LOG_LEVEL names the starting level. An unknown or missing name gives Warn.
TEST(PlanningLogLevel, EnvironmentNamesTheStartingLevel) {
  const std::pair<const char*, gp::LogLevel> cases[] = {
      {"debug", gp::LogLevel::Debug}, {"info", gp::LogLevel::Info}, {"warn", gp::LogLevel::Warn},
      {"error", gp::LogLevel::Error}, {"off", gp::LogLevel::Off},   {"loud", gp::LogLevel::Warn},
      {"INFO", gp::LogLevel::Info},   {"Off", gp::LogLevel::Off}};
  for (const auto& [name, level] : cases) {
    ::setenv("GEODEX_LOG_LEVEL", name, 1);
    EXPECT_EQ(gp::detail::environment_log_level(), level) << name;
  }
  ::unsetenv("GEODEX_LOG_LEVEL");
  EXPECT_EQ(gp::detail::environment_log_level(), gp::LogLevel::Warn);
}
#endif

// A cap of one neighbor gives each added state a single candidate parent. The tree and
// its path around the wall differ from those of the unbounded neighborhood.
TEST(Planning, MaxNeighborsReachesThePlanner) {
  const geodex::Euclidean<2> manifold;
  const std::function<bool(const Eigen::Vector2d&)> valid = [](const Eigen::Vector2d& q) {
    return !(std::abs(q[0]) < 0.1 && q[1] > -0.5);
  };
  auto run = [&](const unsigned int cap) {
    gp::PlanSettings settings;
    settings.iterations = 1000;
    settings.seed = 2;
    settings.smooth = false;
    gp::planners::GreedyRRTstar cfg;
    cfg.max_neighbors = cap;
    settings.planner = cfg;
    return gp::plan(manifold, Eigen::Vector2d(-0.8, -0.8), Eigen::Vector2d(0.8, 0.8), valid,
                    settings);
  };
  const auto capped = run(1);
  const auto unbounded = run(0);
  ASSERT_TRUE(capped.solved);
  ASSERT_TRUE(unbounded.solved);
  EXPECT_FALSE(same_path(capped.raw_path, unbounded.raw_path));
  EXPECT_NE(capped.cost, unbounded.cost);
  EXPECT_TRUE(same_path(capped.raw_path, run(1).raw_path));
}

// plan() checks a start that already meets the goal, both the start itself and the edge
// to the goal.
TEST(Planning, StartAtTheGoalStillChecksValidity) {
  const geodex::Euclidean<2> manifold;
  const std::function<bool(const Eigen::Vector2d&)> wall = [](const Eigen::Vector2d& q) {
    return std::abs(q[0]) > 0.01;
  };
  gp::PlanSettings settings;
  settings.iterations = 100;
  settings.seed = 1;
  EXPECT_FALSE(
      gp::plan(manifold, Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(0.0, 0.0), wall, settings)
          .solved);
  const auto free =
      gp::plan(manifold, Eigen::Vector2d(0.5, 0.0), Eigen::Vector2d(0.5, 0.0), wall, settings);
  EXPECT_TRUE(free.solved);
  EXPECT_EQ(free.path.size(), 2u);
  settings.goal_tolerance = 0.2;
  EXPECT_FALSE(
      gp::plan(manifold, Eigen::Vector2d(-0.05, 0.0), Eigen::Vector2d(0.05, 0.0), wall, settings)
          .solved);
}

// plan() rejects a resolution that is negative, not finite, or finer than
// kMaxEdgeSamples checks across the bounds.
TEST(Planning, CollisionResolutionIsValidated) {
  const geodex::Euclidean<2> manifold;
  gp::PlanSettings settings;
  settings.iterations = 50;
  settings.seed = 1;
  for (const double r : {-1.0, std::numeric_limits<double>::quiet_NaN(), 1e-9}) {
    settings.collision_check_resolution = r;
    EXPECT_THROW((void)gp::plan(manifold, Eigen::Vector2d(-0.5, 0.0), Eigen::Vector2d(0.5, 0.0), {},
                                settings),
                 std::invalid_argument)
        << "resolution " << r;
  }
}

namespace {

using AnisotropicSE2 = geodex::SE2<>;

AnisotropicSE2 anisotropic_se2() {
  return AnisotropicSE2{geodex::SE2LeftInvariantMetric{1.0, 50.0, 2.0},
                        Eigen::Vector3d(0.0, 0.0, -3.14159265),
                        Eigen::Vector3d(10.0, 10.0, 3.14159265)};
}

// A plan around a wall on SE(2) with unequal weights. Its distance breaks the
// triangle inequality, and the pivots that OMPL's nearest-neighbor structure samples
// change the plan.
std::vector<Eigen::Vector3d> anisotropic_plan(const AnisotropicSE2& manifold,
                                              const std::uint64_t seed,
                                              const unsigned int iterations) {
  const std::function<bool(const Eigen::Vector3d&)> valid = [](const Eigen::Vector3d& q) {
    return !(q[0] > 4.0 && q[0] < 6.0 && q[1] < 7.0);
  };
  gp::PlanSettings settings;
  settings.iterations = iterations;
  settings.seed = seed;
  settings.smooth = false;
  return gp::plan(manifold, Eigen::Vector3d(1.0, 1.0, 0.0), Eigen::Vector3d(9.0, 1.0, 0.0), valid,
                  settings)
      .raw_path;
}

}  // namespace

// A manifold seeded alike repeats its sequence of unseeded plans whatever OMPL's
// process-wide generator drew before, as in another process.
TEST(PlanningSampler, SeededManifoldRepeatsItsPlansWhateverOmplDrewBefore) {
  auto sequence = [](const int rngs_before) {
    for (int i = 0; i < rngs_before; ++i) (void)ompl::RNG{};
    AnisotropicSE2 manifold = anisotropic_se2();
    manifold.seed(7);
    std::vector<std::vector<Eigen::Vector3d>> paths;
    for (int i = 0; i < 3; ++i) paths.push_back(anisotropic_plan(manifold, 0, 2000));
    return paths;
  };
  const auto first = sequence(0);
  const auto second = sequence(5);
  for (int i = 0; i < 3; ++i) {
    ASSERT_FALSE(first[i].empty()) << "plan " << i;
    EXPECT_TRUE(same_path(first[i], second[i])) << "plan " << i;
  }
}

// Seeded plans running in several threads, each on its own manifold, give the plan
// run alone, and OMPL's log level is the caller's again once they finish.
TEST(PlanningConcurrency, SeededPlansInThreadsMatchThePlanRunAlone) {
  constexpr int kThreads = 8;
  constexpr int kRounds = 4;
  constexpr unsigned int kIterations = 1500;
  const ompl::msg::LogLevel before = ompl::msg::getLogLevel();
  ompl::msg::setLogLevel(ompl::msg::LOG_ERROR);
  const auto alone = anisotropic_plan(anisotropic_se2(), 5, kIterations);
  ASSERT_FALSE(alone.empty());
  EXPECT_EQ(ompl::msg::getLogLevel(), ompl::msg::LOG_ERROR);

  std::vector<std::vector<Eigen::Vector3d>> out(kThreads * kRounds);
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&out, t] {
      for (int r = 0; r < kRounds; ++r) {
        out[static_cast<std::size_t>(t * kRounds + r)] =
            anisotropic_plan(anisotropic_se2(), 5, kIterations);
      }
    });
  }
  for (auto& th : threads) th.join();
  for (std::size_t i = 0; i < out.size(); ++i) {
    EXPECT_TRUE(same_path(out[i], alone)) << "plan " << i;
  }
  EXPECT_EQ(ompl::msg::getLogLevel(), ompl::msg::LOG_ERROR);
  ompl::msg::setLogLevel(before);
}

// State samplers allocated in other threads, each of which takes a seed from OMPL's
// shared generator, leave a seeded plan as it is when it runs alone.
TEST(PlanningConcurrency, SamplerAllocationsInOtherThreadsLeaveSeededPlansAlone) {
  namespace gio = geodex::integration::ompl;
  constexpr unsigned int kIterations = 1500;
  const auto alone = anisotropic_plan(anisotropic_se2(), 5, kIterations);
  ASSERT_FALSE(alone.empty());
  ob::RealVectorBounds bounds(3);
  bounds.setLow(0.0);
  bounds.setHigh(10.0);
  bounds.setLow(2, -std::numbers::pi);
  bounds.setHigh(2, std::numbers::pi);
  std::atomic<bool> stop{false};
  std::vector<std::thread> allocators;
  for (int t = 0; t < 4; ++t) {
    allocators.emplace_back([&] {
      const auto space =
          std::make_shared<gio::GeodexStateSpace<AnisotropicSE2>>(anisotropic_se2(), bounds);
      while (!stop.load()) (void)space->allocDefaultStateSampler();
    });
  }
  for (int k = 0; k < 4; ++k) {
    EXPECT_TRUE(same_path(anisotropic_plan(anisotropic_se2(), 5, kIterations), alone)) << k;
  }
  stop.store(true);
  for (auto& th : allocators) th.join();
}

// Many short plans overlapping in time leave OMPL's log level as the caller set it.
TEST(PlanningConcurrency, LogLevelIsRestoredAfterOverlappingPlans) {
  constexpr int kThreads = 8;
  constexpr int kPlans = 200;
  const ompl::msg::LogLevel before = ompl::msg::getLogLevel();
  ompl::msg::setLogLevel(ompl::msg::LOG_ERROR);
  const geodex::Euclidean<2> manifold;
  gp::PlanSettings settings;
  settings.seed = 1;
  settings.iterations = 10;
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&] {
      for (int k = 0; k < kPlans; ++k) {
        // Start and goal coincide. Each plan enters and leaves at once.
        (void)gp::plan(manifold, Eigen::Vector2d(0.1, 0.2), Eigen::Vector2d(0.1, 0.2), {},
                       settings);
      }
    });
  }
  for (auto& th : threads) th.join();
  EXPECT_EQ(ompl::msg::getLogLevel(), ompl::msg::LOG_ERROR);
  ompl::msg::setLogLevel(before);
}
