#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <geodex/geodex.hpp>
#include <numbers>

using namespace geodex;

static Eigen::Vector3d point_at_theta(double theta) {
  return Eigen::Vector3d(std::sin(theta), 0.0, std::cos(theta));
}

// ---------------------------------------------------------------------------
// Round metric tests
// ---------------------------------------------------------------------------

class InterpolationRoundTest : public ::testing::Test {
 protected:
  Sphere<> sphere;
  Eigen::Vector3d north{0.0, 0.0, 1.0};
};

TEST_F(InterpolationRoundTest, ConvergesToTarget) {
  auto target = point_at_theta(1.0);
  auto r = discrete_geodesic(sphere, north, target);

  EXPECT_EQ(r.status, InterpolationStatus::Converged);
  ASSERT_GE(r.path.size(), 2u);
  double final_dist = sphere.distance(r.path.back(), target);
  EXPECT_LT(final_dist, 1e-3);
}

TEST_F(InterpolationRoundTest, PathOnSphere) {
  auto target = point_at_theta(1.0);
  auto r = discrete_geodesic(sphere, north, target);

  for (const auto& p : r.path) {
    EXPECT_NEAR(p.norm(), 1.0, 1e-10);
  }
}

TEST_F(InterpolationRoundTest, PathLength) {
  auto target = point_at_theta(1.0);
  auto r = discrete_geodesic(sphere, north, target);

  double total_length = 0.0;
  for (size_t i = 1; i < r.path.size(); ++i) {
    total_length += distance_midpoint(sphere, r.path[i - 1], r.path[i]);
  }

  double expected = sphere.distance(north, target);
  EXPECT_NEAR(total_length, expected, 0.05);
}

TEST_F(InterpolationRoundTest, Antipodal) {
  // At the cut locus, log returns zero, so we report CutLocus status and
  // return a single-point path. The distance between start and target is
  // still pi, but log can't give us a direction.
  Eigen::Vector3d south(0.0, 0.0, -1.0);
  auto r = discrete_geodesic(sphere, north, south);

  EXPECT_EQ(r.status, InterpolationStatus::CutLocus);
  ASSERT_EQ(r.path.size(), 1u);
  EXPECT_NEAR((r.path[0] - north).norm(), 0.0, 1e-12);
}

// ---------------------------------------------------------------------------
// Anisotropic metric tests
// ---------------------------------------------------------------------------

TEST(InterpolationAnisotropic, ConvergesToTarget) {
  Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
  A(0, 0) = 4.0;
  Sphere<2, ConstantSPDMetric<3>> sphere{ConstantSPDMetric<3>{A}};

  Eigen::Vector3d north(0.0, 0.0, 1.0);
  auto target = point_at_theta(0.8);

  auto r = discrete_geodesic(sphere, north, target);
  EXPECT_EQ(r.status, InterpolationStatus::Converged);
  ASSERT_GE(r.path.size(), 2u);

  double final_dist = distance_midpoint(sphere, r.path.back(), target);
  EXPECT_LT(final_dist, 1e-3);
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

TEST_F(InterpolationRoundTest, IdenticalPoints) {
  auto r = discrete_geodesic(sphere, north, north);

  EXPECT_EQ(r.status, InterpolationStatus::DegenerateInput);
  ASSERT_EQ(r.path.size(), 1u);
  EXPECT_NEAR((r.path[0] - north).norm(), 0.0, 1e-12);
}

TEST_F(InterpolationRoundTest, RespectsMaxSteps) {
  auto target = point_at_theta(2.5);

  InterpolationSettings settings;
  settings.max_steps = 3;
  settings.step_size = 0.1;

  auto r = discrete_geodesic(sphere, north, target, settings);

  // Path should have at most max_steps + 1 points (start + up to max_steps steps).
  EXPECT_LE(static_cast<int>(r.path.size()), settings.max_steps + 1);
}

// ---------------------------------------------------------------------------
// Status reporting
// ---------------------------------------------------------------------------

TEST_F(InterpolationRoundTest, ReportsConvergedOnSuccess) {
  auto target = point_at_theta(1.0);
  auto r = discrete_geodesic(sphere, north, target);

  EXPECT_EQ(r.status, InterpolationStatus::Converged);
  EXPECT_GT(r.iterations, 0);
  EXPECT_GT(r.initial_distance, 0.5);
  EXPECT_LT(r.final_distance, 1e-3);
  EXPECT_EQ(r.distortion_halvings, 0);
}

TEST_F(InterpolationRoundTest, ReportsMaxStepsOnTightBudget) {
  // Walk across ~2.5 rad with step_size 0.1 needs ~25 steps; give only 2.
  auto target = point_at_theta(2.5);
  InterpolationSettings settings;
  settings.max_steps = 2;
  settings.step_size = 0.1;

  auto r = discrete_geodesic(sphere, north, target, settings);

  EXPECT_EQ(r.status, InterpolationStatus::MaxStepsReached);
  EXPECT_EQ(r.iterations, 2);
  EXPECT_GT(r.final_distance, settings.convergence_tol);
}

TEST_F(InterpolationRoundTest, ReportsCutLocusOnAntipodal) {
  Eigen::Vector3d south(0.0, 0.0, -1.0);
  auto r = discrete_geodesic(sphere, north, south);

  EXPECT_EQ(r.status, InterpolationStatus::CutLocus);
  EXPECT_EQ(r.iterations, 0);
  ASSERT_EQ(r.path.size(), 1u);
}

TEST_F(InterpolationRoundTest, ReportsDegenerateOnIdenticalInput) {
  auto r = discrete_geodesic(sphere, north, north);

  EXPECT_EQ(r.status, InterpolationStatus::DegenerateInput);
  EXPECT_EQ(r.iterations, 0);
  EXPECT_EQ(r.initial_distance, 0.0);
  EXPECT_EQ(r.final_distance, 0.0);
  ASSERT_EQ(r.path.size(), 1u);
}

// ---------------------------------------------------------------------------
// Monotone distance decrease
// ---------------------------------------------------------------------------

TEST_F(InterpolationRoundTest, MonotoneDistanceDecrease) {
  auto target = point_at_theta(1.5);
  InterpolationSettings settings;
  settings.step_size = 0.2;
  auto r = discrete_geodesic(sphere, north, target, settings);

  ASSERT_GE(r.path.size(), 2u);
  double prev_dist = sphere.distance(r.path.front(), target);
  for (size_t i = 1; i < r.path.size(); ++i) {
    double cur_dist = sphere.distance(r.path[i], target);
    // Allow 1% slack for numerical noise.
    EXPECT_LE(cur_dist, prev_dist * 1.01 + 1e-9)
        << "Distance increased from " << prev_dist << " to " << cur_dist << " at step " << i;
    prev_dist = cur_dist;
  }
}

// ---------------------------------------------------------------------------
// Non-Riemannian retraction — projection retract on the sphere
// ---------------------------------------------------------------------------

TEST(InterpolationNonRiemannian, SphereProjectionRetractionConverges) {
  // Projection retraction is first-order, not the true exp map. The log-fast-path
  // will still make progress (strict monotone decrease), but may be less accurate
  // per step than the true exp map.
  using SphereProj = Sphere<2, SphereRoundMetric, SphereProjectionRetraction>;
  SphereProj sphere;
  Eigen::Vector3d north(0.0, 0.0, 1.0);
  Eigen::Vector3d target(std::sin(1.0), 0.0, std::cos(1.0));

  InterpolationSettings settings;
  settings.step_size = 0.3;
  settings.max_steps = 200;

  auto r = discrete_geodesic(sphere, north, target, settings);
  EXPECT_EQ(r.status, InterpolationStatus::Converged);
  ASSERT_GE(r.path.size(), 2u);

  // Final point should be close to target.
  const double final_dist = sphere.distance(r.path.back(), target);
  EXPECT_LT(final_dist, 1e-2);

  // All points should remain on the unit sphere.
  for (const auto& p : r.path) {
    EXPECT_NEAR(p.norm(), 1.0, 1e-10);
  }
}

TEST(InterpolationNonRiemannian, SE2EulerRetractionConverges) {
  // Euler retraction treats SE(2) as R^2 x S^1 — ignores group structure, but the
  // log-fast-path should still reach target under strict monotone decrease.
  using SE2Euler = SE2<SE2LeftInvariantMetric, SE2EulerRetraction>;
  SE2Euler se2;
  Eigen::Vector3d start(1.0, 1.0, 0.0);
  Eigen::Vector3d target(3.0, 3.0, 0.5);

  InterpolationSettings settings;
  settings.step_size = 0.3;
  auto r = discrete_geodesic(se2, start, target, settings);
  EXPECT_EQ(r.status, InterpolationStatus::Converged);
  EXPECT_LT(r.final_distance, 1e-2);
}

TEST(InterpolationNonRiemannian, SE2AnisotropicWeights) {
  // Anisotropic weights on SE(2) make the Lie group exp/log disagree with the
  // Riemannian geodesic, but the algorithm should still reach the target.
  SE2<SE2LeftInvariantMetric> se2(SE2LeftInvariantMetric{1.0, 1.0, 5.0});
  Eigen::Vector3d start(1.0, 1.0, 0.0);
  Eigen::Vector3d target(4.0, 3.0, 1.0);

  InterpolationSettings settings;
  settings.step_size = 0.3;
  auto r = discrete_geodesic(se2, start, target, settings);
  EXPECT_EQ(r.status, InterpolationStatus::Converged);
  EXPECT_LT(r.final_distance, 1e-2);
}

// ---------------------------------------------------------------------------
// Dynamic vs fixed-size manifolds
// ---------------------------------------------------------------------------

TEST(InterpolationDynamic, TorusDynamicMatchesFixed) {
  Eigen::Vector2d start_fixed(0.5, 0.5);
  Eigen::Vector2d target_fixed(2.5, 2.5);
  Eigen::VectorXd start_dyn(2), target_dyn(2);
  start_dyn << 0.5, 0.5;
  target_dyn << 2.5, 2.5;

  InterpolationSettings settings;
  settings.step_size = 0.3;

  Torus<2> torus_fixed;
  auto r_fixed = discrete_geodesic(torus_fixed, start_fixed, target_fixed, settings);

  Torus<Eigen::Dynamic> torus_dyn(2);
  auto r_dyn = discrete_geodesic(torus_dyn, start_dyn, target_dyn, settings);

  ASSERT_EQ(r_fixed.path.size(), r_dyn.path.size());
  for (size_t i = 0; i < r_fixed.path.size(); ++i) {
    EXPECT_LT((r_fixed.path[i] - r_dyn.path[i]).norm(), 1e-12);
  }
  EXPECT_EQ(r_fixed.status, r_dyn.status);
  EXPECT_EQ(r_fixed.iterations, r_dyn.iterations);
}

// ---------------------------------------------------------------------------
// ConfigurationSpace with KineticEnergyMetric (constant mass matrix)
// ---------------------------------------------------------------------------

TEST(InterpolationConfigSpace, TorusConstantKineticEnergyConverges) {
  // Constant mass matrix = diag(2, 1). The KE metric is flat (point-independent),
  // so the base log is still the Riemannian log of the KE metric (same geodesics).
  // The fast path should apply and the walk should reach the target.
  auto mass_matrix_fn = [](const Eigen::Vector2d& /*q*/) {
    Eigen::Matrix2d M;
    M << 2.0, 0.0, 0.0, 1.0;
    return M;
  };
  auto ke = KineticEnergyMetric{mass_matrix_fn};
  ConfigurationSpace cspace{Torus<2>{}, std::move(ke)};

  Eigen::Vector2d start(0.5, 0.5);
  Eigen::Vector2d target(2.5, 2.5);

  InterpolationSettings settings;
  settings.step_size = 0.3;
  auto r = discrete_geodesic(cspace, start, target, settings);
  EXPECT_EQ(r.status, InterpolationStatus::Converged);
  EXPECT_LT(r.final_distance, 1e-2);

  // For a constant mass matrix, the geodesic is still linear in base coordinates.
  // Check that intermediate path points lie approximately on the line segment.
  for (const auto& p : r.path) {
    // Parameterize: start + t * (target - start), solve for t.
    Eigen::Vector2d delta = target - start;
    double t = (p - start).dot(delta) / delta.squaredNorm();
    Eigen::Vector2d on_line = start + t * delta;
    EXPECT_LT((p - on_line).norm(), 1e-2);
  }
}

// ---------------------------------------------------------------------------
// Workspace reuse
// ---------------------------------------------------------------------------

TEST_F(InterpolationRoundTest, WorkspaceReuseProducesIdenticalResults) {
  auto target = point_at_theta(1.0);
  auto r_no_ws = discrete_geodesic(sphere, north, target);

  InterpolationCache<Sphere<>> ws;
  auto r_with_ws = discrete_geodesic(sphere, north, target, {}, &ws);

  ASSERT_EQ(r_no_ws.path.size(), r_with_ws.path.size());
  for (size_t i = 0; i < r_no_ws.path.size(); ++i) {
    EXPECT_LT((r_no_ws.path[i] - r_with_ws.path[i]).norm(), 1e-12);
  }
  EXPECT_EQ(r_no_ws.status, r_with_ws.status);
  EXPECT_EQ(r_no_ws.iterations, r_with_ws.iterations);

  // Reuse across multiple calls — should still produce identical results.
  auto r_reused = discrete_geodesic(sphere, north, target, {}, &ws);
  ASSERT_EQ(r_with_ws.path.size(), r_reused.path.size());
}
