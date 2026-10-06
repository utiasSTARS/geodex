#include <cmath>

#include <numbers>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "geodex/geodex.hpp"
#include "geodex/metrics/clearance.hpp"

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
  // At the cut locus log returns zero. The walk reports CutLocus and returns a
  // single-point path.
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

  // The path has at most max_steps + 1 points, the start and up to max_steps steps.
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
  // A walk across about 2.5 rad with step_size 0.1 needs about 25 steps. Allow only 2.
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
// Non-Riemannian retraction (projection retraction on the sphere)
// ---------------------------------------------------------------------------

TEST(InterpolationNonRiemannian, SphereProjectionRetractionConverges) {
  // The projection retraction is first-order, not the true exp map. The log fast path
  // still decreases the distance strictly at every step.
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
  // The Euler retraction treats SE(2) as R^2 x S^1 and ignores the group structure. The
  // log fast path still reaches the target with a strict monotone decrease.
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
  // Anisotropic weights on SE(2) make the Lie group exp and log disagree with the
  // Riemannian geodesic. The walk still reaches the target.
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
  // The constant mass matrix diag(2, 1) gives a flat KE metric. The base log is the
  // Riemannian log of the KE metric, the fast path applies, and the walk reaches the
  // target.
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

  // For a constant mass matrix the geodesic is linear in base coordinates. Intermediate
  // path points lie close to the line segment.
  for (const auto& p : r.path) {
    // Solve start + t * (target - start) for t.
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

  // Reusing the workspace across calls gives identical results.
  auto r_reused = discrete_geodesic(sphere, north, target, {}, &ws);
  ASSERT_EQ(r_with_ws.path.size(), r_reused.path.size());
}

// ---------------------------------------------------------------------------
// Midpoint FD surrogate with runtime guard
// ---------------------------------------------------------------------------

// Arc cost, the sum of per-segment Riemannian norms along the path under the
// manifold's metric.
template <typename M>
static double arc_cost(const M& m, const std::vector<typename M::Point>& path) {
  double sum = 0.0;
  for (size_t i = 1; i < path.size(); ++i) {
    const auto v = m.log(path[i - 1], path[i]);
    sum += m.norm(path[i - 1], v);
  }
  return sum;
}

TEST(InterpolationGuardedMidpoint, SE2SDFConformalConverges) {
  // SE(2) with a mild SDFConformalMetric over a circular obstacle. The midpoint FD
  // (default) and the via-log FD (tau=0) both converge, and the fallback counter
  // reports correctly.
  auto sdf = [](const Eigen::Vector3d& q) {
    const double r = std::sqrt(q[0] * q[0] + q[1] * q[1]);
    return r - 1.0;  // unit circle at origin, positive outside
  };

  SE2LeftInvariantMetric base_metric{1.0, 1.0, 0.5};
  SE2<SE2LeftInvariantMetric, SE2LeftExponentialMap> se2{base_metric};
  SDFConformalMetric clearance_metric{base_metric, sdf, 2.0, 2.0};
  ConfigurationSpace cspace{se2, clearance_metric};

  // The path passes above the obstacle. The straight line only grazes the high-c region.
  const Eigen::Vector3d start(-2.0, 1.5, 0.0);
  const Eigen::Vector3d target(2.0, 1.5, 0.0);

  InterpolationSettings midpoint_settings;
  midpoint_settings.step_size = 0.2;
  midpoint_settings.max_steps = 200;
  midpoint_settings.convergence_tol = 1e-3;
  // default fd_midpoint_guard_tau = 0.25

  InterpolationSettings vialog_settings = midpoint_settings;
  vialog_settings.fd_midpoint_guard_tau = 0.0;  // force via-log on every sample

  auto r_midpoint = discrete_geodesic(cspace, start, target, midpoint_settings);
  auto r_vialog = discrete_geodesic(cspace, start, target, vialog_settings);

  EXPECT_EQ(r_midpoint.status, InterpolationStatus::Converged);
  EXPECT_EQ(r_vialog.status, InterpolationStatus::Converged);

  // SE(2) with SE2LeftExponentialMap gives v_ma + v_mb = 0 exactly by the group midpoint
  // identity. The default-tau guard does not trip.
  EXPECT_EQ(r_midpoint.fd_midpoint_fallbacks, 0);

  // Every FD sample trips under tau=0, and the counter is nonzero.
  EXPECT_GT(r_vialog.fd_midpoint_fallbacks, 0);
}

TEST(InterpolationGuardedMidpoint, IdenticalResultsOnRiemannianLog) {
  // When the base log is the Riemannian log of the metric, the walk takes the fast path
  // and does not run the FD path. The midpoint guard does not change the result.
  Sphere<> sphere;
  const Eigen::Vector3d north(0.0, 0.0, 1.0);
  const Eigen::Vector3d target = point_at_theta(1.0);

  InterpolationSettings default_settings;

  InterpolationSettings tau_zero = default_settings;
  tau_zero.fd_midpoint_guard_tau = 0.0;

  auto r_default = discrete_geodesic(sphere, north, target, default_settings);
  auto r_tau_zero = discrete_geodesic(sphere, north, target, tau_zero);

  EXPECT_EQ(r_default.status, InterpolationStatus::Converged);
  EXPECT_EQ(r_tau_zero.status, InterpolationStatus::Converged);

  // Only the fast path runs. The fallback counter stays at zero for every tau.
  EXPECT_EQ(r_default.fd_midpoint_fallbacks, 0);
  EXPECT_EQ(r_tau_zero.fd_midpoint_fallbacks, 0);

  ASSERT_EQ(r_default.path.size(), r_tau_zero.path.size());
  for (size_t i = 0; i < r_default.path.size(); ++i) {
    EXPECT_LT((r_default.path[i] - r_tau_zero.path[i]).norm(), 1e-12);
  }
}

namespace {

// Length of a path of SE(2) poses under the unit-weight metric, which is the flat
// metric of R^2 x S^1 in coordinates.
double flat_se2_length(const std::vector<Eigen::Vector3d>& path) {
  double len = 0.0;
  for (std::size_t k = 1; k < path.size(); ++k) {
    const Eigen::Vector3d d(path[k][0] - path[k - 1][0], path[k][1] - path[k - 1][1],
                            utils::wrap_to_pi(path[k][2] - path[k - 1][2]));
    len += d.norm();
  }
  return len;
}

// Length of a path of SE(3) poses under the unit-weight metric, the translation's
// arc length combined with the rotation angle of each step.
double flat_se3_length(const std::vector<Eigen::Matrix<double, 7, 1>>& path) {
  double len = 0.0;
  for (std::size_t k = 1; k < path.size(); ++k) {
    const Eigen::Vector3d dt = path[k].head<3>() - path[k - 1].head<3>();
    const Eigen::Quaterniond a(path[k - 1][6], path[k - 1][3], path[k - 1][4], path[k - 1][5]);
    const Eigen::Quaterniond b(path[k][6], path[k][3], path[k][4], path[k][5]);
    len += std::hypot(dt.norm(), a.angularDistance(b));
  }
  return len;
}

}  // namespace

// With unit weights the left-invariant metric of SE(2) is the flat metric of
// R^2 x S^1, whose geodesic drives straight while turning at a constant rate. The
// group log follows the longer screw motion, and the walk must not take it as the
// Riemannian log. With the Euler retraction the log is the flat chord.
TEST(InterpolationSE2Flat, DiscreteGeodesicHasTheFlatGeodesicLength) {
  const Eigen::Vector3d a(0.0, 0.0, 0.0);
  InterpolationSettings settings;
  settings.step_size = 0.01;
  settings.max_steps = 2000;
  for (const Eigen::Vector3d b :
       {Eigen::Vector3d(1.0, 0.0, std::numbers::pi / 2), Eigen::Vector3d(2.0, 1.0, -2.0),
        Eigen::Vector3d(0.5, -1.5, 1.0)}) {
    const double flat = std::hypot(b.head<2>().norm(), b[2]);
    const SE2<> group;
    EXPECT_FALSE(is_riemannian_log(group));
    const auto r = discrete_geodesic(group, a, b, settings);
    ASSERT_EQ(r.status, InterpolationStatus::Converged) << b.transpose();
    EXPECT_NEAR(flat_se2_length(r.path), flat, 2e-3 * flat) << b.transpose();
    // The screw motion of the constant twist is longer.
    EXPECT_GT(group.distance(a, b), flat + 1e-2) << b.transpose();

    const SE2<SE2LeftInvariantMetric, SE2EulerRetraction> euler;
    EXPECT_TRUE(is_riemannian_log(euler));
    const auto e = discrete_geodesic(euler, a, b, settings);
    ASSERT_EQ(e.status, InterpolationStatus::Converged) << b.transpose();
    // The walk stops within its convergence tolerance, short of the target.
    EXPECT_NEAR(flat_se2_length(e.path), flat, 2e-3 * flat) << b.transpose();
    EXPECT_LE(flat_se2_length(e.path), flat + 1e-12) << b.transpose();
    EXPECT_NEAR(euler.distance(a, b), flat, 1e-12) << b.transpose();
  }
}

// The Euler retraction's chord is a geodesic under any weights with equal
// translational parts, and the group exponentials never give the Riemannian log.
TEST(InterpolationSE2Flat, RiemannianLogFlagFollowsTheGeometry) {
  using Euler = SE2<SE2LeftInvariantMetric, SE2EulerRetraction>;
  EXPECT_TRUE(is_riemannian_log(Euler{SE2LeftInvariantMetric{2.0, 2.0, 5.0}}));
  EXPECT_FALSE(is_riemannian_log(Euler{SE2LeftInvariantMetric{1.0, 3.0, 1.0}}));
  EXPECT_FALSE(is_riemannian_log(SE2<>{SE2LeftInvariantMetric{2.0, 2.0, 5.0}}));
  EXPECT_FALSE(is_riemannian_log(SE2<SE2LeftInvariantMetric, SE2RightExponentialMap>{}));
  EXPECT_FALSE(is_riemannian_log(SE3<>{}));
  EXPECT_FALSE(is_riemannian_log(SE3<SE3InvariantMetric, SE3RightExponentialMap>{}));
}

// On SE(3) with unit weights the geodesic moves the origin in a straight line while
// rotating at a constant rate, and the discrete geodesic has its length.
TEST(InterpolationSE3Flat, DiscreteGeodesicHasTheFlatGeodesicLength) {
  const SE3<> se3;
  Eigen::Matrix<double, 7, 1> a;
  a << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  Eigen::Matrix<double, 7, 1> b;
  const double angle = std::numbers::pi / 2;
  b << 1.0, 0.0, 0.0, 0.0, 0.0, std::sin(angle / 2), std::cos(angle / 2);
  InterpolationSettings settings;
  settings.step_size = 0.01;
  settings.max_steps = 2000;
  const auto r = discrete_geodesic(se3, a, b, settings);
  ASSERT_EQ(r.status, InterpolationStatus::Converged);
  const double flat = std::hypot(1.0, angle);
  EXPECT_NEAR(flat_se3_length(r.path), flat, 2e-3 * flat);
  EXPECT_GT(se3.distance(a, b), flat + 1e-2);
}
