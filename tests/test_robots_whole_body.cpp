/// @file test_robots_whole_body.cpp
/// @brief Whole-body spaces of the mobile robots, built as the product of an SE(2) base and
/// a robot's joint space, the product's coordinate facts and the product heuristic.

#include <cmath>

#include <numbers>
#include <type_traits>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "geodex/algorithm/precompute_matrix_lower_bound.hpp"
#include "geodex/heuristics/product_lower_bound.hpp"
#include "geodex/manifold/product.hpp"
#include "geodex/manifold/se2.hpp"
#include "geodex/manifold/sphere.hpp"
#include "geodex/metrics/se2_left_invariant.hpp"
#include "geodex/robots/joint_space.hpp"

namespace gr = geodex::robots;
using geodex::SE2LeftInvariantMetric;
using gr::ArmMetric;
using gr::Robot;

namespace {

template <typename M>
concept HasLo = requires(const M& m) { m.lo(); };

/// The SE(2) base of a whole-body space over the rectangle [lo, hi], every heading allowed.
geodex::SE2<> base_space(const SE2LeftInvariantMetric& metric, const Eigen::Vector2d& lo,
                         const Eigen::Vector2d& hi) {
  return geodex::SE2<>(metric, Eigen::Vector3d(lo.x(), lo.y(), -std::numbers::pi),
                       Eigen::Vector3d(hi.x(), hi.y(), std::numbers::pi));
}

}  // namespace

TEST(ProductCoordinateFacts, StackedBoundsPeriodsAndMetric) {
  geodex::Euclidean<3> arm;
  arm.set_sampling_bounds(Eigen::Vector3d(-1, -2, -3), Eigen::Vector3d(1, 2, 3));
  const geodex::SE2<> base(geodex::SE2LeftInvariantMetric(1.0, 4.0, 2.0),
                           Eigen::Vector3d(-5, -6, -std::numbers::pi),
                           Eigen::Vector3d(5, 6, std::numbers::pi));
  const auto space = geodex::make_product(base, arm);
  static_assert(geodex::algorithm::CertifiesOwnMatrixLowerBound<decltype(space)>);
  static_assert(!HasLo<geodex::ProductManifold<geodex::Sphere<>, geodex::Euclidean<2>>>);

  Eigen::VectorXd lo(6), hi(6), periods(6);
  lo << -5, -6, -std::numbers::pi, -1, -2, -3;
  hi << 5, 6, std::numbers::pi, 1, 2, 3;
  periods << 0, 0, 2 * std::numbers::pi, 0, 0, 0;
  EXPECT_EQ(space.lo(), lo);
  EXPECT_EQ(space.hi(), hi);
  EXPECT_EQ(space.periods(), periods);

  Eigen::VectorXd q(6);
  q << 0.3, -0.2, 0.7, 0.1, 0.2, 0.3;
  const Eigen::MatrixXd G = space.coordinate_metric(q);
  EXPECT_LT((G.topLeftCorner(3, 3) - base.coordinate_metric(q.head<3>())).cwiseAbs().maxCoeff(),
            1e-12);
  EXPECT_LT((G.bottomRightCorner(3, 3) - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff(), 1e-12);
  EXPECT_LT(G.topRightCorner(3, 3).cwiseAbs().maxCoeff(), 1e-12);

  // inner_matrix agrees with inner on the tangent basis.
  const Eigen::MatrixXd E = Eigen::MatrixXd::Identity(6, 6);
  const Eigen::MatrixXd B = space.inner_matrix(q, E, E);
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      EXPECT_NEAR(B(i, j), space.inner(q, E.col(i), E.col(j)), 1e-12);
    }
  }
}

TEST(JointSpace, BoundsAndMetrics) {
  constexpr auto R = Robot::Stretch3;
  const auto [lo, hi] = gr::MassMatrix<R>::joint_limits();
  const auto euclidean = gr::joint_space<R, ArmMetric::Euclidean>();
  const auto kinetic = gr::joint_space<R>();
  static_assert(std::is_same_v<decltype(kinetic),
                               const decltype(gr::joint_space<R, ArmMetric::KineticEnergy>())>);
  EXPECT_EQ(euclidean.lo(), Eigen::VectorXd(lo));
  EXPECT_EQ(euclidean.hi(), Eigen::VectorXd(hi));
  EXPECT_EQ(kinetic.lo(), Eigen::VectorXd(lo));
  EXPECT_EQ(kinetic.hi(), Eigen::VectorXd(hi));

  const gr::MassMatrix<R>::Vec q = 0.5 * (lo + hi);
  const Eigen::MatrixXd E = Eigen::MatrixXd::Identity(q.size(), q.size());
  gr::MassMatrix<R> mass;
  EXPECT_LT((kinetic.inner_matrix(q, E, E) - mass(q)).cwiseAbs().maxCoeff(), 1e-12);
  EXPECT_LT((euclidean.inner_matrix(q, E, E) - E).cwiseAbs().maxCoeff(), 1e-12);

  EXPECT_EQ(gr::joint_lower_bound<R>(ArmMetric::KineticEnergy),
            Eigen::MatrixXd(gr::MassLowerBound<R>::matrix()));
  EXPECT_EQ(gr::joint_lower_bound<R>(ArmMetric::Euclidean), E);
}

TEST(WholeBody, DrivesAndDimensions) {
  static_assert(gr::has_planar_base<Robot::Stretch3>);
  static_assert(gr::has_planar_base<Robot::Stretch4>);
  static_assert(gr::has_planar_base<Robot::RidgebackUr5e>);
  static_assert(gr::has_planar_base<Robot::HuskyUr5e>);
  static_assert(!gr::has_planar_base<Robot::Panda>);
  static_assert(gr::base_drive<Robot::Stretch3> == gr::BaseDrive::Differential);
  static_assert(gr::base_drive<Robot::Stretch4> == gr::BaseDrive::Holonomic);
  static_assert(gr::base_drive<Robot::RidgebackUr5e> == gr::BaseDrive::Holonomic);
  static_assert(gr::base_drive<Robot::HuskyUr5e> == gr::BaseDrive::Differential);
  static_assert(gr::MassMatrix<Robot::Stretch3>::Nq == 5);
  static_assert(gr::MassMatrix<Robot::Stretch4>::Nq == 5);
  static_assert(gr::MassMatrix<Robot::RidgebackUr5e>::Nq == 6);
  static_assert(gr::MassMatrix<Robot::HuskyUr5e>::Nq == 6);

  const auto space =
      geodex::make_product(base_space(SE2LeftInvariantMetric::differential_drive(),
                                      Eigen::Vector2d(-2, -3), Eigen::Vector2d(4, 5)),
                           gr::joint_space<Robot::Stretch3>());
  EXPECT_EQ(space.dim(), 8);
  const Eigen::VectorXd lo = space.lo(), hi = space.hi();
  for (int k = 0; k < 50; ++k) {
    const Eigen::VectorXd q = space.random_point();
    ASSERT_EQ(q.size(), 8);
    EXPECT_TRUE((q.array() >= lo.array() - 1e-12).all() && (q.array() <= hi.array() + 1e-12).all());
  }
}

template <Robot R, ArmMetric Metric>
void expect_heuristic_admissible(const SE2LeftInvariantMetric& metric) {
  const auto base = base_space(metric, Eigen::Vector2d(-2, -2), Eigen::Vector2d(2, 2));
  const auto space = geodex::make_product(base, gr::joint_space<R, Metric>());
  const auto h = geodex::heuristics::product_lower_bound(
      {{metric.coordinate_lower_bound(), base.periods()}, {gr::joint_lower_bound<R>(Metric)}});
  EXPECT_EQ(h.periods(), space.periods());
  for (int k = 0; k < 200; ++k) {
    const Eigen::VectorXd a = space.random_point();
    const Eigen::VectorXd b = space.random_point();
    EXPECT_LE(h(a, b), space.distance(a, b) + 1e-9) << "pair " << k;
  }
  // Headings on either side of the branch cut are close.
  Eigen::VectorXd a = space.random_point(), b = a;
  a[2] = std::numbers::pi - 0.05;
  b[2] = -std::numbers::pi + 0.05;
  EXPECT_NEAR(h(a, b), 0.1 * std::sqrt(metric.weights()[2]), 1e-9);
}

template <Robot R>
void expect_heuristic_admissible(const SE2LeftInvariantMetric& metric) {
  expect_heuristic_admissible<R, ArmMetric::KineticEnergy>(metric);
  expect_heuristic_admissible<R, ArmMetric::Euclidean>(metric);
}

TEST(WholeBody, HeuristicIsAdmissibleAndWraps) {
  for (const auto& metric :
       {SE2LeftInvariantMetric::holonomic(), SE2LeftInvariantMetric::differential_drive()}) {
    expect_heuristic_admissible<Robot::Stretch3>(metric);
    expect_heuristic_admissible<Robot::Stretch4>(metric);
    expect_heuristic_admissible<Robot::RidgebackUr5e>(metric);
    expect_heuristic_admissible<Robot::HuskyUr5e>(metric);
  }
}
