/// @file test_configuration_space.cpp
/// @brief Tests for ConfigurationSpace manifold wrapper.

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <geodex/core/concepts.hpp>
#include <geodex/manifold/configuration_space.hpp>
#include <geodex/manifold/euclidean.hpp>
#include <geodex/manifold/torus.hpp>
#include <geodex/metrics/kinetic_energy.hpp>
#include <geodex/metrics/weighted.hpp>

// ---------------------------------------------------------------------------
// Concept checks
// ---------------------------------------------------------------------------

// KineticEnergyMetric with a simple mass matrix function
using MassFn = std::function<Eigen::Matrix2d(const Eigen::Vector2d&)>;
using KE2 = geodex::KineticEnergyMetric<MassFn>;

static_assert(geodex::RiemannianManifold<geodex::ConfigurationSpace<geodex::Torus<2>, KE2>>);
static_assert(geodex::RiemannianManifold<geodex::ConfigurationSpace<geodex::Euclidean<2>, KE2>>);

// WeightedMetric wrapping a flat metric
using WeightedFlat = geodex::WeightedMetric<geodex::TorusFlatMetric<2>>;
static_assert(
    geodex::RiemannianManifold<geodex::ConfigurationSpace<geodex::Torus<2>, WeightedFlat>>);
static_assert(
    geodex::HasInjectivityRadius<geodex::ConfigurationSpace<geodex::Torus<2>, WeightedFlat>>);

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

class ConfigurationSpaceTest : public ::testing::Test {
 protected:
  MassFn mass_fn = [](const Eigen::Vector2d& q) {
    double c = std::cos(q[1]);
    Eigen::Matrix2d M;
    M(0, 0) = 2.0 + c;
    M(0, 1) = 0.5 + 0.3 * c;
    M(1, 0) = M(0, 1);
    M(1, 1) = 1.0;
    return M;
  };
};

TEST_F(ConfigurationSpaceTest, DelegatesTopologyToBase) {
  geodex::Torus<2> base;
  KE2 metric{mass_fn};
  geodex::ConfigurationSpace cspace{base, metric};

  EXPECT_EQ(cspace.dim(), 2);

  // exp/log should use torus wrapping
  Eigen::Vector2d p(0.1, 0.1);
  Eigen::Vector2d v(6.0, 0.0);  // wraps around

  auto q = cspace.exp(p, v);
  // Should wrap: (0.1 + 6.0) mod 2pi ≈ 6.1 - 2pi ≈ -0.183 -> wrapped to [0, 2pi)
  EXPECT_GE(q[0], 0.0);
  EXPECT_LT(q[0], 2.0 * M_PI);

  // log should give shortest path
  Eigen::Vector2d a(0.1, 0.1);
  Eigen::Vector2d b(6.1, 0.1);
  auto vlog = cspace.log(a, b);
  EXPECT_LE(std::abs(vlog[0]), M_PI);
}

TEST_F(ConfigurationSpaceTest, DelegatesGeometryToMetric) {
  geodex::Torus<2> base;
  KE2 metric{mass_fn};
  geodex::ConfigurationSpace cspace{base, metric};

  Eigen::Vector2d q(0.5, 1.0);
  Eigen::Vector2d u(1.0, 0.0);
  Eigen::Vector2d v(0.0, 1.0);

  // inner product should use M(q), not identity
  Eigen::Matrix2d M = mass_fn(q);
  EXPECT_NEAR(cspace.inner(q, u, v), M(0, 1), 1e-12);
  EXPECT_NEAR(cspace.inner(q, u, u), M(0, 0), 1e-12);
}

TEST_F(ConfigurationSpaceTest, DistanceIsPositive) {
  geodex::Torus<2> base;
  KE2 metric{mass_fn};
  geodex::ConfigurationSpace cspace{base, metric};

  Eigen::Vector2d p(1.0, 1.0);
  Eigen::Vector2d q(2.0, 2.0);

  double d = cspace.distance(p, q);
  EXPECT_GT(d, 0.0);
}

TEST_F(ConfigurationSpaceTest, DistanceSymmetric) {
  geodex::Torus<2> base;
  KE2 metric{mass_fn};
  geodex::ConfigurationSpace cspace{base, metric};

  Eigen::Vector2d p(1.0, 1.0);
  Eigen::Vector2d q(2.0, 1.5);

  EXPECT_NEAR(cspace.distance(p, q), cspace.distance(q, p), 1e-10);
}

TEST_F(ConfigurationSpaceTest, GeodesicEndpoints) {
  geodex::Torus<2> base;
  KE2 metric{mass_fn};
  geodex::ConfigurationSpace cspace{base, metric};

  Eigen::Vector2d p(1.0, 1.0);
  Eigen::Vector2d q(2.0, 1.5);

  auto at0 = cspace.geodesic(p, q, 0.0);
  auto at1 = cspace.geodesic(p, q, 1.0);

  EXPECT_NEAR((at0 - p).norm(), 0.0, 1e-10);
  EXPECT_NEAR((at1 - q).norm(), 0.0, 1e-10);
}

TEST_F(ConfigurationSpaceTest, EuclideanBaseWithKineticEnergy) {
  geodex::Euclidean<2> base;
  KE2 metric{mass_fn};
  geodex::ConfigurationSpace cspace{base, metric};

  Eigen::Vector2d p(0.5, 1.0);
  Eigen::Vector2d q(1.5, 2.0);

  double d = cspace.distance(p, q);
  EXPECT_GT(d, 0.0);

  // With identity mass matrix, distance should equal Euclidean distance
  MassFn id_fn = [](const Eigen::Vector2d& /*q*/) { return Eigen::Matrix2d::Identity(); };
  KE2 id_metric{id_fn};
  geodex::ConfigurationSpace id_cspace{geodex::Euclidean<2>{}, id_metric};

  double d_id = id_cspace.distance(p, q);
  double d_eucl = (q - p).norm();
  EXPECT_NEAR(d_id, d_eucl, 1e-10);
}

TEST_F(ConfigurationSpaceTest, AccessBaseAndMetric) {
  geodex::Torus<2> base;
  KE2 metric{mass_fn};
  geodex::ConfigurationSpace cspace{base, metric};

  EXPECT_EQ(cspace.base().dim(), 2);
  // metric() should be accessible
  Eigen::Vector2d q(0.5, 1.0);
  Eigen::Vector2d v(1.0, 0.0);
  EXPECT_NEAR(cspace.metric().inner(q, v, v), cspace.inner(q, v, v), 1e-15);
}
