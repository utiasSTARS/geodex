/// @file test_affine_combined_metric.cpp
/// @brief Tests for `geodex::AffineCombinedMetric` — variadic positive linear
///        combination of Riemannian metric policies.

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <gtest/gtest.h>

#include <array>
#include <random>

#include "geodex/core/metric.hpp"
#include "geodex/metrics/affine_combined.hpp"
#include "geodex/metrics/constant_spd.hpp"
#include "geodex/metrics/identity.hpp"

namespace {

Eigen::Matrix3d random_spd3(std::mt19937& rng) {
  std::normal_distribution<double> n;
  Eigen::Matrix3d A;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) A(i, j) = n(rng);
  return A.transpose() * A + Eigen::Matrix3d::Identity();
}

}  // namespace

TEST(AffineCombined, BinaryIdentitySum) {
  // Two IdentityMetric<3> with coeffs (0.5, 0.5) → identity (sum of weights = 1).
  geodex::AffineCombinedMetric combined({0.5, 0.5}, geodex::IdentityMetric<3>{},
                                        geodex::IdentityMetric<3>{});
  geodex::IdentityMetric<3> reference;

  Eigen::Vector3d p(0.0, 0.0, 0.0);
  Eigen::Vector3d u(1.0, 2.0, 3.0);
  Eigen::Vector3d v(-1.0, 0.5, 2.0);

  EXPECT_NEAR(combined.inner(p, u, v), reference.inner(p, u, v), 1e-12);
  EXPECT_NEAR(combined.norm(p, u), reference.norm(p, u), 1e-12);
}

TEST(AffineCombined, VariadicSPD) {
  // Three ConstantSPDMetric<3> summands: inner_matrix must equal the coeff-weighted sum.
  Eigen::Matrix3d A1 = Eigen::Matrix3d::Identity();
  A1(0, 0) = 4.0;
  Eigen::Matrix3d A2 = Eigen::Matrix3d::Identity() * 2.0;
  Eigen::Matrix3d A3;
  A3 << 3.0, 0.5, 0.0,
        0.5, 2.0, 0.0,
        0.0, 0.0, 1.5;

  const std::array<double, 3> c = {0.25, 0.5, 0.75};
  geodex::AffineCombinedMetric combined(c, geodex::ConstantSPDMetric<3>{A1},
                                        geodex::ConstantSPDMetric<3>{A2},
                                        geodex::ConstantSPDMetric<3>{A3});

  Eigen::Vector3d p(0.0, 0.0, 0.0);
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d M = combined.inner_matrix(p, I, I);
  const Eigen::Matrix3d expected = c[0] * A1 + c[1] * A2 + c[2] * A3;
  EXPECT_TRUE(M.isApprox(expected, 1e-12));
}

TEST(AffineCombined, ConstantSPDAnisotropic) {
  Eigen::Matrix2d A1;
  A1 << 4.0, 0.0, 0.0, 1.0;
  Eigen::Matrix2d A2;
  A2 << 1.0, 0.5, 0.5, 2.0;

  geodex::AffineCombinedMetric combined({1.0, 1.0}, geodex::ConstantSPDMetric<2>{A1},
                                        geodex::ConstantSPDMetric<2>{A2});
  Eigen::Vector2d p(0.0, 0.0);
  const Eigen::Matrix2d I = Eigen::Matrix2d::Identity();
  const Eigen::Matrix2d M = combined.inner_matrix(p, I, I);
  EXPECT_TRUE(M.isApprox(A1 + A2, 1e-12));

  // Spot-check inner against a direct evaluation.
  Eigen::Vector2d u(1.0, -2.0);
  Eigen::Vector2d v(0.5, 0.7);
  EXPECT_NEAR(combined.inner(p, u, v), u.dot((A1 + A2) * v), 1e-12);
}

TEST(AffineCombined, ZeroCoefficientSilencesSummand) {
  // coeffs (1, 0) → identical to the first metric.
  Eigen::Matrix3d A1;
  A1 << 4.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 9.0;
  Eigen::Matrix3d A2;
  A2 << 100.0, 50.0, -10.0,
        50.0,  80.0,  20.0,
        -10.0, 20.0,  60.0;
  geodex::AffineCombinedMetric combined({1.0, 0.0}, geodex::ConstantSPDMetric<3>{A1},
                                        geodex::ConstantSPDMetric<3>{A2});
  geodex::ConstantSPDMetric<3> reference{A1};

  Eigen::Vector3d p(0.0, 0.0, 0.0);
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  EXPECT_TRUE(combined.inner_matrix(p, I, I).isApprox(reference.inner_matrix(p, I, I), 1e-12));

  Eigen::Vector3d u(0.3, -0.7, 1.2);
  Eigen::Vector3d v(1.0, 1.0, 1.0);
  EXPECT_NEAR(combined.inner(p, u, v), reference.inner(p, u, v), 1e-12);
}

TEST(AffineCombined, PreservesSPDUnderPositiveCoeffs) {
  // For 10 random SPD summands and 10 random positive coeffs the resulting
  // inner_matrix must be SPD at any sample point.
  std::mt19937 rng(2026);
  Eigen::Matrix3d A1 = random_spd3(rng);
  Eigen::Matrix3d A2 = random_spd3(rng);
  Eigen::Matrix3d A3 = random_spd3(rng);

  std::uniform_real_distribution<double> coef(0.1, 5.0);
  for (int trial = 0; trial < 10; ++trial) {
    const std::array<double, 3> c = {coef(rng), coef(rng), coef(rng)};
    geodex::AffineCombinedMetric combined(c, geodex::ConstantSPDMetric<3>{A1},
                                          geodex::ConstantSPDMetric<3>{A2},
                                          geodex::ConstantSPDMetric<3>{A3});
    Eigen::Vector3d p(0.0, 0.0, 0.0);
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d M = combined.inner_matrix(p, I, I);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(M, Eigen::EigenvaluesOnly);
    EXPECT_GT(solver.eigenvalues().minCoeff(), 0.0);
  }
}

TEST(AffineCombined, ConceptCompliance) {
  // AffineCombinedMetric is a metric *policy*, not a manifold — it satisfies
  // MetricHasInnerMatrix but does not (and need not) satisfy `Manifold`/`HasMetric`.
  using Combined =
      geodex::AffineCombinedMetric<geodex::IdentityMetric<3>, geodex::IdentityMetric<3>>;
  static_assert(geodex::MetricHasInnerMatrix<Combined, Eigen::Vector3d>);
  SUCCEED();
}

TEST(AffineCombined, AccessorsExposeSummandsAndCoeffs) {
  Eigen::Matrix3d A1 = Eigen::Matrix3d::Identity() * 2.0;
  Eigen::Matrix3d A2 = Eigen::Matrix3d::Identity() * 3.0;
  geodex::AffineCombinedMetric combined({0.25, 0.75}, geodex::ConstantSPDMetric<3>{A1},
                                        geodex::ConstantSPDMetric<3>{A2});
  EXPECT_EQ(combined.coeffs()[0], 0.25);
  EXPECT_EQ(combined.coeffs()[1], 0.75);
  EXPECT_TRUE(combined.metric<0>().weight_matrix().isApprox(A1));
  EXPECT_TRUE(combined.metric<1>().weight_matrix().isApprox(A2));
}
