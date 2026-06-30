/// @file test_heuristics.cpp
/// @brief Tests for admissible heuristics: Zero, Euclidean, EigenvalueLowerBound,
///        MatrixLowerBound (with incremental Loewner-meet update), and detection traits.

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <random>

#include "geodex/heuristics/heuristics.hpp"

namespace gh = geodex::heuristics;

namespace {

/// @brief Generate a fixed sequence of `n` Eigen vectors of dimension `d`.
template <int Dim>
auto random_points(int n, std::uint64_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-3.0, 3.0);
  std::vector<Eigen::Matrix<double, Dim, 1>> xs(n);
  for (int i = 0; i < n; ++i) {
    Eigen::Matrix<double, Dim, 1> v;
    if constexpr (Dim == Eigen::Dynamic) v.resize(4);
    for (int k = 0; k < v.size(); ++k) v[k] = dist(rng);
    xs[i] = v;
  }
  return xs;
}

}  // namespace

// ---------------------------------------------------------------------------
// Zero
// ---------------------------------------------------------------------------

TEST(ZeroHeuristic, AlwaysZero) {
  gh::Zero h;
  Eigen::Vector3d a(1, 2, 3);
  Eigen::Vector3d b(-4, 5, 6);
  EXPECT_EQ(h(a, b), 0.0);
  EXPECT_EQ(h(a, a), 0.0);
}

// ---------------------------------------------------------------------------
// Euclidean
// ---------------------------------------------------------------------------

TEST(EuclideanHeuristic, ChordDistance) {
  gh::Euclidean h;
  Eigen::Vector3d a(0, 0, 0);
  Eigen::Vector3d b(3, 4, 0);
  EXPECT_NEAR(h(a, b), 5.0, 1e-12);
}

TEST(EuclideanHeuristic, ZeroForSamePoint) {
  gh::Euclidean h;
  Eigen::Vector4d a(1, 2, 3, 4);
  EXPECT_NEAR(h(a, a), 0.0, 1e-12);
}

TEST(EuclideanHeuristic, Symmetry) {
  gh::Euclidean h;
  Eigen::Vector3d a(1, 2, 3);
  Eigen::Vector3d b(4, 5, 6);
  EXPECT_NEAR(h(a, b), h(b, a), 1e-12);
}

// ---------------------------------------------------------------------------
// EigenvalueLowerBound
// ---------------------------------------------------------------------------

TEST(EigenvalueLowerBound, ScalesChordByCachedSqrt) {
  gh::EigenvalueLowerBound<gh::Euclidean> h(4.0);  // sqrt(4) = 2
  Eigen::Vector2d a(0, 0);
  Eigen::Vector2d b(3, 0);
  EXPECT_NEAR(h.sqrt_lambda_min(), 2.0, 1e-12);
  EXPECT_NEAR(h(a, b), 2.0 * 3.0, 1e-12);
}

TEST(EigenvalueLowerBound, DefaultBaseIsEuclidean) {
  gh::EigenvalueLowerBound<> h(1.0);  // default BaseT = Euclidean
  Eigen::Vector3d a(0, 0, 0);
  Eigen::Vector3d b(1, 2, 2);
  EXPECT_NEAR(h(a, b), 3.0, 1e-12);  // sqrt(1)*3 = 3
}

TEST(EigenvalueLowerBound, ZeroForSamePoint) {
  gh::EigenvalueLowerBound<gh::Euclidean> h(2.5);
  Eigen::Vector3d a(1, 2, 3);
  EXPECT_NEAR(h(a, a), 0.0, 1e-12);
}

TEST(EigenvalueLowerBound, Symmetry) {
  gh::EigenvalueLowerBound<gh::Euclidean> h(3.0);
  Eigen::Vector3d a(1, 2, 3);
  Eigen::Vector3d b(4, 5, 6);
  EXPECT_NEAR(h(a, b), h(b, a), 1e-12);
}

// ---------------------------------------------------------------------------
// MatrixLowerBound — basic
// ---------------------------------------------------------------------------

TEST(MatrixLowerBound, IdentityMatchesEuclidean) {
  Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  gh::MatrixLowerBound<3> h_mlb(I);
  gh::Euclidean h_euc;
  Eigen::Vector3d a(1, 2, 3);
  Eigen::Vector3d b(-2, 5, 1);
  EXPECT_NEAR(h_mlb(a, b), h_euc(a, b), 1e-12);
}

TEST(MatrixLowerBound, ScalarMatchesEigenvalueBound) {
  // Isotropic M_lower = alpha * I → h(a,b) = sqrt(alpha) * ||a - b||.
  const double alpha = 2.25;
  Eigen::Matrix3d M = alpha * Eigen::Matrix3d::Identity();
  gh::MatrixLowerBound<3> h_mlb(M);
  gh::EigenvalueLowerBound<gh::Euclidean> h_elb(alpha);
  Eigen::Vector3d a(0.1, 0.2, 0.3);
  Eigen::Vector3d b(0.9, -0.3, 0.5);
  EXPECT_NEAR(h_mlb(a, b), h_elb(a, b), 1e-12);
}

TEST(MatrixLowerBound, AnisotropicTighterThanScalar) {
  // M_lower = diag(4, 1). lambda_min = 1.
  // Euclidean-scaled eigenvalue bound: sqrt(1) * ||delta||.
  // MatrixLB: sqrt(delta^T diag(4,1) delta) — tighter in the x direction.
  Eigen::Matrix2d M;
  M << 4.0, 0.0, 0.0, 1.0;
  gh::MatrixLowerBound<2> h_mlb(M);
  gh::EigenvalueLowerBound<gh::Euclidean> h_elb(1.0);

  Eigen::Vector2d a(0, 0);
  Eigen::Vector2d b(1, 0);  // pure-x: MatrixLB should return 2, eigenvalue bound returns 1.
  EXPECT_NEAR(h_mlb(a, b), 2.0, 1e-12);
  EXPECT_NEAR(h_elb(a, b), 1.0, 1e-12);
  EXPECT_GT(h_mlb(a, b), h_elb(a, b));
}

TEST(MatrixLowerBound, Symmetry) {
  Eigen::Matrix3d M = Eigen::Matrix3d::Identity();
  M(0, 0) = 3.0;
  M(2, 2) = 0.5;
  gh::MatrixLowerBound<3> h(M);
  Eigen::Vector3d a(0.2, 0.7, -0.1);
  Eigen::Vector3d b(0.9, -0.3, 0.4);
  EXPECT_NEAR(h(a, b), h(b, a), 1e-12);
}

TEST(MatrixLowerBound, ZeroForSamePoint) {
  Eigen::Matrix3d M = 2.0 * Eigen::Matrix3d::Identity();
  gh::MatrixLowerBound<3> h(M);
  Eigen::Vector3d a(1, 2, 3);
  EXPECT_NEAR(h(a, a), 0.0, 1e-12);
}

TEST(MatrixLowerBound, DynamicDimension) {
  Eigen::MatrixXd M(4, 4);
  M.setIdentity();
  M(0, 0) = 9.0;
  gh::MatrixLowerBound<Eigen::Dynamic> h(M);
  Eigen::VectorXd a(4);
  a << 1, 0, 0, 0;
  Eigen::VectorXd b(4);
  b << 0, 0, 0, 0;
  EXPECT_NEAR(h(a, b), 3.0, 1e-12);  // sqrt(9)*1 = 3
}

TEST(MatrixLowerBound, EigenvalueFloorDominatesBoth) {
  // M_lower = diag(0.25, 0.25), lambda_min floor = 1.
  // For delta = (1, 0), ||L^T delta|| = 0.5, sqrt(1)*||delta|| = 1 — floor wins.
  // For delta = (1, 1), ||L^T delta|| = sqrt(0.5) ≈ 0.707, sqrt(1)*||delta|| = sqrt(2) ≈ 1.414 — floor still wins.
  Eigen::Matrix2d M;
  M << 0.25, 0.0, 0.0, 0.25;
  gh::MatrixLowerBound<2> h(M, /*lambda_min=*/1.0);

  Eigen::Vector2d a(0, 0);
  {
    Eigen::Vector2d b(1, 0);
    EXPECT_NEAR(h(a, b), 1.0, 1e-12);
  }
  {
    Eigen::Vector2d b(1, 1);
    EXPECT_NEAR(h(a, b), std::sqrt(2.0), 1e-12);
  }
  EXPECT_TRUE(h.has_eigenvalue_floor());
}

TEST(MatrixLowerBound, NoFloorDoesNotActivate) {
  Eigen::Matrix2d M = Eigen::Matrix2d::Identity();
  gh::MatrixLowerBound<2> h(M);  // no floor
  EXPECT_FALSE(h.has_eigenvalue_floor());
  Eigen::Vector2d a(0, 0);
  Eigen::Vector2d b(3, 4);
  EXPECT_NEAR(h(a, b), 5.0, 1e-12);  // plain Euclidean, no floor override
}

TEST(MatrixLowerBound, DetAndEigenvaluesAndMatrixAgree) {
  Eigen::Matrix3d M;
  M << 4.0, 0.5, 0.0,
       0.5, 2.0, 0.0,
       0.0, 0.0, 1.0;
  gh::MatrixLowerBound<3> h(M);

  // matrix() reconstructs M from the Cholesky.
  const Eigen::Matrix3d Mr = h.matrix();
  EXPECT_TRUE(Mr.isApprox(M, 1e-10));

  // det() matches direct determinant.
  EXPECT_NEAR(h.det(), M.determinant(), 1e-10);

  // eigenvalues() are ascending and match direct eig.
  const auto evs = h.eigenvalues();
  for (int i = 1; i < evs.size(); ++i) EXPECT_LE(evs[i - 1], evs[i] + 1e-12);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(M, Eigen::EigenvaluesOnly);
  EXPECT_TRUE(evs.isApprox(solver.eigenvalues(), 1e-10));
}

// ---------------------------------------------------------------------------
// MatrixLowerBound — incremental Loewner-meet update
// ---------------------------------------------------------------------------

TEST(MatrixLowerBoundUpdate, UpdateCountStartsAtZero) {
  Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  gh::MatrixLowerBound<3> h(I);
  EXPECT_EQ(h.update_count(), 0);
}

TEST(MatrixLowerBoundUpdate, NotAppliedForLargerMatrix) {
  // If M_new >= M_lower already, update() should return false and not change state.
  Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  gh::MatrixLowerBound<3> h(I);
  const Eigen::Matrix3d M_new = 5.0 * I;
  EXPECT_FALSE(h.update(M_new));
  EXPECT_EQ(h.update_count(), 0);
  EXPECT_TRUE(h.matrix().isApprox(I, 1e-12));
}

TEST(MatrixLowerBoundUpdate, LowersWhenSmaller) {
  // If M_new has an eigendirection smaller than M_lower, update() tightens.
  Eigen::Matrix2d I = Eigen::Matrix2d::Identity();
  gh::MatrixLowerBound<2> h(I);

  Eigen::Matrix2d M_new;
  M_new << 0.25, 0.0, 0.0, 2.0;  // smaller than I in x, larger in y
  EXPECT_TRUE(h.update(M_new));
  EXPECT_EQ(h.update_count(), 1);

  // Post-update: M_lower should match the meet: diag(min(1, 0.25), min(1, 2)) = diag(0.25, 1).
  const Eigen::Matrix2d M_expected = (Eigen::Matrix2d() << 0.25, 0.0, 0.0, 1.0).finished();
  EXPECT_TRUE(h.matrix().isApprox(M_expected, 1e-10));
}

TEST(MatrixLowerBoundUpdate, MonotonicNonIncreasingDet) {
  // Repeated updates with random SPD matrices should never increase det(M_lower).
  Eigen::Matrix3d M0;
  M0 << 4.0, 0.0, 0.0,
        0.0, 4.0, 0.0,
        0.0, 0.0, 4.0;
  gh::MatrixLowerBound<3> h(M0);

  double det_prev = h.det();
  // Feed a sequence of SPD matrices, some smaller in various directions.
  std::vector<Eigen::Matrix3d> obs;
  {
    Eigen::Matrix3d X;
    X << 1.0, 0.5, 0.0,
         0.5, 2.0, 0.0,
         0.0, 0.0, 0.5;
    obs.push_back(X);
  }
  {
    Eigen::Matrix3d X;
    X << 0.3, 0.0, 0.1,
         0.0, 3.0, 0.0,
         0.1, 0.0, 2.0;
    obs.push_back(X);
  }
  {
    Eigen::Matrix3d X;
    X << 2.5, 0.0, 0.0,
         0.0, 0.2, 0.0,
         0.0, 0.0, 1.5;
    obs.push_back(X);
  }

  for (const auto& X : obs) {
    h.update(X);
    const double det_now = h.det();
    EXPECT_LE(det_now, det_prev + 1e-10);
    det_prev = det_now;
  }
}

TEST(MatrixLowerBoundUpdate, LoewnerOrderPreservedAfterOnlineUpdates) {
  // After any sequence of updates with SPD observations M_k, the current M_lower
  // must satisfy M_lower ≤ M_k in the Loewner order for every k (in particular
  // for each M_k we just supplied).
  Eigen::Matrix2d I = Eigen::Matrix2d::Identity();
  gh::MatrixLowerBound<2> h(I);

  std::vector<Eigen::Matrix2d> obs;
  obs.push_back((Eigen::Matrix2d() << 0.5, 0.0, 0.0, 3.0).finished());
  obs.push_back((Eigen::Matrix2d() << 2.0, 0.5, 0.5, 0.4).finished());
  obs.push_back((Eigen::Matrix2d() << 0.7, 0.0, 0.0, 0.7).finished());

  for (const auto& X : obs) h.update(X);

  const Eigen::Matrix2d M_lower = h.matrix();
  for (const auto& X : obs) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(X - M_lower,
                                                          Eigen::EigenvaluesOnly);
    // All eigenvalues of (X - M_lower) must be >= 0 (PSD), modulo tiny numerical slack.
    EXPECT_GE(solver.eigenvalues().minCoeff(), -1e-9);
  }
}

// ---------------------------------------------------------------------------
// Property tests: non-negativity, triangle inequality, dominance
// ---------------------------------------------------------------------------

TEST(ZeroHeuristicProperties, NonNegative) {
  gh::Zero h;
  for (const auto& a : random_points<3>(10, /*seed=*/123)) {
    for (const auto& b : random_points<3>(10, /*seed=*/456)) {
      EXPECT_GE(h(a, b), 0.0);
    }
  }
}

TEST(EuclideanHeuristicProperties, NonNegative) {
  gh::Euclidean h;
  for (const auto& a : random_points<3>(10, /*seed=*/1)) {
    for (const auto& b : random_points<3>(10, /*seed=*/2)) {
      EXPECT_GE(h(a, b), 0.0);
    }
  }
}

TEST(EuclideanHeuristicProperties, TriangleInequality) {
  gh::Euclidean h;
  const auto pts = random_points<3>(20, /*seed=*/7);
  for (std::size_t i = 0; i + 2 < pts.size(); ++i) {
    const auto& a = pts[i];
    const auto& b = pts[i + 1];
    const auto& c = pts[i + 2];
    EXPECT_LE(h(a, c), h(a, b) + h(b, c) + 1e-12);
  }
}

TEST(EigenvalueLowerBoundProperties, NonNegative) {
  gh::EigenvalueLowerBound<gh::Euclidean> h(2.5);
  for (const auto& a : random_points<3>(10, /*seed=*/11)) {
    for (const auto& b : random_points<3>(10, /*seed=*/22)) {
      EXPECT_GE(h(a, b), 0.0);
    }
  }
}

TEST(EigenvalueLowerBoundProperties, TriangleInequality) {
  // h_lb = sqrt(lambda_min) * ||a - b|| inherits the triangle inequality from Euclidean.
  gh::EigenvalueLowerBound<gh::Euclidean> h(2.0);
  const auto pts = random_points<3>(20, /*seed=*/8);
  for (std::size_t i = 0; i + 2 < pts.size(); ++i) {
    const auto& a = pts[i];
    const auto& b = pts[i + 1];
    const auto& c = pts[i + 2];
    EXPECT_LE(h(a, c), h(a, b) + h(b, c) + 1e-12);
  }
}

TEST(EigenvalueLowerBoundProperties, DominatesZero) {
  gh::EigenvalueLowerBound<gh::Euclidean> h_lb(0.5);
  gh::Zero h_z;
  for (const auto& a : random_points<3>(8, /*seed=*/33)) {
    for (const auto& b : random_points<3>(8, /*seed=*/44)) {
      EXPECT_GE(h_lb(a, b), h_z(a, b));
    }
  }
}

TEST(MatrixLowerBoundProperties, NonNegative) {
  Eigen::Matrix3d M;
  M << 4.0, 0.5, 0.0,
       0.5, 2.0, 0.0,
       0.0, 0.0, 1.0;
  gh::MatrixLowerBound<3> h(M);
  for (const auto& a : random_points<3>(10, /*seed=*/55)) {
    for (const auto& b : random_points<3>(10, /*seed=*/66)) {
      EXPECT_GE(h(a, b), 0.0);
    }
  }
}

TEST(MatrixLowerBoundProperties, TriangleInequality) {
  // h(a,b) = ||L^T (a - b)|| is a norm on (a - b), so the triangle inequality holds.
  Eigen::Matrix3d M;
  M << 3.0, 0.4, 0.0,
       0.4, 1.5, 0.0,
       0.0, 0.0, 0.8;
  gh::MatrixLowerBound<3> h(M);
  const auto pts = random_points<3>(20, /*seed=*/9);
  for (std::size_t i = 0; i + 2 < pts.size(); ++i) {
    const auto& a = pts[i];
    const auto& b = pts[i + 1];
    const auto& c = pts[i + 2];
    EXPECT_LE(h(a, c), h(a, b) + h(b, c) + 1e-9);
  }
}

TEST(MatrixLowerBoundProperties, SymmetryDynamic) {
  Eigen::MatrixXd M(4, 4);
  M.setIdentity();
  M(0, 0) = 4.0;
  M(2, 2) = 0.7;
  gh::MatrixLowerBound<Eigen::Dynamic> h(M);
  for (const auto& a : random_points<Eigen::Dynamic>(6, /*seed=*/77)) {
    for (const auto& b : random_points<Eigen::Dynamic>(6, /*seed=*/88)) {
      EXPECT_NEAR(h(a, b), h(b, a), 1e-12);
    }
  }
}

// ---------------------------------------------------------------------------
// Detection traits
// ---------------------------------------------------------------------------

TEST(HeuristicTraits, DetectsMatrixLowerBound) {
  static_assert(gh::is_matrix_lower_bound_v<gh::MatrixLowerBound<3>>);
  static_assert(gh::is_matrix_lower_bound_v<gh::MatrixLowerBound<Eigen::Dynamic>>);
  static_assert(!gh::is_matrix_lower_bound_v<gh::Euclidean>);
  static_assert(!gh::is_matrix_lower_bound_v<gh::Zero>);
  static_assert(!gh::is_matrix_lower_bound_v<gh::EigenvalueLowerBound<gh::Euclidean>>);
  SUCCEED();
}

TEST(HeuristicTraits, DetectsEigenvalueLowerBound) {
  static_assert(gh::is_eigenvalue_lower_bound_v<gh::EigenvalueLowerBound<gh::Euclidean>>);
  static_assert(gh::is_eigenvalue_lower_bound_v<gh::EigenvalueLowerBound<gh::Zero>>);
  static_assert(!gh::is_eigenvalue_lower_bound_v<gh::Euclidean>);
  static_assert(!gh::is_eigenvalue_lower_bound_v<gh::Zero>);
  static_assert(!gh::is_eigenvalue_lower_bound_v<gh::MatrixLowerBound<3>>);
  SUCCEED();
}
