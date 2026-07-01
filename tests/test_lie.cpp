/// @file test_lie.cpp
/// @brief Unit tests for the shared SO(3)/SE(3) Lie-group math in
/// `geodex/utils/lie.hpp`, cross-checked against Eigen oracles.

#include <cmath>

#include <numbers>

#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "geodex/utils/lie.hpp"

// Optional strong oracle: skipped when Eigen's unsupported matrix exponential
// header is unavailable.
#if __has_include(<unsupported/Eigen/MatrixFunctions>)
#include <unsupported/Eigen/MatrixFunctions>
#define GEODEX_HAS_MATRIX_EXP 1
#endif

using namespace geodex::utils;
using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Vec7 = Eigen::Matrix<double, 7, 1>;

namespace {

constexpr double kPi = std::numbers::pi;
const Vector4d kQid(0.0, 0.0, 0.0, 1.0);

/// @brief Component-wise near-equality assertion for fixed-size Eigen vectors.
template <typename A, typename B>
void ExpectVecNear(const A& a, const B& b, double tol, const char* what) {
  ASSERT_EQ(a.size(), b.size()) << what;
  for (int i = 0; i < a.size(); ++i) {
    EXPECT_NEAR(a[i], b[i], tol) << what << " component " << i;
  }
}

/// @brief Align sign of a quaternion/twist to a reference (double cover).
Vector4d AlignSign(Vector4d q, const Vector4d& ref) {
  if (q.dot(ref) < 0.0) q = -q;
  return q;
}

}  // namespace

// ---------------------------------------------------------------------------
// Quaternion algebra
// ---------------------------------------------------------------------------

TEST(LieQuaternion, MultiplyIdentity) {
  Vector4d q = so3_exp(Vector3d(0.3, -0.7, 1.1));
  ExpectVecNear(quat_mul(q, kQid), q, 1e-15, "q * id");
  ExpectVecNear(quat_mul(kQid, q), q, 1e-15, "id * q");
}

TEST(LieQuaternion, MultiplyAssociative) {
  Vector4d a = so3_exp(Vector3d(0.3, -0.7, 1.1));
  Vector4d b = so3_exp(Vector3d(-0.4, 0.2, 0.9));
  Vector4d c = so3_exp(Vector3d(1.3, 0.1, -0.5));
  ExpectVecNear(quat_mul(quat_mul(a, b), c), quat_mul(a, quat_mul(b, c)), 1e-12, "(ab)c == a(bc)");
}

TEST(LieQuaternion, ConjugateGivesIdentity) {
  Vector4d q = so3_exp(Vector3d(0.5, 1.2, -0.8));
  ExpectVecNear(quat_mul(q, quat_conj(q)), kQid, 1e-14, "q * conj(q)");
  ExpectVecNear(quat_mul(quat_conj(q), q), kQid, 1e-14, "conj(q) * q");
}

TEST(LieQuaternion, InverseOfNonUnitQuaternion) {
  Vector4d q(0.5, -1.2, 0.3, 2.0);  // deliberately non-unit
  ExpectVecNear(quat_mul(q, quat_inv(q)), kQid, 1e-14, "q * inv(q)");
  ExpectVecNear(quat_mul(quat_inv(q), q), kQid, 1e-14, "inv(q) * q");
}

TEST(LieQuaternion, InverseEqualsConjugateForUnit) {
  Vector4d q = so3_exp(Vector3d(0.9, -0.1, 0.4));
  ExpectVecNear(quat_inv(q), quat_conj(q), 1e-14, "inv == conj for unit q");
}

TEST(LieQuaternion, NormalizeGivesUnit) {
  Vector4d q(0.5, -1.2, 0.3, 2.0);
  EXPECT_NEAR(quat_normalize(q).norm(), 1.0, 1e-15);
}

TEST(LieQuaternion, CompositionMatchesRotationMatrixProduct) {
  // R(quat_mul(a, b)) == R(a) * R(b)
  Vector4d a = so3_exp(Vector3d(0.3, -0.7, 1.1));
  Vector4d b = so3_exp(Vector3d(-0.4, 0.2, 0.9));
  Matrix3d lhs = quat_to_rotmat(quat_mul(a, b));
  Matrix3d rhs = quat_to_rotmat(a) * quat_to_rotmat(b);
  ExpectVecNear(lhs.reshaped(), rhs.reshaped(), 1e-12, "R(ab) == R(a)R(b)");
}

TEST(LieQuaternion, RotateMatchesRotationMatrix) {
  Vector4d q = so3_exp(Vector3d(0.3, -0.7, 1.1));
  Vector3d v(1.3, -2.1, 0.7);
  ExpectVecNear(quat_rotate(q, v), (quat_to_rotmat(q) * v).eval(), 1e-13, "quat_rotate == R*v");
}

TEST(LieQuaternion, RotmatIsOrthonormal) {
  Vector4d q = so3_exp(Vector3d(0.9, -0.3, 1.7));
  Matrix3d R = quat_to_rotmat(q);
  ExpectVecNear((R.transpose() * R).reshaped(), Matrix3d::Identity().reshaped(), 1e-13, "R^T R = I");
  EXPECT_NEAR(R.determinant(), 1.0, 1e-13);
}

// ---------------------------------------------------------------------------
// SO(3) exponential vs Eigen oracle
// ---------------------------------------------------------------------------

TEST(LieSo3Exp, MatchesEigenAngleAxis) {
  const Vector3d axes[] = {Vector3d(1, 0, 0), Vector3d(0, 1, 0), Vector3d(0, 0, 1),
                           Vector3d(1, 1, 1).normalized(), Vector3d(-2, 1, 0.5).normalized()};
  const double angles[] = {1e-10, 1e-6, 0.1, 1.0, 2.5, kPi - 1e-6};
  for (const Vector3d& axis : axes) {
    for (double theta : angles) {
      Vector4d q = so3_exp(theta * axis);
      Vector4d oracle = Eigen::Quaterniond(Eigen::AngleAxisd(theta, axis)).coeffs();  // xyzw
      ExpectVecNear(q, AlignSign(oracle, q), 1e-12, "so3_exp vs Eigen quaternion");
      // Rotation matrices agree regardless of the sign ambiguity.
      ExpectVecNear(quat_to_rotmat(q).reshaped(),
                    Eigen::AngleAxisd(theta, axis).toRotationMatrix().reshaped(), 1e-11,
                    "so3_exp R vs Eigen R");
    }
  }
}

TEST(LieSo3Exp, TinyAngleIsUnitAndClose) {
  Vector3d omega(1e-10, 0.0, 0.0);
  Vector4d q = so3_exp(omega);
  EXPECT_NEAR(q.norm(), 1.0, 1e-15);
  EXPECT_NEAR(q[0], 5e-11, 1e-20);  // sin(theta/2) ~ theta/2
  EXPECT_NEAR(q[3], 1.0, 1e-18);
}

TEST(LieSo3Exp, ZeroIsIdentityQuaternion) {
  ExpectVecNear(so3_exp(Vector3d::Zero()), kQid, 1e-15, "so3_exp(0) == identity");
}

// ---------------------------------------------------------------------------
// SO(3) logarithm
// ---------------------------------------------------------------------------

TEST(LieSo3Log, RoundTripsExp) {
  const Vector3d omegas[] = {
      Vector3d(1e-10, 0, 0), Vector3d(0.2, -0.3, 0.1), Vector3d(1.0, 1.0, 1.0),
      Vector3d(2.0, -1.0, 0.5),
      (kPi - 1e-4) * Vector3d(0.3, -0.4, 0.8660254037844386).normalized()};
  for (const Vector3d& omega : omegas) {
    ExpectVecNear(so3_log(so3_exp(omega)), omega, 1e-9, "so3_log(so3_exp(w)) == w");
  }
}

TEST(LieSo3Log, DoubleCoverGeneric) {
  Vector4d q = so3_exp(Vector3d(0.7, -0.2, 1.3));
  ExpectVecNear(so3_log(q), so3_log((-q).eval()), 1e-12, "so3_log(q) == so3_log(-q)");
}

TEST(LieSo3Log, DoubleCoverAtPiExactZeroScalar) {
  // theta = pi: scalar part is exactly zero (exercises the lexicographic tie-break).
  Vector4d q(1.0, 0.0, 0.0, 0.0);
  ExpectVecNear(so3_log(q), so3_log((-q).eval()), 1e-12, "so3_log(q) == so3_log(-q) at pi");
  EXPECT_NEAR(so3_log(q).norm(), kPi, 1e-12);
}

TEST(LieSo3Log, IdentityMapsToZero) {
  ExpectVecNear(so3_log(kQid), Vector3d::Zero(), 1e-15, "so3_log(identity) == 0");
}

TEST(LieSo3Log, NormalizesInput) {
  // A non-unit but proportional quaternion must yield the same log.
  Vector4d q = so3_exp(Vector3d(0.4, 0.9, -0.3));
  Vector4d scaled = 3.7 * q;
  ExpectVecNear(so3_log(scaled), so3_log(q), 1e-12, "so3_log ignores scale");
}

// ---------------------------------------------------------------------------
// SE(3) left Jacobian and inverse
// ---------------------------------------------------------------------------

TEST(LieSe3Jacobian, InverseIsTrueInverse) {
  const Vector3d omegas[] = {Vector3d(1e-10, 0, 0), Vector3d(0.4, -0.2, 0.1),
                             Vector3d(2.5, -1.0, 0.5), Vector3d(3.9, 0.2, -1.1)};
  for (const Vector3d& omega : omegas) {
    Matrix3d prod = se3_left_jacobian(omega) * se3_left_jacobian_inverse(omega);
    ExpectVecNear(prod.reshaped(), Matrix3d::Identity().reshaped(), 1e-12, "V * V^{-1} == I");
    Matrix3d prod2 = se3_left_jacobian_inverse(omega) * se3_left_jacobian(omega);
    ExpectVecNear(prod2.reshaped(), Matrix3d::Identity().reshaped(), 1e-12, "V^{-1} * V == I");
  }
}

TEST(LieSe3Jacobian, ZeroOmegaIsIdentity) {
  ExpectVecNear(se3_left_jacobian(Vector3d::Zero()).reshaped(), Matrix3d::Identity().reshaped(),
                1e-15, "V(0) == I");
  ExpectVecNear(se3_left_jacobian_inverse(Vector3d::Zero()).reshaped(),
                Matrix3d::Identity().reshaped(), 1e-15, "V^{-1}(0) == I");
}

// ---------------------------------------------------------------------------
// SE(3) exp / log
// ---------------------------------------------------------------------------

TEST(LieSe3ExpLog, PureTranslation) {
  Vec6 xi;
  xi << 0.5, -0.3, 0.9, 0.0, 0.0, 0.0;
  Vec7 g = se3_exp(xi);
  ExpectVecNear(g.head<3>().eval(), xi.head<3>().eval(), 1e-14, "translation passes through");
  ExpectVecNear(g.tail<4>().eval(), kQid, 1e-14, "rotation is identity");
  ExpectVecNear(se3_log(g), xi, 1e-12, "se3_log recovers pure translation");
}

TEST(LieSe3ExpLog, RoundTrips) {
  const Vector3d omegas[] = {Vector3d(1e-9, 0, 0), Vector3d(0.3, -0.7, 0.2),
                             Vector3d(2.0, 1.0, -1.5)};
  for (const Vector3d& omega : omegas) {
    Vec6 xi;
    xi << 1.1, -0.4, 0.6, omega[0], omega[1], omega[2];
    ExpectVecNear(se3_log(se3_exp(xi)), xi, 1e-9, "se3_log(se3_exp(xi)) == xi");
  }
}

// ---------------------------------------------------------------------------
// SE(3) group operations
// ---------------------------------------------------------------------------

namespace {
Vec7 MakePose(const Vector3d& t, const Vector3d& omega) {
  Vec7 g;
  g.head<3>() = t;
  g.tail<4>() = so3_exp(omega);
  return g;
}
}  // namespace

TEST(LieSe3Group, ComposeWithInverseIsIdentity) {
  Vec7 g = MakePose(Vector3d(1.0, 2.0, -0.5), Vector3d(0.3, -0.6, 0.9));
  Vec7 e = se3_compose(g, se3_inverse(g));
  ExpectVecNear(e.head<3>().eval(), Vector3d::Zero(), 1e-13, "translation cancels");
  Vector4d q = AlignSign(e.tail<4>(), kQid);
  ExpectVecNear(q, kQid, 1e-13, "rotation cancels");

  Vec7 e2 = se3_compose(se3_inverse(g), g);
  ExpectVecNear(e2.head<3>().eval(), Vector3d::Zero(), 1e-13, "translation cancels (left)");
  ExpectVecNear(AlignSign(e2.tail<4>(), kQid), kQid, 1e-13, "rotation cancels (left)");
}

TEST(LieSe3Group, ComposeAssociative) {
  Vec7 g1 = MakePose(Vector3d(1.0, 2.0, -0.5), Vector3d(0.3, -0.6, 0.9));
  Vec7 g2 = MakePose(Vector3d(-0.7, 0.4, 1.2), Vector3d(-0.2, 1.1, 0.4));
  Vec7 g3 = MakePose(Vector3d(0.2, -1.3, 0.8), Vector3d(1.0, 0.1, -0.7));

  Vec7 lhs = se3_compose(se3_compose(g1, g2), g3);
  Vec7 rhs = se3_compose(g1, se3_compose(g2, g3));
  ExpectVecNear(lhs.head<3>().eval(), rhs.head<3>().eval(), 1e-12, "translation assoc");
  ExpectVecNear(lhs.tail<4>().eval(), AlignSign(rhs.tail<4>(), lhs.tail<4>()), 1e-12, "rotation assoc");
}

TEST(LieSe3Group, ComposeMatchesHomogeneousMatrices) {
  // Cross-check composition against 4x4 homogeneous-matrix multiplication.
  Vec7 g1 = MakePose(Vector3d(1.0, 2.0, -0.5), Vector3d(0.3, -0.6, 0.9));
  Vec7 g2 = MakePose(Vector3d(-0.7, 0.4, 1.2), Vector3d(-0.2, 1.1, 0.4));

  auto to_mat = [](const Vec7& g) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = quat_to_rotmat(g.tail<4>());
    T.block<3, 1>(0, 3) = g.head<3>();
    return T;
  };
  Eigen::Matrix4d expected = to_mat(g1) * to_mat(g2);
  Eigen::Matrix4d got = to_mat(se3_compose(g1, g2));
  ExpectVecNear(got.reshaped(), expected.reshaped(), 1e-12, "compose == T1*T2");
}

// ---------------------------------------------------------------------------
// Strong oracle: se3_exp vs the matrix exponential of the 4x4 hat matrix
// ---------------------------------------------------------------------------

#ifdef GEODEX_HAS_MATRIX_EXP
TEST(LieSe3ExpLog, MatchesMatrixExponential) {
  const Vector3d omegas[] = {Vector3d(0.0, 0.0, 0.0), Vector3d(0.5, -0.3, 0.2),
                             Vector3d(1.5, 0.7, -1.1), Vector3d(2.5, -1.0, 0.5)};
  const Vector3d v(0.9, -1.2, 0.4);
  for (const Vector3d& omega : omegas) {
    Vec6 xi;
    xi << v[0], v[1], v[2], omega[0], omega[1], omega[2];

    Eigen::Matrix4d hat = Eigen::Matrix4d::Zero();
    hat.block<3, 3>(0, 0) = skew(omega);
    hat.block<3, 1>(0, 3) = v;
    Eigen::Matrix4d E = hat.exp();

    Vec7 g = se3_exp(xi);
    ExpectVecNear(g.head<3>().eval(), E.block<3, 1>(0, 3).eval(), 1e-11, "se3_exp translation");
    ExpectVecNear(quat_to_rotmat(g.tail<4>()).reshaped(), E.block<3, 3>(0, 0).reshaped(), 1e-11,
                  "se3_exp rotation");
  }
}
#else
TEST(LieSe3ExpLog, MatchesMatrixExponential) {
  GTEST_SKIP() << "unsupported/Eigen/MatrixFunctions not available";
}
#endif
