/// @file lie.hpp
/// @brief Shared SO(3) / SE(3) Lie-group math (quaternions, exp/log, Jacobians).
///
/// This is the foundational geometry header used by both `so3.hpp` and
/// `se3.hpp`. It provides pure, allocation-free free functions on plain Eigen
/// vectors so that the manifold classes can stay thin policy wrappers.
///
/// ## Conventions (read carefully — the whole library depends on these)
///
/// ### Quaternions: scalar-LAST `[x, y, z, w]`
/// A quaternion is stored as an `Eigen::Vector4d` laid out as
/// \f$ q = [x, y, z, w] \f$ (the vector part first, the scalar part **last**).
/// This matches `Eigen::Quaterniond::coeffs()`, so an Eigen quaternion can be
/// compared component-wise against these functions without reordering.
///
/// A *unit* quaternion represents a rotation. Products use the **Hamilton**
/// convention with the composition property
/// \f$ R(\mathrm{quat\_mul}(a, b)) = R(a)\,R(b) \f$, i.e. `quat_mul(a, b)`
/// applies `b` first, then `a` — the same order as multiplying the rotation
/// matrices. A rotation by angle \f$ \theta \f$ about a unit axis \f$ n \f$ is
/// \f$ q = [\,\sin(\theta/2)\,n,\; \cos(\theta/2)\,] \f$.
///
/// ### Twists (se(3) Lie algebra): `[v; omega]`
/// An `Eigen::Matrix<double,6,1>` twist is ordered
/// \f$ \xi = [v_x, v_y, v_z,\; \omega_x, \omega_y, \omega_z] \f$ —
/// translation-velocity **first**, angular-velocity **last**.
///
/// ### SE(3) elements: `[t; quat]`
/// An `Eigen::Matrix<double,7,1>` group element is ordered
/// \f$ g = [t_x, t_y, t_z,\; q_x, q_y, q_z, q_w] \f$ — a translation followed
/// by a scalar-last unit quaternion. The identity is
/// \f$ [0, 0, 0,\; 0, 0, 0, 1] \f$.
///
/// ### Small-angle handling
/// Every trigonometric coefficient that divides by \f$ \theta \f$,
/// \f$ \theta^2 \f$ or \f$ \theta^3 \f$ has a Taylor-series branch near
/// \f$ \theta = 0 \f$ to stay finite and accurate (mirroring the SE(2)
/// V-matrix pattern in `manifold/se2.hpp`). The SO(3) coefficients switch at
/// \f$ \theta < 10^{-8} \f$; the SE(3) V-matrix coefficients switch at
/// \f$ \theta < 10^{-3} \f$ where catastrophic cancellation in
/// \f$ (1-\cos\theta) \f$ / \f$ (\theta-\sin\theta) \f$ would otherwise degrade
/// the direct formula.

#pragma once

#include <cmath>

#include <Eigen/Core>

namespace geodex::utils {

namespace detail {

/// @brief Cross product of two 3-vectors (kept local so the header needs only
/// `<Eigen/Core>`, not `<Eigen/Geometry>`).
inline Eigen::Vector3d cross(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
  return Eigen::Vector3d(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                         a[0] * b[1] - a[1] * b[0]);
}

/// @brief Small-angle threshold for the SO(3) exp/log coefficients (no
/// cancellation there, so the pure-\f$0/0\f$ guard suffices).
inline constexpr double kSo3Eps = 1e-8;

/// @brief Small-angle threshold for the SE(3) V-matrix coefficients. Larger
/// than `kSo3Eps` because \f$(1-\cos\theta)\f$ and \f$(\theta-\sin\theta)\f$
/// lose relative precision to cancellation well before \f$\theta = 10^{-8}\f$.
inline constexpr double kSe3Eps = 1e-3;

}  // namespace detail

// ---------------------------------------------------------------------------
// Skew / hat
// ---------------------------------------------------------------------------

/// @brief Skew-symmetric ("hat") matrix \f$ [\,\omega\,]_\times \f$ of a
/// 3-vector, satisfying \f$ [\omega]_\times v = \omega \times v \f$.
/// @param w The 3-vector \f$ \omega \f$.
/// @return The \f$ 3\times3 \f$ skew-symmetric matrix.
inline Eigen::Matrix3d skew(const Eigen::Vector3d& w) {
  Eigen::Matrix3d K;
  // clang-format off
  K <<   0.0, -w.z(),  w.y(),
       w.z(),    0.0, -w.x(),
      -w.y(),  w.x(),    0.0;
  // clang-format on
  return K;
}

// ---------------------------------------------------------------------------
// Quaternion algebra (scalar-last [x, y, z, w])
// ---------------------------------------------------------------------------

/// @brief Hamilton product \f$ a \otimes b \f$ of two quaternions.
///
/// @details Uses the composition convention
/// \f$ R(\mathrm{quat\_mul}(a, b)) = R(a)\,R(b) \f$.
/// @param a Left quaternion \f$ [x, y, z, w] \f$.
/// @param b Right quaternion \f$ [x, y, z, w] \f$.
/// @return The product \f$ a \otimes b \f$ (not renormalized).
inline Eigen::Vector4d quat_mul(const Eigen::Vector4d& a, const Eigen::Vector4d& b) {
  const double ax = a[0], ay = a[1], az = a[2], aw = a[3];
  const double bx = b[0], by = b[1], bz = b[2], bw = b[3];
  return Eigen::Vector4d(aw * bx + ax * bw + ay * bz - az * by,   // x
                         aw * by - ax * bz + ay * bw + az * bx,   // y
                         aw * bz + ax * by - ay * bx + az * bw,   // z
                         aw * bw - ax * bx - ay * by - az * bz);  // w
}

/// @brief Quaternion conjugate \f$ [-x, -y, -z, w] \f$.
/// @param q Input quaternion.
/// @return The conjugate.
inline Eigen::Vector4d quat_conj(const Eigen::Vector4d& q) {
  return Eigen::Vector4d(-q[0], -q[1], -q[2], q[3]);
}

/// @brief Quaternion inverse \f$ \bar{q} / \lVert q \rVert^2 \f$.
///
/// @note For a unit quaternion this equals `quat_conj(q)`.
/// @param q Input quaternion (need not be unit).
/// @return The multiplicative inverse.
inline Eigen::Vector4d quat_inv(const Eigen::Vector4d& q) {
  return quat_conj(q) / q.squaredNorm();
}

/// @brief Return \f$ q / \lVert q \rVert \f$ (a unit quaternion).
/// @param q Input quaternion (must be non-zero).
/// @return The normalized quaternion.
inline Eigen::Vector4d quat_normalize(const Eigen::Vector4d& q) { return q / q.norm(); }

/// @brief Rotate a 3-vector by a unit quaternion.
///
/// @details Uses the cross-product form
/// \f$ v' = v + 2s\,(u\times v) + 2\,u\times(u\times v) \f$ with
/// \f$ u = [x,y,z] \f$, \f$ s = w \f$; equivalent to `quat_to_rotmat(q) * v`
/// for unit \f$ q \f$.
/// @param q Unit quaternion.
/// @param v Vector to rotate.
/// @return The rotated vector.
inline Eigen::Vector3d quat_rotate(const Eigen::Vector4d& q, const Eigen::Vector3d& v) {
  const Eigen::Vector3d u = q.head<3>();
  const double s = q[3];
  const Eigen::Vector3d t = 2.0 * detail::cross(u, v);
  return v + s * t + detail::cross(u, t);
}

/// @brief Rotation matrix of a unit quaternion.
///
/// @note Assumes @p q is a unit quaternion (a rotation).
/// @param q Unit quaternion \f$ [x, y, z, w] \f$.
/// @return The corresponding \f$ 3\times3 \f$ rotation matrix.
inline Eigen::Matrix3d quat_to_rotmat(const Eigen::Vector4d& q) {
  const double x = q[0], y = q[1], z = q[2], w = q[3];
  const double xx = x * x, yy = y * y, zz = z * z;
  const double xy = x * y, xz = x * z, yz = y * z;
  const double wx = w * x, wy = w * y, wz = w * z;
  Eigen::Matrix3d R;
  // clang-format off
  R << 1.0 - 2.0 * (yy + zz),       2.0 * (xy - wz),       2.0 * (xz + wy),
             2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz),       2.0 * (yz - wx),
             2.0 * (xz - wy),       2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy);
  // clang-format on
  return R;
}

// ---------------------------------------------------------------------------
// SO(3) exponential / logarithm
// ---------------------------------------------------------------------------

/// @brief SO(3) exponential map: axis-angle \f$ \omega \in \mathbb{R}^3 \f$ to
/// a unit quaternion.
///
/// @details With \f$ \theta = \lVert\omega\rVert \f$ the result is
/// \f$ q = [\,\tfrac{\sin(\theta/2)}{\theta}\,\omega,\; \cos(\theta/2)\,] \f$,
/// returned normalized. The vector coefficient uses a Taylor expansion
/// \f$ \tfrac{1}{2} - \tfrac{\theta^2}{48} + \tfrac{\theta^4}{3840} \f$ for
/// small \f$ \theta \f$.
/// @param omega Rotation vector (axis times angle) in the body algebra so(3).
/// @return The unit quaternion \f$ [x, y, z, w] \f$.
inline Eigen::Vector4d so3_exp(const Eigen::Vector3d& omega) {
  const double theta2 = omega.squaredNorm();
  const double theta = std::sqrt(theta2);
  double coeff;  // sin(theta/2) / theta
  if (theta < detail::kSo3Eps) {
    coeff = 0.5 - theta2 / 48.0 + theta2 * theta2 / 3840.0;
  } else {
    coeff = std::sin(0.5 * theta) / theta;
  }
  Eigen::Vector4d q;
  q.head<3>() = coeff * omega;
  q[3] = std::cos(0.5 * theta);
  return quat_normalize(q);
}

/// @brief SO(3) logarithm: unit quaternion to axis-angle
/// \f$ \omega \in \mathbb{R}^3 \f$.
///
/// @details The quaternion is first normalized, then flipped to the
/// \f$ w \ge 0 \f$ hemisphere (double-cover: pick the shortest arc, so
/// \f$ \theta \in [0, \pi] \f$ and `so3_log(q) == so3_log(-q)`). Ties at
/// \f$ w = 0 \f$ (rotation by \f$ \pi \f$) are broken lexicographically on the
/// vector part so the antipodal identity still holds exactly. With
/// \f$ s = \lVert\mathrm{vec}\rVert \f$ and \f$ \theta = 2\,\mathrm{atan2}(s, w) \f$
/// the result is \f$ \tfrac{\theta}{s}\,\mathrm{vec} \f$; the coefficient uses a
/// Taylor expansion \f$ 2 + \tfrac{\theta^2}{12} \f$ for small \f$ s \f$.
/// @param q_in Unit quaternion \f$ [x, y, z, w] \f$ (re-normalized internally).
/// @return The rotation vector \f$ \omega \f$ with \f$ \lVert\omega\rVert \le \pi \f$.
inline Eigen::Vector3d so3_log(const Eigen::Vector4d& q_in) {
  const Eigen::Vector4d q = quat_normalize(q_in);
  double x = q[0], y = q[1], z = q[2], w = q[3];

  // Canonicalize to the w >= 0 hemisphere; break the w == 0 tie lexicographically.
  const bool negate =
      (w < 0.0) ||
      (w == 0.0 && (x < 0.0 || (x == 0.0 && (y < 0.0 || (y == 0.0 && z < 0.0)))));
  if (negate) {
    x = -x;
    y = -y;
    z = -z;
    w = -w;
  }

  const double s = std::sqrt(x * x + y * y + z * z);  // = |sin(theta/2)|
  const double theta = 2.0 * std::atan2(s, w);
  double coeff;  // theta / s
  if (s < detail::kSo3Eps) {
    coeff = 2.0 + theta * theta / 12.0;
  } else {
    coeff = theta / s;
  }
  return Eigen::Vector3d(coeff * x, coeff * y, coeff * z);
}

// ---------------------------------------------------------------------------
// SE(3) left Jacobian (the "V" matrix) and its inverse
// ---------------------------------------------------------------------------

/// @brief SE(3) left Jacobian \f$ V(\omega) \f$ relating the algebra
/// translation to the group translation.
///
/// @details
/// \f$ V(\omega) = I + \tfrac{1-\cos\theta}{\theta^2} K
///     + \tfrac{\theta-\sin\theta}{\theta^3} K^2 \f$
/// with \f$ K = [\omega]_\times \f$, \f$ \theta = \lVert\omega\rVert \f$.
/// The two scalar coefficients fall back to their Taylor series
/// (\f$ \to \tfrac12 \f$ and \f$ \to \tfrac16 \f$) for small \f$ \theta \f$.
/// @param omega Rotation vector.
/// @return The \f$ 3\times3 \f$ matrix \f$ V(\omega) \f$.
inline Eigen::Matrix3d se3_left_jacobian(const Eigen::Vector3d& omega) {
  const Eigen::Matrix3d K = skew(omega);
  const double theta2 = omega.squaredNorm();
  const double theta = std::sqrt(theta2);
  double A;  // (1 - cos theta) / theta^2
  double B;  // (theta - sin theta) / theta^3
  if (theta < detail::kSe3Eps) {
    const double t4 = theta2 * theta2;
    A = 0.5 - theta2 / 24.0 + t4 / 720.0;
    B = 1.0 / 6.0 - theta2 / 120.0 + t4 / 5040.0;
  } else {
    A = (1.0 - std::cos(theta)) / theta2;
    B = (theta - std::sin(theta)) / (theta2 * theta);
  }
  return Eigen::Matrix3d::Identity() + A * K + B * (K * K);
}

/// @brief Inverse of the SE(3) left Jacobian, \f$ V^{-1}(\omega) \f$.
///
/// @details
/// \f$ V^{-1}(\omega) = I - \tfrac12 K + c_2\,K^2 \f$ with
/// \f$ c_2 = \tfrac{1}{\theta^2}\!\left(1 - \tfrac{\theta}{2}\cot\tfrac{\theta}{2}\right) \f$,
/// \f$ K = [\omega]_\times \f$. The coefficient \f$ c_2 \f$ falls back to its
/// Taylor series \f$ \tfrac{1}{12} + \tfrac{\theta^2}{720} + \dots \f$ for small
/// \f$ \theta \f$. This is the exact analytic inverse of `se3_left_jacobian`.
/// @param omega Rotation vector.
/// @return The \f$ 3\times3 \f$ matrix \f$ V^{-1}(\omega) \f$.
inline Eigen::Matrix3d se3_left_jacobian_inverse(const Eigen::Vector3d& omega) {
  const Eigen::Matrix3d K = skew(omega);
  const double theta2 = omega.squaredNorm();
  const double theta = std::sqrt(theta2);
  double c2;  // (1 - (theta/2) cot(theta/2)) / theta^2
  if (theta < detail::kSe3Eps) {
    c2 = 1.0 / 12.0 + theta2 / 720.0 + theta2 * theta2 / 30240.0;
  } else {
    const double half = 0.5 * theta;
    c2 = (1.0 - half * std::cos(half) / std::sin(half)) / theta2;
  }
  return Eigen::Matrix3d::Identity() - 0.5 * K + c2 * (K * K);
}

// ---------------------------------------------------------------------------
// SE(3) exponential / logarithm and group operations
// ---------------------------------------------------------------------------

/// @brief SE(3) exponential map: twist \f$ \xi = [v; \omega] \f$ to a group
/// element \f$ [t; q] \f$.
///
/// @details \f$ t = V(\omega)\,v \f$ and \f$ q = \mathrm{so3\_exp}(\omega) \f$,
/// matching \f$ \exp \begin{bmatrix} [\omega]_\times & v \\ 0 & 0 \end{bmatrix} \f$.
/// @param xi Twist \f$ [v_x, v_y, v_z, \omega_x, \omega_y, \omega_z] \f$.
/// @return SE(3) element \f$ [t_x, t_y, t_z, q_x, q_y, q_z, q_w] \f$.
inline Eigen::Matrix<double, 7, 1> se3_exp(const Eigen::Matrix<double, 6, 1>& xi) {
  const Eigen::Vector3d v = xi.head<3>();
  const Eigen::Vector3d omega = xi.tail<3>();
  Eigen::Matrix<double, 7, 1> g;
  g.head<3>() = se3_left_jacobian(omega) * v;
  g.tail<4>() = so3_exp(omega);
  return g;
}

/// @brief SE(3) logarithm: group element \f$ [t; q] \f$ to a twist
/// \f$ \xi = [v; \omega] \f$.
///
/// @details \f$ \omega = \mathrm{so3\_log}(q) \f$ and
/// \f$ v = V^{-1}(\omega)\,t \f$.
/// @param g SE(3) element \f$ [t_x, t_y, t_z, q_x, q_y, q_z, q_w] \f$.
/// @return Twist \f$ [v_x, v_y, v_z, \omega_x, \omega_y, \omega_z] \f$.
inline Eigen::Matrix<double, 6, 1> se3_log(const Eigen::Matrix<double, 7, 1>& g) {
  const Eigen::Vector3d t = g.head<3>();
  const Eigen::Vector4d q = g.tail<4>();
  const Eigen::Vector3d omega = so3_log(q);
  Eigen::Matrix<double, 6, 1> xi;
  xi.head<3>() = se3_left_jacobian_inverse(omega) * t;
  xi.tail<3>() = omega;
  return xi;
}

/// @brief SE(3) composition \f$ a \cdot b \f$.
///
/// @details \f$ t = t_a + R(q_a)\,t_b \f$ and
/// \f$ q = \mathrm{normalize}(q_a \otimes q_b) \f$.
/// @param a Left SE(3) element \f$ [t; q] \f$.
/// @param b Right SE(3) element \f$ [t; q] \f$.
/// @return The composed element \f$ a \cdot b \f$.
inline Eigen::Matrix<double, 7, 1> se3_compose(const Eigen::Matrix<double, 7, 1>& a,
                                               const Eigen::Matrix<double, 7, 1>& b) {
  const Eigen::Vector3d ta = a.head<3>();
  const Eigen::Vector4d qa = a.tail<4>();
  const Eigen::Vector3d tb = b.head<3>();
  const Eigen::Vector4d qb = b.tail<4>();
  Eigen::Matrix<double, 7, 1> g;
  g.head<3>() = ta + quat_rotate(qa, tb);
  g.tail<4>() = quat_normalize(quat_mul(qa, qb));
  return g;
}

/// @brief SE(3) inverse \f$ a^{-1} \f$.
///
/// @details \f$ q^{-1} = \bar{q_a} \f$ (conjugate, unit assumed) and
/// \f$ t = -R(q^{-1})\,t_a \f$.
/// @param a SE(3) element \f$ [t; q] \f$.
/// @return The inverse element.
inline Eigen::Matrix<double, 7, 1> se3_inverse(const Eigen::Matrix<double, 7, 1>& a) {
  const Eigen::Vector3d ta = a.head<3>();
  const Eigen::Vector4d qa = a.tail<4>();
  const Eigen::Vector4d qinv = quat_conj(qa);
  Eigen::Matrix<double, 7, 1> g;
  g.head<3>() = -quat_rotate(qinv, ta);
  g.tail<4>() = qinv;
  return g;
}

}  // namespace geodex::utils
