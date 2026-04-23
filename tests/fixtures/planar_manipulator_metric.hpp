#pragma once

#include <cmath>

#include <limits>

#include <Eigen/Core>

namespace geodex {

/// Configuration-dependent mass-inertia matrix metric for a 2-link planar arm.
///
/// The metric tensor at configuration q is the 2x2 mass matrix M(q):
///   M11 = I1 + I2 + m1*lc1² + m2*(l1² + lc2² + 2*l1*lc2*cos(q2))
///   M12 = M21 = I2 + m2*(lc2² + l1*lc2*cos(q2))
///   M22 = I2 + m2*lc2²
///
/// This gives a proper Riemannian metric where inner(q, u, v) = u^T M(q) v.
/// M depends only on q2 (the elbow angle).
struct PlanarManipulatorMetric {
  double l1, l2;    // link lengths
  double m1, m2;    // link masses
  double lc1, lc2;  // center-of-mass distances from joint
  double I1, I2;    // moments of inertia

  PlanarManipulatorMetric(double l1 = 1.0, double l2 = 1.0, double m1 = 1.0, double m2 = 1.0,
                          double lc1 = 0.5, double lc2 = 0.5, double I1 = 1.0 / 12.0,
                          double I2 = 1.0 / 12.0)
      : l1(l1), l2(l2), m1(m1), m2(m2), lc1(lc1), lc2(lc2), I1(I1), I2(I2) {}

  Eigen::Matrix2d mass_matrix(const Eigen::Vector2d& q) const {
    double c2 = std::cos(q[1]);
    double h = l1 * lc2 * c2;

    Eigen::Matrix2d M;
    M(0, 0) = I1 + I2 + m1 * lc1 * lc1 + m2 * (l1 * l1 + lc2 * lc2 + 2.0 * h);
    M(0, 1) = I2 + m2 * (lc2 * lc2 + h);
    M(1, 0) = M(0, 1);
    M(1, 1) = I2 + m2 * lc2 * lc2;
    return M;
  }

  double inner(const Eigen::Vector2d& q, const Eigen::Vector2d& u, const Eigen::Vector2d& v) const {
    return u.dot(mass_matrix(q) * v);
  }

  double norm(const Eigen::Vector2d& q, const Eigen::Vector2d& v) const {
    return std::sqrt(inner(q, v, v));
  }

  double injectivity_radius() const { return std::numeric_limits<double>::infinity(); }
};

}  // namespace geodex
