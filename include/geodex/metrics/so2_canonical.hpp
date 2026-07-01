/// @file so2_canonical.hpp
/// @brief Canonical metric on SO(2) — thin wrapper over ConstantSPDMetric<1>.

#pragma once

#include <Eigen/Core>

#include "geodex/core/metric.hpp"
#include "geodex/metrics/constant_spd.hpp"

namespace geodex {

/// @brief Canonical (bi-invariant) metric on SO(2).
///
/// @details The inner product is constant:
/// \f$ \langle u, v \rangle = w\, u\, v \f$ with a single scalar weight \f$ w > 0 \f$.
/// The default \f$ w = 1 \f$ is the unit-speed round metric on the circle, for which
/// the Lie-group `log` is the Riemannian logarithm.
///
/// Implementation: this is `ConstantSPDMetric<1>` with `A = [[w]]`. The scalar
/// `weight_` is preserved alongside the base metric so that
/// `SO2::has_riemannian_log_runtime()` can quickly check unit weight without
/// inspecting the wrapped SPD matrix.
class SO2CanonicalMetric {
 public:
  using Point = Eigen::Matrix<double, 1, 1>;    ///< Angle \f$ \theta \f$ as a 1-vector.
  using Tangent = Eigen::Matrix<double, 1, 1>;  ///< Angular velocity \f$ \omega \f$ as a 1-vector.

  /// @brief Construct with unit weight (round unit circle).
  SO2CanonicalMetric() : SO2CanonicalMetric(1.0) {}

  /// @brief Construct with an explicit scalar weight.
  /// @param w Positive rotational weight; the norm scales as \f$ \sqrt{w} \f$.
  explicit SO2CanonicalMetric(double w)
      : weight_(w), base_(Eigen::Matrix<double, 1, 1>::Constant(w)) {}

  /// @brief Access the scalar weight \f$ w \f$.
  double weight() const { return weight_; }

  /// @brief Compute the inner product via the wrapped ConstantSPDMetric.
  /// @param p Base point (unused for a constant metric).
  /// @param u First tangent vector.
  /// @param v Second tangent vector.
  /// @return The inner product value \f$ w\, u\, v \f$.
  double inner(const Point& p, const Tangent& u, const Tangent& v) const {
    return base_.inner(p, u, v);
  }

  /// @brief Batched inner product via the wrapped ConstantSPDMetric.
  /// @param p Base point.
  /// @param U Matrix whose columns are tangent vectors.
  /// @param V Matrix whose columns are tangent vectors.
  /// @return \f$ U^\top A \, V \f$.
  Eigen::MatrixXd inner_matrix(const Point& p, const Eigen::MatrixXd& U,
                               const Eigen::MatrixXd& V) const {
    return base_.inner_matrix(p, U, V);
  }

  /// @brief Compute the norm \f$ \|v\| = \sqrt{\langle v, v \rangle} = \sqrt{w}\,|v| \f$.
  /// @param p Base point.
  /// @param v Tangent vector.
  /// @return The norm value.
  double norm(const Point& p, const Tangent& v) const { return riemannian_norm(*this, p, v); }

 private:
  double weight_;              ///< Scalar rotational weight \f$ w \f$.
  ConstantSPDMetric<1> base_;  ///< Wrapped SPD metric with `A = [[w]]`.
};

}  // namespace geodex
