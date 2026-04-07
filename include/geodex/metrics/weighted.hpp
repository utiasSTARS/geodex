/// @file weighted.hpp
/// @brief Uniformly scaled metric wrapper.

#pragma once

#include <Eigen/Core>

#include <geodex/core/metric.hpp>

namespace geodex {

/// @brief Metric wrapper that uniformly scales another metric by a constant factor.
///
/// @details The inner product is:
/// \f$ \langle u, v \rangle_q = \alpha \cdot \langle u, v \rangle^{\mathrm{base}}_q \f$
///
/// @tparam MetricT The base metric type.
template <typename MetricT>
struct WeightedMetric {
  MetricT base_;    ///< The base metric.
  double alpha_;    ///< The scaling factor.

  /// @brief Construct a weighted metric.
  /// @param base The base metric instance.
  /// @param alpha The scaling factor (must be positive).
  WeightedMetric(MetricT base, double alpha) : base_(std::move(base)), alpha_(alpha) {}

  /// @brief Compute the scaled inner product.
  /// @param q Configuration point.
  /// @param u First tangent vector.
  /// @param v Second tangent vector.
  /// @return The scaled inner product value.
  template <typename Point, typename Tangent>
  double inner(const Point& q, const Tangent& u, const Tangent& v) const {
    return alpha_ * base_.inner(q, u, v);
  }

  /// @brief Compute the scaled norm.
  /// @param q Configuration point.
  /// @param v Tangent vector.
  /// @return The scaled norm value.
  template <typename Point, typename Tangent>
  double norm(const Point& q, const Tangent& v) const {
    return riemannian_norm(*this, q, v);
  }

  /// @brief Batched inner product: forwards to the base metric's `inner_matrix`
  /// when available and scales the result by \f$\alpha\f$.
  template <typename Point>
  Eigen::MatrixXd inner_matrix(const Point& q, const Eigen::MatrixXd& U,
                                const Eigen::MatrixXd& V) const
    requires requires { base_.inner_matrix(q, U, V); }
  {
    return alpha_ * base_.inner_matrix(q, U, V);
  }

  /// @brief Forward injectivity radius from the base metric if available.
  double injectivity_radius() const
    requires requires { base_.injectivity_radius(); }
  {
    return base_.injectivity_radius();
  }
};

}  // namespace geodex
