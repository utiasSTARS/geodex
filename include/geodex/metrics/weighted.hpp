/// @file weighted.hpp
/// @brief Scaled metric wrapper with constant or configuration-dependent alpha.

#pragma once

#include <type_traits>

#include <Eigen/Core>

#include "geodex/core/metric.hpp"

namespace geodex {

/// @brief Metric wrapper that scales another metric by a constant or
/// configuration-dependent factor \f$\alpha\f$.
///
/// @details The inner product is:
/// \f$ \langle u, v \rangle_q = \alpha(q) \cdot \langle u, v \rangle^{\mathrm{base}}_q \f$
///
/// The alpha parameter can be either:
/// - a constant `double` (e.g. `WeightedMetric{base, 3.0}`)
/// - a callable `Fn(q) -> double` for configuration-dependent scaling
///   (used by `JacobiMetric`, region-avoiding metrics, etc.).
///
/// @tparam MetricT The base metric type.
/// @tparam AlphaT  Scaling factor type — either `double` or a callable.
template <typename MetricT, typename AlphaT = double>
class WeightedMetric {
 public:
  /// @brief Construct a weighted metric.
  /// @param base The base metric to scale.
  /// @param alpha Scaling factor — constant `double` or callable `Fn(q) -> double`.
  WeightedMetric(MetricT base, AlphaT alpha) : base_(std::move(base)), alpha_(std::move(alpha)) {}

  /// @brief Compute the scaled inner product \f$\alpha(q) \langle u, v
  /// \rangle^{\mathrm{base}}_q\f$.
  /// @param q Configuration point.
  /// @param u First tangent vector.
  /// @param v Second tangent vector.
  /// @return The scaled inner product value.
  template <typename Point, typename Tangent>
  double inner(const Point& q, const Tangent& u, const Tangent& v) const {
    return evaluate_alpha(q) * base_.inner(q, u, v);
  }

  /// @brief Compute the scaled norm \f$\|v\|_q = \sqrt{\alpha(q) \langle v, v
  /// \rangle^{\mathrm{base}}_q}\f$.
  /// @param q Configuration point.
  /// @param v Tangent vector.
  /// @return The scaled norm value.
  template <typename Point, typename Tangent>
  double norm(const Point& q, const Tangent& v) const {
    return riemannian_norm(*this, q, v);
  }

  /// @brief Batched inner product: forwards to the base metric's `inner_matrix`
  /// when available and scales the result by \f$\alpha(q)\f$.
  template <typename Point>
  Eigen::MatrixXd inner_matrix(const Point& q, const Eigen::MatrixXd& U,
                               const Eigen::MatrixXd& V) const
    requires requires(const MetricT& m, const Point& p, const Eigen::MatrixXd& A) {
      { m.inner_matrix(p, A, A) } -> std::convertible_to<Eigen::MatrixXd>;
    }
  {
    return evaluate_alpha(q) * base_.inner_matrix(q, U, V);
  }

  /// @brief Forward injectivity radius from the base metric — only valid for
  /// constant-scalar \f$\alpha\f$ (a config-dependent alpha breaks the
  /// uniform-scaling guarantee).
  double injectivity_radius() const
    requires(std::is_arithmetic_v<AlphaT> &&
             requires(const MetricT& m) {
               { m.injectivity_radius() };
             })
  {
    return base_.injectivity_radius();
  }

  /// @brief Access the base metric.
  const MetricT& base() const { return base_; }

  /// @brief Access the scaling factor.
  const AlphaT& alpha() const { return alpha_; }

 private:
  /// @brief Evaluate \f$\alpha(q)\f$ — selects between constant scalar and
  /// callable at compile time.
  template <typename Point>
  double evaluate_alpha(const Point& q) const {
    if constexpr (std::is_invocable_r_v<double, const AlphaT&, const Point&>) {
      return alpha_(q);
    } else {
      return static_cast<double>(alpha_);
    }
  }

  MetricT base_;  ///< The base metric.
  AlphaT alpha_;  ///< The scaling factor (constant or callable).
};

}  // namespace geodex
