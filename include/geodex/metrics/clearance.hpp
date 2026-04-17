/// @file clearance.hpp
/// @brief SDF-based conformal metric for obstacle clearance biasing.

#pragma once

#include <cmath>

#include <type_traits>

#include <Eigen/Core>

#include "geodex/core/metric.hpp"

namespace geodex {

/// @brief Conformal metric that scales a base metric by an obstacle proximity field.
///
/// @details The inner product is:
/// \f$ \langle u, v \rangle_q = c(q) \cdot \langle u, v \rangle^{\mathrm{base}}_q \f$
///
/// where the conformal factor is:
/// \f$ c(q) = 1 + \kappa \exp(-\beta \cdot \mathrm{sdf}(q)) \f$
///
/// This makes the metric expensive near obstacles (low SDF) and unmodified far
/// away (high SDF), causing geodesics to naturally maintain clearance. With
/// multiple obstacles, geodesics follow channels equidistant from obstacle
/// surfaces.
///
/// The SDF callable should return a smooth signed distance field where positive
/// values indicate free space. For circular obstacles, use log-sum-exp
/// smooth-min to avoid gradient discontinuities at Voronoi boundaries. For
/// grid-based distance transforms, use bilinear interpolation.
///
/// @tparam BaseMetricT The base metric type (e.g., `SE2LeftInvariantMetric`).
/// @tparam SDFFn Callable with signature `double(const Point&)` returning the
///               signed distance to the nearest obstacle surface.
template <typename BaseMetricT, typename SDFFn>
class SDFConformalMetric {
 public:
  /// @brief Construct an SDF-based conformal metric.
  /// @param base The base metric to scale.
  /// @param sdf Callable returning signed distance (positive = free space).
  /// @param kappa Strength of obstacle repulsion (higher = more clearance).
  /// @param beta Falloff rate (higher = tighter to obstacle surfaces).
  SDFConformalMetric(BaseMetricT base, SDFFn sdf, double kappa = 5.0, double beta = 3.0)
      : base_(std::move(base)), sdf_(std::move(sdf)), kappa_(kappa), beta_(beta) {}

  /// @brief Compute the scaled inner product \f$ c(q) \langle u, v \rangle^{\mathrm{base}}_q \f$.
  template <typename Point, typename Tangent>
  double inner(const Point& q, const Tangent& u, const Tangent& v) const {
    return conformal_factor(q) * base_.inner(q, u, v);
  }

  /// @brief Compute the scaled norm \f$ \|v\|_q = \sqrt{c(q)} \|v\|^{\mathrm{base}}_q \f$.
  template <typename Point, typename Tangent>
  double norm(const Point& q, const Tangent& v) const {
    return riemannian_norm(*this, q, v);
  }

  /// @brief Batched inner product scaled by the conformal factor.
  template <typename Point>
  Eigen::MatrixXd inner_matrix(const Point& q, const Eigen::MatrixXd& U,
                               const Eigen::MatrixXd& V) const
    requires requires(const BaseMetricT& m, const Point& p, const Eigen::MatrixXd& A) {
      { m.inner_matrix(p, A, A) } -> std::convertible_to<Eigen::MatrixXd>;
    }
  {
    return conformal_factor(q) * base_.inner_matrix(q, U, V);
  }

  /// @brief Access the base metric.
  const BaseMetricT& base() const { return base_; }

  /// @brief Access the SDF callable.
  const SDFFn& sdf() const { return sdf_; }

  /// @brief Get the strength parameter.
  double kappa() const { return kappa_; }

  /// @brief Get the falloff rate parameter.
  double beta() const { return beta_; }

  /// @brief Evaluate the conformal factor at a configuration.
  /// @param q Configuration point.
  /// @return \f$ c(q) = 1 + \kappa \exp(-\beta \cdot \mathrm{sdf}(q)) \f$.
  template <typename Point>
  double conformal_factor(const Point& q) const {
    const double d = sdf_(q);
    return 1.0 + kappa_ * std::exp(-beta_ * d);
  }

 private:
  BaseMetricT base_;
  SDFFn sdf_;
  double kappa_;
  double beta_;
};

}  // namespace geodex
