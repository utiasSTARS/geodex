/// @file configuration_space.hpp
/// @brief Configuration space: a manifold with a custom Riemannian metric overlay.

#pragma once

#include <geodex/algorithm/distance.hpp>
#include <geodex/core/concepts.hpp>

namespace geodex {

/// @brief A configuration space that combines a base manifold's topology with a
/// custom Riemannian metric.
///
/// @details Only topology operations (exp, log, random_point, dim) are delegated
/// to the base manifold. All geometry (inner, norm, distance) comes from `MetricT`.
/// The base manifold's own metric is **never called** by this class — it exists only
/// because the base must be a complete `RiemannianManifold` (which requires exp/log).
///
/// @tparam BaseManifoldT The base manifold type (must provide exp, log, dim, random_point).
/// @tparam MetricT The metric policy type (must provide `inner` and `norm`).
template <typename BaseManifoldT, typename MetricT>
class ConfigurationSpace {
  BaseManifoldT base_;
  MetricT metric_;

 public:
  using Scalar = typename BaseManifoldT::Scalar;   ///< Scalar type from the base manifold.
  using Point = typename BaseManifoldT::Point;     ///< Point type from the base manifold.
  using Tangent = typename BaseManifoldT::Tangent; ///< Tangent vector type from the base manifold.

  /// @brief Construct with a base manifold and a metric.
  /// @param base The base manifold instance.
  /// @param metric The metric policy instance.
  ConfigurationSpace(BaseManifoldT base, MetricT metric)
      : base_(std::move(base)), metric_(std::move(metric)) {}

  /// @name Topology — delegated to base manifold
  /// @{

  /// @brief Return the intrinsic dimension.
  int dim() const { return base_.dim(); }

  /// @brief Sample a random point from the base manifold.
  Point random_point() const { return base_.random_point(); }

  /// @brief Exponential map (or retraction) from the base manifold.
  Point exp(const Point& p, const Tangent& v) const { return base_.exp(p, v); }

  /// @brief Logarithmic map (or inverse retraction) from the base manifold.
  Tangent log(const Point& p, const Point& q) const { return base_.log(p, q); }

  /// @brief Project an ambient vector onto the tangent space at \f$ p \f$.
  ///
  /// @details Delegates to the base manifold's projection.
  Tangent project(const Point& p, const Tangent& v) const
    requires requires(const BaseManifoldT& b) {
      { b.project(p, v) } -> std::same_as<Tangent>;
    }
  {
    return base_.project(p, v);
  }

  /// @}

  /// @name Geometry — delegated to custom metric
  /// @{

  /// @brief Riemannian inner product from the custom metric.
  Scalar inner(const Point& p, const Tangent& u, const Tangent& v) const {
    return metric_.inner(p, u, v);
  }

  /// @brief Riemannian norm from the custom metric.
  Scalar norm(const Point& p, const Tangent& v) const { return metric_.norm(p, v); }

  /// @brief Batched inner product \f$U^\top M(p)\, V\f$ when the custom metric provides it.
  ///
  /// @details Forwards to the metric's `inner_matrix`. This is the main
  /// performance hook for kinetic-energy configuration spaces, where the
  /// expensive mass matrix evaluation is amortized over all \f$d^2\f$ entries
  /// of the tangent-metric tensor in a single call.
  Eigen::MatrixXd inner_matrix(const Point& p, const Eigen::MatrixXd& U,
                                const Eigen::MatrixXd& V) const
    requires requires { metric_.inner_matrix(p, U, V); }
  {
    return metric_.inner_matrix(p, U, V);
  }

  /// @}

  /// @name Derived operations
  /// @{

  /// @brief Geodesic distance via the midpoint approximation.
  Scalar distance(const Point& p, const Point& q) const {
    return distance_midpoint(*this, p, q);
  }

  /// @brief Injectivity radius — forwarded from the metric if available.
  Scalar injectivity_radius() const
    requires requires { metric_.injectivity_radius(); }
  {
    return metric_.injectivity_radius();
  }

  /// @brief Geodesic interpolation between two points.
  Point geodesic(const Point& p, const Point& q, Scalar t) const {
    return exp(p, t * log(p, q));
  }

  /// @}

  /// @brief Access the base manifold.
  const BaseManifoldT& base() const { return base_; }

  /// @brief Access the metric.
  const MetricT& metric() const { return metric_; }
};

}  // namespace geodex
