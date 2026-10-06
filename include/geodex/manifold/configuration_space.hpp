/// @file configuration_space.hpp
/// @brief Configuration space, a manifold with a custom Riemannian metric overlay.

#pragma once

#include <cstdint>

#include <utility>

#include "geodex/algorithm/distance.hpp"
#include "geodex/core/concepts.hpp"
#include "geodex/core/metric.hpp"

namespace geodex {

/// @brief A configuration space that combines a base manifold's topology with a
/// custom Riemannian metric.
///
/// @details The base manifold provides only the topology operations (exp, log,
/// random_point, dim), and `MetricT` provides all geometry (inner, norm, distance).
/// This class does not call the base manifold's own metric. The base must still be a
/// complete `RiemannianManifold`.
///
/// @tparam BaseManifoldT The base manifold type (must provide exp, log, dim, random_point).
/// @tparam MetricT The metric policy type (must provide `inner` and `norm`).
template <typename BaseManifoldT, typename MetricT>
class ConfigurationSpace {
 public:
  using Scalar = typename BaseManifoldT::Scalar;    ///< Scalar type from the base manifold.
  using Point = typename BaseManifoldT::Point;      ///< Point type from the base manifold.
  using Tangent = typename BaseManifoldT::Tangent;  ///< Tangent vector type from the base manifold.
  using SamplerType =
      typename BaseManifoldT::SamplerType;  ///< Sampler policy inherited from the base manifold.

  /// @brief Runtime check whether `log` is the Riemannian logarithm of the custom metric.
  ///
  /// @details Always returns `false`. The base's `log` is the Riemannian log of the
  /// base's native metric, not of the custom metric. `discrete_geodesic` then uses the
  /// finite-difference natural gradient, which follows the energy-minimizing curve
  /// under the custom metric.
  /// @warning `InterpolationSettings::force_log_direction = true` bypasses this and
  /// takes the direction from the base manifold's log. The path then follows the base
  /// metric's geodesic. Use it only when the base log is a reasonable approximation.
  bool has_riemannian_log_runtime() const { return false; }

  /// @brief Construct with a base manifold and a metric.
  /// @param base The base manifold instance.
  /// @param metric The metric policy instance.
  ConfigurationSpace(BaseManifoldT base, MetricT metric)
      : base_(std::move(base)), metric_(std::move(metric)) {}

  /// @name Topology, delegated to the base manifold
  /// @{

  /// @brief Return the intrinsic dimension.
  int dim() const { return base_.dim(); }

  /// @brief Sample a random point from the base manifold.
  Point random_point() const { return base_.random_point(); }

  /// @brief The base manifold's sampler, which random_point() uses.
  decltype(auto) sampler() const
    requires HasSampler<BaseManifoldT>
  {
    return base_.sampler();
  }

  /// @brief Reseed the base manifold's sampler.
  void seed(std::uint64_t s)
    requires requires(BaseManifoldT& b) { b.seed(s); }
  {
    base_.seed(s);
  }

  /// @brief Replace the base manifold's sampler.
  void set_sampler(SamplerType s)
    requires requires(BaseManifoldT& b, SamplerType x) { b.set_sampler(std::move(x)); }
  {
    base_.set_sampler(std::move(s));
  }

  /// @brief Set the base manifold's sampling bounds.
  template <typename Lo, typename Hi>
    requires requires(BaseManifoldT& b, const Lo& lo, const Hi& hi) {
      b.set_sampling_bounds(lo, hi);
    }
  void set_sampling_bounds(const Lo& lo, const Hi& hi) {
    base_.set_sampling_bounds(lo, hi);
  }

  /// @brief Number of unit-cube coordinates that from_unit_cube consumes.
  int unit_cube_dim() const { return base_.unit_cube_dim(); }

  /// @brief Map unit-cube coordinates to a point via the base manifold.
  Point from_unit_cube(Eigen::Ref<const Eigen::VectorXd> u) const {
    return base_.from_unit_cube(u);
  }

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

  /// @name Geometry, delegated to the custom metric
  /// @{

  /// @brief Riemannian inner product from the custom metric.
  Scalar inner(const Point& p, const Tangent& u, const Tangent& v) const {
    return metric_.inner(p, u, v);
  }

  /// @brief Riemannian norm from the custom metric.
  Scalar norm(const Point& p, const Tangent& v) const { return metric_.norm(p, v); }

  /// @brief The custom metric evaluated once at `p`. See `frozen_metric`.
  ///
  /// @details Forwards to the metric's own `metric_at` when it has one. A
  /// metric with only a batched `inner_matrix`, such as a kinetic energy
  /// metric, is frozen as its Gram matrix, and its mass matrix is evaluated once
  /// for all tangent vectors measured at `p`.
  auto metric_at(const Point& p) const
    requires HasMetricAt<MetricT, Point> || MetricHasInnerMatrix<MetricT, Point>
  {
    if constexpr (HasMetricAt<MetricT, Point>) {
      return metric_.metric_at(p);
    } else if constexpr (Tangent::SizeAtCompileTime != Eigen::Dynamic) {
      return gram_metric_at(metric_, p, Tangent::SizeAtCompileTime);
    } else {
      return gram_metric_at(metric_, p, base_.log(p, p).size());
    }
  }

  /// @brief Batched inner product \f$U^\top M(p)\, V\f$ when the custom metric provides it.
  ///
  /// @details Forwards to the metric's `inner_matrix`. Kinetic-energy configuration
  /// spaces evaluate the mass matrix once for all \f$d^2\f$ entries of the
  /// tangent-metric tensor.
  Eigen::MatrixXd inner_matrix(const Point& p, const Eigen::MatrixXd& U,
                               const Eigen::MatrixXd& V) const
    requires requires(const MetricT& m, const Point& q, const Eigen::MatrixXd& A) {
      { m.inner_matrix(q, A, A) } -> std::convertible_to<Eigen::MatrixXd>;
    }
  {
    return metric_.inner_matrix(p, U, V);
  }

  /// @brief The custom metric expressed on coordinate velocities.
  ///
  /// @details Pulls the custom metric back through the base's `coordinate_jacobian`,
  /// \f$ G(q) = J(q)^\top M_{\mathrm{custom}}(q)\, J(q) \f$. The frame belongs to the
  /// base manifold and the metric to the overlay. A conformal overlay such as a
  /// clearance metric still gets a certifiable coordinate bound.
  Eigen::MatrixXd coordinate_metric(const Point& p) const
    requires requires(const BaseManifoldT& b, const MetricT& m, const Point& q,
                      const Eigen::MatrixXd& A) {
      { b.coordinate_jacobian(q) } -> std::convertible_to<Eigen::MatrixXd>;
      { m.inner_matrix(q, A, A) } -> std::convertible_to<Eigen::MatrixXd>;
    }
  {
    const Eigen::MatrixXd J = base_.coordinate_jacobian(p);
    const Eigen::MatrixXd G = metric_.inner_matrix(p, J, J);
    return 0.5 * (G + G.transpose());
  }

  /// @}

  /// @name Derived operations
  /// @{

  /// @brief Geodesic distance via the midpoint approximation.
  Scalar distance(const Point& p, const Point& q) const { return distance_midpoint(*this, p, q); }

  /// @brief Injectivity radius, forwarded from the metric when it has one.
  Scalar injectivity_radius() const
    requires requires(const MetricT& m) {
      { m.injectivity_radius() };
    }
  {
    return metric_.injectivity_radius();
  }

  /// @brief Geodesic interpolation between two points.
  Point geodesic(const Point& p, const Point& q, Scalar t) const { return exp(p, t * log(p, q)); }

  /// @}

  /// @brief Access the base manifold.
  const BaseManifoldT& base() const { return base_; }

  /// @brief Access the metric.
  const MetricT& metric() const { return metric_; }

  /// @brief Forward the base manifold's lower sampling bound when available.
  template <typename B = BaseManifoldT>
    requires requires(const B& b) { b.lo(); }
  auto lo() const {
    return base_.lo();
  }

  /// @brief Forward the base manifold's upper sampling bound when available.
  template <typename B = BaseManifoldT>
    requires requires(const B& b) { b.hi(); }
  auto hi() const {
    return base_.hi();
  }

  /// @brief Forward the base manifold's coordinate periods when available.
  ///
  /// @details Periodicity belongs to the base topology, which the metric overlay
  /// does not change.
  template <typename B = BaseManifoldT>
    requires requires(const B& b) { b.periods(); }
  auto periods() const {
    return base_.periods();
  }

 private:
  BaseManifoldT base_;
  MetricT metric_;
};

}  // namespace geodex
