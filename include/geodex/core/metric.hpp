/// @file metric.hpp
/// @brief HasMetric concept and batched variant for manifolds with a Riemannian inner product.

#pragma once

#include <cmath>

#include <concepts>
#include <utility>

#include <Eigen/Core>

#include "concepts.hpp"

namespace geodex {

/// @brief Default Riemannian norm formula \f$ \|v\|_p = \sqrt{\langle v, v \rangle_p} \f$.
///
/// @details Metrics and manifolds with the canonical induced norm forward to this helper.
template <typename HasInner, typename Point, typename Tangent>
inline double riemannian_norm(const HasInner& h, const Point& p, const Tangent& v) {
  return std::sqrt(h.inner(p, v, v));
}

namespace detail {

template <typename M>
concept HasCompileTimeRiemannianLog = requires { requires M::has_riemannian_log; };

template <typename M>
concept HasRuntimeRiemannianLog = requires(const M& m) {
  { m.has_riemannian_log_runtime() } -> std::convertible_to<bool>;
};

}  // namespace detail

/// @brief A manifold that signals at compile time or at run time that `log` is the
/// Riemannian logarithm of its configured metric.
///
/// @details Algorithms call `is_riemannian_log(m)`, which combines the two signals
/// into one boolean, and do not branch on this concept directly.
template <typename M>
concept HasRiemannianLogSignal =
    detail::HasCompileTimeRiemannianLog<M> || detail::HasRuntimeRiemannianLog<M>;

/// @brief Decide whether `log` coincides with the Riemannian logarithm of `m`'s
/// metric, combining compile-time (`M::has_riemannian_log`) and runtime
/// (`m.has_riemannian_log_runtime()`) signals.
///
/// @details The identity \f$\nabla_g(\tfrac{1}{2}\, d_g^2(\cdot, q))(x) = -\log_x^g(q)\f$
/// holds exactly only when `log` is the Riemannian log of `g`. `discrete_geodesic` uses
/// this to choose between the log-based natural gradient and a finite-difference fallback.
/// The compile-time signal takes precedence, and manifolds with neither return `false`.
template <typename M>
constexpr bool is_riemannian_log(const M& m) {
  if constexpr (detail::HasCompileTimeRiemannianLog<M>) {
    return M::has_riemannian_log;
  } else if constexpr (detail::HasRuntimeRiemannianLog<M>) {
    return m.has_riemannian_log_runtime();
  } else {
    return false;
  }
}

/// @brief A manifold that provides a Riemannian inner product and norm.
///
/// @details Requires
/// - `inner(p, u, v)`, the inner product \f$ \langle u, v \rangle_p \f$ at point \f$ p \f$
/// - `norm(p, v)`, the induced norm \f$ \|v\|_p = \sqrt{\langle v, v \rangle_p} \f$
template <typename M>
concept HasMetric =
    Manifold<M> && requires(const M m, const typename M::Point p, const typename M::Tangent u,
                            const typename M::Tangent v) {
      { m.inner(p, u, v) } -> std::convertible_to<typename M::Scalar>;
      { m.norm(p, v) } -> std::convertible_to<typename M::Scalar>;
    };

/// @brief A manifold that exposes a batched inner-product, computing
/// \f$U^\top M(p) V\f$ in a single call.
///
/// @details An optional hook for point-dependent metrics whose tensor \f$M(p)\f$ is
/// expensive, such as a kinetic-energy metric with forward kinematics. Algorithms that
/// build a \f$d \times d\f$ metric tensor in a tangent basis, such as
/// `natural_gradient_fd`, evaluate \f$M(p)\f$ once instead of \f$d^2\f$ times.
template <typename M>
concept HasBatchInnerMatrix =
    Manifold<M> && requires(const M m, const typename M::Point p, const Eigen::MatrixXd U,
                            const Eigen::MatrixXd V) {
      { m.inner_matrix(p, U, V) } -> std::convertible_to<Eigen::MatrixXd>;
    };

/// @brief Check if a metric type provides a batched `inner_matrix` method.
///
/// @details Used by manifold classes to conditionally expose `inner_matrix`
/// without referencing a specific member variable in a requires-clause.
template <typename MetricT, typename Point>
concept MetricHasInnerMatrix =
    requires(const MetricT m, const Point p, const Eigen::MatrixXd U, const Eigen::MatrixXd V) {
      { m.inner_matrix(p, U, V) } -> std::convertible_to<Eigen::MatrixXd>;
    };

/// @brief A metric or manifold whose evaluation at a point can be split off
/// from its application to tangent vectors.
///
/// @details `metric_at(p)` returns a lightweight object whose `inner(u, v)`
/// equals `inner(p, u, v)` and whose `norm(v)` equals `norm(p, v)`. Metrics with
/// an expensive point-dependent part (a distance-field lookup, a mass matrix)
/// evaluate it once there. Algorithms that measure many tangent vectors at one
/// base point go through `frozen_metric`, which works for every metric.
template <typename M, typename Point>
concept HasMetricAt = requires(const M m, const Point p) {
  { m.metric_at(p) };
};

/// @brief `frozen_metric` fallback that forwards to `inner(p, u, v)` of `m`.
///
/// @details A view over `m` that owns a copy of the point. It must not outlive `m`.
template <typename M, typename Point>
class ForwardingMetricAt {
 public:
  /// @brief Wrap `m` at the point `p`.
  ForwardingMetricAt(const M& m, const Point& p) : m_(&m), p_(p) {}

  /// @brief \f$ \langle u, v \rangle_p \f$ through the wrapped `inner`.
  template <typename Tangent>
  double inner(const Tangent& u, const Tangent& v) const {
    return m_->inner(p_, u, v);
  }

  /// @brief \f$ \|v\|_p \f$ through the wrapped `norm`.
  template <typename Tangent>
  double norm(const Tangent& v) const {
    return m_->norm(p_, v);
  }

 private:
  const M* m_;
  Point p_;
};

/// @brief A metric frozen at one point as its Gram matrix \f$ G(p) \f$ in the
/// tangent coordinates, for metrics that expose `inner_matrix`.
class GramMetricAt {
 public:
  /// @brief Take ownership of a Gram matrix.
  explicit GramMetricAt(Eigen::MatrixXd gram) : gram_(std::move(gram)) {}

  /// @brief \f$ u^\top G v \f$.
  template <typename Tangent>
  double inner(const Tangent& u, const Tangent& v) const {
    return u.dot(gram_ * v);
  }

  /// @brief \f$ \sqrt{v^\top G v} \f$.
  template <typename Tangent>
  double norm(const Tangent& v) const {
    return std::sqrt(inner(v, v));
  }

  /// @brief The frozen Gram matrix.
  const Eigen::MatrixXd& gram() const { return gram_; }

 private:
  Eigen::MatrixXd gram_;
};

/// @brief The Gram matrix of `m` at `p`, `m.inner_matrix(p, I, I)`, as a frozen metric.
///
/// @details Exact for every metric that implements `inner_matrix` as
/// \f$ U^\top G(p) V \f$. A product with the identity does not round.
/// @param m The metric.
/// @param p The point the metric is frozen at.
/// @param tangent_size Number of tangent coordinates, which differs from the number
///   of point coordinates on SO(3), SE(3) and the sphere.
template <typename M, typename Point>
  requires MetricHasInnerMatrix<M, Point>
GramMetricAt gram_metric_at(const M& m, const Point& p, const Eigen::Index tangent_size) {
  const Eigen::MatrixXd eye = Eigen::MatrixXd::Identity(tangent_size, tangent_size);
  return GramMetricAt(m.inner_matrix(p, eye, eye));
}

/// @brief The metric of `m` evaluated once at `p`, see `HasMetricAt`.
///
/// @details Uses `m.metric_at(p)` when `m` provides it and a forwarding view
/// otherwise.
template <typename M, typename Point>
auto frozen_metric(const M& m, const Point& p) {
  if constexpr (HasMetricAt<M, Point>) {
    return m.metric_at(p);
  } else {
    return ForwardingMetricAt<M, Point>(m, p);
  }
}

}  // namespace geodex
