/// @file metric.hpp
/// @brief HasMetric concept and batched variant for manifolds with a Riemannian inner product.

#pragma once

#include <cmath>

#include <concepts>

#include <Eigen/Core>

#include "concepts.hpp"

namespace geodex {

/// @brief Default Riemannian norm formula \f$ \|v\|_p = \sqrt{\langle v, v \rangle_p} \f$.
///
/// @details Shared helper that every metric/manifold with the canonical induced
/// norm can forward to, removing the duplicated `return std::sqrt(inner(p, v, v));`
/// body from each implementation.
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

/// @brief Concept: manifold exposes a compile-time or runtime signal that
/// `log` is the Riemannian logarithm of its currently configured metric.
///
/// @details Algorithms should not branch on this concept directly — use
/// `is_riemannian_log(m)` below, which collapses the two signals into a
/// single boolean.
template <typename M>
concept HasRiemannianLogSignal =
    detail::HasCompileTimeRiemannianLog<M> || detail::HasRuntimeRiemannianLog<M>;

/// @brief Decide whether `log` coincides with the Riemannian logarithm of `m`'s
/// metric, combining compile-time (`M::has_riemannian_log`) and runtime
/// (`m.has_riemannian_log_runtime()`) signals.
///
/// @details On a Riemannian manifold \f$(M, g)\f$ the identity
/// \f$\nabla_g(\tfrac{1}{2}\, d_g^2(\cdot, q))(x) = -\log_x^g(q)\f$
/// holds exactly only when `log` is the Riemannian log of `g`. Algorithms
/// such as `discrete_geodesic` use this resolver to switch between the fast
/// log-based natural gradient and a finite-difference fallback.
///
/// Compile-time signal beats runtime signal. Manifolds with neither return
/// `false` (the FD fallback is always safe).
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
/// @details Requires:
/// - `inner(p, u, v)` — inner product \f$ \langle u, v \rangle_p \f$ at point \f$ p \f$
/// - `norm(p, v)` — induced norm \f$ \|v\|_p = \sqrt{\langle v, v \rangle_p} \f$
template <typename M>
concept HasMetric =
    Manifold<M> && requires(const M m, const typename M::Point p, const typename M::Tangent u,
                            const typename M::Tangent v) {
      { m.inner(p, u, v) } -> std::convertible_to<typename M::Scalar>;
      { m.norm(p, v) } -> std::convertible_to<typename M::Scalar>;
    };

/// @brief A manifold that exposes a batched inner-product, computing
/// \f$U^\top M(p) V\f$ in one call.
///
/// @details This is an optional optimization hook for point-dependent metrics
/// where the expensive part is evaluating the metric tensor \f$M(p)\f$
/// (e.g., forward kinematics for a kinetic-energy metric). Providing
/// `inner_matrix` allows algorithms that compute a \f$d \times d\f$ metric tensor
/// in a tangent basis (`natural_gradient_fd` is the canonical consumer) to
/// evaluate \f$M(p)\f$ **once** instead of \f$d^2\f$ times.
///
/// Algorithms that only need pointwise evaluation use the scalar `inner` path
/// and do not require metrics to provide `inner_matrix`.
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

}  // namespace geodex
