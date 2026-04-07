/// @file metric.hpp
/// @brief HasMetric concept and batched variant for manifolds with a Riemannian inner product.

#pragma once

#include <Eigen/Core>
#include <concepts>

#include "concepts.hpp"

namespace geodex {

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

}  // namespace geodex
