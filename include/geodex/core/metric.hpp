/// @file metric.hpp
/// @brief HasMetric concept for manifolds with a Riemannian inner product.

#pragma once

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

}  // namespace geodex
