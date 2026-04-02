/// @file interpolation.hpp
/// @brief HasGeodesic concept for manifolds with geodesic interpolation and exp/log maps.

#pragma once

#include <concepts>

#include "concepts.hpp"

namespace geodex {

/// @brief A manifold that provides geodesic interpolation, exponential, and logarithmic maps.
///
/// @details Requires:
/// - `geodesic(p, q, t)` — point at parameter \f$ t \in [0, 1] \f$ along the geodesic from
///   \f$ p \f$ to \f$ q \f$
/// - `exp(p, v)` — exponential map \f$ \exp_p(v) \f$ (or retraction)
/// - `log(p, q)` — logarithmic map \f$ \log_p(q) \f$ (or inverse retraction)
template <typename M>
concept HasGeodesic =
    Manifold<M> && requires(const M m, const typename M::Point p, const typename M::Point q,
                            const typename M::Tangent v, const typename M::Scalar t) {
      { m.geodesic(p, q, t) } -> std::same_as<typename M::Point>;
      { m.exp(p, v) } -> std::same_as<typename M::Point>;
      { m.log(p, q) } -> std::same_as<typename M::Tangent>;
    };

}  // namespace geodex
