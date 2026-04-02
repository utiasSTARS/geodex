/// @file distance.hpp
/// @brief HasDistance concept for manifolds with geodesic distance.

#pragma once

#include <concepts>

#include "concepts.hpp"

namespace geodex {

/// @brief A manifold that provides geodesic distance between points.
///
/// @details Requires `distance(p, q)` returning the geodesic distance
/// \f$ d(p, q) \f$ between points \f$ p \f$ and \f$ q \f$.
template <typename M>
concept HasDistance =
    Manifold<M> && requires(const M m, const typename M::Point p, const typename M::Point q) {
      { m.distance(p, q) } -> std::convertible_to<typename M::Scalar>;
    };

}  // namespace geodex
