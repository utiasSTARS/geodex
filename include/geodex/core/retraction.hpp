/// @file retraction.hpp
/// @brief Retraction concept for generalized exponential/logarithmic maps.

#pragma once

#include <concepts>

namespace geodex {

/// @brief A retraction policy providing generalized exp and log operations.
///
/// @details A retraction is a smooth map \f$ R_p : T_p\mathcal{M} \to \mathcal{M} \f$
/// that approximates the exponential map. It must satisfy \f$ R_p(0) = p \f$ and
/// \f$ \mathrm{D}R_p(0) = \mathrm{id} \f$.
///
/// @tparam R The retraction type.
/// @tparam Point The manifold point type.
/// @tparam Tangent The tangent vector type.
template <typename R, typename Point, typename Tangent>
concept Retraction = requires(const R r, const Point& p, const Point& q, const Tangent& v) {
  /// `retract(p, v)` — map tangent vector \f$ v \in T_p\mathcal{M} \f$ to a point on \f$ \mathcal{M} \f$.
  { r.retract(p, v) } -> std::same_as<Point>;
  /// `inverse_retract(p, q)` — map point \f$ q \f$ to a tangent vector at \f$ p \f$.
  { r.inverse_retract(p, q) } -> std::same_as<Tangent>;
};

}  // namespace geodex
