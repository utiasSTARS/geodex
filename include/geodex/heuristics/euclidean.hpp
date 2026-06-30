/// @file euclidean.hpp
/// @brief Euclidean (L2) heuristic between coordinate vectors.

#pragma once

namespace geodex::heuristics {

/// @brief Euclidean (L2) chord-distance heuristic.
///
/// @details Computes the chord distance \f$ \|a - b\|_2 \f$ between two
/// coordinate vectors. Admissible for any manifold where geodesic distance
/// is bounded below by chord distance — specifically, when
/// \f$ \lambda_{\min}(M(q)) \geq 1 \f$ for all \f$ q \in \mathcal{Q} \f$.
/// When \f$ \lambda_{\min} < 1 \f$ in some direction, the geodesic distance
/// may be less than the chord distance, making this heuristic inadmissible
/// (it over-estimates).
struct Euclidean {
  /// @brief Compute \f$ \|a - b\|_2 \f$.
  /// @param a First point.
  /// @param b Second point.
  /// @return The Euclidean chord distance.
  template <typename PointA, typename PointB>
  auto operator()(const PointA& a, const PointB& b) const -> double {
    return (a - b).norm();
  }
};

}  // namespace geodex::heuristics
