/// @file heuristics.hpp
/// @brief Admissible heuristics for motion planning on Riemannian manifolds.

#pragma once

namespace geodex {

/// @brief Euclidean (L2) heuristic between coordinate vectors.
///
/// @details Computes the chord distance \f$ \|a - b\|_2 \f$ between two points.
/// Admissible for any manifold where geodesic distance >= chord distance.
struct EuclideanHeuristic {
  /// @brief Compute \f$ \|a - b\|_2 \f$.
  /// @param a First point.
  /// @param b Second point.
  /// @return The Euclidean distance.
  template <typename PointA, typename PointB>
  auto operator()(const PointA& a, const PointB& b) const -> double {
    return (a - b).norm();
  }
};

}  // namespace geodex
