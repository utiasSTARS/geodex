/// @file zero.hpp
/// @brief Zero heuristic, trivially admissible and completely uninformative.

#pragma once

namespace geodex::heuristics {

/// @brief Zero heuristic that returns \f$ h(a,b) = 0 \f$ for every pair.
///
/// @details The weakest possible admissible heuristic: admissible for any
/// non-negative distance, but carries no information. When used with an
/// informed planner, the informed set degenerates to the full configuration
/// space (uniform sampling), no vertices are pruned, and vertex ordering
/// becomes uninformed.
struct Zero {
  /// @brief Compute \f$ h(a,b) = 0 \f$ for any two points.
  /// @return 0.
  template <typename PointA, typename PointB>
  auto operator()(const PointA&, const PointB&) const -> double {
    return 0.0;
  }
};

}  // namespace geodex::heuristics
