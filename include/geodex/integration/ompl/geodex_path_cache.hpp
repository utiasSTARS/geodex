/// @file geodex_path_cache.hpp
/// @brief Cached discrete geodesic path for amortized arc-length interpolation.

#pragma once

#include <algorithm>
#include <cstring>
#include <vector>

#include <geodex/algorithm/distance.hpp>
#include <geodex/algorithm/interpolation.hpp>
#include <geodex/core/concepts.hpp>

namespace geodex {
namespace ompl_integration {

/// @brief Caches a discrete geodesic path between two points for efficient
/// arc-length-parameterized lookups.
///
/// @details OMPL's `DiscreteMotionValidator` calls `interpolate(s1, s2, j/n)`
/// for `j = 1..n-1` with the same `(s1, s2)` pair. This cache computes the
/// full discrete geodesic once and serves subsequent lookups via binary search
/// on cumulative arc lengths — O(log K) per query instead of recomputing the
/// geodesic each time.
///
/// @tparam M A type satisfying `RiemannianManifold`.
template <RiemannianManifold M>
class GeodesicPathCache {
 public:
  using Point = typename M::Point;
  using Scalar = typename M::Scalar;

  /// @brief Check whether the cache holds a valid path for the given endpoints.
  /// @param f Start point.
  /// @param t Target point.
  /// @param dim Ambient dimension (number of doubles to compare).
  /// @return True if the cache matches and is valid.
  bool matches(const Point& f, const Point& t, unsigned int dim) const {
    if (!valid_) return false;
    return std::memcmp(from_.data(), f.data(), dim * sizeof(double)) == 0 &&
           std::memcmp(to_.data(), t.data(), dim * sizeof(double)) == 0;
  }

  /// @brief Compute the discrete geodesic from \p f to \p t and cache the result.
  /// @param manifold The manifold instance.
  /// @param f Start point.
  /// @param t Target point.
  /// @param settings Interpolation settings (step_size controls path resolution).
  void compute(const M& manifold, const Point& f, const Point& t,
               const InterpolationSettings& settings) {
    from_ = f;
    to_ = t;
    valid_ = false;

    auto result = discrete_geodesic(manifold, f, t, settings, &interp_cache_);

    // Accept Converged, MaxStepsReached, and DegenerateInput.
    // Reject CutLocus, GradientVanished, StepShrunkToZero — caller falls back.
    if (result.status == InterpolationStatus::CutLocus ||
        result.status == InterpolationStatus::GradientVanished ||
        result.status == InterpolationStatus::StepShrunkToZero) {
      return;
    }

    waypoints_ = std::move(result.path);

    // Ensure the target is the final waypoint for exact t=1 lookup.
    if (waypoints_.empty()) {
      waypoints_.push_back(f);
      waypoints_.push_back(t);
    } else if ((waypoints_.back() - t).norm() > 1e-12) {
      waypoints_.push_back(t);
    }

    // Build cumulative arc-length table.
    const auto n = waypoints_.size();
    cum_arc_.resize(n);
    cum_arc_[0] = 0.0;
    for (std::size_t i = 1; i < n; ++i) {
      cum_arc_[i] =
          cum_arc_[i - 1] + static_cast<double>(manifold.distance(waypoints_[i - 1], waypoints_[i]));
    }
    total_arc_ = cum_arc_.back();

    valid_ = true;
  }

  /// @brief Look up the point at arc-length fraction \p t along the cached path.
  /// @param manifold The manifold instance (for local geodesic interpolation).
  /// @param t Parameter in [0, 1] (0 = start, 1 = target).
  /// @return The interpolated point on the cached geodesic path.
  Point at(const M& manifold, double t) const {
    if (t <= 0.0) return waypoints_.front();
    if (t >= 1.0) return waypoints_.back();

    if (waypoints_.size() <= 1 || total_arc_ <= 0.0) {
      return waypoints_.front();
    }

    const double target_s = t * total_arc_;

    // Binary search: find first element > target_s.
    auto it = std::upper_bound(cum_arc_.begin(), cum_arc_.end(), target_s);
    auto idx = static_cast<int>(it - cum_arc_.begin()) - 1;
    idx = std::max(0, std::min(idx, static_cast<int>(waypoints_.size()) - 2));

    const double seg_len = cum_arc_[idx + 1] - cum_arc_[idx];
    const double t_local = (seg_len > 1e-15) ? (target_s - cum_arc_[idx]) / seg_len : 0.0;

    // Local geodesic between adjacent waypoints — close enough for retraction accuracy.
    return manifold.geodesic(waypoints_[idx], waypoints_[idx + 1], t_local);
  }

  /// @brief Whether the cache holds a valid path.
  bool valid() const { return valid_; }

 private:
  Point from_;
  Point to_;
  std::vector<Point> waypoints_;
  std::vector<double> cum_arc_;
  double total_arc_ = 0.0;
  bool valid_ = false;
  InterpolationCache<M> interp_cache_;
};

}  // namespace ompl_integration
}  // namespace geodex
