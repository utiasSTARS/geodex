/// @file footprint_grid_checker.hpp
/// @brief SIMD-accelerated polygon footprint vs. distance grid collision checker.
///
/// Checks whether a polygon footprint at a given SE(2) pose collides with
/// obstacles represented by a distance grid. Unlike point-based checking, this
/// evaluates the entire polygon perimeter against the grid.
///
/// SIMD acceleration pipeline:
///   1. Bounding sphere early-out at the polygon center
///   2. One sincos call rotates all body-frame samples to world frame
///   3. NEON 2-wide batch bilinear interpolation across all samples
///   4. NEON min-reduce to find minimum clearance

#pragma once

#include <cmath>

#include <algorithm>
#include <limits>
#include <vector>

#include <Eigen/Core>

#include "geodex/collision/distance_grid.hpp"
#include "geodex/collision/polygon_footprint.hpp"

#ifdef __ARM_NEON
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <immintrin.h>
#endif

namespace geodex::collision {

/// @brief Polygon-vs-grid collision checker with SIMD acceleration.
///
/// Precomputes a polygon footprint as body-frame perimeter samples. At query
/// time, transforms all samples to world frame and batch-queries the distance
/// grid. Returns either a binary collision result or a continuous signed
/// distance field (minimum grid distance across all samples minus safety margin).
///
/// Thread-safe: uses thread_local scratch buffers for OMPL's parallel motion
/// validation.
class FootprintGridChecker {
 public:
  /// @brief Construct a footprint checker.
  /// @param grid Pointer to the distance grid (must outlive this object).
  /// @param footprint Polygon footprint with precomputed body-frame samples.
  /// @param safety_margin Extra clearance subtracted from distances (default 0).
  FootprintGridChecker(const DistanceGrid* grid, PolygonFootprint footprint,
                       const double safety_margin = 0.0)
      : grid_(grid), footprint_(std::move(footprint)), safety_margin_(safety_margin) {}

  /// @brief Binary collision test. Returns true if the footprint is collision-free.
  bool is_valid(const Eigen::Vector3d& q) const { return min_distance(q) > 0.0; }

  /// @brief SDF: minimum grid distance across all perimeter samples minus safety margin.
  ///
  /// Positive = clear, negative = collision. Suitable for use with
  /// SDFConformalMetric to bias planning away from obstacles.
  template <typename Point>
  double operator()(const Point& q) const {
    return min_distance_impl(q[0], q[1], q[2]);
  }

  /// @brief Get the underlying distance grid.
  const DistanceGrid* grid() const { return grid_; }
  /// @brief Get the polygon footprint.
  const PolygonFootprint& footprint() const { return footprint_; }
  /// @brief Get the safety margin.
  double safety_margin() const { return safety_margin_; }

 private:
  const DistanceGrid* grid_;
  PolygonFootprint footprint_;
  double safety_margin_;

  double min_distance(const Eigen::Vector3d& q) const {
    return min_distance_impl(q[0], q[1], q[2]);
  }

  double min_distance_impl(const double x, const double y, const double theta) const {
    // Early-out: if center distance exceeds bounding radius + margin,
    // the entire footprint is clear.
    const double center_dist = grid_->distance_at(x, y);
    if (center_dist > footprint_.bounding_radius() + safety_margin_) {
      return center_dist - footprint_.bounding_radius() - safety_margin_;
    }
    // If center is deep inside an obstacle, skip the full check.
    if (center_dist < -(footprint_.bounding_radius() + safety_margin_)) {
      return center_dist + footprint_.bounding_radius() - safety_margin_;
    }

    const int np = footprint_.sample_count();
    const int nr = footprint_.sample_count_raw();

    // Thread-local scratch buffers for OMPL thread safety.
    thread_local std::vector<double> wx, wy, dist;
    if (static_cast<int>(wx.size()) < np) {
      wx.resize(np);
      wy.resize(np);
      dist.resize(np);
    }

    // Transform body-frame samples to world frame (1 sincos + NEON rotation).
    footprint_.transform(x, y, theta, wx.data(), wy.data());

    // Batch grid lookup with NEON bilinear interpolation.
    grid_->distance_at_batch(wx.data(), wy.data(), dist.data(), nr);

    // Find minimum distance across all samples.
#ifdef __ARM_NEON
    float64x2_t vmin = vdupq_n_f64(std::numeric_limits<double>::max());
    const int n2 = nr & ~1;
    for (int i = 0; i < n2; i += 2) {
      vmin = vminq_f64(vmin, vld1q_f64(dist.data() + i));
    }
    double min_d = std::min(vgetq_lane_f64(vmin, 0), vgetq_lane_f64(vmin, 1));
    if (n2 < nr) {
      min_d = std::min(min_d, dist[n2]);
    }
#elif defined(__SSE2__)
    __m128d vmin = _mm_set1_pd(std::numeric_limits<double>::max());
    const int n2 = nr & ~1;
    for (int i = 0; i < n2; i += 2) {
      vmin = _mm_min_pd(vmin, _mm_loadu_pd(dist.data() + i));
    }
    double min_d = std::min(_mm_cvtsd_f64(vmin), _mm_cvtsd_f64(_mm_unpackhi_pd(vmin, vmin)));
    if (n2 < nr) {
      min_d = std::min(min_d, dist[n2]);
    }
#else
    double min_d = std::numeric_limits<double>::max();
    for (int i = 0; i < nr; ++i) {
      min_d = std::min(min_d, dist[i]);
    }
#endif

    return min_d - safety_margin_;
  }
};

}  // namespace geodex::collision
