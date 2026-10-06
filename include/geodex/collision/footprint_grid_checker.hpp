/// @file footprint_grid_checker.hpp
/// @brief Collision check of a polygon footprint against a distance grid.
///
/// Checks the whole perimeter of a polygon footprint at an SE(2) pose against a distance
/// grid. A bounding circle at the polygon center settles clear and deep poses first. The
/// grid lookups and the minimum run two samples at a time with NEON on ARM and SSE2 on
/// x86, and one at a time elsewhere.

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
/// The checker is thread-safe. It uses thread_local scratch buffers for OMPL's
/// parallel motion validation.
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

  /// @brief Minimum grid distance across all perimeter samples minus the safety margin.
  ///
  /// Positive when clear and negative in collision. When the center is farther from
  /// the boundary than the bounding radius plus the grid's `lipschitz_slack()`, the
  /// center alone decides. The value then has the sign of the full evaluation and is a
  /// lower bound when clear. The grid must be an unsigned or signed distance transform
  /// (see `DistanceGrid::lipschitz_slack`).
  template <typename Point>
  double operator()(const Point& q) const {
    return min_distance_impl(q[0], q[1], q[2]);
  }

  /// @brief Footprint clearance that is exact only below `cap`.
  ///
  /// @details Returns the value of `operator()` when it lies in (0, `cap`). Above
  /// `cap` it returns a lower bound of at least `cap`, and in collision a value of at
  /// most 0 whose sign alone is meaningful. It skips in one step a run of samples
  /// whose Lipschitz bound stays above the running minimum, and reads far fewer
  /// samples than `operator()`.
  template <typename Point>
  double min_distance_capped(const Point& q, const double cap) const {
    const double x = q[0], y = q[1];
    const double center_dist = grid_->distance_at(x, y);
    const double reach = footprint_.bounding_radius() + grid_->lipschitz_slack();
    if (center_dist > reach + safety_margin_) return center_dist - reach - safety_margin_;
    if (center_dist < -(reach + safety_margin_)) return center_dist + reach - safety_margin_;
    const int nr = footprint_.sample_count_raw();
    const double gap = footprint_.max_sample_gap();
    const double slack = grid_->lipschitz_slack();
    const double thresh_cap = safety_margin_ + cap;
    double ct, st;
    utils::sincos(q[2], &st, &ct);
    double best = std::numeric_limits<double>::max();
    int k = 0;
    while (k < nr) {
      const double bx = footprint_.body_x(k), by = footprint_.body_y(k);
      const double d = grid_->distance_at(ct * bx - st * by + x, st * bx + ct * by + y);
      if (d < best) best = d;
      if (best <= safety_margin_) break;
      const double level = std::min(best, thresh_cap);
      const int skip = gap > 0.0 ? static_cast<int>((d - slack - level) / gap) : 0;
      if (skip > 0) {
        const double bound = d - skip * gap - slack;  // every skipped sample is at least this
        if (bound < best) best = bound;
        k += skip + 1;
      } else {
        ++k;
      }
    }
    return best - safety_margin_;
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
    // Early-outs. Every perimeter sample lies within the bounding radius of the
    // center, and the grid changes by at most that distance plus its slack. A
    // center this far out is clear, and a center this deep in collides.
    const double center_dist = grid_->distance_at(x, y);
    const double reach = footprint_.bounding_radius() + grid_->lipschitz_slack();
    if (center_dist > reach + safety_margin_) return center_dist - reach - safety_margin_;
    if (center_dist < -(reach + safety_margin_)) return center_dist + reach - safety_margin_;

    const int np = footprint_.sample_count();
    const int nr = footprint_.sample_count_raw();

    // Thread-local scratch buffers.
    thread_local std::vector<double> wx, wy, dist;
    if (static_cast<int>(wx.size()) < np) {
      wx.resize(np);
      wy.resize(np);
      dist.resize(np);
    }

    // Body-frame samples to the world frame, with one sincos.
    footprint_.transform(x, y, theta, wx.data(), wy.data());

    // Bilinear grid lookup of every sample.
    grid_->distance_at_batch(wx.data(), wy.data(), dist.data(), nr);

    // Smallest distance over the samples.
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
