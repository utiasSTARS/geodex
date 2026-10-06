/// @file distance_grid.hpp
/// @brief 2D distance transform grid with bilinear interpolation and batch queries.
///
/// Provides:
///   - DistanceGrid: occupancy grid distance field (load from file or construct)
///   - GridSDF: callable wrapper for SDFConformalMetric
///   - InflatedSDF: generic SDF inflation wrapper

#pragma once

#include <cmath>
#include <cstdint>

#include <algorithm>
#include <atomic>
#include <bit>
#include <fstream>
#include <memory>
#include <numbers>
#include <string>
#include <thread>
#include <vector>

#include "geodex/utils/math.hpp"

#ifdef __ARM_NEON
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <immintrin.h>
#endif

namespace geodex::collision {

// ---------------------------------------------------------------------------
// Distance grid
// ---------------------------------------------------------------------------

namespace detail {

/// @brief A flag computed on first use and kept until cleared. Copies carry the value.
class LazyFlag {
 public:
  LazyFlag() = default;
  LazyFlag(const LazyFlag& other) : state_(other.state_.load(std::memory_order_relaxed)) {}
  LazyFlag& operator=(const LazyFlag& other) {
    state_.store(other.state_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    return *this;
  }

  /// @brief The flag, computed by `compute` when it is not known yet.
  template <typename F>
  bool get(const F& compute) const {
    int s = state_.load(std::memory_order_acquire);
    if (s < 0) {
      s = compute() ? 1 : 0;
      state_.store(s, std::memory_order_release);
    }
    return s == 1;
  }

  /// @brief Forget the value. The next `get` computes it again.
  void clear() { state_.store(-1, std::memory_order_release); }

 private:
  mutable std::atomic<int> state_{-1};
};

}  // namespace detail

/// @brief A 2D precomputed distance transform with bilinear interpolation.
///
/// Stores obstacle distances at regular grid points. A query takes world coordinates
/// in meters and returns the interpolated distance to the nearest obstacle, positive
/// in free space and zero or negative in an obstacle.
class DistanceGrid {
 public:
  DistanceGrid() = default;

  /// @brief Construct from raw data.
  /// @param width Grid width in cells.
  /// @param height Grid height in cells.
  /// @param resolution Meters per cell.
  /// @param data Row-major distance values, data[r * width + c].
  DistanceGrid(const int width, const int height, const double resolution, std::vector<double> data)
      : width_(width), height_(height), resolution_(resolution), data_(std::move(data)) {}

  /// @brief Resize for a rebuild and hand back the value array to fill.
  ///
  /// @details Keeps the allocation the grid already holds. A host that rebuilds the
  /// field every cycle does not pay for fresh pages. The returned array holds
  /// `width * height` values in row-major order, unspecified until the caller writes
  /// them.
  std::vector<double>& reset(const int width, const int height, const double resolution) {
    width_ = width;
    height_ = height;
    resolution_ = resolution;
    data_.resize(static_cast<size_t>(width) * height);
    signed_.clear();
    return data_;
  }

  /// @brief Load from the geodex distance transform file format.
  ///
  /// The first line holds `width height resolution`, followed by `height * width`
  /// double values in row-major order.
  /// @return `false` when the file cannot be opened or read.
  bool load(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) return false;
    in >> width_ >> height_ >> resolution_;
    data_.resize(static_cast<size_t>(width_) * height_);
    signed_.clear();
    for (int r = 0; r < height_; ++r) {
      for (int c = 0; c < width_; ++c) {
        in >> data_[static_cast<size_t>(r) * width_ + c];
      }
    }
    return static_cast<bool>(in);
  }

  /// @brief Query distance at world coordinates using bilinear interpolation.
  double distance_at(const double x_m, const double y_m) const {
    const double cx = std::clamp(x_m / resolution_, 0.0, static_cast<double>(width_ - 1));
    const double cy = std::clamp(y_m / resolution_, 0.0, static_cast<double>(height_ - 1));

    const int c0 = static_cast<int>(std::floor(cx));
    const int r0 = static_cast<int>(std::floor(cy));
    const int c1 = std::min(c0 + 1, width_ - 1);
    const int r1 = std::min(r0 + 1, height_ - 1);

    const double fx = cx - c0;
    const double fy = cy - r0;

    const double d00 = data_[static_cast<size_t>(r0) * width_ + c0];
    const double d10 = data_[static_cast<size_t>(r0) * width_ + c1];
    const double d01 = data_[static_cast<size_t>(r1) * width_ + c0];
    const double d11 = data_[static_cast<size_t>(r1) * width_ + c1];

    return (1.0 - fx) * (1.0 - fy) * d00 + fx * (1.0 - fy) * d10 + (1.0 - fx) * fy * d01 +
           fx * fy * d11;
  }

  /// @brief Evaluate the distance at N world-coordinate points.
  ///
  /// On ARM NEON and x86 SSE2, processes 2 points per iteration with
  /// vectorized bilinear interpolation math. Grid value gathering remains
  /// scalar on both architectures.
  void distance_at_batch(const double* x, const double* y, double* out, const int n) const {
    const double inv_res = 1.0 / resolution_;
    const double max_cx = static_cast<double>(width_ - 1);
    const double max_cy = static_cast<double>(height_ - 1);

#ifdef __ARM_NEON
    const float64x2_t vinv = vdupq_n_f64(inv_res);
    const float64x2_t vzero = vdupq_n_f64(0.0);
    const float64x2_t vmcx = vdupq_n_f64(max_cx);
    const float64x2_t vmcy = vdupq_n_f64(max_cy);
    const float64x2_t vone = vdupq_n_f64(1.0);

    const int n2 = n & ~1;
    for (int i = 0; i < n2; i += 2) {
      // Convert to grid coords and clamp.
      float64x2_t vcx = vmulq_f64(vld1q_f64(x + i), vinv);
      float64x2_t vcy = vmulq_f64(vld1q_f64(y + i), vinv);
      vcx = vmaxq_f64(vminq_f64(vcx, vmcx), vzero);
      vcy = vmaxq_f64(vminq_f64(vcy, vmcy), vzero);

      // Floor and fraction.
      const float64x2_t fc0 = vrndmq_f64(vcx);
      const float64x2_t fr0 = vrndmq_f64(vcy);
      const float64x2_t vfx = vsubq_f64(vcx, fc0);
      const float64x2_t vfy = vsubq_f64(vcy, fr0);

      // Extract integer indices with a scalar gather. NEON does not have a gather instruction.
      const int c0_a = static_cast<int>(vgetq_lane_f64(fc0, 0));
      const int c0_b = static_cast<int>(vgetq_lane_f64(fc0, 1));
      const int r0_a = static_cast<int>(vgetq_lane_f64(fr0, 0));
      const int r0_b = static_cast<int>(vgetq_lane_f64(fr0, 1));
      const int c1_a = std::min(c0_a + 1, width_ - 1);
      const int c1_b = std::min(c0_b + 1, width_ - 1);
      const int r1_a = std::min(r0_a + 1, height_ - 1);
      const int r1_b = std::min(r0_b + 1, height_ - 1);

      // Gather 4 grid values per point (8 loads total).
      const float64x2_t d00 = {data_[static_cast<size_t>(r0_a) * width_ + c0_a],
                               data_[static_cast<size_t>(r0_b) * width_ + c0_b]};
      const float64x2_t d10 = {data_[static_cast<size_t>(r0_a) * width_ + c1_a],
                               data_[static_cast<size_t>(r0_b) * width_ + c1_b]};
      const float64x2_t d01 = {data_[static_cast<size_t>(r1_a) * width_ + c0_a],
                               data_[static_cast<size_t>(r1_b) * width_ + c0_b]};
      const float64x2_t d11 = {data_[static_cast<size_t>(r1_a) * width_ + c1_a],
                               data_[static_cast<size_t>(r1_b) * width_ + c1_b]};

      // Bilinear interpolation via NEON FMA.
      const float64x2_t omfx = vsubq_f64(vone, vfx);
      const float64x2_t omfy = vsubq_f64(vone, vfy);
      float64x2_t result = vmulq_f64(vmulq_f64(omfx, omfy), d00);
      result = vfmaq_f64(result, vmulq_f64(vfx, omfy), d10);
      result = vfmaq_f64(result, vmulq_f64(omfx, vfy), d01);
      result = vfmaq_f64(result, vmulq_f64(vfx, vfy), d11);

      vst1q_f64(out + i, result);
    }

    // Handle odd trailing element.
    if (n2 < n) {
      out[n2] = distance_at(x[n2], y[n2]);
    }
#elif defined(__SSE2__)
    const __m128d vinv = _mm_set1_pd(inv_res);
    const __m128d vzero = _mm_setzero_pd();
    const __m128d vmcx = _mm_set1_pd(max_cx);
    const __m128d vmcy = _mm_set1_pd(max_cy);
    const __m128d vone = _mm_set1_pd(1.0);

    const int n2 = n & ~1;
    for (int i = 0; i < n2; i += 2) {
      // Convert to grid coords and clamp.
      __m128d vcx = _mm_mul_pd(_mm_loadu_pd(x + i), vinv);
      __m128d vcy = _mm_mul_pd(_mm_loadu_pd(y + i), vinv);
      vcx = _mm_max_pd(_mm_min_pd(vcx, vmcx), vzero);
      vcy = _mm_max_pd(_mm_min_pd(vcy, vmcy), vzero);

      // Floor and fraction.
      const __m128d fc0 = geodex::utils::geodex_floor_pd(vcx);
      const __m128d fr0 = geodex::utils::geodex_floor_pd(vcy);
      const __m128d vfx = _mm_sub_pd(vcx, fc0);
      const __m128d vfy = _mm_sub_pd(vcy, fr0);

      // Extract integer indices (scalar gather).
      const int c0_a = static_cast<int>(_mm_cvtsd_f64(fc0));
      const int c0_b = static_cast<int>(_mm_cvtsd_f64(_mm_unpackhi_pd(fc0, fc0)));
      const int r0_a = static_cast<int>(_mm_cvtsd_f64(fr0));
      const int r0_b = static_cast<int>(_mm_cvtsd_f64(_mm_unpackhi_pd(fr0, fr0)));
      const int c1_a = std::min(c0_a + 1, width_ - 1);
      const int c1_b = std::min(c0_b + 1, width_ - 1);
      const int r1_a = std::min(r0_a + 1, height_ - 1);
      const int r1_b = std::min(r0_b + 1, height_ - 1);

      // Gather 4 grid values per point (8 loads total).
      // _mm_set_pd(high, low): lane 0 = low (a), lane 1 = high (b).
      const __m128d d00 = _mm_set_pd(data_[static_cast<size_t>(r0_b) * width_ + c0_b],
                                     data_[static_cast<size_t>(r0_a) * width_ + c0_a]);
      const __m128d d10 = _mm_set_pd(data_[static_cast<size_t>(r0_b) * width_ + c1_b],
                                     data_[static_cast<size_t>(r0_a) * width_ + c1_a]);
      const __m128d d01 = _mm_set_pd(data_[static_cast<size_t>(r1_b) * width_ + c0_b],
                                     data_[static_cast<size_t>(r1_a) * width_ + c0_a]);
      const __m128d d11 = _mm_set_pd(data_[static_cast<size_t>(r1_b) * width_ + c1_b],
                                     data_[static_cast<size_t>(r1_a) * width_ + c1_a]);

      // Bilinear interpolation via FMA.
      const __m128d omfx = _mm_sub_pd(vone, vfx);
      const __m128d omfy = _mm_sub_pd(vone, vfy);
      __m128d result = _mm_mul_pd(_mm_mul_pd(omfx, omfy), d00);
      result = geodex::utils::geodex_fmadd_pd(_mm_mul_pd(vfx, omfy), d10, result);
      result = geodex::utils::geodex_fmadd_pd(_mm_mul_pd(omfx, vfy), d01, result);
      result = geodex::utils::geodex_fmadd_pd(_mm_mul_pd(vfx, vfy), d11, result);

      _mm_storeu_pd(out + i, result);
    }

    // Handle odd trailing element.
    if (n2 < n) {
      out[n2] = distance_at(x[n2], y[n2]);
    }
#else
    for (int i = 0; i < n; ++i) {
      out[i] = distance_at(x[i], y[i]);
    }
#endif
  }

  /// @brief Grid width in cells.
  int width() const { return width_; }
  /// @brief Grid height in cells.
  int height() const { return height_; }
  /// @brief Cell size in meters.
  double resolution() const { return resolution_; }

  /// @brief Additive slack of the interpolated field over a 1-Lipschitz bound, with
  /// \f$ |d(p) - d(q)| \le \|p - q\| + \mathrm{slack} \f$ for any two points.
  ///
  /// @details With resolution \f$ h \f$, the bilinear weights keep the corner nodes
  /// within \f$ \sqrt{2}\,h/2 \f$ of the query on average. Node values within \f$ e \f$
  /// of a 1-Lipschitz function give a slack of \f$ 2e + \sqrt{2}\,h \f$. An unsigned
  /// transform has \f$ e = 0 \f$. A grid with a negative node is a signed transform, the
  /// distance to the nearest occupied node minus the distance to the nearest free one,
  /// with \f$ e = \sqrt{2}\,h/2 \f$. The
  /// first call after construction, `load` or `reset` reads every node once.
  double lipschitz_slack() const {
    const bool has_negative = signed_.get([this] {
      return std::any_of(data_.begin(), data_.end(), [](const double v) { return v < 0.0; });
    });
    return (has_negative ? 2.0 : 1.0) * std::numbers::sqrt2 * resolution_;
  }
  /// @brief Raw distance data array.
  const std::vector<double>& data() const { return data_; }

 private:
  int width_ = 0, height_ = 0;
  double resolution_ = 0.05;
  std::vector<double> data_;
  detail::LazyFlag signed_;  ///< some node value is negative
};

// ---------------------------------------------------------------------------
// Grid SDF callable
// ---------------------------------------------------------------------------

/// @brief SDF callable wrapping a DistanceGrid for use with SDFConformalMetric.
///
/// Extracts (x, y) from the configuration point and queries the grid.
class GridSDF {
 public:
  /// @brief Construct from a DistanceGrid pointer.
  explicit GridSDF(const DistanceGrid* grid) : grid_(grid) {}

  /// @brief Evaluate grid-interpolated signed distance at (x, y).
  template <typename Point>
  double operator()(const Point& q) const {
    return grid_->distance_at(q[0], q[1]);
  }

 private:
  const DistanceGrid* grid_;
};

// ---------------------------------------------------------------------------
// Inflated SDF wrapper
// ---------------------------------------------------------------------------

/// @brief Wraps any SDF and subtracts a constant inflation radius.
///
/// Equivalent to a Minkowski expansion of all obstacles by @p inflation, for
/// point-robot queries that account for the robot radius.
template <typename SDFType>
class InflatedSDF {
 public:
  /// @brief Construct with a base SDF and inflation radius.
  InflatedSDF(SDFType sdf, const double inflation) : sdf_(std::move(sdf)), inflation_(inflation) {}

  /// @brief Evaluate inflated signed distance at (x, y).
  template <typename Point>
  double operator()(const Point& q) const {
    return sdf_(q) - inflation_;
  }

  /// @brief Get the inflation radius.
  double inflation() const { return inflation_; }
  /// @brief Get the underlying base SDF.
  const SDFType& base() const { return sdf_; }

 private:
  SDFType sdf_;
  double inflation_;
};

// ---------------------------------------------------------------------------
// Memoized SDF wrapper
// ---------------------------------------------------------------------------

/// @brief Wraps an SDF with a small table of recently queried poses.
///
/// @details Entries are keyed on the exact bit pattern of the first `Dim` coordinates,
/// and a hit returns exactly what the wrapped SDF returned for that pose. Copies share
/// one table, which belongs to the constructing thread. Calls from other threads
/// evaluate the wrapped SDF directly and leave the table alone. Call `clear()` after
/// the wrapped SDF changes, for example after a grid rebuild.
///
/// @tparam SDFType Callable `double(const Point&)`.
/// @tparam Dim Number of leading pose coordinates that identify a query.
/// @tparam Entries Table size, a power of two.
template <typename SDFType, int Dim = 3, int Entries = 256>
class MemoizedSDF {
  static_assert(Dim > 0 && Entries > 0 && (Entries & (Entries - 1)) == 0,
                "MemoizedSDF needs a positive Dim and a power-of-two table size");

 public:
  /// @brief Construct with a base SDF and an empty table owned by this thread.
  explicit MemoizedSDF(SDFType sdf) : sdf_(std::move(sdf)), table_(std::make_shared<Table>()) {}

  /// @brief Evaluate the wrapped SDF at `q`, from the table when `q` was seen recently.
  template <typename Point>
  double operator()(const Point& q) const {
    if (std::this_thread::get_id() != table_->owner) return sdf_(q);
    std::uint64_t key[Dim];
    std::uint64_t h = 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < Dim; ++i) {
      key[i] = std::bit_cast<std::uint64_t>(static_cast<double>(q[i]));
      h = (h ^ key[i]) * 0xC2B2AE3D27D4EB4Full;
      h ^= h >> 29;
    }
    Entry& e = table_->entries[static_cast<std::size_t>(h & (Entries - 1))];
    if (e.used && std::equal(key, key + Dim, e.key)) return e.value;
    e.value = sdf_(q);
    std::copy(key, key + Dim, e.key);
    e.used = true;
    return e.value;
  }

  /// @brief Forget every cached value. Call from the owning thread.
  void clear() const {
    for (auto& e : table_->entries) e.used = false;
  }

  /// @brief Get the underlying base SDF.
  const SDFType& base() const { return sdf_; }

 private:
  struct Entry {
    std::uint64_t key[Dim] = {};
    double value = 0.0;
    bool used = false;
  };
  struct Table {
    std::thread::id owner = std::this_thread::get_id();
    Entry entries[Entries];
  };

  SDFType sdf_;
  std::shared_ptr<Table> table_;
};

}  // namespace geodex::collision
