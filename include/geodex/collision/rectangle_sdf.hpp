/// @file rectangle_sdf.hpp
/// @brief Oriented rectangle obstacles: SDF, SAT collision, and SIMD acceleration.
///
/// Provides:
///   - RectObstacle: oriented rectangle descriptor
///   - rect_corners / rects_overlap: SAT-based overlap test
///   - RectObstacleSoA: NEON-friendly Structure-of-Arrays layout
///   - RectSmoothSDF: SIMD-accelerated smooth-min SDF with bounding-sphere
///     early-out, branchless signed distance, and fast_exp log-sum-exp

#pragma once

#include <cmath>

#include <array>
#include <limits>
#include <vector>

#include <Eigen/Core>

#include "geodex/utils/math.hpp"

#ifdef __ARM_NEON
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <immintrin.h>
#endif

namespace geodex::collision {

// ---------------------------------------------------------------------------
// Oriented rectangle obstacle
// ---------------------------------------------------------------------------

/// @brief An oriented rectangle obstacle defined by center, rotation, and half-extents.
struct RectObstacle {
  double cx, cy, theta;
  double half_length, half_width;
};

/// @brief Compute the 4 corners of an oriented rectangle.
inline std::array<Eigen::Vector2d, 4> rect_corners(const RectObstacle& r) {
  const double ct = std::cos(r.theta), st = std::sin(r.theta);
  const double hl = r.half_length, hw = r.half_width;
  return {{
      {r.cx + ct * (-hl) - st * (-hw), r.cy + st * (-hl) + ct * (-hw)},
      {r.cx + ct * (hl)-st * (-hw), r.cy + st * (hl) + ct * (-hw)},
      {r.cx + ct * (hl)-st * (hw), r.cy + st * (hl) + ct * (hw)},
      {r.cx + ct * (-hl) - st * (hw), r.cy + st * (-hl) + ct * (hw)},
  }};
}

/// @brief Separating Axis Theorem (SAT) overlap test for two oriented rectangles.
/// @return true if the rectangles overlap (collide).
inline bool rects_overlap(const RectObstacle& a, const RectObstacle& b) {
  auto corners_a = rect_corners(a);
  auto corners_b = rect_corners(b);

  auto test_axes = [&](const RectObstacle& r) -> bool {
    const double ct = std::cos(r.theta), st = std::sin(r.theta);
    Eigen::Vector2d axes[2] = {{ct, st}, {-st, ct}};
    for (const auto& axis : axes) {
      double min_a = std::numeric_limits<double>::max();
      double max_a = std::numeric_limits<double>::lowest();
      double min_b = std::numeric_limits<double>::max();
      double max_b = std::numeric_limits<double>::lowest();
      for (const auto& c : corners_a) {
        const double proj = axis.dot(c);
        min_a = std::min(min_a, proj);
        max_a = std::max(max_a, proj);
      }
      for (const auto& c : corners_b) {
        const double proj = axis.dot(c);
        min_b = std::min(min_b, proj);
        max_b = std::max(max_b, proj);
      }
      if (max_a < min_b || max_b < min_a) return true;  // separated
    }
    return false;
  };

  return !test_axes(a) && !test_axes(b);
}

// ---------------------------------------------------------------------------
// SoA layout for NEON-friendly rectangle data
// ---------------------------------------------------------------------------

/// @brief Structure-of-Arrays layout for oriented rectangle obstacles.
///
/// Precomputes sin/cos of each obstacle's orientation and bounding sphere
/// radius. Arrays are padded to a multiple of 2 for NEON 2-wide processing.
class RectObstacleSoA {
 public:
  /// @brief Build SoA arrays from a vector of obstacles with skip distance.
  void build(const std::vector<RectObstacle>& obstacles, const double skip_dist) {
    n_ = static_cast<int>(obstacles.size());
    n_padded_ = (n_ + 1) & ~1;
    auto alloc = [&](std::vector<double>& v, const double fill) { v.resize(n_padded_, fill); };
    alloc(cx_, 0.0);
    alloc(cy_, 0.0);
    alloc(ct_, 1.0);
    alloc(st_, 0.0);
    alloc(half_length_, 0.0);
    alloc(half_width_, 0.0);
    alloc(bounding_r2_, 0.0);
    for (int i = 0; i < n_; ++i) {
      const auto& o = obstacles[i];
      ct_[i] = std::cos(o.theta);
      st_[i] = std::sin(o.theta);
      cx_[i] = o.cx;
      cy_[i] = o.cy;
      half_length_[i] = o.half_length;
      half_width_[i] = o.half_width;
      const double diag = std::sqrt(o.half_length * o.half_length + o.half_width * o.half_width);
      const double br = diag + skip_dist;
      bounding_r2_[i] = br * br;
    }
  }

  /// @brief Number of actual obstacles.
  int count() const { return n_; }
  /// @brief Number of obstacles including SIMD padding.
  int padded_count() const { return n_padded_; }
  /// @brief Obstacle center x-coordinates.
  const double* cx() const { return cx_.data(); }
  /// @brief Obstacle center y-coordinates.
  const double* cy() const { return cy_.data(); }
  /// @brief Precomputed cos(theta) for each obstacle.
  const double* ct() const { return ct_.data(); }
  /// @brief Precomputed sin(theta) for each obstacle.
  const double* st() const { return st_.data(); }
  /// @brief Half-length of each obstacle.
  const double* half_length() const { return half_length_.data(); }
  /// @brief Half-width of each obstacle.
  const double* half_width() const { return half_width_.data(); }
  /// @brief Squared bounding sphere radius for early-out.
  const double* bounding_r2() const { return bounding_r2_.data(); }

 private:
  std::vector<double> cx_, cy_, ct_, st_;
  std::vector<double> half_length_, half_width_, bounding_r2_;
  int n_ = 0, n_padded_ = 0;
};

// ---------------------------------------------------------------------------
// SIMD-accelerated smooth-min SDF for oriented rectangles
// ---------------------------------------------------------------------------

/// @brief Fused SDF + log-sum-exp for oriented rectangles with SIMD acceleration.
///
/// SIMD strategy (ARM NEON, 2-wide float64):
///   - SoA layout: cache-line-friendly, NEON-loadable
///   - 2-wide processing: 2 obstacles per iteration, branchless via vbslq
///   - Bounding sphere early-out: skip obstacles where center distance exceeds
///     diagonal + skip_dist (where exp(-beta*d) ~ 0)
///   - Inflation applied inside the SIMD loop (no separate wrapper needed)
///   - fast_exp_neon for the log-sum-exp accumulation
///
/// Scalar fallback provided for platforms without SIMD.
/// x86 SSE2 path uses 2-wide float64 processing with optional SSE4.1/FMA.
class RectSmoothSDF {
 public:
  /// @brief Construct from obstacles with smoothing and optional inflation.
  RectSmoothSDF(std::vector<RectObstacle> obstacles, const double beta = 20.0,
                const double inflation = 0.0)
      : obstacles_(std::move(obstacles)),
        beta_(beta),
        skip_dist_(20.0 / beta),
        inflation_(inflation) {
    soa_.build(obstacles_, skip_dist_);
    simd_buf_.resize(soa_.padded_count());
  }

  /// @brief Evaluate smooth signed distance at point (x, y).
  template <typename Point>
  double operator()(const Point& q) const {
    const double px = q[0], py = q[1];
    const int np = soa_.padded_count();

#ifdef __ARM_NEON
    return eval_neon(px, py, np);
#elif defined(__SSE2__)
    return eval_sse2(px, py, np);
#else
    return eval_scalar(px, py, np);
#endif
  }

  /// @brief Get the rectangle obstacles.
  const std::vector<RectObstacle>& obstacles() const { return obstacles_; }
  /// @brief Get the log-sum-exp smoothing parameter.
  double beta() const { return beta_; }
  /// @brief Get the inflation radius.
  double inflation() const { return inflation_; }

 private:
  std::vector<RectObstacle> obstacles_;
  RectObstacleSoA soa_;
  double beta_;
  double skip_dist_;
  double inflation_;
  mutable std::vector<double> simd_buf_;  ///< Pre-allocated SIMD scratch buffer.

#ifdef __ARM_NEON
  double eval_neon(const double px, const double py, const int np) const {
    const float64x2_t vpx = vdupq_n_f64(px);
    const float64x2_t vpy = vdupq_n_f64(py);
    const float64x2_t vzero = vdupq_n_f64(0.0);
    const float64x2_t vbeta = vdupq_n_f64(beta_);
    const float64x2_t vskip = vdupq_n_f64(skip_dist_);
    const float64x2_t vinfl = vdupq_n_f64(inflation_);
    const uint64x2_t abs_mask = vdupq_n_u64(0x7FFFFFFFFFFFFFFFULL);

    double* neg_beta_d = simd_buf_.data();
    int total = 0;

    for (int i = 0; i < np; i += 2) {
      const float64x2_t vcx = vld1q_f64(soa_.cx() + i);
      const float64x2_t vcy = vld1q_f64(soa_.cy() + i);

      // Bounding sphere early-out.
      const float64x2_t dx = vsubq_f64(vpx, vcx);
      const float64x2_t dy = vsubq_f64(vpy, vcy);
      const float64x2_t dist2 = vaddq_f64(vmulq_f64(dx, dx), vmulq_f64(dy, dy));
      const float64x2_t vbr2 = vld1q_f64(soa_.bounding_r2() + i);
      const uint64x2_t in_range = vcltq_f64(dist2, vbr2);
      if (vgetq_lane_u64(in_range, 0) == 0 && vgetq_lane_u64(in_range, 1) == 0) continue;

      // Local coordinates via rotation.
      const float64x2_t vct = vld1q_f64(soa_.ct() + i);
      const float64x2_t vst = vld1q_f64(soa_.st() + i);
      const float64x2_t lx = vaddq_f64(vmulq_f64(vct, dx), vmulq_f64(vst, dy));
      const float64x2_t ly = vsubq_f64(vmulq_f64(vct, dy), vmulq_f64(vst, dx));

      const float64x2_t alx = vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(lx), abs_mask));
      const float64x2_t aly = vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(ly), abs_mask));

      const float64x2_t vhl = vld1q_f64(soa_.half_length() + i);
      const float64x2_t vhw = vld1q_f64(soa_.half_width() + i);

      // Exterior distance: max(|l| - half, 0).
      const float64x2_t ex = vmaxq_f64(vsubq_f64(alx, vhl), vzero);
      const float64x2_t ey = vmaxq_f64(vsubq_f64(aly, vhw), vzero);
      const float64x2_t ext2 = vaddq_f64(vmulq_f64(ex, ex), vmulq_f64(ey, ey));
      const float64x2_t ext_d = vsqrtq_f64(ext2);

      // Interior distance: -min(hl - |lx|, hw - |ly|).
      const float64x2_t ix = vsubq_f64(vhl, alx);
      const float64x2_t iy = vsubq_f64(vhw, aly);
      const float64x2_t int_d = vnegq_f64(vminq_f64(ix, iy));

      // Branchless select: ext2 > 0 → exterior, else interior.
      const uint64x2_t is_outside = vcgtq_f64(ext2, vzero);
      float64x2_t d = vbslq_f64(is_outside, ext_d, int_d);

      // Inflation: subtract so clearance triggers at robot edge.
      d = vsubq_f64(d, vinfl);

      // Mask out-of-range obstacles to skip_dist.
      d = vbslq_f64(in_range, d, vskip);

      const float64x2_t nbd = vnegq_f64(vmulq_f64(vbeta, d));
      vst1q_f64(&neg_beta_d[total], nbd);
      total += 2;
    }

    if (total == 0) return skip_dist_;

    // Find global max for numerical stability.
    double global_max = std::numeric_limits<double>::lowest();
    for (int j = 0; j < total; j += 2) {
      const float64x2_t v = vld1q_f64(&neg_beta_d[j]);
      global_max = std::max(global_max, std::max(vgetq_lane_f64(v, 0), vgetq_lane_f64(v, 1)));
    }

    // Sum exp(nbd - max) using NEON fast_exp.
    const float64x2_t vgmax = vdupq_n_f64(global_max);
    float64x2_t vsum = vdupq_n_f64(0.0);
    for (int j = 0; j < total; j += 2) {
      const float64x2_t shifted = vsubq_f64(vld1q_f64(&neg_beta_d[j]), vgmax);
      vsum = vaddq_f64(vsum, utils::fast_exp(shifted));
    }
    const double sum = vgetq_lane_f64(vsum, 0) + vgetq_lane_f64(vsum, 1);

    return -(global_max + std::log(sum)) / beta_;
  }
#endif  // __ARM_NEON

#ifdef __SSE2__
  double eval_sse2(const double px, const double py, const int np) const {
    const __m128d vpx = _mm_set1_pd(px);
    const __m128d vpy = _mm_set1_pd(py);
    const __m128d vzero = _mm_setzero_pd();
    const __m128d vbeta = _mm_set1_pd(beta_);
    const __m128d vskip = _mm_set1_pd(skip_dist_);
    const __m128d vinfl = _mm_set1_pd(inflation_);
    const __m128d abs_mask = _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFFLL));
    const __m128d sign_mask =
        _mm_castsi128_pd(_mm_set1_epi64x(static_cast<int64_t>(0x8000000000000000ULL)));

    double* neg_beta_d = simd_buf_.data();
    int total = 0;

    for (int i = 0; i < np; i += 2) {
      const __m128d vcx = _mm_loadu_pd(soa_.cx() + i);
      const __m128d vcy = _mm_loadu_pd(soa_.cy() + i);

      // Bounding sphere early-out.
      const __m128d dx = _mm_sub_pd(vpx, vcx);
      const __m128d dy = _mm_sub_pd(vpy, vcy);
      const __m128d dist2 = _mm_add_pd(_mm_mul_pd(dx, dx), _mm_mul_pd(dy, dy));
      const __m128d vbr2 = _mm_loadu_pd(soa_.bounding_r2() + i);
      const __m128d in_range = _mm_cmplt_pd(dist2, vbr2);
      if (_mm_movemask_pd(in_range) == 0) continue;

      // Local coordinates via rotation.
      const __m128d vct = _mm_loadu_pd(soa_.ct() + i);
      const __m128d vst = _mm_loadu_pd(soa_.st() + i);
      const __m128d lx = _mm_add_pd(_mm_mul_pd(vct, dx), _mm_mul_pd(vst, dy));
      const __m128d ly = _mm_sub_pd(_mm_mul_pd(vct, dy), _mm_mul_pd(vst, dx));

      const __m128d alx = _mm_and_pd(lx, abs_mask);
      const __m128d aly = _mm_and_pd(ly, abs_mask);

      const __m128d vhl = _mm_loadu_pd(soa_.half_length() + i);
      const __m128d vhw = _mm_loadu_pd(soa_.half_width() + i);

      // Exterior distance: max(|l| - half, 0).
      const __m128d ex = _mm_max_pd(_mm_sub_pd(alx, vhl), vzero);
      const __m128d ey = _mm_max_pd(_mm_sub_pd(aly, vhw), vzero);
      const __m128d ext2 = _mm_add_pd(_mm_mul_pd(ex, ex), _mm_mul_pd(ey, ey));
      const __m128d ext_d = _mm_sqrt_pd(ext2);

      // Interior distance: -min(hl - |lx|, hw - |ly|).
      const __m128d ix = _mm_sub_pd(vhl, alx);
      const __m128d iy = _mm_sub_pd(vhw, aly);
      const __m128d int_d = _mm_xor_pd(_mm_min_pd(ix, iy), sign_mask);

      // Branchless select: ext2 > 0 → exterior, else interior.
      const __m128d is_outside = _mm_cmpgt_pd(ext2, vzero);
      __m128d d = geodex::utils::geodex_blendv_pd(int_d, ext_d, is_outside);

      // Inflation: subtract so clearance triggers at robot edge.
      d = _mm_sub_pd(d, vinfl);

      // Mask out-of-range obstacles to skip_dist.
      d = geodex::utils::geodex_blendv_pd(vskip, d, in_range);

      const __m128d nbd = _mm_xor_pd(_mm_mul_pd(vbeta, d), sign_mask);
      _mm_storeu_pd(&neg_beta_d[total], nbd);
      total += 2;
    }

    if (total == 0) return skip_dist_;

    // Find global max for numerical stability.
    double global_max = std::numeric_limits<double>::lowest();
    for (int j = 0; j < total; j += 2) {
      const __m128d v = _mm_loadu_pd(&neg_beta_d[j]);
      const double l0 = _mm_cvtsd_f64(v);
      const double l1 = _mm_cvtsd_f64(_mm_unpackhi_pd(v, v));
      global_max = std::max(global_max, std::max(l0, l1));
    }

    // Sum exp(nbd - max) using SSE2 fast_exp.
    const __m128d vgmax = _mm_set1_pd(global_max);
    __m128d vsum = _mm_setzero_pd();
    for (int j = 0; j < total; j += 2) {
      const __m128d shifted = _mm_sub_pd(_mm_loadu_pd(&neg_beta_d[j]), vgmax);
      vsum = _mm_add_pd(vsum, utils::fast_exp(shifted));
    }
    const double sum = _mm_cvtsd_f64(vsum) + _mm_cvtsd_f64(_mm_unpackhi_pd(vsum, vsum));

    return -(global_max + std::log(sum)) / beta_;
  }
#endif  // __SSE2__

  double eval_scalar(const double px, const double py, const int np) const {
    thread_local std::vector<double> neg_beta_d;
    neg_beta_d.clear();
    neg_beta_d.reserve(np);

    for (int i = 0; i < soa_.count(); ++i) {
      const double dx = px - soa_.cx()[i];
      const double dy = py - soa_.cy()[i];
      const double dist2 = dx * dx + dy * dy;

      // Bounding sphere early-out.
      if (dist2 >= soa_.bounding_r2()[i]) continue;

      // Local coordinates.
      const double lx = soa_.ct()[i] * dx + soa_.st()[i] * dy;
      const double ly = soa_.ct()[i] * dy - soa_.st()[i] * dx;
      const double alx = std::abs(lx);
      const double aly = std::abs(ly);

      const double hl = soa_.half_length()[i];
      const double hw = soa_.half_width()[i];

      // Signed distance.
      const double ex = std::max(alx - hl, 0.0);
      const double ey = std::max(aly - hw, 0.0);
      const double ext2 = ex * ex + ey * ey;
      double d;
      if (ext2 > 0.0) {
        d = std::sqrt(ext2);  // exterior
      } else {
        d = -std::min(hl - alx, hw - aly);  // interior
      }

      d -= inflation_;
      neg_beta_d.push_back(-beta_ * d);
    }

    if (neg_beta_d.empty()) return skip_dist_;

    double global_max = std::numeric_limits<double>::lowest();
    for (const double v : neg_beta_d) global_max = std::max(global_max, v);

    double sum = 0.0;
    for (const double v : neg_beta_d) sum += utils::fast_exp(v - global_max);

    return -(global_max + std::log(sum)) / beta_;
  }
};

}  // namespace geodex::collision
