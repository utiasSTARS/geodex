/// @file polygon_footprint.hpp
/// @brief Convex polygon footprint with precomputed body-frame perimeter samples.
///
/// The polygon perimeter is discretized into uniformly-spaced sample points at
/// construction time, stored in SoA layout for SIMD-friendly access. At query
/// time, a single sincos call rotates all samples to world frame — amortizing
/// the expensive trig across the entire polygon.

#pragma once

#include <Eigen/Core>
#include <cmath>
#include <geodex/utils/math.hpp>
#include <vector>

#ifdef __ARM_NEON
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <immintrin.h>
#endif

namespace geodex::collision {

/// @brief A convex polygon footprint represented as body-frame perimeter samples.
///
/// Samples are stored in SoA layout (separate x[] and y[] arrays) padded to
/// an even count for NEON 2-wide processing. The transform() method rotates
/// and translates all samples to world frame using a single sincos call and
/// NEON vectorized rotation.
///
/// @note Only convex polygons are supported. For non-convex shapes, interior
/// sampling would be needed (future work).
class PolygonFootprint {
 public:
  /// @brief Construct from polygon vertices with a specified number of samples per edge.
  ///
  /// Vertices define a convex polygon in the body frame, centered at the origin.
  /// Samples are placed uniformly along each edge, including the start vertex
  /// but excluding the end vertex (to avoid duplicates at corners).
  PolygonFootprint(const std::vector<Eigen::Vector2d>& vertices, const int samples_per_edge)
      : samples_per_edge_(samples_per_edge) {
    const int n_edges = static_cast<int>(vertices.size());
    n_ = n_edges * samples_per_edge;
    n_padded_ = (n_ + 1) & ~1;

    body_x_.resize(n_padded_, 0.0);
    body_y_.resize(n_padded_, 0.0);

    int idx = 0;
    for (int e = 0; e < n_edges; ++e) {
      const auto& v0 = vertices[e];
      const auto& v1 = vertices[(e + 1) % n_edges];
      for (int s = 0; s < samples_per_edge; ++s) {
        const double t = static_cast<double>(s) / samples_per_edge;
        body_x_[idx] = (1.0 - t) * v0[0] + t * v1[0];
        body_y_[idx] = (1.0 - t) * v0[1] + t * v1[1];
        ++idx;
      }
    }

    // Bounding radius: max distance from origin across all samples.
    // Track max(r²) then sqrt once — avoids n-1 unnecessary sqrt calls.
    double max_r2 = 0.0;
    for (int i = 0; i < n_; ++i) {
      const double r2 = body_x_[i] * body_x_[i] + body_y_[i] * body_y_[i];
      max_r2 = std::max(max_r2, r2);
    }
    bounding_radius_ = std::sqrt(max_r2);
  }

  /// @brief Convenience factory for a rectangular footprint.
  /// @param half_length Half-extent along the local x-axis.
  /// @param half_width Half-extent along the local y-axis.
  /// @param samples_per_edge Number of sample points per edge.
  static PolygonFootprint rectangle(const double half_length, const double half_width,
                                    const int samples_per_edge) {
    std::vector<Eigen::Vector2d> vertices = {
        {-half_length, -half_width},
        {half_length, -half_width},
        {half_length, half_width},
        {-half_length, half_width},
    };
    return PolygonFootprint(vertices, samples_per_edge);
  }

  /// @brief Total number of perimeter samples (padded to even for NEON).
  int sample_count() const { return n_padded_; }

  /// @brief Actual (unpadded) number of perimeter samples.
  int sample_count_raw() const { return n_; }

  /// @brief Body-frame x-coordinates (SoA, padded).
  const double* body_x() const { return body_x_.data(); }

  /// @brief Body-frame y-coordinates (SoA, padded).
  const double* body_y() const { return body_y_.data(); }

  /// @brief Bounding radius from origin (for early-out tests).
  double bounding_radius() const { return bounding_radius_; }

  /// @brief Transform body-frame samples to world frame at pose (x, y, theta).
  ///
  /// SIMD strategy:
  ///   - One sincos(theta) call shared across all samples
  ///   - NEON 2-wide: rotate + translate 2 points per iteration
  ///   - For a rectangle with samples_per_edge=4: 16 samples → 8 iterations
  ///
  /// @param x World x-coordinate of the polygon center.
  /// @param y World y-coordinate of the polygon center.
  /// @param theta Rotation angle (radians).
  /// @param[out] wx World-frame x-coordinates (must have capacity >= sample_count()).
  /// @param[out] wy World-frame y-coordinates (must have capacity >= sample_count()).
  void transform(const double x, const double y, const double theta, double* wx,
                 double* wy) const {
    double ct, st;
    utils::sincos(theta, &st, &ct);

#ifdef __ARM_NEON
    const float64x2_t vct = vdupq_n_f64(ct);
    const float64x2_t vst = vdupq_n_f64(st);
    const float64x2_t vtx = vdupq_n_f64(x);
    const float64x2_t vty = vdupq_n_f64(y);

    for (int i = 0; i < n_padded_; i += 2) {
      const float64x2_t bx = vld1q_f64(body_x_.data() + i);
      const float64x2_t by = vld1q_f64(body_y_.data() + i);

      // Rotate: rx = ct*bx - st*by, ry = st*bx + ct*by.
      float64x2_t rx = vfmsq_f64(vmulq_f64(vct, bx), vst, by);
      float64x2_t ry = vfmaq_f64(vmulq_f64(vct, by), vst, bx);

      // Translate.
      vst1q_f64(wx + i, vaddq_f64(rx, vtx));
      vst1q_f64(wy + i, vaddq_f64(ry, vty));
    }
#elif defined(__SSE2__)
    const __m128d vct = _mm_set1_pd(ct);
    const __m128d vst = _mm_set1_pd(st);
    const __m128d vtx = _mm_set1_pd(x);
    const __m128d vty = _mm_set1_pd(y);

    for (int i = 0; i < n_padded_; i += 2) {
      const __m128d bx = _mm_loadu_pd(body_x_.data() + i);
      const __m128d by = _mm_loadu_pd(body_y_.data() + i);

      // Rotate: rx = ct*bx - st*by, ry = st*bx + ct*by.
      __m128d rx = geodex::utils::geodex_fnmadd_pd(vst, by, _mm_mul_pd(vct, bx));
      __m128d ry = geodex::utils::geodex_fmadd_pd(vst, bx, _mm_mul_pd(vct, by));

      // Translate.
      _mm_storeu_pd(wx + i, _mm_add_pd(rx, vtx));
      _mm_storeu_pd(wy + i, _mm_add_pd(ry, vty));
    }
#else
    for (int i = 0; i < n_padded_; ++i) {
      const double bx = body_x_[i], by = body_y_[i];
      wx[i] = ct * bx - st * by + x;
      wy[i] = st * bx + ct * by + y;
    }
#endif
  }

 private:
  std::vector<double> body_x_, body_y_;
  int n_ = 0, n_padded_ = 0;
  int samples_per_edge_ = 0;
  double bounding_radius_ = 0.0;
};

}  // namespace geodex::collision
