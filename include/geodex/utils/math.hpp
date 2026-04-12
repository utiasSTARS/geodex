/// @file math.hpp
/// @brief Portable fast-math and SIMD utility functions.
///
/// Provides:
///   - Portable sincos() wrapper (macOS __sincos, POSIX ::sincos)
///   - Schraudolph fast_exp() approximation (~5x faster, ~4% max relative error)
///   - ARM NEON 2-wide helpers and x86 SSE2 2-wide helpers with scalar fallbacks
///
/// SIMD portability: The x86 path requires only SSE2 (baseline x86_64). When
/// SSE4.1 or FMA are available (detected via __SSE4_1__ / __FMA__), faster
/// intrinsics are used automatically. Compile with -march=native for best
/// performance on your machine.
///
/// Used by geodex::collision (SDF evaluation), geodex::SE2 (distance),
/// and anywhere fast transcendental approximations are acceptable.

#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>

#ifdef __ARM_NEON
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <immintrin.h>
#endif

namespace geodex::utils {

// ---------------------------------------------------------------------------
// Portable sincos
// ---------------------------------------------------------------------------

/// @brief Compute sin and cos of @p angle in a single call.
///
/// On macOS/Apple, maps to `__sincos`. On POSIX/Linux, maps to `::sincos`.
/// Falls back to separate `std::sin`/`std::cos` on other platforms.
inline void sincos(const double angle, double* s, double* c) {
#if defined(__APPLE__)
  __sincos(angle, s, c);
#elif defined(_GNU_SOURCE) || defined(__linux__)
  ::sincos(angle, s, c);
#else
  *s = std::sin(angle);
  *c = std::cos(angle);
#endif
}

// ---------------------------------------------------------------------------
// Schraudolph fast exp() approximation
// ---------------------------------------------------------------------------

/// @brief Fast exp() approximation via Schraudolph's IEEE 754 bit trick.
///
/// Maps `x` to `2^(x/ln2)` by writing directly into the exponent bits of a
/// double. ~5x faster than std::exp on ARM. Max relative error ~4% (3.9%).
///
/// Schraudolph's paper writes to the upper 32-bit int of a double using
/// `a = 2^20/ln(2)`, `b = 1023*2^20`, `c = 60801`. The 64-bit adaptation
/// shifts all constants by 2^32 (the upper int occupies bits 32-63 of the
/// int64). The correction c=60801 balances over/under-estimates of the linear
/// chord approximation to 2^f on each unit interval.
///
/// @par Reference
/// Schraudolph, N. N. (1999). A fast, compact approximation of the
/// exponential function. Neural Computation, 11(4), 853-862.
inline double fast_exp(const double x) {
  const double clamped = std::max(x, -700.0);
  // a = 2^52 / ln(2)  (= 2^20/ln(2) * 2^32)
  // b - c = (1023 * 2^20 - 60801) * 2^32 = 4606921280493453312
  const auto i = static_cast<int64_t>(6497320848556798.0 * clamped + 4606921280493453312.0);
  return std::bit_cast<double>(i);
}

// ---------------------------------------------------------------------------
// ARM NEON 2-wide helpers
// ---------------------------------------------------------------------------

#ifdef __ARM_NEON

/// @brief NEON 2-wide fast_exp: process two doubles simultaneously.
///
/// Same Schraudolph approximation as the scalar version with the 2^32-scaled
/// bias correction. Uses vcvtq_s64_f64 (truncation toward zero), matching
/// the scalar static_cast<int64_t> behavior.
inline float64x2_t fast_exp(float64x2_t x) {
  const float64x2_t vmin = vdupq_n_f64(-700.0);
  x = vmaxq_f64(x, vmin);
  const float64x2_t scale = vdupq_n_f64(6497320848556798.0);
  const float64x2_t bias = vdupq_n_f64(4606921280493453312.0);
  const float64x2_t val = vaddq_f64(vmulq_f64(scale, x), bias);
  const int64x2_t ival = vcvtq_s64_f64(val);
  return vreinterpretq_f64_s64(ival);
}

/// @brief Forward rotation: rotate 2 points by angle θ (body → world).
///
/// Standard 2D rotation matrix applied to 2-wide coordinate pairs:
///   rx =  ct * dx - st * dy
///   ry =  st * dx + ct * dy
///
/// where (ct, st) = (cos θ, sin θ) are broadcast values and (dx, dy) are
/// 2-wide coordinate pairs (e.g., body-frame polygon samples).
inline void rotate_2wide(const float64x2_t ct, const float64x2_t st,
                         const float64x2_t dx, const float64x2_t dy,
                         float64x2_t& rx, float64x2_t& ry) {
  rx = vfmsq_f64(vmulq_f64(ct, dx), st, dy);  // ct*dx - st*dy
  ry = vfmaq_f64(vmulq_f64(ct, dy), st, dx);  // st*dx + ct*dy
}

/// @brief Inverse rotation: rotate 2 points by -θ (world → local).
///
/// Transpose of the rotation matrix (R^T = R^{-1}):
///   lx = ct * dx + st * dy
///   ly = ct * dy - st * dx
///
/// Used by SDF code to transform query points into obstacle-local frames.
inline void inverse_rotate_2wide(const float64x2_t ct, const float64x2_t st,
                                 const float64x2_t dx, const float64x2_t dy,
                                 float64x2_t& lx, float64x2_t& ly) {
  lx = vfmaq_f64(vmulq_f64(ct, dx), st, dy);  // ct*dx + st*dy
  ly = vfmsq_f64(vmulq_f64(ct, dy), st, dx);  // ct*dy - st*dx
}

#endif  // __ARM_NEON

// ---------------------------------------------------------------------------
// x86 SSE2 2-wide helpers (with optional SSE4.1 and FMA acceleration)
// ---------------------------------------------------------------------------

#ifdef __SSE2__

// --- Portable wrappers: dispatch to best available ISA at compile time ---

/// @brief Branchless blend: select trueVal where mask is all-1s, falseVal otherwise.
/// Uses SSE4.1 _mm_blendv_pd when available, SSE2 bitwise fallback otherwise.
inline __m128d geodex_blendv_pd(const __m128d falseVal, const __m128d trueVal,
                                const __m128d mask) {
#ifdef __SSE4_1__
  return _mm_blendv_pd(falseVal, trueVal, mask);
#else
  return _mm_or_pd(_mm_and_pd(mask, trueVal), _mm_andnot_pd(mask, falseVal));
#endif
}

/// @brief Floor (round toward -infinity). Uses SSE4.1 when available.
inline __m128d geodex_floor_pd(const __m128d x) {
#ifdef __SSE4_1__
  return _mm_floor_pd(x);
#else
  const __m128d truncated = _mm_cvtepi32_pd(_mm_cvttpd_epi32(x));
  const __m128d needs_correction = _mm_cmpgt_pd(truncated, x);
  return _mm_sub_pd(truncated, _mm_and_pd(needs_correction, _mm_set1_pd(1.0)));
#endif
}

/// @brief Fused multiply-add: a*b + c. Uses FMA3 when available.
inline __m128d geodex_fmadd_pd(const __m128d a, const __m128d b, const __m128d c) {
#ifdef __FMA__
  return _mm_fmadd_pd(a, b, c);
#else
  return _mm_add_pd(_mm_mul_pd(a, b), c);
#endif
}

/// @brief Negated fused multiply-add: -(a*b) + c = c - a*b. Uses FMA3 when available.
inline __m128d geodex_fnmadd_pd(const __m128d a, const __m128d b, const __m128d c) {
#ifdef __FMA__
  return _mm_fnmadd_pd(a, b, c);
#else
  return _mm_sub_pd(c, _mm_mul_pd(a, b));
#endif
}

/// @brief SSE2 2-wide fast_exp: process two doubles simultaneously.
///
/// Same Schraudolph approximation as the scalar version. SSE2 has no packed
/// double-to-int64 conversion, so lanes are extracted to scalar for the
/// static_cast<int64_t> step.
inline __m128d fast_exp(__m128d x) {
  const __m128d vmin = _mm_set1_pd(-700.0);
  x = _mm_max_pd(x, vmin);
  const __m128d scale = _mm_set1_pd(6497320848556798.0);
  const __m128d bias = _mm_set1_pd(4606921280493453312.0);
  const __m128d val = _mm_add_pd(_mm_mul_pd(scale, x), bias);
  // SSE2 has no _mm_cvttpd_epi64: extract lanes, cast scalar, reload.
  alignas(16) double dv[2];
  _mm_store_pd(dv, val);
  alignas(16) int64_t iv[2] = {static_cast<int64_t>(dv[0]), static_cast<int64_t>(dv[1])};
  return _mm_castsi128_pd(_mm_load_si128(reinterpret_cast<const __m128i*>(iv)));
}

/// @brief Forward rotation: rotate 2 points by angle θ (body → world).
///   rx = ct * dx - st * dy
///   ry = st * dx + ct * dy
inline void rotate_2wide(const __m128d ct, const __m128d st, const __m128d dx,
                         const __m128d dy, __m128d& rx, __m128d& ry) {
  rx = geodex_fnmadd_pd(st, dy, _mm_mul_pd(ct, dx));  // ct*dx - st*dy
  ry = geodex_fmadd_pd(st, dx, _mm_mul_pd(ct, dy));   // st*dx + ct*dy
}

/// @brief Inverse rotation: rotate 2 points by -θ (world → local).
///   lx = ct * dx + st * dy
///   ly = ct * dy - st * dx
inline void inverse_rotate_2wide(const __m128d ct, const __m128d st, const __m128d dx,
                                 const __m128d dy, __m128d& lx, __m128d& ly) {
  lx = geodex_fmadd_pd(st, dy, _mm_mul_pd(ct, dx));   // ct*dx + st*dy
  ly = geodex_fnmadd_pd(st, dx, _mm_mul_pd(ct, dy));   // ct*dy - st*dx
}

#endif  // __SSE2__

}  // namespace geodex::utils
