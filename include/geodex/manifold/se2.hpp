/// @file se2.hpp
/// @brief SE(2) manifold with left-invariant metric and multiple retraction policies.

#pragma once

#include <Eigen/Core>
#include <cmath>
#include <geodex/algorithm/distance.hpp>
#include <geodex/core/concepts.hpp>
#include <geodex/core/retraction.hpp>
#include <geodex/core/sampler.hpp>
#include <geodex/metrics/se2_left_invariant.hpp>
#include <geodex/utils/angle.hpp>
#include <geodex/utils/math.hpp>
#include <numbers>
#include <type_traits>

namespace geodex {

// ---------------------------------------------------------------------------
// Retraction policies
// ---------------------------------------------------------------------------

/// @brief True exponential and logarithmic maps on SE(2) (Lie group exp/log).
///
/// @details Uses left translation: \f$ \exp_p(v) = p \cdot \mathrm{Exp}(v) \f$ where
/// \f$ \mathrm{Exp} \f$ is the Lie group exponential at the identity. The matrix
/// \f$ V(\omega) \f$ relates the Lie algebra translation to the group translation.
struct SE2ExponentialMap {
  /// @brief Exponential map \f$ \exp_p(v) \f$ on SE(2).
  /// @param p Base pose \f$ (x, y, \theta) \f$.
  /// @param v Lie algebra velocity \f$ (v_x, v_y, \omega) \f$.
  /// @return The resulting pose on SE(2).
  EIGEN_STRONG_INLINE
  Eigen::Vector3d retract(const Eigen::Vector3d p, const Eigen::Vector3d v) const {
    const double vx = v[0], vy = v[1], omega = v[2];
    const double ct = std::cos(p[2]), st = std::sin(p[2]);

    double tx, ty;
    if (std::abs(omega) < 1e-8) {
      const double half_omega = 0.5 * omega;
      tx = vx - half_omega * vy;
      ty = vy + half_omega * vx;
    } else {
      const double s = std::sin(omega), c = std::cos(omega);
      const double s_o = s / omega;
      const double cm1_o = (1.0 - c) / omega;
      tx = s_o * vx - cm1_o * vy;
      ty = cm1_o * vx + s_o * vy;
    }

    const double dx = ct * tx - st * ty;
    const double dy = st * tx + ct * ty;

    return Eigen::Vector3d(p[0] + dx, p[1] + dy, wrap_to_pi(p[2] + omega));
  }

  /// @brief Logarithmic map \f$ \log_p(q) \f$ on SE(2).
  /// @param p Base pose.
  /// @param q Target pose.
  /// @return Lie algebra velocity at \f$ p \f$ such that \f$ \exp_p(v) = q \f$.
  EIGEN_STRONG_INLINE
  Eigen::Vector3d inverse_retract(const Eigen::Vector3d p, const Eigen::Vector3d q) const {
    const double ct = std::cos(p[2]), st = std::sin(p[2]);
    const double dxw = q[0] - p[0], dyw = q[1] - p[1];
    const double dx = ct * dxw + st * dyw;
    const double dy = -st * dxw + ct * dyw;
    const double dtheta = wrap_to_pi(q[2] - p[2]);

    double vx, vy;
    if (std::abs(dtheta) < 1e-8) {
      const double half_dtheta = 0.5 * dtheta;
      vx = dx + half_dtheta * dy;
      vy = -half_dtheta * dx + dy;
    } else {
      const double half_theta = 0.5 * dtheta;
      const double cot_half = half_theta / std::tan(half_theta);
      vx = cot_half * dx + half_theta * dy;
      vy = -half_theta * dx + cot_half * dy;
    }

    return Eigen::Vector3d(vx, vy, dtheta);
  }
};

/// @brief First-order Euler retraction on SE(2) (cheapest, treats as \f$ \mathbb{R}^2 \times S^1
/// \f$).
///
/// @details Simply adds the tangent vector component-wise with angle wrapping.
/// This is a valid first-order retraction but does not capture the group structure.
struct SE2EulerRetraction {
  /// @brief Euler retraction: \f$ R_p(v) = (p_x + v_x, p_y + v_y, \mathrm{wrap}(p_\theta +
  /// v_\theta)) \f$.
  /// @param p Base pose.
  /// @param v Tangent vector.
  /// @return The retracted pose.
  EIGEN_STRONG_INLINE
  Eigen::Vector3d retract(const Eigen::Vector3d p, const Eigen::Vector3d v) const {
    return Eigen::Vector3d(p[0] + v[0], p[1] + v[1], wrap_to_pi(p[2] + v[2]));
  }

  /// @brief Inverse Euler retraction.
  /// @param p Base pose.
  /// @param q Target pose.
  /// @return Tangent vector at \f$ p \f$.
  EIGEN_STRONG_INLINE
  Eigen::Vector3d inverse_retract(const Eigen::Vector3d p, const Eigen::Vector3d q) const {
    return Eigen::Vector3d(q[0] - p[0], q[1] - p[1], wrap_to_pi(q[2] - p[2]));
  }
};

// Verify retraction concepts.
static_assert(Retraction<SE2ExponentialMap, Eigen::Vector3d, Eigen::Vector3d>);
static_assert(Retraction<SE2EulerRetraction, Eigen::Vector3d, Eigen::Vector3d>);

// ---------------------------------------------------------------------------
// SE(2) manifold
// ---------------------------------------------------------------------------

/// @brief The special Euclidean group \f$ \mathrm{SE}(2) = \mathbb{R}^2 \rtimes \mathrm{SO}(2) \f$.
///
/// @details Poses are represented as \f$ (x, y, \theta) \f$ with \f$ \theta \in [-\pi, \pi) \f$.
/// The manifold is parameterized by a metric policy and a retraction policy, following the
/// same design as Sphere and Torus.
///
/// @tparam MetricT Metric policy (default: SE2LeftInvariantMetric).
/// @tparam RetractionT Retraction policy (default: SE2ExponentialMap).
/// @tparam SamplerT Sampler policy for `random_point()` (default: `StochasticSampler`).
template <typename MetricT = SE2LeftInvariantMetric, typename RetractionT = SE2ExponentialMap,
          typename SamplerT = StochasticSampler>
class SE2 {
 public:
  using Scalar = double;            ///< Scalar type.
  using Point = Eigen::Vector3d;    ///< Pose \f$ (x, y, \theta) \f$.
  using Tangent = Eigen::Vector3d;  ///< Tangent vector \f$ (v_x, v_y, \omega) \f$.

  /// @brief Runtime query: is the currently-configured metric the bi-invariant
  /// Lie group metric (unit weights on `SE2LeftInvariantMetric` paired with the
  /// true `SE2ExponentialMap`)?
  ///
  /// @details Only in this case is the Lie-group `log` the Riemannian logarithm
  /// of the metric, so `discrete_geodesic` can safely take the log direction
  /// as the natural gradient. `discrete_geodesic` calls this method to
  /// activate the fast path on a per-call basis — anisotropic SE2 metrics
  /// fall through to finite differences.
  ///
  /// Because `SE2LeftInvariantMetric::weights_` is a runtime value, this check
  /// cannot be made at compile time.
  bool has_riemannian_log_runtime() const {
    if constexpr (std::is_same_v<MetricT, SE2LeftInvariantMetric> &&
                  std::is_same_v<RetractionT, SE2ExponentialMap>) {
      return metric_.weights().isApprox(Eigen::Vector3d(1.0, 1.0, 1.0));
    } else {
      return false;
    }
  }

  /// @brief Default constructor. Users must call `set_sampling_bounds()` before
  /// using `random_point()` if the default \f$[0,10]^2 \times [-\pi,\pi)\f$ is unsuitable.
  SE2() = default;

  /// @brief Construct with explicit metric.
  /// @param metric The metric policy instance.
  explicit SE2(MetricT metric) : metric_(std::move(metric)) {}

  /// @brief Construct with sampling bounds.
  /// @param lo Lower sampling bounds \f$(x_\min, y_\min, \theta_\min)\f$.
  /// @param hi Upper sampling bounds \f$(x_\max, y_\max, \theta_\max)\f$.
  SE2(const Eigen::Vector3d& lo, const Eigen::Vector3d& hi) : lo_(lo), hi_(hi), sample_buf_(3) {}

  /// @brief Construct with explicit metric and sampling bounds.
  /// @param metric The metric policy instance.
  /// @param lo Lower sampling bounds \f$(x_\min, y_\min, \theta_\min)\f$.
  /// @param hi Upper sampling bounds \f$(x_\max, y_\max, \theta_\max)\f$.
  SE2(MetricT metric, const Eigen::Vector3d& lo, const Eigen::Vector3d& hi)
      : metric_(std::move(metric)), lo_(lo), hi_(hi), sample_buf_(3) {}

  /// @brief Construct with explicit metric, retraction, and sampling bounds.
  /// @param metric The metric policy instance.
  /// @param retraction The retraction policy instance.
  /// @param lo Lower sampling bounds \f$(x_\min, y_\min, \theta_\min)\f$.
  /// @param hi Upper sampling bounds \f$(x_\max, y_\max, \theta_\max)\f$.
  SE2(MetricT metric, RetractionT retraction, const Eigen::Vector3d& lo, const Eigen::Vector3d& hi)
      : metric_(std::move(metric)),
        retraction_(std::move(retraction)),
        lo_(lo),
        hi_(hi),
        sample_buf_(3) {}

  /// @brief Set the sampling bounds.
  void set_sampling_bounds(const Eigen::Vector3d& lo, const Eigen::Vector3d& hi) {
    lo_ = lo;
    hi_ = hi;
  }

  /// @brief Return the intrinsic dimension (always 3).
  int dim() const { return 3; }

  /// @brief Sample a random pose uniformly in the sampling bounds.
  /// @return A random pose \f$ (x, y, \theta) \f$.
  Point random_point() const {
    sampler_.sample_box(3, sample_buf_);
    return Point(lo_[0] + sample_buf_[0] * (hi_[0] - lo_[0]),
                 lo_[1] + sample_buf_[1] * (hi_[1] - lo_[1]),
                 lo_[2] + sample_buf_[2] * (hi_[2] - lo_[2]));
  }

  /// @brief Project an ambient vector onto the tangent space at \f$ p \f$.
  ///
  /// @details The tangent space of SE(2) is \f$ \mathbb{R}^3 \f$ (the Lie algebra
  /// \f$ \mathfrak{se}(2) \f$), so the projection is the identity.
  Tangent project(const Point& /*p*/, const Tangent& v) const { return v; }

  /// @name Metric delegates
  /// @{

  /// @brief Riemannian inner product at \f$ p \f$.
  Scalar inner(const Point& p, const Tangent& u, const Tangent& v) const {
    return metric_.inner(p, u, v);
  }

  /// @brief Riemannian norm at \f$ p \f$.
  Scalar norm(const Point& p, const Tangent& v) const { return metric_.norm(p, v); }

  /// @brief Batched inner product \f$U^\top M(p)\, V\f$ when the metric provides it.
  Eigen::MatrixXd inner_matrix(const Point& p, const Eigen::MatrixXd& U,
                               const Eigen::MatrixXd& V) const
    requires MetricHasInnerMatrix<MetricT, Point>
  {
    return metric_.inner_matrix(p, U, V);
  }

  /// @}

  /// @name Retraction delegates
  /// @{

  /// @brief Exponential map (or retraction) \f$ \exp_p(v) \f$.
  /// @param p Base pose.
  /// @param v Tangent vector.
  /// @return Resulting pose on SE(2).
  Point exp(const Point& p, const Tangent& v) const { return retraction_.retract(p, v); }

  /// @brief Logarithmic map (or inverse retraction) \f$ \log_p(q) \f$.
  /// @param p Base pose.
  /// @param q Target pose.
  /// @return Tangent vector at \f$ p \f$ pointing toward \f$ q \f$.
  Tangent log(const Point& p, const Point& q) const { return retraction_.inverse_retract(p, q); }

  /// @}

  /// @name Derived operations
  /// @{

  /// @brief Geodesic distance \f$ d(p, q) \f$ via the midpoint approximation.
  Scalar distance(const Point& p, const Point& q) const { return distance_midpoint(*this, p, q); }

  /// @brief Geodesic interpolation between \f$ p \f$ and \f$ q \f$ at parameter \f$ t \f$.
  /// @param p Start pose.
  /// @param q End pose.
  /// @param t Interpolation parameter in \f$ [0, 1] \f$.
  /// @return The interpolated pose.
  Point geodesic(const Point& p, const Point& q, Scalar t) const { return exp(p, t * log(p, q)); }

  /// @}

 private:
  MetricT metric_;
  RetractionT retraction_;
  Eigen::Vector3d lo_{0.0, 0.0, -std::numbers::pi};   ///< Lower sampling bounds (x, y, theta).
  Eigen::Vector3d hi_{10.0, 10.0, std::numbers::pi};  ///< Upper sampling bounds (x, y, theta).
  mutable SamplerT sampler_;
  mutable Eigen::VectorXd sample_buf_{3};  ///< Preallocated buffer for sampler output.
};

// Forward declaration for ConfigurationSpace overload below.
template <typename BaseManifold, typename MetricT>
class ConfigurationSpace;

// Verify the composed types satisfy RiemannianManifold.
static_assert(RiemannianManifold<SE2<>>);
static_assert(RiemannianManifold<SE2<SE2LeftInvariantMetric, SE2EulerRetraction>>);

// ---------------------------------------------------------------------------
// distance_midpoint overloads for SE(2)
// ---------------------------------------------------------------------------
//
// The implementation shares trig across the log→exp→log→log chain:
//   - 2 utils::sincos calls
//   - sincos(mid.θ) derived via angle-addition (no trig)
//   - tan(dθ/4) derived via half-angle formula (no trig)
//   - fma() for single-cycle FMADD on ARM
//   - NEON 2-wide for log(m,a) and log(m,b) in parallel
// Only norm() is called from the manifold — preserves metric evaluation.

#ifdef __ARM_NEON
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <immintrin.h>
#endif

namespace detail {

/// @brief SE(2) fused midpoint retraction: computes midpoint + v_diff with
///        minimal trig, then delegates norm to the manifold.
template <RiemannianManifold M>
auto distance_midpoint_se2_impl(const M& m, const Eigen::Vector3d& a,
                                const Eigen::Vector3d& b) -> typename M::Scalar {
  const double dxw = b[0] - a[0], dyw = b[1] - a[1];
  const double dtheta = wrap_to_pi(b[2] - a[2]);
  const double half_dt = 0.5 * dtheta;
  const double quarter_dt = 0.25 * dtheta;
  const double abs_dt = std::abs(dtheta);
  const double abs_hdt = std::abs(half_dt);

  // === sincos #1: a.θ ===
  double sa, ca;
  utils::sincos(a[2], &sa, &ca);

  const double dx = std::fma(ca, dxw, sa * dyw);
  const double dy = std::fma(-sa, dxw, ca * dyw);

  // === sincos #2: dθ/2 ===
  double sh, ch;
  utils::sincos(half_dt, &sh, &ch);

  const double s_dt = 2.0 * sh * ch;
  const double c_dt = std::fma(ch, ch, -(sh * sh));

  // --- log(a, b): V⁻¹(dθ) — branchless ---
  const bool general = abs_dt > 1e-10;
  const double inv_dt = general ? (1.0 / dtheta) : 1.0;
  const double s_o = general ? (s_dt * inv_dt) : 1.0;
  const double cm1_o = general ? ((1.0 - c_dt) * inv_dt) : 0.0;

  const double vx_ab = std::fma(s_o, dx, cm1_o * dy);
  const double vy_ab = std::fma(s_o, dy, -cm1_o * dx);

  // --- exp(a, 0.5·v_ab): V(dθ/2) ---
  const double hvx = 0.5 * vx_ab, hvy = 0.5 * vy_ab;
  const bool hgeneral = abs_hdt > 1e-10;
  const double inv_hdt = hgeneral ? (1.0 / half_dt) : 1.0;
  const double sho = hgeneral ? (sh * inv_hdt) : 1.0;
  const double chm1o = hgeneral ? ((1.0 - ch) * inv_hdt) : 0.0;

  const double tx = std::fma(sho, hvx, -chm1o * hvy);
  const double ty = std::fma(sho, hvy, chm1o * hvx);
  const double mx = std::fma(ca, tx, std::fma(-sa, ty, a[0]));
  const double my = std::fma(sa, tx, std::fma(ca, ty, a[1]));

  // --- sincos(mid.θ) via angle-addition (no trig) ---
  const double cm = std::fma(ca, ch, -(sa * sh));
  const double sm = std::fma(sa, ch, ca * sh);

  // --- log(m,a) and log(m,b) ---
  const double dxw_ma = a[0] - mx, dyw_ma = a[1] - my;
  const double dxw_mb = b[0] - mx, dyw_mb = b[1] - my;

#ifdef __ARM_NEON
  const float64x2_t vcm = vdupq_n_f64(cm);
  const float64x2_t vsm = vdupq_n_f64(sm);
  float64x2_t dxw_pair = {dxw_ma, dxw_mb};
  float64x2_t dyw_pair = {dyw_ma, dyw_mb};

  float64x2_t dx_pair = vfmaq_f64(vmulq_f64(vcm, dxw_pair), vsm, dyw_pair);
  float64x2_t dy_pair = vfmsq_f64(vmulq_f64(vcm, dyw_pair), vsm, dxw_pair);

  double vx_ma, vy_ma, vx_mb, vy_mb;
  if (hgeneral) {
    const double tan_q = sh / (1.0 + ch);
    const double cot_q = quarter_dt / tan_q;
    const float64x2_t vc = vdupq_n_f64(cot_q);
    const float64x2_t vq = {-quarter_dt, quarter_dt};
    const float64x2_t vnq = vnegq_f64(vq);
    float64x2_t vx_pair = vfmaq_f64(vmulq_f64(vc, dx_pair), vq, dy_pair);
    float64x2_t vy_pair = vfmaq_f64(vmulq_f64(vc, dy_pair), vnq, dx_pair);
    vx_ma = vgetq_lane_f64(vx_pair, 0);
    vy_ma = vgetq_lane_f64(vy_pair, 0);
    vx_mb = vgetq_lane_f64(vx_pair, 1);
    vy_mb = vgetq_lane_f64(vy_pair, 1);
  } else {
    const float64x2_t vq = {quarter_dt, -quarter_dt};
    const float64x2_t vnq = {-quarter_dt, quarter_dt};
    float64x2_t vx_pair = vfmaq_f64(dx_pair, vq, dy_pair);
    float64x2_t vy_pair = vfmaq_f64(dy_pair, vnq, dx_pair);
    vx_ma = vgetq_lane_f64(vx_pair, 0);
    vy_ma = vgetq_lane_f64(vy_pair, 0);
    vx_mb = vgetq_lane_f64(vx_pair, 1);
    vy_mb = vgetq_lane_f64(vy_pair, 1);
  }
#elif defined(__SSE2__)
  const __m128d vcm = _mm_set1_pd(cm);
  const __m128d vsm = _mm_set1_pd(sm);
  // _mm_set_pd(high, low): lane 0 = ma, lane 1 = mb
  __m128d dxw_pair = _mm_set_pd(dxw_mb, dxw_ma);
  __m128d dyw_pair = _mm_set_pd(dyw_mb, dyw_ma);

  // dx_pair = cm*dxw + sm*dyw, dy_pair = cm*dyw - sm*dxw
  __m128d dx_pair = utils::geodex_fmadd_pd(vsm, dyw_pair, _mm_mul_pd(vcm, dxw_pair));
  __m128d dy_pair = utils::geodex_fnmadd_pd(vsm, dxw_pair, _mm_mul_pd(vcm, dyw_pair));

  double vx_ma, vy_ma, vx_mb, vy_mb;
  if (hgeneral) {
    const double tan_q = sh / (1.0 + ch);
    const double cot_q = quarter_dt / tan_q;
    const __m128d vc = _mm_set1_pd(cot_q);
    // vq: lane 0 = -quarter_dt (for ma), lane 1 = quarter_dt (for mb)
    const __m128d vq = _mm_set_pd(quarter_dt, -quarter_dt);
    const __m128d vnq = _mm_sub_pd(_mm_setzero_pd(), vq);
    __m128d vx_pair = utils::geodex_fmadd_pd(vq, dy_pair, _mm_mul_pd(vc, dx_pair));
    __m128d vy_pair = utils::geodex_fmadd_pd(vnq, dx_pair, _mm_mul_pd(vc, dy_pair));
    vx_ma = _mm_cvtsd_f64(vx_pair);
    vy_ma = _mm_cvtsd_f64(vy_pair);
    vx_mb = _mm_cvtsd_f64(_mm_unpackhi_pd(vx_pair, vx_pair));
    vy_mb = _mm_cvtsd_f64(_mm_unpackhi_pd(vy_pair, vy_pair));
  } else {
    // vq: lane 0 = quarter_dt (for ma), lane 1 = -quarter_dt (for mb)
    const __m128d vq = _mm_set_pd(-quarter_dt, quarter_dt);
    const __m128d vnq = _mm_set_pd(quarter_dt, -quarter_dt);
    __m128d vx_pair = utils::geodex_fmadd_pd(vq, dy_pair, dx_pair);
    __m128d vy_pair = utils::geodex_fmadd_pd(vnq, dx_pair, dy_pair);
    vx_ma = _mm_cvtsd_f64(vx_pair);
    vy_ma = _mm_cvtsd_f64(vy_pair);
    vx_mb = _mm_cvtsd_f64(_mm_unpackhi_pd(vx_pair, vx_pair));
    vy_mb = _mm_cvtsd_f64(_mm_unpackhi_pd(vy_pair, vy_pair));
  }
#else
  const double dx_ma = cm * dxw_ma + sm * dyw_ma;
  const double dy_ma = -sm * dxw_ma + cm * dyw_ma;
  const double dx_mb = cm * dxw_mb + sm * dyw_mb;
  const double dy_mb = -sm * dxw_mb + cm * dyw_mb;

  double vx_ma, vy_ma, vx_mb, vy_mb;
  if (hgeneral) {
    const double tan_q = sh / (1.0 + ch);
    const double cot_q = quarter_dt / tan_q;
    vx_ma = cot_q * dx_ma - quarter_dt * dy_ma;
    vy_ma = quarter_dt * dx_ma + cot_q * dy_ma;
    vx_mb = cot_q * dx_mb + quarter_dt * dy_mb;
    vy_mb = -quarter_dt * dx_mb + cot_q * dy_mb;
  } else {
    vx_ma = dx_ma + quarter_dt * dy_ma;
    vy_ma = -quarter_dt * dx_ma + dy_ma;
    vx_mb = dx_mb - quarter_dt * dy_mb;
    vy_mb = quarter_dt * dx_mb + dy_mb;
  }
#endif

  Eigen::Vector3d midpoint(mx, my, wrap_to_pi(a[2] + half_dt));
  Eigen::Vector3d v_diff(vx_mb - vx_ma, vy_mb - vy_ma, dtheta);
  return m.norm(midpoint, v_diff);
}

}  // namespace detail

/// @brief Fused distance_midpoint overload for SE2.
template <typename MetricT, typename RetractionT, typename SamplerT>
auto distance_midpoint(const SE2<MetricT, RetractionT, SamplerT>& m, const Eigen::Vector3d& a,
                       const Eigen::Vector3d& b) -> double {
  return detail::distance_midpoint_se2_impl(m, a, b);
}

/// @brief Fused distance_midpoint overload for ConfigurationSpace wrapping an SE(2) base.
///
/// @note ConfigurationSpace is forward-declared here; the full definition is in
///       configuration_space.hpp. This overload is instantiated only when both
///       headers are included, which is the normal usage pattern.
template <typename BaseM, typename MetricT>
  requires std::is_same_v<typename BaseM::Point, Eigen::Vector3d>
auto distance_midpoint(const ConfigurationSpace<BaseM, MetricT>& m, const Eigen::Vector3d& a,
                       const Eigen::Vector3d& b) -> double {
  return detail::distance_midpoint_se2_impl(m, a, b);
}

}  // namespace geodex
