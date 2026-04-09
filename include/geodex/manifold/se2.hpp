/// @file se2.hpp
/// @brief SE(2) manifold with left-invariant metric and multiple retraction policies.

#pragma once

#include <Eigen/Core>
#include <cmath>
#include <geodex/algorithm/distance.hpp>
#include <geodex/core/angle.hpp>
#include <geodex/core/concepts.hpp>
#include <geodex/core/retraction.hpp>
#include <geodex/core/sampler.hpp>
#include <geodex/metrics/se2_left_invariant.hpp>
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
  /// @param p First pose.
  /// @param q Second pose.
  /// @return The approximate geodesic distance.
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

// Verify the composed types satisfy RiemannianManifold.
static_assert(RiemannianManifold<SE2<>>);
static_assert(RiemannianManifold<SE2<SE2LeftInvariantMetric, SE2EulerRetraction>>);

}  // namespace geodex
