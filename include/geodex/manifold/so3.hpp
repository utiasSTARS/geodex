/// @file so3.hpp
/// @brief SO(3) manifold — the rotation group as unit quaternions with a
///        canonical (left-/bi-invariant) metric and body/world retractions.

#pragma once

#include <cmath>

#include <algorithm>
#include <numbers>
#include <type_traits>

#include <Eigen/Core>

#include "geodex/algorithm/distance.hpp"
#include "geodex/core/concepts.hpp"
#include "geodex/core/metric.hpp"
#include "geodex/core/retraction.hpp"
#include "geodex/core/sampler.hpp"
#include "geodex/metrics/so3_canonical.hpp"
#include "geodex/utils/lie.hpp"

namespace geodex {

// ---------------------------------------------------------------------------
// Retraction policies
// ---------------------------------------------------------------------------
//
// Points are unit quaternions `[x, y, z, w]` (Point = Eigen::Vector4d); tangents
// are body angular velocities `omega` (Tangent = Eigen::Vector3d), so Point and
// Tangent differ in size (4 vs 3).

/// @brief Body-frame (left-translation) exponential/logarithm on SO(3).
///
/// @details \f$ \mathrm{retract}(q, \omega) = q \otimes \mathrm{Exp}(\omega) \f$
/// and \f$ \mathrm{inverse\_retract}(q_0, q_1) = \mathrm{Log}(q_0^{-1} \otimes q_1) \f$,
/// the body-frame relative rotation.
struct SO3LeftExponentialMap {
  /// @brief Body exponential map \f$ \exp_q(\omega) = q \otimes \mathrm{Exp}(\omega) \f$.
  /// @param q Base unit quaternion \f$ [x, y, z, w] \f$.
  /// @param omega Body angular velocity in \f$ \mathfrak{so}(3) \f$.
  /// @return The resulting unit quaternion.
  EIGEN_STRONG_INLINE
  Eigen::Vector4d retract(const Eigen::Vector4d& q, const Eigen::Vector3d& omega) const {
    return utils::quat_mul(q, utils::so3_exp(omega));
  }

  /// @brief Body logarithm \f$ \log_{q_0}(q_1) = \mathrm{Log}(q_0^{-1} \otimes q_1) \f$.
  /// @param q0 Base unit quaternion.
  /// @param q1 Target unit quaternion.
  /// @return Body angular velocity \f$ \omega \f$ such that \f$ \exp_{q_0}(\omega) = q_1 \f$.
  EIGEN_STRONG_INLINE
  Eigen::Vector3d inverse_retract(const Eigen::Vector4d& q0, const Eigen::Vector4d& q1) const {
    return utils::so3_log(utils::quat_mul(utils::quat_inv(q0), q1));
  }
};

/// @brief World-frame (right-translation) exponential/logarithm on SO(3).
///
/// @details \f$ \mathrm{retract}(q, \omega) = \mathrm{Exp}(\omega) \otimes q \f$
/// and \f$ \mathrm{inverse\_retract}(q_0, q_1) = \mathrm{Log}(q_1 \otimes q_0^{-1}) \f$,
/// the world-frame relative rotation.
struct SO3RightExponentialMap {
  /// @brief World exponential map \f$ \exp_q(\omega) = \mathrm{Exp}(\omega) \otimes q \f$.
  /// @param q Base unit quaternion \f$ [x, y, z, w] \f$.
  /// @param omega World angular velocity in \f$ \mathfrak{so}(3) \f$.
  /// @return The resulting unit quaternion.
  EIGEN_STRONG_INLINE
  Eigen::Vector4d retract(const Eigen::Vector4d& q, const Eigen::Vector3d& omega) const {
    return utils::quat_mul(utils::so3_exp(omega), q);
  }

  /// @brief World logarithm \f$ \log_{q_0}(q_1) = \mathrm{Log}(q_1 \otimes q_0^{-1}) \f$.
  /// @param q0 Base unit quaternion.
  /// @param q1 Target unit quaternion.
  /// @return World angular velocity \f$ \omega \f$ such that \f$ \exp_{q_0}(\omega) = q_1 \f$.
  EIGEN_STRONG_INLINE
  Eigen::Vector3d inverse_retract(const Eigen::Vector4d& q0, const Eigen::Vector4d& q1) const {
    return utils::so3_log(utils::quat_mul(q1, utils::quat_inv(q0)));
  }
};

// Verify retraction concepts at the SO(3) signature (Point = Vector4d, Tangent = Vector3d).
static_assert(Retraction<SO3LeftExponentialMap, Eigen::Vector4d, Eigen::Vector3d>);
static_assert(Retraction<SO3RightExponentialMap, Eigen::Vector4d, Eigen::Vector3d>);

// ---------------------------------------------------------------------------
// SO(3) manifold
// ---------------------------------------------------------------------------

/// @brief The special orthogonal group \f$ \mathrm{SO}(3) \f$ (3-D rotations).
///
/// @details Rotations are represented as unit quaternions
/// \f$ q = [x, y, z, w] \in S^3 \subset \mathbb{R}^4 \f$ (scalar-last, matching
/// `Eigen::Quaterniond::coeffs()`); the double cover \f$ q \sim -q \f$ represents
/// the same rotation. Tangent vectors are body angular velocities
/// \f$ \omega \in \mathfrak{so}(3) \cong \mathbb{R}^3 \f$, so the intrinsic
/// dimension is 3 while a point occupies 4 coordinates.
///
/// The manifold composes a metric policy and a retraction policy following the
/// same design as Sphere, Torus, and SE(2). With the default `SO3CanonicalMetric`
/// (unit weights) the metric is bi-invariant and `geodesic` is quaternion SLERP.
///
/// @tparam MetricT Metric policy (default: `SO3CanonicalMetric`).
/// @tparam RetractionT Retraction policy (default: `SO3LeftExponentialMap`).
/// @tparam SamplerT Sampler policy for `random_point()` (default: `StochasticSampler`).
template <typename MetricT = SO3CanonicalMetric, typename RetractionT = SO3LeftExponentialMap,
          typename SamplerT = StochasticSampler>
class SO3 {
 public:
  using Scalar = double;            ///< Scalar type.
  using Point = Eigen::Vector4d;    ///< Unit quaternion \f$ [x, y, z, w] \f$.
  using Tangent = Eigen::Vector3d;  ///< Body angular velocity \f$ \omega \f$.

  /// @brief Runtime query: is the Lie-group `log` the Riemannian logarithm of
  /// the currently-configured metric?
  ///
  /// @details True exactly when the metric is isotropic — all three
  /// `SO3CanonicalMetric` weights equal — AND the retraction is one of the two
  /// group exponential maps. In that case the group `log` is the Riemannian
  /// logarithm and `discrete_geodesic` can take the log direction as the natural
  /// gradient; anisotropic weights fall back to finite differences.
  bool has_riemannian_log_runtime() const {
    if constexpr ((std::is_same_v<RetractionT, SO3LeftExponentialMap> ||
                   std::is_same_v<RetractionT, SO3RightExponentialMap>) &&
                  std::is_same_v<MetricT, SO3CanonicalMetric>) {
      const Eigen::Vector3d& w = metric_.weights();
      return std::abs(w[0] - w[1]) < 1e-12 && std::abs(w[1] - w[2]) < 1e-12;
    } else {
      return false;
    }
  }

  /// @brief Default constructor (round bi-invariant metric, body retraction).
  SO3() = default;

  /// @brief Construct with an explicit metric.
  /// @param metric The metric policy instance.
  explicit SO3(MetricT metric) : metric_(std::move(metric)) {}

  /// @brief Construct with an explicit metric and retraction.
  /// @param metric The metric policy instance.
  /// @param retraction The retraction policy instance.
  SO3(MetricT metric, RetractionT retraction)
      : metric_(std::move(metric)), retraction_(std::move(retraction)) {}

  /// @brief Return the intrinsic dimension (always 3).
  int dim() const { return 3; }

  /// @brief Injectivity radius of the round SO(3): \f$ \pi \f$.
  ///
  /// @details The exponential map is a diffeomorphism for rotation angles below
  /// \f$ \pi \f$; at \f$ \pi \f$ (antipodal on \f$ S^3 \f$) the log direction is
  /// non-unique. Returned for the bi-invariant metric; anisotropic metrics have
  /// a smaller effective radius. `discrete_geodesic` uses this to cap steps.
  Scalar injectivity_radius() const { return std::numbers::pi; }

  /// @brief Sample a rotation uniformly (w.r.t. the Haar measure) on SO(3).
  ///
  /// @details Draws four standard normals via the Box-Muller transform over the
  /// sampler's uniform box and normalizes the resulting 4-vector — Marsaglia's
  /// method for a uniform point on \f$ S^3 \f$, which projects to the uniform
  /// (Haar) distribution on SO(3).
  /// @return A valid unit quaternion \f$ [x, y, z, w] \f$.
  Point random_point() const {
    sampler_.sample_box(4, sample_buf_);
    const double u1 = std::max(sample_buf_[0], 1e-300);  // avoid log(0)
    const double u2 = sample_buf_[1];
    const double u3 = std::max(sample_buf_[2], 1e-300);
    const double u4 = sample_buf_[3];
    const double r1 = std::sqrt(-2.0 * std::log(u1));
    const double r2 = std::sqrt(-2.0 * std::log(u3));
    const double a1 = 2.0 * std::numbers::pi * u2;
    const double a2 = 2.0 * std::numbers::pi * u4;
    Point q(r1 * std::cos(a1), r1 * std::sin(a1), r2 * std::cos(a2), r2 * std::sin(a2));
    return q.normalized();
  }

  /// @brief Project an ambient vector onto the tangent space at \f$ p \f$.
  ///
  /// @details Tangent vectors are already the minimal body algebra
  /// \f$ \mathfrak{so}(3) \cong \mathbb{R}^3 \f$, so the projection is the
  /// identity.
  Tangent project(const Point& /*p*/, const Tangent& v) const { return v; }

  /// @name Metric delegates
  /// @{
  //
  // The metric acts on the body algebra (a 3-vector) and ignores its base-point
  // argument, so the manifold's 4-vector quaternion point is not forwarded; a
  // zero 3-vector is passed as the metric's `p`.

  /// @brief Riemannian inner product at \f$ p \f$.
  Scalar inner(const Point& /*p*/, const Tangent& u, const Tangent& v) const {
    return metric_.inner(Eigen::Vector3d::Zero(), u, v);
  }

  /// @brief Riemannian norm at \f$ p \f$.
  Scalar norm(const Point& /*p*/, const Tangent& v) const {
    return metric_.norm(Eigen::Vector3d::Zero(), v);
  }

  /// @brief Batched inner product \f$U^\top M\, V\f$ when the metric provides it.
  Eigen::MatrixXd inner_matrix(const Point& /*p*/, const Eigen::MatrixXd& U,
                               const Eigen::MatrixXd& V) const
    requires MetricHasInnerMatrix<MetricT, Eigen::Vector3d>
  {
    return metric_.inner_matrix(Eigen::Vector3d::Zero(), U, V);
  }

  /// @}

  /// @name Retraction delegates
  /// @{

  /// @brief Exponential map (or retraction) \f$ \exp_p(v) \f$.
  /// @param p Base unit quaternion.
  /// @param v Body angular velocity at \f$ p \f$.
  /// @return The resulting unit quaternion.
  Point exp(const Point& p, const Tangent& v) const { return retraction_.retract(p, v); }

  /// @brief Logarithmic map (or inverse retraction) \f$ \log_p(q) \f$.
  /// @param p Base unit quaternion.
  /// @param q Target unit quaternion.
  /// @return Body angular velocity at \f$ p \f$ such that \f$ \exp_p(v) = q \f$ (shortest arc).
  Tangent log(const Point& p, const Point& q) const { return retraction_.inverse_retract(p, q); }

  /// @}

  /// @name Derived operations
  /// @{

  /// @brief Geodesic distance \f$ d(p, q) \f$ via the midpoint approximation.
  ///
  /// @details Exact here: with the true exp/log the midpoint formula reproduces
  /// the metric geodesic length. For the default isotropic metric this equals
  /// the rotation angle between \f$ p \f$ and \f$ q \f$ in \f$ [0, \pi] \f$.
  Scalar distance(const Point& p, const Point& q) const { return distance_midpoint(*this, p, q); }

  /// @brief Geodesic interpolation between \f$ p \f$ and \f$ q \f$ at parameter \f$ t \f$.
  ///
  /// @details \f$ \exp_p(t\,\log_p(q)) \f$; for the bi-invariant metric this is
  /// exactly quaternion SLERP.
  /// @param p Start unit quaternion.
  /// @param q End unit quaternion.
  /// @param t Interpolation parameter in \f$ [0, 1] \f$.
  /// @return The interpolated unit quaternion.
  Point geodesic(const Point& p, const Point& q, Scalar t) const { return exp(p, t * log(p, q)); }

  /// @}

 private:
  MetricT metric_;
  RetractionT retraction_;
  mutable SamplerT sampler_;
  mutable Eigen::VectorXd sample_buf_{4};  ///< Preallocated buffer for Box-Muller uniform samples.
};

// Verify the composed types satisfy RiemannianManifold.
static_assert(RiemannianManifold<SO3<>>);
static_assert(RiemannianManifold<SO3<SO3CanonicalMetric, SO3RightExponentialMap>>);

}  // namespace geodex
