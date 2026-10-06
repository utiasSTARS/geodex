/// @file so3.hpp
/// @brief SO(3) manifold, the rotation group as unit quaternions with a canonical
///        (left- and bi-invariant) metric and body and world retractions.

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
// Points are unit quaternions `[x, y, z, w]` (Point = Eigen::Vector4d), and tangents
// are body angular velocities `omega` (Tangent = Eigen::Vector3d). Point and Tangent
// differ in size (4 vs 3).

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
/// `Eigen::Quaterniond::coeffs()`), and \f$ q \f$ and \f$ -q \f$ represent the same
/// rotation. Tangent vectors are body angular velocities
/// \f$ \omega \in \mathfrak{so}(3) \cong \mathbb{R}^3 \f$. The intrinsic dimension is 3,
/// and a point occupies 4 coordinates.
///
/// The manifold composes a metric policy and a retraction policy following the
/// same design as Sphere, Torus, and SE(2). With the default `SO3CanonicalMetric`
/// (unit weights) the metric is bi-invariant and `geodesic` is quaternion SLERP.
///
/// @tparam MetricT Metric policy (default `SO3CanonicalMetric`).
/// @tparam RetractionT Retraction policy (default `SO3LeftExponentialMap`).
/// @tparam SamplerT Sampler policy for `random_point()` (default `ScrambledHaltonSampler`).
template <typename MetricT = SO3CanonicalMetric, typename RetractionT = SO3LeftExponentialMap,
          typename SamplerT = ScrambledHaltonSampler>
class SO3 {
 public:
  using Scalar = double;            ///< Scalar type.
  using Point = Eigen::Vector4d;    ///< Unit quaternion \f$ [x, y, z, w] \f$.
  using SamplerType = SamplerT;     ///< Sampler policy backing random_point().
  using Tangent = Eigen::Vector3d;  ///< Body angular velocity \f$ \omega \f$.

  /// @brief Runtime check whether the Lie-group `log` is the Riemannian logarithm of
  /// the configured metric.
  ///
  /// @details True exactly when all three `SO3CanonicalMetric` weights are equal and
  /// the retraction is one of the two group exponential maps. `discrete_geodesic` then
  /// takes the log direction as the natural gradient. Anisotropic weights fall back to
  /// finite differences.
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

  /// @brief Injectivity radius of the round SO(3), \f$ \pi \f$.
  ///
  /// @details The exponential map is a diffeomorphism for rotation angles below
  /// \f$ \pi \f$. At \f$ \pi \f$ (antipodal on \f$ S^3 \f$) the log direction is not
  /// unique. The value holds for the bi-invariant metric, and anisotropic metrics have
  /// a smaller effective radius. `discrete_geodesic` caps steps with it.
  Scalar injectivity_radius() const { return std::numbers::pi; }

  /// @brief Number of unit-cube coordinates that from_unit_cube consumes.
  int unit_cube_dim() const { return 3; }

  /// @brief Map three unit-cube coordinates to a Haar-uniform rotation via
  /// Shoemake's method (Shoemake 1992).
  /// @param u Unit-cube coordinates.
  /// @return A unit quaternion \f$ [x, y, z, w] \f$.
  Point from_unit_cube(Eigen::Ref<const Eigen::VectorXd> u) const {
    detail::require_unit_cube_size(u.size(), 3);
    return utils::uniform_quaternion(u[0], u[1], u[2]);
  }

  /// @brief Sample a rotation uniformly (Haar measure) on SO(3).
  Point random_point() const {
    sample_buf_.resize(3);
    sampler_.sample(3, sample_buf_);
    return from_unit_cube(sample_buf_);
  }

  /// @brief Reseed the sampler for a reproducible random_point sequence.
  void seed(std::uint64_t s)
    requires SeedableSampler<SamplerT>
  {
    sampler_.seed(s);
  }

  /// @brief Replace the sampler.
  void set_sampler(SamplerT s) { sampler_ = std::move(s); }

  /// @brief The sampler behind random_point(). Planning samples through copies of it.
  const SamplerT& sampler() const { return sampler_; }

  /// @brief Project an ambient vector onto the tangent space at \f$ p \f$.
  ///
  /// @details Tangent vectors are already the minimal body algebra
  /// \f$ \mathfrak{so}(3) \cong \mathbb{R}^3 \f$, and the projection is the
  /// identity.
  Tangent project(const Point& /*p*/, const Tangent& v) const { return v; }

  /// @name Metric delegates
  /// @{
  //
  // The metric acts on the body algebra (a 3-vector) and ignores its base-point
  // argument. The manifold passes a zero 3-vector as the metric's `p`, not its
  // 4-vector quaternion.

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
  /// @details Exact here. With the true exp and log, the midpoint formula reproduces
  /// the metric geodesic length. For the default isotropic metric this equals the
  /// rotation angle between \f$ p \f$ and \f$ q \f$ in \f$ [0, \pi] \f$.
  Scalar distance(const Point& p, const Point& q) const { return distance_midpoint(*this, p, q); }

  /// @brief Geodesic interpolation between \f$ p \f$ and \f$ q \f$ at parameter \f$ t \f$.
  ///
  /// @details Returns \f$ \exp_p(t\,\log_p(q)) \f$, which is exactly quaternion SLERP
  /// for the bi-invariant metric.
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
  mutable Eigen::VectorXd sample_buf_{3};  ///< Preallocated buffer for unit-cube samples.
};

// Verify the composed types satisfy RiemannianManifold.
static_assert(RiemannianManifold<SO3<>>);
static_assert(RiemannianManifold<SO3<SO3CanonicalMetric, SO3RightExponentialMap>>);

}  // namespace geodex
