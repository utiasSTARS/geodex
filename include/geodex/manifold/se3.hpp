/// @file se3.hpp
/// @brief SE(3) manifold, a Lie group whose `geodesic` follows the screw motion of a
/// constant twist.

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
#include "geodex/metrics/se3_invariant.hpp"
#include "geodex/utils/lie.hpp"

namespace geodex {

// ---------------------------------------------------------------------------
// Retraction policies
// ---------------------------------------------------------------------------

/// @brief Body-frame (left) group exponential/logarithm on SE(3).
///
/// @details Uses left translation of the group exponential,
/// \f$ \exp_g(\xi) = g \cdot \mathrm{Exp}(\xi) \f$ and
/// \f$ \log_g(h) = \mathrm{Log}(g^{-1} h) \f$, where \f$ \mathrm{Exp}/\mathrm{Log} \f$
/// are the SE(3) group exp/log at the identity (`utils::se3_exp` / `utils::se3_log`).
/// The twist \f$ \xi \f$ is expressed in the body frame of \f$ g \f$.
struct SE3LeftExponentialMap {
  using Point = Eigen::Matrix<double, 7, 1>;    ///< Pose \f$ [t;\,q] \f$.
  using Tangent = Eigen::Matrix<double, 6, 1>;  ///< Body twist \f$ [v;\,\omega] \f$.

  /// @brief Exponential map \f$ \exp_g(\xi) = g \cdot \mathrm{Exp}(\xi) \f$.
  /// @param g Base pose.
  /// @param xi Body-frame twist.
  /// @return The resulting pose on SE(3).
  EIGEN_STRONG_INLINE
  Point retract(const Point& g, const Tangent& xi) const {
    return utils::se3_compose(g, utils::se3_exp(xi));
  }

  /// @brief Logarithmic map \f$ \log_g(h) = \mathrm{Log}(g^{-1} h) \f$.
  /// @param g Base pose.
  /// @param h Target pose.
  /// @return Body-frame twist at \f$ g \f$ such that \f$ \exp_g(\xi) = h \f$.
  EIGEN_STRONG_INLINE
  Tangent inverse_retract(const Point& g, const Point& h) const {
    return utils::se3_log(utils::se3_compose(utils::se3_inverse(g), h));
  }
};

/// @brief World-frame (right) group exponential/logarithm on SE(3).
///
/// @details Uses right translation of the group exponential,
/// \f$ \exp_g(\xi) = \mathrm{Exp}(\xi) \cdot g \f$ and
/// \f$ \log_g(h) = \mathrm{Log}(h\, g^{-1}) \f$. The twist \f$ \xi \f$ is
/// expressed in the fixed world/spatial frame.
struct SE3RightExponentialMap {
  using Point = Eigen::Matrix<double, 7, 1>;    ///< Pose \f$ [t;\,q] \f$.
  using Tangent = Eigen::Matrix<double, 6, 1>;  ///< Spatial twist \f$ [v;\,\omega] \f$.

  /// @brief Exponential map \f$ \exp_g(\xi) = \mathrm{Exp}(\xi) \cdot g \f$.
  /// @param g Base pose.
  /// @param xi Spatial-frame twist.
  /// @return The resulting pose on SE(3).
  EIGEN_STRONG_INLINE
  Point retract(const Point& g, const Tangent& xi) const {
    return utils::se3_compose(utils::se3_exp(xi), g);
  }

  /// @brief Logarithmic map \f$ \log_g(h) = \mathrm{Log}(h\, g^{-1}) \f$.
  /// @param g Base pose.
  /// @param h Target pose.
  /// @return Spatial-frame twist at \f$ g \f$ such that \f$ \exp_g(\xi) = h \f$.
  EIGEN_STRONG_INLINE
  Tangent inverse_retract(const Point& g, const Point& h) const {
    return utils::se3_log(utils::se3_compose(h, utils::se3_inverse(g)));
  }
};

// Verify retraction concepts at the SE(3) point/tangent signature.
static_assert(Retraction<SE3LeftExponentialMap, Eigen::Matrix<double, 7, 1>,
                         Eigen::Matrix<double, 6, 1>>);
static_assert(Retraction<SE3RightExponentialMap, Eigen::Matrix<double, 7, 1>,
                         Eigen::Matrix<double, 6, 1>>);

// ---------------------------------------------------------------------------
// SE(3) manifold
// ---------------------------------------------------------------------------

/// @brief The special Euclidean group \f$ \mathrm{SE}(3) = \mathbb{R}^3 \rtimes \mathrm{SO}(3) \f$.
///
/// @details `exp`, `log` and `geodesic` follow the screw motion of a constant twist,
/// which is not a geodesic of the metric (see `has_riemannian_log_runtime`). Poses
/// are represented as \f$ [t_x, t_y, t_z,\; q_x, q_y, q_z, q_w] \f$ (translation and
/// scalar-last unit quaternion) and tangents as twists \f$ [v;\,\omega] \f$. The
/// class composes a metric policy and a retraction policy, following the same
/// design as Sphere, Torus, and SE(2).
///
/// @tparam MetricT Metric policy (default `SE3InvariantMetric`).
/// @tparam RetractionT Retraction policy (default `SE3LeftExponentialMap`).
/// @tparam SamplerT Sampler policy for `random_point()` (default `ScrambledHaltonSampler`).
template <typename MetricT = SE3InvariantMetric, typename RetractionT = SE3LeftExponentialMap,
          typename SamplerT = ScrambledHaltonSampler>
class SE3 {
 public:
  using Scalar = double;                        ///< Scalar type.
  using Point = Eigen::Matrix<double, 7, 1>;    ///< Pose \f$ [t;\,q] \f$.
  using Tangent = Eigen::Matrix<double, 6, 1>;  ///< Twist \f$ [v;\,\omega] \f$.
  using SamplerType = SamplerT;                 ///< Sampler policy backing random_point().

  /// @brief Runtime check whether `log` is the Riemannian logarithm of the metric,
  /// always false.
  ///
  /// @details SE(3) does not have a bi-invariant Riemannian metric. With unit weights
  /// the invariant metric is \f$ |\dot t|^2 + |\omega|^2 \f$, whose geodesics move the
  /// origin in a straight line while rotating at a constant rate. The screw motion of a
  /// constant twist is longer whenever it rotates while its linear velocity has a
  /// component across the rotation axis. `discrete_geodesic` takes finite-difference
  /// steps on SE(3).
  bool has_riemannian_log_runtime() const { return false; }

  /// @brief Default constructor. Users must call `set_sampling_bounds()` before
  /// using `random_point()` if the default translation box \f$[0,10]^3\f$ is unsuitable.
  SE3() = default;

  /// @brief Construct with an explicit metric.
  /// @param metric The metric policy instance.
  explicit SE3(MetricT metric) : metric_(std::move(metric)) {}

  /// @brief Construct with translation sampling bounds.
  /// @param lo Lower translation bounds \f$(x_\min, y_\min, z_\min)\f$.
  /// @param hi Upper translation bounds \f$(x_\max, y_\max, z_\max)\f$.
  SE3(const Eigen::Vector3d& lo, const Eigen::Vector3d& hi) : lo_(lo), hi_(hi) {}

  /// @brief Construct with an explicit metric and translation sampling bounds.
  /// @param metric The metric policy instance.
  /// @param lo Lower translation bounds.
  /// @param hi Upper translation bounds.
  SE3(MetricT metric, const Eigen::Vector3d& lo, const Eigen::Vector3d& hi)
      : metric_(std::move(metric)), lo_(lo), hi_(hi) {}

  /// @brief Construct with an explicit metric, retraction, and translation bounds.
  /// @param metric The metric policy instance.
  /// @param retraction The retraction policy instance.
  /// @param lo Lower translation bounds.
  /// @param hi Upper translation bounds.
  SE3(MetricT metric, RetractionT retraction, const Eigen::Vector3d& lo, const Eigen::Vector3d& hi)
      : metric_(std::move(metric)), retraction_(std::move(retraction)), lo_(lo), hi_(hi) {}

  /// @brief Set the translation sampling bounds.
  /// @param lo Lower translation bounds \f$(x_\min, y_\min, z_\min)\f$.
  /// @param hi Upper translation bounds \f$(x_\max, y_\max, z_\max)\f$.
  void set_sampling_bounds(const Eigen::Vector3d& lo, const Eigen::Vector3d& hi) {
    lo_ = lo;
    hi_ = hi;
  }

  /// @brief Return the intrinsic dimension (always 6).
  int dim() const { return 6; }

  /// @brief Number of unit-cube coordinates that from_unit_cube consumes.
  int unit_cube_dim() const { return 6; }

  /// @brief Map six unit-cube coordinates to a pose. The first three rescale to
  /// the translation box, the last three give a Haar-uniform rotation via
  /// Shoemake's method (Shoemake 1992).
  /// @param u Unit-cube coordinates.
  /// @return A pose \f$ [t;\,q] \f$ with a unit quaternion part.
  Point from_unit_cube(Eigen::Ref<const Eigen::VectorXd> u) const {
    detail::require_unit_cube_size(u.size(), 6);
    Point g;
    g[0] = lo_[0] + u[0] * (hi_[0] - lo_[0]);
    g[1] = lo_[1] + u[1] * (hi_[1] - lo_[1]);
    g[2] = lo_[2] + u[2] * (hi_[2] - lo_[2]);
    g.tail<4>() = utils::uniform_quaternion(u[3], u[4], u[5]);
    return g;
  }

  /// @brief Sample a random pose, translation uniform in the box and rotation
  /// Haar-uniform on SO(3).
  Point random_point() const {
    sample_buf_.resize(6);
    sampler_.sample(6, sample_buf_);
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
  /// @details The tangent space of SE(3) is the Lie algebra
  /// \f$ \mathfrak{se}(3) \cong \mathbb{R}^6 \f$, and the projection is the identity.
  Tangent project(const Point& /*p*/, const Tangent& v) const { return v; }

  /// @name Metric delegates
  /// @{
  ///
  /// @note The metric acts on 6-vector twists and ignores its base-point
  /// argument, as a constant left-invariant metric. The manifold passes a zero
  /// twist as the metric's `p`, not its 7-vector point.

  /// @brief Riemannian inner product of two twists at \f$ p \f$.
  Scalar inner(const Point& /*p*/, const Tangent& u, const Tangent& v) const {
    return metric_.inner(Tangent::Zero(), u, v);
  }

  /// @brief Riemannian norm of a twist at \f$ p \f$.
  Scalar norm(const Point& /*p*/, const Tangent& v) const {
    return metric_.norm(Tangent::Zero(), v);
  }

  /// @brief Batched inner product \f$U^\top M\, V\f$ when the metric provides it.
  Eigen::MatrixXd inner_matrix(const Point& /*p*/, const Eigen::MatrixXd& U,
                               const Eigen::MatrixXd& V) const
    requires MetricHasInnerMatrix<MetricT, Tangent>
  {
    return metric_.inner_matrix(Tangent::Zero(), U, V);
  }

  /// @}

  /// @name Retraction delegates
  /// @{

  /// @brief Exponential map \f$ \exp_p(v) \f$, the screw motion of the twist \f$ v \f$.
  /// @param p Base pose.
  /// @param v Twist.
  /// @return Resulting pose on SE(3).
  Point exp(const Point& p, const Tangent& v) const { return retraction_.retract(p, v); }

  /// @brief Logarithmic map (or inverse retraction) \f$ \log_p(q) \f$.
  /// @param p Base pose.
  /// @param q Target pose.
  /// @return Twist at \f$ p \f$ pointing toward \f$ q \f$.
  Tangent log(const Point& p, const Point& q) const { return retraction_.inverse_retract(p, q); }

  /// @}

  /// @name Derived operations
  /// @{

  /// @brief Length of `geodesic(p, q, .)` under the metric, through the midpoint formula.
  ///
  /// @details This is the norm of the constant twist, the length of its screw motion,
  /// which is at least the Riemannian distance.
  Scalar distance(const Point& p, const Point& q) const { return distance_midpoint(*this, p, q); }

  /// @brief The screw motion \f$ \exp_p(t\,\log_p(q)) \f$ at parameter \f$ t \f$.
  ///
  /// @details The screw motion of the constant twist from \f$ p \f$ to \f$ q \f$, not a
  /// geodesic of the metric (see `has_riemannian_log_runtime`).
  /// @param p Start pose.
  /// @param q End pose.
  /// @param t Interpolation parameter in \f$ [0, 1] \f$.
  /// @return The interpolated pose.
  Point geodesic(const Point& p, const Point& q, Scalar t) const { return exp(p, t * log(p, q)); }

  /// @}

 private:
  MetricT metric_;
  RetractionT retraction_;
  Eigen::Vector3d lo_{0.0, 0.0, 0.0};     ///< Lower translation sampling bounds.
  Eigen::Vector3d hi_{10.0, 10.0, 10.0};  ///< Upper translation sampling bounds.
  mutable SamplerT sampler_;
  mutable Eigen::VectorXd sample_buf_{6};  ///< Preallocated buffer for unit-cube samples.
};

// Verify the composed types satisfy RiemannianManifold (both retractions).
static_assert(RiemannianManifold<SE3<>>);
static_assert(RiemannianManifold<SE3<SE3InvariantMetric, SE3RightExponentialMap>>);

}  // namespace geodex
