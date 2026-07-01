/// @file se3.hpp
/// @brief SE(3) manifold — a genuine Lie group with coupled screw geodesics.

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
/// @details Uses left translation of the group exponential:
/// \f$ \exp_g(\xi) = g \cdot \mathrm{Exp}(\xi) \f$ and
/// \f$ \log_g(h) = \mathrm{Log}(g^{-1} h) \f$, where \f$ \mathrm{Exp}/\mathrm{Log} \f$
/// are the SE(3) group exp/log at the identity (`utils::se3_exp` / `utils::se3_log`).
/// The twist \f$ \xi \f$ is therefore expressed in the body frame of \f$ g \f$.
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
/// @details Uses right translation of the group exponential:
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
/// @details A genuine Lie group whose geodesics are coupled screw motions. Poses
/// are represented as \f$ [t_x, t_y, t_z,\; q_x, q_y, q_z, q_w] \f$ (translation +
/// scalar-last unit quaternion) and tangents as twists \f$ [v;\,\omega] \f$. The
/// class composes a metric policy and a retraction policy, following the same
/// design as Sphere, Torus, and SE(2).
///
/// @tparam MetricT Metric policy (default: `SE3InvariantMetric`).
/// @tparam RetractionT Retraction policy (default: `SE3LeftExponentialMap`).
/// @tparam SamplerT Sampler policy for `random_point()` (default: `StochasticSampler`).
template <typename MetricT = SE3InvariantMetric, typename RetractionT = SE3LeftExponentialMap,
          typename SamplerT = StochasticSampler>
class SE3 {
 public:
  using Scalar = double;                        ///< Scalar type.
  using Point = Eigen::Matrix<double, 7, 1>;    ///< Pose \f$ [t;\,q] \f$.
  using Tangent = Eigen::Matrix<double, 6, 1>;  ///< Twist \f$ [v;\,\omega] \f$.

  /// @brief Runtime query: is the group `log` the Riemannian logarithm of the
  /// currently-configured metric?
  ///
  /// @details True only when the metric is `SE3InvariantMetric` with unit weights
  /// AND the retraction is one of the group-exponential maps (left or right). In
  /// that case `-log_x(q)` is the length-minimizing descent direction of
  /// \f$ \tfrac12 d^2(\cdot, q) \f$, so `discrete_geodesic` can take the fast
  /// log-based step; anisotropic weights or a non-group retraction fall through to
  /// the finite-difference natural gradient.
  bool has_riemannian_log_runtime() const {
    if constexpr (std::is_same_v<MetricT, SE3InvariantMetric> &&
                  (std::is_same_v<RetractionT, SE3LeftExponentialMap> ||
                   std::is_same_v<RetractionT, SE3RightExponentialMap>)) {
      return metric_.weights().isApprox(Eigen::Matrix<double, 6, 1>::Ones());
    } else {
      return false;
    }
  }

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

  /// @brief Sample a random pose: translation uniform in the box, rotation uniform on SO(3).
  ///
  /// @details The translation is drawn uniformly in \f$[\mathrm{lo}, \mathrm{hi}]\f$.
  /// The rotation is a Haar-uniform unit quaternion, obtained by drawing four
  /// standard normals (via the Box-Muller transform over the configurable
  /// sampler) and normalizing — a point uniform on \f$ S^3 \f$, which is exactly
  /// the uniform (bi-invariant Haar) distribution on SO(3).
  /// @return A random pose \f$ [t;\,q] \f$ with a unit quaternion part.
  Point random_point() const {
    Point g;

    // Translation: 3 uniforms mapped into the box [lo_, hi_].
    sampler_.sample_box(3, sample_buf_);
    g[0] = lo_[0] + sample_buf_[0] * (hi_[0] - lo_[0]);
    g[1] = lo_[1] + sample_buf_[1] * (hi_[1] - lo_[1]);
    g[2] = lo_[2] + sample_buf_[2] * (hi_[2] - lo_[2]);

    // Rotation: 4 uniforms -> 4 normals (Box-Muller) -> normalize onto S^3.
    sampler_.sample_box(4, sample_buf_);
    double n[4];
    for (int i = 0; i < 2; ++i) {
      const double u1 = std::max(sample_buf_[2 * i], 1e-300);  // avoid log(0)
      const double u2 = sample_buf_[2 * i + 1];
      const double r = std::sqrt(-2.0 * std::log(u1));
      const double ang = 2.0 * std::numbers::pi * u2;
      n[2 * i] = r * std::cos(ang);
      n[2 * i + 1] = r * std::sin(ang);
    }
    g.tail<4>() = Eigen::Vector4d(n[0], n[1], n[2], n[3]).normalized();
    return g;
  }

  /// @brief Project an ambient vector onto the tangent space at \f$ p \f$.
  ///
  /// @details The tangent space of SE(3) is the Lie algebra
  /// \f$ \mathfrak{se}(3) \cong \mathbb{R}^6 \f$, so the projection is the identity.
  Tangent project(const Point& /*p*/, const Tangent& v) const { return v; }

  /// @name Metric delegates
  /// @{
  ///
  /// @note The metric acts on 6-vector twists and ignores its base-point
  /// argument (it is left-invariant / constant), so the manifold's 7-vector
  /// point is not forwarded; a zero twist is passed as the metric's `p`.

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

  /// @brief Exponential map (or retraction) \f$ \exp_p(v) \f$ — a screw motion.
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

  /// @brief Geodesic distance \f$ d(p, q) \f$ via the midpoint approximation.
  Scalar distance(const Point& p, const Point& q) const { return distance_midpoint(*this, p, q); }

  /// @brief Geodesic interpolation between \f$ p \f$ and \f$ q \f$ at parameter \f$ t \f$.
  ///
  /// @details Traces the coupled screw motion \f$ \exp_p(t\,\log_p(q)) \f$.
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
  mutable Eigen::VectorXd sample_buf_{4};  ///< Preallocated buffer (max of 3 trans + 4 quat draws).
};

// Verify the composed types satisfy RiemannianManifold (both retractions).
static_assert(RiemannianManifold<SE3<>>);
static_assert(RiemannianManifold<SE3<SE3InvariantMetric, SE3RightExponentialMap>>);

}  // namespace geodex
