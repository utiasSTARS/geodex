/// @file so2.hpp
/// @brief SO(2) manifold, the circle group with a single canonical metric.

#pragma once

#include <cmath>

#include <numbers>
#include <type_traits>

#include <Eigen/Core>

#include "geodex/algorithm/distance.hpp"
#include "geodex/core/concepts.hpp"
#include "geodex/core/metric.hpp"
#include "geodex/core/retraction.hpp"
#include "geodex/core/sampler.hpp"
#include "geodex/metrics/so2_canonical.hpp"
#include "geodex/utils/angle.hpp"

namespace geodex {

// ---------------------------------------------------------------------------
// Retraction policy
// ---------------------------------------------------------------------------

/// @brief True exponential and logarithmic maps on SO(2) (Lie group exp/log).
///
/// @details Angle addition/subtraction wrapped to \f$ [-\pi, \pi) \f$, realizing
/// the shortest-arc geodesic on the circle. SO(2) is abelian, and one map serves
/// as both retraction and inverse.
struct SO2ExponentialMap {
  /// @brief Exponential map \f$ \exp_\theta(v) = \mathrm{wrap}(\theta + v) \f$.
  /// @param theta Base angle as a 1-vector.
  /// @param v Angular velocity as a 1-vector.
  /// @return The resulting angle on SO(2).
  EIGEN_STRONG_INLINE
  Eigen::Matrix<double, 1, 1> retract(const Eigen::Matrix<double, 1, 1> theta,
                                      const Eigen::Matrix<double, 1, 1> v) const {
    Eigen::Matrix<double, 1, 1> out;
    out[0] = utils::wrap_to_pi(theta[0] + v[0]);
    return out;
  }

  /// @brief Logarithmic map \f$ \log_a(b) = \mathrm{wrap}(b - a) \f$.
  /// @param a Base angle as a 1-vector.
  /// @param b Target angle as a 1-vector.
  /// @return Angular velocity at \f$ a \f$ such that \f$ \exp_a(v) = b \f$ (shortest arc).
  EIGEN_STRONG_INLINE
  Eigen::Matrix<double, 1, 1> inverse_retract(const Eigen::Matrix<double, 1, 1> a,
                                              const Eigen::Matrix<double, 1, 1> b) const {
    Eigen::Matrix<double, 1, 1> out;
    out[0] = utils::wrap_to_pi(b[0] - a[0]);
    return out;
  }
};

// Verify retraction concept.
static_assert(
    Retraction<SO2ExponentialMap, Eigen::Matrix<double, 1, 1>, Eigen::Matrix<double, 1, 1>>);

// ---------------------------------------------------------------------------
// SO(2) manifold
// ---------------------------------------------------------------------------

/// @brief The special orthogonal group \f$ \mathrm{SO}(2) \cong S^1 \f$ (the circle group).
///
/// @details A configuration is a single angle \f$ \theta \in [-\pi, \pi) \f$ with
/// wraparound. The manifold is parameterized by a metric policy and a retraction
/// policy, following the same design as Sphere, Torus, and SE(2).
///
/// @tparam MetricT Metric policy (default SO2CanonicalMetric).
/// @tparam RetractionT Retraction policy (default SO2ExponentialMap).
/// @tparam SamplerT Sampler policy for `random_point()` (default `ScrambledHaltonSampler`).
template <typename MetricT = SO2CanonicalMetric, typename RetractionT = SO2ExponentialMap,
          typename SamplerT = ScrambledHaltonSampler>
class SO2 {
 public:
  using Scalar = double;                        ///< Scalar type.
  using Point = Eigen::Matrix<double, 1, 1>;    ///< Angle \f$ \theta \f$.
  using SamplerType = SamplerT;                 ///< Sampler policy backing random_point().
  using Tangent = Eigen::Matrix<double, 1, 1>;  ///< Angular velocity \f$ \omega \f$.

  /// @brief Runtime check whether the configured metric is the bi-invariant round
  /// metric, a unit-weight `SO2CanonicalMetric` with the true `SO2ExponentialMap`.
  ///
  /// @details Only then is the Lie-group `log` the Riemannian logarithm of the
  /// metric, and `discrete_geodesic` can take the log direction as the natural
  /// gradient.
  bool has_riemannian_log_runtime() const {
    if constexpr (std::is_same_v<MetricT, SO2CanonicalMetric> &&
                  std::is_same_v<RetractionT, SO2ExponentialMap>) {
      return std::abs(metric_.weight() - 1.0) < 1e-12;
    } else {
      return false;
    }
  }

  /// @brief Default constructor (unit-weight round circle).
  SO2() = default;

  /// @brief Construct with an explicit metric.
  /// @param metric The metric policy instance.
  explicit SO2(MetricT metric) : metric_(std::move(metric)) {}

  /// @brief Construct with an explicit metric and retraction.
  /// @param metric The metric policy instance.
  /// @param retraction The retraction policy instance.
  SO2(MetricT metric, RetractionT retraction)
      : metric_(std::move(metric)), retraction_(std::move(retraction)) {}

  /// @brief Return the intrinsic dimension (always 1).
  int dim() const { return 1; }

  /// @brief Number of unit-cube coordinates that from_unit_cube consumes.
  int unit_cube_dim() const { return 1; }

  /// @brief Map a unit-cube coordinate to an angle in [-pi, pi).
  Point from_unit_cube(Eigen::Ref<const Eigen::VectorXd> u) const {
    detail::require_unit_cube_size(u.size(), 1);
    Point p;
    p[0] = lo_ + u[0] * (hi_ - lo_);
    return p;
  }

  /// @brief Sample a random angle uniformly in [-pi, pi).
  Point random_point() const {
    sample_buf_.resize(1);
    sampler_.sample(1, sample_buf_);
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

  /// @brief Lower sampling bound \f$ -\pi \f$ as a 1-vector.
  Point lo() const { return Point{lo_}; }

  /// @brief Upper sampling bound \f$ \pi \f$ as a 1-vector.
  Point hi() const { return Point{hi_}; }

  /// @brief Deck-group generator of the single coordinate axis, \f$ 2\pi \f$.
  Point periods() const { return Point{utils::two_pi}; }

  /// @brief Project an ambient vector onto the tangent space at \f$ p \f$.
  ///
  /// @details The tangent space of SO(2) is \f$ \mathbb{R} \f$ (the Lie algebra
  /// \f$ \mathfrak{so}(2) \f$), and the projection is the identity.
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

  /// @brief The frame Jacobian, which is the identity. The angle is its own tangent
  /// coordinate.
  Eigen::MatrixXd coordinate_jacobian(const Point& /*q*/) const {
    return Eigen::MatrixXd::Identity(1, 1);
  }

  /// @brief The metric on the angle's velocity, the metric's own Gram matrix.
  ///
  /// @details `plan()` uses it to certify a Loewner bound that carries the period.
  /// A raw chord across the cut overestimates the wrapped distance.
  Eigen::MatrixXd coordinate_metric(const Point& q) const
    requires MetricHasInnerMatrix<MetricT, Point>
  {
    const Eigen::MatrixXd J = coordinate_jacobian(q);
    return metric_.inner_matrix(q, J, J);
  }

  /// @}

  /// @name Retraction delegates
  /// @{

  /// @brief Exponential map (or retraction) \f$ \exp_p(v) \f$.
  Point exp(const Point& p, const Tangent& v) const { return retraction_.retract(p, v); }

  /// @brief Logarithmic map (or inverse retraction) \f$ \log_p(q) \f$.
  Tangent log(const Point& p, const Point& q) const { return retraction_.inverse_retract(p, q); }

  /// @}

  /// @name Derived operations
  /// @{

  /// @brief Geodesic distance \f$ d(p, q) \f$ via the midpoint approximation.
  Scalar distance(const Point& p, const Point& q) const { return distance_midpoint(*this, p, q); }

  /// @brief Geodesic interpolation between \f$ p \f$ and \f$ q \f$ at parameter \f$ t \f$.
  Point geodesic(const Point& p, const Point& q, Scalar t) const { return exp(p, t * log(p, q)); }

  /// @}

 private:
  MetricT metric_;
  RetractionT retraction_;
  double lo_ = -std::numbers::pi;          ///< Lower sampling bound (fixed circle range).
  double hi_ = std::numbers::pi;           ///< Upper sampling bound (fixed circle range).
  mutable SamplerT sampler_;
  mutable Eigen::VectorXd sample_buf_{1};  ///< Preallocated buffer for sampler output.
};

// Verify the composed type satisfies RiemannianManifold.
static_assert(RiemannianManifold<SO2<>>);
static_assert(HasPeriods<SO2<>>);
static_assert(HasCoordinateMetric<SO2<>>);

}  // namespace geodex
