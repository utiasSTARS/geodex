/// @file so2.hpp
/// @brief SO(2) manifold — the circle group with a single canonical metric.

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
/// @details SO(2) is abelian, so the left- and right-invariant exponential maps
/// coincide and there is no frame distinction: a single retraction suffices.
/// Points and tangents are 1-vectors holding the angle \f$ \theta \f$ and angular
/// velocity \f$ \omega \f$ respectively. The maps are plain angle addition and
/// subtraction wrapped to \f$ [-\pi, \pi) \f$, which realizes the shortest-arc
/// (constant-speed) geodesic on the circle.
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
/// wraparound. SO(2) is a 1-D abelian Lie group, so left- and right-invariant
/// structure coincide and a single exponential map serves as both retraction and
/// its inverse. The manifold is parameterized by a metric policy and a retraction
/// policy, following the same design as Sphere, Torus, and SE(2).
///
/// @tparam MetricT Metric policy (default: SO2CanonicalMetric).
/// @tparam RetractionT Retraction policy (default: SO2ExponentialMap).
/// @tparam SamplerT Sampler policy for `random_point()` (default: `StochasticSampler`).
template <typename MetricT = SO2CanonicalMetric, typename RetractionT = SO2ExponentialMap,
          typename SamplerT = StochasticSampler>
class SO2 {
 public:
  using Scalar = double;                        ///< Scalar type.
  using Point = Eigen::Matrix<double, 1, 1>;    ///< Angle \f$ \theta \f$.
  using Tangent = Eigen::Matrix<double, 1, 1>;  ///< Angular velocity \f$ \omega \f$.

  /// @brief Runtime query: is the currently-configured metric the bi-invariant
  /// round metric (unit weight on `SO2CanonicalMetric` paired with the true
  /// `SO2ExponentialMap`)?
  ///
  /// @details Only in this case is the Lie-group `log` the Riemannian logarithm
  /// of the metric, so `discrete_geodesic` can safely take the log direction as
  /// the natural gradient. Because `SO2CanonicalMetric`'s weight is a runtime
  /// value, this check cannot be made at compile time.
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

  /// @brief Sample a random angle uniformly in \f$ [-\pi, \pi) \f$.
  /// @return A random angle as a 1-vector.
  Point random_point() const {
    sampler_.sample_box(1, sample_buf_);
    Point p;
    p[0] = lo_ + sample_buf_[0] * (hi_ - lo_);
    return p;
  }

  /// @brief Project an ambient vector onto the tangent space at \f$ p \f$.
  ///
  /// @details The tangent space of SO(2) is \f$ \mathbb{R} \f$ (the Lie algebra
  /// \f$ \mathfrak{so}(2) \f$), so the projection is the identity.
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

}  // namespace geodex
