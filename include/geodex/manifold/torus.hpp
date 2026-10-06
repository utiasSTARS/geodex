/// @file torus.hpp
/// @brief Flat torus manifold \f$ T^n \f$ with periodic angle coordinates.

#pragma once

#include <cmath>

#include <numbers>
#include <type_traits>

#include <Eigen/Core>

#include "geodex/algorithm/distance.hpp"
#include "geodex/core/concepts.hpp"
#include "geodex/core/sampler.hpp"
#include "geodex/metrics/constant_spd.hpp"
#include "geodex/metrics/identity.hpp"
#include "geodex/utils/angle.hpp"

namespace geodex {

// ---------------------------------------------------------------------------
// Metric alias
// ---------------------------------------------------------------------------

/// @brief Standard flat metric on \f$ T^n \f$.
///
/// @details The inner product is the standard dot product
/// \f$ \langle u, v \rangle = u \cdot v \f$. The metric is stateless and stores nothing.
template <int Dim = Eigen::Dynamic>
using TorusFlatMetric = IdentityMetric<Dim>;

// ---------------------------------------------------------------------------
// Torus manifold
// ---------------------------------------------------------------------------

/// @brief Flat torus \f$ T^n \f$ parameterized by dimension and metric policy.
///
/// @details Points are represented as angles in \f$ [0, 2\pi)^n \f$.
/// The exp map wraps to \f$ [0, 2\pi) \f$ and the log map wraps differences
/// to \f$ [-\pi, \pi) \f$.
///
/// @tparam Dim Compile-time dimension, or `Eigen::Dynamic`.
/// @tparam MetricT Metric policy (default TorusFlatMetric).
/// @tparam SamplerT Sampler policy for `random_point()` (default `ScrambledHaltonSampler`).
template <int Dim = Eigen::Dynamic, typename MetricT = TorusFlatMetric<Dim>,
          typename SamplerT = ScrambledHaltonSampler>
class Torus {
 public:
  using Scalar = double;                       ///< Scalar type.
  using Point = Eigen::Vector<double, Dim>;    ///< Point type (angles in \f$ [0, 2\pi)^n \f$).
  using SamplerType = SamplerT;                ///< Sampler policy backing random_point().
  using Tangent = Eigen::Vector<double, Dim>;  ///< Tangent vector type.

  /// @brief Runtime check whether `log` is the Riemannian logarithm of the metric.
  ///
  /// @details The exp and log of the torus are addition and wrapping, the Riemannian
  /// log exactly when the metric is the identity (standard flat metric). Anisotropic
  /// SPD metrics are flat as well, but their geodesics are reparameterized. They count
  /// as not log-compatible, and `discrete_geodesic` uses finite differences.
  bool has_riemannian_log_runtime() const {
    if constexpr (std::is_same_v<MetricT, IdentityMetric<Dim>>) {
      return true;
    } else if constexpr (std::is_same_v<MetricT, ConstantSPDMetric<Dim>>) {
      return metric_.weight_matrix().isApprox(
          Eigen::Matrix<double, Dim, Dim>::Identity(dim_, dim_));
    } else {
      return false;
    }
  }

  /// @brief Fixed-dimension constructor.
  Torus()
    requires(Dim != Eigen::Dynamic)
      : dim_(Dim), sample_buf_(Dim) {}

  /// @brief Fixed-dimension constructor with custom metric.
  /// @param metric The metric policy instance.
  explicit Torus(MetricT metric)
    requires(Dim != Eigen::Dynamic)
      : metric_(std::move(metric)), dim_(Dim), sample_buf_(Dim) {}

  /// @brief Dynamic-dimension constructor.
  /// @param n The dimension of the torus.
  explicit Torus(int n)
    requires(Dim == Eigen::Dynamic)
      : metric_(make_default_metric(n)), dim_(n), sample_buf_(n) {}

  /// @brief Dynamic-dimension constructor with custom metric.
  /// @param n The dimension of the torus.
  /// @param metric The metric policy instance.
  Torus(int n, MetricT metric)
    requires(Dim == Eigen::Dynamic)
      : metric_(std::move(metric)), dim_(n), sample_buf_(n) {}

  /// @brief Return the dimension of the torus.
  int dim() const { return dim_; }

  /// @brief Number of unit-cube coordinates that from_unit_cube consumes.
  int unit_cube_dim() const { return dim_; }

  /// @brief Map unit-cube coordinates to angles in [0, 2pi)^n.
  Point from_unit_cube(Eigen::Ref<const Eigen::VectorXd> u) const {
    detail::require_unit_cube_size(u.size(), dim_);
    Point p;
    if constexpr (Dim == Eigen::Dynamic) {
      p.resize(dim_);
    }
    for (int i = 0; i < dim_; ++i) {
      p[i] = u[i] * 2.0 * std::numbers::pi;
    }
    return p;
  }

  /// @brief Sample a uniformly random point in [0, 2pi)^n.
  Point random_point() const {
    sample_buf_.resize(dim_);
    sampler_.sample(dim_, sample_buf_);
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

  /// @brief Lower coordinate bound, \f$ 0 \f$ on every axis.
  Point lo() const { return Point::Zero(dim_); }

  /// @brief Upper coordinate bound, \f$ 2\pi \f$ on every axis.
  Point hi() const { return Point::Constant(dim_, utils::two_pi); }

  /// @brief Deck-group generators of the coordinate axes, \f$ 2\pi \f$ on every axis.
  Point periods() const { return Point::Constant(dim_, utils::two_pi); }

  /// @brief Project an ambient vector onto the tangent space at \f$ p \f$.
  ///
  /// @details The tangent space of \f$ T^n \f$ is \f$ \mathbb{R}^n \f$ everywhere,
  /// and the projection is the identity.
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

  /// @brief The frame Jacobian, which is the identity. The angles are their own
  /// tangent coordinates.
  Eigen::MatrixXd coordinate_jacobian(const Point& /*q*/) const {
    return Eigen::MatrixXd::Identity(dim_, dim_);
  }

  /// @brief The metric on angle velocities, the metric's own Gram matrix.
  ///
  /// @details `plan()` uses it to certify a Loewner bound that carries the periods.
  /// A raw chord across a cut overestimates the wrapped distance.
  Eigen::MatrixXd coordinate_metric(const Point& q) const
    requires MetricHasInnerMatrix<MetricT, Point>
  {
    const Eigen::MatrixXd J = coordinate_jacobian(q);
    const Eigen::MatrixXd G = metric_.inner_matrix(q, J, J);
    return 0.5 * (G + G.transpose());
  }

  /// @}

  /// @name Exp / Log
  /// @{

  /// @brief Exponential map \f$ \exp_p(v) = \mathrm{wrap}(p + v) \f$.
  /// @param p Base point.
  /// @param v Tangent vector.
  /// @return The resulting point, wrapped to \f$ [0, 2\pi)^n \f$.
  Point exp(const Point& p, const Tangent& v) const { return utils::wrap_point<Dim>(p + v); }

  /// @brief Logarithmic map, the shortest-path tangent vector from \f$ p \f$ to \f$ q \f$.
  /// @param p Base point.
  /// @param q Target point.
  /// @return The wrapped difference in \f$ [-\pi, \pi)^n \f$.
  Tangent log(const Point& p, const Point& q) const { return utils::wrap_delta<Dim>(q - p); }

  /// @}

  /// @name Derived operations
  /// @{

  /// @brief Geodesic distance via the midpoint approximation.
  /// @param p First point.
  /// @param q Second point.
  /// @return The distance \f$ d(p, q) \f$.
  Scalar distance(const Point& p, const Point& q) const { return distance_midpoint(*this, p, q); }

  /// @brief Injectivity radius of \f$ T^n \f$, \f$ \pi \f$ (half the period).
  ///
  /// @details Returns the topological value for the default identity metric and
  /// period \f$ 2\pi \f$. For anisotropic custom metrics the effective radius is
  /// \f$ \pi / \sqrt{\lambda_{\max}(A)} \f$. This value is an upper bound, and
  /// `discrete_geodesic` may retry when the true radius is smaller.
  Scalar injectivity_radius() const { return std::numbers::pi; }

  /// @brief Geodesic interpolation between \f$ p \f$ and \f$ q \f$ at parameter \f$ t \f$.
  /// @param p Start point.
  /// @param q End point.
  /// @param t Interpolation parameter in \f$ [0, 1] \f$.
  /// @return The interpolated point, wrapped to \f$ [0, 2\pi)^n \f$.
  Point geodesic(const Point& p, const Point& q, Scalar t) const { return exp(p, t * log(p, q)); }

  /// @}

 private:
  /// @brief Build the default metric for dynamic Torus.
  static MetricT make_default_metric(int n) {
    if constexpr (std::is_constructible_v<MetricT, int>) {
      return MetricT(n);
    } else {
      return MetricT{};
    }
  }

  MetricT metric_;
  int dim_;
  mutable SamplerT sampler_;
  mutable Eigen::VectorXd sample_buf_;  ///< Preallocated buffer for sampler output.
};

// Verify the default types satisfy RiemannianManifold.
static_assert(RiemannianManifold<Torus<2>>);
static_assert(RiemannianManifold<Torus<Eigen::Dynamic>>);
static_assert(HasInjectivityRadius<Torus<2>>);
static_assert(HasPeriods<Torus<2>>);
static_assert(HasPeriods<Torus<Eigen::Dynamic>>);
static_assert(HasCoordinateMetric<Torus<Eigen::Dynamic>>);

}  // namespace geodex
