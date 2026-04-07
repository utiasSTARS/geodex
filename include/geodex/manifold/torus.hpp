/// @file torus.hpp
/// @brief Flat torus manifold \f$ T^n \f$ with periodic angle coordinates.

#pragma once

#include <Eigen/Core>
#include <cmath>
#include <geodex/algorithm/distance.hpp>
#include <geodex/core/angle.hpp>
#include <geodex/core/concepts.hpp>
#include <geodex/core/sampler.hpp>
#include <numbers>
#include <type_traits>

#include <geodex/metrics/constant_spd.hpp>

namespace geodex {

// ---------------------------------------------------------------------------
// Metric alias
// ---------------------------------------------------------------------------

/// @brief The standard flat metric on \f$ T^n \f$.
///
/// @details The inner product is the standard dot product:
/// \f$ \langle u, v \rangle = u \cdot v \f$. This is exactly
/// `ConstantSPDMetric<Dim>` with the identity weight matrix.
template <int Dim = Eigen::Dynamic>
using TorusFlatMetric = ConstantSPDMetric<Dim>;

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
/// @tparam MetricT Metric policy (default: TorusFlatMetric).
/// @tparam SamplerT Sampler policy for `random_point()` (default: `StochasticSampler`).
template <int Dim = Eigen::Dynamic,
          typename MetricT = TorusFlatMetric<Dim>,
          typename SamplerT = StochasticSampler>
class Torus {
  MetricT metric_;
  int dim_;
  mutable SamplerT sampler_;

 public:
  using Scalar = double;                       ///< Scalar type.
  using Point = Eigen::Vector<double, Dim>;    ///< Point type (angles in \f$ [0, 2\pi)^n \f$).
  using Tangent = Eigen::Vector<double, Dim>;  ///< Tangent vector type.

  /// @brief Runtime query: is `log` the Riemannian logarithm of the metric?
  ///
  /// @details Torus topology has trivial exp/log (addition/wrapping), which is
  /// the Riemannian log exactly when the metric is the identity (standard flat
  /// metric). Anisotropic SPDs are still flat in the mathematical sense but
  /// their geodesics are reparameterized, so we mark them as not log-compatible
  /// and `discrete_geodesic` uses finite differences.
  bool has_riemannian_log_runtime() const {
    if constexpr (std::is_same_v<MetricT, ConstantSPDMetric<Dim>>) {
      return metric_.A_.isApprox(
          Eigen::Matrix<double, Dim, Dim>::Identity(dim_, dim_));
    } else {
      return false;
    }
  }

  /// @brief Fixed-dimension constructor.
  Torus()
    requires(Dim != Eigen::Dynamic)
      : dim_(Dim) {}

  /// @brief Fixed-dimension constructor with custom metric.
  /// @param metric The metric policy instance.
  explicit Torus(MetricT metric)
    requires(Dim != Eigen::Dynamic)
      : metric_(std::move(metric)), dim_(Dim) {}

  /// @brief Dynamic-dimension constructor.
  /// @param n The dimension of the torus.
  explicit Torus(int n)
    requires(Dim == Eigen::Dynamic)
      : metric_(make_default_metric(n)), dim_(n) {}

  /// @brief Dynamic-dimension constructor with custom metric.
  /// @param n The dimension of the torus.
  /// @param metric The metric policy instance.
  Torus(int n, MetricT metric)
    requires(Dim == Eigen::Dynamic)
      : metric_(std::move(metric)), dim_(n) {}

 private:
  /// @brief Build the default metric for dynamic Torus.
  static MetricT make_default_metric(int n) {
    if constexpr (std::is_constructible_v<MetricT, int>) {
      return MetricT(n);
    } else {
      return MetricT{};
    }
  }

 public:

  /// @brief Return the dimension of the torus.
  int dim() const { return dim_; }

  /// @brief Sample a uniformly random point in \f$ [0, 2\pi)^n \f$.
  /// @return A random point on the torus.
  Point random_point() const {
    Eigen::VectorXd box(dim_);
    sampler_.sample_box(dim_, box);
    Point p;
    if constexpr (Dim == Eigen::Dynamic) {
      p.resize(dim_);
    }
    for (int i = 0; i < dim_; ++i) {
      p[i] = box[i] * 2.0 * std::numbers::pi;
    }
    return p;
  }

  /// @brief Project an ambient vector onto the tangent space at \f$ p \f$.
  ///
  /// @details The tangent space of \f$ T^n \f$ is \f$ \mathbb{R}^n \f$ everywhere,
  /// so the projection is the identity.
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
    requires requires { metric_.inner_matrix(p, U, V); }
  {
    return metric_.inner_matrix(p, U, V);
  }

  /// @}

  /// @name Exp / Log
  /// @{

  /// @brief Exponential map: \f$ \exp_p(v) = \mathrm{wrap}(p + v) \f$.
  /// @param p Base point.
  /// @param v Tangent vector.
  /// @return The resulting point, wrapped to \f$ [0, 2\pi)^n \f$.
  Point exp(const Point& p, const Tangent& v) const { return wrap_point<Dim>(p + v); }

  /// @brief Logarithmic map: shortest-path tangent vector from \f$ p \f$ to \f$ q \f$.
  /// @param p Base point.
  /// @param q Target point.
  /// @return The wrapped difference in \f$ [-\pi, \pi)^n \f$.
  Tangent log(const Point& p, const Point& q) const { return wrap_delta<Dim>(q - p); }

  /// @}

  /// @name Derived operations
  /// @{

  /// @brief Geodesic distance via the midpoint approximation.
  /// @param p First point.
  /// @param q Second point.
  /// @return The distance \f$ d(p, q) \f$.
  Scalar distance(const Point& p, const Point& q) const { return distance_midpoint(*this, p, q); }

  /// @brief Injectivity radius of \f$ T^n \f$: \f$ \pi \f$ (half the period).
  ///
  /// @details Assumes the default identity metric and period \f$ 2\pi \f$.
  /// Anisotropic custom metrics on the same topology scale the effective
  /// radius by a factor of the metric's spectrum.
  Scalar injectivity_radius() const { return std::numbers::pi; }

  /// @brief Geodesic interpolation between \f$ p \f$ and \f$ q \f$ at parameter \f$ t \f$.
  /// @param p Start point.
  /// @param q End point.
  /// @param t Interpolation parameter in \f$ [0, 1] \f$.
  /// @return The interpolated point, wrapped to \f$ [0, 2\pi)^n \f$.
  Point geodesic(const Point& p, const Point& q, Scalar t) const { return exp(p, t * log(p, q)); }

  /// @}
};

// Verify the default types satisfy RiemannianManifold.
static_assert(RiemannianManifold<Torus<2>>);
static_assert(RiemannianManifold<Torus<Eigen::Dynamic>>);
static_assert(HasInjectivityRadius<Torus<2>>);

}  // namespace geodex
