/// @file euclidean.hpp
/// @brief Euclidean manifold \f$ \mathbb{R}^n \f$ with compile-time or dynamic dimension.

#pragma once

#include <Eigen/Core>
#include <cmath>
#include <geodex/algorithm/distance.hpp>
#include <geodex/core/concepts.hpp>
#include <limits>
#include <random>
#include <type_traits>

#include <geodex/metrics/constant_spd.hpp>

namespace geodex {

// ---------------------------------------------------------------------------
// Metric alias
// ---------------------------------------------------------------------------

/// @brief The standard Euclidean metric on \f$ \mathbb{R}^n \f$.
///
/// @details The inner product is the standard dot product:
/// \f$ \langle u, v \rangle = u \cdot v \f$. This is exactly
/// `ConstantSPDMetric<Dim>` with the identity weight matrix.
template <int Dim = Eigen::Dynamic>
using EuclideanStandardMetric = ConstantSPDMetric<Dim>;

// ---------------------------------------------------------------------------
// Euclidean manifold
// ---------------------------------------------------------------------------

/// @brief Euclidean manifold \f$ \mathbb{R}^n \f$ parameterized by dimension and metric policy.
///
/// @details Supports both compile-time fixed dimension (e.g., `Euclidean<3>`) and
/// runtime dynamic dimension (`Euclidean<Eigen::Dynamic>`). The exp/log maps are
/// trivial (addition/subtraction) since the space is flat.
///
/// @tparam Dim Compile-time dimension, or `Eigen::Dynamic`.
/// @tparam MetricT Metric policy (default: EuclideanStandardMetric).
template <int Dim = Eigen::Dynamic, typename MetricT = EuclideanStandardMetric<Dim>>
class Euclidean {
  MetricT metric_;
  int dim_;

 public:
  using Scalar = double;                       ///< Scalar type.
  using Point = Eigen::Vector<double, Dim>;    ///< Point type.
  using Tangent = Eigen::Vector<double, Dim>;  ///< Tangent vector type.

  /// @brief Runtime query: is `log` the Riemannian logarithm of the metric?
  ///
  /// @details True only when the metric is the identity `ConstantSPDMetric<Dim>`
  /// (the standard Euclidean dot product). Non-identity SPDs live under a
  /// different inner product, so `discrete_geodesic` falls back to finite
  /// differences for those cases.
  bool has_riemannian_log_runtime() const {
    if constexpr (std::is_same_v<MetricT, ConstantSPDMetric<Dim>>) {
      return metric_.A_.isApprox(
          Eigen::Matrix<double, Dim, Dim>::Identity(dim_, dim_));
    } else {
      return false;
    }
  }

  /// @brief Fixed-dimension constructor.
  Euclidean()
    requires(Dim != Eigen::Dynamic)
      : dim_(Dim) {}

  /// @brief Fixed-dimension constructor with custom metric.
  /// @param metric The metric policy instance.
  explicit Euclidean(MetricT metric)
    requires(Dim != Eigen::Dynamic)
      : metric_(std::move(metric)), dim_(Dim) {}

  /// @brief Dynamic-dimension constructor.
  /// @param n The dimension of the space.
  explicit Euclidean(int n)
    requires(Dim == Eigen::Dynamic)
      : metric_(make_default_metric(n)), dim_(n) {}

  /// @brief Dynamic-dimension constructor with custom metric.
  /// @param n The dimension of the space.
  /// @param metric The metric policy instance.
  Euclidean(int n, MetricT metric)
    requires(Dim == Eigen::Dynamic)
      : metric_(std::move(metric)), dim_(n) {}

 private:
  /// @brief Build the default metric for dynamic Euclidean: `ConstantSPDMetric<Dynamic>(n)`
  /// when applicable, otherwise a default-constructed metric.
  static MetricT make_default_metric(int n) {
    if constexpr (std::is_constructible_v<MetricT, int>) {
      return MetricT(n);
    } else {
      return MetricT{};
    }
  }

 public:

  /// @brief Return the dimension of the space.
  int dim() const { return dim_; }

  /// @brief Sample a random point from a standard normal distribution.
  /// @return A point in \f$ \mathbb{R}^n \f$.
  Point random_point() const {
    thread_local std::mt19937 gen{std::random_device{}()};
    std::normal_distribution<double> dist(0.0, 1.0);
    Point p;
    if constexpr (Dim == Eigen::Dynamic) {
      p.resize(dim_);
    }
    for (int i = 0; i < dim_; ++i) {
      p[i] = dist(gen);
    }
    return p;
  }

  /// @brief Project an ambient vector onto the tangent space at \f$ p \f$.
  ///
  /// @details The tangent space of \f$ \mathbb{R}^n \f$ is \f$ \mathbb{R}^n \f$
  /// everywhere, so the projection is the identity.
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

  /// @brief Exponential map: \f$ \exp_p(v) = p + v \f$.
  /// @param p Base point.
  /// @param v Tangent vector.
  /// @return The resulting point.
  Point exp(const Point& p, const Tangent& v) const { return p + v; }

  /// @brief Logarithmic map: \f$ \log_p(q) = q - p \f$.
  /// @param p Base point.
  /// @param q Target point.
  /// @return The tangent vector from \f$ p \f$ to \f$ q \f$.
  Tangent log(const Point& p, const Point& q) const { return q - p; }

  /// @}

  /// @name Derived operations
  /// @{

  /// @brief Geodesic distance via the midpoint approximation.
  /// @param p First point.
  /// @param q Second point.
  /// @return The distance \f$ d(p, q) \f$.
  Scalar distance(const Point& p, const Point& q) const { return distance_midpoint(*this, p, q); }

  /// @brief Injectivity radius of \f$ \mathbb{R}^n \f$: \f$ \infty \f$.
  ///
  /// @details Assumes the default identity metric — anisotropic custom metrics
  /// still have an infinite radius on flat space, so this constant is correct
  /// for every metric on \f$ \mathbb{R}^n \f$.
  Scalar injectivity_radius() const { return std::numeric_limits<double>::infinity(); }

  /// @brief Geodesic interpolation: \f$ (1 - t) p + t q \f$.
  /// @param p Start point.
  /// @param q End point.
  /// @param t Interpolation parameter in \f$ [0, 1] \f$.
  /// @return The interpolated point.
  Point geodesic(const Point& p, const Point& q, Scalar t) const { return (1.0 - t) * p + t * q; }

  /// @}
};

// Verify the default type satisfies RiemannianManifold.
static_assert(RiemannianManifold<Euclidean<3>>);
static_assert(RiemannianManifold<Euclidean<Eigen::Dynamic>>);
static_assert(HasInjectivityRadius<Euclidean<3>>);
static_assert(RiemannianManifold<Euclidean<3, ConstantSPDMetric<3>>>);

}  // namespace geodex
