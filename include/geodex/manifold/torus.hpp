/// @file torus.hpp
/// @brief Flat torus manifold \f$ T^n \f$ with periodic angle coordinates.

#pragma once

#include <Eigen/Core>
#include <cmath>
#include <geodex/algorithm/distance.hpp>
#include <geodex/core/angle.hpp>
#include <geodex/core/concepts.hpp>
#include <numbers>
#include <random>
#include <type_traits>

namespace geodex {

// ---------------------------------------------------------------------------
// Metric
// ---------------------------------------------------------------------------

/// @brief Standard flat metric on \f$ T^n \f$.
///
/// @details The inner product is the standard dot product:
/// \f$ \langle u, v \rangle = u \cdot v \f$.
/// The injectivity radius is \f$ \pi \f$ (half the period).
///
/// @tparam Dim Compile-time dimension, or `Eigen::Dynamic`.
template <int Dim = Eigen::Dynamic>
struct TorusFlatMetric {
  /// @brief Compute the inner product \f$ \langle u, v \rangle = u \cdot v \f$.
  /// @param u First tangent vector.
  /// @param v Second tangent vector.
  /// @return The inner product value.
  double inner(const Eigen::Vector<double, Dim>& /*p*/, const Eigen::Vector<double, Dim>& u,
               const Eigen::Vector<double, Dim>& v) const {
    return u.dot(v);
  }

  /// @brief Compute the flat norm \f$ \|v\| = \sqrt{v \cdot v} \f$.
  /// @param p Base point.
  /// @param v Tangent vector.
  /// @return The norm value.
  double norm(const Eigen::Vector<double, Dim>& p, const Eigen::Vector<double, Dim>& v) const {
    return std::sqrt(inner(p, v, v));
  }

  /// @brief Return the injectivity radius \f$ \pi \f$.
  double injectivity_radius() const { return std::numbers::pi; }
};

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
template <int Dim = Eigen::Dynamic, typename MetricT = TorusFlatMetric<Dim>>
class Torus {
  MetricT metric_;
  int dim_;

 public:
  using Scalar = double;                       ///< Scalar type.
  using Point = Eigen::Vector<double, Dim>;    ///< Point type (angles in \f$ [0, 2\pi)^n \f$).
  using Tangent = Eigen::Vector<double, Dim>;  ///< Tangent vector type.

  /// @brief Compile-time flag: is `log` the Riemannian logarithm of the metric?
  ///
  /// @details Torus topology has trivial exp/log (addition/wrapping), which is
  /// the Riemannian log exactly when the metric is the default flat metric.
  /// Any other metric (e.g. `ConstantSPDMetric`) may still be a flat metric in
  /// the mathematical sense (geodesics coincide with the flat ones up to
  /// reparameterization), but we conservatively mark it as not log-compatible
  /// so `discrete_geodesic` uses finite differences to compute the correct
  /// gradient under that metric.
  static constexpr bool has_riemannian_log =
      std::is_same_v<MetricT, TorusFlatMetric<Dim>>;

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
      : dim_(n) {}

  /// @brief Dynamic-dimension constructor with custom metric.
  /// @param n The dimension of the torus.
  /// @param metric The metric policy instance.
  Torus(int n, MetricT metric)
    requires(Dim == Eigen::Dynamic)
      : metric_(std::move(metric)), dim_(n) {}

  /// @brief Return the dimension of the torus.
  int dim() const { return dim_; }

  /// @brief Sample a uniformly random point in \f$ [0, 2\pi)^n \f$.
  /// @return A random point on the torus.
  Point random_point() const {
    thread_local std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<double> dist(0.0, 2.0 * std::numbers::pi);
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

  /// @brief Injectivity radius — only available when the metric provides it.
  Scalar injectivity_radius() const
    requires requires { metric_.injectivity_radius(); }
  {
    return metric_.injectivity_radius();
  }

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
