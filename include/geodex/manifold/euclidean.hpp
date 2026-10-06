/// @file euclidean.hpp
/// @brief Euclidean manifold \f$ \mathbb{R}^n \f$ with compile-time or dynamic dimension.

#pragma once

#include <cmath>

#include <limits>
#include <stdexcept>
#include <type_traits>

#include <Eigen/Core>

#include "geodex/algorithm/distance.hpp"
#include "geodex/core/concepts.hpp"
#include "geodex/core/sampler.hpp"
#include "geodex/metrics/constant_spd.hpp"
#include "geodex/metrics/identity.hpp"

namespace geodex {

// ---------------------------------------------------------------------------
// Metric alias
// ---------------------------------------------------------------------------

/// @brief The standard Euclidean metric on \f$ \mathbb{R}^n \f$.
///
/// @details The inner product is the standard dot product
/// \f$ \langle u, v \rangle = u \cdot v \f$. The metric is stateless and stores nothing.
template <int Dim = Eigen::Dynamic>
using EuclideanStandardMetric = IdentityMetric<Dim>;

// ---------------------------------------------------------------------------
// Euclidean manifold
// ---------------------------------------------------------------------------

/// @brief Euclidean manifold \f$ \mathbb{R}^n \f$ parameterized by dimension and metric policy.
///
/// @details Supports both compile-time fixed dimension (e.g., `Euclidean<3>`) and
/// runtime dynamic dimension (`Euclidean<Eigen::Dynamic>`). The exp and log maps are
/// addition and subtraction on the flat space.
///
/// @tparam Dim Compile-time dimension, or `Eigen::Dynamic`.
/// @tparam MetricT Metric policy (default EuclideanStandardMetric).
/// @tparam SamplerT Sampler policy for `random_point()` (default `ScrambledHaltonSampler`).
template <int Dim = Eigen::Dynamic, typename MetricT = EuclideanStandardMetric<Dim>,
          typename SamplerT = ScrambledHaltonSampler>
class Euclidean {
 public:
  using Scalar = double;                       ///< Scalar type.
  using Point = Eigen::Vector<double, Dim>;    ///< Point type.
  using SamplerType = SamplerT;                ///< Sampler policy backing random_point().
  using Tangent = Eigen::Vector<double, Dim>;  ///< Tangent vector type.

  /// @brief Runtime check whether `log` is the Riemannian logarithm of the metric.
  ///
  /// @details True only when the metric is the identity, `IdentityMetric<Dim>` or an
  /// identity `ConstantSPDMetric<Dim>`, the standard Euclidean dot product. For other SPD
  /// metrics, `discrete_geodesic` falls back to finite differences.
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

  /// @brief Fixed-dimension constructor with default bounds \f$[-1, 1]^n\f$.
  Euclidean()
    requires(Dim != Eigen::Dynamic)
      : dim_(Dim),
        lo_(Eigen::VectorXd::Constant(Dim, -1.0)),
        hi_(Eigen::VectorXd::Constant(Dim, 1.0)),
        sample_buf_(Dim) {}

  /// @brief Fixed-dimension constructor with custom metric.
  /// @param metric The metric policy instance.
  explicit Euclidean(MetricT metric)
    requires(Dim != Eigen::Dynamic)
      : metric_(std::move(metric)),
        dim_(Dim),
        lo_(Eigen::VectorXd::Constant(Dim, -1.0)),
        hi_(Eigen::VectorXd::Constant(Dim, 1.0)),
        sample_buf_(Dim) {}

  /// @brief Dynamic-dimension constructor.
  /// @param n The dimension of the space.
  explicit Euclidean(int n)
    requires(Dim == Eigen::Dynamic)
      : metric_(make_default_metric(n)),
        dim_(n),
        lo_(Eigen::VectorXd::Constant(n, -1.0)),
        hi_(Eigen::VectorXd::Constant(n, 1.0)),
        sample_buf_(n) {}

  /// @brief Dynamic-dimension constructor with custom metric.
  /// @param n The dimension of the space.
  /// @param metric The metric policy instance.
  Euclidean(int n, MetricT metric)
    requires(Dim == Eigen::Dynamic)
      : metric_(std::move(metric)),
        dim_(n),
        lo_(Eigen::VectorXd::Constant(n, -1.0)),
        hi_(Eigen::VectorXd::Constant(n, 1.0)),
        sample_buf_(n) {}

  /// @brief Set the sampling bounds. `random_point()` returns only values inside
  /// them. `exp`, `log` and the metric do not change.
  /// @throws std::invalid_argument unless `lo` and `hi` hold `dim()` entries each.
  void set_sampling_bounds(const Eigen::VectorXd& lo, const Eigen::VectorXd& hi) {
    if (lo.size() != dim() || hi.size() != dim()) {
      throw std::invalid_argument("Euclidean: sampling bounds must hold dim() entries each");
    }
    lo_ = lo;
    hi_ = hi;
  }

  /// @brief Lower per-dimension sampling bound.
  const Eigen::VectorXd& lo() const { return lo_; }

  /// @brief Upper per-dimension sampling bound.
  const Eigen::VectorXd& hi() const { return hi_; }

  /// @brief Return the dimension of the space.
  int dim() const { return dim_; }

  /// @brief Number of unit-cube coordinates that from_unit_cube consumes.
  int unit_cube_dim() const { return dim_; }

  /// @brief Map unit-cube coordinates to a point by affine rescaling to [lo, hi].
  Point from_unit_cube(Eigen::Ref<const Eigen::VectorXd> u) const {
    detail::require_unit_cube_size(u.size(), dim_);
    Point p;
    if constexpr (Dim == Eigen::Dynamic) {
      p.resize(dim_);
    }
    for (int i = 0; i < dim_; ++i) {
      p[i] = lo_[i] + u[i] * (hi_[i] - lo_[i]);
    }
    return p;
  }

  /// @brief Sample a point uniformly in [lo, hi]^n (default [-1, 1]^n).
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

  /// @brief Project an ambient vector onto the tangent space at \f$ p \f$.
  ///
  /// @details The tangent space of \f$ \mathbb{R}^n \f$ is \f$ \mathbb{R}^n \f$
  /// everywhere, and the projection is the identity.
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

  /// @name Exp / Log
  /// @{

  /// @brief Exponential map \f$ \exp_p(v) = p + v \f$.
  /// @param p Base point.
  /// @param v Tangent vector.
  /// @return The resulting point.
  Point exp(const Point& p, const Tangent& v) const { return p + v; }

  /// @brief Logarithmic map \f$ \log_p(q) = q - p \f$.
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

  /// @brief Injectivity radius of \f$ \mathbb{R}^n \f$, \f$ \infty \f$.
  ///
  /// @details The injectivity radius of flat Euclidean space is infinite for every
  /// metric. Anisotropic custom metrics change geodesic directions, and the space
  /// stays simply connected and geodesically complete.
  Scalar injectivity_radius() const { return std::numeric_limits<double>::infinity(); }

  /// @brief Geodesic interpolation \f$ (1 - t) p + t q \f$.
  /// @param p Start point.
  /// @param q End point.
  /// @param t Interpolation parameter in \f$ [0, 1] \f$.
  /// @return The interpolated point.
  Point geodesic(const Point& p, const Point& q, Scalar t) const { return (1.0 - t) * p + t * q; }

  /// @}

 private:
  /// @brief Build the default metric for dynamic Euclidean, `MetricT(n)` when the metric
  /// takes a size and a default-constructed metric otherwise.
  static MetricT make_default_metric(int n) {
    if constexpr (std::is_constructible_v<MetricT, int>) {
      return MetricT(n);
    } else {
      return MetricT{};
    }
  }

  MetricT metric_;
  int dim_;
  Eigen::VectorXd lo_;                  ///< Lower sampling bounds (default -1^n).
  Eigen::VectorXd hi_;                  ///< Upper sampling bounds (default  1^n).
  mutable SamplerT sampler_;            ///< Sampler used by `random_point`.
  mutable Eigen::VectorXd sample_buf_;  ///< Preallocated buffer for sampler output.
};

// Verify the default type satisfies RiemannianManifold.
static_assert(RiemannianManifold<Euclidean<3>>);
static_assert(RiemannianManifold<Euclidean<Eigen::Dynamic>>);
static_assert(HasInjectivityRadius<Euclidean<3>>);
static_assert(RiemannianManifold<Euclidean<3, ConstantSPDMetric<3>>>);

}  // namespace geodex
