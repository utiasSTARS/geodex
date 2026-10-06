/// @file sphere.hpp
/// @brief Sphere manifold \f$ S^n \f$ with interchangeable metric and retraction policies.

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
#include "geodex/metrics/constant_spd.hpp"
#include "geodex/metrics/identity.hpp"
#include "geodex/utils/normal.hpp"

namespace geodex {

namespace detail {

/// @brief Ambient dimension of \f$ S^n \f$ as a compile-time constant.
///
/// @details \f$ S^n \f$ lives in \f$ \mathbb{R}^{n+1} \f$, and the ambient dim
/// is Dim+1. In the dynamic case it stays dynamic.
template <int Dim>
inline constexpr int sphere_ambient_v = Dim + 1;
template <>
inline constexpr int sphere_ambient_v<Eigen::Dynamic> = Eigen::Dynamic;

}  // namespace detail

// ---------------------------------------------------------------------------
// Metric alias
// ---------------------------------------------------------------------------

/// @brief The standard round (bi-invariant) metric on \f$ S^2 \f$.
///
/// @details The inner product is the ambient Euclidean dot product restricted
/// to tangent vectors, \f$ \langle u, v \rangle_p = u \cdot v \f$. The metric is
/// stateless and stores nothing.
using SphereRoundMetric = IdentityMetric<3>;

// ---------------------------------------------------------------------------
// Retraction policies
// ---------------------------------------------------------------------------

/// @brief True exponential and logarithmic maps on \f$ S^n \f$ (round geometry).
///
/// @details
/// - `retract(p, v)` computes \f$ \exp_p(v) = \cos(\|v\|)\, p + \sin(\|v\|)\, v / \|v\| \f$
/// - `inverse_retract(p, q)` computes \f$ \log_p(q) \f$ via the arc-length formula
///
/// Both methods are polymorphic in the point type, and the same retraction
/// struct serves every \f$ S^n \f$.
struct SphereExponentialMap {
  /// @brief Exponential map \f$ \exp_p(v) \f$ on \f$ S^n \f$.
  /// @param p Base point on the sphere.
  /// @param v Tangent vector at \f$ p \f$.
  /// @return The resulting point on the sphere.
  template <typename Point>
  EIGEN_STRONG_INLINE Point retract(const Point& p, const Point& v) const {
    const double theta = v.norm();
    if (theta < 1e-10) {
      return p;
    }
    const double inv_theta = 1.0 / theta;
    return std::cos(theta) * p + (std::sin(theta) * inv_theta) * v;
  }

  /// @brief Logarithmic map \f$ \log_p(q) \f$ on \f$ S^n \f$.
  /// @param p Base point on the sphere.
  /// @param q Target point on the sphere.
  /// @return Tangent vector at \f$ p \f$ such that \f$ \exp_p(v) = q \f$.
  template <typename Point>
  EIGEN_STRONG_INLINE Point inverse_retract(const Point& p, const Point& q) const {
    const double cos_theta = p.dot(q);
    const Point v = q - cos_theta * p;
    const double sin_theta = v.norm();
    if (sin_theta < 1e-10) {
      // Coincident points give the zero vector. At antipodal points the direction is not
      // unique, and the map also returns the zero vector.
      return Point::Zero(p.size());
    }
    const double theta = std::atan2(sin_theta, cos_theta);
    return (theta / sin_theta) * v;
  }
};

/// @brief First-order projection retraction on \f$ S^n \f$.
///
/// @details
/// - `retract(p, v)` computes \f$ R_p(v) = (p + v) / \|p + v\| \f$
/// - `inverse_retract(p, q)` projects \f$ q - p \f$ onto \f$ T_p S^n \f$
struct SphereProjectionRetraction {
  /// @brief Projection retraction that normalizes \f$ p + v \f$.
  /// @param p Base point on the sphere.
  /// @param v Tangent vector at \f$ p \f$.
  /// @return The retracted point on the sphere.
  template <typename Point>
  EIGEN_STRONG_INLINE Point retract(const Point& p, const Point& v) const {
    return (p + v).normalized();
  }

  /// @brief Inverse projection retraction.
  /// @param p Base point on the sphere.
  /// @param q Target point on the sphere.
  /// @return Tangent vector at \f$ p \f$ approximating \f$ \log_p(q) \f$.
  template <typename Point>
  EIGEN_STRONG_INLINE Point inverse_retract(const Point& p, const Point& q) const {
    const Point d = q - p;
    const Point v = d - d.dot(p) * p;
    if (v.norm() < 1e-10 && p.dot(q) < 0.0) {
      // At antipodal points the direction is not unique, and the map returns the zero vector.
      return Point::Zero(p.size());
    }
    return v;
  }
};

// Verify retraction concepts at the canonical S^2 signature.
static_assert(Retraction<SphereExponentialMap, Eigen::Vector3d, Eigen::Vector3d>);
static_assert(Retraction<SphereProjectionRetraction, Eigen::Vector3d, Eigen::Vector3d>);

// ---------------------------------------------------------------------------
// Sphere manifold
// ---------------------------------------------------------------------------

/// @brief The n-sphere \f$ S^n \f$ parameterized by dimension, metric and retraction.
///
/// @details This class composes a metric policy and a retraction policy to form a
/// complete `RiemannianManifold`. Points are represented as unit vectors in
/// \f$ \mathbb{R}^{n+1} \f$. Tangent vectors live in the orthogonal complement
/// of the base point.
///
/// @tparam Dim Intrinsic dimension (e.g. `2` for \f$ S^2 \f$), or `Eigen::Dynamic`
///   for runtime sizing. Defaults to `2` (the classical round 2-sphere).
/// @tparam MetricT Metric policy (default `IdentityMetric<Dim+1>`).
/// @tparam RetractionT Retraction policy (default `SphereExponentialMap`).
template <int Dim = 2, typename MetricT = IdentityMetric<detail::sphere_ambient_v<Dim>>,
          typename RetractionT = SphereExponentialMap, typename SamplerT = ScrambledHaltonSampler>
class Sphere {
 public:
  /// @brief Ambient dimension (`Dim + 1` for static, `Eigen::Dynamic` otherwise).
  static constexpr int Ambient = detail::sphere_ambient_v<Dim>;

  using Scalar = double;  ///< Scalar type.
  using Point =
      Eigen::Vector<double, Ambient>;  ///< Point type (unit vector in \f$ \mathbb{R}^{n+1} \f$).
  using Tangent = Eigen::Vector<double, Ambient>;  ///< Tangent vector type.
  using SamplerType = SamplerT;                    ///< Sampler policy backing random_point().

  /// @brief Runtime check whether `log` is the Riemannian logarithm of the metric.
  ///
  /// @details `grad((1/2) d^2)(x) = -log_x(q)` holds exactly only when the metric is
  /// the identity, `IdentityMetric<Ambient>` or an identity `ConstantSPDMetric<Ambient>`,
  /// and the retraction is the true exponential map. Other metrics and a projection
  /// retraction fall back to the finite-difference natural gradient.
  bool has_riemannian_log_runtime() const {
    if constexpr (std::is_same_v<RetractionT, SphereExponentialMap>) {
      if constexpr (std::is_same_v<MetricT, IdentityMetric<Ambient>>) {
        return true;
      } else if constexpr (std::is_same_v<MetricT, ConstantSPDMetric<Ambient>>) {
        return metric_.weight_matrix().isApprox(
            Eigen::Matrix<double, Ambient, Ambient>::Identity(dim_ + 1, dim_ + 1));
      } else {
        return false;
      }
    } else {
      return false;
    }
  }

  /// @brief Static-dim default constructor with a default metric and retraction.
  Sphere()
    requires(Dim != Eigen::Dynamic)
      : dim_(Dim) {}

  /// @brief Static-dim constructor with explicit metric and retraction policies.
  explicit Sphere(MetricT metric, RetractionT retraction = {})
    requires(Dim != Eigen::Dynamic)
      : metric_(std::move(metric)), retraction_(std::move(retraction)), dim_(Dim) {}

  /// @brief Dynamic-dim constructor that takes the intrinsic dimension \f$ n \f$ of \f$ S^n \f$.
  explicit Sphere(int n)
    requires(Dim == Eigen::Dynamic)
      : metric_(make_default_metric(n + 1)), dim_(n) {}

  /// @brief Dynamic-dim constructor with explicit metric and retraction.
  Sphere(int n, MetricT metric, RetractionT retraction = {})
    requires(Dim == Eigen::Dynamic)
      : metric_(std::move(metric)), retraction_(std::move(retraction)), dim_(n) {}

  /// @brief Return the intrinsic dimension \f$ n \f$ of \f$ S^n \f$.
  int dim() const { return dim_; }

  /// @brief Number of unit-cube coordinates that from_unit_cube consumes.
  ///
  /// @details 1 for the circle, 2 for the 2-sphere via the equal-area map, and
  /// \f$ n+1 \f$ for higher spheres via normalized normals.
  int unit_cube_dim() const {
    if (dim_ == 1) return 1;
    if (dim_ == 2) return 2;
    return dim_ + 1;
  }

  /// @brief Map unit-cube coordinates to a uniform point on \f$ S^n \f$.
  ///
  /// @details \f$ S^1 \f$ uses the angle map, \f$ S^2 \f$ uses the Archimedes
  /// hat-box equal-area map, and higher spheres use \f$ n+1 \f$ inverse-CDF
  /// normal variates then normalize (Muller 1959).
  /// @param u Unit-cube coordinates.
  /// @return A unit vector in \f$ \mathbb{R}^{n+1} \f$.
  Point from_unit_cube(Eigen::Ref<const Eigen::VectorXd> u) const {
    detail::require_unit_cube_size(u.size(), unit_cube_dim());
    Point p;
    if constexpr (Ambient == Eigen::Dynamic) {
      p.resize(dim_ + 1);
    }
    if (dim_ == 1) {
      const double a = 2.0 * std::numbers::pi * u[0];
      p[0] = std::cos(a);
      p[1] = std::sin(a);
      return p;
    }
    if (dim_ == 2) {
      const double z = 2.0 * u[0] - 1.0;
      const double phi = 2.0 * std::numbers::pi * u[1];
      const double r = std::sqrt(1.0 - z * z);
      p[0] = r * std::cos(phi);
      p[1] = r * std::sin(phi);
      p[2] = z;
      return p;
    }
    for (int i = 0; i <= dim_; ++i) {
      p[i] = utils::normal_quantile(u[i]);
    }
    p.normalize();
    return p;
  }

  /// @brief Sample a uniformly random point on \f$ S^n \f$.
  Point random_point() const {
    const int k = unit_cube_dim();
    sample_buf_.resize(k);
    sampler_.sample(k, sample_buf_);
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

  /// @brief Project a vector onto the tangent space at \f$ p \f$.
  /// @param p Base point on the sphere.
  /// @param v Ambient vector to project.
  /// @return The tangential component of \f$ v \f$.
  Tangent project(const Point& p, const Tangent& v) const { return v - v.dot(p) * p; }

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
  /// @param p Base point on the sphere.
  /// @param v Tangent vector at \f$ p \f$.
  /// @return The resulting point on the sphere.
  Point exp(const Point& p, const Tangent& v) const { return retraction_.retract(p, v); }

  /// @brief Logarithmic map (or inverse retraction) \f$ \log_p(q) \f$.
  /// @param p Base point on the sphere.
  /// @param q Target point on the sphere.
  /// @return Tangent vector at \f$ p \f$ such that \f$ \exp_p(v) \approx q \f$.
  Tangent log(const Point& p, const Point& q) const { return retraction_.inverse_retract(p, q); }

  /// @}

  /// @name Derived operations
  /// @{

  /// @brief Geodesic distance \f$ d(p, q) \f$ via the midpoint approximation.
  /// @param p First point on the sphere.
  /// @param q Second point on the sphere.
  /// @return The geodesic distance.
  Scalar distance(const Point& p, const Point& q) const {
    double dot = p.dot(q);
    if (dot < -1.0 + 1e-10) {
      return std::numbers::pi;
    }
    return distance_midpoint(*this, p, q);
  }

  /// @brief Injectivity radius of the round n-sphere, \f$ \pi \f$.
  ///
  /// @details Returns the topological injectivity radius for the default round
  /// (identity) metric. For anisotropic custom metrics the effective radius is the
  /// smaller \f$ \pi / \sqrt{\lambda_{\max}(A)} \f$, with \f$ \lambda_{\max} \f$ the
  /// largest eigenvalue of the weight matrix. This value is an upper bound.
  /// `discrete_geodesic` caps steps with it and may retry when the true radius is
  /// smaller.
  Scalar injectivity_radius() const { return std::numbers::pi; }

  /// @brief Geodesic interpolation between \f$ p \f$ and \f$ q \f$ at parameter \f$ t \f$.
  /// @param p Start point.
  /// @param q End point.
  /// @param t Interpolation parameter in \f$ [0, 1] \f$.
  /// @return The interpolated point on the sphere.
  Point geodesic(const Point& p, const Point& q, Scalar t) const { return exp(p, t * log(p, q)); }

  /// @}

 private:
  /// @brief Build the default metric for dynamic Sphere (identity of ambient size).
  static MetricT make_default_metric(int ambient_size) {
    if constexpr (std::is_constructible_v<MetricT, int>) {
      return MetricT(ambient_size);
    } else {
      return MetricT{};
    }
  }

  MetricT metric_;
  RetractionT retraction_;
  int dim_;                   ///< Intrinsic dimension (Dim for static, runtime for dynamic).
  mutable SamplerT sampler_;  ///< Sampler used by `random_point`.
  mutable Eigen::VectorXd sample_buf_;  ///< Preallocated buffer for unit-cube samples.
};

// Verify the composed types satisfy RiemannianManifold.
static_assert(RiemannianManifold<Sphere<>>);
static_assert(RiemannianManifold<Sphere<2, ConstantSPDMetric<3>>>);
static_assert(RiemannianManifold<Sphere<2, SphereRoundMetric, SphereProjectionRetraction>>);
static_assert(RiemannianManifold<Sphere<3>>);
static_assert(RiemannianManifold<Sphere<Eigen::Dynamic>>);

}  // namespace geodex
