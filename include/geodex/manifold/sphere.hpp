/// @file sphere.hpp
/// @brief Sphere manifold \f$ S^n \f$ with interchangeable metric and retraction policies.

#pragma once

#include <Eigen/Core>
#include <cmath>
#include <geodex/algorithm/distance.hpp>
#include <geodex/core/concepts.hpp>
#include <geodex/core/debug.hpp>
#include <geodex/core/metric.hpp>
#include <geodex/core/retraction.hpp>
#include <numbers>
#include <random>
#include <type_traits>

#include <geodex/metrics/constant_spd.hpp>

namespace geodex {

namespace detail {

/// @brief Ambient dimension of \f$ S^n \f$ as a compile-time constant.
///
/// @details \f$ S^n \f$ lives in \f$ \mathbb{R}^{n+1} \f$, so the ambient dim
/// is Dim+1. For the dynamic case, ambient stays dynamic.
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
/// to tangent vectors: \f$ \langle u, v \rangle_p = u \cdot v \f$. This is
/// exactly `ConstantSPDMetric<3>` with the identity weight matrix — exposed
/// here under the familiar "round metric" name.
using SphereRoundMetric = ConstantSPDMetric<3>;

// ---------------------------------------------------------------------------
// Retraction policies
// ---------------------------------------------------------------------------

/// @brief True exponential and logarithmic maps on \f$ S^n \f$ (round geometry).
///
/// @details
/// - `retract(p, v)` computes \f$ \exp_p(v) = \cos(\|v\|)\, p + \sin(\|v\|)\, v / \|v\| \f$
/// - `inverse_retract(p, q)` computes \f$ \log_p(q) \f$ via the arc-length formula
///
/// Both methods are polymorphic in the point type, so the same retraction
/// struct serves every \f$ S^n \f$.
struct SphereExponentialMap {
  /// @brief Exponential map \f$ \exp_p(v) \f$ on \f$ S^n \f$.
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
  template <typename Point>
  EIGEN_STRONG_INLINE Point inverse_retract(const Point& p, const Point& q) const {
    const double cos_theta = p.dot(q);
    const Point v = q - cos_theta * p;
    const double sin_theta = v.norm();
    if (sin_theta < 1e-10) {
      if (cos_theta < 0.0) {
        // TODO: handle cut locus properly — direction is non-unique for antipodal points.
        GEODEX_LOG("SphereExponentialMap: log called at cut locus "
                   "(antipodal points). Returning zero vector.");
      }
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
  /// @brief Projection retraction: normalize \f$ p + v \f$.
  template <typename Point>
  EIGEN_STRONG_INLINE Point retract(const Point& p, const Point& v) const {
    return (p + v).normalized();
  }

  /// @brief Inverse projection retraction.
  template <typename Point>
  EIGEN_STRONG_INLINE Point inverse_retract(const Point& p, const Point& q) const {
    const Point d = q - p;
    const Point v = d - d.dot(p) * p;
    if (v.norm() < 1e-10 && p.dot(q) < 0.0) {
      // TODO: handle cut locus properly.
      GEODEX_LOG("SphereProjectionRetraction: inverse_retract called at cut locus "
                 "(antipodal points). Returning zero vector.");
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
/// \f$ \mathbb{R}^{n+1} \f$; tangent vectors live in the orthogonal complement
/// of the base point.
///
/// @tparam Dim Intrinsic dimension (e.g. `2` for \f$ S^2 \f$), or `Eigen::Dynamic`
///   for runtime sizing. Defaults to `2` (the classical round 2-sphere).
/// @tparam MetricT Metric policy (default: `ConstantSPDMetric<Dim+1>` identity).
/// @tparam RetractionT Retraction policy (default: `SphereExponentialMap`).
template <int Dim = 2,
          typename MetricT = ConstantSPDMetric<detail::sphere_ambient_v<Dim>>,
          typename RetractionT = SphereExponentialMap>
class Sphere {
 public:
  /// @brief Ambient dimension (`Dim + 1` for static, `Eigen::Dynamic` otherwise).
  static constexpr int Ambient = detail::sphere_ambient_v<Dim>;

  using Scalar = double;                            ///< Scalar type.
  using Point = Eigen::Vector<double, Ambient>;     ///< Point type (unit vector in \f$ \mathbb{R}^{n+1} \f$).
  using Tangent = Eigen::Vector<double, Ambient>;   ///< Tangent vector type.

 private:
  MetricT metric_;
  RetractionT retraction_;
  int dim_;  ///< Intrinsic dimension (Dim for static, runtime for dynamic).

 public:
  /// @brief Runtime query: is `log` the Riemannian logarithm of the metric?
  ///
  /// @details Only when the metric is the identity `ConstantSPDMetric<Ambient>`
  /// AND the retraction is the true exponential map does
  /// `grad((1/2) d^2)(x) = -log_x(q)` hold exactly. Other metrics and any
  /// projection retraction fall back to finite-difference natural gradient.
  bool has_riemannian_log_runtime() const {
    if constexpr (std::is_same_v<MetricT, ConstantSPDMetric<Ambient>> &&
                  std::is_same_v<RetractionT, SphereExponentialMap>) {
      return metric_.A_.isApprox(
          Eigen::Matrix<double, Ambient, Ambient>::Identity(dim_ + 1, dim_ + 1));
    } else {
      return false;
    }
  }

  /// @brief Static-dim default constructor: default-construct metric/retraction.
  Sphere()
    requires(Dim != Eigen::Dynamic)
      : dim_(Dim) {
    log_construction();
  }

  /// @brief Static-dim constructor with explicit metric and retraction policies.
  explicit Sphere(MetricT metric, RetractionT retraction = {})
    requires(Dim != Eigen::Dynamic)
      : metric_(std::move(metric)), retraction_(std::move(retraction)), dim_(Dim) {
    log_construction();
  }

  /// @brief Dynamic-dim constructor: takes the intrinsic dimension \f$ n \f$ of \f$ S^n \f$.
  explicit Sphere(int n)
    requires(Dim == Eigen::Dynamic)
      : metric_(make_default_metric(n + 1)), dim_(n) {
    log_construction();
  }

  /// @brief Dynamic-dim constructor with explicit metric and retraction.
  Sphere(int n, MetricT metric, RetractionT retraction = {})
    requires(Dim == Eigen::Dynamic)
      : metric_(std::move(metric)), retraction_(std::move(retraction)), dim_(n) {
    log_construction();
  }

  /// @brief Return the intrinsic dimension \f$ n \f$ of \f$ S^n \f$.
  int dim() const { return dim_; }

  /// @brief Sample a uniformly random point on \f$ S^n \f$ via Marsaglia's method.
  /// @return A unit vector in \f$ \mathbb{R}^{n+1} \f$.
  Point random_point() const {
    thread_local std::mt19937 gen{std::random_device{}()};
    std::normal_distribution<double> dist(0.0, 1.0);
    Point p;
    if constexpr (Ambient == Eigen::Dynamic) {
      p.resize(dim_ + 1);
    }
    for (int i = 0; i < p.size(); ++i) {
      p[i] = dist(gen);
    }
    return p.normalized();
  }

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
    requires requires { metric_.inner_matrix(p, U, V); }
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
  Scalar distance(const Point& p, const Point& q) const {
    double dot = p.dot(q);
    if (dot < -1.0 + 1e-10) {
      return std::numbers::pi;
    }
    return distance_midpoint(*this, p, q);
  }

  /// @brief Injectivity radius of the round n-sphere: \f$ \pi \f$.
  ///
  /// @details Assumes the default identity metric. Anisotropic custom metrics
  /// scale the effective radius by a factor of the metric's spectrum — users
  /// with custom metrics must compute their own bound.
  Scalar injectivity_radius() const { return std::numbers::pi; }

  /// @brief Geodesic interpolation between \f$ p \f$ and \f$ q \f$ at parameter \f$ t \f$.
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

  void log_construction() const {
    GEODEX_LOG("Sphere<" << Dim << ", " << detail::type_name<MetricT>() << ", "
                         << detail::type_name<RetractionT>() << "> created (dim=" << dim() << ")");
  }
};

// Verify the composed types satisfy RiemannianManifold.
static_assert(RiemannianManifold<Sphere<>>);
static_assert(RiemannianManifold<Sphere<2, ConstantSPDMetric<3>>>);
static_assert(RiemannianManifold<Sphere<2, SphereRoundMetric, SphereProjectionRetraction>>);
static_assert(RiemannianManifold<Sphere<3>>);
static_assert(RiemannianManifold<Sphere<Eigen::Dynamic>>);

}  // namespace geodex
