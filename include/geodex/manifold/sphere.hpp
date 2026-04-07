/// @file sphere.hpp
/// @brief Sphere manifold \f$ S^2 \f$ with interchangeable metric and retraction policies.

#pragma once

#include <Eigen/Core>
#include <cmath>
#include <geodex/algorithm/distance.hpp>
#include <geodex/core/concepts.hpp>
#include <geodex/core/debug.hpp>
#include <geodex/core/retraction.hpp>
#include <numbers>
#include <random>
#include <string_view>
#include <type_traits>

#include <geodex/metrics/constant_spd.hpp>

namespace geodex {

namespace detail {

/// Extract a clean type name from compiler intrinsics.
template <typename T>
constexpr std::string_view type_name() {
#if defined(__clang__) || defined(__GNUC__)
  // __PRETTY_FUNCTION__ looks like:
  //   "std::string_view geodex::detail::type_name() [T = SphereRoundMetric]"
  std::string_view fn = __PRETTY_FUNCTION__;
  auto start = fn.find("T = ");
  if (start == std::string_view::npos) return fn;
  start += 4;
  auto end = fn.rfind(']');
  if (end == std::string_view::npos) return fn.substr(start);
  return fn.substr(start, end - start);
#else
  return typeid(T).name();
#endif
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Metric policies
// ---------------------------------------------------------------------------

/// @brief Standard round (bi-invariant) metric on \f$ S^2 \f$.
///
/// @details The inner product is the ambient Euclidean dot product restricted
/// to tangent vectors: \f$ \langle u, v \rangle_p = u \cdot v \f$.
/// The injectivity radius is \f$ \pi \f$.
struct SphereRoundMetric {
  /// @brief Compute the inner product \f$ \langle u, v \rangle_p = u \cdot v \f$.
  /// @param u First tangent vector.
  /// @param v Second tangent vector.
  /// @return The inner product value.
  double inner(const Eigen::Vector3d& /*p*/, const Eigen::Vector3d& u,
               const Eigen::Vector3d& v) const {
    return u.dot(v);
  }

  /// @brief Compute the norm \f$ \|v\|_p = \sqrt{\langle v, v \rangle_p} \f$.
  /// @param p Base point.
  /// @param v Tangent vector.
  /// @return The norm value.
  double norm(const Eigen::Vector3d& p, const Eigen::Vector3d& v) const {
    return std::sqrt(inner(p, v, v));
  }

  /// @brief Return the injectivity radius \f$ \pi \f$.
  double injectivity_radius() const { return std::numbers::pi; }
};


// ---------------------------------------------------------------------------
// Retraction policies
// ---------------------------------------------------------------------------

/// @brief True exponential and logarithmic maps on \f$ S^2 \f$ (round geometry).
///
/// @details
/// - `retract(p, v)` computes \f$ \exp_p(v) = \cos(\|v\|)\, p + \sin(\|v\|)\, v / \|v\| \f$
/// - `inverse_retract(p, q)` computes \f$ \log_p(q) \f$ via the arc-length formula
struct SphereExponentialMap {
  /// @brief Exponential map \f$ \exp_p(v) \f$ on \f$ S^2 \f$.
  /// @param p Base point on the sphere.
  /// @param v Tangent vector at \f$ p \f$.
  /// @return The resulting point on the sphere.
  EIGEN_STRONG_INLINE
  Eigen::Vector3d retract(const Eigen::Vector3d p, const Eigen::Vector3d v) const {
    const double theta = v.norm();
    if (theta < 1e-10) {
      return p;
    }
    const double inv_theta = 1.0 / theta;
    return std::cos(theta) * p + (std::sin(theta) * inv_theta) * v;
  }

  /// @brief Logarithmic map \f$ \log_p(q) \f$ on \f$ S^2 \f$.
  /// @param p Base point on the sphere.
  /// @param q Target point on the sphere.
  /// @return Tangent vector at \f$ p \f$ such that \f$ \exp_p(v) = q \f$.
  EIGEN_STRONG_INLINE
  Eigen::Vector3d inverse_retract(const Eigen::Vector3d p, const Eigen::Vector3d q) const {
    const double cos_theta = p.dot(q);
    const Eigen::Vector3d v = q - cos_theta * p;
    const double sin_theta = v.norm();
    if (sin_theta < 1e-10) {
      if (cos_theta < 0.0) {
        // TODO: handle cut locus properly — direction is non-unique for antipodal points.
        // Options: random tangent direction, user-provided fallback, or error.
        GEODEX_LOG("SphereExponentialMap: log called at cut locus "
                   "(antipodal points). Returning zero vector.");
      }
      return Eigen::Vector3d::Zero();
    }
    const double theta = std::atan2(sin_theta, cos_theta);
    return (theta / sin_theta) * v;
  }
};

/// @brief First-order projection retraction on \f$ S^2 \f$.
///
/// @details
/// - `retract(p, v)` computes \f$ R_p(v) = (p + v) / \|p + v\| \f$
/// - `inverse_retract(p, q)` projects \f$ q - p \f$ onto \f$ T_p S^2 \f$
struct SphereProjectionRetraction {
  /// @brief Projection retraction: normalize \f$ p + v \f$.
  /// @param p Base point on the sphere.
  /// @param v Tangent vector at \f$ p \f$.
  /// @return The retracted point on the sphere.
  EIGEN_STRONG_INLINE
  Eigen::Vector3d retract(const Eigen::Vector3d p, const Eigen::Vector3d v) const {
    return (p + v).normalized();
  }

  /// @brief Inverse projection retraction.
  /// @param p Base point on the sphere.
  /// @param q Target point on the sphere.
  /// @return Tangent vector at \f$ p \f$ approximating \f$ \log_p(q) \f$.
  EIGEN_STRONG_INLINE
  Eigen::Vector3d inverse_retract(const Eigen::Vector3d p, const Eigen::Vector3d q) const {
    const Eigen::Vector3d d = q - p;
    const Eigen::Vector3d v = d - d.dot(p) * p;
    if (v.norm() < 1e-10 && p.dot(q) < 0.0) {
      // TODO: handle cut locus properly — direction is non-unique for antipodal points.
      GEODEX_LOG("SphereProjectionRetraction: inverse_retract called at cut locus "
                 "(antipodal points). Returning zero vector.");
      return Eigen::Vector3d::Zero();
    }
    return v;
  }
};

// Verify retraction concepts.
static_assert(Retraction<SphereExponentialMap, Eigen::Vector3d, Eigen::Vector3d>);
static_assert(Retraction<SphereProjectionRetraction, Eigen::Vector3d, Eigen::Vector3d>);

// ---------------------------------------------------------------------------
// Sphere manifold
// ---------------------------------------------------------------------------

/// @brief The 2-sphere \f$ S^2 \f$ parameterized by metric and retraction policies.
///
/// @details This class composes a metric policy and a retraction policy to form a
/// complete `RiemannianManifold`. Points are represented as unit vectors in
/// \f$ \mathbb{R}^3 \f$; tangent vectors live in the orthogonal complement of the
/// base point.
///
/// @tparam MetricT Metric policy (default: SphereRoundMetric).
/// @tparam RetractionT Retraction policy (default: SphereExponentialMap).
template <typename MetricT = SphereRoundMetric, typename RetractionT = SphereExponentialMap>
class Sphere {
  MetricT metric_;
  RetractionT retraction_;

 public:
  using Scalar = double;         ///< Scalar type.
  using Point = Eigen::Vector3d;   ///< Point type (unit vector in \f$ \mathbb{R}^3 \f$).
  using Tangent = Eigen::Vector3d; ///< Tangent vector type.

  /// @brief Compile-time flag: is `log` the Riemannian logarithm of the metric?
  ///
  /// @details True only when the default round metric is paired with the true
  /// exponential map — in which case `grad((1/2) d^2)(x) = -log_x(q)` holds
  /// exactly and algorithms like `discrete_geodesic` can take the log direction
  /// without finite differences. Any other metric (e.g., `ConstantSPDMetric`)
  /// or any projection retraction must fall back to finite-difference
  /// natural gradient.
  static constexpr bool has_riemannian_log =
      std::is_same_v<MetricT, SphereRoundMetric> &&
      std::is_same_v<RetractionT, SphereExponentialMap>;

  /// @brief Default constructor (requires default-constructible policies).
  Sphere() { log_construction(); }

  /// @brief Construct with explicit metric and retraction policies.
  /// @param metric The metric policy instance.
  /// @param retraction The retraction policy instance (default-constructed if omitted).
  explicit Sphere(MetricT metric, RetractionT retraction = {})
      : metric_(std::move(metric)), retraction_(std::move(retraction)) {
    log_construction();
  }

  /// @brief Return the intrinsic dimension (always 2).
  int dim() const { return 2; }

  /// @brief Sample a uniformly random point on \f$ S^2 \f$.
  /// @return A unit vector in \f$ \mathbb{R}^3 \f$.
  Point random_point() const {
    // Uniform sampling on S^2 via standard normal.
    thread_local std::mt19937 gen{std::random_device{}()};
    std::normal_distribution<double> dist(0.0, 1.0);
    Point p(dist(gen), dist(gen), dist(gen));
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
  /// @param p Base point.
  /// @param u First tangent vector.
  /// @param v Second tangent vector.
  /// @return \f$ \langle u, v \rangle_p \f$
  Scalar inner(const Point& p, const Tangent& u, const Tangent& v) const {
    return metric_.inner(p, u, v);
  }

  /// @brief Riemannian norm at \f$ p \f$.
  /// @param p Base point.
  /// @param v Tangent vector.
  /// @return \f$ \|v\|_p \f$
  Scalar norm(const Point& p, const Tangent& v) const { return metric_.norm(p, v); }

  /// @brief Batched inner product \f$U^\top M(p)\, V\f$ when the metric provides it.
  ///
  /// @details Forwards to the metric's `inner_matrix` when available. Exists only
  /// for metrics that implement the optimization hook (e.g. `ConstantSPDMetric`,
  /// `KineticEnergyMetric`); the FD fallback in `discrete_geodesic` uses this
  /// path automatically when both sides support it.
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
  /// @param p Base point.
  /// @param v Tangent vector.
  /// @return Resulting point on the manifold.
  Point exp(const Point& p, const Tangent& v) const { return retraction_.retract(p, v); }

  /// @brief Logarithmic map (or inverse retraction) \f$ \log_p(q) \f$.
  /// @param p Base point.
  /// @param q Target point.
  /// @return Tangent vector at \f$ p \f$ pointing toward \f$ q \f$.
  Tangent log(const Point& p, const Point& q) const { return retraction_.inverse_retract(p, q); }

  /// @}

  /// @name Derived operations
  /// @{

  /// @brief Geodesic distance \f$ d(p, q) \f$ via the midpoint approximation.
  /// @param p First point.
  /// @param q Second point.
  /// @return The approximate geodesic distance.
  Scalar distance(const Point& p, const Point& q) const {
    double dot = p.dot(q);
    if (dot < -1.0 + 1e-10) {
      // Antipodal: geodesic distance = π for round metric.
      // For non-round metrics, this is still the best we can do
      // since log is undefined at the cut locus.
      return std::numbers::pi;
    }
    return distance_midpoint(*this, p, q);
  }

  /// @brief Injectivity radius — only available when the metric provides it.
  /// @return The injectivity radius of the manifold.
  Scalar injectivity_radius() const
    requires requires { metric_.injectivity_radius(); }
  {
    return metric_.injectivity_radius();
  }

  /// @brief Geodesic interpolation between \f$ p \f$ and \f$ q \f$ at parameter \f$ t \f$.
  /// @param p Start point.
  /// @param q End point.
  /// @param t Interpolation parameter in \f$ [0, 1] \f$.
  /// @return The interpolated point.
  Point geodesic(const Point& p, const Point& q, Scalar t) const { return exp(p, t * log(p, q)); }

  /// @}

 private:
  void log_construction() const {
    GEODEX_LOG("Sphere<" << detail::type_name<MetricT>() << ", "
                         << detail::type_name<RetractionT>() << "> created (dim=" << dim() << ")");
  }
};

// Verify the composed types satisfy RiemannianManifold.
static_assert(RiemannianManifold<Sphere<>>);
static_assert(RiemannianManifold<Sphere<ConstantSPDMetric<3>>>);
static_assert(RiemannianManifold<Sphere<SphereRoundMetric, SphereProjectionRetraction>>);

}  // namespace geodex
