/// @file dynamic_manifold.hpp
/// @brief Type-erased manifold and metric for bridging Python-composed types to C++ algorithms.

#pragma once

#include <cstdint>

#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

#include <Eigen/Core>

#include "geodex/algorithm/distance.hpp"
#include "geodex/core/concepts.hpp"
#include "geodex/core/sampler.hpp"

#include "sizes.hpp"

namespace geodex::python {

/// @brief Type-erased metric storing inner/norm as std::function.
///
/// ConfigurationSpace uses it to compose Python-defined metrics with C++ manifolds.
struct DynamicMetric {
  using InnerFn =
      std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&)>;
  using NormFn = std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)>;

  InnerFn inner_fn;
  NormFn norm_fn;

  double inner(const Eigen::VectorXd& p, const Eigen::VectorXd& u, const Eigen::VectorXd& v) const {
    return inner_fn(p, u, v);
  }

  double norm(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const { return norm_fn(p, v); }

  double injectivity_radius() const { return std::numeric_limits<double>::infinity(); }
};

/// @brief Type-erased Riemannian manifold satisfying the `RiemannianManifold` concept.
///
/// Stores all manifold operations as std::function members. C++ template algorithms accept
/// it for Python-composed manifolds, such as a base topology with a custom metric.
class DynamicManifold {
 public:
  using Scalar = double;
  using Point = Eigen::VectorXd;
  using Tangent = Eigen::VectorXd;
  using SamplerType = geodex::DynamicSampler;  ///< type-erased sampler

  using DimFn = std::function<int()>;
  using RandomPointFn = std::function<Point()>;
  using ExpFn = std::function<Point(const Point&, const Tangent&)>;
  using LogFn = std::function<Tangent(const Point&, const Point&)>;
  using InnerFn = std::function<Scalar(const Point&, const Tangent&, const Tangent&)>;
  using NormFn = std::function<Scalar(const Point&, const Tangent&)>;
  using ProjectFn = std::function<Tangent(const Point&, const Tangent&)>;
  using UnitCubeDimFn = std::function<int()>;
  using FromUnitCubeFn = std::function<Point(Eigen::Ref<const Eigen::VectorXd>)>;
  using SamplerFn = std::function<SamplerType()>;
  using GeodesicFn = std::function<Point(const Point&, const Point&, Scalar)>;
  using DistanceFn = std::function<Scalar(const Point&, const Point&)>;

  DynamicManifold() = default;

  DynamicManifold(DimFn dim_fn, RandomPointFn random_point_fn, ExpFn exp_fn, LogFn log_fn,
                  InnerFn inner_fn, NormFn norm_fn, ProjectFn project_fn = nullptr,
                  UnitCubeDimFn unit_cube_dim_fn = nullptr,
                  FromUnitCubeFn from_unit_cube_fn = nullptr)
      : dim_fn_(std::move(dim_fn)),
        random_point_fn_(std::move(random_point_fn)),
        exp_fn_(std::move(exp_fn)),
        log_fn_(std::move(log_fn)),
        inner_fn_(std::move(inner_fn)),
        norm_fn_(std::move(norm_fn)),
        project_fn_(std::move(project_fn)),
        unit_cube_dim_fn_(std::move(unit_cube_dim_fn)),
        from_unit_cube_fn_(std::move(from_unit_cube_fn)) {}

  int dim() const { return dim_fn_(); }
  Point random_point() const { return random_point_fn_(); }

  Point exp(const Point& p, const Tangent& v) const {
    check_point(p, "exp", "p");
    check_tangent(v, "exp", "v");
    return exp_fn_(p, v);
  }

  Tangent log(const Point& p, const Point& q) const {
    check_point(p, "log", "p");
    check_point(q, "log", "q");
    return log_fn_(p, q);
  }

  Scalar inner(const Point& p, const Tangent& u, const Tangent& v) const {
    check_point(p, "inner", "p");
    check_tangent(u, "inner", "u");
    check_tangent(v, "inner", "v");
    return inner_fn_(p, u, v);
  }

  Scalar norm(const Point& p, const Tangent& v) const {
    check_point(p, "norm", "p");
    check_tangent(v, "norm", "v");
    return norm_fn_(p, v);
  }

  /// @brief The wrapped manifold's distance when one was supplied, otherwise
  /// `distance_midpoint`.
  Scalar distance(const Point& p, const Point& q) const {
    if (!distance_fn_) return distance_midpoint(*this, p, q);
    check_point(p, "distance", "p");
    check_point(q, "distance", "q");
    return distance_fn_(p, q);
  }

  /// @brief The wrapped manifold's geodesic when one was supplied, otherwise
  /// `exp(p, t log(p, q))`.
  Point geodesic(const Point& p, const Point& q, Scalar t) const {
    if (!geodesic_fn_) return exp(p, t * log(p, q));
    check_point(p, "geodesic", "p");
    check_point(q, "geodesic", "q");
    return geodesic_fn_(p, q, t);
  }

  /// @brief Supply the wrapped manifold's own geodesic, which `geodesic` then calls.
  void set_geodesic_fn(GeodesicFn fn) { geodesic_fn_ = std::move(fn); }

  /// @brief Supply the wrapped manifold's own distance, which `distance` then calls.
  void set_distance_fn(DistanceFn fn) { distance_fn_ = std::move(fn); }

  Tangent project(const Point& p, const Tangent& v) const {
    if (!project_fn_) {
      throw std::runtime_error("project() not available for this manifold composition");
    }
    check_point(p, "project", "p");
    check_tangent(v, "project", "v");
    return project_fn_(p, v);
  }

  bool has_project() const { return static_cast<bool>(project_fn_); }

  /// @brief Number of unit-cube coordinates the manifold's map consumes.
  int unit_cube_dim() const { return unit_cube_dim_fn_(); }

  /// @brief Map a unit-cube vector to a point via the manifold's own map.
  Point from_unit_cube(Eigen::Ref<const Eigen::VectorXd> u) const { return from_unit_cube_fn_(u); }

  /// @brief Whether this manifold exposes a from_unit_cube map.
  bool has_from_unit_cube() const { return static_cast<bool>(from_unit_cube_fn_); }

  /// @brief Record whether `log` is the Riemannian logarithm of the metric, as the
  /// wrapped manifold reports through `is_riemannian_log`. Planning then resolves
  /// automatic interpolation the way it does for that manifold.
  void set_riemannian_log(const bool riemannian_log) { riemannian_log_ = riemannian_log; }

  /// @brief Whether `log` is the Riemannian logarithm of the metric.
  bool has_riemannian_log_runtime() const { return riemannian_log_; }

  /// @brief Supply the sampler behind random_point(), which sampler() copies.
  void set_sampler_fn(SamplerFn fn) { sampler_fn_ = std::move(fn); }

  /// @brief A copy of the sampler behind random_point(), or a fresh default sampler when
  /// the caller did not supply one. Planning samples through copies of it.
  SamplerType sampler() const { return sampler_fn_ ? sampler_fn_() : SamplerType{}; }

  /// @brief Supply how to reseed the sampler behind random_point().
  void set_seed_fn(std::function<void(std::uint64_t)> fn) { seed_fn_ = std::move(fn); }

  /// @brief Supply how to replace the sampler behind random_point().
  void set_replace_sampler_fn(std::function<void(SamplerType)> fn) {
    replace_sampler_fn_ = std::move(fn);
  }

  /// @brief Reseed the sampler behind random_point().
  /// @throws std::logic_error when the caller did not supply a reseed function.
  void seed(const std::uint64_t s) {
    if (!seed_fn_) throw std::logic_error("this manifold's sampler cannot be reseeded");
    seed_fn_(s);
  }

  /// @brief Replace the sampler behind random_point().
  /// @throws std::logic_error when the caller did not supply a replace function.
  void set_sampler(SamplerType sampler) {
    if (!replace_sampler_fn_) throw std::logic_error("this manifold's sampler cannot be replaced");
    replace_sampler_fn_(std::move(sampler));
  }

  /// @brief Set explicit (lower, upper) sampling bounds on the ambient coordinates.
  void set_bounds(Eigen::VectorXd lo, Eigen::VectorXd hi) {
    bounds_ = std::make_pair(std::move(lo), std::move(hi));
  }

  /// @brief Whether explicit sampling bounds were set.
  bool has_bounds() const { return bounds_.has_value(); }

  /// @brief The explicit (lower, upper) sampling bounds.
  std::pair<Eigen::VectorXd, Eigen::VectorXd> bounds() const { return *bounds_; }

  /// @brief Declare the sizes of points and tangent vectors. Every later call checks
  /// its arguments against them and throws std::invalid_argument on a mismatch. A
  /// size of 0 disables its check.
  void set_sizes(const Eigen::Index point, const Eigen::Index tangent) {
    point_size_ = point;
    tangent_size_ = tangent;
  }

  /// @brief Declare the sizes from one point of the cube map and its zero tangent, when the
  /// manifold has a cube map and does not have sizes yet. It does not touch the sampler.
  void probe_sizes() {
    if (point_size_ > 0 || !has_from_unit_cube()) return;
    const Point p = from_unit_cube(Eigen::VectorXd::Constant(unit_cube_dim(), 0.5));
    set_sizes(p.size(), log_fn_(p, p).size());
  }

  /// @brief Declared size of a point, 0 when unknown.
  Eigen::Index point_size() const { return point_size_; }

  /// @brief Declared size of a tangent vector, 0 when unknown.
  Eigen::Index tangent_size() const { return tangent_size_; }

 private:
  void check_point(const Point& p, const char* where, const char* name) const {
    if (point_size_ > 0) require_size(p, point_size_, where, name);
  }

  void check_tangent(const Tangent& v, const char* where, const char* name) const {
    if (tangent_size_ > 0) require_size(v, tangent_size_, where, name);
  }

  DimFn dim_fn_;
  RandomPointFn random_point_fn_;
  ExpFn exp_fn_;
  LogFn log_fn_;
  InnerFn inner_fn_;
  NormFn norm_fn_;
  ProjectFn project_fn_;
  UnitCubeDimFn unit_cube_dim_fn_;
  FromUnitCubeFn from_unit_cube_fn_;
  SamplerFn sampler_fn_;
  GeodesicFn geodesic_fn_;
  DistanceFn distance_fn_;
  std::function<void(std::uint64_t)> seed_fn_;
  std::function<void(SamplerType)> replace_sampler_fn_;
  bool riemannian_log_ = false;
  std::optional<std::pair<Eigen::VectorXd, Eigen::VectorXd>> bounds_;
  Eigen::Index point_size_ = 0;
  Eigen::Index tangent_size_ = 0;
};

// Verify DynamicManifold satisfies the RiemannianManifold concept.
static_assert(RiemannianManifold<DynamicManifold>);
static_assert(HasSampler<DynamicManifold>);

}  // namespace geodex::python
