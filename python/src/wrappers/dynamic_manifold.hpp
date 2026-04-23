/// @file dynamic_manifold.hpp
/// @brief Type-erased manifold and metric for bridging Python-composed types to C++ algorithms.

#pragma once

#include <functional>
#include <limits>

#include <Eigen/Core>

#include "geodex/algorithm/distance.hpp"
#include "geodex/core/concepts.hpp"

namespace geodex::python {

/// @brief Type-erased metric storing inner/norm as std::function.
///
/// Used to compose Python-defined metrics with C++ manifolds in ConfigurationSpace.
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
/// Stores all manifold operations as std::function members, enabling Python-composed
/// manifolds (e.g., base topology + custom metric) to be passed to C++ template algorithms.
class DynamicManifold {
 public:
  using Scalar = double;
  using Point = Eigen::VectorXd;
  using Tangent = Eigen::VectorXd;

  using DimFn = std::function<int()>;
  using RandomPointFn = std::function<Point()>;
  using ExpFn = std::function<Point(const Point&, const Tangent&)>;
  using LogFn = std::function<Tangent(const Point&, const Point&)>;
  using InnerFn = std::function<Scalar(const Point&, const Tangent&, const Tangent&)>;
  using NormFn = std::function<Scalar(const Point&, const Tangent&)>;
  using ProjectFn = std::function<Tangent(const Point&, const Tangent&)>;

  DynamicManifold() = default;

  DynamicManifold(DimFn dim_fn, RandomPointFn random_point_fn, ExpFn exp_fn, LogFn log_fn,
                  InnerFn inner_fn, NormFn norm_fn, ProjectFn project_fn = nullptr)
      : dim_fn_(std::move(dim_fn)),
        random_point_fn_(std::move(random_point_fn)),
        exp_fn_(std::move(exp_fn)),
        log_fn_(std::move(log_fn)),
        inner_fn_(std::move(inner_fn)),
        norm_fn_(std::move(norm_fn)),
        project_fn_(std::move(project_fn)) {}

  int dim() const { return dim_fn_(); }
  Point random_point() const { return random_point_fn_(); }
  Point exp(const Point& p, const Tangent& v) const { return exp_fn_(p, v); }
  Tangent log(const Point& p, const Point& q) const { return log_fn_(p, q); }

  Scalar inner(const Point& p, const Tangent& u, const Tangent& v) const {
    return inner_fn_(p, u, v);
  }

  Scalar norm(const Point& p, const Tangent& v) const { return norm_fn_(p, v); }

  Scalar distance(const Point& p, const Point& q) const { return distance_midpoint(*this, p, q); }

  Point geodesic(const Point& p, const Point& q, Scalar t) const { return exp(p, t * log(p, q)); }

  Tangent project(const Point& p, const Tangent& v) const {
    if (project_fn_) return project_fn_(p, v);
    throw std::runtime_error("project() not available for this manifold composition");
  }

  bool has_project() const { return static_cast<bool>(project_fn_); }

 private:
  DimFn dim_fn_;
  RandomPointFn random_point_fn_;
  ExpFn exp_fn_;
  LogFn log_fn_;
  InnerFn inner_fn_;
  NormFn norm_fn_;
  ProjectFn project_fn_;
};

// Verify DynamicManifold satisfies the RiemannianManifold concept.
static_assert(RiemannianManifold<DynamicManifold>);

}  // namespace geodex::python
