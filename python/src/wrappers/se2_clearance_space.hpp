/// @file se2_clearance_space.hpp
/// @brief A ConfigurationSpace over SE2 with the clearance metric of a bound C++ SDF, with
/// fixed-size points and without std::function.

#pragma once

#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

#include <Eigen/Core>

#include "geodex/algorithm/distance.hpp"
#include "geodex/metrics/clearance.hpp"
#include "geodex/metrics/se2_left_invariant.hpp"

#include "native_sdf.hpp"
#include "py_se2.hpp"

namespace geodex::python {

/// @brief The parts of `ConfigurationSpace(SE2, ClearanceMetric(base, sdf, kappa, beta))`
/// whose base is an SE2LeftInvariantMetric and whose sdf is a bound geodex SDF.
struct SE2ClearanceParts {
  PySE2 se2;                    ///< the base manifold, which shares its sampler with Python
  SE2LeftInvariantMetric base;  ///< the base metric of the clearance metric
  NativeSdf sdf;                ///< the SDF of the clearance metric
  double kappa;                 ///< obstacle repulsion strength
  double beta;                  ///< falloff rate
};

/// @brief The type-erased space of `ConfigurationSpace(SE2, ClearanceMetric(...))` with
/// points of fixed size 3.
///
/// @details Every operation computes what the type-erased space computes. exp, log and
/// sampling come from the SE2, inner and norm from SDFConformalMetric, distance from
/// `distance_midpoint` and geodesic from `exp(p, t log(p, q))`. The class has the optional
/// members of the type-erased space and no others, and plan() takes the same branches on
/// both. It does not have the `metric_at`, `periods` or `coordinate_metric` of the C++
/// ConfigurationSpace. The SE2 is the one the Python object holds, and an unseeded plan
/// advances its sampler.
///
/// @tparam SE2T The SE2 alternative of `PySE2::V`.
template <typename SE2T>
class SE2ClearanceSpace {
 public:
  using Scalar = double;
  using Point = Eigen::Vector3d;
  using Tangent = Eigen::Vector3d;
  using SamplerType = typename SE2T::SamplerType;
  using Metric = SDFConformalMetric<SE2LeftInvariantMetric, NativeSdf>;

  /// @param se2 The SE2 of the Python object. It holds an `SE2T`.
  /// @param metric The clearance metric.
  SE2ClearanceSpace(std::shared_ptr<PySE2::V> se2, Metric metric)
      : holder_(std::move(se2)), se2_(&std::get<SE2T>(*holder_)), metric_(std::move(metric)) {}

  int dim() const { return se2_->dim(); }
  Point random_point() const { return se2_->random_point(); }
  Point exp(const Point& p, const Tangent& v) const { return se2_->exp(p, v); }
  Tangent log(const Point& p, const Point& q) const { return se2_->log(p, q); }

  /// @brief The tangent space of SE2 at every point is R^3.
  Tangent project(const Point& /*p*/, const Tangent& v) const { return v; }
  bool has_project() const { return true; }

  Scalar inner(const Point& p, const Tangent& u, const Tangent& v) const {
    return metric_.inner(p, u, v);
  }
  Scalar norm(const Point& p, const Tangent& v) const { return metric_.norm(p, v); }
  Scalar distance(const Point& p, const Point& q) const { return distance_midpoint(*this, p, q); }
  Point geodesic(const Point& p, const Point& q, const Scalar t) const {
    return exp(p, t * log(p, q));
  }

  int unit_cube_dim() const { return se2_->unit_cube_dim(); }
  Point from_unit_cube(Eigen::Ref<const Eigen::VectorXd> u) const {
    return se2_->from_unit_cube(u);
  }
  bool has_from_unit_cube() const { return true; }

  /// @brief A copy of the SE2's sampler.
  SamplerType sampler() const { return se2_->sampler(); }

  /// @brief The base's log is not the Riemannian logarithm of the clearance metric.
  bool has_riemannian_log_runtime() const { return false; }

  /// @brief The space does not declare sampling bounds.
  bool has_bounds() const { return false; }

  /// @brief Not available. `has_bounds()` is false.
  std::pair<Eigen::VectorXd, Eigen::VectorXd> bounds() const {
    throw std::logic_error("SE2ClearanceSpace does not declare bounds");
  }

  /// @brief The clearance metric.
  const Metric& metric() const { return metric_; }

 private:
  std::shared_ptr<PySE2::V> holder_;
  const SE2T* se2_;
  Metric metric_;
};

/// @brief Call `f` with the typed space of `parts` and return its result.
template <typename F>
decltype(auto) visit_se2_clearance(const SE2ClearanceParts& parts, F&& f) {
  const std::shared_ptr<PySE2::V>& se2 = parts.se2.variant();
  const SDFConformalMetric<SE2LeftInvariantMetric, NativeSdf> metric(parts.base, parts.sdf,
                                                                     parts.kappa, parts.beta);
  return std::visit(
      [&](const auto& alternative) -> decltype(auto) {
        using SE2T = std::remove_cvref_t<decltype(alternative)>;
        return f(SE2ClearanceSpace<SE2T>(se2, metric));
      },
      *se2);
}

}  // namespace geodex::python
