/// @file py_se2.hpp
/// @brief Python wrapper for geodex::SE2 with variant-based retraction dispatch.

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <variant>

#include <Eigen/Core>

#include "geodex/algorithm/precompute_matrix_lower_bound.hpp"
#include "geodex/manifold/se2.hpp"

#include "dynamic_manifold.hpp"
#include "sampler_kind.hpp"

namespace geodex::python {

class PySE2 {
 public:
  using Left = SE2<SE2LeftInvariantMetric, SE2LeftExponentialMap, DynamicSampler>;
  using Right = SE2<SE2LeftInvariantMetric, SE2RightExponentialMap, DynamicSampler>;
  using Euler = SE2<SE2LeftInvariantMetric, SE2EulerRetraction, DynamicSampler>;
  using V = std::variant<Left, Right, Euler>;

  PySE2(double wx = 1.0, double wy = 1.0, double wtheta = 1.0,
        const std::string& retraction = "exponential", const std::string& frame = "body",
        double x_lo = 0.0, double x_hi = 10.0, double y_lo = 0.0, double y_hi = 10.0,
        const std::string& sampler = "scrambled")
      : retraction_name_(retraction), frame_name_(frame) {
    SE2LeftInvariantMetric metric(wx, wy, wtheta);
    const Eigen::Vector3d lo(x_lo, y_lo, -std::numbers::pi);
    const Eigen::Vector3d hi(x_hi, y_hi, std::numbers::pi);
    if (frame != "body" && frame != "world") {
      throw std::invalid_argument("Unknown frame: '" + frame + "'. Options: 'body', 'world'");
    }
    if (retraction == "exponential") {
      if (frame == "world") {
        impl_ = std::make_shared<V>(std::in_place_type<Right>,
            metric, SE2RightExponentialMap{}, lo, hi);
      } else {
        impl_ = std::make_shared<V>(std::in_place_type<Left>,
            metric, SE2LeftExponentialMap{}, lo, hi);
      }
    } else if (retraction == "euler") {
      // The Euler retraction adds world-frame rates and ignores `frame`.
      impl_ = std::make_shared<V>(std::in_place_type<Euler>, metric, SE2EulerRetraction{}, lo, hi);
    } else {
      throw std::invalid_argument("Unknown retraction: '" + retraction +
                                  "'. Options: 'exponential', 'euler'");
    }
    if (sampler != "scrambled") set_sampler(sampler);
  }

  int dim() const {
    return std::visit([](const auto& m) { return m.dim(); }, *impl_);
  }

  Eigen::Vector3d random_point() const {
    return std::visit([](const auto& m) { return m.random_point(); }, *impl_);
  }

  /// @brief Reseed the sampler for reproducible sampling.
  void seed(std::uint64_t s) { std::visit([&](auto& m) { m.seed(s); }, *impl_); }

  /// @brief Switch the sampler kind at runtime.
  void set_sampler(const std::string& kind) {
    std::visit([&](auto& m) { m.set_sampler(make_sampler(kind)); }, *impl_);
  }

  double inner(const Eigen::Vector3d& p, const Eigen::Vector3d& u, const Eigen::Vector3d& v) const {
    return std::visit([&](const auto& m) { return m.inner(p, u, v); }, *impl_);
  }

  double norm(const Eigen::Vector3d& p, const Eigen::Vector3d& v) const {
    return std::visit([&](const auto& m) { return m.norm(p, v); }, *impl_);
  }

  Eigen::Vector3d exp(const Eigen::Vector3d& p, const Eigen::Vector3d& v) const {
    return std::visit([&](const auto& m) { return m.exp(p, v); }, *impl_);
  }

  Eigen::Vector3d log(const Eigen::Vector3d& p, const Eigen::Vector3d& q) const {
    return std::visit([&](const auto& m) { return m.log(p, q); }, *impl_);
  }

  double distance(const Eigen::Vector3d& p, const Eigen::Vector3d& q) const {
    return std::visit([&](const auto& m) { return m.distance(p, q); }, *impl_);
  }

  Eigen::Vector3d geodesic(const Eigen::Vector3d& p, const Eigen::Vector3d& q, double t) const {
    return std::visit([&](const auto& m) { return m.geodesic(p, q, t); }, *impl_);
  }

  /// @brief Deck-group generators of the coordinate axes, \f$ (0, 0, 2\pi) \f$.
  Eigen::Vector3d periods() const {
    return std::visit([](const auto& m) { return m.periods(); }, *impl_);
  }

  /// @brief The metric on coordinate velocities, \f$ J^\top M J \f$.
  Eigen::Matrix3d coordinate_metric(const Eigen::Vector3d& q) const {
    return std::visit([&](const auto& m) { return m.coordinate_metric(q); }, *impl_);
  }

  /// @brief Certify the periods-aware Loewner lower bound for this metric.
  ///
  /// @details Runs constraint generation over the coordinate metric. The result is
  /// admissible for coordinate chords and wraps at the theta cut.
  heuristics::MatrixLowerBound<Eigen::Dynamic> matrix_lower_bound() const {
    return std::visit(
        [](const auto& m) { return algorithm::precompute_matrix_lower_bound(m).heuristic(); },
        *impl_);
  }

  DynamicManifold to_dynamic_manifold() const {
    auto shared = impl_;
    DynamicManifold dm{
        [shared]() { return std::visit([](const auto& m) { return m.dim(); }, *shared); },
        [shared]() -> Eigen::VectorXd {
          return std::visit([](const auto& m) -> Eigen::VectorXd { return m.random_point(); },
                            *shared);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& v) -> Eigen::VectorXd {
          Eigen::Vector3d p3(p), v3(v);
          return std::visit([&](const auto& m) -> Eigen::VectorXd { return m.exp(p3, v3); },
                            *shared);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& q) -> Eigen::VectorXd {
          Eigen::Vector3d p3(p), q3(q);
          return std::visit([&](const auto& m) -> Eigen::VectorXd { return m.log(p3, q3); },
                            *shared);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& u,
                 const Eigen::VectorXd& v) -> double {
          Eigen::Vector3d p3(p), u3(u), v3(v);
          return std::visit([&](const auto& m) { return m.inner(p3, u3, v3); }, *shared);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& v) -> double {
          Eigen::Vector3d p3(p), v3(v);
          return std::visit([&](const auto& m) { return m.norm(p3, v3); }, *shared);
        },
        // SE2 is parameterized as (x, y, theta) in R^3, and its tangent space is R^3.
        [](const Eigen::VectorXd& /*p*/, const Eigen::VectorXd& v) { return v; },
        [shared]() { return std::visit([](const auto& m) { return m.unit_cube_dim(); }, *shared); },
        [shared](Eigen::Ref<const Eigen::VectorXd> u) -> Eigen::VectorXd {
          return std::visit([&](const auto& m) -> Eigen::VectorXd { return m.from_unit_cube(u); },
                            *shared);
        }};
    dm.set_sampler_fn([shared]() {
      return std::visit([](const auto& m) { return DynamicSampler(m.sampler()); }, *shared);
    });
    dm.set_riemannian_log(
        std::visit([](const auto& m) { return geodex::is_riemannian_log(m); }, *shared));
    return dm;
  }

  /// @brief Factory for car-like SE2 with turning radius constraint.
  static PySE2 car_like(const double turning_radius, const double lateral_penalty = 100.0,
                        const std::string& retraction = "exponential",
                        const std::string& frame = "body", const double x_lo = 0.0,
                        const double x_hi = 10.0, const double y_lo = 0.0, const double y_hi = 10.0,
                        const std::string& sampler = "scrambled") {
    const auto metric = SE2LeftInvariantMetric::car_like(turning_radius, lateral_penalty);
    return PySE2(metric.weights()[0], metric.weights()[1], metric.weights()[2], retraction, frame,
                 x_lo, x_hi, y_lo, y_hi, sampler);
  }

  std::string repr() const {
    return "SE2(retraction='" + retraction_name_ + "', frame='" + frame_name_ + "')";
  }

  /// @brief Whether log returns a body twist, forward component first. Only the
  /// body-frame exponential map does.
  bool log_is_body_twist() const {
    return retraction_name_ == "exponential" && frame_name_ == "body";
  }

  /// @brief The wrapped manifold, shared with every copy of this wrapper.
  const std::shared_ptr<V>& variant() const { return impl_; }

 private:
  /// The wrapped manifold, shared with every DynamicManifold this wrapper hands out. A plan
  /// advances this manifold's own sampler.
  std::shared_ptr<V> impl_;
  std::string retraction_name_;
  std::string frame_name_;
};

}  // namespace geodex::python
