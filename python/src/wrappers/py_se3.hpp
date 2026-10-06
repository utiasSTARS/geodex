/// @file py_se3.hpp
/// @brief Python wrapper for geodex::SE3 with variant-based frame (retraction) dispatch.

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <variant>

#include <Eigen/Core>

#include "geodex/manifold/se3.hpp"

#include "dynamic_manifold.hpp"
#include "sampler_kind.hpp"

namespace geodex::python {

class PySE3 {
 public:
  /// @brief Pose \f$ [t_x, t_y, t_z,\; q_x, q_y, q_z, q_w] \in \mathbb{R}^7 \f$.
  using Point = Eigen::Matrix<double, 7, 1>;
  /// @brief Body/spatial twist \f$ [v;\,\omega] \in \mathbb{R}^6 \f$.
  using Tangent = Eigen::Matrix<double, 6, 1>;

  /// @brief Left/right group-exponential variants (body vs. world frame).
  using Left = SE3<SE3InvariantMetric, SE3LeftExponentialMap, DynamicSampler>;
  using Right = SE3<SE3InvariantMetric, SE3RightExponentialMap, DynamicSampler>;
  using V = std::variant<Left, Right>;

  PySE3(const std::string& frame = "body", double w_trans = 1.0, double w_rot = 1.0,
        double x_lo = 0.0, double x_hi = 10.0, double y_lo = 0.0, double y_hi = 10.0,
        double z_lo = 0.0, double z_hi = 10.0, const std::string& sampler = "scrambled")
      : frame_name_(frame) {
    SE3InvariantMetric metric(w_trans, w_rot);
    const Eigen::Vector3d lo(x_lo, y_lo, z_lo);
    const Eigen::Vector3d hi(x_hi, y_hi, z_hi);
    if (frame == "body") {
      impl_ = std::make_shared<V>(std::in_place_type<Left>,
          metric, SE3LeftExponentialMap{}, lo, hi);
    } else if (frame == "world") {
      impl_ = std::make_shared<V>(std::in_place_type<Right>,
          metric, SE3RightExponentialMap{}, lo, hi);
    } else {
      throw std::invalid_argument("Unknown frame: '" + frame + "'. Options: 'body', 'world'");
    }
    if (sampler != "scrambled") set_sampler(sampler);
  }

  int dim() const {
    return std::visit([](const auto& m) { return m.dim(); }, *impl_);
  }

  Point random_point() const {
    return std::visit([](const auto& m) { return m.random_point(); }, *impl_);
  }

  /// @brief Reseed the sampler for reproducible sampling.
  void seed(std::uint64_t s) { std::visit([&](auto& m) { m.seed(s); }, *impl_); }

  /// @brief Switch the sampler kind at runtime.
  void set_sampler(const std::string& kind) {
    std::visit([&](auto& m) { m.set_sampler(make_sampler(kind)); }, *impl_);
  }

  double inner(const Point& p, const Tangent& u, const Tangent& v) const {
    return std::visit([&](const auto& m) { return m.inner(p, u, v); }, *impl_);
  }

  double norm(const Point& p, const Tangent& v) const {
    return std::visit([&](const auto& m) { return m.norm(p, v); }, *impl_);
  }

  Point exp(const Point& p, const Tangent& v) const {
    return std::visit([&](const auto& m) { return m.exp(p, v); }, *impl_);
  }

  Tangent log(const Point& p, const Point& q) const {
    return std::visit([&](const auto& m) { return m.log(p, q); }, *impl_);
  }

  double distance(const Point& p, const Point& q) const {
    return std::visit([&](const auto& m) { return m.distance(p, q); }, *impl_);
  }

  Point geodesic(const Point& p, const Point& q, double t) const {
    return std::visit([&](const auto& m) { return m.geodesic(p, q, t); }, *impl_);
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
          Eigen::Matrix<double, 7, 1> p7(p);
          Eigen::Matrix<double, 6, 1> v6(v);
          return std::visit([&](const auto& m) -> Eigen::VectorXd { return m.exp(p7, v6); },
                            *shared);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& q) -> Eigen::VectorXd {
          Eigen::Matrix<double, 7, 1> p7(p), q7(q);
          return std::visit([&](const auto& m) -> Eigen::VectorXd { return m.log(p7, q7); },
                            *shared);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& u,
                 const Eigen::VectorXd& v) -> double {
          Eigen::Matrix<double, 7, 1> p7(p);
          Eigen::Matrix<double, 6, 1> u6(u), v6(v);
          return std::visit([&](const auto& m) { return m.inner(p7, u6, v6); }, *shared);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& v) -> double {
          Eigen::Matrix<double, 7, 1> p7(p);
          Eigen::Matrix<double, 6, 1> v6(v);
          return std::visit([&](const auto& m) { return m.norm(p7, v6); }, *shared);
        },
        // SE(3)'s tangent space is the Lie algebra se(3) ≅ R^6, and project is the identity.
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

  std::string repr() const { return "SE3(frame='" + frame_name_ + "')"; }

 private:
  /// The wrapped manifold, shared with every DynamicManifold this wrapper hands out. A plan
  /// advances this manifold's own sampler.
  std::shared_ptr<V> impl_;
  std::string frame_name_;
};

}  // namespace geodex::python
