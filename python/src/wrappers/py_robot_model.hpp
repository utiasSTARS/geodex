/// @file py_robot_model.hpp
/// @brief Python wrapper exposing a built-in robot as a configuration space.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <Eigen/Core>

#include "geodex/heuristics/matrix_lower_bound.hpp"
#include "geodex/heuristics/product_lower_bound.hpp"
#include "geodex/manifold/product.hpp"
#include "geodex/manifold/se2.hpp"
#include "geodex/metrics/se2_left_invariant.hpp"
#include "geodex/robots/joint_space.hpp"
#include "geodex/robots/mass_lower_bound.hpp"
#include "geodex/robots/mass_matrix.hpp"

#include "dynamic_manifold.hpp"
#include "sampler_kind.hpp"

namespace geodex::python {

/// @brief Metric and base-region choices for a robot model.
struct RobotModelOptions {
  robots::ArmMetric metric = robots::ArmMetric::KineticEnergy;  ///< metric on the arm joints
  std::optional<SE2LeftInvariantMetric> base;  ///< base metric, empty to match the drive
  Eigen::Vector2d xy_lo{-5.0, -5.0};           ///< lower corner of the base region
  Eigen::Vector2d xy_hi{5.0, 5.0};             ///< upper corner of the base region
};

/// @brief A built-in robot as a configuration space.
///
/// A fixed-base robot is R^dof with joint-limit bounds. A robot with a planar base is
/// SE(2) x R^n with the base pose (x, y, theta) first. The arm metric is the CRBA mass
/// matrix M(q) or the Euclidean metric. The model holds the heuristic of the precomputed
/// Loewner lower bound of its metric, and informed samplers load the bound as a constant.
class PyRobotModel {
 public:
  /// @brief Metric tensor callable, q -> G(q) on coordinate velocities.
  using MassFn = std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>;

  PyRobotModel(std::string name, int dof, Eigen::VectorXd lo, Eigen::VectorXd hi,
               DynamicManifold manifold, heuristics::MatrixLowerBound<Eigen::Dynamic> heuristic,
               MassFn mass_fn = {}, robots::BaseDrive drive = robots::BaseDrive::None,
               int sharp_coordinates = 0)
      : name_(std::move(name)),
        dof_(dof),
        lo_(std::move(lo)),
        hi_(std::move(hi)),
        manifold_(std::move(manifold)),
        heuristic_(std::move(heuristic)),
        mass_fn_(std::move(mass_fn)),
        drive_(drive),
        sharp_coordinates_(sharp_coordinates) {}

  /// @brief Lowercase robot name, such as "panda". It is also the VAMP kernel name.
  const std::string& name() const { return name_; }

  /// @brief Number of configuration coordinates, the base pose included.
  int dof() const { return dof_; }

  /// @brief Configuration-space dimension, equal to dof.
  int dim() const { return dof_; }

  /// @brief Per-coordinate (lower, upper) bounds, the base region first, then joint limits.
  std::pair<Eigen::VectorXd, Eigen::VectorXd> joint_limits() const { return {lo_, hi_}; }

  /// @brief Sample a configuration uniformly within the bounds.
  Eigen::VectorXd random_point() const { return manifold_.random_point(); }

  /// @brief Reseed the sampler behind random_point() and unseeded plans.
  void seed(std::uint64_t s) { manifold_.seed(s); }

  /// @brief Switch the sampler kind ("scrambled", "halton" or "random").
  void set_sampler(const std::string& kind) { manifold_.set_sampler(make_sampler(kind)); }

  /// @brief Type-erased configuration space for extract_manifold. Python does not see it.
  DynamicManifold to_dynamic_manifold() const { return manifold_; }

  /// @brief Per-coordinate periods, 2 pi on the base heading and empty for a fixed base.
  const Eigen::VectorXd& periods() const { return heuristic_.periods(); }

  /// @brief Admissible heuristic of the model's metric, wrapped on the heading.
  heuristics::MatrixLowerBound<Eigen::Dynamic> heuristic() const { return heuristic_; }

  /// @brief Drive of the mobile base, None for a fixed-base robot.
  robots::BaseDrive drive() const { return drive_; }

  /// @brief Leading coordinates whose path may keep corners, the base pose when the base
  /// metric makes sideways motion cost more than driving, and 0 otherwise.
  int sharp_coordinates() const { return sharp_coordinates_; }

  /// @brief Metric tensor on coordinate velocities at @p q.
  ///
  /// For a fixed-base robot under the kinetic-energy metric, this is the joint-space mass
  /// matrix M(q). For a mobile robot, the base block is the SE(2) metric on (x, y, theta)
  /// velocities. Use it to score a path under the metric of its plan.
  /// @throws std::runtime_error when the model does not have a metric tensor.
  Eigen::MatrixXd mass_matrix(const Eigen::VectorXd& q) const {
    if (!mass_fn_) throw std::runtime_error("robot model '" + name_ + "' has no mass matrix");
    if (q.size() != dof_) {
      throw std::invalid_argument("mass_matrix: expected a configuration of size " +
                                  std::to_string(dof_) + ", got " + std::to_string(q.size()));
    }
    return mass_fn_(q);
  }

  /// @brief Whether mass_matrix() is available.
  bool has_mass_matrix() const { return static_cast<bool>(mass_fn_); }

  std::string repr() const {
    return "RobotModel(name=" + name_ + ", dof=" + std::to_string(dof_) + ")";
  }

 private:
  std::string name_;
  int dof_;
  Eigen::VectorXd lo_;
  Eigen::VectorXd hi_;
  DynamicManifold manifold_;
  heuristics::MatrixLowerBound<Eigen::Dynamic> heuristic_;
  MassFn mass_fn_;
  robots::BaseDrive drive_;
  int sharp_coordinates_;
};

namespace detail {

/// @brief Type-erase a geodex manifold whose tangent space is its ambient space.
///
/// Every lambda shares the space. Its metric evaluation buffers are single-threaded, like
/// OMPL's solve.
template <typename SpaceT>
DynamicManifold erase(std::shared_ptr<SpaceT> space, const Eigen::VectorXd& lo,
                      const Eigen::VectorXd& hi) {
  using P = typename SpaceT::Point;
  using T = typename SpaceT::Tangent;
  DynamicManifold m{
      [space] { return space->dim(); },
      [space]() -> Eigen::VectorXd { return space->random_point(); },
      [space](const Eigen::VectorXd& p, const Eigen::VectorXd& v) -> Eigen::VectorXd {
        return space->exp(P(p), T(v));
      },
      [space](const Eigen::VectorXd& p, const Eigen::VectorXd& q) -> Eigen::VectorXd {
        return space->log(P(p), P(q));
      },
      [space](const Eigen::VectorXd& p, const Eigen::VectorXd& u, const Eigen::VectorXd& v) {
        return space->inner(P(p), T(u), T(v));
      },
      [space](const Eigen::VectorXd& p, const Eigen::VectorXd& v) {
        return space->norm(P(p), T(v));
      },
      [](const Eigen::VectorXd& /*p*/, const Eigen::VectorXd& v) { return v; },
      [space] { return space->unit_cube_dim(); },
      [space](Eigen::Ref<const Eigen::VectorXd> u) -> Eigen::VectorXd {
        return space->from_unit_cube(u);
      }};
  m.set_geodesic_fn([space](const Eigen::VectorXd& p, const Eigen::VectorXd& q,
                            const double t) -> Eigen::VectorXd {
    return space->geodesic(P(p), P(q), t);
  });
  m.set_distance_fn([space](const Eigen::VectorXd& p, const Eigen::VectorXd& q) {
    return space->distance(P(p), P(q));
  });
  m.set_sampler_fn([space] { return DynamicSampler(space->sampler()); });
  m.set_seed_fn([space](std::uint64_t s) { space->seed(s); });
  m.set_replace_sampler_fn([space](DynamicSampler s) { space->set_sampler(std::move(s)); });
  m.set_riemannian_log(geodex::is_riemannian_log(*space));
  m.set_bounds(lo, hi);
  return m;
}

/// @brief PyRobotModel of robot @p R over the joint space @p joints.
template <robots::Robot R, typename JointSpace>
PyRobotModel make_robot_model(const RobotModelOptions& options, JointSpace joints) {
  constexpr int n = robots::MassMatrix<R>::Nq;
  const std::string name(robots::name(R));
  if constexpr (robots::has_planar_base<R>) {
    const SE2LeftInvariantMetric base_metric =
        options.base.value_or(robots::base_drive<R> == robots::BaseDrive::Differential
                                  ? SE2LeftInvariantMetric::differential_drive()
                                  : SE2LeftInvariantMetric::holonomic());
    const SE2<> base(base_metric,
                     Eigen::Vector3d(options.xy_lo.x(), options.xy_lo.y(), -std::numbers::pi),
                     Eigen::Vector3d(options.xy_hi.x(), options.xy_hi.y(), std::numbers::pi));
    using Space = ProductManifold<SE2<>, JointSpace>;
    auto space = std::make_shared<Space>(make_product(base, std::move(joints)));
    const Eigen::VectorXd lo = space->lo();
    const Eigen::VectorXd hi = space->hi();
    auto heuristic =
        heuristics::product_lower_bound({{base_metric.coordinate_lower_bound(), base.periods()},
                                         {robots::joint_lower_bound<R>(options.metric)}});
    auto metric = [space](const Eigen::VectorXd& q) -> Eigen::MatrixXd {
      return space->coordinate_metric(q);
    };
    return PyRobotModel{name,
                        3 + n,
                        lo,
                        hi,
                        erase(space, lo, hi),
                        std::move(heuristic),
                        metric,
                        robots::base_drive<R>,
                        robots::sharp_base_coordinates(base_metric)};
  } else {
    auto space = std::make_shared<JointSpace>(std::move(joints));
    const Eigen::VectorXd lo = space->lo();
    const Eigen::VectorXd hi = space->hi();
    auto metric = [space](const Eigen::VectorXd& q) -> Eigen::MatrixXd {
      const Eigen::MatrixXd E = Eigen::MatrixXd::Identity(n, n);
      return space->inner_matrix(typename JointSpace::Point(q), E, E);
    };
    return PyRobotModel{
        name,
        n,
        lo,
        hi,
        erase(space, lo, hi),
        heuristics::MatrixLowerBound<Eigen::Dynamic>(
            robots::joint_lower_bound<R>(options.metric)),
        metric};
  }
}

}  // namespace detail

/// @brief Build a PyRobotModel for a built-in robot.
template <robots::Robot R>
PyRobotModel make_robot_model(const RobotModelOptions& options = {}) {
  static_assert(robots::MassLowerBound<R>::available,
                "robot is missing a precomputed Loewner mass lower bound");
  static_assert(robots::MassMatrix<R>::Nq == robots::MassLowerBound<R>::Nv,
                "robot layer assumes Nq == Nv");
  static_assert(robots::MassLowerBound<R>::converged, "robot ships a non-converged Loewner bound");
  using robots::ArmMetric;
  if (options.metric == ArmMetric::KineticEnergy) {
    return detail::make_robot_model<R>(
        options, robots::joint_space<R, ArmMetric::KineticEnergy, DynamicSampler>());
  }
  return detail::make_robot_model<R>(
      options, robots::joint_space<R, ArmMetric::Euclidean, DynamicSampler>());
}

/// @brief Build a PyRobotModel by its public name, e.g. "panda".
/// @throws std::invalid_argument for an unknown name.
inline PyRobotModel make_robot_model(const std::string& name,
                                     const RobotModelOptions& options = {}) {
  const auto robot = robots::robot_from_name(name);
  if (!robot) {
    std::string known;
    for (const auto r : robots::registered_robots()) {
      known += (known.empty() ? "" : ", ") + std::string(robots::name(r));
    }
    throw std::invalid_argument("Unknown robot '" + name + "'. Supported: " + known + ".");
  }
  return robots::visit(*robot, [&]<robots::Robot R>() { return make_robot_model<R>(options); });
}

}  // namespace geodex::python
