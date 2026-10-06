/// @file
/// @brief Joint spaces of the built-in robots and Loewner lower bounds of their metrics.
///
/// `joint_space<R>()` is the joint coordinates of robot @p R with its joint limits as
/// sampling bounds, under the kinetic-energy metric of its precomputed mass matrix or under the
/// Euclidean metric. `joint_lower_bound<R>()` is the constant matrix below that metric,
/// the bound `heuristics::MatrixLowerBound` takes.
///
/// A robot with a mobile base plans on a product of a base space and its joint space.
/// `heuristics::product_lower_bound` combines the bounds of the two factors.
///
/// @code
/// namespace gr = geodex::robots;
/// constexpr auto R = gr::Robot::Stretch4;
/// const auto metric = geodex::SE2LeftInvariantMetric::holonomic();
/// const geodex::SE2<> base(metric, Eigen::Vector3d(-3, -3, -std::numbers::pi),
///                          Eigen::Vector3d(3, 3, std::numbers::pi));
/// const auto space = geodex::make_product(base, gr::joint_space<R>());
/// const auto heuristic = geodex::heuristics::product_lower_bound(
///     {{metric.coordinate_lower_bound(), base.periods()},
///      {gr::joint_lower_bound<R>(gr::ArmMetric::KineticEnergy)}});
/// @endcode

#pragma once

#include <utility>

#include <Eigen/Core>

#include "geodex/core/sampler.hpp"
#include "geodex/manifold/configuration_space.hpp"
#include "geodex/manifold/euclidean.hpp"
#include "geodex/metrics/kinetic_energy.hpp"
#include "geodex/metrics/se2_left_invariant.hpp"
#include "geodex/robots/mass_lower_bound.hpp"
#include "geodex/robots/mass_matrix.hpp"

namespace geodex::robots {

/// @brief Metric on a robot's joint coordinates.
enum class ArmMetric {
  KineticEnergy,  ///< the precomputed CRBA mass matrix \f$ M(q) \f$
  Euclidean,      ///< the identity on joint coordinates
};

/// @brief Joint coordinates of robot @p R with its joint limits as sampling bounds.
///
/// @tparam R A built-in robot.
/// @tparam Metric Metric on the joint coordinates.
/// @tparam SamplerT Sampler behind `random_point()` and the planner's samples.
/// @return `Euclidean<n>` under `ArmMetric::Euclidean`. Under `ArmMetric::KineticEnergy`,
///         a `ConfigurationSpace` over it with `KineticEnergyMetric<MassMatrix<R>>`.
template <Robot R, ArmMetric Metric = ArmMetric::KineticEnergy,
          typename SamplerT = ScrambledHaltonSampler>
auto joint_space() {
  using MM = MassMatrix<R>;
  constexpr int n = MM::Nq;
  Euclidean<n, EuclideanStandardMetric<n>, SamplerT> joints;
  const auto [lo, hi] = MM::joint_limits();
  joints.set_sampling_bounds(lo, hi);
  if constexpr (Metric == ArmMetric::Euclidean) {
    return joints;
  } else {
    return ConfigurationSpace<decltype(joints), KineticEnergyMetric<MM>>(
        std::move(joints), KineticEnergyMetric<MM>(MM{}));
  }
}

/// @brief Loewner lower bound of the metric of `joint_space<R, metric>()` over the joint
/// limits.
///
/// @details Under `ArmMetric::KineticEnergy` it is `MassLowerBound<R>::matrix()`, and under
/// `ArmMetric::Euclidean` the identity.
/// @tparam R A built-in robot.
/// @param metric Metric on the joint coordinates.
template <Robot R>
Eigen::MatrixXd joint_lower_bound(const ArmMetric metric) {
  if (metric == ArmMetric::KineticEnergy) return MassLowerBound<R>::matrix();
  return Eigen::MatrixXd::Identity(MassMatrix<R>::Nq, MassMatrix<R>::Nq);
}

/// @brief Leading coordinates of a mobile robot's path that may keep corners, the value of
/// `algorithm::PathSmoothingSettings::sharp_coordinates` for its plans.
///
/// @details A base whose metric makes sideways motion cost more than driving turns at a
/// corner. Its pose, the first three coordinates, may keep the corner while the smoother rounds
/// the arm's joints. It returns 0 for any other base.
inline int sharp_base_coordinates(const SE2LeftInvariantMetric& base) {
  return base.weights()[1] > base.weights()[0] ? 3 : 0;
}

}  // namespace geodex::robots
