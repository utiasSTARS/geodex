/// @file geodex_optimization_objective.hpp
/// @brief OMPL optimization objective using geodesic cost and admissible heuristic.

#pragma once

#include <ompl/base/OptimizationObjective.h>
#include <ompl/base/SpaceInformation.h>

#include <geodex/algorithm/heuristics.hpp>
#include <geodex/integration/ompl/geodex_informed_sampler.hpp>
#include <geodex/integration/ompl/geodex_state_space.hpp>

namespace geodex {
namespace ompl_integration {

namespace ob = ompl::base;

/// @brief OMPL optimization objective for geodex manifolds.
///
/// @details Uses geodesic distance for motion cost (via `si->distance()`) and an
/// admissible heuristic (default: Euclidean chord distance) for `motionCostHeuristic`
/// and `costToGo`. This enables informed planners (InformedRRT*, BIT*) to focus
/// sampling in promising regions.
///
/// @tparam ManifoldT A type satisfying `geodex::RiemannianManifold`.
/// @tparam HeuristicT Callable with signature `double(Point, Point)`. Defaults to
///         `EuclideanHeuristic` which computes \f$ \|a - b\|_2 \f$.
template <typename ManifoldT, typename HeuristicT = EuclideanHeuristic>
class GeodexOptimizationObjective : public ob::OptimizationObjective {
 public:
  using Point = typename ManifoldT::Point;   ///< Manifold point type.
  using StateType = GeodexState<ManifoldT>;  ///< OMPL state type.

  /// @brief Construct the objective.
  /// @param si OMPL space information (distance uses geodesic metric).
  /// @param goal_coords Goal point coordinates for costToGo evaluation.
  /// @param heuristic Admissible heuristic functor.
  GeodexOptimizationObjective(const ob::SpaceInformationPtr& si, const Point& goal_coords,
                              HeuristicT heuristic = HeuristicT{})
      : ob::OptimizationObjective(si), goal_coords_(goal_coords), heuristic_(std::move(heuristic)) {
    description_ = "Geodex geodesic distance with admissible heuristic";
    setCostToGoHeuristic(
        [this](const ob::State* s, const ob::Goal*) { return this->costToGoHeuristic(s); });
  }

  /// @brief State cost (zero for path-length objectives).
  ob::Cost stateCost(const ob::State* /*s*/) const override { return ob::Cost(0.0); }

  /// @brief Motion cost is the geodesic distance (via si->distance).
  ob::Cost motionCost(const ob::State* s1, const ob::State* s2) const override {
    return ob::Cost(si_->distance(s1, s2));
  }

  /// @brief Admissible heuristic for motion cost between two states.
  ob::Cost motionCostHeuristic(const ob::State* s1, const ob::State* s2) const override {
    const auto* a = s1->as<StateType>();
    const auto* b = s2->as<StateType>();
    return ob::Cost(heuristic_(a->asEigen(), b->asEigen()));
  }

  /// @brief Allocate a direct informed sampler for this objective.
  ob::InformedSamplerPtr allocInformedStateSampler(const ob::ProblemDefinitionPtr& probDefn,
                                                   unsigned int maxNumberCalls) const override {
    return std::make_shared<GeodexDirectInfSampler<HeuristicT>>(probDefn, maxNumberCalls,
                                                                heuristic_);
  }

 private:
  /// @brief Admissible cost-to-go: heuristic distance from state to goal.
  ob::Cost costToGoHeuristic(const ob::State* state) const {
    const auto* s = state->as<StateType>();
    return ob::Cost(heuristic_(s->asEigen(), goal_coords_));
  }

  Point goal_coords_;
  HeuristicT heuristic_;
};

}  // namespace ompl_integration
}  // namespace geodex
