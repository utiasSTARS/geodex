/// @file geodex_optimization_objective.hpp
/// @brief OMPL optimization objective using geodesic cost and admissible heuristic.

#pragma once

#include <optional>

#include <ompl/base/OptimizationObjective.h>
#include <ompl/base/SpaceInformation.h>

#include "geodex/algorithm/heuristics.hpp"
#include "geodex/integration/ompl/geodex_informed_sampler.hpp"
#include "geodex/integration/ompl/geodex_state_space.hpp"

namespace geodex::integration::ompl {

using geodex::EuclideanHeuristic;
using geodex::RiemannianManifold;

namespace ob = ::ompl::base;

/// @brief OMPL optimization objective for geodex manifolds.
///
/// @details Uses geodesic distance for motion cost (via `si->distance()`) and an
/// admissible heuristic (default: Euclidean chord distance) for `motionCostHeuristic`
/// and `costToGo`. This enables informed planners (InformedRRT*, BIT*) to focus
/// sampling in promising regions.
///
/// When `setIntegratedArcCost(true)` is enabled, `motionCost` returns the sum
/// of per-segment Riemannian distances along the cached discrete-geodesic arc
/// instead of the endpoint-only `si->distance()`. This makes the planner's
/// parent-selection and rewiring reflect the actual curved arc the local
/// planner will traverse under the custom metric, rather than a scalar
/// midpoint approximation of its endpoints. Falls back to endpoint distance
/// when the cache cannot hold a valid path for the pair.
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

  /// @brief Opt into integrated-arc motion cost.
  ///
  /// @details When enabled, `motionCost(s1, s2)` computes the arc cost by
  /// summing per-segment Riemannian distances along the cached discrete
  /// geodesic from `s1` to `s2`, triggering a compute when the cache doesn't
  /// hold the pair. When disabled (default), `motionCost` uses
  /// `si->distance()`.
  void setIntegratedArcCost(bool enabled) { integrated_arc_cost_ = enabled; }

  /// @brief Whether integrated-arc cost is enabled.
  bool usesIntegratedArcCost() const { return integrated_arc_cost_; }

  /// @brief State cost (zero for path-length objectives).
  ob::Cost stateCost(const ob::State* /*s*/) const override { return ob::Cost(0.0); }

  /// @brief Motion cost: endpoint distance by default, arc cost when enabled.
  ob::Cost motionCost(const ob::State* s1, const ob::State* s2) const override {
    if (integrated_arc_cost_) {
      if (auto cost = tryArcCost(s1, s2); cost.has_value()) {
        return ob::Cost(*cost);
      }
    }
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

  /// @brief Compute the integrated arc cost if the state space is a
  /// `GeodexStateSpace<ManifoldT>`; otherwise returns nullopt so the caller
  /// falls back to endpoint distance. Populates the cache when needed.
  std::optional<double> tryArcCost(const ob::State* s1, const ob::State* s2) const {
    const auto* space = dynamic_cast<const GeodexStateSpace<ManifoldT>*>(si_->getStateSpace().get());
    if (!space) return std::nullopt;
    const auto* a = s1->as<StateType>();
    const auto* b = s2->as<StateType>();
    Point pa = a->asEigen();
    Point pb = b->asEigen();
    space->ensureGeodesicCached(pa, pb);
    const auto& cache = space->getGeodesicCache();
    if (!cache.valid()) return std::nullopt;
    return cache.total_arc_cost();
  }

  Point goal_coords_;
  HeuristicT heuristic_;
  bool integrated_arc_cost_ = false;
};

}  // namespace geodex::integration::ompl
