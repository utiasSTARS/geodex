/// @file geodex_informed_sampler.hpp
/// @brief Direct informed sampler for GeodexStateSpace with PHS specialization.

#pragma once

#include <ompl/base/goals/GoalSampleableRegion.h>
#include <ompl/base/samplers/InformedStateSampler.h>
#include <ompl/util/ProlateHyperspheroid.h>
#include <ompl/util/RandomNumbers.h>

#include <cmath>
#include <geodex/algorithm/heuristics.hpp>
#include <geodex/integration/ompl/geodex_state_space.hpp>
#include <memory>
#include <type_traits>
#include <vector>

namespace geodex {
namespace ompl_integration {

namespace ob = ompl::base;

/// @brief Direct informed sampler for GeodexStateSpace.
///
/// @details For `EuclideanHeuristic`, samples directly from a prolate
/// hyperspheroid (PHS) centered on the start-goal line. For other heuristics,
/// falls back to heuristic-guided rejection sampling.
///
/// @tparam HeuristicT Callable with signature `double(Point, Point)`.
template <typename HeuristicT = EuclideanHeuristic>
class GeodexDirectInfSampler : public ob::InformedSampler {
 public:
  /// @brief Construct the informed sampler.
  /// @param probDefn Problem definition (provides start/goal states).
  /// @param maxNumberCalls Maximum sampling attempts per call.
  /// @param heuristic Admissible heuristic functor.
  GeodexDirectInfSampler(const ob::ProblemDefinitionPtr& probDefn, unsigned int maxNumberCalls,
                         HeuristicT heuristic = HeuristicT{})
      : ob::InformedSampler(probDefn, maxNumberCalls), heuristic_(std::move(heuristic)) {
    // Extract start and goal coordinates
    const auto* startState = probDefn_->getStartState(0);
    // Sample a goal state (GoalSampleableRegion doesn't expose getState directly)
    auto* goalState = space_->allocState();
    probDefn_->getGoal()->as<ob::GoalSampleableRegion>()->sampleGoal(goalState);

    unsigned int dim = space_->getDimension();
    start_coords_.resize(dim);
    goal_coords_.resize(dim);
    space_->copyToReals(start_coords_, startState);
    space_->copyToReals(goal_coords_, goalState);
    space_->freeState(goalState);

    // Create a base sampler for uniform fallback
    baseSampler_ = space_->allocStateSampler();

    if constexpr (std::is_same_v<HeuristicT, EuclideanHeuristic>) {
      // Build the prolate hyperspheroid from the two foci
      phs_ = std::make_shared<ompl::ProlateHyperspheroid>(dim, start_coords_.data(),
                                                          goal_coords_.data());
    }
  }

  /// @brief Sample a state uniformly from the informed region {x : h(s,x) + h(x,g) <= maxCost}.
  bool sampleUniform(ob::State* statePtr, const ob::Cost& maxCost) override {
    if constexpr (std::is_same_v<HeuristicT, EuclideanHeuristic>) {
      return samplePHS(statePtr, maxCost);
    } else {
      return sampleRejection(statePtr, maxCost);
    }
  }

  /// @brief Sample from the annular informed region (minCost < cost <= maxCost).
  bool sampleUniform(ob::State* statePtr, const ob::Cost& minCost,
                     const ob::Cost& maxCost) override {
    // Sample from outer PHS/region and reject if inside inner region
    for (unsigned int i = 0; i < numIters_; ++i) {
      if (sampleUniform(statePtr, maxCost)) {
        double cost = heuristicCost(statePtr);
        if (cost > minCost.value()) {
          return true;
        }
      }
    }
    return false;
  }

  /// @brief Whether this sampler has an analytic measure of the informed region.
  bool hasInformedMeasure() const override {
    return std::is_same_v<HeuristicT, EuclideanHeuristic>;
  }

  /// @brief Measure (volume) of the informed region at the given cost.
  double getInformedMeasure(const ob::Cost& currentCost) const override {
    if constexpr (std::is_same_v<HeuristicT, EuclideanHeuristic>) {
      if (std::isinf(currentCost.value())) {
        return space_->getMeasure();
      }
      if (currentCost.value() < phs_->getMinTransverseDiameter()) {
        return 0.0;
      }
      return phs_->getPhsMeasure(currentCost.value());
    } else {
      return space_->getMeasure();
    }
  }

  /// @brief Heuristic cost of a solution path through the given state.
  ob::Cost heuristicSolnCost(const ob::State* statePtr) const override {
    return ob::Cost(heuristicCost(statePtr));
  }

 private:
  /// @brief Compute h(start, state) + h(state, goal) using the heuristic.
  double heuristicCost(const ob::State* statePtr) const {
    std::vector<double> coords(space_->getDimension());
    space_->copyToReals(coords, statePtr);

    if constexpr (std::is_same_v<HeuristicT, EuclideanHeuristic>) {
      // Use PHS path length (L2 distance through foci) directly
      return phs_->getPathLength(coords.data());
    } else {
      // Map to Eigen for the heuristic callable
      Eigen::Map<const Eigen::VectorXd> s(start_coords_.data(), start_coords_.size());
      Eigen::Map<const Eigen::VectorXd> g(goal_coords_.data(), goal_coords_.size());
      Eigen::Map<const Eigen::VectorXd> x(coords.data(), coords.size());
      return heuristic_(s, x) + heuristic_(x, g);
    }
  }

  /// @brief Direct PHS sampling (EuclideanHeuristic only).
  bool samplePHS(ob::State* statePtr, const ob::Cost& maxCost) {
    // Before a solution is found, maxCost is infinity — fall back to uniform sampling
    if (std::isinf(maxCost.value())) {
      baseSampler_->sampleUniform(statePtr);
      return true;
    }
    if (maxCost.value() < phs_->getMinTransverseDiameter()) {
      return false;
    }
    phs_->setTransverseDiameter(maxCost.value());

    unsigned int dim = space_->getDimension();
    std::vector<double> coords(dim);

    for (unsigned int i = 0; i < numIters_; ++i) {
      rng_.uniformProlateHyperspheroid(phs_, coords.data());
      space_->copyFromReals(statePtr, coords);
      if (space_->satisfiesBounds(statePtr)) {
        return true;
      }
    }
    return false;
  }

  /// @brief Rejection sampling for generic heuristics.
  bool sampleRejection(ob::State* statePtr, const ob::Cost& maxCost) {
    for (unsigned int i = 0; i < numIters_; ++i) {
      baseSampler_->sampleUniform(statePtr);
      if (heuristicCost(statePtr) <= maxCost.value()) {
        return true;
      }
    }
    return false;
  }

  HeuristicT heuristic_;
  std::vector<double> start_coords_;
  std::vector<double> goal_coords_;
  ob::StateSamplerPtr baseSampler_;
  ompl::RNG rng_;

  // Only used for EuclideanHeuristic
  std::shared_ptr<ompl::ProlateHyperspheroid> phs_;
};

}  // namespace ompl_integration
}  // namespace geodex
