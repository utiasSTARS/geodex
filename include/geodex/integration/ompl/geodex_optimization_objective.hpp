/// @file geodex_optimization_objective.hpp
/// @brief OMPL optimization objective using geodesic cost and admissible heuristic.

#pragma once

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include <ompl/base/OptimizationObjective.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorBounds.h>

#include "geodex/heuristics/euclidean.hpp"
#include "geodex/integration/ompl/cost_bound_feedback.hpp"
#include "geodex/integration/ompl/geodex_informed_sampler.hpp"
#include "geodex/integration/ompl/geodex_state_space.hpp"

namespace geodex::integration::ompl {

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
///         `geodex::heuristics::Euclidean` which computes \f$ \|a - b\|_2 \f$.
template <typename ManifoldT, typename HeuristicT = geodex::heuristics::Euclidean>
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
      : ob::OptimizationObjective(si),
        goal_coords_(goal_coords),
        heuristic_(std::move(heuristic)),
        feedback_(std::make_shared<CostBoundFeedback>()) {
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

  /// @brief Enable or disable the sampler's auto-refresh of the cost-bound
  /// channel. Defaults to enabled.
  ///
  /// @details When enabled (default), the sampler chains onto pdef's
  /// intermediate-solution callback and also rechecks `pdef->getSolutionCount()`
  /// at every `sampleUniform`, recomputing `heuristic_path_cost` and
  /// `greedy_cost` from the latest exact solution (gated by a 5%
  /// relative-improvement threshold). When disabled, the sampler installs no
  /// callback and skips the count check; the caller is responsible for keeping
  /// the bounds up-to-date via `setHeuristicPathCost` / `setGreedyCost`.
  /// Call this *before* the planner allocates its sampler (i.e. before `ss.solve()`).
  void setSelfRefreshEnabled(const bool enabled) const {
    feedback_->self_refresh_enabled = enabled;
  }

  /// @brief Whether the sampler will auto-refresh the cost bounds.
  auto getSelfRefreshEnabled() const -> bool { return feedback_->self_refresh_enabled; }

  /// @brief State cost (zero for path-length objectives).
  ob::Cost stateCost(const ob::State* /*s*/) const override { return ob::Cost(0.0); }

  /// @brief Motion cost: endpoint distance by default, arc cost when enabled.
  ob::Cost motionCost(const ob::State* s1, const ob::State* s2) const override {
    motion_cost_calls_.fetch_add(1, std::memory_order_relaxed);
    if (integrated_arc_cost_) {
      if (auto cost = tryArcCost(s1, s2); cost.has_value()) {
        return ob::Cost(*cost);
      }
    }
    return ob::Cost(si_->distance(s1, s2));
  }

  /// @brief Total number of `motionCost` invocations since construction.
  auto getMotionCostCallCount() const -> std::uint64_t {
    return motion_cost_calls_.load(std::memory_order_relaxed);
  }

  /// @brief Sampling stats from the most recently allocated informed sampler.
  /// @details Returns a default-constructed `SamplingStats` if the sampler has
  /// been deallocated by OMPL or no sampler has yet been allocated.
  auto getLastSamplerStats() const -> SamplingStats {
    if (auto sampler = last_sampler_.lock()) {
      return sampler->getSamplingStats();
    }
    return {};
  }

  /// @name Greedy informed sampling
  /// @{
  ///
  /// @details Tightening of the informed-set cost bound using the
  /// "maximum heuristic cost along the current solution path" rule
  /// from the G-RRT* algorithm.
  ///
  /// `setGreedyBiasingRatio(r)` sets the mixture probability between the full
  /// PHS and the tighter greedy ellipsoid (`r = 0` disables, `r = 1` always
  /// uses the greedy bound). `setGreedyCost(c)` injects the per-iteration
  /// bound; `computeGreedyCost(path)` is the convenience that produces it
  /// from a sequence of solution-path states.
  ///
  /// @see Phone Thiha Kyaw, Anh Vu Le, Rajesh Elara Mohan, Jonathan Kelly.
  ///   "Greedy Heuristics for Sampling-Based Motion Planning in
  ///   High-Dimensional State Spaces." arXiv:2405.03411 (2024).

  /// @brief Set the fraction of samples drawn from the greedy ellipsoid.
  /// @param ratio Value in `[0, 1]`. `0` disables greedy biasing.
  void setGreedyBiasingRatio(const double ratio) const {
    feedback_->greedy_biasing_ratio = ratio;
  }

  /// @brief Get the current greedy biasing ratio.
  auto getGreedyBiasingRatio() const -> double { return feedback_->greedy_biasing_ratio; }

  /// @brief Set the greedy cost bound.
  /// @details Typically the maximum heuristic cost along the current solution
  /// path: \f$ \max_{p \in \text{path}} [h(s, p) + h(p, g)] \f$. Visible to
  /// the next `sampleUniform` call on every sampler the objective has
  /// allocated.
  void setGreedyCost(const double cost) const { feedback_->greedy_cost = cost; }

  /// @brief Get the current greedy cost bound.
  auto getGreedyCost() const -> double { return feedback_->greedy_cost; }

  /// @brief Compute the greedy cost from a sequence of solution-path states.
  /// @details Returns \f$ \max_{p \in \text{path}} [h(s_0, p) + h(p, s_g)] \f$
  /// where \f$ s_0 \f$ is the first state and \f$ s_g \f$ the last.
  /// Returns `+inf` for an empty path.
  template <typename StatePtr>
  auto computeGreedyCost(const std::vector<StatePtr>& path_states) const -> double {
    if (path_states.empty()) return std::numeric_limits<double>::infinity();
    // Materialize copies of the endpoints (used in every loop iteration); inner
    // points stay as Eigen::Map views since they're consumed once per iter.
    const Point start_pt = path_states.front()->template as<StateType>()->asEigen();
    const Point goal_pt = path_states.back()->template as<StateType>()->asEigen();
    double c_max = -std::numeric_limits<double>::infinity();
    for (const auto& sp : path_states) {
      const auto pt = sp->template as<StateType>()->asEigen();  // view, no copy
      const double cost = heuristic_(start_pt, pt) + heuristic_(pt, goal_pt);
      if (cost > c_max) c_max = cost;
    }
    return c_max;
  }

  /// @}
  /// @name Heuristic-path-cost tightening
  /// @{
  ///
  /// @details Tightening of the informed-set cost bound using the
  /// admissibility identity \f$ \sum_i h(p_i, p_{i+1}) \le c_{\text{best}} \f$,
  /// which holds for any admissible \f$ h \f$.
  /// `setHeuristicPathCost(c)` injects that sum;
  /// `computeHeuristicPathCost(path)` produces it from a state sequence. The
  /// sampler consumes the value via `CostBoundFeedback` in subsequent
  /// `sampleUniform` calls and uses it as the effective cost bound when finite.

  /// @brief Set the heuristic-path-cost bound.
  /// @details \f$ \sum_i h(p_i, p_{i+1}) \f$ along the current solution path,
  /// always \f$ \le c_{\text{best}} \f$ for admissible \f$ h \f$. The sampler
  /// uses this as the effective cost bound when finite.
  void setHeuristicPathCost(const double cost) const {
    feedback_->heuristic_path_cost = cost;
  }

  /// @brief Get the current heuristic-path-cost bound.
  auto getHeuristicPathCost() const -> double { return feedback_->heuristic_path_cost; }

  /// @brief Compute the heuristic path cost from a sequence of states.
  /// @details Returns \f$ \sum_i h(p_i, p_{i+1}) \f$. Returns `+inf` for paths
  /// with fewer than two states.
  template <typename StatePtr>
  auto computeHeuristicPathCost(const std::vector<StatePtr>& path_states) const -> double {
    if (path_states.size() < 2) return std::numeric_limits<double>::infinity();
    double total = 0.0;
    for (std::size_t i = 0; i + 1 < path_states.size(); ++i) {
      const auto a = path_states[i]->template as<StateType>()->asEigen();      // view
      const auto b = path_states[i + 1]->template as<StateType>()->asEigen();  // view
      total += heuristic_(a, b);
    }
    return total;
  }

  /// @}

  /// @brief Admissible heuristic for motion cost between two states.
  ob::Cost motionCostHeuristic(const ob::State* s1, const ob::State* s2) const override {
    const auto* a = s1->as<StateType>();
    const auto* b = s2->as<StateType>();
    return ob::Cost(heuristic_(a->asEigen(), b->asEigen()));
  }

  /// @brief Allocate a direct informed sampler for this objective.
  ///
  /// @details Forwards the underlying `GeodexStateSpace`'s coordinate bounds
  /// (when present) so the MatrixLowerBound branch can use the clipped-AABB
  /// strategy. For other state-space types or unbounded ones, the sampler
  /// receives empty bounds and falls back to PHS-with-rejection sampling.
  ob::InformedSamplerPtr allocInformedStateSampler(const ob::ProblemDefinitionPtr& probDefn,
                                                   unsigned int maxNumberCalls) const override {
    ob::RealVectorBounds bounds(0);
    const auto* state_space = si_->getStateSpace().get();
    if (const auto* gss = dynamic_cast<const GeodexStateSpace<ManifoldT>*>(state_space)) {
      bounds = gss->getBounds();
    }
    auto sampler = std::make_shared<GeodexDirectInfSampler<HeuristicT>>(
        probDefn, maxNumberCalls, heuristic_, bounds, feedback_);
    last_sampler_ = sampler;
    return sampler;
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
  std::shared_ptr<CostBoundFeedback> feedback_;
  bool integrated_arc_cost_ = false;
  mutable std::atomic<std::uint64_t> motion_cost_calls_{0};
  mutable std::weak_ptr<GeodexDirectInfSampler<HeuristicT>> last_sampler_;
};

}  // namespace geodex::integration::ompl
