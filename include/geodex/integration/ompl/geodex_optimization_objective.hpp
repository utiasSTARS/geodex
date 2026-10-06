/// @file geodex_optimization_objective.hpp
/// @brief OMPL optimization objective using geodesic cost and admissible heuristic.

#pragma once

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
/// @details Uses geodesic distance (`si->distance()`) for motion cost and an admissible
/// heuristic, the Euclidean chord distance by default, for `motionCostHeuristic` and
/// `costToGo`. Informed planners (InformedRRT*, BIT*) focus sampling with it.
/// `setIntegratedArcCost(true)` switches the motion cost to the integrated arc cost.
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
  /// @details When enabled, `motionCost(s1, s2)` sums the per-segment Riemannian
  /// distances along the cached discrete geodesic from `s1` to `s2` and computes it
  /// when the cache does not hold the pair. It falls back to the endpoint distance
  /// when the cache cannot hold a valid path. By default, it uses `si->distance()`.
  void setIntegratedArcCost(bool enabled) { integrated_arc_cost_ = enabled; }

  /// @brief Whether integrated-arc cost is enabled.
  bool usesIntegratedArcCost() const { return integrated_arc_cost_; }

  /// @brief Enable or disable the sampler's auto-refresh of the cost-bound
  /// channel. Defaults to enabled.
  ///
  /// @details When enabled, the sampler chains onto the intermediate-solution callback
  /// of pdef and checks `pdef->getSolutionCount()` at every `sampleUniform`. It
  /// recomputes `heuristic_path_cost` and `greedy_cost` whenever the latest exact
  /// solution is cheaper. When disabled, the caller keeps the bounds current with
  /// `setHeuristicPathCost` and `setGreedyCost`. Call this before `ss.solve()`.
  void setSelfRefreshEnabled(const bool enabled) const {
    feedback_->self_refresh_enabled = enabled;
  }

  /// @brief Whether the sampler will auto-refresh the cost bounds.
  auto getSelfRefreshEnabled() const -> bool { return feedback_->self_refresh_enabled; }

  /// @brief Allow narrowing the sampling bound to the heuristic path cost.
  void setNarrowToHeuristicPathCost(const bool enabled) const {
    feedback_->narrow_to_heuristic_path_cost = enabled;
  }

  /// @brief Whether the sampling bound may be narrowed to the heuristic path cost.
  auto getNarrowToHeuristicPathCost() const -> bool {
    return feedback_->narrow_to_heuristic_path_cost;
  }

  /// @brief State cost (zero for path-length objectives).
  ob::Cost stateCost(const ob::State* /*s*/) const override { return ob::Cost(0.0); }

  /// @brief Motion cost, the endpoint distance by default and the arc cost when enabled.
  ob::Cost motionCost(const ob::State* s1, const ob::State* s2) const override {
    if (integrated_arc_cost_) {
      if (auto cost = tryArcCost(s1, s2); cost.has_value()) {
        return ob::Cost(*cost);
      }
    }
    return ob::Cost(si_->distance(s1, s2));
  }

  /// @brief Sampling stats from the most recently allocated informed sampler.
  /// @details Returns a default-constructed `SamplingStats` if the sampler has
  /// been deallocated by OMPL or before any sampler is allocated.
  auto getLastSamplerStats() const -> SamplingStats {
    if (auto sampler = last_sampler_.lock()) {
      return sampler->getSamplingStats();
    }
    return {};
  }

  /// @brief Sampling stats summed over every informed sampler still alive, one
  /// per tree for a bidirectional planner such as G-RRT*.
  /// @details Counters add. The strategy and volume fields come from the most
  /// recently allocated sampler.
  auto getSamplerStats() const -> SamplingStats {
    SamplingStats total = getLastSamplerStats();
    total.total_attempts = total.phs_rejections = total.bounds_rejections = 0;
    total.accepted = total.uniform_samples = total.focused_sample_count = 0;
    for (const auto& weak : samplers_) {
      if (const auto sampler = weak.lock()) {
        const SamplingStats s = sampler->getSamplingStats();
        total.total_attempts += s.total_attempts;
        total.phs_rejections += s.phs_rejections;
        total.bounds_rejections += s.bounds_rejections;
        total.accepted += s.accepted;
        total.uniform_samples += s.uniform_samples;
        total.focused_sample_count += s.focused_sample_count;
      }
    }
    return total;
  }

  /// @name Greedy informed sampling
  /// @{
  ///
  /// @details Tightens the informed-set cost bound with the G-RRT* rule, the maximum
  /// heuristic cost along the current solution path. `setGreedyBiasingRatio(r)` sets
  /// the mixture probability of the greedy ellipsoid against the full PHS (0 disables
  /// it). `setGreedyCost(c)` sets the bound, and `computeGreedyCost(path)` computes it.
  ///
  /// @see Phone Thiha Kyaw, Anh Vu Le, Rajesh Elara Mohan, Jonathan Kelly.
  ///   "Greedy Heuristics for Sampling-Based Motion Planning in
  ///   High-Dimensional State Spaces." Autonomous Robots, 2026. arXiv:2405.03411.

  /// @brief Set the fraction of samples from the greedy ellipsoid.
  /// @param ratio Value in `[0, 1]`. `0` disables greedy biasing.
  void setGreedyBiasingRatio(const double ratio) const {
    feedback_->greedy_biasing_ratio = ratio;
  }

  /// @brief Get the current greedy biasing ratio.
  auto getGreedyBiasingRatio() const -> double { return feedback_->greedy_biasing_ratio; }

  /// @brief Set the greedy cost bound.
  /// @details Typically the maximum heuristic cost along the current solution
  /// path, \f$ \max_{p \in \text{path}} [h(s, p) + h(p, g)] \f$. The next
  /// `sampleUniform` call of every sampler the objective has allocated sees it.
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
    // Copy the endpoints, which every iteration reads. Inner points stay Eigen::Map
    // views and are read once.
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
  /// @details Tightens the informed-set cost bound with
  /// \f$ \sum_i h(p_i, p_{i+1}) \le c_{\text{best}} \f$, which holds for any admissible
  /// \f$ h \f$. `setHeuristicPathCost(c)` sets that sum, and `computeHeuristicPathCost`
  /// computes it. The sampler reads it through `CostBoundFeedback` when it is finite.

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
  /// @details Passes the coordinate bounds of the underlying `GeodexStateSpace` to the
  /// clipped-AABB strategy and takes the sampler's streams from that space, with the
  /// manifold's sampler and the space's seed. On an embedded manifold it rejection-samples
  /// on the manifold. Other state spaces give empty bounds and PHS rejection sampling.
  ob::InformedSamplerPtr allocInformedStateSampler(const ob::ProblemDefinitionPtr& probDefn,
                                                   unsigned int maxNumberCalls) const override {
    using SamplerType = typename ManifoldT::SamplerType;
    ob::RealVectorBounds bounds(0);
    std::optional<InformedSamplerStreams<SamplerType>> streams;
    const auto* gss = dynamic_cast<const GeodexStateSpace<ManifoldT>*>(si_->getStateSpace().get());
    if (gss) {
      bounds = gss->getBounds();
      // Braced initialization evaluates left to right. The spatial stream is taken
      // before the scalar seed.
      streams =
          InformedSamplerStreams<SamplerType>{gss->allocManifoldSampler(), gss->nextStreamSeed()};
    }
    auto sampler = std::make_shared<GeodexDirectInfSampler<HeuristicT, SamplerType>>(
        probDefn, maxNumberCalls, heuristic_, bounds, feedback_, std::move(streams));
    // Turn off direct sampling on an embedded manifold. A sample in its ambient
    // coordinates would leave the manifold.
    if (gss && gss->getManifold().dim() != static_cast<int>(gss->getDimension())) {
      sampler->setDirectSampling(false);
    }
    last_sampler_ = sampler;
    std::erase_if(samplers_, [](const auto& weak) { return weak.expired(); });
    samplers_.push_back(sampler);
    return sampler;
  }

 private:
  /// @brief Admissible cost-to-go, the heuristic distance from the state to the goal.
  ob::Cost costToGoHeuristic(const ob::State* state) const {
    const auto* s = state->as<StateType>();
    return ob::Cost(heuristic_(s->asEigen(), goal_coords_));
  }

  /// @brief Compute the integrated arc cost when the state space is a
  /// `GeodexStateSpace<ManifoldT>`, and fill the cache when needed. Returns nullopt
  /// otherwise, and the caller uses the endpoint distance.
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
  mutable std::weak_ptr<GeodexDirectInfSampler<HeuristicT, typename ManifoldT::SamplerType>>
      last_sampler_;
  mutable std::vector<
      std::weak_ptr<GeodexDirectInfSampler<HeuristicT, typename ManifoldT::SamplerType>>>
      samplers_;  ///< every allocated sampler, for getSamplerStats
};

}  // namespace geodex::integration::ompl
