/// @file cost_bound_feedback.hpp
/// @brief Shared cost-bound feedback channel between objective and sampler.

#pragma once

#include <limits>

namespace geodex::integration::ompl {

/// @brief Shared cost-bound feedback channel between
/// `GeodexOptimizationObjective` and `GeodexDirectInfSampler`.
///
/// @details The objective mints one of these as a `std::shared_ptr` and
/// forwards it to each sampler at `allocInformedStateSampler` time, so a
/// caller's mid-plan updates (e.g. after each `solve()` checkpoint) become
/// visible to the next `sampleUniform` invocation. Both bounds default to
/// \f$ +\infty \f$ (disabled).
///
/// Either bound, when finite and tighter than the planner's `c_best`, narrows
/// the effective informed-set cost used by the sampler without altering the
/// admissibility of the underlying heuristic.
struct CostBoundFeedback {
  /// @brief Greedy cost bound — typically \f$ \max_{p \in \text{path}}
  /// [h(s, p) + h(p, g)] \f$.
  /// @details When finite and `greedy_biasing_ratio > 0`, a fraction
  /// `greedy_biasing_ratio` of samples are drawn from a tighter ellipsoid
  /// bounded by `greedy_cost` rather than the planner's `c_best`.
  double greedy_cost = std::numeric_limits<double>::infinity();

  /// @brief Heuristic-path-cost bound — \f$ \sum_i h(p_i, p_{i+1}) \f$ along
  /// the current solution path.
  /// @details Always \f$ \le c_{\text{best}} \f$ for admissible \f$ h \f$.
  /// When finite, the sampler uses \f$ \min(c_{\text{best}}, \text{HPC}) \f$
  /// as the effective cost bound.
  double heuristic_path_cost = std::numeric_limits<double>::infinity();

  /// @brief Fraction of samples drawn from the greedy ellipsoid in
  /// \f$ [0, 1] \f$. `0` disables greedy biasing; `1` always picks greedy when
  /// `greedy_cost` is finite.
  double greedy_biasing_ratio = 0.0;

  /// @brief Whether the sampler should auto-refresh `heuristic_path_cost` and
  /// `greedy_cost` from the planner's intermediate solutions.
  /// @details When `true` (default), the sampler chains itself onto pdef's
  /// intermediate-solution callback and also checks `pdef->getSolutionCount()`
  /// at every `sampleUniform`, recomputing both bounds from the latest path
  /// (5%-relative-improvement gated). When `false`, the sampler installs no
  /// callback and skips the count check; the caller is fully responsible for
  /// keeping the bounds up-to-date via the objective's `setHeuristicPathCost`
  /// and `setGreedyCost` setters. Toggle before `ss.solve()` is first called
  /// so the value is visible at sampler construction.
  bool self_refresh_enabled = true;
};

}  // namespace geodex::integration::ompl
