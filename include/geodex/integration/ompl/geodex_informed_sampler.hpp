/// @file geodex_informed_sampler.hpp
/// @brief Direct informed sampler for GeodexStateSpace with PHS, scaled-PHS,
///        and latent-space ellipsoid specializations.

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/goals/GoalSampleableRegion.h>
#include <ompl/base/samplers/InformedStateSampler.h>
#include <ompl/base/spaces/RealVectorBounds.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/util/ProlateHyperspheroid.h>
#include <ompl/util/RandomNumbers.h>

#include "geodex/heuristics/euclidean.hpp"
#include "geodex/heuristics/traits.hpp"
#include "geodex/integration/ompl/cost_bound_feedback.hpp"
#include "geodex/integration/ompl/geodex_state_space.hpp"

namespace geodex::integration::ompl {

namespace ob = ::ompl::base;

/// @brief Diagnostic counters and last-computed quantities from a sampler run.
///
/// @details Counters accumulate across `sampleUniform` calls and are reset by
/// `GeodexDirectInfSampler::resetSamplingStats()`. The boolean and volume
/// fields reflect the most recent strategy decision in the MatrixLB branch.
struct SamplingStats {
  /// Total `sampleUniform` inner-loop iterations across all calls.
  unsigned long total_attempts = 0;
  /// Iterations rejected for PHS-membership violation (clipped-AABB strategy).
  unsigned long phs_rejections = 0;
  /// Iterations rejected for coordinate-bounds violation.
  unsigned long bounds_rejections = 0;
  /// Iterations that returned a valid sample.
  unsigned long accepted = 0;
  /// `true` when the MatrixLB branch is using the clipped-AABB strategy.
  bool using_clipped_aabb = true;
  /// Volume of the latest clipped AABB (MatrixLB branch).
  double clipped_aabb_volume = 0.0;
  /// Volume of the latest latent PHS (MatrixLB branch).
  double phs_volume = 0.0;
  /// Times the volume-ratio fallback redirected to uniform sampling.
  unsigned long uniform_fallback_count = 0;
  /// Times a sample was drawn from a tighter cost bound (greedy biasing).
  unsigned long focused_sample_count = 0;
  /// Most recent PHS volume divided by C-space measure.
  double last_volume_ratio = 0.0;
};

/// @brief Direct informed sampler for GeodexStateSpace.
///
/// @details Dispatches over the heuristic type at compile time:
///
/// - `geodex::heuristics::Euclidean` — direct prolate hyperspheroid (PHS)
///   sampling in original coordinates.
/// - `geodex::heuristics::EigenvalueLowerBound<Base>` — PHS sampling with the
///   transverse diameter scaled by \f$ 1/\sqrt{\lambda_{\min}} \f$. The
///   informed set
///   \f$ \{x : \sqrt{\lambda_{\min}}(\|x-s\|+\|x-g\|) \le c\} \f$
///   is exactly a PHS with effective cost \f$ c/\sqrt{\lambda_{\min}} \f$.
/// - `geodex::heuristics::MatrixLowerBound<Dim>` — latent-space PHS sampling.
///   The Cholesky factor \f$ L \f$ of \f$ M_{\mathrm{lower}} = LL^\top \f$ maps
///   coordinates to a latent isotropic Euclidean space \f$ y = L^\top x \f$
///   where standard PHS sampling applies; samples back-transform via
///   \f$ x = L^{-\top} y \f$. When OMPL bounds are supplied via the ctor, the
///   sampler additionally chooses adaptively per cost level between sampling
///   the latent PHS directly (rejecting for bounds) and sampling the clipped
///   latent AABB (intersection of PHS-AABB with the latent image of the
///   bounds parallelotope, rejecting for PHS membership and bounds).
/// - Other heuristics — heuristic-guided rejection sampling.
///
/// @see Phone Thiha Kyaw, Jonathan Kelly. "Direct Informed Sampling on
///   Riemannian Manifolds via Loewner Order Lower Bounds." arXiv:2606.02879
///   (2026).
///
/// **Self-refresh.** When a feedback channel is supplied, every
/// `sampleUniform` call cheaply checks whether the problem definition has
/// gained a new solution since the last call (one count comparison). On a
/// new exact solution, the sampler walks the path to recompute the
/// heuristic-path-cost and greedy-cost bounds and writes them back to
/// the feedback channel, gated by a 5% relative-improvement threshold.
/// This means `planner->solve(planning_time)` tightens the informed set
//  without an external checkpoint loop. Callers that want explicit control of
//  the bounds can either omit the feedback channel or use the objective's
/// `setHeuristicPathCost` / `setGreedyCost` setters as a manual override
/// (the threshold gate prevents the auto-refresh from clobbering values
/// the caller just set, unless the new path is a strictly bigger
/// improvement than the manually set bound).
///
/// @tparam HeuristicT Callable with signature `double(Point, Point)`.
template <typename HeuristicT = geodex::heuristics::Euclidean>
class GeodexDirectInfSampler : public ob::InformedSampler {
  static constexpr bool kIsEuclidean = std::is_same_v<HeuristicT, geodex::heuristics::Euclidean>;
  static constexpr bool kIsEigenvalueLB =
      geodex::heuristics::is_eigenvalue_lower_bound_v<HeuristicT>;
  static constexpr bool kIsMatrixLB = geodex::heuristics::is_matrix_lower_bound_v<HeuristicT>;
  static constexpr bool kHasDirectSampling = kIsEuclidean || kIsEigenvalueLB || kIsMatrixLB;

  /// @brief Default volume-ratio fallback threshold for inadmissibly-large PHSes.
  static constexpr double kDefaultVolumeRatioThreshold = 20.0;
  static constexpr double kCostTolerance = 1e-12;

 public:
  /// @brief Construct the informed sampler.
  /// @param probDefn Problem definition (provides start/goal states).
  /// @param maxNumberCalls Maximum sampling attempts per call.
  /// @param heuristic Admissible heuristic functor.
  /// @param bounds Optional coordinate bounds. When non-empty and the heuristic
  ///        is `MatrixLowerBound`, enables adaptive clipped-AABB sampling.
  /// @param feedback Optional shared feedback channel for greedy biasing and
  ///        heuristic-path-cost tightening, typically minted by the objective.
  ///
  /// @todo Multi-start / multi-goal support.
  GeodexDirectInfSampler(const ob::ProblemDefinitionPtr& probDefn, unsigned int maxNumberCalls,
                         HeuristicT heuristic = HeuristicT{},
                         const ob::RealVectorBounds& bounds = ob::RealVectorBounds(0),
                         std::shared_ptr<CostBoundFeedback> feedback = nullptr)
      : ob::InformedSampler(probDefn, maxNumberCalls),
        heuristic_(std::move(heuristic)),
        feedback_(std::move(feedback)) {
    const auto* startState = probDefn_->getStartState(0);
    auto* goalState = space_->allocState();
    probDefn_->getGoal()->as<ob::GoalSampleableRegion>()->sampleGoal(goalState);

    const unsigned int dim = space_->getDimension();
    start_coords_.resize(dim);
    goal_coords_.resize(dim);
    space_->copyToReals(start_coords_, startState);
    space_->copyToReals(goal_coords_, goalState);
    space_->freeState(goalState);

    baseSampler_ = space_->allocStateSampler();

    coords_buf_.resize(dim);
    latent_buf_.resize(dim);
    orig_buf_.resize(dim);
    latent_eigen_buf_.resize(dim);

    if constexpr (kIsEuclidean) {
      phs_ = std::make_shared<::ompl::ProlateHyperspheroid>(dim, start_coords_.data(),
                                                            goal_coords_.data());
    } else if constexpr (kIsEigenvalueLB) {
      sqrt_lambda_min_ = heuristic_.sqrt_lambda_min();
      phs_ = std::make_shared<::ompl::ProlateHyperspheroid>(dim, start_coords_.data(),
                                                            goal_coords_.data());
    } else if constexpr (kIsMatrixLB) {
      const auto& llt = heuristic_.llt();
      const Eigen::MatrixXd L = llt.matrixL();
      Lt_ = llt.matrixU();
      L_inv_t_ = Lt_.template triangularView<Eigen::Upper>().solve(
          Eigen::MatrixXd::Identity(dim, dim));
      const double det_L = L.determinant();
      det_M_lower_ = det_L * det_L;

      Eigen::Map<const Eigen::VectorXd> s(start_coords_.data(), dim);
      Eigen::Map<const Eigen::VectorXd> g(goal_coords_.data(), dim);
      const Eigen::VectorXd ys = Lt_ * s;
      const Eigen::VectorXd yg = Lt_ * g;
      latent_start_.assign(ys.data(), ys.data() + dim);
      latent_goal_.assign(yg.data(), yg.data() + dim);
      phs_ = std::make_shared<::ompl::ProlateHyperspheroid>(dim, latent_start_.data(),
                                                            latent_goal_.data());

      if (bounds.low.size() == dim && dim > 0) {
        has_latent_bounds_ = true;
        latent_bounds_lo_.resize(dim);
        latent_bounds_hi_.resize(dim);
        clipped_lo_.resize(dim);
        clipped_hi_.resize(dim);
        // AABB of the parallelotope L^T B in latent space:
        //   y_i_min = sum_j min(Lt(i,j) * lo_j, Lt(i,j) * hi_j)
        //   y_i_max = sum_j max(Lt(i,j) * lo_j, Lt(i,j) * hi_j)
        for (unsigned int i = 0; i < dim; ++i) {
          double lo_sum = 0.0;
          double hi_sum = 0.0;
          for (unsigned int j = 0; j < dim; ++j) {
            const double a = Lt_(i, j) * bounds.low[j];
            const double b = Lt_(i, j) * bounds.high[j];
            lo_sum += std::min(a, b);
            hi_sum += std::max(a, b);
          }
          latent_bounds_lo_[i] = lo_sum;
          latent_bounds_hi_[i] = hi_sum;
        }
      }
    }

    // Self-fetch wiring. When a feedback channel is supplied, the sampler
    // chains itself onto pdef's intermediate-solution callback so it can pull
    // the latest exact-solution path directly out of the planner the moment
    // a new one is found, without any user-side wiring. The wrapper is gated
    // on a shared `alive` sentinel so that even if pdef outlives the sampler
    // (e.g. another planner reuses pdef), the lambda becomes a passthrough
    // to whatever callback was previously installed.
    if (feedback_ && probDefn_ && feedback_->self_refresh_enabled) {
      auto saved = probDefn_->getIntermediateSolutionCallback();
      auto alive = sampler_alive_;
      auto* self = this;
      probDefn_->setIntermediateSolutionCallback(
          [self, alive, saved](const ob::Planner* p,
                               const std::vector<const ob::State*>& spath,
                               const ob::Cost cost) {
            if (*alive && spath.size() >= 2) self->applyPathToFeedback(spath);
            if (saved) saved(p, spath, cost);
          });

      // Bootstrap: if pdef already carries an exact solution at sampler-alloc
      // time (e.g. unit tests that addSolutionPath before calling
      // allocInformedStateSampler), refresh feedback from it now.
      if (probDefn_->hasExactSolution()) {
        if (auto path = std::dynamic_pointer_cast<::ompl::geometric::PathGeometric>(
                probDefn_->getSolutionPath())) {
          applyPathToFeedback(path->getStates());
          last_seen_solution_count_ = probDefn_->getSolutionCount();
        }
      }
    }
  }

  /// @brief Deactivate the chained intermediate-solution callback installed in
  /// the ctor so that subsequent fires are passthroughs to the previous
  /// callback rather than dereferencing a destroyed sampler.
  ~GeodexDirectInfSampler() override {
    if (sampler_alive_) *sampler_alive_ = false;
  }

  /// @brief Sample uniformly from the informed region {x : h(s,x) + h(x,g) <= maxCost}.
  bool sampleUniform(ob::State* statePtr, const ob::Cost& maxCost) override {
    maybeRefreshFromSolution();
    const ob::Cost effective = narrowCost(maxCost);
    return sampleUniformWithAttempts(statePtr, effective, numIters_);
  }

  /// @brief Sample from the annular informed region (minCost <= cost <= maxCost).
  ///
  /// @details Uses inclusive lower bound (`>=`) to match OMPL's
  /// `RejectionInfSampler` (`isCostEquivalentTo || isCostBetterThan`).
  bool sampleUniform(ob::State* statePtr, const ob::Cost& minCost,
                     const ob::Cost& maxCost) override {
    maybeRefreshFromSolution();
    const ob::Cost effective = narrowCost(maxCost);
    for (unsigned int i = 0; i < numIters_; ++i) {
      if (sampleUniformWithAttempts(statePtr, effective, 1u)) {
        if (costAtLeast(heuristicCost(statePtr), minCost.value())) {
          return true;
        }
      }
    }
    return false;
  }

  /// @brief Whether this sampler has an analytic measure of the informed region.
  bool hasInformedMeasure() const override { return kHasDirectSampling; }

  /// @brief Measure (volume) of the informed region at the given cost.
  double getInformedMeasure(const ob::Cost& currentCost) const override {
    if (!kHasDirectSampling || std::isinf(currentCost.value())) {
      return space_->getMeasure();
    }
    if constexpr (kIsEigenvalueLB) {
      const double effective = currentCost.value() / sqrt_lambda_min_;
      if (effective < phs_->getMinTransverseDiameter()) return 0.0;
      return phs_->getPhsMeasure(effective);
    } else if constexpr (kIsMatrixLB) {
      if (currentCost.value() < phs_->getMinTransverseDiameter()) return 0.0;
      return phs_->getPhsMeasure(currentCost.value()) / std::sqrt(det_M_lower_);
    } else {
      // kIsEuclidean
      if (currentCost.value() < phs_->getMinTransverseDiameter()) return 0.0;
      return phs_->getPhsMeasure(currentCost.value());
    }
  }

  /// @brief Heuristic cost of a solution path through the given state.
  ob::Cost heuristicSolnCost(const ob::State* statePtr) const override {
    return ob::Cost(heuristicCost(statePtr));
  }

  /// @brief Snapshot of the sampler's diagnostic counters.
  auto getSamplingStats() const -> SamplingStats { return stats_; }

  /// @brief Reset all diagnostic counters to zero.
  void resetSamplingStats() { stats_ = SamplingStats{}; }

  /// @brief Set the volume-ratio fallback threshold for inadmissibly-large PHSes.
  /// @details If the latent PHS volume exceeds the C-space measure by this
  /// factor, the sampler falls back to bounded-domain rejection instead of
  /// sampling from the PHS and rejecting on bounds. For `MatrixLowerBound`, this
  /// is uniform over the latent bounds parallelotope via the linear map
  /// \f$ y = L^\top x \f$. Only consulted by direct-sampling branches.
  /// Default: `kDefaultVolumeRatioThreshold` (20).
  /// Set to `0` to disable the fallback.
  void setVolumeRatioThreshold(const double ratio) { volume_ratio_threshold_ = ratio; }

  /// @brief Get the current volume-ratio fallback threshold.
  auto getVolumeRatioThreshold() const -> double { return volume_ratio_threshold_; }

  /// @brief Latent-space bounds AABB (the image of the original-space bounds
  /// box under \f$ L^\top \f$). Populated only for `MatrixLowerBound`
  /// heuristics with non-empty ctor bounds.
  /// @returns `true` and writes `lo`/`hi` when populated; `false` otherwise.
  auto getLatentBoundsAABB(Eigen::VectorXd& lo, Eigen::VectorXd& hi) const -> bool {
    if constexpr (kIsMatrixLB) {
      if (has_latent_bounds_) {
        lo = latent_bounds_lo_;
        hi = latent_bounds_hi_;
        return true;
      }
    }
    return false;
  }

  /// @brief Most recent clipped AABB (intersection of the latent PHS AABB and
  /// the latent bounds AABB). Populated only after `sampleUniform` has been
  /// called on a finite cost in the MatrixLB branch with non-empty bounds.
  /// @returns `true` and writes `lo`/`hi` when the clipped strategy is active;
  ///          `false` otherwise.
  auto getClippedAABB(Eigen::VectorXd& lo, Eigen::VectorXd& hi) const -> bool {
    if constexpr (kIsMatrixLB) {
      if (has_latent_bounds_ && using_clipped_aabb_) {
        lo = clipped_lo_;
        hi = clipped_hi_;
        return true;
      }
    }
    return false;
  }

 private:
  /// @todo Deterministic (Halton) sampling support. Currently all randomness
  ///   routes through OMPL's `::ompl::RNG rng_`:
  ///     - `rng_.uniform01()`                 (greedy biasing coin flip)
  ///     - `rng_.uniformProlateHyperspheroid` (PHS sampling, 3 sites)
  ///     - `rng_.uniformReal(lo, hi)`         (clipped-AABB sampling)
  ///   Extend the OMPL allocator to forward the sampler choice. See
  ///   `core/sampler.hpp` for the existing `Sampler` concept.

  bool sampleUniformWithAttempts(ob::State* statePtr, const ob::Cost& effective,
                                 const unsigned int maxAttempts) {
    // Volume-ratio fallback for ALL direct-sampling heuristics (Euclidean,
    // EigenvalueLB, MatrixLB). When the informed region's volume exceeds
    // volume_ratio_threshold_ * C-space measure, sample-then-reject-for-bounds
    // has a catastrophic acceptance rate (most PHS samples land outside the
    // joint-limits box). Fall back to bounded-domain rejection instead. For
    // MatrixLB, this is equivalent to uniform rejection from the latent bounds
    // parallelotope under y = L^T x.
    if constexpr (kHasDirectSampling) {
      if (std::isfinite(effective.value())) {
        double check_volume = getInformedMeasure(effective);
        const double space_measure = space_->getMeasure();

        // For MatrixLB the uncapped PHS can be enormous in volume (stretched
        // along small-eigenvalue axes) while its AABB intersected with the
        // coordinate bounds is reasonable. Use the tighter of the two volumes
        // so the threshold check doesn't redirect to uniform when clipped-AABB
        // sampling would still be efficient.
        if constexpr (kIsMatrixLB) {
          updateClippedAABB(effective.value());
          if (has_latent_bounds_ && stats_.clipped_aabb_volume > 0.0) {
            const double clipped_original =
                stats_.clipped_aabb_volume / std::sqrt(det_M_lower_);
            if (clipped_original < check_volume) check_volume = clipped_original;
          }
        }

        stats_.last_volume_ratio = (space_measure > 0.0) ? check_volume / space_measure : 0.0;
        if (volume_ratio_threshold_ > 0.0 &&
            check_volume > volume_ratio_threshold_ * space_measure) {
          ++stats_.uniform_fallback_count;
          return sampleRejection(statePtr, effective, maxAttempts);
        }
      }
    }

    if constexpr (kIsMatrixLB) {
      return sampleMatrixLBLatent(statePtr, effective, maxAttempts);
    } else if constexpr (kIsEigenvalueLB) {
      return sampleEigenvalueLBPHS(statePtr, effective, maxAttempts);
    } else if constexpr (kIsEuclidean) {
      return sampleEuclideanPHS(statePtr, effective, maxAttempts);
    } else {
      return sampleRejection(statePtr, effective, maxAttempts);
    }
  }

  static double costTolerance(const double cost) {
    return kCostTolerance * std::max(1.0, std::abs(cost));
  }

  static bool costAtMost(const double cost, const double bound) {
    return cost <= bound + costTolerance(bound);
  }

  static bool costAtLeast(const double cost, const double bound) {
    return cost + costTolerance(bound) >= bound;
  }

  static bool costBelow(const double cost, const double bound) {
    return cost < bound - costTolerance(bound);
  }

  /// @brief Narrow `maxCost` using the shared cost-bound feedback channel.
  ///
  /// @details If `feedback_->heuristic_path_cost` is finite and tighter, take
  /// it. If `feedback_->greedy_biasing_ratio > 0` and `feedback_->greedy_cost`
  /// is finite, draw a uniform variate to decide whether to further narrow to
  /// the greedy bound. Bumps `stats_.focused_sample_count` on greedy hits.
  ob::Cost narrowCost(const ob::Cost& maxCost) {
    if (!feedback_) return maxCost;
    double effective = maxCost.value();
    // Only narrow with HPC when the planner actually has a finite cost bound.
    // Narrowing an `inf` maxCost down to a stale HPC (e.g. from a previous
    // solve cycle, or before the planner has propagated its first solution
    // back) yields an over-tight informed region the planner can't justify.
    if (!std::isinf(effective) && std::isfinite(feedback_->heuristic_path_cost)) {
      effective = std::min(effective, feedback_->heuristic_path_cost);
    }
    if (feedback_->greedy_biasing_ratio > 0.0 && std::isfinite(feedback_->greedy_cost) &&
        rng_.uniform01() < feedback_->greedy_biasing_ratio) {
      effective = std::min(effective, feedback_->greedy_cost);
      ++stats_.focused_sample_count;
    }
    return ob::Cost(effective);
  }

  /// @brief Compute h(start, state) + h(state, goal).
  double heuristicCost(const ob::State* statePtr) const {
    space_->copyToReals(coords_buf_, statePtr);
    if constexpr (kIsEuclidean) {
      return phs_->getPathLength(coords_buf_.data());
    } else {
      Eigen::Map<const Eigen::VectorXd> s(start_coords_.data(), start_coords_.size());
      Eigen::Map<const Eigen::VectorXd> g(goal_coords_.data(), goal_coords_.size());
      Eigen::Map<const Eigen::VectorXd> x(coords_buf_.data(), coords_buf_.size());
      return heuristic_(s, x) + heuristic_(x, g);
    }
  }

  /// @brief Direct PHS sampling for `geodex::heuristics::Euclidean`.
  bool sampleEuclideanPHS(ob::State* statePtr, const ob::Cost& maxCost,
                          const unsigned int maxAttempts) {
    if (std::isinf(maxCost.value())) {
      baseSampler_->sampleUniform(statePtr);
      ++stats_.uniform_fallback_count;
      return true;
    }
    const double minTD = phs_->getMinTransverseDiameter();
    if (costBelow(maxCost.value(), minTD)) {
      return false;
    }
    if (costAtMost(maxCost.value(), minTD)) {
      return sampleOriginalFocalSegment(statePtr, maxAttempts);
    }
    phs_->setTransverseDiameter(maxCost.value());

    for (unsigned int i = 0; i < maxAttempts; ++i) {
      ++stats_.total_attempts;
      rng_.uniformProlateHyperspheroid(phs_, coords_buf_.data());
      space_->copyFromReals(statePtr, coords_buf_);
      if (space_->satisfiesBounds(statePtr)) {
        ++stats_.accepted;
        return true;
      }
      ++stats_.bounds_rejections;
    }
    return false;
  }

  /// @brief PHS sampling with cost scaled by 1/sqrt(lambda_min).
  bool sampleEigenvalueLBPHS(ob::State* statePtr, const ob::Cost& maxCost,
                             const unsigned int maxAttempts) {
    if (std::isinf(maxCost.value())) {
      baseSampler_->sampleUniform(statePtr);
      ++stats_.uniform_fallback_count;
      return true;
    }
    double effective = maxCost.value() / sqrt_lambda_min_;
    const double minTD = phs_->getMinTransverseDiameter();
    if (costBelow(effective, minTD)) {
      // Inadmissible heuristic — fall back to uniform.
      baseSampler_->sampleUniform(statePtr);
      ++stats_.uniform_fallback_count;
      return true;
    }
    if (costAtMost(effective, minTD)) {
      return sampleOriginalFocalSegment(statePtr, maxAttempts);
    }
    phs_->setTransverseDiameter(effective);

    for (unsigned int i = 0; i < maxAttempts; ++i) {
      ++stats_.total_attempts;
      rng_.uniformProlateHyperspheroid(phs_, coords_buf_.data());
      space_->copyFromReals(statePtr, coords_buf_);
      if (space_->satisfiesBounds(statePtr)) {
        ++stats_.accepted;
        return true;
      }
      ++stats_.bounds_rejections;
    }
    return false;
  }

  /// @brief Latent-space ellipsoidal sampling for `MatrixLowerBound`.
  ///
  /// @details Picks adaptively between PHS-in-latent and clipped-AABB-in-latent
  /// based on whichever has the smaller sampling volume.
  bool sampleMatrixLBLatent(ob::State* statePtr, const ob::Cost& maxCost,
                            const unsigned int maxAttempts) {
    if (std::isinf(maxCost.value())) {
      baseSampler_->sampleUniform(statePtr);
      ++stats_.uniform_fallback_count;
      return true;
    }
    const double minTD = phs_->getMinTransverseDiameter();
    if (costBelow(maxCost.value(), minTD)) {
      // Inadmissible heuristic — fall back to uniform.
      baseSampler_->sampleUniform(statePtr);
      ++stats_.uniform_fallback_count;
      return true;
    }
    const unsigned int dim = space_->getDimension();
    if (costAtMost(maxCost.value(), minTD)) {
      return sampleLatentFocalSegment(statePtr, dim, maxAttempts);
    }
    phs_->setTransverseDiameter(maxCost.value());

    if (has_latent_bounds_) {
      updateClippedAABB(maxCost.value());
    }

    if (has_latent_bounds_ && using_clipped_aabb_) {
      return sampleFromClippedAABB(statePtr, maxCost, dim, maxAttempts);
    }
    return sampleFromPHSLatent(statePtr, dim, maxAttempts);
  }

  /// @brief Recompute the clipped AABB and decide which strategy to use.
  ///
  /// @details Intersects the latent PHS AABB with the latent bounds AABB and
  /// compares volumes. Result is cached against `last_c_best_`.
  void updateClippedAABB(double c_best) {
    if (!has_latent_bounds_ || !std::isfinite(c_best)) {
      using_clipped_aabb_ = false;
      return;
    }
    if (std::abs(c_best - last_c_best_) < 1e-15) return;  // cached
    last_c_best_ = c_best;

    const unsigned int dim = space_->getDimension();
    Eigen::Map<const Eigen::VectorXd> ys(latent_start_.data(), dim);
    Eigen::Map<const Eigen::VectorXd> yg(latent_goal_.data(), dim);
    const Eigen::VectorXd center = (ys + yg) / 2.0;
    const double d_foci = (yg - ys).norm();
    const double a = c_best / 2.0;
    const double c = d_foci / 2.0;
    const double b_sq = a * a - c * c;
    if (b_sq <= 0.0) {
      using_clipped_aabb_ = false;
      return;
    }
    const Eigen::VectorXd u = (yg - ys).normalized();

    double clipped_vol = 1.0;
    for (unsigned int i = 0; i < dim; ++i) {
      const double half_ext = std::sqrt(b_sq + (a * a - b_sq) * u[i] * u[i]);
      const double phs_lo = center[i] - half_ext;
      const double phs_hi = center[i] + half_ext;
      clipped_lo_[i] = std::max(phs_lo, latent_bounds_lo_[i]);
      clipped_hi_[i] = std::min(phs_hi, latent_bounds_hi_[i]);
      const double extent = clipped_hi_[i] - clipped_lo_[i];
      if (extent <= 0.0) {
        using_clipped_aabb_ = false;
        stats_.using_clipped_aabb = false;
        stats_.clipped_aabb_volume = 0.0;
        return;
      }
      clipped_vol *= extent;
    }
    const double phs_vol = phs_->getPhsMeasure(c_best);
    using_clipped_aabb_ = (clipped_vol < phs_vol);
    stats_.using_clipped_aabb = using_clipped_aabb_;
    stats_.clipped_aabb_volume = clipped_vol;
    stats_.phs_volume = phs_vol;
  }

  bool sampleOriginalFocalSegment(ob::State* statePtr, const unsigned int maxAttempts) {
    const unsigned int dim = space_->getDimension();
    for (unsigned int i = 0; i < maxAttempts; ++i) {
      ++stats_.total_attempts;
      const double t = rng_.uniform01();
      for (unsigned int d = 0; d < dim; ++d) {
        coords_buf_[d] = (1.0 - t) * start_coords_[d] + t * goal_coords_[d];
      }
      space_->copyFromReals(statePtr, coords_buf_);
      if (space_->satisfiesBounds(statePtr)) {
        ++stats_.accepted;
        return true;
      }
      ++stats_.bounds_rejections;
    }
    return false;
  }

  bool sampleLatentFocalSegment(ob::State* statePtr, unsigned int dim,
                                const unsigned int maxAttempts) {
    for (unsigned int i = 0; i < maxAttempts; ++i) {
      ++stats_.total_attempts;
      const double t = rng_.uniform01();
      for (unsigned int d = 0; d < dim; ++d) {
        latent_buf_[d] = (1.0 - t) * latent_start_[d] + t * latent_goal_[d];
      }
      Eigen::Map<const Eigen::VectorXd> y(latent_buf_.data(), dim);
      latent_eigen_buf_.noalias() = L_inv_t_ * y;
      Eigen::Map<Eigen::VectorXd>(orig_buf_.data(), dim) = latent_eigen_buf_;
      space_->copyFromReals(statePtr, orig_buf_);
      if (space_->satisfiesBounds(statePtr)) {
        ++stats_.accepted;
        return true;
      }
      ++stats_.bounds_rejections;
    }
    return false;
  }

  /// @brief Sample from the latent PHS, reject for original-space bounds.
  bool sampleFromPHSLatent(ob::State* statePtr, unsigned int dim, const unsigned int maxAttempts) {
    for (unsigned int i = 0; i < maxAttempts; ++i) {
      ++stats_.total_attempts;
      rng_.uniformProlateHyperspheroid(phs_, latent_buf_.data());
      Eigen::Map<const Eigen::VectorXd> y(latent_buf_.data(), dim);
      latent_eigen_buf_.noalias() = L_inv_t_ * y;
      Eigen::Map<Eigen::VectorXd>(orig_buf_.data(), dim) = latent_eigen_buf_;
      space_->copyFromReals(statePtr, orig_buf_);
      if (space_->satisfiesBounds(statePtr)) {
        ++stats_.accepted;
        return true;
      }
      ++stats_.bounds_rejections;
    }
    return false;
  }

  /// @brief Sample from the clipped AABB, reject for PHS membership and bounds.
  bool sampleFromClippedAABB(ob::State* statePtr, const ob::Cost& maxCost, unsigned int dim,
                             const unsigned int maxAttempts) {
    for (unsigned int i = 0; i < maxAttempts; ++i) {
      ++stats_.total_attempts;
      for (unsigned int d = 0; d < dim; ++d) {
        latent_buf_[d] = rng_.uniformReal(clipped_lo_[d], clipped_hi_[d]);
      }
      const double path_len = phs_->getPathLength(latent_buf_.data());
      if (!costAtMost(path_len, maxCost.value())) {
        ++stats_.phs_rejections;
        continue;
      }

      Eigen::Map<const Eigen::VectorXd> y(latent_buf_.data(), dim);
      latent_eigen_buf_.noalias() = L_inv_t_ * y;
      Eigen::Map<Eigen::VectorXd>(orig_buf_.data(), dim) = latent_eigen_buf_;
      space_->copyFromReals(statePtr, orig_buf_);
      if (space_->satisfiesBounds(statePtr)) {
        ++stats_.accepted;
        return true;
      }
      ++stats_.bounds_rejections;
    }
    return false;
  }

  /// @brief Refresh feedback costs from the latest solution in the problem
  /// definition.
  ///
  /// @details Returns early if no new solution exists. Otherwise computes:
  ///   - heuristic path cost: \f$ \sum_i h(p_i, p_{i+1}) \f$
  ///   - greedy cost: \f$ \max_p [h(s,p) + h(p,g)] \f$
  /// and writes both to the feedback channel. Updates are gated by
  /// `kCostUpdateThreshold` to ignore minor changes. Does nothing if no
  /// feedback channel was supplied; in that case callers should use
  /// `setHeuristicPathCost` / `setGreedyCost` directly.
  ///
  /// Called by planners that update pdef mid-solve and by tests that call
  /// `addSolutionPath` manually. Planners using `intermediateSolutionCallback`
  /// reach the same logic via the wrapper installed in the constructor; both
  /// paths go through `applyPathToFeedback`.
  void maybeRefreshFromSolution() {
    if (!feedback_) return;
    if (!feedback_->self_refresh_enabled) return;
    if (!probDefn_) return;

    const std::size_t count = probDefn_->getSolutionCount();
    if (count == last_seen_solution_count_) return;
    last_seen_solution_count_ = count;
    if (!probDefn_->hasExactSolution()) return;

    const auto path = std::dynamic_pointer_cast<::ompl::geometric::PathGeometric>(
        probDefn_->getSolutionPath());
    if (!path) return;
    applyPathToFeedback(path->getStates());
  }

  /// @brief Compute heuristic-path-cost and greedy-cost from a planner-emitted
  /// solution path and write to the shared feedback channel, gated by
  /// `kCostUpdateThreshold` so optimisation noise does not churn the bound.
  /// Templated on the state-pointer container element so we accept both
  /// `std::vector<State*>` (PathGeometric::getStates) and
  /// `std::vector<const State*>` (intermediateSolutionCallback's spath).
  template <typename StatePtr>
  void applyPathToFeedback(const std::vector<StatePtr>& states) {
    if (!feedback_) return;
    if (states.size() < 2) return;

    const unsigned int dim = space_->getDimension();
    std::vector<double> a_buf(dim);
    std::vector<double> b_buf(dim);

    // Heuristic-path-cost: Σ h(p_i, p_{i+1}) over consecutive waypoints.
    double hpc = 0.0;
    space_->copyToReals(b_buf, states[0]);
    for (std::size_t i = 1; i < states.size(); ++i) {
      std::swap(a_buf, b_buf);
      space_->copyToReals(b_buf, states[i]);
      const Eigen::Map<const Eigen::VectorXd> a(a_buf.data(), dim);
      const Eigen::Map<const Eigen::VectorXd> b(b_buf.data(), dim);
      hpc += heuristic_(a, b);
    }

    // Greedy cost: max_p [h(s, p) + h(p, g)] over all waypoints.
    const Eigen::Map<const Eigen::VectorXd> s(start_coords_.data(), dim);
    const Eigen::Map<const Eigen::VectorXd> g(goal_coords_.data(), dim);
    double gc = -std::numeric_limits<double>::infinity();
    for (const auto* st : states) {
      space_->copyToReals(a_buf, st);
      const Eigen::Map<const Eigen::VectorXd> p(a_buf.data(), dim);
      const double cost = heuristic_(s, p) + heuristic_(p, g);
      if (cost > gc) gc = cost;
    }

    // Threshold gate: skip the write if the previous bound was already
    // close. The first refresh (prev = +inf) always passes.
    if (std::isfinite(feedback_->heuristic_path_cost)) {
      const double prev = feedback_->heuristic_path_cost;
      if (prev <= 0.0 || (prev - hpc) / prev < kCostUpdateThreshold) return;
    }

    feedback_->heuristic_path_cost = hpc;
    feedback_->greedy_cost = gc;
  }

  /// @brief Heuristic-guided rejection sampling for non-trait heuristics.
  bool sampleRejection(ob::State* statePtr, const ob::Cost& maxCost,
                       const unsigned int maxAttempts) {
    for (unsigned int i = 0; i < maxAttempts; ++i) {
      ++stats_.total_attempts;
      baseSampler_->sampleUniform(statePtr);
      if (costAtMost(heuristicCost(statePtr), maxCost.value())) {
        ++stats_.accepted;
        return true;
      }
      ++stats_.phs_rejections;
    }
    return false;
  }

  HeuristicT heuristic_;
  std::shared_ptr<CostBoundFeedback> feedback_;
  std::vector<double> start_coords_;
  std::vector<double> goal_coords_;
  ob::StateSamplerPtr baseSampler_;
  ::ompl::RNG rng_;
  double volume_ratio_threshold_ = kDefaultVolumeRatioThreshold;

  std::shared_ptr<::ompl::ProlateHyperspheroid> phs_;

  // EigenvalueLB-only.
  double sqrt_lambda_min_ = 1.0;

  // MatrixLB-only — latent-space transforms.
  Eigen::MatrixXd Lt_;
  Eigen::MatrixXd L_inv_t_;
  double det_M_lower_ = 1.0;
  std::vector<double> latent_start_;
  std::vector<double> latent_goal_;

  // MatrixLB-only — clipped-AABB strategy state.
  bool has_latent_bounds_ = false;
  Eigen::VectorXd latent_bounds_lo_;
  Eigen::VectorXd latent_bounds_hi_;
  Eigen::VectorXd clipped_lo_;
  Eigen::VectorXd clipped_hi_;
  bool using_clipped_aabb_ = true;
  double last_c_best_ = -1.0;

  // Pre-allocated buffers — reused across sampleUniform calls to avoid heap
  // allocation in the hot path. Single-threaded planner usage assumed.
  mutable std::vector<double> coords_buf_;
  std::vector<double> latent_buf_;
  std::vector<double> orig_buf_;
  Eigen::VectorXd latent_eigen_buf_;

  // Self-refresh state — see maybeRefreshFromSolution().
  std::size_t last_seen_solution_count_ = 0;
  static constexpr double kCostUpdateThreshold = 0.05;

  // Sentinel shared with the chained intermediate-solution callback wrapper
  // installed on pdef in the ctor. Set to false in the dtor so that wrappers
  // outliving the sampler degrade to a passthrough rather than dereferencing
  // freed memory.
  std::shared_ptr<bool> sampler_alive_ = std::make_shared<bool>(true);

  SamplingStats stats_{};
};

}  // namespace geodex::integration::ompl
