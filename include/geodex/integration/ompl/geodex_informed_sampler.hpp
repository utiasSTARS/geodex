/// @file geodex_informed_sampler.hpp
/// @brief Direct informed sampler for GeodexStateSpace with PHS, scaled-PHS,
///        and latent-space ellipsoid specializations.

#pragma once

#include <cmath>
#include <cstdint>

#include <algorithm>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
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

#include "geodex/core/sampler.hpp"
#include "geodex/heuristics/euclidean.hpp"
#include "geodex/heuristics/traits.hpp"
#include "geodex/integration/ompl/cost_bound_feedback.hpp"
#include "geodex/integration/ompl/geodex_state_space.hpp"
#include "geodex/utils/angle.hpp"
#include "geodex/utils/normal.hpp"
#include "geodex/utils/random.hpp"

namespace geodex::integration::ompl {

namespace ob = ::ompl::base;

/// @brief The streams of an informed sampler, one spatial sampler and the
/// seed of its scalar generator.
///
/// @details `GeodexOptimizationObjective` allocates both from its
/// `GeodexStateSpace`, with the manifold's sampler and the space's seed.
template <typename SamplerT>
struct InformedSamplerStreams {
  SamplerT spatial;           ///< hyperspheroid and clipped-AABB samples
  std::uint64_t scalar_seed;  ///< greedy coin, lift choice and overlap accept
};

/// @brief Counters and last-computed quantities from a sampler run.
///
/// @details Counters accumulate across `sampleUniform` calls. The boolean and ratio
/// fields reflect the most recent strategy decision.
struct SamplingStats {
  /// Total `sampleUniform` inner-loop iterations across all calls.
  unsigned long total_attempts = 0;
  /// Iterations rejected for PHS-membership violation (clipped-AABB strategy and
  /// heuristic rejection sampling).
  unsigned long phs_rejections = 0;
  /// Iterations rejected for coordinate-bounds violation.
  unsigned long bounds_rejections = 0;
  /// Samples returned after passing an informed test, a PHS sample inside the bounds
  /// or a sample that met the heuristic bound.
  unsigned long accepted = 0;
  /// `true` when the MatrixLB branch is using the clipped-AABB strategy.
  bool using_clipped_aabb = true;
  /// Samples returned by plain uniform sampling without an informed test, before the
  /// first solution or when the cost bound falls below the heuristic's
  /// start-to-goal bound.
  unsigned long uniform_samples = 0;
  /// Samples from a tighter cost bound (greedy biasing).
  unsigned long focused_sample_count = 0;
  /// Most recent PHS volume divided by C-space measure.
  double last_volume_ratio = 0.0;
};

/// @brief Direct informed sampler for GeodexStateSpace.
///
/// @details Dispatches over the heuristic type at compile time.
///
/// - `geodex::heuristics::Euclidean` samples the prolate hyperspheroid (PHS) in the
///   original coordinates.
/// - `geodex::heuristics::EigenvalueLowerBound<Base>` samples a PHS with effective cost
///   \f$ c/\sqrt{\lambda_{\min}} \f$, which is exactly the informed set
///   \f$ \{x : \sqrt{\lambda_{\min}}(\|x-s\|+\|x-g\|) \le c\} \f$.
/// - `geodex::heuristics::MatrixLowerBound<Dim>` samples a PHS in the latent isotropic
///   space \f$ y = L^\top x \f$, with \f$ M_{\mathrm{lower}} = LL^\top \f$, and maps back
///   with \f$ x = L^{-\top} y \f$. With OMPL bounds it picks, per cost level, the latent
///   PHS or the clipped latent AABB, whichever has the smaller volume.
/// - Other heuristics use heuristic-guided rejection sampling.
///
/// @see Phone Thiha Kyaw, Jonathan Kelly. "Direct Informed Sampling on
///   Riemannian Manifolds via Loewner Order Lower Bounds." IEEE Robotics and
///   Automation Letters (RA-L), 2026. arXiv:2606.02879.
///
/// With a feedback channel, every `sampleUniform` call compares the solution count of
/// the problem definition with the last one. On a new exact solution that is cheaper
/// than the last one used, the sampler recomputes the heuristic-path-cost and
/// greedy-cost bounds and writes them to the channel. `planner->solve(planning_time)`
/// then tightens the informed set on its own. To set the bounds yourself, omit the
/// channel or turn the self-refresh off and use `setHeuristicPathCost` and
/// `setGreedyCost`.
///
/// @tparam HeuristicT Callable with signature `double(Point, Point)`.
/// @tparam SamplerT Sampler policy for the hyperspheroid and clipped-AABB samples;
///   the objective forwards the manifold's `SamplerType`.
template <typename HeuristicT = geodex::heuristics::Euclidean,
          typename SamplerT = geodex::DynamicSampler>
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
  /// @param streams Optional streams for the spatial and scalar samples. Empty
  ///        takes both seeds from geodex's thread-local seed source.
  ///
  /// @note Uses the first start state and one sampled goal state.
  GeodexDirectInfSampler(const ob::ProblemDefinitionPtr& probDefn, unsigned int maxNumberCalls,
                         HeuristicT heuristic = HeuristicT{},
                         const ob::RealVectorBounds& bounds = ob::RealVectorBounds(0),
                         std::shared_ptr<CostBoundFeedback> feedback = nullptr,
                         std::optional<InformedSamplerStreams<SamplerT>> streams = std::nullopt)
      : ob::InformedSampler(probDefn, maxNumberCalls),
        heuristic_(std::move(heuristic)),
        feedback_(std::move(feedback)),
        ld_(streams ? std::move(streams->spatial) : SamplerT{}),
        scalar_rng_(streams ? streams->scalar_seed : geodex::detail::seed_source()()) {
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
      phs_.push_back(std::make_shared<::ompl::ProlateHyperspheroid>(
          dim, start_coords_.data(), goal_coords_.data()));
    } else if constexpr (kIsEigenvalueLB) {
      sqrt_lambda_min_ = heuristic_.sqrt_lambda_min();
      phs_.push_back(std::make_shared<::ompl::ProlateHyperspheroid>(
          dim, start_coords_.data(), goal_coords_.data()));
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
      phs_.push_back(std::make_shared<::ompl::ProlateHyperspheroid>(
          dim, latent_start_.data(), latent_goal_.data()));
      lift_probe_.resize(dim);

      // The heuristic wraps periodic coordinates. The informed set is the projection
      // of a union of PHS over goal lifts, and the bounds supply the fundamental
      // domain it folds into.
      if (bounds.low.size() == dim && dim > 0) buildLifts(bounds, dim);

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

    // Fixed foci make the minimum reachable cost and the closest lift
    // constants of the sampler.
    if (!phs_.empty()) {
      min_td_ = phs_.front()->getMinTransverseDiameter();
      for (std::size_t k = 1; k < phs_.size(); ++k) {
        const double td = phs_[k]->getMinTransverseDiameter();
        if (td < min_td_) {
          min_td_ = td;
          best_lift_ = k;
        }
      }
    }

    // With a feedback channel, chain the sampler onto the intermediate-solution
    // callback of pdef. The sampler reads each new exact solution as the planner
    // finds it. A shared `alive` sentinel turns the wrapper into a passthrough to
    // the previous callback when pdef outlives the sampler, for example when
    // another planner reuses pdef.
    if (feedback_ && probDefn_ && feedback_->self_refresh_enabled) {
      auto saved = probDefn_->getIntermediateSolutionCallback();
      auto alive = sampler_alive_;
      auto* self = this;
      probDefn_->setIntermediateSolutionCallback(
          [self, alive, saved](const ob::Planner* p,
                               const std::vector<const ob::State*>& spath,
                               const ob::Cost cost) {
            if (*alive && spath.size() >= 2) self->applyPathToFeedback(spath, cost.value());
            if (saved) saved(p, spath, cost);
          });

      // Refresh the feedback now when pdef already carries an exact solution, for
      // example when the caller added a solution path before allocating the sampler.
      if (probDefn_->hasExactSolution()) {
        if (auto path = std::dynamic_pointer_cast<::ompl::geometric::PathGeometric>(
                probDefn_->getSolutionPath())) {
          applyPathToFeedback(path->getStates(), path->cost(opt_).value());
          last_seen_solution_count_ = probDefn_->getSolutionCount();
        }
      }
    }
  }

  /// @brief Deactivate the chained intermediate-solution callback installed in
  /// the ctor. Later calls pass through to the previous callback and do not
  /// touch the destroyed sampler.
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
  bool hasInformedMeasure() const override { return direct_; }

  /// @brief Measure (volume) of the informed region at the given cost.
  double getInformedMeasure(const ob::Cost& currentCost) const override {
    if (!direct_ || std::isinf(currentCost.value())) {
      return space_->getMeasure();
    }
    if constexpr (kIsEigenvalueLB) {
      const double effective = currentCost.value() / sqrt_lambda_min_;
      if (effective < phs_[0]->getMinTransverseDiameter()) return 0.0;
      return phs_[0]->getPhsMeasure(effective);
    } else if constexpr (kIsMatrixLB) {
      const double measure = unionMeasure(currentCost.value());
      if (measure == 0.0) return 0.0;
      // Summing lifts overcounts their overlap, and the union projects into the
      // fundamental domain. The C-space measure is the tighter upper bound.
      if (phs_.size() > 1) return std::min(measure, space_->getMeasure());
      return measure;
    } else {
      // kIsEuclidean
      if (currentCost.value() < phs_[0]->getMinTransverseDiameter()) return 0.0;
      return phs_[0]->getPhsMeasure(currentCost.value());
    }
  }

  /// @brief Heuristic cost of a solution path through the given state.
  ob::Cost heuristicSolnCost(const ob::State* statePtr) const override {
    return ob::Cost(heuristicCost(statePtr));
  }

  /// @brief Snapshot of the sampler's counters.
  auto getSamplingStats() const -> SamplingStats { return stats_; }

  /// @brief Choose between direct sampling and heuristic rejection sampling.
  ///
  /// @details Direct sampling samples in the state space's coordinates and is valid only
  /// when they are the manifold's intrinsic coordinates. When off, every sample
  /// rejection-samples over the base state sampler and stays on the manifold, as an
  /// embedded manifold such as a sphere or SO(3) needs. On by default when the
  /// heuristic has a direct strategy.
  void setDirectSampling(const bool enabled) { direct_ = enabled && kHasDirectSampling; }

  /// @brief Whether sampling uses the heuristic's direct strategy.
  auto getDirectSampling() const -> bool { return direct_; }

  /// @brief Set the volume-ratio fallback threshold for inadmissibly-large PHSes.
  /// @details When the latent PHS volume exceeds the C-space measure by this factor,
  /// the direct-sampling branches fall back to bounded-domain rejection. The default
  /// is `kDefaultVolumeRatioThreshold` (20), and `0` disables the fallback.
  void setVolumeRatioThreshold(const double ratio) { volume_ratio_threshold_ = ratio; }

  /// @brief Get the current volume-ratio fallback threshold.
  auto getVolumeRatioThreshold() const -> double { return volume_ratio_threshold_; }

 private:
  /// @brief Uncapped union measure in original coordinates, overlap counted
  /// once per lift. The volume-ratio check uses it. The public measure is capped.
  double unionMeasure(const double c) const {
    double latent = 0.0;
    for (const auto& phs : phs_) {
      if (c >= phs->getMinTransverseDiameter()) latent += phs->getPhsMeasure(c);
    }
    return latent / std::sqrt(det_M_lower_);
  }

  /// @brief Build one PHS per deck-group lift of the goal.
  ///
  /// @details With bounds one period wide on each periodic axis, the deck
  /// element realizing the wrapped heuristic has components in {-P, 0, +P}.
  /// \f$ k \in \{-1, 0, +1\} \f$ per axis then enumerates the informed set exactly.
  void buildLifts(const ob::RealVectorBounds& bounds, const unsigned int dim) {
    const Eigen::VectorXd& periods = heuristic_.periods();
    if (periods.size() != static_cast<int>(dim)) return;

    std::vector<int> axes;
    for (unsigned int i = 0; i < dim; ++i) {
      if (periods[i] > 0.0) axes.push_back(static_cast<int>(i));
    }
    if (axes.empty()) return;
    if (axes.size() > kMaxPeriodicAxes) {
      lifts_capped_ = true;
      return;
    }

    std::size_t n_combos = 1;
    for (std::size_t k = 0; k < axes.size(); ++k) n_combos *= 3;

    wrap_periods_ = periods;
    wrap_center_.resize(dim);
    for (unsigned int i = 0; i < dim; ++i) {
      wrap_center_[i] = 0.5 * (bounds.low[i] + bounds.high[i]);
    }

    lift_offsets_.push_back(Eigen::VectorXd::Zero(dim));  // parallel to phs_[0]
    Eigen::Map<const Eigen::VectorXd> g(goal_coords_.data(), dim);
    for (std::size_t combo = 0; combo < n_combos; ++combo) {
      Eigen::VectorXd offset = Eigen::VectorXd::Zero(dim);
      std::size_t rem = combo;
      bool is_base = true;
      for (const int ax : axes) {
        const int k = static_cast<int>(rem % 3) - 1;
        rem /= 3;
        if (k != 0) is_base = false;
        offset[ax] = k * periods[ax];
      }
      if (is_base) continue;
      const Eigen::VectorXd lifted = Lt_ * (g + offset);
      lift_offsets_.push_back(std::move(offset));
      phs_.push_back(std::make_shared<::ompl::ProlateHyperspheroid>(dim, latent_start_.data(),
                                                                    lifted.data()));
    }
  }

  /// @brief Set the reachable lifts to the current cost and build the measure
  /// CDF over all lifts, flat where a lift is unreachable. Mirrors
  /// `ompl::base::PathLengthDirectInfSampler::updatePhsDefinitions`.
  /// @return Total measure, 0 when no lift is reachable.
  double activateLifts(const double c_best) {
    lift_cdf_.resize(phs_.size());
    n_active_ = 0;
    first_active_ = 0;
    double total = 0.0;
    for (std::size_t k = 0; k < phs_.size(); ++k) {
      if (phs_[k]->getMinTransverseDiameter() < c_best) {
        phs_[k]->setTransverseDiameter(c_best);
        total += phs_[k]->getPhsMeasure(c_best);
        if (n_active_ == 0) first_active_ = k;
        ++n_active_;
      }
      lift_cdf_[k] = total;
    }
    return total;
  }

  /// @brief Sample a lift index with probability proportional to its measure.
  /// Mirrors `ompl::base::PathLengthDirectInfSampler::randomPhsPtr`.
  std::size_t pickLift(const double total) {
    if (n_active_ <= 1 || total <= 0.0) return first_active_;
    const double u = ldUniform01() * total;
    return static_cast<std::size_t>(std::upper_bound(lift_cdf_.begin(), lift_cdf_.end(), u) -
                                    lift_cdf_.begin());
  }

  /// @brief Fold the periodic coordinates of `orig_buf_` into the fundamental domain.
  void foldIntoDomain(const unsigned int dim) {
    if (wrap_periods_.size() != static_cast<int>(dim)) return;
    for (unsigned int i = 0; i < dim; ++i) {
      if (wrap_periods_[i] <= 0.0) continue;
      orig_buf_[i] = wrap_center_[i] + geodex::utils::wrap_delta_period(
                                           orig_buf_[i] - wrap_center_[i], wrap_periods_[i]);
    }
  }

  /// @brief Multiple-importance-sampling accept for the folded sample in `orig_buf_`.
  ///
  /// @details Folding is many-to-one. Accept with probability \f$ 1/N \f$, where
  /// \f$ N \f$ counts the (lattice translate, goal lift) pairs that fit the budget.
  /// The set is symmetric, and `lift_offsets_` serves both roles.
  /// Mirrors `ompl::base::PathLengthDirectInfSampler`'s `keepSample`. Costs
  /// lifts-squared `getPathLength` calls per attempt.
  bool acceptFoldedSample(const unsigned int dim, const double c_best) {
    if (phs_.size() == 1) return true;
    Eigen::Map<const Eigen::VectorXd> x(orig_buf_.data(), dim);
    int count = 0;
    for (const auto& offset : lift_offsets_) {
      lift_probe_.noalias() = Lt_ * (x + offset);
      for (const auto& phs : phs_) {
        if (costAtMost(phs->getPathLength(lift_probe_.data()), c_best)) ++count;
      }
    }
    if (count <= 1) return count == 1;
    return ldUniform01() * count < 1.0;
  }

  /// @brief A pseudo-random variate in [0, 1) for scalar control decisions.
  double ldUniform01() { return geodex::utils::uniform01(scalar_rng_); }

  /// @brief Fill out[0..dim-1] with one low-discrepancy point in the box [lo, hi).
  void ldUniformInBox(const unsigned int dim, const double* lo, const double* hi, double* out) {
    ld_cube_.resize(static_cast<int>(dim));
    ld_.sample(static_cast<int>(dim), ld_cube_);
    for (unsigned int d = 0; d < dim; ++d) out[d] = lo[d] + (hi[d] - lo[d]) * ld_cube_[d];
  }

  /// @brief A low-discrepancy point uniform in the unit `dim`-ball, direction from
  /// inverse-normal Gaussians and radius U^{1/dim}, matching OMPL's uniformInBall.
  void ldUniformInBall(const unsigned int dim, double* out) {
    ld_cube_.resize(static_cast<int>(dim) + 1);
    ld_.sample(static_cast<int>(dim) + 1, ld_cube_);
    double norm_sq = 0.0;
    for (unsigned int d = 0; d < dim; ++d) {
      const double g = geodex::utils::normal_quantile(ld_cube_[d]);
      out[d] = g;
      norm_sq += g * g;
    }
    const double radius = std::pow(std::clamp(ld_cube_[dim], 0.0, 1.0), 1.0 / dim);
    const double scale = (norm_sq > 1e-300) ? radius / std::sqrt(norm_sq) : 0.0;
    for (unsigned int d = 0; d < dim; ++d) out[d] *= scale;
  }

  /// @brief A low-discrepancy hyperspheroid point, a unit-ball sample mapped through
  /// the PHS transform.
  void ldUniformPHS(const std::shared_ptr<::ompl::ProlateHyperspheroid>& phs, double* out) {
    const unsigned int dim = phs->getDimension();
    ld_ball_.resize(dim);
    ldUniformInBall(dim, ld_ball_.data());
    phs->transform(ld_ball_.data(), out);
  }

  bool sampleUniformWithAttempts(ob::State* statePtr, const ob::Cost& effective,
                                 const unsigned int maxAttempts) {
    if (!direct_) {
      if (std::isinf(effective.value())) {
        baseSampler_->sampleUniform(statePtr);
        ++stats_.uniform_samples;
        return true;
      }
      return sampleRejection(statePtr, effective, maxAttempts);
    }
    // Volume-ratio fallback for every direct-sampling heuristic (Euclidean,
    // EigenvalueLB, MatrixLB). When the informed region's volume exceeds
    // volume_ratio_threshold_ times the C-space measure, most PHS samples land
    // outside the joint-limits box. Fall back to bounded-domain rejection, which for
    // MatrixLB is uniform rejection from the latent bounds parallelotope under
    // y = L^T x.
    if constexpr (kHasDirectSampling) {
      if (std::isfinite(effective.value())) {
        // The capped public measure cannot exceed the trigger. The check runs on
        // the uncapped union for MatrixLB.
        double check_volume;
        if constexpr (kIsMatrixLB) {
          check_volume = unionMeasure(effective.value());
        } else {
          check_volume = getInformedMeasure(effective);
        }
        const double space_measure = space_->getMeasure();

        // For MatrixLB the uncapped PHS can be enormous (stretched along
        // small-eigenvalue axes) while its AABB clipped to the coordinate bounds
        // stays small. Use the tighter of the two volumes. The check then does not
        // redirect to uniform while clipped-AABB sampling is still efficient.
        if constexpr (kIsMatrixLB) {
          if (phs_.size() == 1) {
            updateClippedAABB(effective.value());
            if (has_latent_bounds_ && clipped_aabb_volume_ > 0.0) {
              const double clipped_original = clipped_aabb_volume_ / std::sqrt(det_M_lower_);
              if (clipped_original < check_volume) check_volume = clipped_original;
            }
          }
        }

        stats_.last_volume_ratio = (space_measure > 0.0) ? check_volume / space_measure : 0.0;
        if (volume_ratio_threshold_ > 0.0 &&
            check_volume > volume_ratio_threshold_ * space_measure) {
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
  /// is finite, sample a uniform variate to decide whether to further narrow to
  /// the greedy bound. Bumps `stats_.focused_sample_count` on greedy hits.
  ob::Cost narrowCost(const ob::Cost& maxCost) {
    if (!feedback_) return maxCost;
    double effective = maxCost.value();
    // Narrow with HPC only when the planner has a finite cost bound. A stale HPC,
    // for example from a previous solve or from before the planner reports its
    // first solution, would give an informed region that is too tight.
    if (feedback_->narrow_to_heuristic_path_cost && !std::isinf(effective) &&
        std::isfinite(feedback_->heuristic_path_cost)) {
      effective = std::min(effective, feedback_->heuristic_path_cost);
    }
    if (feedback_->greedy_biasing_ratio > 0.0 && std::isfinite(feedback_->greedy_cost) &&
        ldUniform01() < feedback_->greedy_biasing_ratio) {
      effective = std::min(effective, feedback_->greedy_cost);
      ++stats_.focused_sample_count;
    }
    return ob::Cost(effective);
  }

  /// @brief Compute h(start, state) + h(state, goal).
  double heuristicCost(const ob::State* statePtr) const {
    space_->copyToReals(coords_buf_, statePtr);
    if constexpr (kIsEuclidean) {
      return phs_[0]->getPathLength(coords_buf_.data());
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
      ++stats_.uniform_samples;
      return true;
    }
    const double minTD = phs_[0]->getMinTransverseDiameter();
    if (costBelow(maxCost.value(), minTD)) {
      return false;
    }
    if (costAtMost(maxCost.value(), minTD)) {
      return sampleFocalSegment(statePtr, maxAttempts);
    }
    phs_[0]->setTransverseDiameter(maxCost.value());

    for (unsigned int i = 0; i < maxAttempts; ++i) {
      ++stats_.total_attempts;
      ldUniformPHS(phs_[0], coords_buf_.data());
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
      ++stats_.uniform_samples;
      return true;
    }
    double effective = maxCost.value() / sqrt_lambda_min_;
    const double minTD = phs_[0]->getMinTransverseDiameter();
    if (costBelow(effective, minTD)) {
      // Inadmissible heuristic, fall back to uniform.
      baseSampler_->sampleUniform(statePtr);
      ++stats_.uniform_samples;
      return true;
    }
    if (costAtMost(effective, minTD)) {
      return sampleFocalSegment(statePtr, maxAttempts);
    }
    phs_[0]->setTransverseDiameter(effective);

    for (unsigned int i = 0; i < maxAttempts; ++i) {
      ++stats_.total_attempts;
      ldUniformPHS(phs_[0], coords_buf_.data());
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
  /// @details Picks the latent PHS or the clipped latent AABB, whichever has the
  /// smaller sampling volume. The clipped AABB covers a single PHS only, and a
  /// multi-lift union always takes the PHS path.
  bool sampleMatrixLBLatent(ob::State* statePtr, const ob::Cost& maxCost,
                            const unsigned int maxAttempts) {
    // With the lift enumeration capped, a single PHS misses wrapped optima. Sample
    // by heuristic rejection, which is complete for the wrapped bound.
    if (lifts_capped_) return sampleRejection(statePtr, maxCost, maxAttempts);
    if (std::isinf(maxCost.value())) {
      baseSampler_->sampleUniform(statePtr);
      ++stats_.uniform_samples;
      return true;
    }
    const double minTD = min_td_;
    if (costBelow(maxCost.value(), minTD)) {
      // Inadmissible heuristic, fall back to uniform.
      baseSampler_->sampleUniform(statePtr);
      ++stats_.uniform_samples;
      return true;
    }
    const unsigned int dim = space_->getDimension();
    if (costAtMost(maxCost.value(), minTD)) {
      return sampleFocalSegment(statePtr, maxAttempts);
    }

    if (has_latent_bounds_ && phs_.size() == 1) {
      updateClippedAABB(maxCost.value());
      if (using_clipped_aabb_) {
        phs_[0]->setTransverseDiameter(maxCost.value());
        return sampleFromClippedAABB(statePtr, maxCost, dim, maxAttempts);
      }
    }
    return sampleFromPHSLatent(statePtr, dim, maxCost.value(), maxAttempts);
  }

  /// @brief Recompute the clipped AABB and decide which strategy to use.
  ///
  /// @details Intersects the latent PHS AABB with the latent bounds AABB and
  /// compares volumes. The result is cached against `last_c_best_`.
  void updateClippedAABB(double c_best) {
    // The clipped AABB brackets one PHS and cannot represent a lift union.
    if (!has_latent_bounds_ || phs_.size() > 1 || !std::isfinite(c_best)) {
      using_clipped_aabb_ = false;
      stats_.using_clipped_aabb = false;
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
      stats_.using_clipped_aabb = false;
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
        clipped_aabb_volume_ = 0.0;
        return;
      }
      clipped_vol *= extent;
    }
    const double phs_vol = phs_[0]->getPhsMeasure(c_best);
    using_clipped_aabb_ = (clipped_vol < phs_vol);
    stats_.using_clipped_aabb = using_clipped_aabb_;
    clipped_aabb_volume_ = clipped_vol;
  }

  /// @brief Sample the focal segment, the informed set at exactly the minimum cost.
  ///
  /// @details The segment belongs to the closest goal lift, which is the lift the
  /// heuristic realizes. Without lifts this is the plain start-to-goal segment.
  bool sampleFocalSegment(ob::State* statePtr, const unsigned int maxAttempts) {
    const unsigned int dim = space_->getDimension();
    for (unsigned int i = 0; i < maxAttempts; ++i) {
      ++stats_.total_attempts;
      const double t = ldUniform01();
      for (unsigned int d = 0; d < dim; ++d) {
        const double g =
            goal_coords_[d] + (lift_offsets_.empty() ? 0.0 : lift_offsets_[best_lift_][d]);
        orig_buf_[d] = (1.0 - t) * start_coords_[d] + t * g;
      }
      foldIntoDomain(dim);
      space_->copyFromReals(statePtr, orig_buf_);
      if (space_->satisfiesBounds(statePtr)) {
        ++stats_.accepted;
        return true;
      }
      ++stats_.bounds_rejections;
    }
    return false;
  }

  /// @brief Sample the union of latent lift PHS, reject for original-space bounds.
  ///
  /// @details Samples a lift with probability proportional to its measure, folds the
  /// back-transformed point into the fundamental domain, then de-biases the lift
  /// overlap. With one lift this is plain PHS sampling and the fold is a no-op.
  bool sampleFromPHSLatent(ob::State* statePtr, unsigned int dim, const double c_best,
                           const unsigned int maxAttempts) {
    const double total = activateLifts(c_best);
    if (n_active_ == 0) return false;

    for (unsigned int i = 0; i < maxAttempts; ++i) {
      ++stats_.total_attempts;
      ldUniformPHS(phs_[pickLift(total)], latent_buf_.data());
      Eigen::Map<const Eigen::VectorXd> y(latent_buf_.data(), dim);
      latent_eigen_buf_.noalias() = L_inv_t_ * y;
      Eigen::Map<Eigen::VectorXd>(orig_buf_.data(), dim) = latent_eigen_buf_;
      foldIntoDomain(dim);
      space_->copyFromReals(statePtr, orig_buf_);
      if (!space_->satisfiesBounds(statePtr)) {
        ++stats_.bounds_rejections;
        continue;
      }
      if (!acceptFoldedSample(dim, c_best)) continue;
      ++stats_.accepted;
      return true;
    }
    return false;
  }

  /// @brief Sample from the clipped AABB, reject for PHS membership and bounds.
  bool sampleFromClippedAABB(ob::State* statePtr, const ob::Cost& maxCost, unsigned int dim,
                             const unsigned int maxAttempts) {
    for (unsigned int i = 0; i < maxAttempts; ++i) {
      ++stats_.total_attempts;
      ldUniformInBox(dim, clipped_lo_.data(), clipped_hi_.data(), latent_buf_.data());
      const double path_len = phs_[0]->getPathLength(latent_buf_.data());
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
  /// @details On a new exact solution, `applyPathToFeedback` computes the heuristic path
  /// cost \f$ \sum_i h(p_i, p_{i+1}) \f$ and the greedy cost \f$ \max_p [h(s,p) + h(p,g)] \f$
  /// and writes them to the feedback channel. The callback wrapper installed in the
  /// constructor calls the same function. Without a feedback channel it does nothing.
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
    applyPathToFeedback(path->getStates(), path->cost(opt_).value());
  }

  /// @brief Compute heuristic-path-cost and greedy-cost from a solution path
  /// and write them to the shared feedback channel. Templated on the state
  /// pointer to accept both `std::vector<State*>` (PathGeometric::getStates) and
  /// `std::vector<const State*>` (intermediateSolutionCallback).
  template <typename StatePtr>
  void applyPathToFeedback(const std::vector<StatePtr>& states, const double solution_cost) {
    if (!feedback_) return;
    if (states.size() < 2) return;

    const unsigned int dim = space_->getDimension();
    std::vector<double> a_buf(dim);
    std::vector<double> b_buf(dim);

    // Heuristic path cost, Σ h(p_i, p_{i+1}) over consecutive waypoints.
    double hpc = 0.0;
    space_->copyToReals(b_buf, states[0]);
    for (std::size_t i = 1; i < states.size(); ++i) {
      std::swap(a_buf, b_buf);
      space_->copyToReals(b_buf, states[i]);
      const Eigen::Map<const Eigen::VectorXd> a(a_buf.data(), dim);
      const Eigen::Map<const Eigen::VectorXd> b(b_buf.data(), dim);
      hpc += heuristic_(a, b);
    }

    // Greedy cost, max_p [h(s, p) + h(p, g)] over all waypoints.
    const Eigen::Map<const Eigen::VectorXd> s(start_coords_.data(), dim);
    const Eigen::Map<const Eigen::VectorXd> g(goal_coords_.data(), dim);
    double gc = -std::numeric_limits<double>::infinity();
    for (const auto* st : states) {
      space_->copyToReals(a_buf, st);
      const Eigen::Map<const Eigen::VectorXd> p(a_buf.data(), dim);
      const double cost = heuristic_(s, p) + heuristic_(p, g);
      if (cost > gc) gc = cost;
    }

    const double cost = std::isfinite(solution_cost) ? solution_cost : hpc;
    // Only a cheaper path moves the bounds, and they never loosen. Compare on the
    // path cost. A better path can have a larger `hpc`.
    if (cost >= last_solution_cost_) return;

    last_solution_cost_ = cost;
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
  SamplerT ld_;                  ///< spatial samples (hyperspheroid, clipped AABB)
  std::mt19937_64 scalar_rng_;   ///< scalar control decisions
  Eigen::VectorXd ld_cube_;      ///< reused unit-cube sample for the ld_ helpers
  std::vector<double> ld_ball_;  ///< reused unit-ball point for hyperspheroid samples
  double volume_ratio_threshold_ = kDefaultVolumeRatioThreshold;
  bool direct_ = kHasDirectSampling;  ///< see setDirectSampling

  /// Deck-group lifts of the informed set. `phs_[0]` is the base lift, and every
  /// other entry is the same PHS with the goal translated by `lift_offsets_[k]`.
  /// Size 1 whenever no coordinate is periodic.
  std::vector<std::shared_ptr<::ompl::ProlateHyperspheroid>> phs_;
  std::vector<Eigen::VectorXd> lift_offsets_;  ///< coordinate offsets, [0] is zero
  Eigen::VectorXd wrap_periods_;               ///< per-axis period, empty when aperiodic
  Eigen::VectorXd wrap_center_;                ///< fundamental-domain center
  std::vector<double> lift_cdf_;               ///< cumulative measures, flat when unreachable
  std::size_t n_active_ = 0;                   ///< lifts reachable at the current cost
  std::size_t first_active_ = 0;               ///< index of the first reachable lift
  Eigen::VectorXd lift_probe_;                 ///< reused latent probe for the overlap count
  double min_td_ = 0.0;                        ///< minimum focal separation over lifts
  std::size_t best_lift_ = 0;                  ///< lift with the minimum focal separation

  /// Cap on periodic axes for lift enumeration (3^n lifts). Past this the
  /// sampler degrades to heuristic rejection, which stays complete.
  static constexpr std::size_t kMaxPeriodicAxes = 4;
  bool lifts_capped_ = false;

  // EigenvalueLB-only.
  double sqrt_lambda_min_ = 1.0;

  // MatrixLB only, latent-space transforms.
  Eigen::MatrixXd Lt_;
  Eigen::MatrixXd L_inv_t_;
  double det_M_lower_ = 1.0;
  std::vector<double> latent_start_;
  std::vector<double> latent_goal_;

  // MatrixLB only, clipped-AABB strategy state.
  bool has_latent_bounds_ = false;
  Eigen::VectorXd latent_bounds_lo_;
  Eigen::VectorXd latent_bounds_hi_;
  Eigen::VectorXd clipped_lo_;
  Eigen::VectorXd clipped_hi_;
  bool using_clipped_aabb_ = true;
  double clipped_aabb_volume_ = 0.0;  ///< Latent volume of the latest clipped AABB.
  double last_c_best_ = -1.0;

  // Buffers reused across sampleUniform calls. The hot path does not allocate.
  // The sampler assumes a single-threaded planner.
  mutable std::vector<double> coords_buf_;
  std::vector<double> latent_buf_;
  std::vector<double> orig_buf_;
  Eigen::VectorXd latent_eigen_buf_;

  /// Cost of the solution the published bounds came from.
  double last_solution_cost_ = std::numeric_limits<double>::infinity();

  // Self-refresh state, see maybeRefreshFromSolution().
  std::size_t last_seen_solution_count_ = 0;

  // Sentinel shared with the chained intermediate-solution callback wrapper on
  // pdef. The dtor sets it to false, and a wrapper that outlives the sampler
  // becomes a passthrough.
  std::shared_ptr<bool> sampler_alive_ = std::make_shared<bool>(true);

  SamplingStats stats_{};
};

}  // namespace geodex::integration::ompl
