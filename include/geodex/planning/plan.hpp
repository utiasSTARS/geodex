/// @file plan.hpp
/// @brief Motion planning on a geodex manifold with OMPL.
///
/// @details `plan()` sets up the OMPL problem, runs the planner and smooths the path.
/// Requires OMPL. The `geodex.hpp` umbrella header does not include it.

#pragma once

#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <concepts>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <ompl/base/MotionValidator.h>
#include <ompl/base/Planner.h>
#include <ompl/base/PlannerTerminationCondition.h>
#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorBounds.h>
#include <ompl/base/terminationconditions/IterationTerminationCondition.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/geometric/planners/rrt/GreedyRRTstar.h>
#include <ompl/util/Console.h>
#include <ompl/util/RandomNumbers.h>

#include "geodex/algorithm/path_smoothing.hpp"
#include "geodex/algorithm/precompute_matrix_lower_bound.hpp"
#include "geodex/core/concepts.hpp"
#include "geodex/core/metric.hpp"
#include "geodex/core/sampler.hpp"
#include "geodex/heuristics/euclidean.hpp"
#include "geodex/heuristics/matrix_lower_bound.hpp"
#include "geodex/integration/ompl/directional_motion_validator.hpp"
#include "geodex/integration/ompl/geodex_optimization_objective.hpp"
#include "geodex/integration/ompl/geodex_state_space.hpp"
#include "geodex/integration/ompl/validity_checker.hpp"

namespace geodex::planning {

namespace ob = ::ompl::base;
namespace og = ::ompl::geometric;

using geodex::integration::ompl::InterpolationMode;

/// @brief Settings of the planner.
namespace planners {

/// @brief Asymptotically optimal informed planner (G-RRT*).
struct GreedyRRTstar {
  double range = 0.0;                        ///< step size, 0 lets OMPL choose
  double greedy_ratio = 0.9;                 ///< fraction of samples from the greedy ellipsoid
  double rewire_factor = 1.1;                ///< rewiring radius scale
  bool greedy_cost_for_tree_pruning = true;  ///< prune to the greedy set when greedy_ratio > 0
  unsigned int max_neighbors = 0;            ///< cap on the k-nearest neighborhood, 0 = none
};

}  // namespace planners

/// @brief How much the planners of `plan()` print.
enum class LogLevel { Debug, Info, Warn, Error, Off };

namespace detail {

/// @brief The level that the environment variable `GEODEX_LOG_LEVEL` names (debug, info, warn,
/// error or off, in any case), `Warn` when it is unset or unknown.
inline LogLevel environment_log_level() {
  const char* value = std::getenv("GEODEX_LOG_LEVEL");
  std::string name = value == nullptr ? "" : value;
  for (char& c : name) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (name == "debug") return LogLevel::Debug;
  if (name == "info") return LogLevel::Info;
  if (name == "error") return LogLevel::Error;
  if (name == "off") return LogLevel::Off;
  return LogLevel::Warn;
}

inline std::atomic<LogLevel>& log_level_storage() {
  static std::atomic<LogLevel> level{environment_log_level()};
  return level;
}

inline ::ompl::msg::LogLevel to_ompl(const LogLevel level) {
  switch (level) {
    case LogLevel::Debug:
      return ::ompl::msg::LOG_DEBUG;
    case LogLevel::Info:
      return ::ompl::msg::LOG_INFO;
    case LogLevel::Warn:
      return ::ompl::msg::LOG_WARN;
    case LogLevel::Error:
      return ::ompl::msg::LOG_ERROR;
    case LogLevel::Off:
      return ::ompl::msg::LOG_NONE;
  }
  return ::ompl::msg::LOG_WARN;
}

/// @brief OMPL's log level, shared by the running plans.
struct OmplLogState {
  std::mutex mutex;
  int plans = 0;                                        ///< running plans
  ::ompl::msg::LogLevel saved = ::ompl::msg::LOG_WARN;  ///< OMPL's level before they started
};

inline OmplLogState& ompl_log_state() {
  static OmplLogState state;
  return state;
}

/// @brief Sets OMPL's log level while a plan runs. The last plan to finish restores it.
class ScopedOmplLogLevel {
 public:
  explicit ScopedOmplLogLevel(const LogLevel level) {
    OmplLogState& state = ompl_log_state();
    const std::lock_guard lock(state.mutex);
    if (state.plans++ == 0) state.saved = ::ompl::msg::getLogLevel();
    ::ompl::msg::setLogLevel(to_ompl(level));
  }
  ScopedOmplLogLevel(const ScopedOmplLogLevel&) = delete;
  ScopedOmplLogLevel& operator=(const ScopedOmplLogLevel&) = delete;
  ~ScopedOmplLogLevel() {
    OmplLogState& state = ompl_log_state();
    const std::lock_guard lock(state.mutex);
    if (--state.plans == 0) ::ompl::msg::setLogLevel(state.saved);
  }
};

}  // namespace detail

/// @brief Set how much `plan()` prints. The default, `Warn`, prints warnings and errors. The
/// environment variable `GEODEX_LOG_LEVEL` (debug, info, warn, error or off, in any case) sets
/// the starting level. At `Info`, a plan that runs its planner prints its times.
inline void set_log_level(const LogLevel level) { detail::log_level_storage().store(level); }

/// @brief The log level of `plan()`.
inline LogLevel log_level() { return detail::log_level_storage().load(); }

/// @brief Settings of `plan()`. The defaults solve most problems.
struct PlanSettings {
  double time = 1.0;            ///< time budget in seconds
  unsigned int iterations = 0;  ///< iteration budget, 0 uses the time budget

  /// @brief Seconds of refinement after the first solution, within `time`. 0 refines for
  /// the whole budget. Ignored when `iterations` is set.
  double refine_time = 0.0;
  planners::GreedyRRTstar planner{};  ///< parameters of G-RRT*

  /// @brief Spacing of the edge checks of the planner and the smoother, in coordinate
  /// distance. 0 uses OMPL's default for the planner, and for the smoother the one in
  /// `smoothing` or, when that is 0 too, one hundredth of the bounds' diagonal. With
  /// `smoothing.edge_travel`, the smoother keeps `smoothing.collision_check_resolution`.
  /// @throws std::invalid_argument from `plan()` when negative, not finite, or so fine
  /// that an edge across the bounds needs more than `algorithm::kMaxEdgeSamples` checks.
  double collision_check_resolution = 0.0;

  /// @brief Curve the planner follows between two states. `BaseGeodesic`, the default, uses
  /// `manifold.geodesic()`. `RiemannianGeodesic` uses the discrete geodesic of the metric.
  /// `Auto` uses `manifold.geodesic()` when the manifold's `log` is the Riemannian logarithm of
  /// its metric or a motion validator is installed, and the discrete geodesic otherwise.
  InterpolationMode interp = InterpolationMode::BaseGeodesic;

  /// @brief Distance to the goal that counts as reaching it. A start this close plans the
  /// single edge to the goal.
  double goal_tolerance = 0.0;

  /// @brief Coordinate limits as (lower, upper), such as joint limits. The planner and the
  /// smoother stay inside them. Without limits, the plan uses the bounds the manifold
  /// declares, if any. Limits that are not finite throw `std::invalid_argument`.
  std::optional<std::pair<Eigen::VectorXd, Eigen::VectorXd>> limits;

  /// @brief Seed of the plan. A nonzero seed gives the same plan on every run under an
  /// `iterations` budget. 0 takes a seed from the manifold's own sampler and advances it.
  /// Successive unseeded plans differ. Run concurrent unseeded plans on copies of the
  /// manifold.
  std::uint64_t seed = 0;
  bool smooth = true;  ///< run `smooth_path` on the planner's path

  /// @brief Settings of `smooth_path`. A nonzero `collision_check_resolution` above overrides
  /// the one inside, except with `edge_travel`, whose resolution must then be positive.
  geodex::algorithm::PathSmoothingSettings smoothing{};
};

/// @brief Result of `plan()`.
///
/// @details When `smoothed` is set, `path` is the smoother's output with its edges
/// interpolated by `manifold.geodesic()`. Otherwise `path` follows the planner's path,
/// densified along the planner's curve when that differs from `manifold.geodesic()`.
template <typename PointT>
struct PlanResult {
  bool solved = false;           ///< an exact solution was found
  bool smoothed = false;         ///< `path` is the smoother's output
  std::vector<PointT> path;      ///< returned path
  std::vector<PointT> raw_path;  ///< planner waypoints before smoothing
  double cost = std::numeric_limits<double>::infinity();  ///< metric length of `path`
  double time_ms = 0.0;                                   ///< planning time
  double smooth_ms = 0.0;                                 ///< smoothing time

  /// @brief Time from the start of the search to the first exact solution, -1 without one.
  /// The rest of `time_ms` refines that solution.
  double first_solution_ms = -1.0;

  /// @brief Termination checks before the first exact solution, counted like
  /// `PlanSettings::iterations`, 0 without one.
  std::uint64_t first_solution_iterations = 0;

  /// @name Informed sampling of G-RRT*
  /// @{
  std::uint64_t informed_samples = 0;  ///< samples that passed an informed test
  std::uint64_t focused_samples = 0;   ///< samples aimed at the tighter greedy set
  std::uint64_t uniform_samples = 0;   ///< uniform samples without an informed test
  /// @}
};

namespace detail {

/// @brief Halton points that estimate the sampled region of a manifold without bounds.
inline constexpr int kBoundsSamples = 512;

/// @brief Relative pad on each side of the estimated region.
inline constexpr double kBoundsPad = 0.02;

/// @brief Smallest pad.
inline constexpr double kBoundsMinPad = 1e-6;

/// @brief Default edge-check spacing of the smoother, per bounds diagonal.
inline constexpr double kSmoothingResolutionFraction = 0.01;

/// @brief Axis-aligned bounds of the planning problem, holding the start and the goal.
template <typename ManifoldT>
ob::RealVectorBounds derive_bounds(
    const ManifoldT& manifold, const typename ManifoldT::Point& start,
    const typename ManifoldT::Point& goal,
    const std::optional<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& limits = std::nullopt) {
  // Given or declared limits are used as they are.
  auto from_limits = [&](const Eigen::VectorXd& blo, const Eigen::VectorXd& bhi) {
    const int amb = static_cast<int>(start.size());
    if (blo.size() != amb || bhi.size() != amb) {
      throw std::invalid_argument("plan: limits must match the manifold ambient size");
    }
    ob::RealVectorBounds bounds(amb);
    for (int i = 0; i < amb; ++i) {
      bounds.setLow(
          i, std::min({blo[i], static_cast<double>(start[i]), static_cast<double>(goal[i])}));
      bounds.setHigh(
          i, std::max({bhi[i], static_cast<double>(start[i]), static_cast<double>(goal[i])}));
    }
    return bounds;
  };
  if (limits) return from_limits(limits->first, limits->second);
  if constexpr (requires(const ManifoldT& m) {
                  m.has_bounds();
                  m.bounds();
                }) {
    if (manifold.has_bounds()) {
      const auto [blo, bhi] = manifold.bounds();
      return from_limits(blo, bhi);
    }
  }

  const int cube_dim = manifold.unit_cube_dim();
  geodex::HaltonSampler sampler;
  Eigen::VectorXd cube(cube_dim);
  sampler.sample(cube_dim, cube);
  auto first = manifold.from_unit_cube(cube);
  const int amb = static_cast<int>(first.size());

  Eigen::VectorXd lo = first;
  Eigen::VectorXd hi = first;
  for (int k = 1; k < kBoundsSamples; ++k) {
    sampler.sample(cube_dim, cube);
    auto p = manifold.from_unit_cube(cube);
    for (int i = 0; i < amb; ++i) {
      lo[i] = std::min(lo[i], static_cast<double>(p[i]));
      hi[i] = std::max(hi[i], static_cast<double>(p[i]));
    }
  }

  ob::RealVectorBounds bounds(amb);
  for (int i = 0; i < amb; ++i) {
    double lo_i = std::min({lo[i], static_cast<double>(start[i]), static_cast<double>(goal[i])});
    double hi_i = std::max({hi[i], static_cast<double>(start[i]), static_cast<double>(goal[i])});
    const double pad = std::max(kBoundsPad * (hi_i - lo_i), kBoundsMinPad);
    bounds.setLow(i, lo_i - pad);
    bounds.setHigh(i, hi_i + pad);
  }
  return bounds;
}

/// @brief A copy of the manifold with its sampling box set to `limits`, when the manifold has
/// a sampling box of the limits' size. Other manifolds are returned unchanged.
template <typename ManifoldT>
ManifoldT sampling_within(
    const ManifoldT& manifold,
    const std::optional<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& limits) {
  ManifoldT out = manifold;
  if constexpr (requires(ManifoldT& m, const Eigen::VectorXd& v) {
                  m.set_sampling_bounds(v, v);
                  manifold.lo().size();
                }) {
    if (limits && manifold.lo().size() == limits->first.size()) {
      out.set_sampling_bounds(limits->first, limits->second);
    }
  }
  return out;
}

/// @brief Metric length of a path.
template <typename ManifoldT>
double path_cost(const ManifoldT& manifold,
                 const std::vector<typename ManifoldT::Point>& path) {
  double c = 0.0;
  for (std::size_t i = 1; i < path.size(); ++i) {
    c += static_cast<double>(manifold.distance(path[i - 1], path[i]));
  }
  return c;
}

/// @brief Seed of a plan. 0 takes one from the manifold's sampler through one
/// `random_point()`.
template <typename ManifoldT>
std::uint64_t stream_seed(const ManifoldT& manifold, const std::uint64_t seed) {
  if (seed != 0) return seed;
  const auto point = manifold.random_point();
  std::uint64_t h = 0x9E3779B97F4A7C15ULL;
  for (Eigen::Index i = 0; i < point.size(); ++i) {
    // splitmix64 over the coordinates' bits
    h ^= std::bit_cast<std::uint64_t>(static_cast<double>(point[i]));
    h += 0x9E3779B97F4A7C15ULL;
    h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ULL;
    h = (h ^ (h >> 27)) * 0x94D049BB133111EBULL;
    h ^= h >> 31;
  }
  return h;
}

/// @brief The bound as a heuristic that wraps at the periodic axes. A bound that couples a
/// periodic axis shrinks to its smallest eigenvalue times the identity.
inline geodex::heuristics::MatrixLowerBound<Eigen::Dynamic> wrapped_heuristic(
    const geodex::algorithm::PrecomputeMatrixLowerBoundResult& bound) {
  try {
    return bound.heuristic();
  } catch (const std::invalid_argument&) {
    const Eigen::Index n = bound.M_lower.rows();
    const double lambda =
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(bound.M_lower, Eigen::EigenvaluesOnly)
            .eigenvalues()
            .minCoeff();
    return {lambda * Eigen::MatrixXd::Identity(n, n), bound.periods};
  }
}

/// @brief Seed OMPL's process-wide generator, which seeds every OMPL RNG. The caller holds
/// `ompl_rng_mutex()`. OMPL's warning about reseeding is muted, and a seed of 0 becomes 1.
inline void seed_ompl(const std::uint64_t seed) {
  const auto folded = static_cast<std::uint_least32_t>(seed ^ (seed >> 32));
  OmplLogState& state = ompl_log_state();
  const std::lock_guard lock(state.mutex);
  const ::ompl::msg::LogLevel level = ::ompl::msg::getLogLevel();
  ::ompl::msg::setLogLevel(::ompl::msg::LOG_NONE);
  ::ompl::RNG::setSeed(folded != 0 ? folded : 1);
  ::ompl::msg::setLogLevel(level);
}

/// @brief Point validity that also rejects points outside the limits. Keeps the batched
/// form of the wrapped validity.
template <typename Point, typename ValidityT>
class BoundedValidity {
 public:
  /// @param valid The caller's validity, held by reference.
  /// @param bounds Limits, checked only when `check_bounds` is set.
  BoundedValidity(const ValidityT& valid, const ob::RealVectorBounds& bounds,
                  const bool check_bounds)
      : valid_(valid), bounds_(bounds), check_bounds_(check_bounds) {}

  /// @brief Inside the limits and valid.
  bool operator()(const Point& q) const { return inside(q) && static_cast<bool>(valid_(q)); }

  /// @brief All `n` points inside the limits and valid.
  bool batch(const Point* q, const std::size_t n) const
    requires geodex::algorithm::BatchValidity<ValidityT, Point>
  {
    for (std::size_t i = 0; i < n; ++i) {
      if (!inside(q[i])) return false;
    }
    return static_cast<bool>(valid_.batch(q, n));
  }

  /// @brief Block size of the wrapped validity.
  std::size_t batch_size() const
    requires geodex::algorithm::BatchValidity<ValidityT, Point>
  {
    return static_cast<std::size_t>(valid_.batch_size());
  }

 private:
  bool inside(const Point& q) const {
    if (!check_bounds_) return true;
    for (unsigned int k = 0; k < bounds_.low.size(); ++k) {
      const double v = static_cast<double>(q[static_cast<int>(k)]);
      if (v < bounds_.low[k] || v > bounds_.high[k]) return false;
    }
    return true;
  }

  const ValidityT& valid_;
  const ob::RealVectorBounds& bounds_;
  bool check_bounds_;
};

/// @brief Implementation of `plan()`.
template <typename ManifoldT, typename HeuristicT, typename ValidityT>
PlanResult<typename ManifoldT::Point> plan_impl(
    const ManifoldT& manifold, const typename ManifoldT::Point& start,
    const typename ManifoldT::Point& goal, const ValidityT& valid_fn, const PlanSettings& settings,
    const HeuristicT& heuristic,
    const std::function<std::shared_ptr<ompl::base::MotionValidator>(
        const ompl::base::SpaceInformationPtr&)>& motion_validator_factory) {
  using Point = typename ManifoldT::Point;
  using StateSpace = geodex::integration::ompl::GeodexStateSpace<ManifoldT>;
  using StateType = geodex::integration::ompl::GeodexState<ManifoldT>;

  static_assert(geodex::SeedableSampler<typename ManifoldT::SamplerType>,
                "plan() needs a manifold sampler with seed()");

  const ScopedOmplLogLevel quiet(log_level());
  PlanResult<Point> result;

  const int ambient = static_cast<int>(
      manifold.from_unit_cube(Eigen::VectorXd::Constant(manifold.unit_cube_dim(), 0.5)).size());
  if (static_cast<int>(start.size()) != ambient || static_cast<int>(goal.size()) != ambient) {
    throw std::invalid_argument("plan: start and goal must match the manifold ambient size");
  }
  if (!std::isfinite(settings.collision_check_resolution) ||
      settings.collision_check_resolution < 0.0) {
    throw std::invalid_argument("plan: collision_check_resolution must be finite and >= 0");
  }
  if (!std::isfinite(settings.smoothing.collision_check_resolution) ||
      settings.smoothing.collision_check_resolution < 0.0) {
    throw std::invalid_argument(
        "plan: smoothing.collision_check_resolution must be finite and >= 0");
  }

  if (settings.limits &&
      (!settings.limits->first.allFinite() || !settings.limits->second.allFinite())) {
    throw std::invalid_argument("plan: limits must be finite");
  }

  // Every random choice of the plan derives from this seed.
  const std::uint64_t seed = detail::stream_seed(manifold, settings.seed);
  const ob::RealVectorBounds bounds = detail::derive_bounds(manifold, start, goal, settings.limits);
  const int amb = ambient;
  double diag2 = 0.0;
  for (int i = 0; i < amb; ++i) {
    const double w = bounds.high[i] - bounds.low[i];
    diag2 += w * w;
  }
  const double diag = std::sqrt(diag2);
  if (settings.collision_check_resolution > 0.0 &&
      diag / settings.collision_check_resolution > geodex::algorithm::kMaxEdgeSamples) {
    throw std::invalid_argument(
        "plan: collision_check_resolution asks for more than kMaxEdgeSamples checks on an edge "
        "across the planning bounds");
  }

  // The planner samples inside the limits, and a derived heuristic bound holds there.
  const ManifoldT sampled = detail::sampling_within(manifold, settings.limits);
  auto space = std::make_shared<StateSpace>(sampled, bounds);
  space->setSamplerSeed(seed);
  const InterpolationMode interp =
      settings.interp == InterpolationMode::Auto && motion_validator_factory
          ? InterpolationMode::BaseGeodesic
          : settings.interp;
  space->setInterpolationMode(interp);
  space->setCollisionResolution(settings.collision_check_resolution);

  // Leaving given or declared limits is invalid. Estimated sampling bounds are not limits.
  bool physical_bounds = settings.limits.has_value();
  if constexpr (requires(const ManifoldT& m) { m.has_bounds(); }) {
    physical_bounds = physical_bounds || manifold.has_bounds();
  }
  const BoundedValidity<Point, ValidityT> bounded_valid(valid_fn, bounds, physical_bounds);

  auto si = std::make_shared<ob::SpaceInformation>(space);
  si->setStateValidityChecker(geodex::integration::ompl::make_validity_checker<ManifoldT>(
      si, [&bounded_valid](const auto& q) { return bounded_valid(Point(q)); }));

  // Smoother settings.
  geodex::algorithm::PathSmoothingSettings smooth = settings.smoothing;
  if (smooth.edge_travel) {
    // The smoother's resolution is in the units of the edge travel bound.
    if (!(smooth.collision_check_resolution > 0.0)) {
      throw std::invalid_argument(
          "plan: smoothing.edge_travel needs a positive smoothing.collision_check_resolution");
    }
  } else if (settings.collision_check_resolution > 0.0) {
    smooth.collision_check_resolution = settings.collision_check_resolution;
  } else if (smooth.collision_check_resolution == 0.0) {
    smooth.collision_check_resolution = kSmoothingResolutionFraction * diag;
  }
  // Without batched point checks, a motion validator decides the smoother's edges too.
  if constexpr (!geodex::algorithm::BatchValidity<ValidityT, Point>) {
    if (motion_validator_factory && !smooth.edge_validator) {
      smooth.edge_validator = [&](const Eigen::Ref<const Eigen::VectorXd>& p,
                                  const Eigen::Ref<const Eigen::VectorXd>& q) {
        ob::ScopedState<StateSpace> a(space);
        ob::ScopedState<StateSpace> b(space);
        for (int k = 0; k < amb; ++k) {
          a->values[k] = p[k];
          b->values[k] = q[k];
        }
        return si->checkMotion(a.get(), b.get());
      };
    }
  }

  // A manifold that certifies its own Loewner bound replaces the default Euclidean
  // heuristic with it. An explicit heuristic is kept.
  constexpr bool kDerivesBound =
      std::is_same_v<HeuristicT, geodex::heuristics::Euclidean> &&
      geodex::algorithm::CertifiesOwnMatrixLowerBound<ManifoldT>;
  using Heuristic = std::conditional_t<kDerivesBound,
                                       geodex::heuristics::MatrixLowerBound<Eigen::Dynamic>,
                                       HeuristicT>;
  auto make_heuristic = [&]() -> Heuristic {
    if constexpr (kDerivesBound) {
      return detail::wrapped_heuristic(geodex::algorithm::precompute_matrix_lower_bound(sampled));
    } else {
      return heuristic;
    }
  };

  // Compute the bound before taking the lock.
  Heuristic planner_heuristic = make_heuristic();

  // Seed OMPL's generator and create the OMPL objects under one lock.
  std::unique_lock ompl_rngs(geodex::integration::ompl::detail::ompl_rng_mutex());
  detail::seed_ompl(seed);

  // Install the motion validator, if any.
  if (motion_validator_factory) si->setMotionValidator(motion_validator_factory(si));
  si->setup();

  // A start within the goal tolerance plans the single edge to the goal, checked like a
  // smoother edge.
  const double start_goal_dist = manifold.distance(start, goal);
  if (start_goal_dist <= settings.goal_tolerance) {
    ompl_rngs.unlock();
    space->setInterpolationMode(InterpolationMode::BaseGeodesic);
    geodex::algorithm::PathSmoothingProfile profile;
    const geodex::algorithm::detail::Oracle<ManifoldT, BoundedValidity<Point, ValidityT>> oracle(
        manifold, bounded_valid, smooth, smooth.collision_check_resolution, profile);
    if (oracle.point(start) && oracle.edge(start, goal, true)) {
      result.solved = true;
      result.first_solution_ms = 0.0;
      result.raw_path = {start, goal};
      result.path = {start, goal};
      result.cost = start_goal_dist;
    }
    return result;
  }

  auto pdef = std::make_shared<ob::ProblemDefinition>(si);
  ob::ScopedState<StateSpace> start_state(space);
  ob::ScopedState<StateSpace> goal_state(space);
  for (int i = 0; i < amb; ++i) {
    start_state->values[i] = start[i];
    goal_state->values[i] = goal[i];
  }
  pdef->setStartAndGoalStates(start_state, goal_state, settings.goal_tolerance);

  const planners::GreedyRRTstar& cfg = settings.planner;
  auto objective =
      std::make_shared<geodex::integration::ompl::GeodexOptimizationObjective<ManifoldT, Heuristic>>(
          si, goal, std::move(planner_heuristic));
  objective->setGreedyBiasingRatio(cfg.greedy_ratio);
  pdef->setOptimizationObjective(objective);

  auto planner = std::make_shared<og::GreedyRRTstar>(si);
  if (cfg.range > 0.0) planner->setRange(cfg.range);
  planner->setRewireFactor(cfg.rewire_factor);
  planner->setMaxNeighbors(cfg.max_neighbors);
  // Pruning to the greedy bound belongs to greedy sampling. A greedy ratio of 0 prunes to the
  // informed set of the solution cost.
  planner->setGreedyCostForTreePruning(cfg.greedy_cost_for_tree_pruning && cfg.greedy_ratio > 0.0);
  // The sampler applies the greedy ratio.
  planner->setGreedyBiasingRatio(0.0);
  // Direct informed sampling needs intrinsic coordinates.
  planner->setInformedSampling(amb == manifold.dim());
  planner->setProblemDefinition(pdef);
  planner->setup();
  ompl_rngs.unlock();

  using Clock = std::chrono::steady_clock;
  const auto t0 = Clock::now();
  // The termination condition records the first exact solution through G-RRT*'s best cost.
  std::optional<Clock::time_point> first;
  std::uint64_t evaluations = 0;
  auto observe = [&](const Clock::time_point now) {
    ++evaluations;
    if (!first && std::isfinite(planner->bestCost().value())) {
      first = now;
      result.first_solution_iterations = evaluations - 1;
    }
  };
  // An iteration budget makes a seeded plan reproducible.
  if (settings.iterations > 0) {
    ob::IterationTerminationCondition iterations(settings.iterations);
    planner->solve(ob::PlannerTerminationCondition([&] {
      observe(Clock::now());
      return iterations.eval();
    }));
  } else if (settings.refine_time > 0.0) {
    // Stop `refine_time` after the first solution or at the time budget.
    const auto budget = std::chrono::duration<double>(settings.time);
    const auto refine = std::chrono::duration<double>(settings.refine_time);
    planner->solve(ob::PlannerTerminationCondition([&] {
      const auto now = Clock::now();
      if (now - t0 >= budget) return true;
      observe(now);
      return first.has_value() && now - *first >= refine;
    }));
  } else {
    const ob::PlannerTerminationCondition timed =
        ob::timedPlannerTerminationCondition(settings.time);
    planner->solve(ob::PlannerTerminationCondition([&] {
      observe(Clock::now());
      return timed();
    }));
  }
  const auto t1 = Clock::now();
  result.time_ms = 1000.0 * std::chrono::duration<double>(t1 - t0).count();
  // A solution of the last iteration appears only after the search.
  if (!first && pdef->hasExactSolution()) {
    first = t1;
    result.first_solution_iterations = evaluations;
  }
  if (first) {
    result.first_solution_ms = 1000.0 * std::chrono::duration<double>(*first - t0).count();
  }
  const auto stats = objective->getSamplerStats();
  result.informed_samples = stats.accepted;
  result.focused_samples = stats.focused_sample_count;
  result.uniform_samples = stats.uniform_samples;

  if (!pdef->hasExactSolution()) {
    OMPL_INFORM("geodex plan: no exact solution, search %.1f ms", result.time_ms);
    return result;
  }
  auto geo = std::dynamic_pointer_cast<og::PathGeometric>(pdef->getSolutionPath());
  if (!geo || geo->getStateCount() < 2) return result;

  result.raw_path.reserve(geo->getStateCount());
  for (const auto* s : geo->getStates()) {
    result.raw_path.push_back(s->template as<StateType>()->asEigen());
  }
  result.solved = true;

  // When the planner followed a curve other than manifold.geodesic(), densify the path along
  // that curve.
  const bool planner_chords = interp == InterpolationMode::BaseGeodesic ||
                              (interp == InterpolationMode::Auto && is_riemannian_log(manifold));
  std::vector<Point> planned = result.raw_path;
  if (!planner_chords) {
    og::PathGeometric dense(*geo);
    dense.interpolate();
    planned.clear();
    planned.reserve(dense.getStateCount());
    for (const auto* s : dense.getStates()) {
      planned.push_back(s->template as<StateType>()->asEigen());
    }
  }

  std::vector<Point> final_path = planned;
  if (settings.smooth && planned.size() >= 2) {
    space->setInterpolationMode(InterpolationMode::BaseGeodesic);
    const auto t_smooth = std::chrono::steady_clock::now();
    auto smoothed = geodex::algorithm::smooth_path(manifold, bounded_valid, planned, smooth);
    result.smooth_ms =
        1000.0 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t_smooth).count();
    if (smoothed.collision_free && !smoothed.path.empty()) {
      result.smoothed = smoothed.profile.fallback < 3;
      final_path = std::move(smoothed.path);
    }
  }

  result.cost = path_cost(manifold, final_path);
  result.path = std::move(final_path);
  OMPL_INFORM(
      "geodex plan: first solution %.1f ms at %llu iterations, refinement %.1f ms, smoothing "
      "%.1f ms, %zu waypoints",
      result.first_solution_ms, static_cast<unsigned long long>(result.first_solution_iterations),
      result.time_ms - result.first_solution_ms, result.smooth_ms, result.path.size());
  return result;
}

}  // namespace detail

/// @brief Plan a collision-free path from start to goal on a manifold.
///
/// @param manifold Any geodex `RiemannianManifold`.
/// @param start Start point.
/// @param goal Goal point.
/// @param is_valid Returns true for a valid point. Empty means free space.
/// @param settings Plan settings.
/// @param heuristic Admissible heuristic of the informed planner. On a manifold that
///   certifies its own Loewner bound, such as SE(2), SO(2) or the torus, the default is
///   replaced by that bound. An explicit heuristic is kept.
/// @param motion_validator_factory Optional factory of an OMPL motion validator, such as
///   VAMP's edge check. It decides the planner's edges, and the smoother's edges when
///   `is_valid` does not check blocks of points.
///
/// @details A plan with a nonzero seed and an `iterations` budget gives the same result on
/// every run, also while other plans run in other threads.
///
/// @note Two limits come from OMPL. OMPL code that creates an `ompl::RNG` in another thread
/// while a plan sets up can change that plan. One seed can give different plans on Linux and
/// macOS when the manifold's distance breaks the triangle inequality, as on SE(2) with unequal
/// weights.
///
/// @code
/// auto result = geodex::planning::plan(manifold, start, goal, is_valid);
/// if (result.solved) use_path(result.path);
/// @endcode
template <typename ManifoldT, typename HeuristicT = geodex::heuristics::Euclidean>
PlanResult<typename ManifoldT::Point> plan(
    const ManifoldT& manifold, const typename ManifoldT::Point& start,
    const typename ManifoldT::Point& goal,
    const std::function<bool(const typename ManifoldT::Point&)>& is_valid,
    const PlanSettings& settings = {}, const HeuristicT& heuristic = {},
    const std::function<std::shared_ptr<ompl::base::MotionValidator>(
        const ompl::base::SpaceInformationPtr&)>& motion_validator_factory = {}) {
  using Point = typename ManifoldT::Point;
  if (is_valid) {
    return detail::plan_impl(manifold, start, goal, is_valid, settings, heuristic,
                             motion_validator_factory);
  }
  const std::function<bool(const Point&)> free_space = [](const Point&) { return true; };
  return detail::plan_impl(manifold, start, goal, free_space, settings, heuristic,
                           motion_validator_factory);
}

/// @brief `plan()` with a validity that also tests a block of points at once
/// (`algorithm::BatchValidity`), such as `integration::vamp::VampValidity`. The smoother
/// then checks edges in blocks.
template <typename ManifoldT, typename HeuristicT = geodex::heuristics::Euclidean,
          typename ValidityT>
  requires geodex::algorithm::BatchValidity<ValidityT, typename ManifoldT::Point> &&
           std::predicate<const ValidityT&, const typename ManifoldT::Point&>
PlanResult<typename ManifoldT::Point> plan(
    const ManifoldT& manifold, const typename ManifoldT::Point& start,
    const typename ManifoldT::Point& goal, const ValidityT& is_valid,
    const PlanSettings& settings = {}, const HeuristicT& heuristic = {},
    const std::function<std::shared_ptr<ompl::base::MotionValidator>(
        const ompl::base::SpaceInformationPtr&)>& motion_validator_factory = {}) {
  return detail::plan_impl(manifold, start, goal, is_valid, settings, heuristic,
                           motion_validator_factory);
}

}  // namespace geodex::planning
