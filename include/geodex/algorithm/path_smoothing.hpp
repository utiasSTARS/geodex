/// @file path_smoothing.hpp
/// @brief Metric-aware path smoothing with corner rounding.
///
/// @details `smooth_path` shortens a valid path under the manifold's metric by shortcutting
/// and local energy descent, then rounds its corners into C² curves. Every waypoint and every
/// edge of the smoothed path passes the validity check, and the returned waypoints lie at equal
/// steps along it.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <chrono>
#include <concepts>
#include <functional>
#include <numbers>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

#include "geodex/core/concepts.hpp"
#include "geodex/core/metric.hpp"
#include "geodex/utils/ordered_sum.hpp"
#include "geodex/utils/random.hpp"

namespace geodex::algorithm {

/// @brief Test of the edge between two configurations.
using EdgePredicate = std::function<bool(const Eigen::Ref<const Eigen::VectorXd>&,
                                         const Eigen::Ref<const Eigen::VectorXd>&)>;

/// @brief Length of the edge between two configurations.
using EdgeLength = std::function<double(const Eigen::Ref<const Eigen::VectorXd>&,
                                        const Eigen::Ref<const Eigen::VectorXd>&)>;

/// @brief Settings of `smooth_path`. The defaults work in meters and in radians.
struct PathSmoothingSettings {
  /// @brief Largest spacing of the validity checks along an edge, as the coordinate norm of
  /// `log(a, b)`, or in the units of `edge_travel` when that is set. 0 uses one hundredth of
  /// the input path's coordinate length and ignores `edge_travel`. A negative or
  /// non-finite value, or one that needs more than `kMaxEdgeSamples` checks on an edge,
  /// throws `std::invalid_argument`.
  double collision_check_resolution = 0.0;

  /// @brief Optional bound on how far the checked geometry moves along the edge from `a` to
  /// `b`, in the units of a positive `collision_check_resolution`. The checks then space the
  /// edge by this bound instead of the coordinate norm of `log(a, b)`. The bound must hold for
  /// every part of the edge in proportion to its share of the geodesic parameter, as a bound
  /// on the speed does. `robots::plan` and the Python `plan` with a robot scene set it to the
  /// travel of the robot's collision spheres. A negative or non-finite value throws
  /// `std::invalid_argument`.
  EdgeLength edge_travel;

  /// @brief Longest step between the returned waypoints, as the coordinate norm of `log`. 0
  /// leaves the step without a limit.
  ///
  /// @details The returned waypoints lie on the smoothed path at equal steps between the ends
  /// and the corners without a curve. The step is the longest one, up to this limit, whose
  /// edges stay within `corner_tolerance` of the smoothed path. The smoother checks the
  /// smoothed path. A validity margin as large as the motion over `corner_tolerance` covers
  /// the spaced one.
  double output_spacing = 0.0;

  /// @brief Seed of the random shortcuts.
  std::uint64_t seed = 42;

  /// @brief Optional proof that a whole edge is valid. True skips the edge's checks, and
  /// false leaves the decision to them. The proof must be sound.
  EdgePredicate edge_provably_clear;

  /// @brief Optional edge test that replaces the sampled checks, such as a planner's motion
  /// validator. The smoother tests edges in path order, and the test may be asymmetric. The
  /// evenly spaced edges of the returned path do not pass through it. Give an asymmetric
  /// constraint also as `path_predicate`.
  EdgePredicate edge_validator;

  /// @brief Optional test of the whole path. A change that fails it is rejected. When the
  /// evenly spaced path fails it, `smooth_path` returns the smoothed path's own waypoints.
  std::function<bool(const std::vector<Eigen::VectorXd>&)> path_predicate;

  /// @brief Round the corners of the smoothed path into C² curves.
  ///
  /// @details Each curve is a quintic Bézier curve in the tangent space at the corner, mapped
  /// to the manifold by `exp`. It leaves the incoming edge and joins the outgoing edge with
  /// their directions and zero acceleration. A curve is kept when it passes the edge test,
  /// adds at most 0.1 percent of metric length and stays in the plane of its two edges.
  /// Otherwise it shrinks by a factor of two, up to five times, and the corner stays when
  /// every size fails.
  bool round_corners = true;

  /// @brief Largest distance between a rounding curve and the edges between its samples, and
  /// between the smoothed path and the evenly spaced edges, as the coordinate norm of `log`. A
  /// value that is not positive and finite throws `std::invalid_argument`.
  double corner_tolerance = 1e-4;

  /// @brief Largest turning angle of a rounded corner in radians, under the metric. A sharper
  /// corner, such as a reversal, stays. A value outside [0, pi] throws
  /// `std::invalid_argument` when `round_corners` is on.
  double corner_max_angle = 0.5 * std::numbers::pi;

  /// @brief Number of leading tangent coordinates whose path may keep a corner, such as the
  /// pose of a differential-drive base before the joints of its arm.
  ///
  /// @details Where the curve through every coordinate stays smaller than its full size, and
  /// at a corner that turns more than `corner_max_angle`, the smoother also tries a curve that
  /// keeps the corner in these coordinates and rounds the others. These coordinates run along
  /// the two edges and turn at the corner, and the larger of the two curves stays. The
  /// manifold's `exp` and `log` must act on these coordinates apart from the others, as on a
  /// product manifold. 0 rounds every coordinate together, and so does a value that covers
  /// every coordinate. A negative value throws `std::invalid_argument`.
  int sharp_coordinates = 0;
};

/// @brief Time and work counters of one `smooth_path` call.
struct PathSmoothingProfile {
  double total_ms = 0.0;       ///< Whole call.
  double shortcut_ms = 0.0;    ///< Shortcutting.
  double descent_ms = 0.0;     ///< Subdivision and energy descent.
  double resample_ms = 0.0;    ///< Output resampling.
  double rounding_ms = 0.0;    ///< Corner rounding.
  double certify_ms = 0.0;     ///< Final check, fallbacks included.
  long point_checks = 0;       ///< Configurations checked.
  long batch_calls = 0;        ///< Batched validity calls.
  long edge_checks = 0;        ///< Edges tested.
  long edge_proofs = 0;        ///< Edges settled by `edge_provably_clear`.
  long shortcut_attempts = 0;  ///< Shortcuts tried.
  long shortcuts = 0;          ///< Shortcuts accepted.
  long relax_visits = 0;       ///< Waypoint visits of the energy descent.
  long relax_moves = 0;        ///< Waypoint moves that passed the edge checks.
  long rounded_corners = 0;    ///< Corners replaced by a curve.
  long cusps = 0;              ///< Corners that turn more than `corner_max_angle`.
  long kept_corners = 0;       ///< Corners that stay after every curve size failed a check.
  long split_corners = 0;      ///< Rounded corners that keep the corner in `sharp_coordinates`.
  long rounding_retries = 0;   ///< Curve sizes that failed a check.
  int input_waypoints = 0;     ///< Size of the input path.
  int output_waypoints = 0;    ///< Size of the returned path.

  /// @brief Returned stage. 0 is the smoothed path, 1 the optimized waypoints before
  /// resampling, 2 the first shortcut round and 3 the input.
  int fallback = 0;
};

/// @brief Result of `smooth_path`.
template <typename PointT>
struct PathSmoothingResult {
  /// @brief Value of `first_invalid_index` for a path that passes the check.
  static constexpr std::size_t npos = static_cast<std::size_t>(-1);

  std::vector<PointT> path;                ///< Returned path, endpoints kept.
  double length = 0.0;                     ///< Metric length of `path`.
  bool collision_free = false;             ///< The smoothed path passed the check.
  std::size_t first_invalid_index = npos;  ///< First waypoint that fails or starts a failing edge.
  PathSmoothingProfile profile;            ///< Time and work counters.
};

/// @brief A validity functor that also tests a block of configurations at once.
///
/// @details `batch(p, n)` returns true when all `n` points are valid, and `batch_size()` is
/// the block size to hand over. SIMD checkers such as VAMP test a block in about the time of
/// one point.
template <typename F, typename PointT>
concept BatchValidity = requires(const F& f, const PointT* p, std::size_t n) {
  { f.batch(p, n) } -> std::convertible_to<bool>;
  { f.batch_size() } -> std::convertible_to<std::size_t>;
};

/// @brief Most validity checks `smooth_path` places on one edge.
inline constexpr double kMaxEdgeSamples = 1e7;

namespace detail {

// Constants of the method. None depends on scale.
inline constexpr int kSegments = 48;                 // descent segments per path
inline constexpr int kBestFirstWaypoints = 32;       // paths this sparse get best-first shortcuts
inline constexpr int kLevels = 3;                    // coarse-to-fine levels of the first descent
inline constexpr int kRounds = 3;                    // shortcut and descent rounds
inline constexpr int kAttemptsPerWaypoint = 8;       // shortcut attempts per waypoint and round
inline constexpr int kMinAttempts = 64;              // shortcut attempts per round, at least
inline constexpr int kMaxSweeps = 40;                // descent sweeps per level, at most
inline constexpr double kMoveTolerance = 1e-3;       // relative energy decrease of a move
inline constexpr int kBacktracks = 8;                // step reductions by 2 per move
inline constexpr double kResolutionFraction = 0.01;  // default check spacing, per path length

// Constants of corner rounding.
inline constexpr double kRoundingCostTolerance = 1e-3;   // metric length a curve may add
inline constexpr int kRoundingShrinks = 5;               // reductions by 2 before a corner stays
inline constexpr double kRoundingPlaneTolerance = 2e-3;  // motion outside the corner's plane
inline constexpr double kRoundingMinShare = 0.1;         // smallest share of a shared edge
inline constexpr int kRoundingBisections = 2;            // bisections after a failed size
inline constexpr int kCurveMinSamples = 4;               // fewest sample intervals of a curve
inline constexpr int kCurveRefinements = 4;              // doublings of the sample count
inline constexpr double kCurveMerge = 1e-3;              // merge distance, per corner_tolerance

using Clock = std::chrono::steady_clock;

inline double ms_since(const Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

/// @brief A zero vector shaped like `ref`.
template <typename V>
V zero_like(const V& ref) {
  if constexpr (V::SizeAtCompileTime == Eigen::Dynamic) {
    return V::Zero(ref.size());
  } else {
    return V::Zero();
  }
}

/// @brief Interior sample indices 1..m of an edge split into m + 1 intervals, in
/// breadth-first bisection order.
inline const std::vector<int>& bisection_order(const int m) {
  thread_local std::vector<int> order;
  thread_local std::vector<std::pair<int, int>> queue;
  order.clear();
  queue.clear();
  queue.emplace_back(0, m + 1);
  for (std::size_t head = 0; head < queue.size(); ++head) {
    const auto [lo, hi] = queue[head];
    const int mid = lo + (hi - lo) / 2;
    if (mid == lo) continue;
    order.push_back(mid);
    if (mid - lo > 1) queue.emplace_back(lo, mid);
    if (hi - mid > 1) queue.emplace_back(mid, hi);
  }
  return order;
}

/// @brief Point and edge checks of one smoothing call, with counters.
template <RiemannianManifold M, typename ValidityFn>
class Oracle {
 public:
  using Point = typename M::Point;

  Oracle(const M& manifold, const ValidityFn& validity, const PathSmoothingSettings& settings,
         const double resolution, PathSmoothingProfile& profile)
      : manifold_(manifold),
        validity_(validity),
        settings_(settings),
        resolution_(resolution),
        profile_(profile) {}

  /// @brief Number of check intervals of the edge from `a` to `b`, the smallest count whose
  /// spacing is at most the resolution. A ratio within 1e-9 of an integer counts as that
  /// integer.
  int intervals(const Point& a, const Point& b) const {
    if (resolution_ <= 0.0) return 1;
    double length = 0.0;
    if (settings_.edge_travel && settings_.collision_check_resolution > 0.0) {
      length = settings_.edge_travel(a, b);
      if (!(std::isfinite(length) && length >= 0.0)) {
        throw std::invalid_argument("smooth_path: edge_travel must be finite and >= 0");
      }
    } else {
      length = utils::ordered_norm(manifold_.log(a, b));
    }
    const double count = std::ceil(length / resolution_ - 1e-9);
    if (!(count <= kMaxEdgeSamples)) {
      throw std::invalid_argument(
          "smooth_path: collision_check_resolution asks for more than kMaxEdgeSamples samples on "
          "an edge");
    }
    return std::max(1, static_cast<int>(count));
  }

  /// @brief Validity of one configuration.
  bool point(const Point& p) const {
    ++profile_.point_checks;
    return static_cast<bool>(validity_(p));
  }

  /// @brief Validity of the geodesic from `a` to `b`, `a` excluded. With `with_end`, `b` is
  /// tested first.
  bool edge(const Point& a, const Point& b, const bool with_end = false) const {
    ++profile_.edge_checks;
    if (settings_.edge_validator) {
      return (!with_end || point(b)) && settings_.edge_validator(a, b);
    }
    if (settings_.edge_provably_clear && settings_.edge_provably_clear(a, b)) {
      ++profile_.edge_proofs;
      return !with_end || point(b);
    }
    const int m = intervals(a, b) - 1;
    if (m <= 0) return !with_end || point(b);
    const auto& order = bisection_order(m);
    const double inv = 1.0 / static_cast<double>(m + 1);
    if constexpr (BatchValidity<ValidityFn, Point>) {
      const std::size_t w = std::max<std::size_t>(validity_.batch_size(), 1);
      thread_local std::vector<Point> block;
      if (block.size() < w) block.resize(w);
      std::size_t k = 0;
      const auto flush = [&] {
        profile_.point_checks += static_cast<long>(k);
        ++profile_.batch_calls;
        const bool ok = static_cast<bool>(validity_.batch(block.data(), k));
        k = 0;
        return ok;
      };
      if (with_end) block[k++] = b;
      for (const int idx : order) {
        if (k == w && !flush()) return false;
        block[k++] = manifold_.geodesic(a, b, static_cast<double>(idx) * inv);
      }
      return k == 0 || flush();
    } else {
      if (with_end && !point(b)) return false;
      for (const int idx : order) {
        if (!point(manifold_.geodesic(a, b, static_cast<double>(idx) * inv))) return false;
      }
      return true;
    }
  }

  /// @brief Validity of the part of the geodesic from `a` to `b` strictly between the
  /// parameters `t0` and `t1`, at the samples of the whole edge. When the whole edge passes
  /// `edge`, every part of it passes too. With an edge validator, the caller tests the whole
  /// edge.
  bool piece(const Point& a, const Point& b, const double t0, const double t1) const {
    ++profile_.edge_checks;
    if (settings_.edge_provably_clear && settings_.edge_provably_clear(a, b)) {
      ++profile_.edge_proofs;
      return true;
    }
    const int n = intervals(a, b);
    // Indices of the edge's samples inside the part.
    const int lo = static_cast<int>(std::floor(t0 * n + 1e-9)) + 1;
    const int hi = std::min(n - 1, static_cast<int>(std::ceil(t1 * n - 1e-9)) - 1);
    if (hi < lo) return true;
    const auto& order = bisection_order(hi - lo + 1);
    const double inv = 1.0 / static_cast<double>(n);
    if constexpr (BatchValidity<ValidityFn, Point>) {
      const std::size_t w = std::max<std::size_t>(validity_.batch_size(), 1);
      thread_local std::vector<Point> block;
      if (block.size() < w) block.resize(w);
      std::size_t k = 0;
      const auto flush = [&] {
        profile_.point_checks += static_cast<long>(k);
        ++profile_.batch_calls;
        const bool ok = static_cast<bool>(validity_.batch(block.data(), k));
        k = 0;
        return ok;
      };
      for (const int idx : order) {
        if (k == w && !flush()) return false;
        block[k++] = manifold_.geodesic(a, b, static_cast<double>(lo - 1 + idx) * inv);
      }
      return k == 0 || flush();
    } else {
      for (const int idx : order) {
        if (!point(manifold_.geodesic(a, b, static_cast<double>(lo - 1 + idx) * inv))) {
          return false;
        }
      }
      return true;
    }
  }

 private:
  const M& manifold_;
  const ValidityFn& validity_;
  const PathSmoothingSettings& settings_;
  double resolution_;
  PathSmoothingProfile& profile_;
};

/// @brief Implementation of `smooth_path`.
template <RiemannianManifold M, typename ValidityFn>
class Smoother {
 public:
  using Point = typename M::Point;
  using Tangent = typename M::Tangent;
  using Result = PathSmoothingResult<Point>;

  Smoother(const M& manifold, const ValidityFn& validity, const PathSmoothingSettings& settings)
      : manifold_(manifold), validity_(validity), settings_(settings) {}

  Result run(const std::vector<Point>& input) {
    const auto t_start = Clock::now();
    Result result;
    auto& prof = result.profile;
    prof.input_waypoints = static_cast<int>(input.size());
    if (input.empty()) return result;

    std::vector<Point> path = without_repeats(input);
    double extent = 0.0;
    for (std::size_t k = 1; k < path.size(); ++k) extent += coord_extent(path[k - 1], path[k]);
    const double resolution = settings_.collision_check_resolution > 0.0
                                  ? settings_.collision_check_resolution
                                  : kResolutionFraction * extent;
    const Oracle<M, ValidityFn> oracle(manifold_, validity_, settings_, resolution, prof);
    oracle_ = &oracle;
    prof_ = &prof;
    // Metric lengths are integrated at this step.
    quad_step_ = extent / static_cast<double>(4 * kSegments);
    // The energy descent needs one tangent coordinate per dimension.
    intrinsic_ = path.size() >= 2 &&
                 static_cast<int>(manifold_.log(path[0], path[1]).size()) == manifold_.dim();
    sharp_ = 0;
    if (path.size() >= 2 && settings_.sharp_coordinates <
                                static_cast<int>(manifold_.log(path[0], path[1]).size())) {
      sharp_ = settings_.sharp_coordinates;
    }

    std::vector<Point> shortcut_only;
    if (path.size() >= 3) {
      std::mt19937_64 rng(settings_.seed);
      for (int round = 0; round < kRounds; ++round) {
        auto t0 = Clock::now();
        const int removed = static_cast<int>(path.size()) <= kBestFirstWaypoints
                                ? shortcut_best_first(path)
                                : shortcut(path, rng);
        prof.shortcut_ms += ms_since(t0);
        if (round == 0) shortcut_only = path;
        if (round > 0 && removed == 0) break;
        if (!intrinsic_) continue;
        t0 = Clock::now();
        descend(path, round == 0 ? kLevels : 1);
        prof.descent_ms += ms_since(t0);
      }
    }

    std::vector<Point> smoothed = path;
    if (settings_.output_spacing > 0.0 && path.size() > 2) {
      const auto t0 = Clock::now();
      smoothed = resample(path);
      prof.resample_ms += ms_since(t0);
    }

    // Round the corners. The rounded path checks itself, and the stages without rounding
    // follow when it fails.
    std::vector<Point> rounded;
    std::vector<unsigned char> corners;
    if (settings_.round_corners && path.size() > 2) {
      const auto t0 = Clock::now();
      const double certify0 = prof.certify_ms;
      merge_distance_ = kCurveMerge * settings_.corner_tolerance;
      rounded = round_path_corners(path, corners);
      prof.rounding_ms += ms_since(t0) - (prof.certify_ms - certify0);
    }
    if (!rounded.empty() && predicate_ok(rounded)) {
      finish(result, std::move(rounded), 0, t_start, corners);
      return result;
    }
    prof.rounded_corners = prof.cusps = prof.kept_corners = prof.split_corners = 0;

    // Check the returned path. A stage that fails hands over to the one before it, down to
    // the input. A failing input is returned with collision_free false.
    const auto t_cert = Clock::now();
    const std::vector<Point>* stages[] = {&smoothed, &path, &shortcut_only, &input};
    std::size_t bad = Result::npos;
    int stage = 0;
    for (; stage < 4; ++stage) {
      const auto& candidate = *stages[stage];
      if (candidate.empty()) continue;
      if (stage > 0 && stage < 3 && candidate == *stages[stage - 1]) continue;
      if (stage < 3 && !predicate_ok(candidate)) continue;
      bad = first_invalid(candidate);
      if (bad == Result::npos) break;
    }
    prof.certify_ms += ms_since(t_cert);
    if (stage < 4) {
      finish(result, *stages[stage], stage, t_start);
    } else {
      result.first_invalid_index = bad;
      finish(result, input, 3, t_start);
    }
    return result;
  }

 private:
  /// Fills `result` with `path`. A path that passed the check is spaced evenly between the
  /// waypoints marked in `corners`, or between its ends and turns when `corners` is empty.
  void finish(Result& result, std::vector<Point> path, const int stage,
              const Clock::time_point t_start,
              const std::vector<unsigned char>& corners = {}) const {
    auto& prof = result.profile;
    result.path = std::move(path);
    result.collision_free = result.first_invalid_index == Result::npos;
    prof.fallback = stage;
    if (result.collision_free && result.path.size() >= 2) {
      const auto t0 = Clock::now();
      reparameterize(result.path, corners);
      prof.resample_ms += ms_since(t0);
    }
    for (std::size_t k = 1; k < result.path.size(); ++k) {
      result.length += static_cast<double>(manifold_.distance(result.path[k - 1], result.path[k]));
    }
    prof.output_waypoints = static_cast<int>(result.path.size());
    prof.total_ms = ms_since(t_start);
  }

  // --- even spacing -------------------------------------------------------

  /// Respaces `path` at equal steps between the waypoints marked in `corners`, at the longest
  /// step up to `output_spacing` whose edges stay within `corner_tolerance` of `path`. Without
  /// marks, the ends and the turns of `path` remain. Each try takes a step a tenth shorter
  /// than the last. `path` stays when no step passes or the path predicate rejects the result.
  void reparameterize(std::vector<Point>& path, const std::vector<unsigned char>& corners) const {
    const std::vector<unsigned char> stay = corners.size() == path.size() ? corners : turns(path);
    std::vector<double> at(path.size(), 0.0);  // coordinate arclength of each waypoint
    for (std::size_t k = 1; k < path.size(); ++k) {
      at[k] = at[k - 1] + coord_extent(path[k - 1], path[k]);
    }
    if (!(at.back() > 0.0)) return;
    const double cap = settings_.output_spacing > 0.0 ? settings_.output_spacing : at.back();
    for (double step = std::min(cap, at.back()); at.back() / step <= kMaxEdgeSamples;
         step *= 0.9) {
      std::vector<Point> out = spaced(path, at, step, stay, settings_.corner_tolerance);
      if (!out.empty()) {
        if (predicate_ok(out)) path = std::move(out);
        return;
      }
    }
  }

  /// `p` at equal steps no longer than `step` between the waypoints marked in `stay`, with `at`
  /// the coordinate arclength of each waypoint. Returns an empty path when an edge lies
  /// farther than `tolerance` from a waypoint of `p` in its span.
  std::vector<Point> spaced(const std::vector<Point>& p, const std::vector<double>& at,
                            const double step, const std::vector<unsigned char>& stay,
                            const double tolerance) const {
    std::vector<Point> out{p.front()};
    std::vector<double> pos{0.0};  // arclength of each returned waypoint
    std::size_t a = 0;
    for (std::size_t b = 1; b < p.size(); ++b) {
      if (b + 1 < p.size() && !stay[b]) continue;
      const double n = std::max(1.0, std::ceil((at[b] - at[a]) / step - 1e-9));
      std::size_t k = a + 1;
      for (double j = 1.0; j < n; j += 1.0) {
        const double s = at[a] + (at[b] - at[a]) * (j / n);
        while (at[k] < s) ++k;
        out.push_back(manifold_.geodesic(p[k - 1], p[k], (s - at[k - 1]) / (at[k] - at[k - 1])));
        pos.push_back(s);
      }
      out.push_back(p[b]);
      pos.push_back(at[b]);
      a = b;
    }
    std::size_t k = 1;
    for (std::size_t j = 1; j < out.size(); ++j) {
      while (k < p.size() && at[k] <= pos[j - 1]) ++k;
      for (; k < p.size() && at[k] < pos[j]; ++k) {
        const double t = (at[k] - pos[j - 1]) / (pos[j] - pos[j - 1]);
        if (!(coord_extent(manifold_.geodesic(out[j - 1], out[j], t), p[k]) <= tolerance)) {
          return {};
        }
      }
    }
    return out;
  }

  /// Marks the ends of `path` and the waypoints where it turns.
  std::vector<unsigned char> turns(const std::vector<Point>& path) const {
    std::vector<unsigned char> turn(path.size(), 1);
    for (std::size_t k = 1; k + 1 < path.size(); ++k) {
      const Tangent u = manifold_.log(path[k], path[k - 1]);
      const Tangent v = manifold_.log(path[k], path[k + 1]);
      const double uv = utils::ordered_norm(u) * utils::ordered_norm(v);
      turn[k] = uv > 0.0 && -u.dot(v) >= (1.0 - 1e-9) * uv ? 0 : 1;
    }
    return turn;
  }

  // --- geometry -----------------------------------------------------------

  double coord_extent(const Point& a, const Point& b) const {
    return utils::ordered_norm(manifold_.log(a, b));
  }

  /// Metric length of the geodesic from `a` to `b`, integrated in pieces no longer than the
  /// quadrature step.
  double metric_length(const Point& a, const Point& b) const {
    const int pieces =
        quad_step_ > 0.0 ? static_cast<int>(std::ceil(coord_extent(a, b) / quad_step_)) : 1;
    if (pieces <= 1) return static_cast<double>(manifold_.distance(a, b));
    double len = 0.0;
    Point prev = a;
    for (int k = 1; k <= pieces; ++k) {
      Point next = k == pieces ? b : manifold_.geodesic(a, b, static_cast<double>(k) / pieces);
      len += static_cast<double>(manifold_.distance(prev, next));
      prev = std::move(next);
    }
    return len;
  }

  std::vector<Point> without_repeats(const std::vector<Point>& in) const {
    std::vector<Point> out;
    out.reserve(in.size());
    for (const auto& p : in) {
      if (out.empty() || coord_extent(out.back(), p) > 0.0) out.push_back(p);
    }
    if (out.size() == 1 && in.size() > 1) out.push_back(in.back());
    return out;
  }

  bool predicate_ok(const std::vector<Point>& path) const {
    if (!settings_.path_predicate) return true;
    std::vector<Eigen::VectorXd> tmp;
    tmp.reserve(path.size());
    for (const auto& q : path) tmp.emplace_back(q);
    return settings_.path_predicate(tmp);
  }

  /// Validity of the part of the waypoint edge from `pts[j]` to `pts[j + 1]` between the
  /// parameters `t0` and `t1`, at the samples of that edge. An edge validator decides the whole
  /// edge once.
  bool piece_valid(const std::vector<Point>& pts, const std::size_t j, const double t0,
                   const double t1) const {
    if (settings_.edge_validator) {
      signed char& known = whole_edge_ok_[j];
      if (known < 0) known = oracle_->edge(pts[j], pts[j + 1]) ? 1 : 0;
      return known == 1;
    }
    return oracle_->piece(pts[j], pts[j + 1], t0, t1);
  }

  /// Parameter of `x` on the geodesic from `a` to `b`.
  double along(const Point& a, const Point& b, const Point& x) const {
    const double whole = coord_extent(a, b);
    return whole > 0.0 ? std::clamp(coord_extent(a, x) / whole, 0.0, 1.0) : 0.0;
  }

  /// First failing waypoint or edge of a rounded path from `assemble`. A straight part of the
  /// waypoint edge from `pts[j]` to `pts[j + 1]` is checked at the samples of that edge, and the
  /// points `filler` marks on it count as parts of it.
  std::size_t first_invalid_rounded(const std::vector<Point>& out,
                                    const std::vector<std::size_t>& owner,
                                    const std::vector<unsigned char>& filler,
                                    const std::vector<Point>& pts) const {
    for (std::size_t k = 0; k < out.size(); ++k) {
      if (!filler[k] && !oracle_->point(out[k])) return k;
      if (k + 1 == out.size()) break;
      const std::size_t o = owner[k];
      bool ok = false;
      if (o % 2 == 1) {
        const Point& a = pts[o / 2];
        const Point& b = pts[o / 2 + 1];
        ok = piece_valid(pts, o / 2, along(a, b, out[k]), along(a, b, out[k + 1]));
      } else {
        ok = oracle_->edge(out[k], out[k + 1]);
      }
      if (!ok) return k;
    }
    return Result::npos;
  }

  std::size_t first_invalid(const std::vector<Point>& path) const {
    for (std::size_t k = 0; k < path.size(); ++k) {
      if (!oracle_->point(path[k])) return k;
      if (k + 1 < path.size() && !oracle_->edge(path[k], path[k + 1])) return k;
    }
    return Result::npos;
  }

  // --- shortcutting -------------------------------------------------------

  /// Best-first shortcuts on a sparse path. The valid shortcut that saves the most metric
  /// length is taken until none is left. A blocked chord is not checked twice.
  int shortcut_best_first(std::vector<Point>& path) const {
    const std::size_t n0 = path.size();
    std::vector<std::size_t> id(n0);  // original index of each remaining waypoint
    for (std::size_t k = 0; k < n0; ++k) id[k] = k;
    std::vector<unsigned char> blocked(n0 * n0, 0);
    int removed = 0;
    struct Candidate {
      double saving;
      std::size_t i, j;
    };
    std::vector<Candidate> cands;
    while (path.size() > 2) {
      std::vector<double> prefix(path.size(), 0.0);
      for (std::size_t k = 1; k < path.size(); ++k) {
        prefix[k] = prefix[k - 1] + metric_length(path[k - 1], path[k]);
      }
      cands.clear();
      for (std::size_t i = 0; i + 2 < path.size(); ++i) {
        for (std::size_t j = i + 2; j < path.size(); ++j) {
          if (blocked[id[i] * n0 + id[j]]) continue;
          const double sub = prefix[j] - prefix[i];
          const double direct = static_cast<double>(manifold_.distance(path[i], path[j]));
          if (direct < sub) cands.push_back({sub - direct, i, j});
        }
      }
      std::sort(cands.begin(), cands.end(),
                [](const Candidate& x, const Candidate& y) { return x.saving > y.saving; });
      bool accepted = false;
      for (const auto& c : cands) {
        ++prof_->shortcut_attempts;
        const double sub = prefix[c.j] - prefix[c.i];
        if (!(metric_length(path[c.i], path[c.j]) < sub * (1.0 - 1e-9))) continue;
        if (!oracle_->edge(path[c.i], path[c.j])) {
          blocked[id[c.i] * n0 + id[c.j]] = 1;
          continue;
        }
        if (settings_.path_predicate) {
          std::vector<Point> trial;
          trial.insert(trial.end(), path.begin(), path.begin() + c.i + 1);
          trial.insert(trial.end(), path.begin() + c.j, path.end());
          if (!predicate_ok(trial)) continue;
        }
        path.erase(path.begin() + c.i + 1, path.begin() + c.j);
        id.erase(id.begin() + c.i + 1, id.begin() + c.j);
        removed += static_cast<int>(c.j - c.i - 1);
        ++prof_->shortcuts;
        accepted = true;
        break;
      }
      if (!accepted) break;
    }
    return removed;
  }

  /// One round of random shortcuts.
  int shortcut(std::vector<Point>& path, std::mt19937_64& rng) const {
    std::vector<double> prefix(path.size(), 0.0);
    for (std::size_t k = 1; k < path.size(); ++k) {
      prefix[k] = prefix[k - 1] + metric_length(path[k - 1], path[k]);
    }
    const long attempts = std::max<long>(
        kMinAttempts, static_cast<long>(kAttemptsPerWaypoint) * static_cast<long>(path.size()));
    int removed = 0;
    for (long attempt = 0; attempt < attempts; ++attempt) {
      const int n = static_cast<int>(path.size());
      if (n <= 2) break;
      ++prof_->shortcut_attempts;
      int i = static_cast<int>(utils::uniform_index(rng, static_cast<std::uint64_t>(n)));
      int j = static_cast<int>(utils::uniform_index(rng, static_cast<std::uint64_t>(n)));
      if (i > j) std::swap(i, j);
      if (j - i < 2) continue;
      const double sub = prefix[j] - prefix[i];
      // The distance screens candidates, and the integrated length decides.
      if (!(static_cast<double>(manifold_.distance(path[i], path[j])) < sub)) continue;
      const double direct = metric_length(path[i], path[j]);
      if (!(direct < sub * (1.0 - 1e-9))) continue;
      if (!oracle_->edge(path[i], path[j])) continue;
      if (settings_.path_predicate) {
        std::vector<Point> trial;
        trial.reserve(path.size() - static_cast<std::size_t>(j - i - 1));
        trial.insert(trial.end(), path.begin(), path.begin() + i + 1);
        trial.insert(trial.end(), path.begin() + j, path.end());
        if (!predicate_ok(trial)) continue;
      }
      path.erase(path.begin() + i + 1, path.begin() + j);
      prefix.erase(prefix.begin() + i + 1, prefix.begin() + j);
      const double saved = direct - sub;
      for (std::size_t k = static_cast<std::size_t>(i) + 1; k < prefix.size(); ++k) {
        prefix[k] += saved;
      }
      removed += j - i - 1;
      ++prof_->shortcuts;
    }
    return removed;
  }

  // --- energy descent -----------------------------------------------------

  /// Splits every edge into geodesic pieces no longer than `h` in metric length. An edge
  /// whose pieces fail a check stays whole.
  void subdivide(std::vector<Point>& path, const double h) const {
    std::vector<Point> out;
    std::vector<Point> pieces_pts;
    out.reserve(path.size() * 2);
    out.push_back(path.front());
    for (std::size_t k = 1; k < path.size(); ++k) {
      const double len = metric_length(path[k - 1], path[k]);
      const int pieces = std::max(1, static_cast<int>(std::ceil(len / h - 1e-9)));
      pieces_pts.clear();
      bool ok = true;
      for (int s = 1; s < pieces && ok; ++s) {
        pieces_pts.push_back(
            manifold_.geodesic(path[k - 1], path[k], static_cast<double>(s) / pieces));
        const Point& from = s == 1 ? path[k - 1] : pieces_pts[pieces_pts.size() - 2];
        ok = oracle_->edge(from, pieces_pts.back(), true);
      }
      if (ok && pieces > 1) ok = oracle_->edge(pieces_pts.back(), path[k]);
      if (ok) out.insert(out.end(), pieces_pts.begin(), pieces_pts.end());
      out.push_back(path[k]);
    }
    if (out.size() > path.size() && predicate_ok(out)) path = std::move(out);
  }

  /// Coarse-to-fine descent, finest level last.
  void descend(std::vector<Point>& path, const int levels) const {
    double total = 0.0;
    for (std::size_t k = 1; k < path.size(); ++k) total += metric_length(path[k - 1], path[k]);
    if (total <= 0.0) return;
    for (int level = levels - 1; level >= 0; --level) {
      subdivide(path, total / static_cast<double>(std::max(2, kSegments >> level)));
      relax(path);
    }
  }

  /// The metric evaluated once at `p`.
  auto frozen_at(const Point& p) const {
    if constexpr (HasMetricAt<M, Point>) {
      return manifold_.metric_at(p);
    } else {
      return ForwardingMetricAt<M, Point>(manifold_, p);
    }
  }

  /// Gram matrix of a frozen metric in tangent coordinates.
  template <typename Frozen>
  static Eigen::MatrixXd gram(const Frozen& g, const Tangent& shape) {
    if constexpr (requires { g.gram(); }) {
      return g.gram();
    } else {
      const auto n = static_cast<Eigen::Index>(shape.size());
      Eigen::MatrixXd G(n, n);
      Tangent ei = zero_like(shape), ej = zero_like(shape);
      for (Eigen::Index i = 0; i < n; ++i) {
        ei.setZero();
        ei[i] = 1.0;
        for (Eigen::Index j = i; j < n; ++j) {
          ej.setZero();
          ej[j] = 1.0;
          G(i, j) = G(j, i) = static_cast<double>(g.inner(ei, ej));
        }
      }
      return G;
    }
  }

  /// Outcome of one waypoint visit.
  enum class Visit {
    kMoved,      ///< The waypoint moved.
    kSettled,    ///< No trial passed its tests.
    kPredicate,  ///< Only the path predicate rejected a trial.
  };

  /// Moves waypoint `i` by a Gauss-Newton step on the energy of its two edges, shrinking the
  /// step by a factor of two until the energy drops and both edges stay valid. The first trial
  /// takes `2^-first` of the step. With `taken`, the energy alone decides, the edges stay
  /// unchecked, and `taken` receives the exponent of the step taken.
  template <typename FrozenA>
  Visit visit(std::vector<Point>& path, const std::size_t i, const FrozenA& ga, const int first = 0,
              int* taken = nullptr) const {
    if (first >= kBacktracks) return Visit::kSettled;
    const Point& a = path[i - 1];
    const Point& b = path[i + 1];
    const Point p0 = path[i];
    const double h = std::min(coord_extent(p0, a), coord_extent(p0, b));
    if (h <= 0.0) return Visit::kSettled;
    const Tangent v10 = manifold_.log(a, p0);
    const Tangent v20 = manifold_.log(p0, b);
    const Eigen::MatrixXd ga_gram = gram(ga, v10);
    auto energy = [&](const Point& p, const Tangent& v1, const Tangent& v2) {
      return utils::ordered_quadratic_form(v1, ga_gram, v1) +
             static_cast<double>(manifold_.inner(p, v2, v2));
    };
    const double e0 = energy(p0, v10, v20);
    const auto n = static_cast<Eigen::Index>(v20.size());

    // Forward differences give the gradient and the Jacobians of both edge tangents.
    const double eps = 1e-6 * h;
    Eigen::VectorXd g(n);
    Eigen::MatrixXd J1(n, n), J2(n, n);
    Tangent e = zero_like(v20);
    for (Eigen::Index j = 0; j < n; ++j) {
      e.setZero();
      e[j] = eps;
      const Point pj = manifold_.exp(p0, e);
      const Tangent v1 = manifold_.log(a, pj);
      const Tangent v2 = manifold_.log(pj, b);
      g[j] = (energy(pj, v1, v2) - e0) / eps;
      for (Eigen::Index r = 0; r < n; ++r) {
        J1(r, j) = (v1[r] - v10[r]) / eps;
        J2(r, j) = (v2[r] - v20[r]) / eps;
      }
    }
    const double gn = g.norm();
    if (!(gn > 0.0) || !std::isfinite(gn)) return Visit::kSettled;

    // A gradient step replaces a Gauss-Newton step that does not descend.
    const Eigen::MatrixXd H =
        2.0 * (J1.transpose() * ga_gram * J1 + J2.transpose() * gram(frozen_at(p0), v20) * J2);
    Eigen::VectorXd step = -H.partialPivLu().solve(g);
    if (!step.allFinite() || step.dot(g) >= 0.0) step = -(h / gn) * g;
    const double sn = step.norm();
    if (sn > h) step *= h / sn;

    Tangent delta = zero_like(v20);
    double scale = std::ldexp(1.0, -first);
    Visit outcome = Visit::kSettled;
    for (int bt = first; bt < kBacktracks; ++bt, scale *= 0.5) {
      for (Eigen::Index j = 0; j < n; ++j) delta[j] = scale * step[j];
      const Point p = manifold_.exp(p0, delta);
      if (!(energy(p, manifold_.log(a, p), manifold_.log(p, b)) < e0 * (1.0 - kMoveTolerance))) {
        continue;
      }
      if (taken != nullptr) {
        path[i] = p;
        *taken = bt;
        return Visit::kMoved;
      }
      if (!oracle_->edge(a, p, true) || !oracle_->edge(p, b)) continue;
      path[i] = p;
      if (settings_.path_predicate && !predicate_ok(path)) {
        path[i] = p0;
        outcome = Visit::kPredicate;
        continue;
      }
      return Visit::kMoved;
    }
    return outcome;
  }

  /// Gauss-Seidel sweeps in alternating direction until a sweep moves nothing. A settled
  /// waypoint is skipped until a neighbor moves. Without a path predicate and an edge validator,
  /// `relax_lazy` moves the waypoints on the energy alone and checks their edges after a period
  /// of sweeps.
  void relax(std::vector<Point>& path) const {
    const std::size_t n = path.size();
    if (n < 3) return;
    if (!settings_.path_predicate && !settings_.edge_validator) {
      relax_lazy(path);
      return;
    }
    std::vector<unsigned char> settled(n, 0);
    for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
      int moved = 0;
      for (std::size_t k = 1; k + 1 < n; ++k) {
        const std::size_t i = (sweep % 2 == 0) ? k : n - 1 - k;
        if (settled[i]) continue;
        ++prof_->relax_visits;
        const Visit v = visit(path, i, frozen_at(path[i - 1]));
        if (v == Visit::kMoved) {
          ++moved;
          ++prof_->relax_moves;
          settled[i - 1] = settled[i] = settled[i + 1] = 0;
        } else if (v == Visit::kSettled) {
          settled[i] = 1;
        }
      }
      if (moved == 0) break;
    }
  }

  /// Gauss-Seidel sweeps in alternating direction that move waypoints on the energy alone.
  /// `validate` checks the moved waypoints after a period of sweeps. The period starts at one
  /// sweep, doubles after a validation that takes back no move, and returns to one sweep
  /// after one that does. The sweeps stop when a sweep moves nothing and every move passes. A
  /// validation follows the last sweep when a move is unchecked.
  void relax_lazy(std::vector<Point>& path) const {
    const std::size_t n = path.size();
    std::vector<unsigned char> settled(n, 0);
    std::vector<int> first(n, 0);       // exponent of each waypoint's first trial
    std::vector<int> taken(n, -1);      // exponent of the largest unchecked step, -1 without one
    std::vector<Point> checked = path;  // the path at the last validation
    int period = 1;
    int unchecked = 0;  // sweeps with a move since the last validation
    for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
      int moved = 0;
      for (std::size_t k = 1; k + 1 < n; ++k) {
        const std::size_t i = (sweep % 2 == 0) ? k : n - 1 - k;
        if (settled[i]) continue;
        ++prof_->relax_visits;
        int bt = -1;
        if (visit(path, i, frozen_at(path[i - 1]), first[i], &bt) == Visit::kMoved) {
          taken[i] = taken[i] < 0 ? bt : std::min(taken[i], bt);
          ++moved;
          settled[i - 1] = settled[i] = settled[i + 1] = 0;
        } else {
          settled[i] = 1;
        }
      }
      if (moved > 0) ++unchecked;
      if (unchecked == 0) break;
      if (moved > 0 && unchecked < period && sweep + 1 < kMaxSweeps) continue;
      const int undone = validate(path, checked, taken, first, settled);
      if (moved == 0 && undone == 0) break;
      period = undone == 0 ? std::min(2 * period, kMaxSweeps) : 1;
      unchecked = 0;
    }
  }

  /// Checks the edges at the waypoints with an unchecked move and takes back the moves at a
  /// failing edge until every edge passes. A waypoint taken back returns to `checked`, its next
  /// trial starts at half the failed step, and its neighbors leave the settled state. A move that
  /// passes doubles the first step of its waypoint's next visit. Returns the number of moves
  /// taken back.
  int validate(std::vector<Point>& path, std::vector<Point>& checked, std::vector<int>& taken,
               std::vector<int>& first, std::vector<unsigned char>& settled) const {
    const std::size_t n = path.size();
    std::vector<unsigned char> dirty(n - 1, 0);  // the edge from k to k + 1 needs a check
    for (std::size_t i = 1; i + 1 < n; ++i) {
      if (taken[i] >= 0) dirty[i - 1] = dirty[i] = 1;
    }
    int undone = 0;
    const auto take_back = [&](const std::size_t j) {
      path[j] = checked[j];
      first[j] = taken[j] + 1;
      taken[j] = -1;
      settled[j] = first[j] >= kBacktracks ? 1 : 0;
      settled[j - 1] = settled[j + 1] = 0;
      ++undone;
    };
    for (bool again = true; again;) {
      again = false;
      for (std::size_t k = 0; k + 1 < n; ++k) {
        if (!dirty[k]) continue;
        dirty[k] = 0;
        const bool start_moved = taken[k] >= 0;
        const bool end_moved = taken[k + 1] >= 0;
        // An edge between two checked waypoints is an edge of the checked path.
        if (!start_moved && !end_moved) continue;
        if (oracle_->edge(path[k], path[k + 1], end_moved)) continue;
        // Taking back a waypoint changes its other edge. That edge needs a new check when its
        // far end moved.
        if (start_moved) {
          take_back(k);
          if (taken[k - 1] >= 0) {
            dirty[k - 1] = 1;
            again = true;
          }
        }
        if (end_moved) {
          take_back(k + 1);
          if (taken[k + 2] >= 0) dirty[k + 1] = 1;
        }
      }
    }
    for (std::size_t i = 1; i + 1 < n; ++i) {
      if (taken[i] < 0) continue;
      checked[i] = path[i];
      taken[i] = -1;
      first[i] = std::max(0, first[i] - 1);
      ++prof_->relax_moves;
    }
    return undone;
  }

  // --- output resampling --------------------------------------------------

  /// Even metric arclength resampling onto configurations the edge checks already passed.
  /// Returns `pts` when the path predicate rejects the result.
  std::vector<Point> resample(const std::vector<Point>& pts) const {
    std::vector<Point> out = resample_points(pts);
    if (!predicate_ok(out)) return pts;
    return out;
  }

  /// `resample` without the path predicate.
  std::vector<Point> resample_points(const std::vector<Point>& pts) const {
    std::vector<Point> grid;
    grid.push_back(pts.front());
    for (std::size_t k = 1; k < pts.size(); ++k) {
      const int n = oracle_->intervals(pts[k - 1], pts[k]);
      for (int j = 1; j < n; ++j) {
        grid.push_back(manifold_.geodesic(pts[k - 1], pts[k],
                                          static_cast<double>(j) * (1.0 / static_cast<double>(n))));
      }
      grid.push_back(pts[k]);
    }
    std::vector<double> arc(grid.size(), 0.0);
    for (std::size_t k = 1; k < grid.size(); ++k) {
      arc[k] = arc[k - 1] + static_cast<double>(manifold_.distance(grid[k - 1], grid[k]));
    }
    const double spacing = settings_.output_spacing;
    std::vector<Point> out;
    out.push_back(grid.front());
    std::size_t i = 0;
    while (i + 1 < grid.size()) {
      std::size_t j = i + 1;
      while (j + 1 < grid.size() && arc[j + 1] - arc[i] <= spacing * (1.0 + 1e-9)) ++j;
      while (j > i + 1 && !oracle_->edge(grid[i], grid[j])) j = i + (j - i) / 2;
      out.push_back(grid[j]);
      i = j;
    }
    return out;
  }

  // --- corner rounding ----------------------------------------------------

  /// Samples of one corner's curve, from the incoming edge to the outgoing edge. The samples
  /// are empty when the corner stays.
  using Samples = std::vector<Point>;

  /// The path with its corners rounded and every curve sampled. `corners` marks its ends and
  /// the corners without a curve.
  std::vector<Point> round_path_corners(const std::vector<Point>& in,
                                        std::vector<unsigned char>& corners) const {
    const std::vector<Point> pts = merge_straight(in);
    const std::size_t n = pts.size();
    whole_edge_ok_.assign(n - 1, -1);
    // Turning angle of every waypoint under the metric, 0 at the ends and at cusps that stay.
    // Two curves share the edge between them in proportion to their angles.
    std::vector<double> turn(n, 0.0);
    std::vector<unsigned char> cusp(n, 0);
    for (std::size_t k = 1; k + 1 < n; ++k) {
      const Tangent p = manifold_.log(pts[k], pts[k - 1]);
      const Tangent q = manifold_.log(pts[k], pts[k + 1]);
      const double pp = static_cast<double>(manifold_.inner(pts[k], p, p));
      const double qq = static_cast<double>(manifold_.inner(pts[k], q, q));
      if (!(pp > 0.0) || !(qq > 0.0) || !std::isfinite(pp * qq)) continue;
      const double c = -static_cast<double>(manifold_.inner(pts[k], p, q)) / std::sqrt(pp * qq);
      if (c < std::cos(settings_.corner_max_angle) - 1e-9) {
        cusp[k] = 1;
        if (sharp_ > 0) turn[k] = std::acos(std::clamp(c, -1.0, 1.0));
        continue;
      }
      turn[k] = std::acos(std::clamp(c, -1.0, 1.0));
    }
    std::vector<double> len(n, 0.0);  // metric length of the edge from k - 1 to k
    for (std::size_t k = 1; k < n; ++k) len[k] = metric_length(pts[k - 1], pts[k]);
    const auto share = [&](const std::size_t a, const std::size_t b, const std::size_t k) {
      const double other = k == a ? turn[b] : turn[a];
      if (other <= 0.0) return 1.0;
      return std::clamp(turn[k] / (turn[k] + other), kRoundingMinShare, 1.0 - kRoundingMinShare);
    };
    std::vector<Samples> curves(n);
    std::vector<std::size_t> turn_at(n, Result::npos);  // corner sample of a split curve
    Point prev_end = pts.front();
    for (std::size_t k = 1; k + 1 < n; ++k) {
      if (cusp[k]) ++prof_->cusps;
      if (turn[k] > 0.0) {
        const double r = std::min(share(k - 1, k, k) * len[k], share(k, k + 1, k) * len[k + 1]);
        const double t0 = along(pts[k - 1], pts[k], prev_end);
        Rounding best;
        if (!cusp[k]) best = round_corner(pts, k, t0, len[k], len[k + 1], r, false);
        // Where the curve through every coordinate stays smaller, the leading coordinates may
        // keep the corner.
        bool split = false;
        if (sharp_ > 0 && best.size < r) {
          Rounding split_curve = round_corner(pts, k, t0, len[k], len[k + 1], r, true);
          if (!split_curve.curve.empty() && split_curve.size > best.size) {
            best = std::move(split_curve);
            split = true;
          }
        }
        if (!best.curve.empty()) {
          ++prof_->rounded_corners;
          if (split) ++prof_->split_corners;
          turn_at[k] = best.turn_at;
        } else if (best.kept) {
          ++prof_->kept_corners;
        }
        curves[k] = std::move(best.curve);
        // A cusp that stays leaves its edges to its neighbors.
        if (cusp[k] && curves[k].empty()) turn[k] = 0.0;
      }
      prev_end = curves[k].empty() ? pts[k] : curves[k].back();
    }
    std::vector<unsigned char> use(n, 0);
    for (std::size_t k = 1; k + 1 < n; ++k) use[k] = curves[k].empty() ? 0 : 1;
    const auto drop = [&](const std::size_t k) {
      if (k == 0 || k + 1 >= n || !use[k]) return false;
      use[k] = 0;
      --prof_->rounded_corners;
      if (!cusp[k]) ++prof_->kept_corners;
      if (turn_at[k] != Result::npos) --prof_->split_corners;
      return true;
    };
    if (settings_.path_predicate && !predicate_ok(assemble(pts, curves, turn_at, use, false))) {
      // Add the curves one at a time in path order and keep those the predicate accepts.
      std::vector<unsigned char> accepted(n, 0);
      for (std::size_t k = 1; k + 1 < n; ++k) {
        if (!use[k]) continue;
        accepted[k] = 1;
        if (!predicate_ok(assemble(pts, curves, turn_at, accepted, false))) accepted[k] = 0;
      }
      for (std::size_t k = 1; k + 1 < n; ++k) {
        if (!accepted[k]) drop(k);
      }
    }
    // Check the rounded path. A failing edge drops the curves at both ends of the waypoint
    // edge it lies on, and the check repeats.
    const auto t_cert = Clock::now();
    std::vector<Point> out;
    for (;;) {
      std::vector<std::size_t> owner;
      std::vector<unsigned char> filler;
      out = assemble(pts, curves, turn_at, use, true, &owner, &corners, &filler);
      const std::size_t bad = first_invalid_rounded(out, owner, filler, pts);
      if (bad == Result::npos) break;
      const std::size_t o = owner[std::min(bad, owner.size() - 1)];
      bool dropped = drop(o / 2);
      if (o % 2 == 1) dropped = drop(o / 2 + 1) || dropped;
      if (!dropped) {
        out.clear();
        break;
      }
    }
    prof_->certify_ms += ms_since(t_cert);
    return out;
  }

  /// Removes every interior waypoint whose neighbors are joined by a valid edge no longer
  /// than the two edges it replaces. Returns `in` when the path predicate rejects the result.
  std::vector<Point> merge_straight(const std::vector<Point>& in) const {
    std::vector<Point> out;
    out.reserve(in.size());
    out.push_back(in.front());
    for (std::size_t k = 1; k + 1 < in.size(); ++k) {
      const double two = metric_length(out.back(), in[k]) + metric_length(in[k], in[k + 1]);
      if (metric_length(out.back(), in[k + 1]) <= two * (1.0 + 1e-9) &&
          oracle_->edge(out.back(), in[k + 1])) {
        continue;
      }
      out.push_back(in[k]);
    }
    out.push_back(in.back());
    if (out.size() < in.size() && !predicate_ok(out)) return in;
    return out;
  }

  /// The curve of one corner.
  struct Rounding {
    Samples curve;                   ///< Samples, empty when the corner stays.
    double size = 0.0;               ///< Metric length the curve reaches into both edges.
    bool kept = false;               ///< Every size failed a check.
    std::size_t turn_at = Result::npos;  ///< Sample where the leading coordinates turn.
  };

  /// The curve of corner `k`. The curve reaches the metric length `r` into both edges and
  /// shrinks `r` by a factor of two when it fails a check. The path before the corner ends at
  /// the parameter `prev_t` of the incoming edge. With `split`, the leading `sharp_`
  /// coordinates keep the corner and the others are rounded.
  Rounding round_corner(const std::vector<Point>& pts, const std::size_t k, const double prev_t,
                        const double len_in, const double len_out, double r,
                        const bool split) const {
    const Point& c = pts[k];
    const Tangent p = manifold_.log(c, pts[k - 1]);
    const Tangent q = manifold_.log(c, pts[k + 1]);
    // The curve must start on the incoming edge.
    const Point start = manifold_.exp(c, (0.5 * r / len_in) * p);
    if (coord_extent(manifold_.geodesic(pts[k - 1], c, 1.0 - 0.5 * r / len_in), start) >
        1e-3 * settings_.corner_tolerance) {
      return {.kept = true};
    }
    const auto accept = [&](const double size, Rounding& out) {
      const double f = size / len_in;
      const double g = size / len_out;
      Samples& s = out.curve;
      out.turn_at = Result::npos;
      if (!(split ? sample_split_curve(c, p, q, f, g, s, out.turn_at)
                  : sample_curve(c, p, q, f, g, s))) {
        ++prof_->rounding_retries;
        return false;
      }
      double len = 0.0;
      for (std::size_t j = 1; j < s.size(); ++j) len += metric_length(s[j - 1], s[j]);
      const double replaced = metric_length(s.front(), c) + metric_length(c, s.back());
      // The leading coordinates of a split curve stay on the two edges, and the others stay
      // in the plane of theirs.
      if (len <= (1.0 + kRoundingCostTolerance) * replaced && (split || in_plane(c, p, q, s)) &&
          curve_valid(pts, k - 1, prev_t, 1.0 - f, s)) {
        out.size = size;
        return true;
      }
      ++prof_->rounding_retries;
      return false;
    };
    // A curve narrower than corner_tolerance stays within the tolerance of its corner.
    const auto narrow = [&](const double size) {
      return std::max(size / len_in * utils::ordered_norm(p),
                      size / len_out * utils::ordered_norm(q)) < settings_.corner_tolerance;
    };
    if (narrow(r)) return {};
    Rounding s;
    for (int attempt = 0; attempt <= kRoundingShrinks && !narrow(r); ++attempt, r *= 0.5) {
      if (!accept(r, s)) continue;
      // After a failure, bisect between the failing size and the passing one.
      double lo = r;
      double hi = 2.0 * r;
      Rounding t;
      for (int step = 0; attempt > 0 && step < kRoundingBisections; ++step) {
        const double mid = std::sqrt(lo * hi);
        if (accept(mid, t)) {
          lo = mid;
          std::swap(s, t);
        } else {
          hi = mid;
        }
      }
      return s;
    }
    return {.kept = true};
  }

  /// True when no edge between the samples `s` moves outside the plane of `p` and `q` by
  /// more than `kRoundingPlaneTolerance` of its motion, under the metric at `c`.
  bool in_plane(const Point& c, const Tangent& p, const Tangent& q, const Samples& s) const {
    if (!intrinsic_) return true;
    const double pp = static_cast<double>(manifold_.inner(c, p, p));
    const double pq = static_cast<double>(manifold_.inner(c, p, q));
    const double qq = static_cast<double>(manifold_.inner(c, q, q));
    const double det = pp * qq - pq * pq;
    if (!(det > 1e-12 * pp * qq)) return true;
    for (std::size_t j = 1; j < s.size(); ++j) {
      const Tangent v = manifold_.log(s[j - 1], s[j]);
      const double vp = static_cast<double>(manifold_.inner(c, v, p));
      const double vq = static_cast<double>(manifold_.inner(c, v, q));
      const double vv = static_cast<double>(manifold_.inner(c, v, v));
      // Squared length of the part of v in the plane.
      const double in = (qq * vp * vp - 2.0 * pq * vp * vq + pp * vq * vq) / det;
      if (vv - in > kRoundingPlaneTolerance * kRoundingPlaneTolerance * vv) return false;
    }
    return true;
  }

  /// Point of the curve at corner `c` for `t` in [0, 1], exp(c, beta(t)) for the quintic
  /// Bézier curve beta with control points f p, 2/3 f p, 1/3 f p, 1/3 g q, 2/3 g q, g q.
  Point curve_point(const Point& c, const Tangent& p, const Tangent& q, const double f,
                    const double g, const double t) const {
    const double s = 1.0 - t;
    const double b0 = s * s * s * s * s;
    const double b1 = 5.0 * t * s * s * s * s;
    const double b2 = 10.0 * t * t * s * s * s;
    const double b3 = 10.0 * t * t * t * s * s;
    const double b4 = 5.0 * t * t * t * t * s;
    const double b5 = t * t * t * t * t;
    const double a = f * (b0 + (2.0 / 3.0) * b1 + (1.0 / 3.0) * b2);
    const double b = g * ((1.0 / 3.0) * b3 + (2.0 / 3.0) * b4 + b5);
    const Tangent v = a * p + b * q;
    return manifold_.exp(c, v);
  }

  /// Weights of `p` and `q` in the curve at parameter `t`, as in curve_point.
  static void bezier_weights(const double f, const double g, const double t, double& a,
                             double& b) {
    const double s = 1.0 - t;
    a = f * (s * s * s * s * s + (2.0 / 3.0) * 5.0 * t * s * s * s * s +
             (1.0 / 3.0) * 10.0 * t * t * s * s * s);
    b = g * ((1.0 / 3.0) * 10.0 * t * t * t * s * s + (2.0 / 3.0) * 5.0 * t * t * t * t * s +
             t * t * t * t * t);
  }

  /// Point of the split curve at corner `c` for `t` in [0, 1]. The trailing coordinates follow
  /// curve_point. The leading `sharp_` coordinates run along the incoming edge to the corner and
  /// on along the outgoing edge, as far in total as the curve has moved along `p` and `q`. `np`
  /// and `nq` are the norms of the leading parts of `p` and `q`.
  Point split_point(const Point& c, const Tangent& p, const Tangent& q, const double f,
                    const double g, const double t, const double np, const double nq) const {
    double a = 0.0;
    double b = 0.0;
    bezier_weights(f, g, t, a, b);
    Tangent v = a * p + b * q;
    const double covered = (f - a) * np + b * nq;
    if (np > 0.0 && covered <= f * np) {
      v.head(sharp_) = (f - covered / np) * p.head(sharp_);
    } else if (nq > 0.0) {
      v.head(sharp_) = ((covered - f * np) / nq) * q.head(sharp_);
    } else {
      v.head(sharp_).setZero();
    }
    return manifold_.exp(c, v);
  }

  /// Samples the split curve into `s` like sample_curve, with the sample `turn_at` where the
  /// leading coordinates pass the corner.
  bool sample_split_curve(const Point& c, const Tangent& p, const Tangent& q, const double f,
                          const double g, Samples& s, std::size_t& turn_at) const {
    const double np = utils::ordered_norm(Eigen::VectorXd(p.head(sharp_)));
    const double nq = utils::ordered_norm(Eigen::VectorXd(q.head(sharp_)));
    // The leading coordinates reach the corner where they have covered the incoming part.
    double lo = 0.0;
    double hi = 1.0;
    for (int i = 0; i < 60; ++i) {
      const double mid = 0.5 * (lo + hi);
      double a = 0.0;
      double b = 0.0;
      bezier_weights(f, g, mid, a, b);
      ((f - a) * np + b * nq < f * np ? lo : hi) = mid;
    }
    const double tm = 0.5 * (lo + hi);
    const double tol = (1.0 - kCurveMerge) * settings_.corner_tolerance;
    const double e = std::max(f * utils::ordered_norm(p), g * utils::ordered_norm(q));
    const double need = std::ceil(std::sqrt(5.0 * e / (8.0 * tol)));
    if (!(need <= kMaxEdgeSamples)) {
      throw std::invalid_argument(
          "smooth_path: corner_tolerance asks for more than kMaxEdgeSamples samples on a "
          "corner");
    }
    int m = std::max(kCurveMinSamples, static_cast<int>(need));
    std::vector<double> ts;
    const auto fill = [&] {
      ts.clear();
      s.clear();
      for (int j = 0; j <= m; ++j) {
        const double tj = static_cast<double>(j) / static_cast<double>(m);
        if (std::abs(tj - tm) <= 1e-12) {
          ts.push_back(tm);
          continue;
        }
        if (!ts.empty() && ts.back() < tm && tj > tm) ts.push_back(tm);
        ts.push_back(tj);
      }
      for (const double tj : ts) s.push_back(split_point(c, p, q, f, g, tj, np, nq));
      turn_at = static_cast<std::size_t>(std::find(ts.begin(), ts.end(), tm) - ts.begin());
    };
    for (int round = 0; round <= kCurveRefinements; ++round, m *= 2) {
      fill();
      for (int grow = 0; grow < 4 && settings_.output_spacing > 0.0; ++grow) {
        double longest = 0.0;
        for (std::size_t j = 1; j < s.size(); ++j) {
          longest = std::max(longest, metric_length(s[j - 1], s[j]));
        }
        if (longest <= settings_.output_spacing) break;
        m = static_cast<int>(std::ceil(m * 1.05 * longest / settings_.output_spacing));
        if (!(m <= kMaxEdgeSamples)) {
          throw std::invalid_argument(
              "smooth_path: output_spacing asks for more than kMaxEdgeSamples samples on a "
              "corner");
        }
        fill();
      }
      bool close = true;
      for (std::size_t j = 0; j + 1 < ts.size() && close; ++j) {
        const Point mid = split_point(c, p, q, f, g, 0.5 * (ts[j] + ts[j + 1]), np, nq);
        close = coord_extent(manifold_.geodesic(s[j], s[j + 1], 0.5), mid) <= tol;
      }
      if (close) return true;
    }
    return false;
  }

  /// Samples the curve at `m + 1` uniform parameters.
  void fill_curve(const Point& c, const Tangent& p, const Tangent& q, const double f,
                  const double g, const int m, Samples& s) const {
    s.clear();
    s.reserve(static_cast<std::size_t>(m) + 1);
    for (int j = 0; j <= m; ++j) {
      s.push_back(curve_point(c, p, q, f, g, static_cast<double>(j) / static_cast<double>(m)));
    }
  }

  /// Samples the curve into `s` with every edge between samples within `corner_tolerance` of
  /// the curve and, with `output_spacing` set, no longer than it. Returns false when a
  /// measured gap stays above the tolerance.
  bool sample_curve(const Point& c, const Tangent& p, const Tangent& q, const double f,
                    const double g, Samples& s) const {
    const double tol = (1.0 - kCurveMerge) * settings_.corner_tolerance;
    // The second derivative of beta is at most 5 max(|f p|, |g q|) in coordinates, and an
    // edge over a parameter step h lies within h^2 / 8 of it from the curve.
    const double e = std::max(f * utils::ordered_norm(p), g * utils::ordered_norm(q));
    const double need = std::ceil(std::sqrt(5.0 * e / (8.0 * tol)));
    if (!(need <= kMaxEdgeSamples)) {
      throw std::invalid_argument(
          "smooth_path: corner_tolerance asks for more than kMaxEdgeSamples samples on a "
          "corner");
    }
    int m = std::max(kCurveMinSamples, static_cast<int>(need));
    for (int round = 0; round <= kCurveRefinements; ++round, m *= 2) {
      fill_curve(c, p, q, f, g, m, s);
      for (int grow = 0; grow < 4 && settings_.output_spacing > 0.0; ++grow) {
        double longest = 0.0;
        for (std::size_t j = 1; j < s.size(); ++j) {
          longest = std::max(longest, metric_length(s[j - 1], s[j]));
        }
        if (longest <= settings_.output_spacing) break;
        m = static_cast<int>(std::ceil(m * 1.05 * longest / settings_.output_spacing));
        if (!(m <= kMaxEdgeSamples)) {
          throw std::invalid_argument(
              "smooth_path: output_spacing asks for more than kMaxEdgeSamples samples on a "
              "corner");
        }
        fill_curve(c, p, q, f, g, m, s);
      }
      bool close = true;
      for (int j = 0; j < m && close; ++j) {
        const Point mid = curve_point(c, p, q, f, g, (static_cast<double>(j) + 0.5) / m);
        const std::size_t i = static_cast<std::size_t>(j);
        close = coord_extent(manifold_.geodesic(s[i], s[i + 1], 0.5), mid) <= tol;
      }
      if (close) return true;
    }
    return false;
  }

  /// True when the samples `s` pass the edge test, middle edges first, and the part of the
  /// waypoint edge from `pts[j]` to `pts[j + 1]` between the parameters `t0` and `t1`, where
  /// the curve starts, passes at the samples of that edge.
  bool curve_valid(const std::vector<Point>& pts, const std::size_t j, const double t0,
                   const double t1, const Samples& s) const {
    if (!oracle_->point(s.front())) return false;
    if (!piece_valid(pts, j, t0, t1)) return false;
    const std::vector<int> order = bisection_order(static_cast<int>(s.size()) - 1);
    for (const int idx : order) {
      const auto j = static_cast<std::size_t>(idx - 1);
      if (!oracle_->edge(s[j], s[j + 1], true)) return false;
    }
    return true;
  }

  /// The path through `pts` with the curves of the corners marked in `use`. With `spaced` and
  /// `output_spacing` set, the edges between curves are resampled, and `filler` marks the points
  /// this adds. `owner` receives for every returned point the part of the path that starts
  /// there, 2 k for the curve of corner k and 2 j + 1 for the waypoint edge from j to j + 1.
  /// `corner` marks the ends, the corners without a curve and the sample `turn_at[k]` of a split
  /// curve.
  std::vector<Point> assemble(const std::vector<Point>& pts, const std::vector<Samples>& curves,
                              const std::vector<std::size_t>& turn_at,
                              const std::vector<unsigned char>& use, const bool spaced,
                              std::vector<std::size_t>* owner = nullptr,
                              std::vector<unsigned char>* corner = nullptr,
                              std::vector<unsigned char>* filler = nullptr) const {
    std::vector<Point> out;
    std::vector<std::size_t> own;
    std::vector<unsigned char> mark;
    std::vector<unsigned char> fill;
    out.push_back(pts.front());
    own.push_back(1);
    mark.push_back(1);
    fill.push_back(0);
    const auto straight_to = [&](const Point& b, const std::size_t edge, const bool end = false) {
      own.back() = 2 * edge + 1;
      if (coord_extent(out.back(), b) <= merge_distance_) {
        // The path ends exactly at the last waypoint.
        if (end && out.size() > 1) {
          out.back() = b;
          mark.back() = 1;
          fill.back() = 0;
        } else if (end) {
          out.push_back(b);
          own.push_back(2 * edge + 1);
          mark.push_back(1);
          fill.push_back(0);
        }
        return;
      }
      if (spaced && settings_.output_spacing > 0.0) {
        const std::vector<Point> piece = resample_points({out.back(), b});
        for (std::size_t i = 1; i + 1 < piece.size(); ++i) {
          out.push_back(piece[i]);
          own.push_back(2 * edge + 1);
          mark.push_back(0);
          fill.push_back(1);
        }
      }
      out.push_back(b);
      own.push_back(2 * edge + 1);
      mark.push_back(0);
      fill.push_back(0);
    };
    for (std::size_t k = 1; k + 1 < pts.size(); ++k) {
      if (!use[k]) {
        straight_to(pts[k], k - 1);
        mark.back() = 1;
        continue;
      }
      const Samples& s = curves[k];
      straight_to(s.front(), k - 1);
      own.back() = 2 * k;
      for (std::size_t j = 1; j < s.size(); ++j) {
        out.push_back(s[j]);
        own.push_back(2 * k);
        mark.push_back(j == turn_at[k] ? 1 : 0);
        fill.push_back(0);
      }
    }
    straight_to(pts.back(), pts.size() - 2, true);
    if (owner) *owner = std::move(own);
    if (corner) *corner = std::move(mark);
    if (filler) *filler = std::move(fill);
    return out;
  }

  const M& manifold_;
  const ValidityFn& validity_;
  const PathSmoothingSettings& settings_;
  const Oracle<M, ValidityFn>* oracle_ = nullptr;
  PathSmoothingProfile* prof_ = nullptr;
  double quad_step_ = 0.0;
  double merge_distance_ = 0.0;  // coordinate distance below which two points coincide
  bool intrinsic_ = false;       // tangent coordinates have dim() entries
  int sharp_ = 0;                // leading coordinates that may keep a corner
  mutable std::vector<signed char> whole_edge_ok_;  // edge validator's answer per waypoint edge
};

}  // namespace detail

/// @brief Shorten and smooth a valid path under the manifold's metric.
///
/// @details Alternates shortcutting with local descent on the discrete path energy
/// \f$ \sum_k \|\log_{q_k} q_{k+1}\|^2_{q_k} \f$, then rounds the corners into C² curves.
/// The energy descent needs `dim()` tangent coordinates. Other manifolds only shortcut and
/// round corners.
///
/// @note Validity is checked at `collision_check_resolution`, and a path may touch an obstacle
/// between two checks.
///
/// @tparam M A `RiemannianManifold`.
/// @tparam ValidityFn Callable `bool(const Point&)`, true for a valid point. A functor that
/// also satisfies `BatchValidity` receives the edge checks in blocks.
/// @param manifold The manifold.
/// @param validity Point validity.
/// @param path Input path, such as a planner's output.
/// @param settings Smoother settings.
/// @return The smoothed path at equal steps. `collision_free` is true when every waypoint of
/// the smoothed path passes `validity` and every edge, interpolated with `manifold.geodesic()`,
/// passes the edge test. Next to a rounding curve, a part of an edge of the shortened path passes
/// at the samples of the whole edge, and an edge validator tests the whole edge. The returned
/// waypoints lie on that path, and the edges between them stay within `corner_tolerance` of it.
/// A smoothed path that fails falls back to an earlier stage, down to the input.
template <RiemannianManifold M, typename ValidityFn>
PathSmoothingResult<typename M::Point> smooth_path(const M& manifold, const ValidityFn& validity,
                                                   const std::vector<typename M::Point>& path,
                                                   const PathSmoothingSettings& settings = {}) {
  if (!std::isfinite(settings.collision_check_resolution) ||
      settings.collision_check_resolution < 0.0) {
    throw std::invalid_argument("smooth_path: collision_check_resolution must be finite and >= 0");
  }
  if (!std::isfinite(settings.output_spacing) || settings.output_spacing < 0.0) {
    throw std::invalid_argument("smooth_path: output_spacing must be finite and >= 0");
  }
  if (!(std::isfinite(settings.corner_tolerance) && settings.corner_tolerance > 0.0)) {
    throw std::invalid_argument("smooth_path: corner_tolerance must be finite and > 0");
  }
  if (settings.round_corners &&
      !(settings.corner_max_angle >= 0.0 && settings.corner_max_angle <= std::numbers::pi)) {
    throw std::invalid_argument("smooth_path: corner_max_angle must lie in [0, pi]");
  }
  if (settings.sharp_coordinates < 0) {
    throw std::invalid_argument("smooth_path: sharp_coordinates must be >= 0");
  }
  return detail::Smoother<M, ValidityFn>(manifold, validity, settings).run(path);
}

}  // namespace geodex::algorithm
