/// @file geodex_state_space.hpp
/// @brief OMPL integration that adapts geodex manifolds to ompl::base::StateSpace.

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#include <algorithm>
#include <limits>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <ompl/base/StateSampler.h>
#include <ompl/base/StateSpace.h>
#include <ompl/base/spaces/RealVectorBounds.h>

#include "geodex/algorithm/interpolation.hpp"
#include "geodex/core/concepts.hpp"
#include "geodex/core/metric.hpp"
#include "geodex/core/sampler.hpp"
#include "geodex/utils/random.hpp"

namespace geodex::integration::ompl {

using geodex::discrete_geodesic;
using geodex::InterpolationCache;
using geodex::InterpolationSettings;
using geodex::InterpolationStatus;
using geodex::is_riemannian_log;
using geodex::RiemannianManifold;

namespace ob = ::ompl::base;

namespace detail {

/// @brief Guards OMPL's process-wide seed generator.
///
/// @details Each OMPL `RNG` takes its seed from that generator when it is created.
/// `planning::plan()` reseeds the generator and creates its planner under this lock,
/// and a `GeodexStateSpace` allocates its state samplers under it. Concurrent plans do
/// not take each other's seeds.
inline std::recursive_mutex& ompl_rng_mutex() {
  static std::recursive_mutex mutex;
  return mutex;
}

}  // namespace detail

/// @brief Interpolation strategy for GeodexStateSpace::interpolate().
enum class InterpolationMode {
  Auto,                ///< Identity metric -> base geodesic, custom metric -> discrete geodesic.
  BaseGeodesic,        ///< Always use closed-form manifold.geodesic(p, q, t).
  RiemannianGeodesic,  ///< Always use iterative discrete geodesic with FD natural gradient.
};

// ---------------------------------------------------------------------------
// GeodesicPathCache, which amortizes discrete_geodesic across same-endpoint queries
// ---------------------------------------------------------------------------

/// @brief Caches a discrete geodesic path between two points for efficient
/// arc-length-parameterized lookups.
///
/// @details OMPL's `DiscreteMotionValidator` calls `interpolate(s1, s2, j/n)`
/// for `j = 1..n-1` with the same `(s1, s2)` pair. This cache computes the
/// full discrete geodesic once and serves later lookups by binary search on
/// cumulative arc lengths, O(log K) per query.
///
/// @note Not thread-safe. The owning `GeodexStateSpace` stores this cache as a
/// `mutable` member and mutates it from the `const` `interpolate()` method. OMPL's
/// standard planners run one `interpolate()` call per state-space instance at a time.
/// Concurrent planners must synchronize externally or use one state space per thread.
///
/// @tparam M A type satisfying `RiemannianManifold`.
template <RiemannianManifold M>
class GeodesicPathCache {
 public:
  using Point = typename M::Point;    ///< Manifold point type.
  using Scalar = typename M::Scalar;  ///< Manifold scalar type.

  /// @brief Check whether the cache holds a valid path for the given endpoints.
  /// @param f Start point.
  /// @param t Target point.
  /// @param dim Ambient dimension (number of doubles to compare).
  /// @return True if the cache matches and is valid.
  bool matches(const Point& f, const Point& t, unsigned int dim) const {
    if (!valid_) return false;
    return std::memcmp(from_.data(), f.data(), dim * sizeof(double)) == 0 &&
           std::memcmp(to_.data(), t.data(), dim * sizeof(double)) == 0;
  }

  /// @brief Compute the discrete geodesic from \p f to \p t and cache the result.
  /// @param manifold The manifold instance.
  /// @param f Start point.
  /// @param t Target point.
  /// @param settings Interpolation settings (step_size controls path resolution).
  void compute(const M& manifold, const Point& f, const Point& t,
               const InterpolationSettings& settings) {
    from_ = f;
    to_ = t;
    valid_ = false;

    auto result = discrete_geodesic(manifold, f, t, settings, &interp_cache_);

    // Accept Converged, MaxStepsReached and DegenerateInput.
    // Reject CutLocus, GradientVanished and StepShrunkToZero, and the caller falls back.
    if (result.status == InterpolationStatus::CutLocus ||
        result.status == InterpolationStatus::GradientVanished ||
        result.status == InterpolationStatus::StepShrunkToZero) {
      return;
    }

    waypoints_ = std::move(result.path);

    // Ensure the target is the final waypoint for exact t=1 lookup.
    if (waypoints_.empty()) {
      waypoints_.push_back(f);
      waypoints_.push_back(t);
    } else if ((waypoints_.back() - t).norm() > 1e-12) {
      waypoints_.push_back(t);
    }

    // Build cumulative arc-length table.
    const auto n = waypoints_.size();
    cum_arc_.resize(n);
    cum_arc_[0] = 0.0;
    for (std::size_t i = 1; i < n; ++i) {
      cum_arc_[i] = cum_arc_[i - 1] +
                    static_cast<double>(manifold.distance(waypoints_[i - 1], waypoints_[i]));
    }
    total_arc_ = cum_arc_.back();

    valid_ = true;
  }

  /// @brief Look up the point at arc-length fraction \p t along the cached path.
  /// @param manifold The manifold instance (for local geodesic interpolation).
  /// @param t Parameter in [0, 1] (0 = start, 1 = target).
  /// @return The interpolated point on the cached geodesic path.
  Point at(const M& manifold, double t) const {
    if (t <= 0.0) return waypoints_.front();
    if (t >= 1.0) return waypoints_.back();

    if (waypoints_.size() <= 1 || total_arc_ <= 0.0) {
      return waypoints_.front();
    }

    const double target_s = t * total_arc_;

    // Binary search for the first element > target_s.
    const auto it = std::upper_bound(cum_arc_.begin(), cum_arc_.end(), target_s);
    auto idx = static_cast<int>(it - cum_arc_.begin()) - 1;
    idx = std::max(0, std::min(idx, static_cast<int>(waypoints_.size()) - 2));

    const double seg_len = cum_arc_[idx + 1] - cum_arc_[idx];
    const double t_local = (seg_len > 1e-15) ? (target_s - cum_arc_[idx]) / seg_len : 0.0;

    // Local geodesic between adjacent waypoints, accurate to the retraction.
    return manifold.geodesic(waypoints_[idx], waypoints_[idx + 1], t_local);
  }

  /// @brief Whether the cache holds a valid path.
  bool valid() const { return valid_; }

  /// @brief Total Riemannian arc length along the cached discrete geodesic.
  ///
  /// @details Sum of per-segment `manifold.distance(waypoints[i], waypoints[i+1])`
  /// values computed when the path was cached. For a valid cache this is the cost
  /// of the arc under the configured metric. An invalid cache returns 0.0.
  double total_arc_cost() const { return total_arc_; }

 private:
  Point from_;
  Point to_;
  std::vector<Point> waypoints_;
  std::vector<double> cum_arc_;
  double total_arc_ = 0.0;
  bool valid_ = false;
  InterpolationCache<M> interp_cache_;
};

// ---------------------------------------------------------------------------
// GeodexStateSpace, which adapts a geodex RiemannianManifold to ompl::base::StateSpace
// ---------------------------------------------------------------------------

template <typename ManifoldT>
  requires geodex::RiemannianManifold<ManifoldT>
class GeodexStateSpace;

// ---------------------------------------------------------------------------
// GeodexState, the state type that stores ambient-space coordinates
// ---------------------------------------------------------------------------

/// @brief State type for GeodexStateSpace, storing ambient-space coordinates.
///
/// @details Wraps a raw `double*` array with Eigen map accessors for
/// convenient conversion between OMPL states and geodex manifold points.
///
/// @tparam ManifoldT The geodex manifold type.
template <typename ManifoldT>
class GeodexState : public ob::State {
 public:
  double* values = nullptr;  ///< Raw coordinate array (owned by the state space).
  unsigned int dim = 0;      ///< Runtime ambient dimension, required for dynamic-size Point types.

  /// @brief Read-only Eigen map of the state coordinates.
  /// @details Returns a non-owning `Eigen::Map<const Vector>` view onto `values`.
  /// To mutate the state, write through `values` or an explicit `Eigen::Map<Vector>`.
  auto asEigen() const {
    using Point = typename ManifoldT::Point;
    constexpr int Dim = Point::SizeAtCompileTime;
    if constexpr (Dim == Eigen::Dynamic) {
      return Eigen::Map<const Eigen::Vector<double, Dim>>(values, dim);
    } else {
      return Eigen::Map<const Eigen::Vector<double, Dim>>(values);
    }
  }
};

// ---------------------------------------------------------------------------
// GeodexStateSampler
// ---------------------------------------------------------------------------

/// @brief State sampler for GeodexStateSpace.
///
/// @details Samples uniformly through one stream of the manifold's own sampler
/// (see `GeodexStateSpace::allocManifoldSampler`), and near-uniformly or from a
/// Gaussian by sampling tangent vectors and applying the exponential map. The
/// tangent samples come from a generator seeded from the same stream and reproduce
/// on any machine.
///
/// @tparam ManifoldT The geodex manifold type.
template <typename ManifoldT>
class GeodexStateSampler : public ob::StateSampler {
  using StateSpace = GeodexStateSpace<ManifoldT>;
  using StateType = GeodexState<ManifoldT>;

 public:
  /// @brief Construct a sampler on the next sampler stream of `space`.
  /// @param space The GeodexStateSpace to sample from.
  /// @throws std::invalid_argument when `space` is not a `GeodexStateSpace<ManifoldT>`.
  explicit GeodexStateSampler(const ob::StateSpace* space)
      : GeodexStateSampler(space, checked(space)->nextStreamSeed()) {}

  /// @brief Sample a state uniformly on the manifold via its from_unit_cube map.
  ///
  /// @details Samples a low-discrepancy point in the unit cube and maps it through the
  /// manifold's measure-preserving map. Curved manifolds are sampled uniformly in their
  /// own volume, not in raw ambient coordinates.
  void sampleUniform(ob::State* state) override {
    auto* s = state->as<StateType>();
    const auto* space = static_cast<const StateSpace*>(space_);
    const auto& manifold = space->getManifold();
    const int cube_dim = manifold.unit_cube_dim();
    cube_buf_.resize(cube_dim);
    sampler_.sample(cube_dim, cube_buf_);
    const auto point = manifold.from_unit_cube(cube_buf_);
    const unsigned int dim = space->getDimension();
    for (unsigned int i = 0; i < dim; ++i) {
      s->values[i] = point[static_cast<int>(i)];
    }
  }

  /// @brief Sample a state uniformly near another state at a given distance.
  /// @param state Output state.
  /// @param near Reference state to sample around.
  /// @param distance Desired geodesic distance from \p near.
  void sampleUniformNear(ob::State* state, const ob::State* near, double distance) override {
    auto* s = state->as<StateType>();
    const auto* n = near->as<StateType>();
    const auto* space = static_cast<const StateSpace*>(space_);
    const auto& manifold = space->getManifold();
    const auto& bounds = space->getBounds();
    const unsigned int dim = space->getDimension();

    // Sample a random tangent vector at 'near', scale to desired distance
    using Point = typename ManifoldT::Point;
    using Tangent = typename ManifoldT::Tangent;
    Point p_near = n->asEigen();

    Tangent v;
    if constexpr (Point::SizeAtCompileTime == Eigen::Dynamic) {
      v.resize(dim);
    }
    for (unsigned int i = 0; i < dim; ++i) {
      v[i] = geodex::utils::normal(tangent_rng_, 0.0, 1.0);
    }
    double v_norm = manifold.norm(p_near, v);
    if (v_norm > 1e-12) {
      v *= distance / v_norm;
    }

    Point result = manifold.exp(p_near, v);

    // Clamp to bounds
    for (unsigned int i = 0; i < dim; ++i) {
      s->values[i] = std::clamp(result[i], bounds.low[i], bounds.high[i]);
    }
  }

  /// @brief Sample a state from a Gaussian distribution centered at \p mean.
  /// @param state Output state.
  /// @param mean Center of the Gaussian.
  /// @param stdDev Standard deviation in the tangent space.
  void sampleGaussian(ob::State* state, const ob::State* mean, double stdDev) override {
    auto* s = state->as<StateType>();
    const auto* m = mean->as<StateType>();
    const auto* space = static_cast<const StateSpace*>(space_);
    const auto& manifold = space->getManifold();
    const auto& bounds = space->getBounds();
    const unsigned int dim = space->getDimension();

    using Point = typename ManifoldT::Point;
    using Tangent = typename ManifoldT::Tangent;
    Point p_mean = m->asEigen();

    Tangent v;
    if constexpr (Point::SizeAtCompileTime == Eigen::Dynamic) {
      v.resize(dim);
    }
    for (unsigned int i = 0; i < dim; ++i) {
      v[i] = geodex::utils::normal(tangent_rng_, 0.0, stdDev);
    }

    Point result = manifold.exp(p_mean, v);

    for (unsigned int i = 0; i < dim; ++i) {
      s->values[i] = std::clamp(result[i], bounds.low[i], bounds.high[i]);
    }
  }

 private:
  GeodexStateSampler(const ob::StateSpace* space, const std::uint64_t seed)
      : ob::StateSampler(space),
        sampler_(checked(space)->manifoldSampler(seed)),
        tangent_rng_(seed ^ 0x9E3779B97F4A7C15ULL) {}

  static const StateSpace* checked(const ob::StateSpace* space) {
    const auto* s = dynamic_cast<const StateSpace*>(space);
    if (s == nullptr) {
      throw std::invalid_argument(
          "GeodexStateSampler: space is not a GeodexStateSpace of this "
          "manifold type");
    }
    return s;
  }

  typename ManifoldT::SamplerType sampler_;  ///< one stream of the manifold's sampler
  std::mt19937_64 tangent_rng_;              ///< tangent samples of the near and Gaussian samplers
  Eigen::VectorXd cube_buf_;                 ///< Reused unit-cube buffer.
};

// ---------------------------------------------------------------------------
// GeodexStateSpace
// ---------------------------------------------------------------------------

/// @brief OMPL state space adapter for geodex Riemannian manifolds.
///
/// @details Wraps a geodex `RiemannianManifold` as an `ompl::base::StateSpace`,
/// delegating distance, interpolation, and sampling to the manifold's operations.
/// States store ambient-space coordinates as raw `double*` arrays.
///
/// @tparam ManifoldT A type satisfying `geodex::RiemannianManifold`.
template <typename ManifoldT>
  requires geodex::RiemannianManifold<ManifoldT>
class GeodexStateSpace : public ob::StateSpace {
 public:
  using Point = typename ManifoldT::Point;              ///< Manifold point type.
  using Tangent = typename ManifoldT::Tangent;          ///< Manifold tangent type.
  using SamplerType = typename ManifoldT::SamplerType;  ///< Manifold sampler type.
  using StateType = GeodexState<ManifoldT>;             ///< OMPL state type.

  /// @brief Construct a state space from a manifold and bounds.
  /// @param manifold The geodex manifold instance.
  /// @param bounds Axis-aligned bounds for the ambient coordinates.
  GeodexStateSpace(ManifoldT manifold, ob::RealVectorBounds bounds)
      : manifold_(std::move(manifold)), bounds_(std::move(bounds)) {
    setName("GeodexStateSpace");
    type_ = ob::STATE_SPACE_UNKNOWN;

    if constexpr (Point::SizeAtCompileTime != Eigen::Dynamic) {
      ambient_dim_ = static_cast<unsigned int>(Point::SizeAtCompileTime);
    } else {
      // Read the ambient size from an actual point. A type-erased embedded manifold
      // stores more ambient coordinates than its intrinsic dimension.
      const int cube_dim = manifold_.unit_cube_dim();
      ambient_dim_ = static_cast<unsigned int>(
          manifold_.from_unit_cube(Eigen::VectorXd::Constant(cube_dim, 0.5)).size());
    }

    if (bounds_.low.size() != ambient_dim_ || bounds_.high.size() != ambient_dim_) {
      throw std::invalid_argument(
          "GeodexStateSpace: bounds have " + std::to_string(bounds_.low.size()) + " lower and " +
          std::to_string(bounds_.high.size()) + " upper entries, the manifold's points have " +
          std::to_string(ambient_dim_) + " coordinates");
    }
  }

  /// @brief Access the underlying geodex manifold.
  const ManifoldT& getManifold() const { return manifold_; }

  /// @brief Access the coordinate bounds.
  const ob::RealVectorBounds& getBounds() const { return bounds_; }

  /// @brief Seed the streams of the samplers this space allocates after this call.
  ///
  /// @details Every sampler a planner allocates through this space, the state
  /// samplers and the informed sampler, is a copy of the manifold's sampler
  /// on its own stream. The sampler kind carries over, and the streams stay
  /// independent. The stream seeds come from a generator seeded here, or without a
  /// seed from geodex's thread-local seed source (`geodex::set_default_seed`).
  void setSamplerSeed(const std::uint64_t seed) {
    const std::lock_guard lock(stream_mutex_);
    stream_seeds_.emplace(seed);
  }

  /// @brief Seed of the next sampler stream. Safe to call from several threads.
  std::uint64_t nextStreamSeed() const {
    const std::lock_guard lock(stream_mutex_);
    return stream_seeds_ ? (*stream_seeds_)() : geodex::detail::seed_source()();
  }

  /// @brief A sampler for one new stream, a copy of the manifold's sampler reseeded
  /// to the next stream seed.
  ///
  /// @details A manifold without `sampler()` contributes a default-constructed
  /// `SamplerType`. A sampler without `seed()` is copied unchanged, and every stream
  /// repeats the same sequence.
  SamplerType allocManifoldSampler() const { return manifoldSampler(nextStreamSeed()); }

  /// @brief A copy of the manifold's sampler reseeded to `seed`, or copied unchanged
  /// when the sampler does not have `seed()`.
  SamplerType manifoldSampler(const std::uint64_t seed) const {
    SamplerType out = [&]() -> SamplerType {
      if constexpr (geodex::HasSampler<ManifoldT>) {
        return manifold_.sampler();
      } else {
        return SamplerType{};
      }
    }();
    if constexpr (geodex::SeedableSampler<SamplerType>) out.seed(seed);
    return out;
  }

  /// @brief Set the minimum collision checking resolution in coordinate distance.
  ///
  /// @details When set to a positive value, `validSegmentCount()` ensures at
  /// least `ceil(coord_distance / resolution)` checks along each edge,
  /// independent of OMPL's `longestValidSegmentFraction`. This prevents thin
  /// walls from being missed when `getMaximumExtent()` is large.
  ///
  /// @param resolution Minimum distance (meters) between collision checks.
  ///        Use 0.0 to disable (OMPL default only).
  /// @throws std::invalid_argument when `resolution` is negative or not finite.
  void setCollisionResolution(const double resolution) {
    if (!(resolution >= 0.0) || !std::isfinite(resolution)) {
      throw std::invalid_argument("GeodexStateSpace: collision resolution must be finite and >= 0");
    }
    collision_resolution_ = resolution;
  }

  /// @brief Get the collision checking resolution.
  double getCollisionResolution() const { return collision_resolution_; }

  /// @brief Set the interpolation strategy.
  /// @param mode The desired interpolation mode.
  void setInterpolationMode(const InterpolationMode mode) { interpolation_mode_ = mode; }

  /// @brief Get the current interpolation mode.
  InterpolationMode getInterpolationMode() const { return interpolation_mode_; }

  /// @brief Set the interpolation settings for discrete geodesic computation.
  /// @param settings The settings (step_size, convergence_tol, max_steps, etc.).
  void setInterpolationSettings(const InterpolationSettings& settings) {
    interpolation_settings_ = settings;
  }

  /// @brief Get the current interpolation settings.
  const InterpolationSettings& getInterpolationSettings() const { return interpolation_settings_; }

  /// @brief Set the step size for discrete geodesic interpolation.
  ///
  /// @details Controls the maximum Riemannian distance between consecutive
  /// waypoints in the cached geodesic path. Smaller values increase resolution
  /// (and computation cost). Default is 0.5.
  ///
  /// @param step_size Maximum step size per iteration.
  void setGeodesicStepSize(const double step_size) {
    interpolation_settings_.step_size = step_size;
  }

  // -- StateSpace interface --

  /// @brief Return the ambient dimension of the state space.
  unsigned int getDimension() const override { return ambient_dim_; }

  /// @brief Return the maximum extent of the state space.
  ///
  /// @details Returns the larger of the coordinate diagonal and the Riemannian
  /// distance between the bounding-box corners. The second term covers anisotropic
  /// metrics, whose Riemannian extent can exceed the coordinate diagonal.
  ///
  /// @note Under angle wrapping or a configuration-dependent metric the corner
  /// distance can underestimate the extent.
  double getMaximumExtent() const override {
    // Coordinate diagonal (baseline for isotropic metrics)
    double diag2 = 0.0;
    for (unsigned int i = 0; i < ambient_dim_; ++i) {
      double d = bounds_.high[i] - bounds_.low[i];
      diag2 += d * d;
    }
    double coord_extent = std::sqrt(diag2);

    // Riemannian distance between bounding box corners
    Point lo, hi;
    if constexpr (Point::SizeAtCompileTime == Eigen::Dynamic) {
      lo.resize(ambient_dim_);
      hi.resize(ambient_dim_);
    }
    for (unsigned int i = 0; i < ambient_dim_; ++i) {
      lo[i] = bounds_.low[i];
      hi[i] = bounds_.high[i];
    }
    double riem_extent = manifold_.distance(lo, hi);

    return std::max(coord_extent, riem_extent);
  }

  /// @brief Return the volume of the bounding box.
  double getMeasure() const override {
    double vol = 1.0;
    for (unsigned int i = 0; i < ambient_dim_; ++i) {
      vol *= bounds_.high[i] - bounds_.low[i];
    }
    return vol;
  }

  /// @brief Clamp state coordinates to the bounds.
  void enforceBounds(ob::State* state) const override {
    auto* s = state->as<StateType>();
    for (unsigned int i = 0; i < ambient_dim_; ++i) {
      s->values[i] = std::clamp(s->values[i], bounds_.low[i], bounds_.high[i]);
    }
  }

  /// @brief Check whether all coordinates satisfy the bounds.
  bool satisfiesBounds(const ob::State* state) const override {
    const auto* s = state->as<StateType>();
    for (unsigned int i = 0; i < ambient_dim_; ++i) {
      if (s->values[i] < bounds_.low[i] - std::numeric_limits<double>::epsilon() ||
          s->values[i] > bounds_.high[i] + std::numeric_limits<double>::epsilon()) {
        return false;
      }
    }
    return true;
  }

  /// @brief Copy state coordinates.
  void copyState(ob::State* destination, const ob::State* source) const override {
    auto* dst = destination->as<StateType>();
    const auto* src = source->as<StateType>();
    std::memcpy(dst->values, src->values, ambient_dim_ * sizeof(double));
  }

  /// @brief Compute the geodesic distance between two states.
  double distance(const ob::State* state1, const ob::State* state2) const override {
    const auto* s1 = state1->as<StateType>();
    const auto* s2 = state2->as<StateType>();
    return manifold_.distance(s1->asEigen(), s2->asEigen());
  }

  /// @brief Check whether two states are equal (within tolerance).
  bool equalStates(const ob::State* state1, const ob::State* state2) const override {
    const auto* s1 = state1->as<StateType>();
    const auto* s2 = state2->as<StateType>();
    for (unsigned int i = 0; i < ambient_dim_; ++i) {
      if (std::abs(s1->values[i] - s2->values[i]) > 1e-12) return false;
    }
    return true;
  }

  /// @brief Geodesic interpolation between two states.
  ///
  /// @details When `is_riemannian_log()` is true (identity metric with matching
  /// retraction), uses `geodesic(p, q, t)` directly. For other metrics, computes a
  /// discrete geodesic by Riemannian natural gradient descent, caches it by the
  /// (from, to) pair and looks up by arc length. Repeated calls with one pair, as in
  /// `DiscreteMotionValidator::checkMotion`, share the computation.
  ///
  /// @param from Start state.
  /// @param to End state.
  /// @param t Interpolation parameter in [0, 1].
  /// @param state Output state.
  void interpolate(const ob::State* from, const ob::State* to, double t,
                   ob::State* state) const override {
    const auto* f = from->as<StateType>();
    const auto* tgt = to->as<StateType>();
    auto* s = state->as<StateType>();

    const bool use_base_geodesic =
        (interpolation_mode_ == InterpolationMode::BaseGeodesic) ||
        (interpolation_mode_ == InterpolationMode::Auto && is_riemannian_log(manifold_));
    if (use_base_geodesic) {
      Point result = manifold_.geodesic(f->asEigen(), tgt->asEigen(), t);
      for (unsigned int i = 0; i < ambient_dim_; ++i) s->values[i] = result[i];
      return;
    }

    // The endpoints do not need the cache.
    if (t <= 0.0) {
      copyState(state, from);
      return;
    }
    if (t >= 1.0) {
      copyState(state, to);
      return;
    }

    // Compute the cache on a miss.
    Point p_from = f->asEigen();
    Point p_to = tgt->asEigen();
    if (!geodesic_cache_.matches(p_from, p_to, ambient_dim_)) {
      geodesic_cache_.compute(manifold_, p_from, p_to, interpolation_settings_);
    }

    // Fall back on a convergence failure (cut locus, vanished gradient, ...).
    if (!geodesic_cache_.valid()) {
      Point result = manifold_.geodesic(p_from, p_to, t);
      for (unsigned int i = 0; i < ambient_dim_; ++i) s->values[i] = result[i];
      return;
    }

    // Arc-length lookup from cache.
    Point result = geodesic_cache_.at(manifold_, t);
    for (unsigned int i = 0; i < ambient_dim_; ++i) s->values[i] = result[i];
  }

  /// @brief Allocate the default state sampler.
  ///
  /// @details Creates the sampler under `detail::ompl_rng_mutex()`. OMPL's base
  /// sampler takes a seed from OMPL's shared generator on construction.
  ob::StateSamplerPtr allocDefaultStateSampler() const override {
    const std::lock_guard lock(detail::ompl_rng_mutex());
    return std::make_shared<GeodexStateSampler<ManifoldT>>(this);
  }

  /// @brief Allocate a new state.
  ob::State* allocState() const override {
    auto* s = new StateType();
    s->values = new double[ambient_dim_];
    s->dim = ambient_dim_;
    return s;
  }

  /// @brief Free a state.
  void freeState(ob::State* state) const override {
    auto* s = state->as<StateType>();
    delete[] s->values;
    delete s;
  }

  /// @brief Get a pointer to the coordinate at the given index.
  double* getValueAddressAtIndex(ob::State* state, unsigned int index) const override {
    if (index >= ambient_dim_) return nullptr;
    return state->as<StateType>()->values + index;
  }

  /// @brief Copy state coordinates to a vector of doubles.
  void copyToReals(std::vector<double>& reals, const ob::State* source) const override {
    const auto* s = source->as<StateType>();
    reals.resize(ambient_dim_);
    for (unsigned int i = 0; i < ambient_dim_; ++i) reals[i] = s->values[i];
  }

  /// @brief Set state coordinates from a vector of doubles.
  void copyFromReals(ob::State* destination, const std::vector<double>& reals) const override {
    auto* d = destination->as<StateType>();
    for (unsigned int i = 0; i < ambient_dim_; ++i) d->values[i] = reals[i];
  }

  /// @brief Read-only access to the internal cached discrete-geodesic path.
  ///
  /// @details `GeodexOptimizationObjective` reads it for the integrated arc cost of
  /// the last `interpolate()` pair. The cache holds at most one (s1, s2) pair. Check
  /// `matches()` before using it.
  const GeodesicPathCache<ManifoldT>& getGeodesicCache() const { return geodesic_cache_; }

  /// @brief Fill the cache for the given endpoint pair, or do nothing on a hit.
  ///
  /// @details The optimization objective calls it for the arc cost of a pair that
  /// is not cached yet. It uses the configured `InterpolationSettings`.
  void ensureGeodesicCached(const Point& from, const Point& to) const {
    if (!geodesic_cache_.matches(from, to, ambient_dim_)) {
      geodesic_cache_.compute(manifold_, from, to, interpolation_settings_);
    }
  }

  /// @brief Number of collision checks for a motion between two states.
  ///
  /// @details Returns the larger of OMPL's default count (metric distance over
  /// `getLongestValidSegmentLength()`) and the coordinate distance over
  /// `collision_resolution_`, or over the same fraction of the bounds' coordinate
  /// diagonal when that is 0. A metric that makes a far coordinate motion short does
  /// not thin out the checks. The count is at most the largest `unsigned int`.
  unsigned int validSegmentCount(const ob::State* s1, const ob::State* s2) const override {
    unsigned int n = ob::StateSpace::validSegmentCount(s1, s2);
    const auto* a = s1->as<StateType>();
    const auto* b = s2->as<StateType>();
    double dist2 = 0.0, diag2 = 0.0;
    for (unsigned int i = 0; i < ambient_dim_; ++i) {
      const double d = a->values[i] - b->values[i];
      const double w = bounds_.high[i] - bounds_.low[i];
      dist2 += d * d;
      diag2 += w * w;
    }
    const double spacing = collision_resolution_ > 0.0
                               ? collision_resolution_
                               : getLongestValidSegmentFraction() * std::sqrt(diag2);
    if (spacing > 0.0 && std::isfinite(spacing)) {
      const double n_coord = std::ceil(std::sqrt(dist2) / spacing);
      constexpr auto kMax = std::numeric_limits<unsigned int>::max();
      n = n_coord >= static_cast<double>(kMax) ? kMax
                                               : std::max(n, static_cast<unsigned int>(n_coord));
    }
    return std::max(n, 1u);
  }

 private:
  ManifoldT manifold_;
  ob::RealVectorBounds bounds_;
  unsigned int ambient_dim_;
  double collision_resolution_ = 0.0;
  InterpolationMode interpolation_mode_ = InterpolationMode::Auto;
  InterpolationSettings interpolation_settings_;
  mutable GeodesicPathCache<ManifoldT> geodesic_cache_;
  mutable std::optional<std::mt19937_64> stream_seeds_;  ///< set by setSamplerSeed
  mutable std::mutex stream_mutex_;                      ///< guards stream_seeds_
};

}  // namespace geodex::integration::ompl
