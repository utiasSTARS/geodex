/// @file geodex_state_space.hpp
/// @brief OMPL integration: adapts geodex manifolds to ompl::base::StateSpace.

#pragma once

#include <ompl/base/StateSampler.h>
#include <ompl/base/StateSpace.h>
#include <ompl/base/spaces/RealVectorBounds.h>

#include <Eigen/Core>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <geodex/core/concepts.hpp>
#include <limits>

namespace geodex {
namespace ompl_integration {

namespace ob = ompl::base;

// ---------------------------------------------------------------------------
// GeodexStateSpace — adapts a geodex RiemannianManifold to ompl::base::StateSpace
// ---------------------------------------------------------------------------

template <typename ManifoldT>
  requires geodex::RiemannianManifold<ManifoldT>
class GeodexStateSpace;

// ---------------------------------------------------------------------------
// GeodexState — state type storing ambient-space coordinates
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

  /// @brief Read-only Eigen map of the state coordinates.
  auto asEigen() const {
    using Point = typename ManifoldT::Point;
    constexpr int Dim = Point::SizeAtCompileTime;
    return Eigen::Map<const Eigen::Vector<double, Dim>>(values);
  }

  /// @brief Mutable Eigen map of the state coordinates.
  auto asEigen() {
    using Point = typename ManifoldT::Point;
    constexpr int Dim = Point::SizeAtCompileTime;
    return Eigen::Map<Eigen::Vector<double, Dim>>(values);
  }
};

// ---------------------------------------------------------------------------
// GeodexStateSampler
// ---------------------------------------------------------------------------

/// @brief State sampler for GeodexStateSpace.
///
/// @details Provides uniform, near-uniform, and Gaussian sampling on the
/// manifold by sampling tangent vectors and applying the exponential map.
///
/// @tparam ManifoldT The geodex manifold type.
template <typename ManifoldT>
class GeodexStateSampler : public ob::StateSampler {
  using StateSpace = GeodexStateSpace<ManifoldT>;
  using StateType = GeodexState<ManifoldT>;

 public:
  /// @brief Construct a sampler for the given state space.
  /// @param space The GeodexStateSpace to sample from.
  explicit GeodexStateSampler(const ob::StateSpace* space) : ob::StateSampler(space) {}

  /// @brief Sample a state uniformly within the bounds.
  void sampleUniform(ob::State* state) override {
    auto* s = state->as<StateType>();
    const auto* space = static_cast<const StateSpace*>(space_);
    const auto& bounds = space->getBounds();
    const unsigned int dim = space->getDimension();
    for (unsigned int i = 0; i < dim; ++i) {
      s->values[i] = rng_.uniformReal(bounds.low[i], bounds.high[i]);
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
      v[i] = rng_.gaussian01();
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
      v[i] = rng_.gaussian(0.0, stdDev);
    }

    Point result = manifold.exp(p_mean, v);

    for (unsigned int i = 0; i < dim; ++i) {
      s->values[i] = std::clamp(result[i], bounds.low[i], bounds.high[i]);
    }
  }
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
  using Point = typename ManifoldT::Point;      ///< Manifold point type.
  using Tangent = typename ManifoldT::Tangent;  ///< Manifold tangent type.
  using StateType = GeodexState<ManifoldT>;     ///< OMPL state type.

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
      ambient_dim_ = static_cast<unsigned int>(manifold_.dim());
    }

    assert(bounds_.low.size() == ambient_dim_);
    assert(bounds_.high.size() == ambient_dim_);
  }

  /// @brief Access the underlying geodex manifold.
  const ManifoldT& getManifold() const { return manifold_; }

  /// @brief Access the coordinate bounds.
  const ob::RealVectorBounds& getBounds() const { return bounds_; }

  /// @brief Set the minimum collision checking resolution in coordinate distance.
  ///
  /// @details When set to a positive value, `validSegmentCount()` ensures at
  /// least `ceil(coord_distance / resolution)` checks along each edge,
  /// independent of OMPL's `longestValidSegmentFraction`. This prevents thin
  /// walls from being missed when `getMaximumExtent()` is large.
  ///
  /// @param resolution Minimum distance (meters) between collision checks.
  ///        Use 0.0 to disable (OMPL default only).
  void setCollisionResolution(double resolution) { collision_resolution_ = resolution; }

  /// @brief Get the collision checking resolution.
  double getCollisionResolution() const { return collision_resolution_; }

  // -- StateSpace interface --

  /// @brief Return the ambient dimension of the state space.
  unsigned int getDimension() const override { return ambient_dim_; }

  /// @brief Return the maximum extent of the state space.
  ///
  /// @todo Validate this fix properly. Current approach (max of coordinate
  /// diagonal and Riemannian corner-to-corner distance) fixes the connect-loop
  /// hang with anisotropic metrics, but the corner-to-corner Riemannian
  /// distance may not be the true maximum extent (angle wrapping, non-diagonal
  /// pairs, configuration-dependent metrics). Needs formal analysis and tests
  /// across different manifold types.
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
  /// @param from Start state.
  /// @param to End state.
  /// @param t Interpolation parameter in [0, 1].
  /// @param state Output state.
  void interpolate(const ob::State* from, const ob::State* to, double t,
                   ob::State* state) const override {
    const auto* f = from->as<StateType>();
    const auto* tgt = to->as<StateType>();
    auto* s = state->as<StateType>();
    Point result = manifold_.geodesic(f->asEigen(), tgt->asEigen(), t);
    for (unsigned int i = 0; i < ambient_dim_; ++i) {
      s->values[i] = result[i];
    }
  }

  /// @brief Allocate the default state sampler.
  ob::StateSamplerPtr allocDefaultStateSampler() const override {
    return std::make_shared<GeodexStateSampler<ManifoldT>>(this);
  }

  /// @brief Allocate a new state.
  ob::State* allocState() const override {
    auto* s = new StateType();
    s->values = new double[ambient_dim_];
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

  /// @brief Number of collision checks for a motion between two states.
  ///
  /// @details Returns the maximum of OMPL's default segment count and a
  /// coordinate-distance-based count derived from `collision_resolution_`.
  unsigned int validSegmentCount(const ob::State* s1, const ob::State* s2) const override {
    unsigned int n = ob::StateSpace::validSegmentCount(s1, s2);
    if (collision_resolution_ > 0.0) {
      const auto* a = s1->as<StateType>();
      const auto* b = s2->as<StateType>();
      double dist2 = 0.0;
      for (unsigned int i = 0; i < ambient_dim_; ++i) {
        double d = a->values[i] - b->values[i];
        dist2 += d * d;
      }
      unsigned int n_coord =
          static_cast<unsigned int>(std::ceil(std::sqrt(dist2) / collision_resolution_));
      n = std::max(n, n_coord);
    }
    return std::max(n, 1u);
  }

 private:
  ManifoldT manifold_;
  ob::RealVectorBounds bounds_;
  unsigned int ambient_dim_;
  double collision_resolution_ = 0.0;
};

}  // namespace ompl_integration
}  // namespace geodex
