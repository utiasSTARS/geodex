/// @file robot_impl.hpp
/// @brief Collision-checker and motion-validator templates over a VAMP kernel (internal).
///
/// Each kernel's translation unit instantiates these against its @c vamp::robots::X
/// struct and its generated `SweepModel`, and registers the result. The header pulls in
/// VAMP SIMD intrinsics, and only translation units inside the @c geodex_vamp static
/// archive include it. The archive's SIMD compile options are PRIVATE and do not reach
/// consumer translation units.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <vamp/vector.hh>

// Eigen derives the alignment of fixed-size storage from its size alone. Under the 16-byte cap
// of the archive, it requests 16 bytes for arrays of VAMP's SIMD vectors, which need 32 with
// AVX. GCC ignores the weaker request and Clang rejects it. These specializations leave such
// arrays at the alignment of their element type.
namespace Eigen::internal {
template <typename SimdT, std::size_t Rows, std::size_t ScalarsPerRow, int Size>
struct compute_default_alignment<::vamp::Vector<SimdT, Rows, ScalarsPerRow>, Size> {
  enum { value = 0 };
};
template <typename SimdT, std::size_t Rows, std::size_t ScalarsPerRow>
struct compute_default_alignment<::vamp::Vector<SimdT, Rows, ScalarsPerRow>, Dynamic> {
  enum { value = EIGEN_MAX_ALIGN_BYTES };
};
}  // namespace Eigen::internal

#include <ompl/base/MotionValidator.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/State.h>

#include "geodex/integration/vamp/registry.hpp"
#include "geodex/manifold/se2.hpp"
#include "check_motion.hpp"
#include "sweep_model.hpp"
#include "vamp_env.hpp"

namespace geodex::integration::vamp::detail {

/// @brief True when every coordinate of @p q lies inside the kernel's joint limits.
///
/// The limits are precomputed with the sweep data from the robot's URDFs. A state on a limit
/// can land a few ulps outside after a float round trip, and the tolerance absorbs that.
template <std::size_t N>
bool in_joint_box(const double* q, const SweepModel<N>& sweep) {
  constexpr double tol = 1e-6;
  for (std::size_t i = 0; i < N; ++i) {
    if (!(q[i] >= sweep.lower[i] - tol && q[i] <= sweep.upper[i] + tol)) return false;
  }
  return true;
}

/// @brief Check one rake of `W` configurations, honoring an attached body when present.
template <typename VampRobot, std::size_t W>
bool fkcc_rake(const VampEnvT& env,
               const typename VampRobot::template ConfigurationBlock<W>& block) {
  if (env.attachments) return VampRobot::template fkcc_attach<W>(env, block);
  return VampRobot::template fkcc<W>(env, block);
}

/// @brief Check @p count row-major configurations of the robot's dimension in rakes.
///
/// A short final rake repeats its last configuration. The check is a conjunction over
/// lanes, and a duplicate does not change the answer. Callers check the joint box.
template <typename VampRobot>
bool rows_collision_free(const VampEnvT& env, const double* values, int count) {
  constexpr std::size_t W = ::vamp::FloatVectorWidth;
  constexpr std::size_t D = VampRobot::dimension;
  alignas(32) std::array<float, W> lane{};
  for (int base = 0; base < count; base += static_cast<int>(W)) {
    const int n = std::min(static_cast<int>(W), count - base);
    typename VampRobot::template ConfigurationBlock<W> block;
    for (std::size_t j = 0; j < D; ++j) {
      for (std::size_t l = 0; l < W; ++l) {
        const int row = base + std::min(static_cast<int>(l), n - 1);
        lane[l] = static_cast<float>(values[static_cast<std::ptrdiff_t>(row) * D + j]);
      }
      block[j] = ::vamp::FloatVector<W>(lane.data(), false);
    }
    if (!fkcc_rake<VampRobot, W>(env, block)) return false;
  }
  return true;
}

/// @brief An environment a checker or validator uses on its own.
///
/// VAMP poses an attached body inside the environment on every check, and checks that
/// share one environment with an attachment would overwrite each other's poses. An
/// environment with an attachment is copied, attachment included, and its checks are
/// serialized. An environment without an attachment is shared as is.
class OwnedEnv {
 public:
  explicit OwnedEnv(EnvHandle env) : env_(std::move(env)) {
    if (env_cast(env_).attachments) {
      env_ = EnvHandle{std::make_shared<VampEnvT>(env_cast(env_))};
      attached_ = true;
    }
  }

  /// @brief Run @p check on the environment, serialized when it holds an attachment.
  template <typename F>
  auto with(F&& check) const {
    if (!attached_) return check(env_cast(env_));
    const std::lock_guard<std::mutex> lock(mutex_);
    return check(env_cast(env_));
  }

  /// @brief The environment's handle.
  const EnvHandle& handle() const { return env_; }

  /// @brief Whether the environment holds an attachment.
  bool attached() const { return attached_; }

 private:
  EnvHandle env_;
  bool attached_ = false;
  mutable std::mutex mutex_;
};

template <typename VampRobot>
class VampCollisionCheckerImpl : public CollisionChecker {
 public:
  using Sweep = SweepModel<VampRobot::dimension>;

  VampCollisionCheckerImpl(EnvHandle env, const Sweep& sweep)
      : env_(std::move(env)), sweep_(sweep) {}

  auto is_valid(const double* values, int dim) const -> bool override {
    if (dim != static_cast<int>(VampRobot::dimension)) return false;
    if (!in_joint_box(values, sweep_)) return false;
    return env_.with(
        [&](const VampEnvT& env) { return rows_collision_free<VampRobot>(env, values, 1); });
  }

  auto batch_width() const -> int override { return static_cast<int>(::vamp::FloatVectorWidth); }

  auto all_valid(const double* values, int dim, int count) const -> bool override {
    if (dim != static_cast<int>(VampRobot::dimension)) {
      throw std::invalid_argument("geodex::integration::vamp: all_valid expects dimension " +
                                  std::to_string(VampRobot::dimension) + ", got " +
                                  std::to_string(dim));
    }
    for (int i = 0; i < count; ++i) {
      if (!in_joint_box(values + static_cast<std::ptrdiff_t>(i) * dim, sweep_)) return false;
    }
    if (count <= 0) return true;
    return env_.with(
        [&](const VampEnvT& env) { return rows_collision_free<VampRobot>(env, values, count); });
  }

 private:
  OwnedEnv env_;
  const Sweep& sweep_;
};

/// @brief OMPL motion validator on VAMP's straight-chord edge check.
///
/// Matches edges that are straight lines in the joint coordinates, the case for a
/// fixed-base arm on a Euclidean chart. Checks every 1/resolution of joint distance.
template <typename VampRobot>
class VampMotionValidatorImpl : public ompl::base::MotionValidator {
 public:
  VampMotionValidatorImpl(const ompl::base::SpaceInformationPtr& si, EnvHandle env)
      : ompl::base::MotionValidator(si), env_(std::move(env)) {}

  auto checkMotion(const ompl::base::State* s1,
                   const ompl::base::State* s2) const -> bool override {
    return env_.with([&](const VampEnvT&) {
      return check_motion_impl<VampRobot>(s1, s2, env_.handle());
    });
  }

  auto checkMotion(const ompl::base::State* s1, const ompl::base::State* s2,
                   std::pair<ompl::base::State*, double>& /*lastValid*/) const
      -> bool override {
    return checkMotion(s1, s2);
  }

 private:
  OwnedEnv env_;
};

/// @brief OMPL motion validator for a planar-base kernel that samples the state space's
/// own interpolation.
///
/// Samples each edge at evenly spaced parameters of `StateSpace::interpolate`, at least
/// `validSegmentCount` and at least the kernel's resolution, and checks them in SIMD
/// rakes. The base edge is a group geodesic on SE(2) whose heading wraps. A straight
/// chord in the coordinates would check a different motion.
template <typename VampRobot>
class VampInterpolatedMotionValidatorImpl : public ompl::base::MotionValidator {
 public:
  using Sweep = SweepModel<VampRobot::dimension>;

  VampInterpolatedMotionValidatorImpl(const ompl::base::SpaceInformationPtr& si, EnvHandle env,
                                      const Sweep& sweep)
      : ompl::base::MotionValidator(si), env_(std::move(env)), sweep_(sweep) {}

  auto checkMotion(const ompl::base::State* s1,
                   const ompl::base::State* s2) const -> bool override {
    constexpr std::size_t W = ::vamp::FloatVectorWidth;
    constexpr std::size_t D = VampRobot::dimension;
    const auto& space = si_->getStateSpace();
    if (space->getDimension() != D) return false;
    // At least the kernel's own edge resolution, in steps per unit of configuration
    // distance. Coordinates 0 to 2 are the planar base (x, y, theta), and the heading
    // difference takes the short way round.
    double d2 = 0.0;
    for (std::size_t j = 0; j < D; ++j) {
      double d = *space->getValueAddressAtIndex(s2, static_cast<unsigned int>(j)) -
                 *space->getValueAddressAtIndex(s1, static_cast<unsigned int>(j));
      if (j == 2) d = std::remainder(d, 2.0 * std::numbers::pi);
      d2 += d * d;
    }
    const auto by_resolution =
        static_cast<unsigned int>(std::ceil(std::sqrt(d2) * VampRobot::resolution));
    const unsigned int n = std::max({space->validSegmentCount(s1, s2), by_resolution, 1u});

    ompl::base::State* tmp = si_->allocState();
    std::array<double, W * D> rows{};
    bool ok = true;
    // Parameters k / n for k = 1..n. The planner contract makes the start state valid.
    for (unsigned int k0 = 1; ok && k0 <= n; k0 += static_cast<unsigned int>(W)) {
      const int m = static_cast<int>(std::min<unsigned int>(W, n - k0 + 1));
      for (int l = 0; l < m; ++l) {
        const double t = static_cast<double>(k0 + static_cast<unsigned int>(l)) / n;
        space->interpolate(s1, s2, t, tmp);
        for (std::size_t j = 0; j < D; ++j) {
          rows[static_cast<std::size_t>(l) * D + j] =
              *space->getValueAddressAtIndex(tmp, static_cast<unsigned int>(j));
        }
      }
      for (int l = 0; ok && l < m; ++l) ok = in_joint_box(&rows[static_cast<std::size_t>(l) * D], sweep_);
      ok = ok && env_.with([&](const VampEnvT& env) {
             return rows_collision_free<VampRobot>(env, rows.data(), m);
           });
    }
    si_->freeState(tmp);
    if (ok) {
      ++valid_;
    } else {
      ++invalid_;
    }
    return ok;
  }

  auto checkMotion(const ompl::base::State* s1, const ompl::base::State* s2,
                   std::pair<ompl::base::State*, double>& /*lastValid*/) const
      -> bool override {
    return checkMotion(s1, s2);
  }

 private:
  OwnedEnv env_;
  const Sweep& sweep_;
};

/// @brief A copy of @p env with every obstacle grown by @p pad on every side.
///
/// Spheres and capsules gain @p pad in radius, and boxes gain @p pad on each half
/// extent, which contains the box grown by a ball of radius @p pad. Each list is sorted
/// again by its early-exit distance, as VAMP requires. Point clouds, height fields and
/// finite cylinders cannot be grown this way, and the loaders do not create them.
inline auto padded_environment(const VampEnvT& env, float pad) -> VampEnvT {
  if (!env.pointclouds.empty() || !env.heightfields.empty() || !env.cylinders.empty()) {
    throw std::invalid_argument(
        "geodex::integration::vamp: a scene with point clouds, height fields or finite "
        "cylinders cannot be padded");
  }
  using FV = ::vamp::FloatVector<::vamp::FloatVectorWidth>;
  const FV p(pad);
  VampEnvT out = env;
  for (auto& s : out.spheres) {
    s.r = s.r + p;
    s.min_distance = s.min_distance - p;
  }
  for (auto* list : {&out.capsules, &out.z_aligned_capsules}) {
    for (auto& c : *list) {
      c.r = c.r + p;
      c.min_distance = c.min_distance - p;
    }
  }
  for (auto* list : {&out.cuboids, &out.z_aligned_cuboids}) {
    for (auto& c : *list) {
      c.axis_1_r = c.axis_1_r + p;
      c.axis_2_r = c.axis_2_r + p;
      c.axis_3_r = c.axis_3_r + p;
      c.min_distance = c.compute_min_distance();
    }
  }
  const auto by_distance = [](const auto& a, const auto& b) {
    return a.min_distance.to_array()[0] < b.min_distance.to_array()[0];
  };
  std::sort(out.spheres.begin(), out.spheres.end(), by_distance);
  std::sort(out.capsules.begin(), out.capsules.end(), by_distance);
  std::sort(out.z_aligned_capsules.begin(), out.z_aligned_capsules.end(), by_distance);
  std::sort(out.cuboids.begin(), out.cuboids.end(), by_distance);
  std::sort(out.z_aligned_cuboids.begin(), out.z_aligned_cuboids.end(), by_distance);
  return out;
}

/// @brief Largest distance of a sphere attached in @p env from the end-effector origin, or
/// 0 without an attached body.
inline double attached_reach(const VampEnvT& env) {
  double a = 0.0;
  if (env.attachments) {
    for (const auto& s : env.attachments->spheres) {
      const auto x = s.x.to_array()[0], y = s.y.to_array()[0], z = s.z.to_array()[0];
      a = std::max(a, std::sqrt(double(x * x + y * y + z * z)));
    }
  }
  return a;
}

/// @brief Sphere travel bound of a kernel with the spheres attached in @p env, the bound of
/// `VampCertifiedMotionValidatorImpl`.
template <std::size_t N>
TravelBound travel_bound(const SweepModel<N>& sweep, const VampEnvT& env) {
  const bool held = static_cast<bool>(env.attachments);
  const double a = attached_reach(env);
  TravelBound out;
  out.planar_base = sweep.planar_base;
  std::size_t first = 0;
  if (sweep.planar_base) {
    out.base = held ? std::max(sweep.base_reach, sweep.ee_base_reach + a) : sweep.base_reach;
    first = 3;
  }
  for (std::size_t j = first; j < N; ++j) {
    out.joint.push_back(held
                            ? std::max(sweep.reach[j], sweep.ee_reach[j] + a * sweep.ee_rotation[j])
                            : sweep.reach[j]);
  }
  return out;
}

/// @brief OMPL motion validator that certifies a whole edge against the obstacles.
///
/// The edge is the curve geodex's robot spaces interpolate. The arm joints move linearly,
/// and a planar base follows the SE(2) group exponential with its constant body twist
/// `(vx, vy, omega)`. Along it, no sphere center moves faster than
/// `|(vx, vy)| + |omega| base_reach + sum_j reach_j |dq_j|` per unit of the edge
/// parameter (`SweepModel`, with an attached body folded in through the end-effector
/// bounds). The edge is cut into pieces along which every center moves at most `step`.
/// Checking each piece's ends against the obstacles grown by `step / 2` covers every
/// sphere position in between. A piece whose end fails is halved, with half the
/// padding, down to `step / 2^kLevels`. A failing end that also collides unpadded
/// rejects the edge at once. Self-collision is checked at the piece ends without
/// padding. With a planar base, the base position must stay inside the space's bounds
/// along the whole arc, not only at the checked states.
template <typename VampRobot>
class VampCertifiedMotionValidatorImpl : public ompl::base::MotionValidator {
 public:
  static constexpr std::size_t D = VampRobot::dimension;
  using Sweep = SweepModel<D>;
  using Config = std::array<double, D>;

  /// Halvings of a piece before an edge that still grazes an obstacle is rejected.
  static constexpr int kLevels = 8;

  VampCertifiedMotionValidatorImpl(const ompl::base::SpaceInformationPtr& si, EnvHandle env,
                                   const Sweep& sweep, double step)
      : ompl::base::MotionValidator(si), env_(std::move(env)), sweep_(sweep), step_(step) {
    if (!(step > 0.0) || !std::isfinite(step)) {
      throw std::invalid_argument(
          "geodex::integration::vamp: the certified motion check needs a positive step");
    }
    const VampEnvT& raw = env_cast(env_.handle());
    for (int level = 0; level <= kLevels; ++level) {
      padded_.push_back(padded_environment(raw, static_cast<float>(pad(level))));
    }
    bound_ = travel_bound(sweep_, raw);
    has_attachment_ = static_cast<bool>(raw.attachments);
  }

  auto checkMotion(const ompl::base::State* s1,
                   const ompl::base::State* s2) const -> bool override {
    double last = 0.0;
    return check(s1, s2, &last);
  }

  auto checkMotion(const ompl::base::State* s1, const ompl::base::State* s2,
                   std::pair<ompl::base::State*, double>& lastValid) const -> bool override {
    double last = 0.0;
    const bool ok = check(s1, s2, &last);
    if (!ok) {
      if (lastValid.first != nullptr) {
        const auto& space = si_->getStateSpace();
        const Config q = config_at(read(s1), edge(read(s1), read(s2)), last);
        for (std::size_t j = 0; j < D; ++j) {
          *space->getValueAddressAtIndex(lastValid.first, static_cast<unsigned int>(j)) = q[j];
        }
      }
      lastValid.second = last;
    }
    return ok;
  }

 private:
  /// Base twist (planar base) and joint change of an edge.
  struct Edge {
    Eigen::Vector3d twist = Eigen::Vector3d::Zero();
    Config delta{};
  };

  double pad(int level) const { return step_ / static_cast<double>(2 << level); }

  Config read(const ompl::base::State* s) const {
    const auto& space = si_->getStateSpace();
    Config q{};
    for (std::size_t j = 0; j < D; ++j) {
      q[j] = *space->getValueAddressAtIndex(s, static_cast<unsigned int>(j));
    }
    return q;
  }

  Edge edge(const Config& a, const Config& b) const {
    Edge e;
    for (std::size_t j = 0; j < D; ++j) e.delta[j] = b[j] - a[j];
    if (sweep_.planar_base) {
      e.twist = SE2LeftExponentialMap{}.inverse_retract(Eigen::Vector3d(a[0], a[1], a[2]),
                                                        Eigen::Vector3d(b[0], b[1], b[2]));
    }
    return e;
  }

  Config config_at(const Config& a, const Edge& e, double t) const {
    Config q{};
    for (std::size_t j = 0; j < D; ++j) q[j] = a[j] + t * e.delta[j];
    if (sweep_.planar_base) {
      const Eigen::Vector3d p =
          SE2LeftExponentialMap{}.retract(Eigen::Vector3d(a[0], a[1], a[2]), t * e.twist);
      q[0] = p[0];
      q[1] = p[1];
      q[2] = p[2];
    }
    return q;
  }

  /// Bound on the distance any sphere center travels over the whole edge.
  double travel(const Edge& e) const {
    Config v = e.delta;
    if (sweep_.planar_base) {
      for (int k = 0; k < 3; ++k) v[static_cast<std::size_t>(k)] = e.twist[k];
    }
    return bound_(v.data()) * (1.0 + 1e-9);
  }

  /// With a planar base, whether the base stays inside the space's bounds along the arc.
  /// The base origin moves with constant body velocity. Its x and y are extreme only at
  /// the ends and where the heading of that velocity is a multiple of pi / 2.
  bool arc_in_bounds(const Config& a, const Edge& e) const {
    if (!sweep_.planar_base) return true;
    const double speed = std::hypot(e.twist[0], e.twist[1]);
    const double omega = e.twist[2];
    if (speed == 0.0 || std::abs(omega) < 1e-12) return true;
    const double phi = a[2] + std::atan2(e.twist[1], e.twist[0]);
    const double half_pi = 0.5 * std::numbers::pi;
    const double k_lo = std::ceil(std::min(phi, phi + omega) / half_pi);
    const double k_hi = std::floor(std::max(phi, phi + omega) / half_pi);
    ompl::base::State* s = si_->allocState();
    const auto& space = si_->getStateSpace();
    bool ok = true;
    for (double k = k_lo; ok && k <= k_hi; k += 1.0) {
      const double t = std::clamp((k * half_pi - phi) / omega, 0.0, 1.0);
      const Config q = config_at(a, e, t);
      for (std::size_t j = 0; j < D; ++j) {
        *space->getValueAddressAtIndex(s, static_cast<unsigned int>(j)) = q[j];
      }
      ok = si_->satisfiesBounds(s);
    }
    si_->freeState(s);
    return ok;
  }

  template <typename F>
  bool guarded(F&& f) const {
    if (!has_attachment_) return f();
    const std::lock_guard<std::mutex> lock(padded_mutex_);
    return f();
  }

  /// Which of @p ts are free against the obstacles grown for @p level, checked in SIMD
  /// rakes. A rake that fails is resolved point by point. Each rake takes points spread
  /// over the whole list, as VAMP's own edge check does, and a collision shows early. A
  /// point that fails the grown obstacles is checked against the scene itself at once.
  /// A hit there stops the check, and @p hit holds its parameter.
  std::vector<char> free_points(const Config& a, const Edge& e, const std::vector<double>& ts,
                                int level, std::optional<double>* hit) const {
    constexpr std::size_t W = ::vamp::FloatVectorWidth;
    const VampEnvT& env = padded_[static_cast<std::size_t>(level)];
    std::vector<char> free(ts.size(), 0);
    std::array<double, W * D> rows{};
    std::array<std::size_t, W> which{};
    const std::size_t stride = (ts.size() + W - 1) / W;
    for (std::size_t first = 0; first < stride; ++first) {
      std::size_t m = 0;
      for (std::size_t i = first; i < ts.size(); i += stride) {
        const Config q = config_at(a, e, ts[i]);
        std::copy(q.begin(), q.end(), rows.begin() + static_cast<std::ptrdiff_t>(m * D));
        which[m++] = i;
      }
      if (rows_collision_free<VampRobot>(env, rows.data(), static_cast<int>(m))) {
        for (std::size_t l = 0; l < m; ++l) free[which[l]] = 1;
        continue;
      }
      for (std::size_t l = 0; l < m; ++l) {
        free[which[l]] = rows_collision_free<VampRobot>(env, &rows[l * D], 1);
        if (free[which[l]]) continue;
        const bool clear = env_.with([&](const VampEnvT& raw) {
          return rows_collision_free<VampRobot>(raw, &rows[l * D], 1);
        });
        if (!clear) {
          *hit = ts[which[l]];
          return free;
        }
      }
    }
    return free;
  }

  /// Check the edge level by level. At level L every piece moves each sphere center at
  /// most step / 2^L, and its ends are checked against the obstacles grown by half that.
  /// The pieces with a failing end are halved for the next level. An end that fails the
  /// unpadded scene too rejects the edge at once. A rejected edge reports its start as
  /// the last valid state in @p last. A longer prefix is not certified.
  bool check(const ompl::base::State* s1, const ompl::base::State* s2, double* last) const {
    const Config a = read(s1), b = read(s2);
    *last = 0.0;
    bool ok = in_joint_box(b.data(), sweep_);
    const Edge e = edge(a, b);
    ok = ok && arc_in_bounds(a, e);
    if (ok) {
      const auto n = static_cast<std::size_t>(std::max(1.0, std::ceil(travel(e) / step_)));
      std::vector<std::pair<double, double>> pieces;
      pieces.reserve(n);
      for (std::size_t k = 0; k < n; ++k) {
        pieces.emplace_back(static_cast<double>(k) / static_cast<double>(n),
                            static_cast<double>(k + 1) / static_cast<double>(n));
      }
      ok = guarded([&] {
        for (int level = 0; level <= kLevels; ++level) {
          // Piece ends at this level, each once.
          std::vector<double> ts;
          ts.reserve(2 * pieces.size());
          for (const auto& [t0, t1] : pieces) {
            if (ts.empty() || ts.back() != t0) ts.push_back(t0);
            ts.push_back(t1);
          }
          std::optional<double> hit;
          const auto free = free_points(a, e, ts, level, &hit);
          if (hit) return false;
          if (std::all_of(free.begin(), free.end(), [](char f) { return f != 0; })) return true;
          // Halve every piece with a failing end.
          std::vector<std::pair<double, double>> next;
          std::size_t i = 0;
          for (const auto& [t0, t1] : pieces) {
            while (i < ts.size() && ts[i] < t0) ++i;
            // ts[i] is t0 and ts[i + 1] is t1. Each piece pushed both of its ends.
            if (free[i] && free[i + 1]) continue;
            const double tm = 0.5 * (t0 + t1);
            next.emplace_back(t0, tm);
            next.emplace_back(tm, t1);
          }
          pieces.swap(next);
        }
        return false;
      });
    }
    if (ok) {
      ++valid_;
    } else {
      ++invalid_;
    }
    return ok;
  }

  OwnedEnv env_;
  const Sweep& sweep_;
  double step_;
  std::vector<VampEnvT> padded_;
  mutable std::mutex padded_mutex_;
  TravelBound bound_;
  bool has_attachment_ = false;
};

/// @brief Sphere centers and radii of the kernel at @p q, in the world frame.
template <typename VampRobot>
auto sphere_positions(const double* q) -> std::vector<std::array<double, 4>> {
  constexpr std::size_t W = ::vamp::FloatVectorWidth;
  typename VampRobot::template ConfigurationBlock<W> block;
  for (std::size_t i = 0; i < VampRobot::dimension; ++i) {
    block[i] = ::vamp::FloatVector<W>(static_cast<float>(q[i]));
  }
  typename VampRobot::template Spheres<W> out;
  VampRobot::template sphere_fk<W>(block, out);
  std::vector<std::array<double, 4>> spheres(VampRobot::n_spheres);
  for (std::size_t s = 0; s < VampRobot::n_spheres; ++s) {
    spheres[s] = {out.x[{s, 0}], out.y[{s, 0}], out.z[{s, 0}], out.r[{s, 0}]};
  }
  return spheres;
}

}  // namespace geodex::integration::vamp::detail
