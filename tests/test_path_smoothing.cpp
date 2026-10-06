// Tests for geodex::algorithm::smooth_path.

#include <cmath>
#include <cstdint>

#include <algorithm>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "geodex/algorithm/path_smoothing.hpp"
#include "geodex/manifold/configuration_space.hpp"
#include "geodex/manifold/euclidean.hpp"
#include "geodex/manifold/se2.hpp"
#include "geodex/manifold/se3.hpp"
#include "geodex/manifold/so3.hpp"
#include "geodex/manifold/sphere.hpp"
#include "geodex/metrics/clearance.hpp"
#include "geodex/metrics/constant_spd.hpp"
#include "geodex/metrics/kinetic_energy.hpp"
#include "geodex/metrics/se2_left_invariant.hpp"
#include "geodex/utils/random.hpp"

namespace ga = geodex::algorithm;

namespace {

using Vec2 = Eigen::Vector2d;

// A disc obstacle of radius r at c, as a validity predicate and a distance.
struct Disc {
  Vec2 c;
  double r;
  template <typename P>
  double distance(const P& q) const {
    return std::hypot(q[0] - c[0], q[1] - c[1]) - r;
  }
  template <typename P>
  bool operator()(const P& q) const {
    return distance(q) > 0.0;
  }
};

// The disc shrunk by `tolerance`. A returned path lies within the corner tolerance of a checked
// path and keeps clear of the disc shrunk by that tolerance.
Disc shrunk(const Disc& disc, const double tolerance) { return {disc.c, disc.r - tolerance}; }

// A detour around a disc at the origin, from (-2, 0) to (2, 0).
std::vector<Vec2> detour() {
  return {{-2.0, 0.0}, {-1.5, 1.5}, {-0.5, 1.6}, {0.0, 2.0}, {0.6, 1.4}, {1.5, 1.5}, {2.0, 0.0}};
}

// Coordinate length of a polyline.
template <typename P>
double coord_length(const std::vector<P>& path) {
  double len = 0.0;
  for (std::size_t k = 1; k < path.size(); ++k) len += (path[k] - path[k - 1]).norm();
  return len;
}

// Checks the certification contract independently. Every waypoint is valid, and every
// edge is valid on a grid with spacing at most `res`.
template <typename M, typename V>
bool certified(const M& m, const std::vector<typename M::Point>& path, const V& valid,
               const double res) {
  for (std::size_t k = 0; k < path.size(); ++k) {
    if (!valid(path[k])) return false;
    if (k + 1 == path.size()) break;
    const double ext = m.log(path[k], path[k + 1]).norm();
    const int n = std::max(1, static_cast<int>(std::ceil(ext / res)));
    for (int j = 1; j < n; ++j) {
      if (!valid(m.geodesic(path[k], path[k + 1], static_cast<double>(j) / n))) return false;
    }
  }
  return true;
}

// Two-link planar arm mass matrix, anisotropic in the elbow angle.
struct PlanarArmMass {
  Eigen::Matrix2d operator()(const Vec2& q) const {
    const double h = 0.5 * std::cos(q[1]);
    const double m00 = 1.0 / 12 + 1.0 / 12 + 0.25 + (1.0 + 0.25 + 2 * h);
    const double m01 = 1.0 / 12 + (0.25 + h);
    Eigen::Matrix2d m;
    m << m00, m01, m01, 1.0 / 12 + 0.25;
    return m;
  }
};

}  // namespace

TEST(PathSmoothing, ShortensAroundAnObstacleAndCertifies) {
  const geodex::Euclidean<2> m;
  const Disc disc{{0.0, 0.0}, 1.0};
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.01;
  const auto r = ga::smooth_path(m, disc, detour(), s);
  ASSERT_TRUE(r.collision_free);
  EXPECT_EQ(r.first_invalid_index, decltype(r)::npos);
  EXPECT_TRUE(certified(m, r.path, shrunk(disc, s.corner_tolerance), 0.01));
  EXPECT_EQ(r.path.front(), detour().front());
  EXPECT_EQ(r.path.back(), detour().back());
  // The shortest path hugs the disc, two tangents of length sqrt(3) and an arc of pi / 3.
  const double optimum = 2.0 * std::sqrt(3.0) + std::acos(-1.0) / 3.0;
  EXPECT_LT(r.length, coord_length(detour()));
  EXPECT_LT(r.length, 1.03 * optimum);
  EXPECT_GE(r.length, optimum - 1e-6);
  EXPECT_NEAR(r.length, coord_length(r.path), 1e-9);
}

TEST(PathSmoothing, FreeSpacePathBecomesTheChord) {
  const geodex::Euclidean<2> m;
  const auto r = ga::smooth_path(
      m, [](const Vec2&) { return true; }, detour());
  ASSERT_TRUE(r.collision_free);
  EXPECT_NEAR(r.length, 4.0, 1e-9);
}

TEST(PathSmoothing, UncertifiableInputIsReturnedWithTheTruth) {
  // The middle edge's chord crosses the disc, and every shortcut crosses it too.
  const geodex::Euclidean<2> m;
  const Disc disc{{0.0, 0.0}, 1.0};
  const std::vector<Vec2> path = {{-2.0, 0.0}, {-1.2, 0.1}, {1.2, 0.1}, {2.0, 0.0}};
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.01;
  const auto r = ga::smooth_path(m, disc, path, s);
  EXPECT_FALSE(r.collision_free);
  EXPECT_EQ(r.path, path);
  EXPECT_EQ(r.first_invalid_index, 1u);
  EXPECT_EQ(r.profile.fallback, 3);
}

TEST(PathSmoothing, NeverReportsAnInvalidPathAsCertified) {
  // The input crosses a thin wall between its samples at the coarse resolution. The
  // certification at a fine resolution must reject it.
  const geodex::Euclidean<2> m;
  auto valid = [](const Vec2& q) { return std::abs(q[0] - 0.013) > 0.002 || q[1] > 0.5; };
  const std::vector<Vec2> path = {{-1.0, 0.0}, {-0.5, 0.0}, {0.5, 0.0}, {1.0, 0.0}};
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.001;
  const auto r = ga::smooth_path(m, valid, path, s);
  EXPECT_TRUE(r.collision_free == certified(m, r.path, valid, 0.001));
  if (r.collision_free) {
    for (const auto& q : r.path) EXPECT_TRUE(valid(q));
  }
}

TEST(PathSmoothing, SameSeedGivesTheSamePath) {
  const geodex::Euclidean<2> m;
  const Disc disc{{0.0, 0.0}, 1.0};
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.01;
  const auto a = ga::smooth_path(m, disc, detour(), s);
  const auto b = ga::smooth_path(m, disc, detour(), s);
  ASSERT_EQ(a.path.size(), b.path.size());
  for (std::size_t k = 0; k < a.path.size(); ++k) EXPECT_EQ(a.path[k], b.path[k]);
}

TEST(PathSmoothing, ScalingTheProblemScalesThePath) {
  // Every default except `corner_tolerance` derives from the path's own scale. With that
  // tolerance scaled too, scaling a problem by a power of two is exact and gives the
  // scaled path exactly.
  const geodex::Euclidean<2> m;
  constexpr double kScale = 128.0;
  const Disc disc{{0.0, 0.0}, 1.0};
  auto scaled_valid = [&](const Vec2& q) { return disc(q / kScale); };
  std::vector<Vec2> big = detour();
  for (auto& q : big) q *= kScale;
  ga::PathSmoothingSettings big_s;
  big_s.corner_tolerance *= kScale;
  const auto small_r = ga::smooth_path(m, disc, detour());
  const auto big_r = ga::smooth_path(m, scaled_valid, big, big_s);
  EXPECT_GT(small_r.profile.rounded_corners, 0);
  ASSERT_TRUE(small_r.collision_free);
  ASSERT_TRUE(big_r.collision_free);
  ASSERT_EQ(small_r.path.size(), big_r.path.size());
  for (std::size_t k = 0; k < small_r.path.size(); ++k) {
    EXPECT_NEAR((big_r.path[k] / kScale - small_r.path[k]).norm(), 0.0, 1e-12);
  }
  EXPECT_NEAR(big_r.length / kScale, small_r.length, 1e-12);
}

TEST(PathSmoothing, ClearanceMetricMovesThePathAwayFromTheObstacle) {
  const Disc disc{{0.0, 0.0}, 1.0};
  auto sdf = [&](const Vec2& q) { return disc.distance(q); };
  using Identity = geodex::Euclidean<2>;
  using Base = geodex::ConstantSPDMetric<2>;
  const Base flat{Eigen::Matrix2d::Identity()};
  const geodex::ConfigurationSpace<Identity, geodex::SDFConformalMetric<Base, decltype(sdf)>>
      clearance{Identity{}, geodex::SDFConformalMetric<Base, decltype(sdf)>{flat, sdf, 4.0, 3.0}};
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.01;
  const auto plain = ga::smooth_path(Identity{}, disc, detour(), s);
  const auto aware = ga::smooth_path(clearance, disc, detour(), s);
  ASSERT_TRUE(plain.collision_free);
  ASSERT_TRUE(aware.collision_free);
  auto min_clearance = [&](const std::vector<Vec2>& p) {
    double best = 1e9;
    for (std::size_t k = 0; k + 1 < p.size(); ++k) {
      for (int j = 0; j <= 16; ++j)
        best = std::min(best, sdf(p[k] + (j / 16.0) * (p[k + 1] - p[k])));
    }
    return best;
  };
  EXPECT_LT(min_clearance(plain.path), 0.02);
  EXPECT_GT(min_clearance(aware.path), 0.15);
}

TEST(PathSmoothing, KineticEnergyMetricBendsTheArmPath) {
  // In free space, the Euclidean optimum is the chord. Under the arm's kinetic
  // energy the optimum bends, and the smoother finds a path cheaper than the chord.
  using KE = geodex::KineticEnergyMetric<PlanarArmMass>;
  const geodex::ConfigurationSpace<geodex::Euclidean<2>, KE> arm{geodex::Euclidean<2>{},
                                                                 KE{PlanarArmMass{}}};
  const std::vector<Vec2> path = {{-1.5, -2.0}, {0.0, 0.0}, {1.5, 2.0}};
  auto free = [](const Vec2&) { return true; };
  const auto flat = ga::smooth_path(geodex::Euclidean<2>{}, free, path);
  const auto curved = ga::smooth_path(arm, free, path);
  ASSERT_TRUE(curved.collision_free);
  double off_chord = 0.0;
  const Vec2 dir = (path.back() - path.front()).normalized();
  for (const auto& q : curved.path) {
    const Vec2 d = q - path.front();
    off_chord = std::max(off_chord, (d - d.dot(dir) * dir).norm());
  }
  EXPECT_LT(coord_length(flat.path), 5.0 + 1e-9);
  EXPECT_GT(off_chord, 0.05);
  const std::vector<Vec2> chord = {path.front(), path.back()};
  double chord_ke = 0.0;
  for (int j = 0; j < 256; ++j) {
    const Vec2 a = path.front() + (j / 256.0) * (path.back() - path.front());
    const Vec2 b = path.front() + ((j + 1) / 256.0) * (path.back() - path.front());
    chord_ke += arm.distance(a, b);
  }
  EXPECT_LT(curved.length, chord_ke - 1e-3);
}

TEST(PathSmoothing, SE2PathStaysCertified) {
  using SE2 = geodex::SE2<>;
  const SE2 m{geodex::SE2LeftInvariantMetric{1.0, 20.0, 0.5}};
  const Disc disc{{2.0, 0.0}, 0.8};
  const std::vector<Eigen::Vector3d> path = {
      {0.0, 0.0, 0.0}, {1.0, 1.2, 0.5}, {2.0, 1.4, 0.0}, {3.0, 1.2, -0.5}, {4.0, 0.0, 0.0}};
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.02;
  const auto r = ga::smooth_path(m, disc, path, s);
  ASSERT_TRUE(r.collision_free);
  EXPECT_TRUE(certified(m, r.path, disc, 0.02));
  double raw = 0.0;
  for (std::size_t k = 1; k < path.size(); ++k) raw += m.distance(path[k - 1], path[k]);
  EXPECT_LT(r.length, raw);
}

TEST(PathSmoothing, EmbeddedManifoldShortcutsOnly) {
  // Sphere tangents are ambient. Only shortcutting runs, and the path stays on the sphere.
  const geodex::Sphere<> m;
  auto valid = [](const Eigen::Vector3d& q) { return q[2] < 0.9; };
  std::vector<Eigen::Vector3d> path;
  for (int k = 0; k <= 8; ++k) {
    const double t = k / 8.0 * std::acos(-1.0) * 0.9;
    path.emplace_back(std::cos(t), std::sin(t) * 0.8, std::sin(t) * 0.6);
  }
  for (auto& q : path) q.normalize();
  const auto r = ga::smooth_path(m, valid, path);
  ASSERT_TRUE(r.collision_free);
  for (const auto& q : r.path) EXPECT_NEAR(q.norm(), 1.0, 1e-9);
  EXPECT_LE(r.path.size(), path.size());
}

TEST(PathSmoothing, OutputSpacingIsEvenAndKeepsTheEndpoints) {
  const geodex::Euclidean<2> m;
  const Disc disc{{0.0, 0.0}, 1.0};
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.01;
  s.output_spacing = 0.1;
  s.round_corners = false;
  const auto r = ga::smooth_path(m, disc, detour(), s);
  ASSERT_TRUE(r.collision_free);
  EXPECT_EQ(r.profile.fallback, 0);
  EXPECT_EQ(r.path.front(), detour().front());
  EXPECT_EQ(r.path.back(), detour().back());
  // Every step is at most the spacing.
  for (std::size_t k = 1; k < r.path.size(); ++k) {
    EXPECT_LE((r.path[k] - r.path[k - 1]).norm(), 0.1 + 1e-9);
  }
  EXPECT_TRUE(certified(m, r.path, disc, 0.01));
}

TEST(PathSmoothing, PathPredicateIsRespected) {
  const geodex::Euclidean<2> m;
  const Disc disc{{0.0, 0.0}, 1.0};
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.01;
  // Every waypoint stays at least 0.3 from the disc.
  s.path_predicate = [&](const std::vector<Eigen::VectorXd>& p) {
    for (const auto& q : p) {
      if (disc.distance(q) < 0.3) return false;
    }
    return true;
  };
  const auto r = ga::smooth_path(m, disc, detour(), s);
  ASSERT_TRUE(r.collision_free);
  for (const auto& q : r.path) EXPECT_GE(disc.distance(q), 0.3 - 1e-12);
  EXPECT_LT(r.length, coord_length(detour()));
}

TEST(PathSmoothing, EdgeProofSkipsSamplesWithoutChangingTheResult) {
  // A proof that is sound but never fires, and one that fires in free space.
  const geodex::Euclidean<2> m;
  const Disc disc{{0.0, 0.0}, 1.0};
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.01;
  const auto plain = ga::smooth_path(m, disc, detour(), s);
  s.edge_provably_clear = [](const Eigen::Ref<const Eigen::VectorXd>&,
                             const Eigen::Ref<const Eigen::VectorXd>&) { return false; };
  const auto silent = ga::smooth_path(m, disc, detour(), s);
  ASSERT_EQ(plain.path.size(), silent.path.size());
  for (std::size_t k = 0; k < plain.path.size(); ++k) EXPECT_EQ(plain.path[k], silent.path[k]);
  // A sound proof from clearances. An edge is clear when the endpoint clearances
  // together exceed its length.
  s.edge_provably_clear = [&](const Eigen::Ref<const Eigen::VectorXd>& a,
                              const Eigen::Ref<const Eigen::VectorXd>& b) {
    return disc.distance(a) + disc.distance(b) > (a - b).norm();
  };
  const auto proved = ga::smooth_path(m, disc, detour(), s);
  ASSERT_TRUE(proved.collision_free);
  EXPECT_GT(proved.profile.edge_proofs, 0);
  EXPECT_LT(proved.profile.point_checks, plain.profile.point_checks);
  EXPECT_TRUE(certified(m, proved.path, shrunk(disc, s.corner_tolerance), 0.01));
}

TEST(PathSmoothing, EdgeTravelSpacesTheChecks) {
  const geodex::Euclidean<2> m;
  const Disc disc{{0.0, 0.0}, 1.0};
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.01;
  const auto plain = ga::smooth_path(m, disc, detour(), s);
  // The coordinate norm itself gives the default spacing and the same path.
  s.edge_travel = [](const Eigen::Ref<const Eigen::VectorXd>& a,
                     const Eigen::Ref<const Eigen::VectorXd>& b) {
    return geodex::utils::ordered_norm(b - a);
  };
  const auto same = ga::smooth_path(m, disc, detour(), s);
  ASSERT_EQ(plain.path.size(), same.path.size());
  for (std::size_t k = 0; k < plain.path.size(); ++k) EXPECT_EQ(plain.path[k], same.path[k]);
  EXPECT_EQ(plain.profile.point_checks, same.profile.point_checks);
  // Half the length places fewer samples on the edges.
  s.edge_travel = [](const Eigen::Ref<const Eigen::VectorXd>& a,
                     const Eigen::Ref<const Eigen::VectorXd>& b) {
    return 0.5 * geodex::utils::ordered_norm(b - a);
  };
  const auto sparse = ga::smooth_path(m, disc, detour(), s);
  ASSERT_TRUE(sparse.collision_free);
  EXPECT_LT(sparse.profile.point_checks, plain.profile.point_checks);
  // A derived resolution is in coordinates and ignores the bound.
  s.collision_check_resolution = 0.0;
  const auto derived = ga::smooth_path(m, disc, detour(), s);
  s.edge_travel = nullptr;
  const auto derived_plain = ga::smooth_path(m, disc, detour(), s);
  EXPECT_EQ(derived.profile.point_checks, derived_plain.profile.point_checks);
  // A bound that is negative or not a number is rejected.
  s.collision_check_resolution = 0.01;
  for (const double bad : {-1.0, std::numeric_limits<double>::quiet_NaN()}) {
    s.edge_travel = [bad](const Eigen::Ref<const Eigen::VectorXd>&,
                          const Eigen::Ref<const Eigen::VectorXd>&) { return bad; };
    EXPECT_THROW((void)ga::smooth_path(m, disc, detour(), s), std::invalid_argument) << bad;
  }
}

TEST(PathSmoothing, EdgeValidatorDecidesEdges) {
  const geodex::Euclidean<2> m;
  const Disc disc{{0.0, 0.0}, 1.0};
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.01;
  long calls = 0;
  s.edge_validator = [&](const Eigen::Ref<const Eigen::VectorXd>& a,
                         const Eigen::Ref<const Eigen::VectorXd>& b) {
    ++calls;
    for (int j = 1; j < 400; ++j) {
      if (!disc(Eigen::Vector2d(a + (j / 400.0) * (b - a)))) return false;
    }
    return true;
  };
  const auto r = ga::smooth_path(m, disc, detour(), s);
  ASSERT_TRUE(r.collision_free);
  EXPECT_GT(calls, 0);
  // With a validator that rejects every edge, the smoother returns the input and reports it
  // invalid.
  s.edge_validator = [](const Eigen::Ref<const Eigen::VectorXd>&,
                        const Eigen::Ref<const Eigen::VectorXd>&) { return false; };
  const auto rejected = ga::smooth_path(m, disc, detour(), s);
  EXPECT_FALSE(rejected.collision_free);
  EXPECT_EQ(rejected.first_invalid_index, 0u);
  EXPECT_EQ(rejected.path, detour());
}

namespace {

// Scripted batch oracle with the same answers as the scalar form, counting calls.
struct BatchDisc {
  Disc disc;
  std::size_t width;
  long* batches;
  bool operator()(const Vec2& q) const { return disc(q); }
  std::size_t batch_size() const { return width; }
  bool batch(const Vec2* q, std::size_t n) const {
    ++*batches;
    if (n > width) ADD_FAILURE() << "a block of " << n << " points exceeds the width " << width;
    for (std::size_t i = 0; i < n; ++i) {
      if (!disc(q[i])) return false;
    }
    return true;
  }
};

}  // namespace

TEST(PathSmoothing, BatchOracleGivesTheScalarResult) {
  const geodex::Euclidean<2> m;
  const Disc disc{{0.0, 0.0}, 1.0};
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.005;
  const auto scalar = ga::smooth_path(m, disc, detour(), s);
  // Width 1 and 2 fill the block with the edge's end point and one sample.
  for (const std::size_t width : {1u, 2u, 3u, 8u}) {
    long batches = 0;
    const auto batched = ga::smooth_path(m, BatchDisc{disc, width, &batches}, detour(), s);
    EXPECT_GT(batches, 0) << "width " << width;
    EXPECT_EQ(batched.profile.batch_calls, batches) << "width " << width;
    ASSERT_EQ(scalar.path.size(), batched.path.size()) << "width " << width;
    for (std::size_t k = 0; k < scalar.path.size(); ++k) {
      EXPECT_EQ(scalar.path[k], batched.path[k]) << "width " << width;
    }
  }
}

TEST(PathSmoothing, WorkStaysBounded) {
  // The smoother makes fewer than 60000 point checks and 6000 relaxation visits on the
  // detour.
  const geodex::Euclidean<2> m;
  const Disc disc{{0.0, 0.0}, 1.0};
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.01;
  const auto r = ga::smooth_path(m, disc, detour(), s);
  EXPECT_LT(r.profile.point_checks, 60000);
  EXPECT_LT(r.profile.relax_visits, 6000);
}

TEST(FrozenMetric, EqualsTheMetricAtThePoint) {
  const Eigen::Vector3d p(0.3, -0.2, 0.7), u(0.1, 0.4, -0.3), v(-0.2, 0.5, 0.25);
  const geodex::SE2LeftInvariantMetric se2{1.0, 50.0, 2.0};
  EXPECT_EQ(geodex::frozen_metric(se2, p).inner(u, v), se2.inner(p, u, v));

  auto sdf = [](const Eigen::Vector3d& q) { return 0.5 - q[0] * q[0]; };
  const geodex::SDFConformalMetric conformal{se2, sdf, 0.25, 10.0};
  EXPECT_EQ(conformal.metric_at(p).inner(u, v), conformal.inner(p, u, v));
  EXPECT_EQ(conformal.metric_at(p).norm(v), conformal.norm(p, v));

  using KE = geodex::KineticEnergyMetric<PlanarArmMass>;
  const geodex::ConfigurationSpace<geodex::Euclidean<2>, KE> arm{geodex::Euclidean<2>{},
                                                                 KE{PlanarArmMass{}}};
  const Vec2 q(0.4, 1.1), a(0.3, -0.7), b(-0.1, 0.9);
  // The Gram matrix is exact. The product with it can round in another order.
  EXPECT_NEAR(arm.metric_at(q).inner(a, b), arm.inner(q, a, b), 1e-12);
  EXPECT_NEAR(geodex::frozen_metric(arm, q).inner(a, b), arm.inner(q, a, b), 1e-12);
}

// On SO(3) and SE(3), a point has more coordinates than a tangent. The frozen Gram matrix
// takes the tangent size.
TEST(FrozenMetric, GramMatrixIsSizedByTheTangent) {
  using Inertia = geodex::ConstantSPDMetric<Eigen::Dynamic>;
  const Eigen::MatrixXd I3 = Eigen::Vector3d(1.0, 4.0, 9.0).asDiagonal();
  const geodex::ConfigurationSpace<geodex::SO3<>, Inertia> so3{geodex::SO3<>{}, Inertia{I3}};
  const auto q = so3.from_unit_cube(Eigen::Vector3d(0.2, 0.7, 0.4));
  const Eigen::Vector3d a(0.3, -0.7, 0.2), b(-0.1, 0.9, 0.5);
  ASSERT_EQ(so3.metric_at(q).gram().rows(), 3);
  EXPECT_NEAR(so3.metric_at(q).inner(a, b), so3.inner(q, a, b), 1e-12);

  const Eigen::MatrixXd I6 =
      (Eigen::VectorXd(6) << 2.0, 2.0, 2.0, 1.0, 4.0, 9.0).finished().asDiagonal();
  const geodex::ConfigurationSpace<geodex::SE3<>, Inertia> se3{geodex::SE3<>{}, Inertia{I6}};
  const auto g =
      se3.from_unit_cube((Eigen::VectorXd(6) << 0.1, 0.5, 0.9, 0.2, 0.7, 0.4).finished());
  const geodex::SE3<>::Tangent u =
      (Eigen::VectorXd(6) << 0.1, -0.2, 0.3, 0.3, -0.7, 0.2).finished();
  const geodex::SE3<>::Tangent w =
      (Eigen::VectorXd(6) << -0.4, 0.2, 0.1, -0.1, 0.9, 0.5).finished();
  ASSERT_EQ(se3.metric_at(g).gram().rows(), 6);
  EXPECT_NEAR(se3.metric_at(g).inner(u, w), se3.inner(g, u, w), 1e-12);
}

// The descent on a rigid body's inertia metric freezes that Gram matrix.
TEST(PathSmoothing, InertiaMetricOnSO3Smooths) {
  using Inertia = geodex::ConstantSPDMetric<Eigen::Dynamic>;
  using Space = geodex::ConfigurationSpace<geodex::SO3<>, Inertia>;
  const Eigen::MatrixXd I3 = Eigen::Vector3d(1.0, 4.0, 9.0).asDiagonal();
  const Space space{geodex::SO3<>{}, Inertia{I3}};
  const geodex::SO3<> so3;
  std::vector<Space::Point> path{so3.from_unit_cube(Eigen::Vector3d(0.5, 0.5, 0.5))};
  for (const Eigen::Vector3d v :
       {Eigen::Vector3d(0.6, 0.0, 0.0), Eigen::Vector3d(0.0, 0.6, 0.0),
        Eigen::Vector3d(0.0, 0.0, 0.6), Eigen::Vector3d(0.3, -0.3, 0.2)}) {
    path.push_back(so3.exp(path.back(), v));
  }
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.01;
  const auto r = ga::smooth_path(
      space, [](const Space::Point&) { return true; }, path, s);
  ASSERT_TRUE(r.collision_free);
  EXPECT_GT(r.profile.relax_visits, 0);
  for (const auto& q : r.path) EXPECT_NEAR(q.norm(), 1.0, 1e-12);
}

TEST(PathSmoothing, AsymmetricEdgeValidatorIsHonoredInPathOrder) {
  // Edges must not move backwards in x, a stand-in for a bound on reverse driving. The
  // validator sees every edge in path order, and the output moves forward only.
  const geodex::Euclidean<2> m;
  const Disc disc{{0.0, 0.0}, 1.0};
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.01;
  auto forward_and_clear = [&](const Eigen::Ref<const Eigen::VectorXd>& a,
                               const Eigen::Ref<const Eigen::VectorXd>& b) {
    if (b[0] < a[0] - 1e-12) return false;
    for (int j = 1; j < 200; ++j) {
      if (!disc(Eigen::Vector2d(a + (j / 200.0) * (b - a)))) return false;
    }
    return true;
  };
  s.edge_validator = forward_and_clear;
  const auto r = ga::smooth_path(m, disc, detour(), s);
  ASSERT_TRUE(r.collision_free);
  for (std::size_t k = 0; k + 1 < r.path.size(); ++k) {
    EXPECT_GE(r.path[k + 1][0], r.path[k][0] - 1e-12) << "edge " << k;
  }
  EXPECT_TRUE(certified(m, r.path, shrunk(disc, s.corner_tolerance), 0.005));
  EXPECT_LT(r.length, coord_length(detour()));
}

// The smoother rejects a negative, non-finite or too fine resolution. The finest one asks
// for more than kMaxEdgeSamples samples on an edge.
TEST(PathSmoothing, CollisionResolutionIsValidated) {
  const geodex::Euclidean<2> m;
  const Disc disc{{0.0, 0.0}, 1.0};
  ga::PathSmoothingSettings s;
  for (const double r :
       {-1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
    s.collision_check_resolution = r;
    EXPECT_THROW((void)ga::smooth_path(m, disc, detour(), s), std::invalid_argument) << r;
  }
  s.collision_check_resolution = 1e-9;
  const std::vector<Vec2> through = {{-2.0, 0.0}, {2.0, 0.0}};
  EXPECT_THROW((void)ga::smooth_path(m, disc, through, s), std::invalid_argument);
}

namespace {

// A box obstacle decided by comparisons alone. It gives the same answer on every machine.
struct Box {
  template <typename P>
  bool operator()(const P& q) const {
    return !(std::abs(q[0]) < 1.0 && std::abs(q[1]) < 1.0);
  }
};

// 120 waypoints zigzagging above the box, dense enough for the random shortcuts.
std::vector<Vec2> dense_zigzag() {
  std::vector<Vec2> path;
  for (int k = 0; k < 120; ++k) {
    const double x = -2.0 + 4.0 * static_cast<double>(k) / 119.0;
    const double y = (k == 0 || k == 119) ? 0.0 : 1.5 + ((k % 2 == 1) ? 0.25 : -0.25);
    path.emplace_back(x, y);
  }
  return path;
}

}  // namespace

// The random values map the engine output with a fixed algorithm, and a seed gives the
// same values with every standard library. The seeded smoother then shortens the zigzag
// to within 2 percent of the shortest path over the box, 2 + 2 sqrt(2).
TEST(PathSmoothing, SeededShortcutsUsePortableRandomValues) {
  std::mt19937_64 rng(42);
  std::vector<std::uint64_t> values;
  for (int i = 0; i < 8; ++i) values.push_back(geodex::utils::uniform_index(rng, 100));
  EXPECT_EQ(values, (std::vector<std::uint64_t>{6, 24, 50, 62, 81, 28, 36, 44}));

  const geodex::Euclidean<2> m;
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.01;
  s.round_corners = false;
  const auto input = dense_zigzag();
  const auto r = ga::smooth_path(m, Box{}, input, s);
  ASSERT_TRUE(r.collision_free);
  EXPECT_EQ(r.path.front(), input.front());
  EXPECT_EQ(r.path.back(), input.back());
  EXPECT_GT(r.profile.shortcuts, 0);
  const double shortest = 2.0 + 2.0 * std::sqrt(2.0);
  EXPECT_GE(r.length, shortest - 1e-9);
  EXPECT_LT(r.length, 1.02 * shortest);
}

// ---------------------------------------------------------------------------
// Corner rounding.
// ---------------------------------------------------------------------------

namespace {

// Turning angle at every interior point, between the incoming direction -log(c, previous)
// and the outgoing direction log(c, next), measured with the metric at c.
template <typename M>
std::vector<double> turning_angles(const M& m, const std::vector<typename M::Point>& path) {
  std::vector<double> out;
  for (std::size_t k = 1; k + 1 < path.size(); ++k) {
    const auto p = m.log(path[k], path[k - 1]);
    const auto q = m.log(path[k], path[k + 1]);
    const double c =
        -m.inner(path[k], p, q) / std::sqrt(m.inner(path[k], p, p) * m.inner(path[k], q, q));
    out.push_back(std::acos(std::clamp(c, -1.0, 1.0)));
  }
  return out;
}

double max_of(const std::vector<double>& v) {
  return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end());
}

// Interior points whose turning angle exceeds `angle` and twice that of both neighbors.
// A sampled C2 curve turns about as much at neighboring points and has none.
int sharp_corners(const std::vector<double>& turns, const double angle) {
  int n = 0;
  for (std::size_t k = 0; k < turns.size(); ++k) {
    const double before = k > 0 ? turns[k - 1] : 0.0;
    const double after = k + 1 < turns.size() ? turns[k + 1] : 0.0;
    if (turns[k] > angle && turns[k] > 2.0 * std::max(before, after)) ++n;
  }
  return n;
}

// Distance from point x to the polyline `path` in coordinates.
template <typename P>
double distance_to_polyline(const P& x, const std::vector<P>& path) {
  double best = std::numeric_limits<double>::infinity();
  for (std::size_t k = 0; k + 1 < path.size(); ++k) {
    const P d = path[k + 1] - path[k];
    const double t = std::clamp((x - path[k]).dot(d) / d.squaredNorm(), 0.0, 1.0);
    best = std::min(best, (x - (path[k] + t * d)).norm());
  }
  return best;
}

using PlanarArm =
    geodex::ConfigurationSpace<geodex::Euclidean<2>, geodex::KineticEnergyMetric<PlanarArmMass>>;

PlanarArm planar_arm() {
  return PlanarArm{geodex::Euclidean<2>{},
                   geodex::KineticEnergyMetric<PlanarArmMass>{PlanarArmMass{}}};
}

// A path the arm's kinetic energy bends into many small corners.
std::vector<Vec2> arm_path() { return {{-1.5, -2.0}, {0.0, 0.0}, {1.5, 2.0}}; }

}  // namespace

TEST(CornerRounding, TurnsSmoothlyAroundAnObstacle) {
  const geodex::Euclidean<2> m;
  const Disc disc{{0.0, 0.0}, 1.0};
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.01;
  s.round_corners = false;
  const auto sharp = ga::smooth_path(m, disc, detour(), s);
  s.round_corners = true;
  const auto r = ga::smooth_path(m, disc, detour(), s);
  ASSERT_TRUE(r.collision_free);
  EXPECT_EQ(r.profile.fallback, 0);
  EXPECT_TRUE(certified(m, r.path, shrunk(disc, s.corner_tolerance), 0.01));
  EXPECT_EQ(r.path.front(), detour().front());
  EXPECT_EQ(r.path.back(), detour().back());
  EXPECT_GT(r.profile.rounded_corners, 0);
  EXPECT_EQ(r.profile.kept_corners + r.profile.cusps, 0);
  // Without rounding, the path turns by more than 0.1 rad at a waypoint. With it, the turn
  // spreads over many waypoints, within about 1.5 steps over the disc's radius at each.
  EXPECT_GT(max_of(turning_angles(m, sharp.path)), 0.1);
  EXPECT_LT(max_of(turning_angles(m, r.path)), 0.035);
}

namespace {

// The path with every edge split into pieces no longer than `h`.
std::vector<Vec2> densified(const std::vector<Vec2>& path, const double h) {
  std::vector<Vec2> out{path.front()};
  for (std::size_t k = 1; k < path.size(); ++k) {
    const int n = std::max(1, static_cast<int>(std::ceil((path[k] - path[k - 1]).norm() / h)));
    for (int j = 1; j <= n; ++j)
      out.push_back(path[k - 1] + (j / double(n)) * (path[k] - path[k - 1]));
  }
  return out;
}

}  // namespace

// A C2 curve sampled more densely has smaller jumps of its finite-difference second
// derivative and smaller turns between chords. A corner sampled more densely keeps its turn
// and its jumps grow.
TEST(CornerRounding, TurnsShrinkWithTheTolerance) {
  const auto arm = planar_arm();
  auto free = [](const Vec2&) { return true; };
  ga::PathSmoothingSettings s;
  s.round_corners = false;
  const auto sharp = ga::smooth_path(arm, free, arm_path(), s);
  s.round_corners = true;
  s.corner_tolerance = 1e-4;
  const auto coarse = ga::smooth_path(arm, free, arm_path(), s);
  s.corner_tolerance = 1e-6;
  const auto fine = ga::smooth_path(arm, free, arm_path(), s);
  ASSERT_TRUE(coarse.collision_free);
  ASSERT_TRUE(fine.collision_free);
  ASSERT_GT(coarse.profile.rounded_corners, 0);
  EXPECT_EQ(coarse.profile.rounded_corners, fine.profile.rounded_corners);
  // A hundred times finer tolerance gives a turn at least five times smaller at every waypoint.
  EXPECT_LT(5.0 * max_of(turning_angles(arm, fine.path)), max_of(turning_angles(arm, coarse.path)));
  // The path without rounding, sampled densely, keeps its turns.
  const auto sharp_fine = densified(sharp.path, 1e-3);
  EXPECT_NEAR(max_of(turning_angles(arm, sharp_fine)), max_of(turning_angles(arm, sharp.path)),
              1e-9);
  EXPECT_GT(max_of(turning_angles(arm, sharp.path)), 20.0 * max_of(turning_angles(arm, fine.path)));
}

// The edges between returned samples stay within corner_tolerance of the curve. The
// densely sampled path stands in for the curve itself.
TEST(CornerRounding, EdgesStayWithinTheToleranceOfTheCurve) {
  const auto arm = planar_arm();
  auto free = [](const Vec2&) { return true; };
  ga::PathSmoothingSettings s;
  s.corner_tolerance = 1e-8;
  const auto dense = ga::smooth_path(arm, free, arm_path(), s);
  for (const double tol : {1e-2, 1e-3}) {
    s.corner_tolerance = tol;
    const auto r = ga::smooth_path(arm, free, arm_path(), s);
    ASSERT_EQ(r.profile.rounded_corners, dense.profile.rounded_corners) << tol;
    // The evenly spaced edges stay within the tolerance of the curve's samples, which stay
    // within the tolerance of the curve.
    double gap = 0.0;
    for (const auto& x : dense.path) gap = std::max(gap, distance_to_polyline(x, r.path));
    EXPECT_LE(gap, 2.0 * tol + 1e-7) << tol;
    EXPECT_GT(gap, 0.01 * tol) << tol;
  }
}

TEST(CornerRounding, ShrinksNextToAnObstacle) {
  // The path wraps the disc, and some curves cut into it at full size.
  const geodex::Euclidean<2> m;
  const Disc disc{{0.0, 0.0}, 1.0};
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.01;
  const auto r = ga::smooth_path(m, disc, detour(), s);
  ASSERT_TRUE(r.collision_free);
  EXPECT_GT(r.profile.rounding_retries, 0);
  EXPECT_GT(r.profile.rounded_corners, 0);
  EXPECT_TRUE(certified(m, r.path, shrunk(disc, s.corner_tolerance), 0.01));
  // In free space, no curve fails a check.
  const auto open = ga::smooth_path(
      planar_arm(), [](const Vec2&) { return true; }, arm_path());
  EXPECT_GT(open.profile.rounded_corners, 0);
  EXPECT_EQ(open.profile.rounding_retries, 0);
}

TEST(CornerRounding, KeepsTheMetricCost) {
  ga::PathSmoothingSettings on, off;
  off.round_corners = false;
  on.collision_check_resolution = off.collision_check_resolution = 0.01;
  // Euclidean. A curve lies inside its corner and is shorter.
  const geodex::Euclidean<2> m;
  const Disc disc{{0.0, 0.0}, 1.0};
  EXPECT_LE(ga::smooth_path(m, disc, detour(), on).length,
            ga::smooth_path(m, disc, detour(), off).length);
  // Kinetic energy and an SE(2) metric with a costly sideways motion. A curve may add at
  // most kRoundingCostTolerance of the length it replaces.
  const auto arm = planar_arm();
  auto free = [](const Vec2&) { return true; };
  const double arm_on = ga::smooth_path(arm, free, arm_path(), on).length;
  const double arm_off = ga::smooth_path(arm, free, arm_path(), off).length;
  EXPECT_LE(arm_on, (1.0 + ga::detail::kRoundingCostTolerance) * arm_off);
  const geodex::SE2<> se2{geodex::SE2LeftInvariantMetric{1.0, 20.0, 0.5}};
  const Disc post{{2.0, 0.0}, 0.8};
  const std::vector<Eigen::Vector3d> path = {
      {0.0, 0.0, 0.0}, {1.0, 1.2, 0.5}, {2.0, 1.4, 0.0}, {3.0, 1.2, -0.5}, {4.0, 0.0, 0.0}};
  const auto se2_on = ga::smooth_path(se2, post, path, on);
  const auto se2_off = ga::smooth_path(se2, post, path, off);
  ASSERT_TRUE(se2_on.collision_free);
  EXPECT_GT(se2_on.profile.rounded_corners, 0);
  EXPECT_LE(se2_on.length, (1.0 + ga::detail::kRoundingCostTolerance) * se2_off.length);
}

TEST(CornerRounding, IsDeterministic) {
  const geodex::Euclidean<2> m;
  const Disc disc{{0.0, 0.0}, 1.0};
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.01;
  s.output_spacing = 0.05;
  const auto a = ga::smooth_path(m, disc, detour(), s);
  const auto b = ga::smooth_path(m, disc, detour(), s);
  ASSERT_GT(a.profile.rounded_corners, 0);
  EXPECT_EQ(a.path, b.path);
  EXPECT_EQ(a.profile.point_checks, b.profile.point_checks);
}

TEST(CornerRounding, OutputSpacingBoundsEveryStep) {
  const geodex::Euclidean<2> m;
  const Disc disc{{0.0, 0.0}, 1.0};
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.01;
  s.output_spacing = 0.1;
  const auto r = ga::smooth_path(m, disc, detour(), s);
  ASSERT_TRUE(r.collision_free);
  ASSERT_GT(r.profile.rounded_corners, 0);
  EXPECT_EQ(r.path.front(), detour().front());
  EXPECT_EQ(r.path.back(), detour().back());
  for (std::size_t k = 1; k < r.path.size(); ++k) {
    EXPECT_LE((r.path[k] - r.path[k - 1]).norm(), 0.1 + 1e-9) << k;
  }
}

// Largest over smallest step between consecutive waypoints.
double step_ratio(const std::vector<Vec2>& path) {
  double lo = std::numeric_limits<double>::infinity();
  double hi = 0.0;
  for (std::size_t k = 1; k < path.size(); ++k) {
    const double step = (path[k] - path[k - 1]).norm();
    lo = std::min(lo, step);
    hi = std::max(hi, step);
  }
  return hi / lo;
}

// By default, the waypoints lie at one even step, the longest whose edges stay within the
// corner tolerance of the smoothed path. They do not cluster around the disc.
TEST(OutputSpacing, DefaultStepIsEven) {
  const geodex::Euclidean<2> m;
  const Disc disc{{0.0, 0.0}, 1.0};
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.01;
  const auto r = ga::smooth_path(m, disc, detour(), s);
  ASSERT_TRUE(r.collision_free);
  ASSERT_EQ(r.profile.kept_corners + r.profile.cusps, 0);
  EXPECT_GT(r.path.size(), 10u);
  EXPECT_LT(step_ratio(r.path), 1.001);
  EXPECT_EQ(r.path.front(), detour().front());
  EXPECT_EQ(r.path.back(), detour().back());
  EXPECT_TRUE(certified(m, r.path, shrunk(disc, s.corner_tolerance), 0.01));
}

// With output_spacing set, the steps stay even and no longer than it.
TEST(OutputSpacing, StepsStayWithinTheOutputSpacing) {
  const geodex::Euclidean<2> m;
  const Disc disc{{0.0, 0.0}, 1.0};
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.01;
  const auto longest = ga::smooth_path(m, disc, detour(), s);
  s.output_spacing = 0.01;
  const auto r = ga::smooth_path(m, disc, detour(), s);
  ASSERT_TRUE(r.collision_free);
  EXPECT_GT(r.path.size(), longest.path.size());
  EXPECT_LT(step_ratio(r.path), 1.001);
  for (std::size_t k = 1; k < r.path.size(); ++k) {
    EXPECT_LE((r.path[k] - r.path[k - 1]).norm(), 0.01 + 1e-12) << k;
  }
}

// A looser corner tolerance gives a longer step.
TEST(OutputSpacing, TheToleranceSetsTheStep) {
  const geodex::Euclidean<2> m;
  const Disc disc{{0.0, 0.0}, 1.0};
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.01;
  const auto fine = ga::smooth_path(m, disc, detour(), s);
  s.corner_tolerance = 1e-3;
  const auto coarse = ga::smooth_path(m, disc, detour(), s);
  ASSERT_TRUE(fine.collision_free && coarse.collision_free);
  EXPECT_LT(coarse.path.size(), fine.path.size());
  EXPECT_LT(step_ratio(coarse.path), 1.001);
}

// Without rounding, the path keeps its turns, and the edges between two turns have equal
// length.
TEST(OutputSpacing, TurnsRemainWithoutRounding) {
  const geodex::Euclidean<2> m;
  const Disc disc{{0.0, 0.0}, 1.0};
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.01;
  s.round_corners = false;
  s.output_spacing = 0.05;
  const auto r = ga::smooth_path(m, disc, detour(), s);
  ASSERT_TRUE(r.collision_free);
  EXPECT_TRUE(certified(m, r.path, disc, 0.01));
  int straight = 0;
  for (std::size_t k = 1; k + 1 < r.path.size(); ++k) {
    const Vec2 a = r.path[k] - r.path[k - 1];
    const Vec2 b = r.path[k + 1] - r.path[k];
    EXPECT_LE(a.norm(), 0.05 + 1e-12) << k;
    if (a.normalized().dot(b.normalized()) < 1.0 - 1e-9) continue;
    ++straight;
    EXPECT_NEAR(a.norm(), b.norm(), 1e-9) << k;
  }
  EXPECT_GT(straight, 0);
}

// Without rounding, the turns of a path stay waypoints. An arc whose waypoints each turn by
// 2e-5 rad has no turn, and its evenly spaced edges stay within the corner tolerance of it.
TEST(OutputSpacing, AGentleArcStaysWithinTheToleranceWithoutRounding) {
  const geodex::Euclidean<2> m;
  // An arc of radius 100 and length 2 through 1001 waypoints, each turning by 2e-5 rad. The
  // validity is the band within 1e-8 of the circle. It admits every edge of the arc and no
  // chord across two of them.
  const double radius = 100.0;
  const Vec2 center(0.0, radius);
  std::vector<Vec2> arc;
  for (int k = 0; k <= 1000; ++k) {
    const double t = 0.002 * k / radius;
    arc.emplace_back(radius * std::sin(t), radius * (1.0 - std::cos(t)));
  }
  const auto band = [&](const Vec2& q) { return std::abs((q - center).norm() - radius) <= 1e-8; };
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.001;
  s.round_corners = false;
  const auto r = ga::smooth_path(m, band, arc, s);
  ASSERT_TRUE(r.collision_free);
  EXPECT_GT(r.path.size(), 2u);
  for (std::size_t k = 1; k < r.path.size(); ++k) {
    for (const double t : {0.25, 0.5, 0.75}) {
      const Vec2 q = r.path[k - 1] + t * (r.path[k] - r.path[k - 1]);
      EXPECT_LE(std::abs((q - center).norm() - radius), s.corner_tolerance + 1e-8) << k;
    }
  }
}

// A differential drive drives 1 m, turns along an arc and backs up along it. A rounding
// curve ends exactly on the reversal. The reversal stays a waypoint of the spaced path.
TEST(OutputSpacing, ACuspACurveEndsOnStays) {
  const geodex::SE2<> se2{geodex::SE2LeftInvariantMetric{1.0, 10.0, 2.0}};
  const double angle = 0.4;
  const double radius = 0.8;
  const Eigen::Vector3d b(1.0, 0.0, 0.0);
  const Eigen::Vector3d c(1.0 + radius * std::sin(angle), radius * (1.0 - std::cos(angle)), angle);
  const std::vector<Eigen::Vector3d> path = {{0.0, 0.0, 0.0}, b, c, se2.geodesic(c, b, 0.7)};
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.01;
  s.path_predicate = [far = c[0] - 0.02](const std::vector<Eigen::VectorXd>& p) {
    return std::any_of(p.begin(), p.end(), [&](const Eigen::VectorXd& q) { return q[0] >= far; });
  };
  const auto free = [](const Eigen::Vector3d&) { return true; };
  s.output_spacing = 0.02;
  const auto spaced = ga::smooth_path(se2, free, path, s);
  ASSERT_TRUE(spaced.collision_free);
  ASSERT_EQ(spaced.profile.cusps, 1);
  // The reversal is the waypoint farthest along x, and the path turns back there.
  const auto far = std::max_element(spaced.path.begin(), spaced.path.end(),
                                    [](const auto& p, const auto& q) { return p[0] < q[0]; });
  ASSERT_NE(far, spaced.path.begin());
  ASSERT_NE(far + 1, spaced.path.end());
  EXPECT_GE((*far)[0], c[0] - 0.02);
  EXPECT_LT((*(far + 1))[0], (*far)[0]);
}

// When the path predicate rejects the evenly spaced path, the smoother's own waypoints
// return.
TEST(OutputSpacing, ARejectedSpacingKeepsTheSmoothersWaypoints) {
  const geodex::Euclidean<2> m;
  const auto free = [](const Vec2&) { return true; };
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.01;
  const auto even = ga::smooth_path(m, free, detour(), s);
  s.path_predicate = [n = even.path.size()](const std::vector<Eigen::VectorXd>& path) {
    return path.size() != n;
  };
  const auto r = ga::smooth_path(m, free, detour(), s);
  ASSERT_TRUE(r.collision_free);
  EXPECT_NE(r.path.size(), even.path.size());
  EXPECT_GT(step_ratio(r.path), 1.1);
}

// On the sphere, the steps along a great circle are equal and no longer than the spacing.
TEST(OutputSpacing, EqualStepsOnTheSphere) {
  const geodex::Sphere<> m;
  const auto free = [](const Eigen::Vector3d&) { return true; };
  std::vector<Eigen::Vector3d> path;
  for (int k = 0; k <= 4; ++k) {
    const double t = k / 4.0 * 2.0;
    path.push_back(Eigen::Vector3d(std::cos(t), std::sin(t), 0.05 * (k % 2)).normalized());
  }
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.01;
  s.output_spacing = 0.07;
  const auto r = ga::smooth_path(m, free, path, s);
  ASSERT_TRUE(r.collision_free);
  ASSERT_GE(r.path.size(), 3u);
  const double step = m.distance(r.path[0], r.path[1]);
  for (std::size_t k = 1; k < r.path.size(); ++k) {
    EXPECT_NEAR(r.path[k].norm(), 1.0, 1e-12);
    EXPECT_NEAR(m.distance(r.path[k - 1], r.path[k]), step, 1e-9) << k;
  }
  EXPECT_LE(step, 0.07 + 1e-12);
}

// A differential drive drives 2 m forward and backs up 1 m along the same line. A path
// predicate makes it reach x = 1.95, and the corner turns by 180 degrees under the
// metric. The corner stays, and a robot stops there.
TEST(CornerRounding, KeepsACusp) {
  const geodex::SE2<> se2{geodex::SE2LeftInvariantMetric{1.0, 50.0, 2.0}};
  const std::vector<Eigen::Vector3d> path = {{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.01;
  s.path_predicate = [](const std::vector<Eigen::VectorXd>& p) {
    return std::any_of(p.begin(), p.end(), [](const Eigen::VectorXd& q) { return q[0] >= 1.95; });
  };
  auto free = [](const Eigen::Vector3d&) { return true; };
  const auto r = ga::smooth_path(se2, free, path, s);
  ASSERT_TRUE(r.collision_free);
  EXPECT_EQ(r.profile.cusps, 1);
  EXPECT_GT(max_of(turning_angles(se2, r.path)), 0.99 * std::acos(-1.0));
  // A limit of pi rounds the reversal. The curve then stops at its tip, where both
  // tangent directions cancel.
  s.corner_max_angle = std::acos(-1.0);
  EXPECT_EQ(ga::smooth_path(se2, free, path, s).profile.cusps, 0);
}

// Driving straight, turning in place and driving on becomes one smooth motion. The curve
// is built from body twists through the group's log and exp.
TEST(CornerRounding, RoundsATurnInPlace) {
  const geodex::SE2<> se2{geodex::SE2LeftInvariantMetric{1.0, 50.0, 2.0}};
  const auto corridor = [](const Eigen::Vector3d& q) {
    return (std::abs(q[1]) <= 0.3 && q[0] <= 2.3) || (q[0] >= 1.7 && q[0] <= 2.3 && q[1] >= -0.3);
  };
  const double half_pi = 0.5 * std::acos(-1.0);
  const std::vector<Eigen::Vector3d> path = {
      {0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {2.0, 0.0, half_pi}, {2.0, 2.0, half_pi}};
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.01;
  const auto r = ga::smooth_path(se2, corridor, path, s);
  ASSERT_TRUE(r.collision_free);
  EXPECT_GT(r.profile.rounded_corners, 0);
  EXPECT_EQ(r.profile.cusps, 0);
  EXPECT_EQ(sharp_corners(turning_angles(se2, r.path), 0.05), r.profile.kept_corners);
}

// On curved manifolds, the curve lives in the tangent space at the corner. The samples stay
// on the manifold, and the path turns smoothly under the metric.
TEST(CornerRounding, RoundsCornersOnCurvedManifolds) {
  {
    // A cap at the north pole blocks the great circle from start to goal.
    const geodex::Sphere<> m;
    auto valid = [](const Eigen::Vector3d& q) { return q[2] < 0.8; };
    std::vector<Eigen::Vector3d> path = {
        {1.0, 0.0, 0.3}, {0.6, 0.6, 0.5}, {0.0, 0.9, 0.4}, {-0.6, 0.6, 0.5}, {-1.0, 0.0, 0.3}};
    for (auto& q : path) q.normalize();
    ga::PathSmoothingSettings s;
    s.collision_check_resolution = 0.005;
    const auto r = ga::smooth_path(m, valid, path, s);
    ASSERT_TRUE(r.collision_free);
    EXPECT_GT(r.profile.rounded_corners, 0);
    for (const auto& q : r.path) EXPECT_NEAR(q.norm(), 1.0, 1e-12);
    EXPECT_EQ(sharp_corners(turning_angles(m, r.path), 0.05), r.profile.kept_corners);
  }
  {
    using Inertia = geodex::ConstantSPDMetric<Eigen::Dynamic>;
    using Space = geodex::ConfigurationSpace<geodex::SO3<>, Inertia>;
    const Space space{geodex::SO3<>{}, Inertia{Eigen::Vector3d(1.0, 4.0, 9.0).asDiagonal()}};
    const geodex::SO3<> so3;
    std::vector<Space::Point> path{so3.from_unit_cube(Eigen::Vector3d(0.5, 0.5, 0.5))};
    for (const Eigen::Vector3d v : {Eigen::Vector3d(0.6, 0.0, 0.0), Eigen::Vector3d(0.0, 0.6, 0.0),
                                    Eigen::Vector3d(0.0, 0.0, 0.6)}) {
      path.push_back(so3.exp(path.back(), v));
    }
    ga::PathSmoothingSettings s;
    s.collision_check_resolution = 0.01;
    s.round_corners = false;
    const auto sharp = ga::smooth_path(
        space, [](const Space::Point&) { return true; }, path, s);
    s.round_corners = true;
    const auto r = ga::smooth_path(
        space, [](const Space::Point&) { return true; }, path, s);
    ASSERT_TRUE(r.collision_free);
    EXPECT_GT(r.profile.rounded_corners, 0);
    for (const auto& q : r.path) EXPECT_NEAR(q.norm(), 1.0, 1e-12);
    EXPECT_LT(max_of(turning_angles(space, r.path)),
              0.5 * max_of(turning_angles(space, sharp.path)));
    EXPECT_LE(r.length, (1.0 + ga::detail::kRoundingCostTolerance) * sharp.length);
  }
  {
    // A rigid body moves around a ball while it turns.
    using Inertia = geodex::ConstantSPDMetric<Eigen::Dynamic>;
    using Space = geodex::ConfigurationSpace<geodex::SE3<>, Inertia>;
    const Eigen::MatrixXd I6 =
        (Eigen::VectorXd(6) << 2.0, 2.0, 2.0, 1.0, 4.0, 9.0).finished().asDiagonal();
    const Space space{geodex::SE3<>{}, Inertia{I6}};
    const auto pose = [](double x, double y, double yaw) {
      Space::Point g;
      g << x, y, 0.0, 0.0, 0.0, std::sin(0.5 * yaw), std::cos(0.5 * yaw);
      return g;
    };
    const std::vector<Space::Point> path = {pose(0.0, 0.0, 0.0), pose(0.5, 1.0, 0.4),
                                            pose(1.5, 1.0, 0.8), pose(2.0, 0.0, 1.2)};
    auto outside = [](const Space::Point& g) {
      return (g.head<3>() - Eigen::Vector3d(1.0, 0.0, 0.0)).norm() > 0.7;
    };
    ga::PathSmoothingSettings s;
    s.collision_check_resolution = 0.01;
    s.round_corners = false;
    const auto sharp = ga::smooth_path(space, outside, path, s);
    s.round_corners = true;
    const auto r = ga::smooth_path(space, outside, path, s);
    ASSERT_TRUE(r.collision_free);
    EXPECT_GT(r.profile.rounded_corners, 0);
    for (const auto& g : r.path) EXPECT_NEAR(g.tail<4>().norm(), 1.0, 1e-12);
    EXPECT_EQ(sharp_corners(turning_angles(space, r.path), 0.05), r.profile.kept_corners);
    EXPECT_LT(sharp_corners(turning_angles(space, r.path), 0.05),
              sharp_corners(turning_angles(space, sharp.path), 0.05));
    EXPECT_LE(r.length, (1.0 + ga::detail::kRoundingCostTolerance) * sharp.length);
  }
}

TEST(CornerRounding, CornerSettingsAreValidated) {
  const geodex::Euclidean<2> m;
  const Disc disc{{0.0, 0.0}, 1.0};
  ga::PathSmoothingSettings s;
  for (const double t : {0.0, -1e-4, std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity()}) {
    s.corner_tolerance = t;
    EXPECT_THROW((void)ga::smooth_path(m, disc, detour(), s), std::invalid_argument) << t;
  }
  s.corner_tolerance = 1e-4;
  for (const double a : {-0.1, 3.2, std::numeric_limits<double>::quiet_NaN()}) {
    s.corner_max_angle = a;
    EXPECT_THROW((void)ga::smooth_path(m, disc, detour(), s), std::invalid_argument) << a;
  }
  // Without rounding, the smoother reads corner_tolerance for the spacing and not
  // corner_max_angle.
  s.round_corners = false;
  EXPECT_NO_THROW((void)ga::smooth_path(m, disc, detour(), s));
  s.corner_tolerance = -1.0;
  EXPECT_THROW((void)ga::smooth_path(m, disc, detour(), s), std::invalid_argument);
}

// A piece of an edge is checked at the samples of the whole edge. On its own, the piece would
// place its samples elsewhere and could find a point that the edge's samples step over.
TEST(CornerRounding, APieceIsCheckedAtTheSamplesOfItsEdge) {
  const geodex::Euclidean<2> m;
  const ga::PathSmoothingSettings s;
  ga::PathSmoothingProfile profile;
  std::vector<Vec2> seen;
  const auto record = [&seen](const Vec2& q) {
    seen.push_back(q);
    return true;
  };
  const ga::detail::Oracle<geodex::Euclidean<2>, decltype(record)> oracle(m, record, s, 0.1,
                                                                          profile);
  const Vec2 a(0.0, 0.0);
  const Vec2 b(1.05, 0.0);
  ASSERT_TRUE(oracle.edge(a, b));
  std::vector<double> edge;
  for (const Vec2& q : seen) edge.push_back(q[0]);
  std::sort(edge.begin(), edge.end());
  ASSERT_EQ(edge.size(), 10u);  // 11 intervals of 0.0955

  seen.clear();
  ASSERT_TRUE(oracle.piece(a, b, 0.2, 0.7));
  std::vector<double> piece;
  for (const Vec2& q : seen) piece.push_back(q[0]);
  std::sort(piece.begin(), piece.end());
  // The edge's samples 3 to 7 lie inside, and the piece checks exactly those.
  ASSERT_EQ(piece.size(), 5u);
  for (std::size_t i = 0; i < piece.size(); ++i) EXPECT_EQ(piece[i], edge[i + 2]) << i;

  // Checked as an edge of its own, the same piece samples other points.
  seen.clear();
  ASSERT_TRUE(oracle.edge(m.geodesic(a, b, 0.2), m.geodesic(a, b, 0.7)));
  EXPECT_TRUE(std::any_of(seen.begin(), seen.end(), [&](const Vec2& q) {
    return std::none_of(edge.begin(), edge.end(), [&](const double x) { return x == q[0]; });
  }));
}

namespace {

// Largest angle between consecutive steps of coordinates [from, to) along a path, where both
// steps move.
template <typename P>
double max_turn(const std::vector<P>& path, const int from, const int to) {
  double worst = 0.0;
  for (std::size_t k = 1; k + 1 < path.size(); ++k) {
    const Eigen::VectorXd u = (path[k] - path[k - 1]).segment(from, to - from);
    const Eigen::VectorXd v = (path[k + 1] - path[k]).segment(from, to - from);
    if (u.norm() < 1e-9 || v.norm() < 1e-9) continue;
    worst = std::max(worst, std::acos(std::clamp(u.dot(v) / (u.norm() * v.norm()), -1.0, 1.0)));
  }
  return worst;
}

}  // namespace

// The first two coordinates must turn sharply at (1, 0), and the path keeps within 0.1 of
// its corner. Rounded together, the last two turn as sharply. With sharp_coordinates, the
// first two keep the corner and the last two turn along a curve.
TEST(CornerRounding, LeadingCoordinatesKeepTheCornerAndTheOthersTurnSmoothly) {
  using Vec4 = Eigen::Vector4d;
  const geodex::Euclidean<4> m;
  const std::vector<Vec4> path = {
      {0.0, 0.0, 0.0, 0.0}, {1.0, 0.0, 1.0, 0.0}, {1.0, 1.0, 1.0, 1.0}};
  const std::vector<Vec2> lead = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}};
  const auto valid = [&](const Vec4& q) {
    return distance_to_polyline(Vec2(q.head<2>()), lead) < 1e-3 &&
           distance_to_polyline(q, path) < 0.1;
  };
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.005;
  const auto together = ga::smooth_path(m, valid, path, s);
  s.sharp_coordinates = 2;
  const auto split = ga::smooth_path(m, valid, path, s);
  ASSERT_TRUE(together.collision_free && split.collision_free);
  EXPECT_EQ(together.profile.split_corners, 0);
  EXPECT_GE(split.profile.split_corners, 1);
  EXPECT_EQ(split.profile.kept_corners, 0);
  // The first two coordinates still turn by about 90 degrees at one waypoint.
  EXPECT_GT(max_turn(split.path, 0, 2), 0.45 * std::acos(-1.0));
  EXPECT_LT(max_turn(split.path, 2, 4), 0.25 * max_turn(together.path, 2, 4));
  EXPECT_TRUE(certified(m, split.path, valid, 0.005));
  EXPECT_EQ(split.path.front(), path.front());
  EXPECT_EQ(split.path.back(), path.back());
}

// Covering every coordinate leaves nothing to round apart, and the curves stay the same.
TEST(CornerRounding, SharpCoordinatesCoveringEveryCoordinateChangeNothing) {
  const geodex::Euclidean<2> m;
  const Disc disc{{0.0, 0.0}, 1.0};
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.01;
  const auto plain = ga::smooth_path(m, disc, detour(), s);
  s.sharp_coordinates = 2;
  const auto covered = ga::smooth_path(m, disc, detour(), s);
  EXPECT_EQ(plain.path, covered.path);
  s.sharp_coordinates = -1;
  EXPECT_THROW((void)ga::smooth_path(m, disc, detour(), s), std::invalid_argument);
}

// The edge from A to C passes at its own samples, and between them a stretch of it is invalid,
// as where a taut edge grazes an obstacle. A cap blocks the shortcut from A to B, and on the
// sphere the smoother only shortcuts. The curve at C starts on the edge past the stretch, and
// the stretch does not change the result.
TEST(CornerRounding, AStretchBetweenTheSamplesOfAnEdgeLeavesTheCurve) {
  using Vec3 = Eigen::Vector3d;
  const geodex::Sphere<> sphere;
  const Vec3 a(1.0, 0.0, 0.0);
  const Vec3 c(0.0, 1.0, 0.0);
  const Vec3 b = Vec3(-0.3, 1.0, 1.0).normalized();
  const Vec3 cap = (a + b).normalized();
  const double resolution = 0.07;
  const double intervals = std::ceil(0.5 * std::acos(-1.0) / resolution - 1e-9);
  const auto outside_cap = [&](const Vec3& q) {
    return std::acos(std::min(1.0, q.dot(cap))) > 0.35;
  };
  // Between t = 0.02 and 0.12 along the edge, only the edge's samples are valid.
  const auto stretch = [&](const Vec3& q) {
    if (std::abs(q[2]) > 1e-12 || q[0] < 0.0 || q[1] < 0.0) return true;
    const double t = std::atan2(q[1], q[0]) / (0.5 * std::acos(-1.0));
    if (!(t > 0.02 && t < 0.12)) return true;
    return std::abs(t * intervals - std::round(t * intervals)) < 1e-6;
  };
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = resolution;
  const auto plain = ga::smooth_path(sphere, outside_cap, std::vector<Vec3>{a, c, b}, s);
  const auto with_stretch = ga::smooth_path(
      sphere, [&](const Vec3& q) { return outside_cap(q) && stretch(q); },
      std::vector<Vec3>{a, c, b}, s);
  ASSERT_TRUE(plain.collision_free && with_stretch.collision_free);
  EXPECT_EQ(plain.profile.rounded_corners, 1);
  EXPECT_EQ(with_stretch.profile.rounded_corners, 1);
  EXPECT_EQ(with_stretch.profile.kept_corners, 0);
  EXPECT_EQ(with_stretch.path, plain.path);
}
