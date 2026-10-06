/// @file test_collision.cpp
/// @brief Tests for geodex::collision module.

#include <cmath>

#include <algorithm>
#include <limits>
#include <numbers>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "geodex/collision/circle_sdf.hpp"
#include "geodex/collision/distance_grid.hpp"
#include "geodex/collision/footprint_grid_checker.hpp"
#include "geodex/collision/polygon_footprint.hpp"
#include "geodex/collision/rectangle_sdf.hpp"
#include "geodex/utils/math.hpp"

using namespace geodex::collision;

// ---------------------------------------------------------------------------
// Fast exp
// ---------------------------------------------------------------------------

TEST(FastExp, ApproximatesStdExp) {
  // Schraudolph's trick with bias correction (c=60801, scaled by 2^32 for the 64-bit
  // adaptation) has a maximum relative error of about 4%. A linear chord approximation to
  // 2^f on each unit interval does not do better.
  for (double x = -10.0; x <= 10.0; x += 0.5) {
    const double approx = geodex::utils::fast_exp(x);
    const double exact = std::exp(x);
    EXPECT_NEAR(approx / exact, 1.0, 0.04) << "x=" << x;
  }
}

TEST(FastExp, ClampsLargeNegative) {
  const double result = geodex::utils::fast_exp(-800.0);
  EXPECT_GE(result, 0.0);
  EXPECT_LT(result, 1e-300);
}

// ---------------------------------------------------------------------------
// CircleSDF
// ---------------------------------------------------------------------------

TEST(CircleSDF, DistanceOutside) {
  CircleSDF c(0.0, 0.0, 1.0);
  Eigen::Vector3d q(3.0, 0.0, 0.0);
  EXPECT_NEAR(c(q), 2.0, 1e-12);
}

TEST(CircleSDF, DistanceInside) {
  CircleSDF c(0.0, 0.0, 2.0);
  Eigen::Vector3d q(0.5, 0.0, 0.0);
  EXPECT_NEAR(c(q), -1.5, 1e-12);
}

TEST(CircleSDF, OnBoundary) {
  CircleSDF c(1.0, 1.0, 0.5);
  Eigen::Vector3d q(1.5, 1.0, 0.0);
  EXPECT_NEAR(c(q), 0.0, 1e-12);
}

TEST(CircleSmoothSDF, SingleCircle) {
  CircleSmoothSDF sdf({CircleSDF(0.0, 0.0, 1.0)}, 20.0);
  Eigen::Vector3d q(3.0, 0.0, 0.0);
  EXPECT_NEAR(sdf(q), 2.0, 0.05);  // smooth-min ~= exact for single obstacle
}

TEST(CircleSmoothSDF, TwoCircles) {
  CircleSmoothSDF sdf({CircleSDF(0.0, 0.0, 1.0), CircleSDF(5.0, 0.0, 1.0)}, 20.0);
  // The midpoint between the two circles lies 1.5 from each.
  Eigen::Vector3d q(2.5, 0.0, 0.0);
  double d = sdf(q);
  EXPECT_GT(d, 0.0);
  // Smooth-min is slightly less than the hard min of 1.5.
  EXPECT_LT(d, 1.5);
  EXPECT_GT(d, 1.0);
}

TEST(CircleSmoothSDF, IsFree) {
  CircleSmoothSDF sdf({CircleSDF(0.0, 0.0, 1.0)}, 20.0);
  Eigen::Vector3d inside(0.0, 0.0, 0.0);
  Eigen::Vector3d outside(3.0, 0.0, 0.0);
  EXPECT_FALSE(sdf.is_free(inside));
  EXPECT_TRUE(sdf.is_free(outside));
}

// ---------------------------------------------------------------------------
// RectObstacle and SAT
// ---------------------------------------------------------------------------

TEST(RectOverlap, OverlappingRects) {
  RectObstacle a{0.0, 0.0, 0.0, 1.0, 1.0};
  RectObstacle b{1.5, 0.0, 0.0, 1.0, 1.0};
  EXPECT_TRUE(rects_overlap(a, b));
}

TEST(RectOverlap, NonOverlappingRects) {
  RectObstacle a{0.0, 0.0, 0.0, 1.0, 1.0};
  RectObstacle b{5.0, 0.0, 0.0, 1.0, 1.0};
  EXPECT_FALSE(rects_overlap(a, b));
}

TEST(RectOverlap, RotatedOverlap) {
  RectObstacle a{0.0, 0.0, 0.0, 2.0, 0.5};
  RectObstacle b{1.0, 1.0, std::numbers::pi / 4.0, 2.0, 0.5};
  // Diagonal rectangle overlaps axis-aligned one.
  EXPECT_TRUE(rects_overlap(a, b));
}

TEST(RectCorners, AxisAligned) {
  RectObstacle r{0.0, 0.0, 0.0, 1.0, 0.5};
  auto corners = rect_corners(r);
  EXPECT_NEAR(corners[0][0], -1.0, 1e-12);
  EXPECT_NEAR(corners[0][1], -0.5, 1e-12);
  EXPECT_NEAR(corners[2][0], 1.0, 1e-12);
  EXPECT_NEAR(corners[2][1], 0.5, 1e-12);
}

// ---------------------------------------------------------------------------
// RectSmoothSDF
// ---------------------------------------------------------------------------

TEST(RectSmoothSDF, OutsideDistance) {
  RectSmoothSDF sdf({RectObstacle{0.0, 0.0, 0.0, 1.0, 1.0}}, 20.0);
  // A point inside the bounding sphere, br = skip_dist + diag = 1.0 + 1.414 = 2.414.
  Eigen::Vector3d q(1.5, 0.0, 0.0);
  double d = sdf(q);
  EXPECT_NEAR(d, 0.5, 0.05);
}

TEST(RectSmoothSDF, InsideDistance) {
  RectSmoothSDF sdf({RectObstacle{0.0, 0.0, 0.0, 2.0, 2.0}}, 20.0);
  Eigen::Vector3d q(0.0, 0.0, 0.0);
  double d = sdf(q);
  EXPECT_LT(d, 0.0);
}

TEST(RectSmoothSDF, InflationReducesDistance) {
  RectSmoothSDF no_infl({RectObstacle{0.0, 0.0, 0.0, 1.0, 1.0}}, 20.0, 0.0);
  RectSmoothSDF with_infl({RectObstacle{0.0, 0.0, 0.0, 1.0, 1.0}}, 20.0, 0.5);
  // A point inside the bounding sphere. The SDF there is computed, not clipped.
  Eigen::Vector3d q(1.5, 0.0, 0.0);
  EXPECT_NEAR(no_infl(q) - with_infl(q), 0.5, 0.05);
}

// ---------------------------------------------------------------------------
// DistanceGrid
// ---------------------------------------------------------------------------

TEST(DistanceGrid, BilinearInterpolation) {
  // 3x3 grid with known values.
  std::vector<double> data = {0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0};
  DistanceGrid grid(3, 3, 1.0, data);

  // Exact grid points.
  EXPECT_NEAR(grid.distance_at(0.0, 0.0), 0.0, 1e-12);
  EXPECT_NEAR(grid.distance_at(1.0, 0.0), 1.0, 1e-12);
  EXPECT_NEAR(grid.distance_at(1.0, 1.0), 2.0, 1e-12);

  // The bilinear value at (0.5, 0.5) is avg(0, 1, 1, 2) = 1.0.
  EXPECT_NEAR(grid.distance_at(0.5, 0.5), 1.0, 1e-12);
}

TEST(DistanceGrid, BatchMatchesScalar) {
  std::vector<double> data = {0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0,
                              2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 6.0};
  DistanceGrid grid(4, 4, 1.0, data);

  double xs[] = {0.5, 1.3, 2.7, 0.1, 1.8};
  double ys[] = {0.5, 1.7, 0.2, 2.5, 2.1};
  double batch_out[5], scalar_out[5];

  grid.distance_at_batch(xs, ys, batch_out, 5);
  for (int i = 0; i < 5; ++i) {
    scalar_out[i] = grid.distance_at(xs[i], ys[i]);
  }

  for (int i = 0; i < 5; ++i) {
    EXPECT_NEAR(batch_out[i], scalar_out[i], 1e-12) << "i=" << i;
  }
}

TEST(GridSDF, Callable) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
  DistanceGrid grid(2, 2, 1.0, data);
  GridSDF sdf(&grid);

  Eigen::Vector3d q(0.5, 0.5, 0.0);
  EXPECT_NEAR(sdf(q), 2.5, 1e-12);
}

TEST(InflatedSDF, SubtractsInflation) {
  std::vector<double> data = {5.0, 5.0, 5.0, 5.0};
  DistanceGrid grid(2, 2, 1.0, data);
  GridSDF base_sdf(&grid);
  InflatedSDF inflated(base_sdf, 1.5);

  Eigen::Vector3d q(0.0, 0.0, 0.0);
  EXPECT_NEAR(inflated(q), 3.5, 1e-12);
}

// ---------------------------------------------------------------------------
// PolygonFootprint
// ---------------------------------------------------------------------------

TEST(PolygonFootprint, RectangleSampleCount) {
  auto fp = PolygonFootprint::rectangle(2.0, 1.0, 4);
  // 4 edges * 4 samples = 16, padded to even = 16.
  EXPECT_EQ(fp.sample_count_raw(), 16);
  EXPECT_EQ(fp.sample_count() % 2, 0);
}

TEST(PolygonFootprint, BoundingRadius) {
  auto fp = PolygonFootprint::rectangle(3.0, 4.0, 2);
  // Diagonal of 3x4 rect = 5.0
  EXPECT_NEAR(fp.bounding_radius(), 5.0, 1e-12);
}

TEST(PolygonFootprint, TransformIdentity) {
  auto fp = PolygonFootprint::rectangle(1.0, 0.5, 2);
  const int n = fp.sample_count();
  std::vector<double> wx(n), wy(n);

  // Zero rotation, zero translation.
  fp.transform(0.0, 0.0, 0.0, wx.data(), wy.data());

  // World coords should match body coords.
  for (int i = 0; i < fp.sample_count_raw(); ++i) {
    EXPECT_NEAR(wx[i], fp.body_x()[i], 1e-12);
    EXPECT_NEAR(wy[i], fp.body_y()[i], 1e-12);
  }
}

TEST(PolygonFootprint, TransformRotation90) {
  auto fp = PolygonFootprint::rectangle(1.0, 0.5, 1);
  // 4 edges * 1 sample = 4 samples (just corners).
  const int n = fp.sample_count();
  std::vector<double> wx(n), wy(n);

  fp.transform(0.0, 0.0, std::numbers::pi / 2.0, wx.data(), wy.data());

  // A 90 deg rotation maps (x,y) to (-y,x).
  for (int i = 0; i < fp.sample_count_raw(); ++i) {
    EXPECT_NEAR(wx[i], -fp.body_y()[i], 1e-10);
    EXPECT_NEAR(wy[i], fp.body_x()[i], 1e-10);
  }
}

TEST(PolygonFootprint, TransformTranslation) {
  auto fp = PolygonFootprint::rectangle(1.0, 0.5, 1);
  const int n = fp.sample_count();
  std::vector<double> wx(n), wy(n);

  fp.transform(10.0, 20.0, 0.0, wx.data(), wy.data());

  for (int i = 0; i < fp.sample_count_raw(); ++i) {
    EXPECT_NEAR(wx[i], fp.body_x()[i] + 10.0, 1e-12);
    EXPECT_NEAR(wy[i], fp.body_y()[i] + 20.0, 1e-12);
  }
}

// ---------------------------------------------------------------------------
// FootprintGridChecker
// ---------------------------------------------------------------------------

TEST(FootprintGridChecker, ClearInFreeSpace) {
  // A uniform distance field with 5.0 m clearance everywhere.
  std::vector<double> data(100 * 100, 5.0);
  DistanceGrid grid(100, 100, 0.1, data);  // 10m x 10m world

  auto fp = PolygonFootprint::rectangle(0.5, 0.3, 4);
  FootprintGridChecker checker(&grid, fp, 0.1);

  Eigen::Vector3d q(5.0, 5.0, 0.5);  // center of the world
  EXPECT_TRUE(checker.is_valid(q));
  EXPECT_GT(checker(q), 0.0);
}

TEST(FootprintGridChecker, CollisionInObstacle) {
  // Create a grid with a wall at x = 5m (columns 50+).
  std::vector<double> data(100 * 100, 5.0);
  for (int r = 0; r < 100; ++r) {
    for (int c = 50; c < 100; ++c) {
      // Distance decreases as we approach the wall.
      data[r * 100 + c] = static_cast<double>(c - 50) * 0.1;
    }
  }
  DistanceGrid grid(100, 100, 0.1, data);

  auto fp = PolygonFootprint::rectangle(0.5, 0.3, 4);
  FootprintGridChecker checker(&grid, fp, 0.0);

  // The robot sits at the wall edge, and its footprint extends into the wall.
  Eigen::Vector3d q(5.0, 5.0, 0.0);
  EXPECT_FALSE(checker.is_valid(q));

  // The robot is well away from the wall and clear.
  Eigen::Vector3d q2(2.0, 5.0, 0.0);
  EXPECT_TRUE(checker.is_valid(q2));
}

TEST(FootprintGridChecker, SafetyMargin) {
  std::vector<double> data(100 * 100, 1.0);  // 1m clearance everywhere
  DistanceGrid grid(100, 100, 0.1, data);

  auto fp = PolygonFootprint::rectangle(0.2, 0.2, 2);

  // With 0.5m safety margin, effective clearance = 1.0 - 0.5 = 0.5.
  FootprintGridChecker small_margin(&grid, fp, 0.5);
  EXPECT_TRUE(small_margin.is_valid(Eigen::Vector3d(5.0, 5.0, 0.0)));

  // With 1.5m safety margin, effective clearance = 1.0 - 1.5 = -0.5.
  FootprintGridChecker big_margin(&grid, fp, 1.5);
  EXPECT_FALSE(big_margin.is_valid(Eigen::Vector3d(5.0, 5.0, 0.0)));
}

TEST(FootprintGridChecker, SDFCallable) {
  std::vector<double> data(100 * 100, 3.0);
  DistanceGrid grid(100, 100, 0.1, data);

  auto fp = PolygonFootprint::rectangle(0.2, 0.2, 2);
  FootprintGridChecker checker(&grid, fp, 0.5);

  Eigen::Vector3d q(5.0, 5.0, 0.0);
  double sdf = checker(q);
  // Early-out returns center_dist - (bounding_radius + slack) - safety_margin.
  const double expected = 3.0 - fp.bounding_radius() - grid.lipschitz_slack() - 0.5;
  EXPECT_NEAR(sdf, expected, 1e-12);
  EXPECT_GT(sdf, 0.0);
}

namespace {

// A distance transform of lethal nodes. Every node holds its distance to the nearest
// one, and node (c, r) sits at world (c h, r h).
DistanceGrid distance_transform(const int w, const int h, const double res,
                                const std::vector<Eigen::Vector2i>& lethal) {
  std::vector<double> data(static_cast<std::size_t>(w) * h);
  for (int r = 0; r < h; ++r) {
    for (int c = 0; c < w; ++c) {
      double best = std::numeric_limits<double>::infinity();
      for (const auto& l : lethal) best = std::min(best, std::hypot(c - l[0], r - l[1]) * res);
      data[static_cast<std::size_t>(r) * w + c] = best;
    }
  }
  return DistanceGrid(w, h, res, std::move(data));
}

// The signed transform of an occupancy grid. Every node holds its distance to the nearest
// occupied node minus its distance to the nearest free one.
DistanceGrid signed_transform(const int w, const int h, const double res,
                              const std::function<bool(int, int)>& occupied) {
  std::vector<double> data(static_cast<std::size_t>(w) * h);
  for (int r = 0; r < h; ++r) {
    for (int c = 0; c < w; ++c) {
      double to_occ = std::numeric_limits<double>::infinity();
      double to_free = std::numeric_limits<double>::infinity();
      for (int rr = 0; rr < h; ++rr) {
        for (int cc = 0; cc < w; ++cc) {
          const double d = std::hypot(c - cc, r - rr) * res;
          if (occupied(cc, rr)) {
            to_occ = std::min(to_occ, d);
          } else {
            to_free = std::min(to_free, d);
          }
        }
      }
      data[static_cast<std::size_t>(r) * w + c] = to_occ - to_free;
    }
  }
  return DistanceGrid(w, h, res, std::move(data));
}

// Two blocks and a bar of occupied cells.
bool blocks(const int c, const int r) {
  return (c >= 18 && c < 24 && r >= 18 && r < 26) || (c >= 45 && c < 52 && r >= 40 && r < 44) ||
         (c >= 30 && c < 33 && r >= 55 && r < 70);
}

// Clearance with every perimeter sample read, the value the shortcuts must bound.
double exact_clearance(const DistanceGrid& grid, const PolygonFootprint& fp, const double margin,
                       const Eigen::Vector3d& q) {
  double best = std::numeric_limits<double>::infinity();
  const double c = std::cos(q[2]), s = std::sin(q[2]);
  for (int i = 0; i < fp.sample_count_raw(); ++i) {
    const double bx = fp.body_x(i), by = fp.body_y(i);
    best = std::min(best, grid.distance_at(c * bx - s * by + q[0], s * bx + c * by + q[1]));
  }
  return best - margin;
}

}  // namespace

// The bilinear field of a distance transform is sqrt(2)-Lipschitz, and stays
// within its slack of 1-Lipschitz.
TEST(DistanceGrid, InterpolatedFieldRespectsItsLipschitzSlack) {
  constexpr double h = 0.1;
  const auto grid = distance_transform(30, 30, h, {{10, 10}, {18, 22}, {24, 7}});
  EXPECT_DOUBLE_EQ(grid.lipschitz_slack(), std::numbers::sqrt2 * h);
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> u(0.0, 29.0 * h);
  for (int k = 0; k < 20000; ++k) {
    const Eigen::Vector2d p(u(rng), u(rng)), q(u(rng), u(rng));
    const double change = std::abs(grid.distance_at(p[0], p[1]) - grid.distance_at(q[0], q[1]));
    EXPECT_LE(change, (p - q).norm() + grid.lipschitz_slack() + 1e-12);
  }
  // Next to a lethal node, along the diagonal, the field is steeper than 1.
  const double eps = 1e-4 * h;
  const double slope = grid.distance_at(h * 10 + eps, h * 10 + eps) / (std::numbers::sqrt2 * eps);
  EXPECT_GT(slope, 1.4);
}

// A square whose corner sample sits on a lethal node. In the interpolated field its
// center lies farther from the node than the bounding radius. A shortcut that treats the
// field as 1-Lipschitz calls this pose clear.
TEST(FootprintGridChecker, CenterShortcutIsSoundForTheInterpolatedField) {
  constexpr double h = 0.1;
  const auto grid = distance_transform(30, 30, h, {{10, 10}});
  const auto fp = PolygonFootprint::rectangle(0.5 * h, 0.5 * h, 2);
  const double margin = 0.01 * h;  // makes the sample on the node a clear collision
  const FootprintGridChecker checker(&grid, fp, margin);
  const Eigen::Vector3d q(10.5 * h, 10.5 * h, 0.0);
  ASSERT_GT(grid.distance_at(q[0], q[1]), fp.bounding_radius() + margin);
  ASSERT_NEAR(exact_clearance(grid, fp, margin, q), -margin, 1e-12);
  EXPECT_FALSE(checker.is_valid(q));
  EXPECT_LE(checker(q), 0.0);
  EXPECT_LE(checker.min_distance_capped(q, 0.0), 0.0);
  EXPECT_LE(checker.min_distance_capped(q, 1.0), 0.0);
}

// An edge runs diagonally into a lethal node, with samples half a cell diagonal apart.
// The field falls faster than 1 over the last step. A skip that treats the field as
// 1-Lipschitz jumps over the sample on the node.
TEST(FootprintGridChecker, CappedSkipIsSoundForTheInterpolatedField) {
  constexpr double h = 0.1;
  const auto grid = distance_transform(40, 40, h, {{20, 20}});
  const double a = std::numbers::sqrt2 * h;  // four samples per edge, half a diagonal apart
  const auto fp = PolygonFootprint::rectangle(a, 0.5 * h, 4);
  const double theta = -0.75 * std::numbers::pi;
  const double c = std::cos(theta), s = std::sin(theta);
  // Body vertex (-a, -b), the first sample, lands at the node plus half a cell diagonal.
  const Eigen::Vector2d first(20.5 * h, 20.5 * h);
  const Eigen::Vector2d offset(c * -a - s * -0.5 * h, s * -a + c * -0.5 * h);
  const Eigen::Vector3d q(first[0] - offset[0], first[1] - offset[1], theta);
  const double margin = 0.01 * h;  // makes the sample on the node a clear collision
  const FootprintGridChecker checker(&grid, fp, margin);
  ASSERT_LT(grid.distance_at(q[0], q[1]), fp.bounding_radius());  // no center shortcut
  ASSERT_NEAR(exact_clearance(grid, fp, margin, q), -margin, 1e-12);
  EXPECT_LE(checker.min_distance_capped(q, 0.0), 0.0);
  EXPECT_LE(checker.min_distance_capped(q, 1.0), 0.0);
  EXPECT_FALSE(checker.is_valid(q));
}

// Over random poses every value bounds the exact clearance. A clear value bounds it
// from below with the right sign, a colliding value bounds it from above, and the capped
// value equals operator() below the cap.
TEST(FootprintGridChecker, ValuesBoundTheExactClearance) {
  constexpr double h = 0.05;
  const auto unsigned_grid =
      distance_transform(80, 80, h, {{20, 20}, {21, 21}, {50, 30}, {60, 60}, {35, 55}});
  const auto signed_grid = signed_transform(80, 80, h, blocks);
  const auto fp = PolygonFootprint::rectangle(0.25, 0.15, 6);
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> pos(0.6, 3.4), ang(-std::numbers::pi, std::numbers::pi);
  for (const DistanceGrid* grid : {&unsigned_grid, &signed_grid}) {
    for (const double margin : {0.0, 0.05}) {
      const FootprintGridChecker checker(grid, fp, margin);
      for (int k = 0; k < 4000; ++k) {
        const Eigen::Vector3d q(pos(rng), pos(rng), ang(rng));
        const double exact = exact_clearance(*grid, fp, margin, q);
        if (std::abs(exact) < 1e-9) continue;
        const double v = checker(q);
        EXPECT_EQ(v > 0.0, exact > 0.0) << "pose " << q.transpose();
        if (v > 0.0) {
          EXPECT_LE(v, exact + 1e-12);
        } else {
          EXPECT_GE(v, exact - 1e-12);
        }
        for (const double cap : {0.0, 0.1, 0.5}) {
          const double capped = checker.min_distance_capped(q, cap);
          EXPECT_EQ(capped > 0.0, exact > 0.0) << "pose " << q.transpose() << " cap " << cap;
          if (capped > 0.0) EXPECT_LE(capped, exact + 1e-12);
          if (v > 0.0 && v < cap) EXPECT_NEAR(capped, v, 1e-12);
          if (v >= cap) EXPECT_GE(capped, cap - 1e-12);
        }
      }
    }
  }
}

TEST(FootprintGridChecker, Accessors) {
  std::vector<double> data(100 * 100, 3.0);
  DistanceGrid grid(100, 100, 0.1, data);

  const auto fp = PolygonFootprint::rectangle(0.5, 0.3, 4);
  const FootprintGridChecker checker(&grid, fp, 0.25);

  EXPECT_EQ(checker.grid(), &grid);
  EXPECT_EQ(checker.footprint().sample_count(), fp.sample_count());
  EXPECT_DOUBLE_EQ(checker.safety_margin(), 0.25);
}

// A grid with a negative node is a signed transform. Its slack is twice the unsigned
// one, and the field stays within it. After reset() the slack follows the new values.
TEST(DistanceGrid, SignedTransformDoublesTheSlack) {
  constexpr double h = 0.1;
  auto grid = signed_transform(40, 40, h, [](const int c, const int r) {
    return (c >= 12 && c < 20 && r >= 10 && r < 16) || (c >= 25 && c < 27 && r >= 5 && r < 35);
  });
  EXPECT_DOUBLE_EQ(grid.lipschitz_slack(), 2.0 * std::numbers::sqrt2 * h);
  std::mt19937 rng(17);
  std::uniform_real_distribution<double> u(0.0, 39.0 * h);
  for (int k = 0; k < 20000; ++k) {
    const Eigen::Vector2d p(u(rng), u(rng)), q(u(rng), u(rng));
    const double change = std::abs(grid.distance_at(p[0], p[1]) - grid.distance_at(q[0], q[1]));
    EXPECT_LE(change, (p - q).norm() + grid.lipschitz_slack() + 1e-12);
  }

  auto& values = grid.reset(3, 3, h);
  std::fill(values.begin(), values.end(), 1.0);
  EXPECT_DOUBLE_EQ(grid.lipschitz_slack(), std::numbers::sqrt2 * h);
  const DistanceGrid copy = grid;
  EXPECT_DOUBLE_EQ(copy.lipschitz_slack(), std::numbers::sqrt2 * h);
}
