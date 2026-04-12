/// @file test_collision.cpp
/// @brief Tests for geodex::collision module.

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <geodex/collision/circle_sdf.hpp>
#include <geodex/collision/distance_grid.hpp>
#include <geodex/collision/footprint_grid_checker.hpp>
#include <geodex/collision/polygon_footprint.hpp>
#include <geodex/collision/rectangle_sdf.hpp>
#include <geodex/utils/math.hpp>
#include <numbers>

using namespace geodex::collision;

// ---------------------------------------------------------------------------
// Fast exp
// ---------------------------------------------------------------------------

TEST(FastExp, ApproximatesStdExp) {
  // Schraudolph's trick with bias correction (c=60801, scaled by 2^32 for
  // the 64-bit adaptation) gives ~4% max relative error — the inherent limit
  // of a linear chord approximation to 2^f on each unit interval.
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
  // Midpoint between two circles: distance to each = 1.5.
  Eigen::Vector3d q(2.5, 0.0, 0.0);
  double d = sdf(q);
  EXPECT_GT(d, 0.0);
  // smooth-min should be slightly less than hard min (1.5).
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
  // Use a point within bounding sphere (skip_dist=1.0, diag=1.414, so br=2.414).
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
  // Use a point within bounding sphere so SDF is computed, not clipped.
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

  // Bilinear midpoint: (0.5, 0.5) = avg(0, 1, 1, 2) = 1.0.
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

  // After 90 deg rotation: (x,y) → (-y,x).
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
  // Uniform distance field: everywhere 5.0m clear.
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

  // Robot right at the wall edge — footprint extends into wall.
  Eigen::Vector3d q(5.0, 5.0, 0.0);
  EXPECT_FALSE(checker.is_valid(q));

  // Robot well away from wall — should be clear.
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
  // Early-out returns center_dist - bounding_radius - safety_margin.
  // = 3.0 - sqrt(0.2^2+0.2^2) - 0.5 ~ 2.217
  const double expected = 3.0 - fp.bounding_radius() - 0.5;
  EXPECT_NEAR(sdf, expected, 0.01);
  EXPECT_GT(sdf, 0.0);
}
