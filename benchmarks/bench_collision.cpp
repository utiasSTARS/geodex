/// @file bench_collision.cpp
/// @brief Micro-benchmarks for collision primitives and fast math utilities.

#include <benchmark/benchmark.h>

#include <Eigen/Core>
#include <cmath>
#include <geodex/collision/collision.hpp>
#include <geodex/utils/math.hpp>
#include <vector>

using namespace geodex::collision;

// ---------------------------------------------------------------------------
// Helper: generate N circles spaced along the x-axis
// ---------------------------------------------------------------------------

static std::vector<CircleSDF> make_circles(const int n) {
  std::vector<CircleSDF> cs;
  cs.reserve(n);
  for (int i = 0; i < n; ++i) {
    cs.emplace_back(static_cast<double>(i) * 2.0, 0.0, 0.5);
  }
  return cs;
}

// ---------------------------------------------------------------------------
// Helper: generate N oriented rectangles spaced along the x-axis
// ---------------------------------------------------------------------------

static std::vector<RectObstacle> make_rects(const int n) {
  std::vector<RectObstacle> rs;
  rs.reserve(n);
  for (int i = 0; i < n; ++i) {
    rs.push_back({static_cast<double>(i) * 3.0, 0.0, 0.1 * i, 1.0, 0.5});
  }
  return rs;
}

// ---------------------------------------------------------------------------
// Helper: synthetic 100x100 distance grid (5m x 5m at 0.05m resolution)
// ---------------------------------------------------------------------------

static DistanceGrid make_grid() {
  const int w = 100, h = 100;
  std::vector<double> data(static_cast<size_t>(w) * h);
  for (int r = 0; r < h; ++r) {
    for (int c = 0; c < w; ++c) {
      // Distance to nearest border (simple approximation).
      data[static_cast<size_t>(r) * w + c] =
          0.05 * std::min({r, c, h - 1 - r, w - 1 - c});
    }
  }
  return DistanceGrid(w, h, 0.05, std::move(data));
}

static const DistanceGrid grid = make_grid();

// ---------------------------------------------------------------------------
// Fixed test data
// ---------------------------------------------------------------------------

static const Eigen::Vector3d query_near(2.5, 2.5, 0.3);    // near grid center
static const Eigen::Vector3d query_border(0.1, 0.1, 0.0);  // near grid border (obstacles)
static const Eigen::Vector2d query_2d(2.5, 2.5);

// ===========================================================================
// Circle SDF benchmarks
// ===========================================================================

static void BM_CircleSDF(benchmark::State& state) {
  CircleSDF sdf(5.0, 5.0, 1.0);
  for (auto _ : state) {
    benchmark::DoNotOptimize(sdf(query_2d));
  }
}
BENCHMARK(BM_CircleSDF);

static void BM_CircleSmoothSDF(benchmark::State& state) {
  const int n = static_cast<int>(state.range(0));
  CircleSmoothSDF sdf(make_circles(n));
  for (auto _ : state) {
    benchmark::DoNotOptimize(sdf(query_2d));
  }
}
BENCHMARK(BM_CircleSmoothSDF)->Arg(5)->Arg(10)->Arg(20)->Arg(50);

static void BM_CircleSmoothSDF_IsFree(benchmark::State& state) {
  const int n = static_cast<int>(state.range(0));
  CircleSmoothSDF sdf(make_circles(n));
  // Query point that is free (far from all circles).
  const Eigen::Vector2d free_pt(100.0, 100.0);
  for (auto _ : state) {
    benchmark::DoNotOptimize(sdf.is_free(free_pt));
  }
}
BENCHMARK(BM_CircleSmoothSDF_IsFree)->Arg(5)->Arg(10)->Arg(20)->Arg(50);

// ===========================================================================
// Rectangle SDF benchmarks
// ===========================================================================

static void BM_RectSmoothSDF(benchmark::State& state) {
  const int n = static_cast<int>(state.range(0));
  RectSmoothSDF sdf(make_rects(n));
  for (auto _ : state) {
    benchmark::DoNotOptimize(sdf(query_2d));
  }
}
BENCHMARK(BM_RectSmoothSDF)->Arg(5)->Arg(10)->Arg(20)->Arg(50);

// ===========================================================================
// Distance grid benchmarks
// ===========================================================================

static void BM_DistanceGrid_Single(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(grid.distance_at(2.5, 2.5));
  }
}
BENCHMARK(BM_DistanceGrid_Single);

static void BM_DistanceGrid_Batch(benchmark::State& state) {
  const int n = static_cast<int>(state.range(0));
  std::vector<double> xs(n), ys(n), out(n);
  for (int i = 0; i < n; ++i) {
    // Spread queries across the grid.
    xs[i] = 0.05 * (i % 100);
    ys[i] = 0.05 * (i / 100 % 100);
  }
  for (auto _ : state) {
    grid.distance_at_batch(xs.data(), ys.data(), out.data(), n);
    benchmark::DoNotOptimize(out.data());
  }
  state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_DistanceGrid_Batch)->Arg(16)->Arg(32)->Arg(64)->Arg(128);

// ===========================================================================
// Polygon footprint benchmarks
// ===========================================================================

static void BM_PolygonFootprint_Transform(benchmark::State& state) {
  const int spe = static_cast<int>(state.range(0));
  PolygonFootprint fp = PolygonFootprint::rectangle(0.5, 0.3, spe);
  const int np = fp.sample_count();
  std::vector<double> wx(np), wy(np);
  for (auto _ : state) {
    fp.transform(2.5, 2.5, 0.3, wx.data(), wy.data());
    benchmark::DoNotOptimize(wx.data());
  }
}
BENCHMARK(BM_PolygonFootprint_Transform)->Arg(2)->Arg(4)->Arg(8);

// ===========================================================================
// Footprint grid checker benchmarks
// ===========================================================================

static void BM_FootprintGridChecker_EarlyOut(benchmark::State& state) {
  PolygonFootprint fp = PolygonFootprint::rectangle(0.1, 0.05, 4);
  FootprintGridChecker checker(&grid, std::move(fp));
  // Query at grid center — far from borders, triggers bounding sphere early-out.
  for (auto _ : state) {
    benchmark::DoNotOptimize(checker(query_near));
  }
}
BENCHMARK(BM_FootprintGridChecker_EarlyOut);

static void BM_FootprintGridChecker_FullCheck(benchmark::State& state) {
  PolygonFootprint fp = PolygonFootprint::rectangle(0.1, 0.05, 4);
  FootprintGridChecker checker(&grid, std::move(fp));
  // Query near border — center distance close to bounding radius, forces full check.
  for (auto _ : state) {
    benchmark::DoNotOptimize(checker(query_border));
  }
}
BENCHMARK(BM_FootprintGridChecker_FullCheck);

// ===========================================================================
// Fast math utility benchmarks
// ===========================================================================

static void BM_FastExp(benchmark::State& state) {
  double x = -3.7;
  for (auto _ : state) {
    benchmark::DoNotOptimize(geodex::utils::fast_exp(x));
    x += 1e-9;  // prevent constant folding
  }
}
BENCHMARK(BM_FastExp);

static void BM_StdExp(benchmark::State& state) {
  double x = -3.7;
  for (auto _ : state) {
    benchmark::DoNotOptimize(std::exp(x));
    x += 1e-9;
  }
}
BENCHMARK(BM_StdExp);

static void BM_Sincos(benchmark::State& state) {
  double angle = 1.23;
  double s, c;
  for (auto _ : state) {
    geodex::utils::sincos(angle, &s, &c);
    benchmark::DoNotOptimize(s);
    benchmark::DoNotOptimize(c);
    angle += 1e-9;
  }
}
BENCHMARK(BM_Sincos);

static void BM_SeparateSinCos(benchmark::State& state) {
  double angle = 1.23;
  for (auto _ : state) {
    benchmark::DoNotOptimize(std::sin(angle));
    benchmark::DoNotOptimize(std::cos(angle));
    angle += 1e-9;
  }
}
BENCHMARK(BM_SeparateSinCos);
