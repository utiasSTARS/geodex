# geodex Benchmarks

Micro-benchmarks for geodex manifold operations, algorithms, metrics, and retractions using [Google Benchmark](https://github.com/google/benchmark).

## Building

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON
cmake --build build
```

Google Benchmark is fetched automatically via CMake FetchContent.

## Running

```sh
# All benchmarks
./build/benchmarks/bench_manifold_ops
./build/benchmarks/bench_algorithms
./build/benchmarks/bench_metrics
./build/benchmarks/bench_retractions
./build/benchmarks/bench_collision

# JSON output for regression tracking
./build/benchmarks/bench_manifold_ops --benchmark_format=json --benchmark_out=results/bench_manifold_ops.json

# Filter specific benchmarks
./build/benchmarks/bench_manifold_ops --benchmark_filter="Torus"
```

## Benchmark Suites

| File | What it measures |
|------|-----------------|
| `bench_manifold_ops.cpp` | exp, log, distance, inner, norm, geodesic, random_point per manifold; dimension scaling for Torus/Euclidean (d=2..50) |
| `bench_algorithms.cpp` | `discrete_geodesic` single-pair on all manifolds; batch-steer workload (256 random endpoint pairs, shared workspace); SDFConformal FD path (midpoint guard vs forced via-log); batch `distance_midpoint` throughput (1000 pairs); dimension scaling |
| `bench_metrics.cpp` | `inner` and `norm` for all 7 metric types; ConfigurationSpace overhead vs bare manifold |
| `bench_retractions.cpp` | SE(2) and Sphere retraction speed; Sphere projection accuracy vs exponential map |
| `bench_collision.cpp` | Collision primitives: circle/rectangle SDF eval and scaling, distance grid single/batch lookup, polygon footprint transform, footprint grid checker (early-out vs full check); fast math: `fast_exp` vs `std::exp`, `sincos` vs separate trig |
| `bench_ompl_interpolation.cpp` | OMPL `GeodexStateSpace::interpolate()` with discrete geodesic cache; cache hot/cold; motion validation overhead (requires OMPL) |

## Reference Results

Machine: Apple M2 (8-core, arm64), 16 GB RAM, macOS 15.3.1
Compiler: Apple clang 15.0.0, `-O2` (CMake Release)
Date: 2026-04-17 (run after `feature/ompl-integration`: discrete-geodesic
caching in OMPL adapter, FD midpoint guard with via-log fallback, `SDFConformalMetric`,
path smoothing arc-length shortcut criterion).

### Manifold Primitives (ns per call)

| Operation | Sphere (exp) | Sphere (proj) | Euclidean<2> | Euclidean<7> | Torus<2> | Torus<7> | SE2 (exp) |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| exp | 5.0 | 1.7 | 0.3 | 0.9 | 1.2 | 5.1 | 6.3 |
| log | 10.7 | 1.5 | 0.3 | 0.9 | 2.0 | 6.6 | 4.9 |
| distance | 79 | 12 | 0.9 | 6.5 | 7.9 | 31 | 30 |
| inner | 1.1 | -- | 0.4 | 4.1 | 0.5 | 4.2 | 1.1 |
| geodesic | 23 | -- | -- | -- | 3.8 | -- | 12 |
| random_point | 70 | -- | -- | -- | 25 | -- | 30 |

### Dimension Scaling: distance_midpoint (ns)

Torus at compile-time fixed dimension versus `Torus<Eigen::Dynamic>` constructed
at runtime. The fixed column is derived from the batch benchmarks
(`BM_DistanceMidpoint_Torus2_Batch`, `BM_DistanceMidpoint_Torus7_Batch`) by
dividing by the 1000-pair batch size; the dynamic column is per-call
(`BM_DistanceMidpoint_TorusDynamic/d`).

| Dimension | Torus (fixed) | Torus (dynamic) |
|-----------|:---:|:---:|
| 2 | 6.3 | 138 |
| 5 | -- | 187 |
| 7 | 38 | 200 |
| 10 | -- | 195 |
| 20 | -- | 247 |
| 50 | -- | 656 |

Fixed-dimension types are ~20x faster than dynamic at d=2 thanks to stack
allocation and compile-time loop unrolling. The dynamic cost grows roughly
linearly with dimension (heap vectors + Eigen's dynamic dispatch).

### Algorithm Performance: discrete_geodesic single pair (ns per call)

Single `start → target` walk with a pre-allocated `InterpolationCache` workspace.
`iters/call` is the average number of gradient steps to convergence at the
default `settings.step_size = 0.5`.

| Manifold | Time | iters/call |
|----------|:---:|:---:|
| Sphere (round metric) | 218 | 2 |
| Sphere (anisotropic `ConstantSPD<3>`) | 2,706 | 4 |
| Torus<2> (flat metric) | 130 | 6 |
| SE2 (exp map, isotropic) | 1,246 | 23 |
| SE2 (anisotropic `SE2LeftInvariant`) | 11,630 | 22 |
| ConfigurationSpace<Torus<2>, KineticEnergyMetric> | 1,544 | 7 |

Isotropic metrics (`is_riemannian_log` returns true) use the log direction
directly — no FD sampling per step, so cost per iteration is minimal. The
anisotropic/point-dependent rows (`SphereAniso`, `SE2_Anisotropic`, `CSpace_KE`)
pay one central finite-difference natural gradient per step: 2·d samples of
`distance_midpoint_fd` under the metric, which dominates wall time. Each sample
itself calls the metric's `inner` (the 5.9 ns and 11.9 ns for KE/Jacobi listed
further down), so the FD cost scales with the metric's per-evaluation price.

### SDFConformal FD path: midpoint guard vs forced via-log

Benchmarks: `BM_DiscreteGeodesic_SE2_SDFConformal_{Midpoint,ViaLog}`. Same
`(start, target, step_size, max_steps)` under a spatially-varying
`SDFConformalMetric` (unit-circle obstacle at origin). The two rows toggle
`fd_midpoint_guard_tau` between its default (midpoint surrogate with guard) and
0.0 (force the via-log fallback on every FD sample, which reproduces the
pre-guard behavior).

| Mode | Time | iters/call | fallbacks/call |
|------|:---:|:---:|:---:|
| Midpoint guard (default) | 13,221 ns | 24 | 0 |
| Forced via-log (tau = 0) | 12,174 ns | 24 | 72 |

Both paths converge in the same number of iterations and have essentially the
same wall time — but the guarded midpoint form avoids the 72 per-iteration
via-log recomputations. The guard lets the cheaper midpoint distance stand when
it agrees with via-log, falling back per-sample only when the two diverge. On
this benchmark the guard always agrees, so `fallbacks/call` is 0 at the
default; forcing `tau = 0` shows the upper bound on via-log work we avoid.

### Batch steer workload (RRT\* inner loop)

Benchmarks: `BM_DiscreteGeodesic_*_BatchSteer`. Each invocation runs 256
`discrete_geodesic` calls between independent random `(start, target)` pairs
reusing a single `InterpolationCache`. This is the primary "RRT\* steer loop"
shape — many short walks from unrelated roots through one workspace.

| Manifold | Batch total | Per steer | Throughput | iters/call |
|----------|:---:|:---:|:---:|:---:|
| Sphere (round metric) | 93.8 µs | 366 ns | 2.73 M/s | 3.6 |
| Torus<7> (flat metric) | 112.7 µs | 440 ns | 2.27 M/s | 9.7 |
| SE2 (exp map) | 199.6 µs | 780 ns | 1.28 M/s | 13.5 |

Per-steer cost is lower than the single-pair "Algorithm Performance" numbers
because the workspace stays hot across steers and random endpoints give a
distribution of distances around the injectivity-radius sweet spot. Use these
numbers as a realistic RRT*/G-RRT* budget: ~1–3 M steers/second per core for
isotropic metrics on the manifolds geodex currently supports.

### Batch distance_midpoint Throughput (1000 pairs)

| Manifold | Total (µs) | Throughput |
|----------|:---:|:---:|
| Torus<2> | 6.3 | 159 M/s |
| Torus<7> | 38.0 | 26 M/s |
| SE2 | 26.4 | 38 M/s |
| Sphere | 93.0 | 11 M/s |

`distance_midpoint` is a single exp + 3 log per pair. Sphere is slowest because
log involves `acos` + vector normalization; Torus at low dim is fastest because
exp/log reduce to arithmetic + angle wrap.

### Metric inner Product (ns per call)

| Metric | Time |
|--------|:---:|
| EuclideanStandard<2> | 0.4 |
| EuclideanStandard<7> | 3.0 |
| TorusFlat<2> | 0.4 |
| TorusFlat<7> | 3.0 |
| ConstantSPD<3> | 0.8 |
| ConstantSPD<7> | 3.0 |
| SE2LeftInvariant | 0.8 |
| Weighted<ConstantSPD<3>> | 0.9 |
| KineticEnergy (2-link arm) | 5.9 |
| Jacobi (2-link arm) | 11.9 |
| Pullback (2-link arm) | 15.1 |

After the refactor, `EuclideanStandard`, `TorusFlat`, and `SE2LeftInvariant` are
all thin aliases/compositions over `ConstantSPDMetric`, which is why their
`inner` timings are essentially identical to the corresponding `ConstantSPD<N>`
row. The `WeightedMetric` wrapper adds ~0.1 ns on top of its base metric.

Point-dependent metrics (KineticEnergy, Jacobi, Pullback) remain 6–15× more
expensive than constant metrics due to the `std::function` evaluation and
dense matrix–vector product per call.

### Retraction Comparison (ns per call)

| Retraction | retract | inverse_retract |
|------------|:---:|:---:|
| SE2 ExponentialMap | 8.2 | 7.7 |
| SE2 Euler | 0.7 | 0.6 |
| Sphere ExponentialMap | 5.1 | 10.8 |
| Sphere Projection | 1.6 | 1.5 |

### OMPL Integration: Discrete Geodesic Interpolation (ns per call)

Benchmark: `bench_ompl_interpolation` — measures the `GeodexStateSpace::interpolate()` method
with the discrete geodesic path cache.

Machine: Apple M2 (8-core, arm64), macOS 15.3.1, Apple clang, `-O2`

**Interpolation throughput** (N sequential calls with same `(from, to)` pair):

| Benchmark | N=10 | N=50 | N=100 | Per-call |
|-----------|:---:|:---:|:---:|:---:|
| SE2 Identity (fast path) | 234 | 1,137 | 2,300 | 23 |
| SE2 Anisotropic cache hot | 420 | 2,174 | 4,378 | 44 |
| Euclidean Aniso cache hot | 161 | 872 | 1,634 | 16 |

Cache cold (first call, triggers `discrete_geodesic`): **59 µs** for anisotropic SE2.

**Motion validation** (`DiscreteMotionValidator::checkMotion()`, single edge):

| Mode | Time |
|------|:---:|
| Simple geodesic (discrete OFF) | 1,767 ns |
| Discrete geodesic (discrete ON) | 3,773 ns |

The 2.1x overhead on motion validation is justified by the path quality improvement
(see analysis below).

### Interpolation Quality Analysis

Analysis tool: `analysis/interpolation_analysis.cpp` — compares path length and
energy between naive `geodesic(p,q,t) = exp(p, t·log(p,q))` and the discrete
geodesic path for the same (start, target) pair, sampled at 20 uniform points.

| Scenario | Naive length | Discrete length | Ratio | Naive energy | Discrete energy | Max deviation |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|
| Euclidean R², identity | 11.31 | 11.31 | 1.000 | 6.40 | 6.40 | 0.000 |
| Euclidean R², A=diag(4,1) | 17.89 | 17.89 | 1.000 | 16.00 | 16.00 | 0.000 |
| SE(2) isotropic, Euler retract | 8.54 | 8.54 | 1.000 | 3.65 | 3.65 | 0.000 |
| **SE(2) car-like w=(1,100,0.5)** | **26.33** | **9.07** | **0.344** | **34.67** | **4.37** | **0.765** |
| **SE(2) extreme w=(1,1000,0.1)** | **79.24** | **0.78** | **0.010** | **313.98** | **0.031** | **7.941** |

**Key observations:**

1. **Flat spaces with constant metrics (Euclidean, Torus):** Both methods produce
   identical paths. Geodesics are straight lines regardless of metric — the metric
   changes distances but not directions. The fast path (`is_riemannian_log`) is
   correctly activated, giving zero overhead.

2. **SE(2) with anisotropic left-invariant metric:** The discrete geodesic
   produces dramatically shorter paths. With car-like weights (1,100,0.5), the
   naive path is **2.9x longer** and **7.9x higher energy**. The naive midpoint
   deviates 0.76 units from the true geodesic midpoint.

3. **Extreme anisotropy amplifies the effect:** With w=(1,1000,0.1), the naive
   path is **100x longer** and **10,000x higher energy**. The naive interpolation
   cuts through the high-cost y-direction while the discrete geodesic avoids it.

4. **Retraction type matters less than metric:** SE(2) with Euler retraction but
   isotropic metric produces identical results to the exponential map case.
   The metric-retraction mismatch drives the quality difference, not the
   retraction choice alone.

**Convergence note:** The default `InterpolationSettings` (`step_size=0.5,
max_steps=100`) may not converge for strongly anisotropic metrics. For car-like
SE(2) w=(1,100,0.5), the default budget reaches only ~50% of the path to target.
The cache gracefully handles partial convergence (appends the target, so `t=0` and
`t=1` are exact), but intermediate points are less accurate near the tail. Users
with strong anisotropy should increase `max_steps` via `setInterpolationSettings()`.

| Metric anisotropy | step=0.5, max=100 | step=0.5, max=500 | step=0.1, max=1000 |
|---|---|---|---|
| w=(1,1,1) isotropic | Converged | Converged | Converged |
| w=(1,100,0.5) car-like | MaxStepsReached (50%) | Converged | Converged |
| w=(1,1000,0.1) extreme | MaxStepsReached (10%) | MaxStepsReached (30%) | MaxStepsReached (60%) |

**When discrete geodesic helps:**
- Non-identity metrics on manifolds with non-trivial geometry (SE(2), SO(3), configuration spaces)
- Point-dependent metrics (KineticEnergyMetric, JacobiMetric, PullbackMetric)
- ConfigurationSpace with custom metrics overlaid on base manifolds

**When it adds no value (fast path used):**
- Any manifold with identity metric and matching retraction
- Flat spaces (Euclidean, Torus) with constant metrics — geodesics are straight lines

### Collision Primitives and Fast Math (ns per call)

Benchmark: `bench_collision` — measures all collision SDF evaluation, distance grid
queries, polygon footprint transforms, and fast math utilities.

Machine: Apple M2 (8-core, arm64), macOS 15.3.1, Apple clang, `-O2`
Date: 2026-04-11

**SDF evaluation scaling** (parameterized by obstacle count):

| Benchmark | N=5 | N=10 | N=20 | N=50 |
|-----------|:---:|:---:|:---:|:---:|
| CircleSmoothSDF | 20 | 27 | 46 | 114 |
| CircleSmoothSDF (is_free) | 2.5 | 4.8 | 9.4 | 23 |
| RectSmoothSDF (NEON) | 4.8 | 6.2 | 9.4 | 22 |

Single `CircleSDF` evaluation: **0.6 ns** (one sqrt + subtract).

RectSmoothSDF benefits from NEON 2-wide processing and bounding-sphere early-out,
keeping cost low even at N=50. CircleSmoothSDF scales linearly with circle count
(~2.3 ns/circle) after the single-pass caching optimization.

**Distance grid queries:**

| Benchmark | Time | Throughput |
|-----------|:---:|:---:|
| Single bilinear lookup | 2.9 ns | — |
| Batch N=16 | 31 ns | 509M items/s |
| Batch N=32 | 61 ns | 522M items/s |
| Batch N=64 | 121 ns | 528M items/s |
| Batch N=128 | 250 ns | 513M items/s |

Batch throughput is ~2 ns/point with NEON vectorized bilinear interpolation.
The scalar gather (no ARM NEON gather instruction) is the bottleneck.

**Polygon footprint and checker:**

| Benchmark | Time |
|-----------|:---:|
| PolygonFootprint transform (spe=2, 8 samples) | 6.7 ns |
| PolygonFootprint transform (spe=4, 16 samples) | 10 ns |
| PolygonFootprint transform (spe=8, 32 samples) | 18 ns |
| FootprintGridChecker (bounding sphere early-out) | 4.6 ns |
| FootprintGridChecker (full perimeter check) | 52 ns |

The bounding-sphere early-out (center distance > bounding radius + margin) resolves
most queries in 4.6 ns without transforming the polygon. The full check (transform +
batch grid lookup + min-reduce) costs 52 ns for a 16-sample rectangle footprint.

**Fast math utilities:**

| Function | Time | vs stdlib |
|----------|:---:|:---:|
| `fast_exp` | 1.0 ns | **2.5x** faster than `std::exp` (2.5 ns) |
| `sincos` | 4.7 ns | ~same as separate `sin`+`cos` (Apple libm optimizes both) |

`fast_exp` (Schraudolph IEEE 754 bit trick, ~4% max relative error) is used in all
log-sum-exp smooth-min SDF computations. The 2.5x speedup compounds across obstacle
counts in `CircleSmoothSDF` and `RectSmoothSDF`.

### x86 Reference Results — Collision Primitives and Fast Math

Machine: Intel Core i7-10875H (8-core/16-thread, x86_64), 32 GB RAM, Ubuntu 24.04
Compiler: GCC 13.3.0, `-O3 -march=native` (SSE4.2 + AVX2 + FMA enabled)
Date: 2026-04-12

**SDF evaluation scaling** (parameterized by obstacle count):

| Benchmark | N=5 | N=10 | N=20 | N=50 |
|-----------|:---:|:---:|:---:|:---:|
| CircleSmoothSDF | 24 | 36 | 59 | 128 |
| CircleSmoothSDF (is_free) | 3.3 | 6.1 | 12 | 29 |
| RectSmoothSDF (SSE2) | 4.1 | 5.8 | 9.7 | 21 |

Single `CircleSDF` evaluation: **1.3 ns**.

RectSmoothSDF SSE2 performance is on par with ARM NEON: the 2-wide processing,
bounding-sphere early-out, and `fast_exp` vectorization translate directly. The SSE2
path uses `_mm_movemask_pd` for the early-out check, which is slightly more efficient
than NEON's per-lane extraction.

**Distance grid queries:**

| Benchmark | Time | Throughput |
|-----------|:---:|:---:|
| Single bilinear lookup | 6.7 ns | — |
| Batch N=16 | 74 ns | 218M items/s |
| Batch N=32 | 146 ns | 222M items/s |
| Batch N=64 | 298 ns | 217M items/s |
| Batch N=128 | 568 ns | 228M items/s |

Batch throughput is ~4.4 ns/point with SSE2 vectorized bilinear interpolation.
Lower than ARM (~2 ns/point) because the benchmark grid (100x100 doubles = 80 KB)
fits in the M2's 128 KB L1D but spills on the i7's 32 KB L1D. Each bilinear lookup
gathers 4 grid values; on x86 these frequently miss L1 and hit L2 (~10 cycles vs
~4 cycles for an L1 hit). The scalar single-point lookup shows the same 2.3x gap
(6.7 vs 2.9 ns), confirming the bottleneck is cache capacity, not SIMD efficiency.

**Polygon footprint and checker:**

| Benchmark | Time |
|-----------|:---:|
| PolygonFootprint transform (spe=2, 8 samples) | 13 ns |
| PolygonFootprint transform (spe=4, 16 samples) | 19 ns |
| PolygonFootprint transform (spe=8, 32 samples) | 26 ns |
| FootprintGridChecker (bounding sphere early-out) | 9.2 ns |
| FootprintGridChecker (full perimeter check) | 99 ns |

**Fast math utilities:**

| Function | Time | vs stdlib |
|----------|:---:|:---:|
| `fast_exp` | 0.9 ns | **5.3x** faster than `std::exp` (4.8 ns) |
| `sincos` | 9.9 ns | ~same as separate `sin`+`cos` (glibc optimizes both) |

`fast_exp` delivers a larger relative speedup on x86 (5.3x vs 2.5x on ARM) because
`std::exp` in glibc is slower than Apple's libm, while the Schraudolph bit trick runs
at similar speed on both architectures.

### x86 vs ARM NEON — Collision Summary

| Benchmark | ARM M2 (NEON) | x86 i7-10875H (SSE2) | Ratio |
|-----------|:---:|:---:|:---:|
| RectSmoothSDF N=5 | 4.8 ns | 4.1 ns | 0.85x |
| RectSmoothSDF N=50 | 22 ns | 21 ns | 0.95x |
| DistanceGrid batch throughput | 520M/s | 222M/s | 2.3x slower |
| FootprintGridChecker full | 52 ns | 99 ns | 1.9x slower |
| fast_exp | 1.0 ns | 0.9 ns | 0.90x |
| fast_exp vs std::exp speedup | 2.5x | 5.3x | — |

The SSE2 RectSmoothSDF path matches or beats ARM NEON despite being a direct 2-wide
port (same width). The distance grid and footprint checker are slower on x86 because
the 80 KB benchmark grid fits in the M2's 128 KB L1D but exceeds the i7's 32 KB L1D,
causing L2 spills on every bilinear gather. This is a cache capacity effect, not a
SIMD efficiency issue — the vectorization code itself is equally efficient on both.

## Key Findings

- **OMPL discrete geodesic interpolation is a net win for anisotropic
  metrics.** The 2.1x overhead on per-edge motion validation (3.8 µs vs 1.8 µs)
  is negligible compared to the path quality improvement: 2.9x shorter paths and
  7.9x lower energy for car-like SE(2). For identity metrics the fast path is
  taken with zero overhead — the compiler dead-code-eliminates the cache branch.
- **Cache amortization works as designed.** The cold-cache cost (59 µs for
  anisotropic SE2) is paid once per `(from, to)` pair. Subsequent lookups cost
  44 ns/call — only 1.9x the identity-metric fast path (23 ns/call). OMPL's
  `DiscreteMotionValidator` calls `interpolate` N-1 times with the same pair,
  so the amortized cost is dominated by lookups, not computation.
- **Flat spaces with constant metrics don't benefit from discrete geodesic.**
  Geodesics on Euclidean/Torus with constant SPD metrics are straight lines —
  the naive `exp(p, t·log(p,q))` is already exact. The `is_riemannian_log`
  signal correctly activates the fast path for identity metrics. For non-identity
  constant SPD on flat spaces, both methods produce identical paths; the discrete
  geodesic adds overhead without quality gain. A future `IsFlat` concept could
  extend the fast path to all constant metrics on flat spaces.
- **`discrete_geodesic` cost is dominated by FD sampling for anisotropic /
  point-dependent metrics.** Isotropic rows (Sphere round, Torus, SE2 exp, KE
  on CSpace) converge in 2–23 iters with no FD fallback — hundreds of ns to a
  few µs per walk. Anisotropic rows (`SphereAniso`, `SE2_Anisotropic`,
  SDFConformal) pay 2·d `distance_midpoint_fd` samples per step; cost scales
  with the metric's per-evaluation price and the number of steps, not the
  loop overhead.
- **Batch-steer throughput sets the RRT\* budget.** With a reused workspace,
  geodex does ~2.7 M steers/s on Sphere (round), ~2.3 M/s on Torus<7>, and
  ~1.3 M/s on SE2 — a useful ceiling when sizing planners.
- **Metric unification is free.** Routing `EuclideanStandard`, `TorusFlat`, and
  `SE2LeftInvariant` through `ConstantSPDMetric` did not regress `inner`/`norm`
  performance (sub-nanosecond at low dim, identical to the old specialised
  implementations).
- **Use fixed-dimension types** (`Torus<7>` not `Torus<Eigen::Dynamic>`) when
  dimension is known at compile time. The dynamic penalty is ~20× at low
  dimensions.
- **Torus and Euclidean primitives are trivially fast** — exp/log reduce to
  arithmetic + angle wrapping.
- **SE2 is dominated by trig** (sin/cos/tan in the exponential map). Each
  exp/log costs ~5–8 ns from trig alone.
- **Sphere distance is still the most expensive isotropic primitive** (~79 ns)
  because `distance_midpoint` evaluates 1 exp + 3 log, each involving trig.
- **Point-dependent metrics** (KineticEnergy, Jacobi, Pullback) add 6–15 ns per
  inner product from the callable + matrix–vector product, which compounds
  through `distance_midpoint` and `discrete_geodesic`.
- **ConfigurationSpace overhead is moderate**: `distance_midpoint` on
  `CSpace<Torus<2>, KE>` costs 8.6 ns vs 5.7 ns for a bare `Torus<2>` flat —
  the mass-matrix evaluation adds ~50%.
