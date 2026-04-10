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

# JSON output for regression tracking
./build/benchmarks/bench_manifold_ops --benchmark_format=json --benchmark_out=results/bench_manifold_ops.json

# Filter specific benchmarks
./build/benchmarks/bench_manifold_ops --benchmark_filter="Torus"
```

## Benchmark Suites

| File | What it measures |
|------|-----------------|
| `bench_manifold_ops.cpp` | exp, log, distance, inner, norm, geodesic, random_point per manifold; dimension scaling for Torus/Euclidean (d=2..50) |
| `bench_algorithms.cpp` | `discrete_geodesic` on all manifolds; batch `distance_midpoint` throughput (1000 pairs); dimension scaling |
| `bench_metrics.cpp` | `inner` and `norm` for all 7 metric types; ConfigurationSpace overhead vs bare manifold |
| `bench_retractions.cpp` | SE(2) and Sphere retraction speed; Sphere projection accuracy vs exponential map |
| `bench_ompl_interpolation.cpp` | OMPL `GeodexStateSpace::interpolate()` with discrete geodesic cache; cache hot/cold; motion validation overhead (requires OMPL) |

## Reference Results

Machine: Apple M2 (8-core, arm64), 16 GB RAM, macOS 15.3.1
Compiler: Apple clang 15.0.0, `-O2` (CMake Release)
Date: 2026-04-08 (re-run after `feature/refactoring`: metric/sampler policy split,
`discrete_geodesic` hardening, and `ConstantSPDMetric` unification)

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

| Dimension | Torus (fixed) | Torus (dynamic) | Euclidean (dynamic) |
|-----------|:---:|:---:|:---:|
| 2 | 7.9 | 156 | 96 |
| 5 | -- | 208 | 123 |
| 7 | 31 | 228 | 132 |
| 10 | -- | 218 | 121 |
| 20 | -- | 289 | 155 |
| 50 | -- | 892 | 550 |

Fixed-dimension types remain ~20x faster than dynamic at d=2 thanks to stack
allocation and compile-time loop unrolling.

### Algorithm Performance (ns per call)

| Manifold | discrete_geodesic |
|----------|:---:|
| Sphere (round metric) | 231 |
| Sphere (anisotropic) | 1,399 |
| Torus<2> (flat metric) | 138 |
| SE2 (exp map) | 1,240 |
| SE2 (anisotropic SE2LeftInvariant) | 7,293 |
| ConfigurationSpace<Torus<2>, KineticEnergyMetric> | 1,290 |

`discrete_geodesic` got roughly **5–8× faster** across the board compared to the
pre-refactor numbers (e.g. Sphere round 1432→231 ns, SE2 10639→1240 ns). The
speedup comes from the hardened natural-gradient loop landing in far fewer
iterations on well-conditioned inputs (`iters/call` is 2–7 for isotropic
metrics, 22–23 for anisotropic ones).

### Batch distance_midpoint Throughput (1000 pairs)

| Manifold | Total (us) | Throughput |
|----------|:---:|:---:|
| Torus<2> | 6.2 | 161M/s |
| Torus<7> | 41.3 | 24M/s |
| SE2 | 41.9 | 24M/s |
| Sphere | 88.9 | 11M/s |

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
- **`discrete_geodesic` is now the biggest win from the refactor** — the
  hardened natural-gradient loop converges in a handful of iterations for
  isotropic metrics, giving a 5–8× speedup on Sphere/Torus/SE2 without changing
  the public API.
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
