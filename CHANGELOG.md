# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Fixed

<br>

## Released

### [0.1.1] - 2026-04-23

#### Added - new major features
- Discrete geodesic interpolation algorithm (`discrete_geodesic`).
- New collision checking module: smooth-SDF primitives (`CircleSmoothSDF`, `RectangleSmoothSDF`), `GridSDF`, `PolygonFootprint`, `FootprintGridChecker`.
- `SDFConformalMetric` — turns any base metric into an obstacle-aware metric via a smooth SDF callable.
- `smooth_path()` - metric-aware shortcutting and collision-constrained L-BFGS energy minimization.
- `SE2LeftInvariantMetric::car_like(radius, lateral_penalty)` static factory for turning-radius-constrained SE(2) planning.
- n-dimensional Sphere
    - Sphere<Dim> now supports any dimensions
- OMPL integration
  - GeodexStateSpace<Manifold> adapts any RiemannianManifold to OMPL's StateSpace.
  - GeodexOptimizationObjective<Manifold, Heuristic> for geodesic distance cost + admissible heuristic.
  - GeodexDirectInfSampler<Manifold, Heuristic> for informed sampling (PHS for Euclidean heuristic, rejection otherwise).
  - GeodexValidityChecker for OMPL motion validation.
- `Sampler` concept with `StochasticSampler` and `HaltonSampler`; all manifolds take a `SamplerT` template parameter.
- CMake install targets and find_package(geodex) support
- New python bindings and tests
- Examples: `sphere_interpolation` (C++ and Python), `se2_tutorial` (holonomic / diff-drive / clearance / parking on a real costmap), `minimum_energy_planning` (planar arm under KE and Jacobi metrics).
- Documentation updates
    - New SE2 planning tutorial
    - Minimum energy planning tutorial now includes planning with OMPL section
    - New concept page for discrete geodesic interpolation algorithm
    - Redesigned landing page, and vendored MathJax for offline builds.

#### Changed
- `SE2` sampling bounds unified into `lo`/`hi` `Vector3d` over `(x, y, θ)`; default θ bounds `[−π, π)`.
- `injectivity_radius()` moved from metrics onto manifolds.
- `Sphere` exp/log/distance parameterized on the metric (was round-metric-only).
- Composable metric refactors
    - WeightedMetric — uniform scalar (or configuration-dependent callable) scaling wrapper around any base metric.
    - JacobiMetric — now composed over KineticEnergyMetric + WeightedMetric; static_assert callability checks on construction.
    - SE2LeftInvariantMetric — composed over WeightedMetric + ConstantSPDMetric.
- `type_name<T>()` moved to `core/debug.hpp`; `MetricHasInnerMatrix` concept and `is_riemannian_log()` resolver added in `core/metric.hpp`.
- All manifolds preallocate a sample_buf_ for random_point() (no per-call allocation)
- clang-format applied repo-wide

### [0.1.0] - 2026-04-02

Initial public release.

#### Added
- C++20 concept hierarchy: `Manifold`, `RiemannianManifold`, `HasMetric`, `HasDistance`, `HasGeodesic`, `HasInjectivityRadius`
- Manifold implementations: `Sphere`, `Euclidean`, `Torus`, `SE2`, `ConfigurationSpace`
- Metric policies: `ConstantSPDMetric`, `SE2LeftInvariantMetric`, `KineticEnergyMetric`, `JacobiMetric`, `PullbackMetric`, `WeightedMetric`
- Retraction policies: `SphereExponentialMap`, `SphereProjectionRetraction`, `SE2ExponentialMap`, `SE2EulerRetraction`
- Algorithm: `distance_midpoint` (geodesic distance approximation)
- Python bindings via nanobind (`pip install geodex`)
- Sphinx + Doxygen documentation
- C++ and Python examples: `sphere_basics`, `sphere_distance`, `minimum_energy_grid`
- GoogleTest test suite
- CI with GitHub Actions (build, test, coverage, Python, docs)
