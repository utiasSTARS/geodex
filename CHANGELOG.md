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

### [0.2.0] - 2026-06-30

#### Added - new major features
- Built-in robot dynamics (`geodex::robots`) — an always-on `geodex_robots` archive with **no Pinocchio dependency**:
  - `robots::MassMatrix<Robot::R>` — precompiled CRBA joint-space mass matrix `M(q)` for **Panda, UR5, Fetch, Baxter, and PR2**, code-generated per robot from its URDF and post-processed for SIMD-friendly trigonometry. Fully fixed-size at compile time and roughly 2× faster than Pinocchio's CRBA.
  - `robots::MassLowerBound<Robot::R>::matrix()` — a certified constant SPD matrix that lower-bounds `M(q)` in the Loewner order over each robot's joint-limit box, shipped precomputed so planners load a constant instead of running `precompute_matrix_lower_bound` at startup.
- Admissible heuristics (`geodex::heuristics`):
  - `heuristics::MatrixLowerBound` — informed-sampling heuristic built from a constant SPD Loewner lower bound of the metric.
  - `heuristics::EigenvalueLowerBound` — scalar minimum-eigenvalue heuristic.
- `algorithm::precompute_matrix_lower_bound()` — certifies a constant SPD Loewner lower bound of a configuration-dependent metric over a box.
- `algorithm::simplify_path()` — metric-aware random shortcutting that only accepts collision-free, lower-energy subpaths.
- `metrics::AffineCombinedMetric` — variadic non-negative affine combination of metric policies, with a deduction guide for `AffineCombinedMetric({c0, c1}, m0, m1)`.
- `lo()` / `hi()` bound accessors on the built-in manifolds.
- OMPL integration — direct informed sampling, cost-bound feedback, and solver diagnostics for the geodesic optimization objective.
- Optional Pinocchio integration (`GEODEX_PINOCCHIO=ON`, namespace `geodex::integration::pinocchio`) — runtime URDF mass matrix, frame Jacobian, and pullback-metric builders for arbitrary URDFs.
- Optional VAMP integration (`GEODEX_VAMP=ON`, namespace `geodex::integration::vamp`) — SIMD collision checking, scene loading, and motion validation along manifold geodesics.
- Robot descriptions (URDF + meshes) for Panda and PR2, plus MotionBenchMaker (MBM) benchmark problems, under `data/`.
- Example: `manipulator_planning` — kinetic-energy (mass-matrix) planning on the Panda using the OMPL and VAMP integrations.
- Benchmark: `bench_robots` — precompiled CRBA vs Pinocchio microbenchmark.
- Python bindings for the new heuristics, `simplify_path`, `precompute_matrix_lower_bound`, `AffineCombinedMetric`, and the optional Pinocchio/VAMP integrations.

#### Changed
- Heuristics moved into a dedicated `geodex::heuristics` namespace and `heuristics/` header directory. The Euclidean heuristic is renamed: `geodex::EuclideanHeuristic` (`geodex/algorithm/heuristics.hpp`) → `geodex::heuristics::Euclidean` (`geodex/heuristics/heuristics.hpp`).

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
