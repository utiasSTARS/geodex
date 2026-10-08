# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- The MoveIt 2 page shows the shelf move on a real FR3 under the kinetic-energy metric and under the Euclidean metric, side by side.
- The README and the documentation cite the geodex paper with its arXiv identifier.

### Changed
- The robot guides show the planning time, the time to the first solution and the iteration budget of each plan in tiles.
- The FR3 recording on the landing page and in the README shows the move from the middle compartment of the shelf to the top compartment, the move of the MoveIt 2 demo.

### Fixed

<br>

## Released

### [1.0.0] - 2026-10-06

geodex 1.0 plans collision-free, metric-aware robot motion in a single planning call, from Python or C++, with the whole stack in one `pip install`.

#### Highlights
- **Plan with a single call.** `geodex.plan(robot, start, goal, collision=scene)` returns a smooth, collision-free path. `planning::plan()` is the same call in C++.
- **Everything from pip.** `pip install pygeodex` now includes planning, SIMD collision checking and the built-in robots on Linux and macOS.
- **Nine robots, ready to plan.** Franka Panda and FR3, UR5, Baxter, PR2, Hello Robot Stretch 3 and Stretch 4, and a UR5e on a Clearpath Ridgeback or Husky.
- **Whole-body mobile manipulation.** Base and arm are planned together, and a holonomic or a differential-drive base is one metric setting. `make_product` and `heuristics::product_lower_bound` build the space and its heuristic for any base and arm.
- **Smooth paths.** `smooth_path` rounds the corners of a path into C² curves, checked like every other edge and at most 0.1 percent longer under the metric than the corners they replace. A time parameterization such as TOTG or TOPP-RA no longer stops at every waypoint. On a differential-drive base, the base pose may keep a corner where the base turns, and the arm's joints still turn along curves (`PathSmoothingSettings.sharp_coordinates`). `PathSmoothingSettings.round_corners = False` keeps the corners. `PathSmoothingSettings.edge_travel` spaces the edge checks by a bound on how far the checked geometry moves, and a robot plan spaces them by each edge's own sphere travel.
- **ROS 2 plugins.** A Nav2 global planner and a MoveIt planner, in their own repositories, for Jazzy and Lyrical.
- **Planning times.** Every plan reports the time and the iterations to its first solution (`first_solution_ms`, `first_solution_iterations`) next to the search and smoothing times (`time_ms`, `smooth_ms`). At `LogLevel.Info`, or with `GEODEX_LOG_LEVEL=info`, `plan()` prints them.
- **Reproducible.** A seed and an iteration budget give the same path on every run on the same platform, and one pixi workspace builds every dependency from one exact release or commit.
- **New documentation.** Every example comes in Python and C++, with interactive 3D views and robot guides.

#### Upgrading from 0.2
- `collision_resolution` is now `collision_check_resolution` in `PlanSettings` and `PathSmoothingSettings`, and it sets only the spacing of the edge checks. The MoveIt plugin's `planner.collision_resolution` and the Nav2 plugin's `collision_resolution` are renamed the same way.
- `simplify_path` and the earlier smoothers are replaced by `smooth_path`, which `plan()` runs for you. It returns the smoothed path at evenly spaced waypoints, at the longest step that follows its rounded corners within `corner_tolerance`, and `output_spacing` limits the step.
- The default sampler is scrambled Halton. Use `PseudoRandomSampler` for independent uniform samples.
- `plan()` joins states with the manifold's geodesic by default (`interp="base_geodesic"`). `interp="riemannian_geodesic"` follows the discrete geodesic of the metric, and `interp="auto"` picks between the two.
- `geodex.planners.RRTConnect` is removed, and `plan()` always runs G-RRT*. `PlanSettings.planner` holds the parameters of `GreedyRRTstar`.
- The Fetch robot is removed.
- Installed CMake targets are `geodex::robots`, `geodex::vamp` and `geodex::pinocchio`, and `find_package(geodex 1.0)` accepts any 1.x release.
- The examples follow the docs sections, and `examples/README.md` lists them. `pixi run quickstart` replaces `pixi run plan`.
- The sphere, `minimum_energy_grid` and `manipulator_planning` examples, the MotionBenchMaker sample problems and the `BUILD_BENCHMARKS` option are removed.

#### Fixes
- The built-in heuristics could overestimate the cost for some robot configurations.
- Arrays of the wrong size from Python now raise `ValueError` instead of crashing.
- Planning from several threads at once is now safe.
- A seeded plan in Python and the same plan in C++ could differ in the last bits and then return different smoothed paths. On one build they now return the same path bit for bit.
- A seeded plan returns the same path in a process that has planned before.
- With `greedy_ratio=0`, G-RRT* pruned its trees to the greedy set. It now prunes them to the informed set.
- With `limits` set, a C++ `plan()` on a space with a sampling box, such as `Euclidean`, sampled the box instead of the limits and could fail to find a path when the limits reached outside [-1, 1].

### [0.2.1] - 2026-07-01

#### Added - new major features
- Lie-group manifolds with canonical metrics and group-exponential retractions:
  - `SO2` — the circle group `S¹`; a 1-D angle in `[−π, π)` with a canonical (bi-invariant) metric.
  - `SO3` — points are unit quaternions `[x, y, z, w]`, tangents are body angular velocities; geodesics are quaternion SLERP. Selectable `body` (left-invariant) / `world` (right-invariant) frame.
  - `SE3` — a genuine Lie group; points are `[t, quat]` 7-vectors, tangents are `[v; ω]` twists; geodesics are coupled screw motions. Selectable `body` / `world` frame.
- `ProductManifold<Ms...>` — the Riemannian product of several manifolds (e.g. `R^n × SE(2)` for a mobile manipulator); exposed in Python as `geodex.Product([...])`.
- `SE2RightExponentialMap` — a world-frame (right-invariant) SE(2) retraction alongside the existing body-frame one.
- Shared Lie-group math (`geodex/utils/lie.hpp`): quaternion algebra, `so3_exp`/`so3_log`, `se3_exp`/`se3_log`, and the SE(3) left Jacobian.
- Python bindings for `SO2`, `SO3`, `SE3`, and `Product`.
- `.waypoints` on `discrete_geodesic`, `smooth_path`, and `simplify_path` results — the path as the original `list[np.ndarray]`.
- Distributed on PyPI as **`pygeodex`**: `pip install pygeodex` (imports as `geodex`) — a lean, dependency-light nanobind `abi3` wheel for CPython 3.12+.

#### Changed
- Result `.path` now returns an `(N, d)` float64 NumPy array instead of a `list[np.ndarray]`; the previous list is available unchanged as `.waypoints`.

#### Fixed
- `discrete_geodesic` sizes its finite-difference tangent basis by the tangent dimension rather than the point dimension, so it is correct on manifolds whose point representation differs from their tangent dimension (e.g. `SO3`, `SE3`).

### [0.2.0] - 2026-06-30

#### Added - new major features
- Built-in robot dynamics (`geodex::robots`) — an always-on `geodex_robots` archive with **no Pinocchio dependency**:
  - `robots::MassMatrix<Robot::R>` — precompiled CRBA joint-space mass matrix `M(q)` for **Panda, UR5, Fetch, Baxter, and PR2**, code-generated per robot from its URDF and post-processed for SIMD-friendly trigonometry. Fully fixed-size at compile time.
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
