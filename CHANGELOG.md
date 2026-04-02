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
