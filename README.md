# geodex

[![CI](https://github.com/utiasSTARS/geodex/actions/workflows/ci.yml/badge.svg)](https://github.com/utiasSTARS/geodex/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/utiasSTARS/geodex/graph/badge.svg?token=SJXNZZZQ9S)](https://codecov.io/github/utiasSTARS/geodex)

**geodex** is a general-purpose software framework for planning on Riemannian manifolds.

Unlike traditional algorithms that operate in Euclidean space, geodex works directly with the intrinsic geometry of the state space. Manifolds are defined via C++20 concepts (`RiemannianManifold`, `HasMetric`, `HasGeodesic`, etc.) so algorithms work generically over any conforming type with zero overhead. Metrics and retractions are injected as template policies, making it easy to swap between e.g. true exponential maps and cheaper approximations.

## Key Features

- **Generic Manifolds** — Out-of-the-box support for $S^n$, $R^n$, $T^n$, SE(2), and custom manifolds via template policies
- **Performance First** — Header-only C++ core with many built-in retractions and anisotropic metrics
- **Python Bindings** — First-class support for Python (`pip install geodex`)

## Roadmap

- [ ] [OMPL](https://ompl.kavrakilab.org/) and [VAMP](https://github.com/KavrakiLab/vamp) integrations (Planning on Riemannian manifolds with state-of-the-art sampling-based planners)
- [ ] Nav2 and MoveIt 2 plugins (Geometry-aware planning for ROS 2 mobile robots and manipulators)

## Getting Started

All installation instructions, C++/Python tutorials, and API references are available at our documentation site:

👉 **[Read the Documentation](https://geodex.readthedocs.io)**

## Citation

geodex accompanies the paper "[Geometry-Aware Sampling-Based Motion Planning on Riemannian Manifolds](https://arxiv.org/abs/2602.00992)" accepted to [WAFR 2026](https://algorithmic-robotics.org/):
```bibtex
@article{kyaw2026geometry,
  title={Geometry-Aware Sampling-Based Motion Planning on Riemannian Manifolds},
  author={Kyaw, Phone Thiha and Kelly, Jonathan},
  journal={arXiv preprint arXiv:2602.00992},
  year={2026}
}
```

## License

geodex is licensed under the [Apache License 2.0](LICENSE).
