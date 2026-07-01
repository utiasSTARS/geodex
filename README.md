# geodex

[![CI](https://github.com/utiasSTARS/geodex/actions/workflows/ci.yml/badge.svg)](https://github.com/utiasSTARS/geodex/actions/workflows/ci.yml)
[![docs](https://app.readthedocs.org/projects/geodex/badge/?version=latest&style=default)](https://app.readthedocs.org/projects/geodex)
[![codecov](https://codecov.io/github/utiasSTARS/geodex/graph/badge.svg?token=SJXNZZZQ9S)](https://codecov.io/github/utiasSTARS/geodex)
[![PyPI](https://img.shields.io/pypi/v/pygeodex)](https://pypi.org/project/pygeodex/)
[![Python](https://img.shields.io/pypi/pyversions/pygeodex)](https://pypi.org/project/pygeodex/)

**geodex** is a general-purpose software framework for planning on Riemannian manifolds.

Unlike traditional algorithms that operate in Euclidean space, geodex works directly with the intrinsic geometry of the state space. Manifolds are defined via C++20 concepts (`RiemannianManifold`, `HasMetric`, `HasGeodesic`, etc.) so algorithms work generically over any conforming type with zero overhead. Metrics and retractions are injected as template policies, making it easy to swap between e.g. true exponential maps and cheaper approximations.

## Key Features

- **Generic Manifolds** — Out-of-the-box support for Sⁿ, Rⁿ, Tⁿ, the Lie groups SO(2)/SO(3)/SE(2)/SE(3) with group-exponential geodesics, and product manifolds (e.g. Rⁿ × SE(2)), plus custom manifolds via template policies
- **Performance First** — C++20 core with zero-overhead generic algorithms, many built-in retractions, and anisotropic metrics
- **Precompiled Robots** — Built-in CRBA mass-matrix metrics for five manipulators (Panda, UR5, Fetch, Baxter, PR2), plus certified admissible heuristics for informed sampling
- **Python Bindings** — First-class Python support (`pip install pygeodex`)

## Roadmap

- [x] [OMPL](https://ompl.kavrakilab.org/) and [VAMP](https://github.com/KavrakiLab/vamp) integrations (Planning on Riemannian manifolds with state-of-the-art sampling-based planners)
- [ ] Nav2 and MoveIt 2 plugins (Geometry-aware planning for ROS 2 mobile robots and manipulators)

## Getting Started

All installation instructions, C++/Python tutorials, and API references are available at our documentation site:

👉 **[Read the Documentation](https://geodex.readthedocs.io)**

## Citation

geodex accompanies the WAFR'26 paper "[Geometry-Aware Sampling-Based Motion Planning on Riemannian Manifolds](https://arxiv.org/abs/2602.00992)":
```bibtex
@inproceedings{kyaw2026geometry,
  address = {Oulu, Finland},
  author = {Phone Thiha Kyaw and Jonathan Kelly},
  booktitle = {Proceedings of the 17th World Symposium on the Algorithmic Foundations of Robotics {(WAFR)}},
  date = {2026-06-15/2026-06-17},
  month = {Jun. 15--17},
  title = {Geometry-Aware Sampling-Based Motion Planning on {Riemannian} Manifolds},
  url = {https://arxiv.org/abs/2602.00992},
  year = {2026}
}
```

If you use matrix lower-bound heuristic in your research, please cite:

```bibtex
@article{kyaw2026direct,
  title={Direct Informed Sampling on Riemannian Manifolds via Loewner Order Lower Bounds},
  author = {Phone Thiha Kyaw and Jonathan Kelly},
  journal={arXiv preprint arXiv:2606.02879},
  url = {https://arxiv.org/abs/2606.02879},
  year={2026}
}
```

If you use Greedy RRT* (G-RRT*) in your research, please cite:

```bibtex
@article{kyaw2025greedy,
  title={Greedy heuristics for sampling-based motion planning in high-dimensional state spaces},
  author={Kyaw, Phone Thiha and Le, Anh Vu and Elara, Mohan Rajesh and Kelly, Jonathan},
  journal={arXiv preprint arXiv:2405.03411},
  url = {https://arxiv.org/abs/2405.03411},
  year={2025}
}
```

## License

Copyright © 2026 Space and Terrestrial Autonomous Robotic Systems (STARS) Lab.

geodex is licensed under the [Apache License 2.0](LICENSE).
