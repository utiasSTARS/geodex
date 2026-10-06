<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/utiasSTARS/geodex/master/docs/_static/logo-dark.svg">
    <img alt="geodex logo" src="https://raw.githubusercontent.com/utiasSTARS/geodex/master/docs/_static/logo-light.svg" width="150">
  </picture>
</p>

<h1 align="center">geodex</h1>

<p align="center">
  <a href="https://github.com/utiasSTARS/geodex/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/utiasSTARS/geodex/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://codecov.io/gh/utiasSTARS/geodex"><img alt="codecov" src="https://codecov.io/gh/utiasSTARS/geodex/graph/badge.svg"></a>
  <a href="https://pypi.org/project/pygeodex/"><img alt="PyPI" src="https://img.shields.io/pypi/v/pygeodex"></a>
  <a href="https://geodex.readthedocs.io"><img alt="Documentation" src="https://readthedocs.org/projects/geodex/badge/?version=latest"></a>
  <a href="https://github.com/utiasSTARS/geodex/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
</p>

**geodex** plans robot motion on Riemannian manifolds, in Python and in C++. You give it a
configuration space, a metric and a collision scene, and it returns a smooth, collision-free path
that is short under that metric.

<table>
  <tr>
    <td width="50%"><img alt="A Franka FR3 arm moves a cracker box from the bottom compartment of a shelf to the top compartment and back" src="https://raw.githubusercontent.com/utiasSTARS/geodex/master/docs/_static/videos/real-fr3-shelf.webp" width="100%"></td>
    <td width="50%"><img alt="A Clearpath Jackal drives between office desks and through a narrow passage" src="https://raw.githubusercontent.com/utiasSTARS/geodex/master/docs/_static/videos/real-jackal-office.webp" width="100%"></td>
  </tr>
  <tr>
    <td>A Franka FR3 moves a box between shelf compartments with the MoveIt 2 plugin under the kinetic-energy metric.</td>
    <td>A Clearpath Jackal plans on SE(2) with the Nav2 plugin and drives through a narrow passage of an office.</td>
  </tr>
</table>

```python
import numpy as np
import geodex

robot = geodex.robots.Panda()  # seven joints, the kinetic-energy metric
scene = geodex.Scene()
scene.add_box(position=[0.4, 0.0, 0.3], size=[0.1, 0.1, 0.8])

start = np.array([-1.1, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
goal = np.array([1.1, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
result = geodex.plan(robot, start, goal, scene, settings=geodex.PlanSettings(iterations=1500, seed=1))
print(result.solved, result.cost, result.path.shape)
```

## Features

- **Plan in Python or C++.** `geodex.plan` in Python and `geodex::planning::plan` in C++ run
  G-RRT* with informed sampling, then smooth the path under the same metric.
- **Built-in robots.** geodex includes the Franka Panda and FR3, UR5, Baxter, PR2, Hello Robot
  Stretch 3 and Stretch 4, and a UR5e on a Clearpath Ridgeback or Husky, each with VAMP collision
  checking and its mass matrix.
- **Whole-body planning.** geodex plans the base and the arm together on SE(2) × ℝⁿ. A
  holonomic and a differential-drive base differ only in the base metric.
- **Collision-free edges.** Robot paths are checked along every edge, not only at waypoints.
- **Manifolds and metrics.** geodex has ℝⁿ, tori, spheres, SO(2), SO(3), SE(2), SE(3) and their
  products, with kinetic-energy, left-invariant, clearance and custom metrics.
- **ROS 2.** A Nav2 global planner ([geodex_nav2_planner](https://github.com/utiasSTARS/geodex_nav2_planner))
  and a MoveIt planner ([geodex_moveit](https://github.com/utiasSTARS/geodex_moveit)) run geodex
  in ROS 2 Jazzy and Lyrical.
- **Reproducible.** A seed and an iteration budget give the same path on every run on the same
  platform.

## Installation

### Python

```sh
pip install pygeodex
```

The package imports as `geodex` and needs Python 3.12 or newer. On Linux (x86-64 and aarch64,
glibc 2.28 or newer) and macOS (arm64 and x86-64), it includes planning, collision checking and the
built-in robots. On x86-64, collision checking of the built-in robots requires a CPU with AVX2 and
FMA, and planning with a Python validity function works on any CPU. On Windows and on Linux with
musl, it includes the geometry core only (manifolds, metrics, sampling, distances, interpolation
and the smoother).

### Everything with pixi

```sh
git clone https://github.com/utiasSTARS/geodex.git && cd geodex
pixi run test        # build the OMPL fork with G-RRT*, VAMP, geodex and its Python module, then run the tests
pixi run quickstart  # plan the quickstart example, a Panda arm around a post
pixi run docs        # build the documentation site into build/docs/sphinx
```

`pixi run install` installs geodex and the OMPL fork into `.pixi/prefix` (`$GEODEX_PREFIX`).

### C++ package

Install Ninja, yaml-cpp and Boost 1.68 or newer with its serialization and program_options
libraries first, for example with your system's package manager.

```sh
git clone --branch v1.0.0 https://github.com/utiasSTARS/geodex.git
cmake -S geodex -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<prefix> \
  -DGEODEX_OMPL=ON -DGEODEX_BUILD_OMPL=ON -DGEODEX_VAMP=ON
cmake --build build && cmake --install build
```

The build also compiles the OMPL fork that holds the G-RRT* planner and installs it next to geodex.
`scripts/install_geodex.sh <prefix>` runs the same steps. A C++ project then adds the prefix to
`CMAKE_PREFIX_PATH` and links the package.

```cmake
find_package(geodex 1.0 CONFIG REQUIRED COMPONENTS ompl vamp)
target_link_libraries(my_planner PRIVATE geodex::geodex ompl::ompl)
```

## Documentation

The **[geodex documentation](https://geodex.readthedocs.io)** has the tutorials, the robot guides
and the API reference for Python and C++. To build the site locally, run `pixi run docs` and open
`build/docs/sphinx/index.html`.

## Citation

If you use geodex in your research, please cite the geodex paper.

```bibtex
@article{kyaw2026geodex,
  title   = {geodex: A Library for Motion Planning on {Riemannian} Manifolds},
  author  = {Kyaw, Phone Thiha and Wei, Ben and Samavi, Sepehr and
             {Rogel Garcia}, Miguel Angel and Kelly, Jonathan},
  journal = {arXiv preprint arXiv:26XX.XXXXX},
  year    = {2026},
  url     = {https://arxiv.org/abs/26XX.XXXXX}
}
```

geodex implements the methods of the papers below. Please also cite the ones your work uses.

Midpoint geodesic distance and interpolation under a Riemannian metric:

```bibtex
@inproceedings{kyaw2026geometry,
  title     = {Geometry-Aware Sampling-Based Motion Planning on {Riemannian} Manifolds},
  author    = {Kyaw, Phone Thiha and Kelly, Jonathan},
  booktitle = {Proceedings of the 17th World Symposium on the Algorithmic Foundations
               of Robotics (WAFR)},
  address   = {Oulu, Finland},
  month     = jun,
  year      = {2026},
  url       = {https://arxiv.org/abs/2602.00992}
}
```

The Loewner lower bounds behind the admissible heuristics and the informed sampling:

```bibtex
@article{kyaw2026loewner,
  title   = {Direct Informed Sampling on {Riemannian} Manifolds via {Loewner} Order
             Lower Bounds},
  author  = {Kyaw, Phone Thiha and Kelly, Jonathan},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2026},
  url     = {https://arxiv.org/abs/2606.02879}
}
```

The G-RRT\* planner:

```bibtex
@article{kyaw2026greedy,
  title   = {Greedy Heuristics for Sampling-Based Motion Planning in High-Dimensional
             State Spaces},
  author  = {Kyaw, Phone Thiha and Le, Anh Vu and Mohan, Rajesh Elara and
             Kelly, Jonathan},
  journal = {Autonomous Robots},
  year    = {2026},
  url     = {https://arxiv.org/abs/2405.03411}
}
```

## License

Copyright © 2026 Space and Terrestrial Autonomous Robotic Systems (STARS) Lab.

geodex is licensed under the [Apache License 2.0](https://github.com/utiasSTARS/geodex/blob/master/LICENSE).