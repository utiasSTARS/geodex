# geodex examples

Each example is a Python script with a C++ version next to it, and each one is the code of a
page in the [geodex documentation](https://geodex.readthedocs.io).

## Running the examples

Install geodex with `pip install pygeodex` and run a Python example from the repository root.

```sh
python examples/getting_started/quickstart.py
```

The C++ examples build with `-DBUILD_EXAMPLES=ON`. The planning examples also need
`-DGEODEX_OMPL=ON`, and the robot examples need `-DGEODEX_VAMP=ON` as well (see Building from
Source in the documentation). In a pixi checkout, `pixi run build-cpp` builds all of them.

```sh
pixi run build-cpp
./build/examples/getting_started/quickstart
```

The SE(2) tutorial and the C++ navigation example load their maps from the current directory.
Run them from their own folder.

```sh
cd examples/tutorials
python se2_planning.py
```

## Getting started

| Example | Description | Page |
|---|---|---|
| `getting_started/first_plan` | Plans a differential-drive robot around a disc on SE(2). | Installation |
| `getting_started/quickstart` | Plans a Franka Panda around a post under its kinetic-energy metric. | Quickstart |
| `getting_started/reproducibility` | Repeats plans with seeds, iteration budgets and time budgets. | Reproducibility |

## Concepts

| Example | Description | Page |
|---|---|---|
| `concepts/metrics` | Measures the same motions under different metrics. | Metrics as Robot Models |
| `concepts/sampling` | Samples random points on manifolds and chooses and seeds their samplers. | Sampling on Manifolds |
| `concepts/discrete_geodesic` | Computes discrete geodesics on the sphere under two metrics. | Discrete Geodesic Interpolation |
| `concepts/planning` | Plans on the sphere and builds an admissible heuristic for a planar arm. | Planning |
| `concepts/smoothing` | Smooths a path on its own, with an edge proof, and inside `plan`. | Path Smoothing |

## Tutorials

| Example | Description | Page |
|---|---|---|
| `tutorials/geodex_basics` | Creates manifolds and computes exp, log, distances, geodesics and samples. | geodex Basics |
| `tutorials/minimum_energy_planning` | Plans a two-link arm under the Euclidean, kinetic-energy and Jacobi metrics. | Minimum-Energy Planning on Configuration Manifolds |
| `tutorials/se2_planning` | Plans a disc robot, a differential-drive robot and a car on SE(2). | SE(2) Motion Planning |

## Robots

| Example | Description | Page |
|---|---|---|
| `robots/catalog` | Lists the built-in robots. | Robot Guides |
| `robots/manipulation/arm_ke` | Moves a box between two shelf bays with a Franka FR3. | Manipulation |
| `robots/navigation/bases` | Plans three Clearpath bases through an office. | Navigation |
| `robots/mobile_manipulation/stretch` | Plans one kitchen task with a Stretch 3 and a Stretch 4. | Whole-Body Planning |
| `robots/mobile_manipulation/clearpath` | Plans a UR5e on a Husky and on a Ridgeback. | Whole-Body Planning |

## CMake project

`cmake_project` is a C++ project that finds an installed geodex with `find_package` and
plans a path. The Installation page shows its `CMakeLists.txt`.

```sh
cmake -S examples/cmake_project -B build/cmake_project -DCMAKE_PREFIX_PATH=<prefix>
cmake --build build/cmake_project
./build/cmake_project/my_planner
```

## ROS 2

The ROS 2 demos are in the plugin repositories,
[geodex_nav2_planner](https://github.com/utiasSTARS/geodex_nav2_planner) and
[geodex_moveit](https://github.com/utiasSTARS/geodex_moveit).
