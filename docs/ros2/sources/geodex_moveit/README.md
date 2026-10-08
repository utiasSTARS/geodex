<h1 align="center">geodex_moveit</h1>

<p align="center">
  A <a href="https://moveit.picknik.ai">MoveIt 2</a> planner plugin for manipulators, built on
  <a href="https://github.com/utiasSTARS/geodex">geodex</a>.
</p>

<p align="center">
  <a href="https://github.com/utiasSTARS/geodex_moveit/actions/workflows/jazzy.yml"><img alt="Jazzy" src="https://github.com/utiasSTARS/geodex_moveit/actions/workflows/jazzy.yml/badge.svg"></a>
  <a href="https://github.com/utiasSTARS/geodex_moveit/actions/workflows/lyrical.yml"><img alt="Lyrical" src="https://github.com/utiasSTARS/geodex_moveit/actions/workflows/lyrical.yml/badge.svg"></a>
  <a href="https://docs.ros.org"><img alt="ROS 2 Jazzy and Lyrical" src="https://img.shields.io/badge/ROS%202-Jazzy%20%7C%20Lyrical-22314E?logo=ros&logoColor=white"></a>
  <a href="https://github.com/utiasSTARS/geodex"><img alt="geodex version" src="https://img.shields.io/badge/dynamic/regex?url=https%3A%2F%2Fraw.githubusercontent.com%2FutiasSTARS%2Fgeodex_moveit%2Fmain%2Fscripts%2Fdeps.env&search=GEODEX_REF%3A-v%28%5B0-9.%5D%2B%29&replace=%241&label=geodex&color=2980b9"></a>
  <a href="https://geodex.readthedocs.io/en/latest/ros2/moveit.html"><img alt="Documentation" src="https://img.shields.io/badge/docs-geodex.readthedocs.io-8ca1af?logo=readthedocs&logoColor=white"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
</p>

<table>
  <tr>
    <td width="50%"><img alt="A Franka FR3 arm moves a cracker box from the middle compartment of a shelf to the top compartment. The lower arm stays in place and the wrist turns the box." src="doc/real_fr3_shelf_kinetic_energy.webp" width="100%"></td>
    <td width="50%"><img alt="The same FR3 arm makes the same move and swings the whole arm from its base" src="doc/real_fr3_shelf_euclidean.webp" width="100%"></td>
  </tr>
  <tr>
    <td>A Franka FR3 moves a box to the top compartment of a shelf under the kinetic-energy metric.</td>
    <td>The same move under the Euclidean metric.</td>
  </tr>
</table>

`geodex_moveit` plans joint-space paths for fixed-base arms with G-RRT* under a Euclidean or a
kinetic-energy metric, and smooths them with geodex's metric-aware smoother.

- **Plans in milliseconds.** The planner finds a first path in under a millisecond and returns a
  smooth trajectory in tens of milliseconds.
- **Two metrics.** `metric: kinetic_energy` moves the light wrist joints and spares the heavy base
  joints. `metric: euclidean` returns the shortest joint path.
- **Fast collision checking.** With `checker: vamp`, VAMP checks geodex's sphere model of the
  robot against the planning scene, attached objects included. `checker: moveit` plans for any
  robot.
- **Reproducible.** A seed and an iteration budget give the same trajectory on every run.

## ROS 2 distributions

| ROS 2 | MoveIt 2 | Build and test |
|---|---|---|
| Jazzy | 2.12 | [![Jazzy](https://github.com/utiasSTARS/geodex_moveit/actions/workflows/jazzy.yml/badge.svg)](https://github.com/utiasSTARS/geodex_moveit/actions/workflows/jazzy.yml) |
| Lyrical | 2.15 | [![Lyrical](https://github.com/utiasSTARS/geodex_moveit/actions/workflows/lyrical.yml/badge.svg)](https://github.com/utiasSTARS/geodex_moveit/actions/workflows/lyrical.yml) |

Each workflow builds and tests both packages on Ubuntu with apt and with pixi, and on macOS arm64
with pixi. `geodex_moveit` is the plugin, of type `geodex_moveit/GeodexPlanner`, a
`planning_interface::PlannerManager`. `geodex_moveit_demos` holds the demos and the robot
configurations.

The [MoveIt 2 guide](https://geodex.readthedocs.io/en/latest/ros2/moveit.html) of the geodex
documentation has the walkthrough and every parameter.

## Quickstart with pixi

[pixi](https://pixi.sh) 0.81 or newer installs ROS 2, MoveIt 2 and RViz from RoboStack and builds
geodex, VAMP and the OMPL fork that holds G-RRT*.

1. Clone the repository.

   ```sh
   git clone https://github.com/utiasSTARS/geodex_moveit.git
   cd geodex_moveit
   ```

2. Build geodex and both packages, and run the tests.

   ```sh
   pixi run -e jazzy test        # or -e lyrical
   ```

3. Start the FR3 shelf demo on mock hardware with RViz.

   ```sh
   pixi shell -e jazzy
   source install/jazzy/setup.bash
   ros2 launch geodex_moveit_demos fr3_shelf.launch.py mock_hardware:=true rviz:=true
   ```

4. In a second shell of the same environment, plan and execute the move of the box.

   ```sh
   ros2 run geodex_moveit_demos fr3_shelf_plan.py
   ```

`GEODEX_SOURCE_DIR=/path/to/geodex` in front of a `pixi run` command builds against a local
geodex tree. `ros2 launch geodex_moveit_demos panda_demo.launch.py` starts a Panda demo with the
geodex and OMPL pipelines.

## Use it with your robot

Load `geodex_moveit/config/geodex_planning.yaml` under a pipeline name, and set the robot's
parameters in the same namespace.

```yaml
robot: panda              # a geodex robot model, or empty with checker: moveit
checker: vamp             # vamp (geodex's sphere model) or moveit (the planning scene)
metric: kinetic_energy    # or euclidean
obstacle_padding: 0.0241  # clearance of planned configurations from every obstacle
```

`geodex_moveit_demos/launch/panda_demo.launch.py` shows the whole setup, and
`geodex_moveit/config/geodex_moveit.example.yaml` lists every parameter.

- The plugin supports only fixed-base arms at the moment. Plan mobile manipulators with geodex
  directly.
- The plugin refuses requests with path or trajectory constraints.
- Keep `ValidateSolution` in the pipeline, and raise `obstacle_padding` when it rejects a
  trajectory.
- With `planner.seed` and `planner.max_iterations` set, a request returns the same path on every
  run.

## Attached objects

The vamp checker checks an object attached in the planning scene as one sphere that encloses it.
For an object that passes close to obstacles, give a sphere cover instead, as x, y, z and radius
per sphere in the frame of the link that holds the object.

```yaml
attached_object:
  spheres: [0.0, -0.1, 0.0, 0.036, 0.0, 0.1, 0.0, 0.036]
```

`geodex_moveit_demos/scripts/attached_object_spheres.py` turns the spheres that
[foam](https://github.com/CoMMALab/foam) fits to a mesh into this parameter.

## Building without pixi

On a machine with ROS 2 Jazzy or Lyrical and MoveIt 2, build geodex first and then the plugin.

1. Build geodex, the OMPL fork and VAMP into a prefix of their own.

   ```sh
   bash scripts/build_deps.sh /opt/geodex_deps
   ```

2. Install the ROS dependencies.

   ```sh
   rosdep install --from-paths . --ignore-src --skip-keys geodex
   ```

3. Build both packages against that prefix.

   ```sh
   colcon build --packages-select geodex_moveit geodex_moveit_demos \
     --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/geodex_deps
   ```

4. Run the tests.

   ```sh
   colcon test --packages-select geodex_moveit geodex_moveit_demos
   colcon test-result --verbose
   ```

## Citation

If you use this plugin in your research, please cite the [geodex paper](https://arxiv.org/abs/2610.09165).

```bibtex
@article{kyaw2026geodex,
  title   = {geodex: A Library for Motion Planning on {Riemannian} Manifolds},
  author  = {Kyaw, Phone Thiha and Wei, Ben and Samavi, Sepehr and
             {Rogel Garcia}, Miguel Angel and Kelly, Jonathan},
  journal = {arXiv preprint arXiv:2610.09165},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.09165}
}
```

## License

Copyright © 2026 Space and Terrestrial Autonomous Robotic Systems (STARS) Lab.

Apache-2.0. See `LICENSE`.
