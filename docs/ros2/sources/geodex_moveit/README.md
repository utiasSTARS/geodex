# geodex_moveit

`geodex_moveit` is a MoveIt 2 planner plugin built on
[geodex](https://github.com/utiasSTARS/geodex). It plans joint paths for fixed-base arms with
G-RRT* under a Euclidean or a kinetic-energy metric, and smooths them with geodex's metric-aware
smoother.

<table>
  <tr>
    <td width="50%"><img alt="A Franka FR3 arm moves a box between the compartments of a shelf" src="doc/real_fr3_shelf.webp" width="100%"></td>
    <td width="50%"><img alt="RViz view of the FR3 moving a box from the middle compartment of a shelf to the top compartment" src="doc/rviz_fr3_shelf.gif" width="100%"></td>
  </tr>
  <tr>
    <td>A Franka FR3 moves a box between shelf compartments under the kinetic-energy metric.</td>
    <td>The FR3 shelf demo of this repository in RViz.</td>
  </tr>
</table>

| | |
|---|---|
| ROS 2 | Jazzy (MoveIt 2.12) and Lyrical (MoveIt 2.15) |
| platforms | Linux x86-64, macOS arm64 |
| plugin type | `geodex_moveit/GeodexPlanner`, a `planning_interface::PlannerManager` |
| packages | `geodex_moveit` (the plugin), `geodex_moveit_demos` (demos and robot configurations) |
| license | Apache-2.0 |

The [geodex documentation](https://geodex.readthedocs.io) has the guide and every parameter.

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
geodex tree. `pixi run -e jazzy demo` starts a Panda demo with the geodex and OMPL pipelines.

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
   bash scripts/build_geodex.sh /opt/geodex_deps
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

## License

Apache-2.0. See `LICENSE`.
