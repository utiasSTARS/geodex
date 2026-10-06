# geodex_nav2_planner

`geodex_nav2_planner` is a Nav2 global planner built on
[geodex](https://github.com/utiasSTARS/geodex). It plans on SE(2) with G-RRT* under a metric that
weights forward, sideways and turning motion, checks the full footprint polygon against the
costmap, and smooths the path with geodex's metric-aware smoother.

<table>
  <tr>
    <td width="50%"><img alt="A Clearpath Jackal drives between office desks and through a narrow passage" src="doc/real_jackal_office.webp" width="100%"></td>
    <td width="50%"><img alt="RViz view of an office map with a planned path and the robot's footprint along it" src="doc/rviz_office.png" width="100%"></td>
  </tr>
  <tr>
    <td>A Clearpath Jackal drives a planned path through a narrow passage of an office.</td>
    <td>The office demo of this repository in RViz.</td>
  </tr>
</table>

| | |
|---|---|
| ROS 2 | Jazzy (Nav2 1.3) and Lyrical (Nav2 1.5) |
| platforms | Linux x86-64, macOS arm64 |
| plugin type | `geodex_nav2_planner::GeodexSE2Planner`, a `nav2_core::GlobalPlanner` |
| license | Apache-2.0 |

The [geodex documentation](https://geodex.readthedocs.io) has the guide and every parameter.

## Quickstart with pixi

[pixi](https://pixi.sh) 0.81 or newer installs ROS 2, Nav2 and RViz from RoboStack and builds
geodex and the OMPL fork that holds G-RRT*.

1. Clone the repository.

   ```sh
   git clone https://github.com/utiasSTARS/geodex_nav2_planner.git
   cd geodex_nav2_planner
   ```

2. Build geodex and the plugin, and run the tests.

   ```sh
   pixi run -e jazzy test        # or -e lyrical
   ```

3. Start the office demo with RViz.

   ```sh
   pixi shell -e jazzy
   source install/jazzy/setup.bash
   ros2 launch geodex_nav2_planner office.launch.py robot:=jackal rviz:=true
   ```

4. In a second shell of the same environment, request a path.

   ```sh
   ros2 run geodex_nav2_planner office_plan.py
   ```

`GEODEX_SOURCE_DIR=/path/to/geodex` in front of a `pixi run` command builds against a local
geodex tree.

## Use it in Nav2

Set the planner server's plugin and the metric weights of your base.

```yaml
planner_server:
  ros__parameters:
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "geodex_nav2_planner::GeodexSE2Planner"
      wx: 1.0      # forward motion
      wy: 50.0     # sideways motion, 50 for a differential drive, 1.0 for a holonomic base
      wtheta: 2.0  # rotation
```

`config/nav2_params.yaml` holds the planner server and global costmap blocks of a working
bring-up, and `config/geodex_nav2_planner.example.yaml` lists every parameter.

- The planner checks the footprint against the lethal cells. The global costmap does not need an
  inflation layer.
- `safety_margin` sets how far the footprint stays from every lethal cell.
- For a holonomic base, set `max_reverse_run: -1.0`.
- With a nonzero `seed` and a `refine_iterations` budget, a request returns the same path on every
  run.
- `publish_footprints` publishes the footprint along the path for RViz.

`config/robots/` holds the footprints and weights of four Clearpath bases.

| file | base | `wy` | `max_reverse_run` |
|---|---|---|---|
| `jackal.yaml` | skid steer | 50 | 0.2 |
| `husky.yaml` | skid steer | 50 | 0.2 |
| `ridgeback.yaml` | mecanum | 1 | -1 |
| `dingo_o.yaml` | mecanum | 1 | -1 |

## Building without pixi

On a machine with ROS 2 Jazzy or Lyrical and Nav2, build geodex first and then the plugin.

1. Build geodex and the OMPL fork into a prefix of their own.

   ```sh
   bash scripts/build_deps.sh /opt/geodex_deps
   ```

2. Install the ROS dependencies.

   ```sh
   rosdep install --from-paths . --ignore-src --skip-keys geodex
   ```

3. Build the plugin against that prefix.

   ```sh
   colcon build --packages-select geodex_nav2_planner \
     --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/geodex_deps
   ```

4. Run the tests.

   ```sh
   colcon test --packages-select geodex_nav2_planner
   colcon test-result --verbose
   ```

## Credits

Sepehr Samavi and Phone Thiha Kyaw, STARS Lab, University of Toronto.

## License

Apache-2.0. See `LICENSE`.
