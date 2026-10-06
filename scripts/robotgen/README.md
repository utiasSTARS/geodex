# Robot data generation

These scripts regenerate the data geodex ships for its built-in robots. Normal builds do not run them and compile the committed results.

## Run

On Linux x86_64, from the repository root:

```sh
pixi run --manifest-path scripts/robotgen/pixi.toml scripts/robotgen/generate.sh          # all robots
pixi run --manifest-path scripts/robotgen/pixi.toml scripts/robotgen/generate.sh stretch3 # one robot
```

`WORK` (default `build/robotgen`) holds the tools, the source caches and the logs. `JOBS` sets the build parallelism and `SAMPLES` the self-collision sampling (default 100000). A full run of all nine robots takes about an hour.

## What it writes

| robot | inputs | outputs |
|---|---|---|
| `stretch3`, `stretch4`, `ridgeback_ur5e`, `husky_ur5e` | pinned upstream descriptions (`third_party/robot_descriptions.cmake`) and the recipe `robots/<robot>.json` | `data/robots/<robot>/` dynamics URDF, sphere URDF and SRDF; VAMP kernel `<robot>.hh`; CRBA; bound; sweep data |
| `fr3_arm_gripper` | pinned `franka_description` and `ros2_robotiq_gripper`, and the recipe `robots/fr3_arm_gripper.json` | `data/robots/fr3_gripper/` dynamics URDF, sphere URDF and SRDF; VAMP kernel `fr3_arm_gripper.hh`; CRBA `fr3_gripper`; bound; sweep data |
| `ur5` | `robots/ur5_expansion.urdf` and the pinned robowflex_resources meshes | `data/robots/ur5/ur5.urdf` (Robotiq and FT 300 meshes replaced by their bounding boxes); CRBA; bound; sweep data |
| `panda`, `baxter`, `pr2` | vendored dynamics URDFs in `data/robots/<robot>/` | CRBA; bound; sweep data |

The VAMP kernels of Panda, UR5 and Baxter are VAMP's own, and PR2's is vendored. Their sweep data comes from the sphere model each kernel was built from.

- The CRBA is `src/robots/generated/<name>_crba.{cpp,hpp}`, written by `pinocchio_codegen` and `post_process_sincos_simd.py`. An arm of nested segments, such as the Stretch arm, is one coordinate with the segments as URDF mimic joints, and the CRBA returns `A^T M(A r + b) A` in that coordinate.
- The bound is `src/robots/generated/<name>_bound.hpp`, a constant `M_lower` with `M(q) >= M_lower` on the whole joint box. `precompute_robot_bound.cpp` shapes it, and `crba_certify.hpp` proves it by an interval branch and bound over the shipped CRBA expression.
- The sweep data is `include/geodex/integration/vamp/robots/generated/<robot>_sweep.hh`. It holds the kernel's joint limits and bounds on how far any sphere center moves per unit of each coordinate (triangle inequality over the kinematic chain). The VAMP motion validator uses them.

## Files

- `generate.sh` builds cricket and foam at their pins and the geodex tools, fetches and checks the sources, and runs the steps above for each robot.
- `robotgen.py` does the per-robot work (`fetch`, `build`, `kernel`, `crba`, `sweep`, `fit`, `pin`, `get`).
- `verify.py` and `verify_kernel.cpp` check each generated robot against pinocchio (kernel sphere positions, self-collision results, the coupled CRBA, and every link and the end effector against the unmodified upstream description). The report also holds the sphere fit.
- `mesh_boxes.py` replaces the meshes a URDF names with their bounding boxes.
- `prepare_pr2_assets.py` wrote the vendored PR2 assets in `data/robots/pr2/`. `generate.sh` does not run it.
- `pinocchio_codegen.cmake` builds `pinocchio_codegen` and adds the `regenerate_robots` targets, which run it and `post_process_sincos_simd.py`.
- `crba_template.py` turns a shipped CRBA source into a template over the scalar type. The bound tool's build runs it.
- `add_robot.sh` and `update_robot_registry.py` add a robot of your own from its URDF (see `docs/robots/add-a-robot.rst`).
- `robots/` holds one recipe per robot, the top-level xacro files of the Clearpath robots and the FR3, and the UR5 expansion.
- `patches/` holds the cricket patch.
- `pixi.toml` and `pixi.lock` pin the environment.

## Reproducibility

The tools build without `-march=native` and fast math, and every seed is fixed. Two x86-64 hosts write the same bytes.
