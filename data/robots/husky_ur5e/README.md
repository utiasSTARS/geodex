# Clearpath Husky with a UR5e

Whole-body model of a Clearpath Husky (A200, four-wheel skid steer, differential drive) with a Universal Robots UR5e on the default top plate, built from the upstream macros by `scripts/robotgen/robots/husky_ur5e.urdf.xacro`.

| file | contents |
|---|---|
| `husky_ur5e_dynamics.urdf` | Fixed-base arm, kinematics and inertials only. Source of the CRBA and its Loewner bound. |
| `husky_ur5e_spherized.urdf` | Whole-body model with collision spheres. Source of the VAMP kernel `husky_ur5e.hh`. |
| `husky_ur5e.srdf` | Self-collision link pairs the kernel skips. |
| `LICENSE.clearpath`, `LICENSE.universal_robots` | Licenses of the upstream descriptions. |

- Sources: `clearpathrobotics/clearpath_common` tag 2.9.17 (commit `726d7fba367f18e53899d3ac03f0b276380ae847`, archive sha256 `28981666117d72c939c2e13c3b535dc6dbf221c00898284dd9f20376fa717c56`) and `UniversalRobots/Universal_Robots_ROS2_Description` tag 4.3.1 (commit `ae333289875f9ba5a9ea6649a54036efb5ccabee`, archive sha256 `821b66bbb5f2188da49aa0020ffd2141378fdc68327c0cfa4c0f23ebf87ac997`), the same pins as the Ridgeback.
- Licenses: BSD-3-Clause (Clearpath, Copyright (c) 2023 clearpathrobotics) and BSD-3-Clause (Universal Robots descriptions for the UR5e). No meshes are vendored.
- Configuration: `(x, y, theta, arm_0_shoulder_pan_joint, arm_0_shoulder_lift_joint, arm_0_elbow_joint, arm_0_wrist_1_joint, arm_0_wrist_2_joint, arm_0_wrist_3_joint)`. The floor is 132.28 mm below `base_link` (outdoor wheel radius 0.1651 m, axle 0.03282 m above `base_link`), so the planar chain lifts the base by that much.
- The chassis and top plate are covered by a 7 x 4 x 2 grid of spheres that circumscribe their bounding box. The wheels get six foam spheres each, which reach 53 mm past the tyre sides and 42 mm below the floor, so a floor obstacle must sit lower than that.
- Regenerate: `scripts/robotgen/generate.sh husky_ur5e`.
