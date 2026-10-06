# Franka FR3 with a Robotiq 2F-85 (`fr3_arm_gripper`)

A Franka FR3 with a Robotiq 2F-85 on its ISO 9409-1-50-4-M6 coupling, built from the upstream macros by `scripts/robotgen/robots/fr3_arm_gripper.urdf.xacro` and the recipe `scripts/robotgen/robots/fr3_arm_gripper.json`.

| file | contents |
|---|---|
| `fr3_arm_gripper_dynamics.urdf` | Fixed-base arm with the gripper, kinematics and inertials only. Source of the CRBA `src/robots/generated/fr3_gripper_crba.*` and of its Loewner bound. |
| `fr3_arm_gripper_spherized.urdf` | The same model with foam collision spheres. Source of the VAMP kernel `include/geodex/integration/vamp/robots/generated/fr3_arm_gripper.hh`. |
| `fr3_arm_gripper.srdf` | Self-collision link pairs the kernel skips. |
| `LICENSE.franka_description`, `LICENSE.ros2_robotiq_gripper` | Licenses of the upstream descriptions, with the franka_description NOTICE. |

- Sources: `frankarobotics/franka_description` tag 2.9.0 (commit `7aeeddc449edf8d62b594f9e36a81da53e7796f9`, archive sha256 `cf625af64ea29a03358d6adc049accc60e622d02a6f6434f33beeb703cf308c2`) for the FR3 kinematics, joint limits, inertials and collision meshes, and `PickNikRobotics/ros2_robotiq_gripper` commit `a74d007d8f2f06dc6a503ad21038ba869d4999a6` (archive sha256 `7f471e183d0f34e0200aecd1840e0d56fa0e17df4639a3e493e408d40fc9835b`) for the 2F-85 and its coupling (`ur_to_robotiq_adapter`).
- Licenses: Apache-2.0 with a NOTICE (franka_description, Copyright 2023 Franka Robotics GmbH) and BSD-3-Clause (ros2_robotiq_gripper, Copyright (c) 2022 PickNik Robotics). No meshes are vendored.
- Configuration: `(fr3_joint1, ..., fr3_joint7)` with the franka_description limits. The coupling sits on the flange `fr3_link8` turned a quarter turn about its axis, and the fingers close along the flange y axis. The accelerometer frames of franka_description are left out.
- Fingers: `robotiq_85_left_knuckle_joint` is held at 0.44844 rad and every finger joint follows it, which puts the inner faces of the two finger pads 40.0 mm apart. The fully open gripper measures 85.0 mm.
- Tool centre point: `2f85_tcp` sits on the gripper axis 0.142 m from `robotiq_85_base_link`, midway between the finger pads and at the height of their centres, 0.153 m from the flange. Its x axis points along the closing direction and its z axis out of the fingers.
- Spheres: 60 foam spheres (medial, depth 1, branch 8, meshes scaled by 0.9 before fitting), with per-link budgets in the recipe. `fr3_link0` has 5, `fr3_link1` to `fr3_link4` and `fr3_link6` have 4 each, `fr3_link5` has 8, `fr3_link7` has 3, the coupling 2, the gripper base 4 and each finger 9. No collision-mesh point lies more than 22.9 mm outside the spheres, the spheres add 7.9 L (55 percent of the mesh volume) outside the meshes, and 5 percent of the mesh volume lies outside the spheres (`scripts/robotgen/robotgen.py fit`). The base spheres reach 36 mm below the mounting plane, and a floor or table obstacle must sit lower than that.
- Regenerate: `scripts/robotgen/generate.sh fr3_arm_gripper`.
