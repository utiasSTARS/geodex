# Hello Robot Stretch 3

Whole-body model of the Stretch 3 (SE3) with the DexWrist 3 and the SG3 gripper on its differential-drive base.

| file | contents |
|---|---|
| `stretch3_dynamics.urdf` | Fixed-base arm, kinematics and inertials only. Source of the CRBA in `src/robots/generated/stretch3_crba.*` and of its Loewner bound. |
| `stretch3_spherized.urdf` | Whole-body model with foam collision spheres. Source of the VAMP kernel `include/geodex/integration/vamp/robots/generated/stretch3.hh`. |
| `stretch3.srdf` | Self-collision link pairs the kernel skips. |
| `LICENSE.md` | License of the upstream description, which these files derive from. |

- Source: PyPI `hello-robot-stretch-urdf==0.1.2` (wheel sha256 `86a73bc111252b48001ee9abdc0c15cb792f291e88a1b678b5e1b3b0d83affe0`), file `stretch_urdf/SE3/stretch_description_SE3_eoa_wrist_dw3_tool_sg3.urdf`.
- License: Clear BSD, Copyright (c) 2021-2024 Hello Robot Inc. No upstream meshes are vendored; the files hold kinematics, inertials and sphere approximations derived from the description.
- Configuration: `(x, y, theta, joint_lift, joint_arm, joint_wrist_yaw, joint_wrist_pitch, joint_wrist_roll)`. A prismatic x, prismatic y and revolute yaw chain puts `base_link` on the floor at world z = 0.
- Telescoping arm: `joint_arm` is the total extension, 0 to 0.52 m. It is a massless driver joint, and the segment joints `joint_arm_l3` to `joint_arm_l0` mimic it with multiplier 0.25 each, so each segment extends its share as on the robot. The CRBA is taken in the independent coordinates, `A^T M(A r) A`.
- Fixed at zero: wheels, head pan and tilt, gripper fingers. Collision geometry dropped: ArUco marker plates, the base IMU.
- The base spheres reach 50 mm below the floor, so a floor obstacle must sit lower than that.
- Regenerate: `scripts/robotgen/generate.sh stretch3`.
