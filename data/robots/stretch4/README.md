# Hello Robot Stretch 4

Whole-body model of the Stretch 4 (SE4, batch francis) with the DexWrist 4 and the SG4 gripper on its three-omniwheel holonomic base.

| file | contents |
|---|---|
| `stretch4_dynamics.urdf` | Fixed-base arm, kinematics and inertials only. Source of the CRBA in `src/robots/generated/stretch4_crba.*` and of its Loewner bound. |
| `stretch4_spherized.urdf` | Whole-body model with foam collision spheres. Source of the VAMP kernel `include/geodex/integration/vamp/robots/generated/stretch4.hh`. |
| `stretch4.srdf` | Self-collision link pairs the kernel skips. |
| `LICENSE.md` | License of the upstream description, which these files derive from. |

- Source: PyPI `hello-robot-stretch4-urdf==2026.8.21` (wheel sha256 `a7b74b8534b2502d5a43dc6507fe72f6e95762e3a0927dd8460a24feb855bbba`), expanded with `stretch4_urdf.get_urdf(model_name="SE4", batch_name="francis", tool_name="eoa_wrist_dw4_tool_sg4")`. The wheel is the pinned source.
- License: Clear BSD, Copyright (c) 2021-2026 Hello Robot Inc. No upstream meshes are vendored.
- Configuration: `(x, y, theta, lift_joint, arm_joint, wrist_yaw_joint, wrist_pitch_joint, wrist_roll_joint)`. The planar chain attaches at `base_footprint`, which sits on the floor 28 mm below `base_link`.
- Telescoping arm: `arm_joint` is the total extension, 0 to 0.52 m, driving `arm_l1_joint` to `arm_l4_joint` with multiplier 0.25 each (no mimic tags upstream).
- Wrist pitch limits (-1.135 to 4.276 rad) equal the yaw limits upstream. `stretch4_body` d6175993 `robot_params_SE4.py` gives `range_deg [-65, 245]` for yaw, pitch and roll alike, so the URDF matches the driver's servo range.
- The upstream `arm_l4` collision mesh is inside out; the generator re-orients it before spherization.
- Sphere-model conservatism: wrist roll between about -1.6 and -0.4 rad collides with the yaw link at every pitch, while the meshes touch there only at extreme pitch.
- The base spheres reach 29 mm below the floor.
- Regenerate: `scripts/robotgen/generate.sh stretch4`.
