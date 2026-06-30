# PR2 Assets

These files define geodex's fixed-base dual-arm PR2 model:

- `pr2.urdf` is used for precompiled CRBA generation.
- `pr2_spherized.urdf` is used by cricket to generate the VAMP collision kernel.
- `pr2.srdf` is the MoveIt PR2 semantic collision model used by cricket.

The movable planning joints are the 14 arm joints from `robot.yaml`'s
`dual_arm` planning group, in left-arm then right-arm order. Dataset problems
store joint values by name and are converted to this order at load time. Base,
torso, head, and gripper joints are fixed. Continuous arm roll joints are
represented as bounded revolute joints over `[-pi, pi]` so Pinocchio and VAMP
use a 14-dimensional configuration vector.

Source material:

- PR2 description meshes/URDF lineage: `petercorke/robotics-toolbox-python`
  commit `07ad1259338d6dfd41af6e5e50973e315284a912`, under
  `rtb-data/rtbdata/xacro/pr2_description`.
- SRDF: `ros-planning/moveit_pr2/pr2_moveit_config/config/pr2.srdf`.
- Spherized URDF seed: cricket's PR2 resource, normalized by
  `scripts/prepare_pr2_assets.py`.
