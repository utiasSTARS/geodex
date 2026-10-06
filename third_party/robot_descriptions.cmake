# Pinned upstream robot descriptions and the tools that turn them into geodex robot data
# with scripts/robotgen/generate.sh. Only the maintainer regeneration downloads them.

# Hello Robot Stretch 3 and Stretch 4 descriptions (Clear BSD), as PyPI wheels.
set(GEODEX_STRETCH_URDF_URL
    "https://files.pythonhosted.org/packages/py3/h/hello-robot-stretch-urdf/hello_robot_stretch_urdf-0.1.2-py3-none-any.whl")
set(GEODEX_STRETCH_URDF_SHA256 "86a73bc111252b48001ee9abdc0c15cb792f291e88a1b678b5e1b3b0d83affe0")
set(GEODEX_STRETCH4_URDF_URL
    "https://files.pythonhosted.org/packages/py3/h/hello-robot-stretch4-urdf/hello_robot_stretch4_urdf-2026.8.21-py3-none-any.whl")
set(GEODEX_STRETCH4_URDF_SHA256 "a7b74b8534b2502d5a43dc6507fe72f6e95762e3a0927dd8460a24feb855bbba")

# Clearpath platform descriptions (BSD-3-Clause), tag 2.9.17.
set(GEODEX_CLEARPATH_COMMON_REF "726d7fba367f18e53899d3ac03f0b276380ae847")
set(GEODEX_CLEARPATH_COMMON_URL
    "https://github.com/clearpathrobotics/clearpath_common/archive/${GEODEX_CLEARPATH_COMMON_REF}.tar.gz")
set(GEODEX_CLEARPATH_COMMON_SHA256 "28981666117d72c939c2e13c3b535dc6dbf221c00898284dd9f20376fa717c56")

# Universal Robots ROS 2 description (BSD-3-Clause for UR3 to UR16e), tag 4.3.1.
set(GEODEX_UR_DESCRIPTION_REF "ae333289875f9ba5a9ea6649a54036efb5ccabee")
set(GEODEX_UR_DESCRIPTION_URL
    "https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/archive/${GEODEX_UR_DESCRIPTION_REF}.tar.gz")
set(GEODEX_UR_DESCRIPTION_SHA256 "821b66bbb5f2188da49aa0020ffd2141378fdc68327c0cfa4c0f23ebf87ac997")

# KavrakiLab robowflex_resources (MIT), the source of the UR5 with its Robotiq gripper and
# FT 300 sensor, and the recorded origin of the Panda and Baxter URDFs.
set(GEODEX_ROBOWFLEX_RESOURCES_REF "fb37f078fe27d5327781913ee130c3f0f2d70c0b")
set(GEODEX_ROBOWFLEX_RESOURCES_URL
    "https://github.com/KavrakiLab/robowflex_resources/archive/${GEODEX_ROBOWFLEX_RESOURCES_REF}.tar.gz")
set(GEODEX_ROBOWFLEX_RESOURCES_SHA256 "494ed07ad6c0ea9c0deabc6f1592589262d5fe8292ee414d9fda28789dd962a9")

# Franka Robotics franka_description (Apache-2.0), tag 2.9.0, the FR3 kinematics, joint limits,
# inertials and collision meshes.
set(GEODEX_FRANKA_DESCRIPTION_REF "7aeeddc449edf8d62b594f9e36a81da53e7796f9")
set(GEODEX_FRANKA_DESCRIPTION_URL
    "https://github.com/frankarobotics/franka_description/archive/${GEODEX_FRANKA_DESCRIPTION_REF}.tar.gz")
set(GEODEX_FRANKA_DESCRIPTION_SHA256 "cf625af64ea29a03358d6adc049accc60e622d02a6f6434f33beeb703cf308c2")

# PickNik ros2_robotiq_gripper (BSD-3-Clause), the Robotiq 2F-85 and its ISO 9409-1-50-4-M6
# coupling on the FR3.
set(GEODEX_ROBOTIQ_DESCRIPTION_REF "a74d007d8f2f06dc6a503ad21038ba869d4999a6")
set(GEODEX_ROBOTIQ_DESCRIPTION_URL
    "https://github.com/PickNikRobotics/ros2_robotiq_gripper/archive/${GEODEX_ROBOTIQ_DESCRIPTION_REF}.tar.gz")
set(GEODEX_ROBOTIQ_DESCRIPTION_SHA256 "7f471e183d0f34e0200aecd1840e0d56fa0e17df4639a3e493e408d40fc9835b")

# Generation tools, cloned with submodules at these commits. generate.sh applies
# scripts/robotgen/patches/cricket-base-position-braces.patch to cricket.
set(GEODEX_CRICKET_REF "2280986b5b0563f47365be04263ad8a79532790f")
set(GEODEX_CRICKET_GIT "https://github.com/CoMMALab/cricket.git")
set(GEODEX_FOAM_REF "116928f71aaa7c40356d79c84d3c9ff1f4497d90")
set(GEODEX_FOAM_GIT "https://github.com/CoMMALab/foam.git")
