# Franka Emika Panda

- URDF `urdf/panda.urdf`: expanded from `franka_ros` `franka_description/robots/panda/panda.urdf.xacro` and hand-edited to fix inertia parameters (see its header). Meshes under `meshes/` come from the same package. robowflex_resources (KavrakiLab, commit `fb37f078fe27d5327781913ee130c3f0f2d70c0b`) records the upstream as `frankaemika/franka_ros` commit `edba362bc216d7169f14801c92af70f4291a0f76`.
- License: Apache-2.0, Copyright 2017 Franka Emika GmbH (`franka_description` package.xml and the `franka_ros` NOTICE). The NOTICE text must travel with redistributions.
- The VAMP kernel is VAMP's own `vamp/robots/panda.hh`, compiled from the pinned VAMP source.
