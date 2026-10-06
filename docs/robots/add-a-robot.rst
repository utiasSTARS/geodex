Add a Robot
===========

In this guide, we add a robot to geodex, a Clearpath Husky A200 with a UR5e. A pipeline turns
the robot's URDF into a VAMP collision model, a mass matrix generated as C++ and a lower bound
of that mass matrix for the planner's heuristic. The pipeline runs once per robot on Linux
x86-64, and a regular geodex build compiles its output.

.. robot-scene:: robots-gallery-husky-ur5e
   :spheres:
   :aspect: 6/5
   :alt: A Clearpath Husky with a UR5e, with the 130 collision spheres of its VAMP model drawn
         over its meshes.

   The Husky with a UR5e and the 130 spheres of its VAMP model. The Spheres button hides them.

1. Write a recipe
-----------------

A recipe in ``scripts/robotgen/robots/<name>.json`` describes the robot to the pipeline. The
Husky's recipe names its description, its base and drive, the joints the planner moves and the
number of spheres per link.

.. literalinclude:: ../../scripts/robotgen/robots/husky_ur5e.json
   :language: json

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Meaning
   * - ``name``, ``vamp_struct``
     - The robot's name in geodex, and the name of the model struct in its VAMP kernel.
   * - ``source``
     - The description, an archive listed in ``third_party/robot_descriptions.cmake``, and how
       to expand it into a URDF.
   * - ``base_link``, ``base_height``
     - The link that the planar base joints attach to, and its height above the floor.
   * - ``drive``
     - ``differential`` or ``holonomic``. A mobile robot's recipe has it, and the pipeline adds
       the planar base joints :math:`(x, y, \theta)`. A fixed-base arm leaves it out.
   * - ``planning_joints``
     - The joints the planner moves, in order. Every other joint is fixed.
   * - ``end_effector``
     - The link that VAMP attaches held objects to.
   * - ``spherize``, ``box_covers``
     - The number of spheres per link, and links covered by a grid of spheres instead.

2. Generate the robot
---------------------

For a mobile robot, add ``husky_ur5e=differential`` to ``GEODEX_ROBOT_BASES`` in
``cmake/robots_manifest.cmake``. Then register the robot and run the pipeline.

.. code-block:: bash

   python scripts/robotgen/update_robot_registry.py --name husky_ur5e \
       --urdf data/robots/husky_ur5e/husky_ur5e_dynamics.urdf
   pixi run --manifest-path scripts/robotgen/pixi.toml scripts/robotgen/generate.sh husky_ur5e

``generate.sh`` writes the robot's sphere model under ``data/robots/husky_ur5e/``, its VAMP
kernel and sweep data under ``include/geodex/integration/vamp/robots/generated/``, and its mass
matrix and lower bound under ``src/robots/generated/``. It ends by comparing them with
Pinocchio on the original description.

3. Connect the kernel
---------------------

Add the robot's name to ``cmake/vamp_robots.cmake``, and write a header
``include/geodex/integration/vamp/robots/<name>.hpp`` that includes the kernel and its sweep
data and names the kernel's model.

.. literalinclude:: ../../include/geodex/integration/vamp/robots/husky_ur5e.hpp
   :language: cpp
   :start-at: #include
   :end-at: }  // namespace geodex::integration::vamp::detail

The Python class of the robot is a model struct and a ``bind_mobile`` or ``bind_fixed`` call in
``python/src/bind_robots.cpp``.

4. Rebuild and list the robots
------------------------------

After a rebuild, ``geodex.robots.available()`` lists the new robot, and it plans like every
other robot.

.. code-pair:: robots/catalog catalog

5. Try it: add a fixed-base arm
-------------------------------

A recipe without ``drive`` builds a fixed-base arm. The recipe of the Franka FR3 with its
Robotiq 2F-85 is ``scripts/robotgen/robots/fr3_arm_gripper.json``. Write a recipe for your arm
the same way and run steps 2 to 4.

Where to go next
----------------

- :doc:`mobile-manipulation` plans a whole-body motion for a generated robot.
- :doc:`/concepts/planning` covers the planner's heuristic, which uses the lower bound.
