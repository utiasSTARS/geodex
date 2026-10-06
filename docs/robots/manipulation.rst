Manipulation
============

In this guide, we plan a Franka FR3 with a Robotiq 2F-85 gripper that moves a box it holds from
one compartment of a shelf to another. We plan the move with the arm's kinetic-energy metric, measure the
path, and then plan it again with the Euclidean metric on the joint angles.

.. robot-scene:: robots-fr3-kinetic-energy
   :hero:
   :alt: An FR3 moving a purple box from the middle compartment of a shelf to the top
         compartment, with faint copies of the arm along its path and the trace of its gripper
         in orange.

   The finished plan. The faint copies show the arm along the path, and the orange curve
   traces the gripper's tool center point from the start (green) to the goal (orange). Drag to
   orbit, and use the controls to pause the arm or move it along the path.

The full script is ``examples/robots/manipulation/arm_ke.py``, and the C++ version is
``examples/robots/manipulation/arm_ke.cpp``.

1. Load the robot and the scene
-------------------------------

``geodex.robots.Fr3Gripper()`` is the arm's configuration space, its seven joint angles within
their limits, with the kinetic-energy metric of its mass matrix. The scene is the shelf scene of
the MoveIt demo in :doc:`/ros2/moveit`, an IKEA BILLY shelf of seven boards that
``geodex.Scene`` holds as boxes. VAMP :footcite:`thomason2024vamp` checks the arm's 60 collision
spheres against them. The gripper holds a 0.04 x 0.24 x 0.16 m box between its two finger
pads 40 mm apart. 112 spheres contain the box and reach at most 1 cm outside it.
``geodex.vamp.attach_spheres`` fixes them to the gripper's tool center point (TCP), with the box's
center 0.055 m along the TCP's z axis, and the check moves them with the arm.

.. code-pair:: robots/manipulation/arm_ke load

It prints ``fr3_arm_gripper joints: 7``. The collision check sees the arm and the box as these
spheres.

.. robot-scene:: robots-fr3-collision
   :spheres:
   :alt: The FR3 moving the box as its blue collision spheres, with the translucent purple spheres
         that contain the box.

   The kinetic-energy plan of step 2 with the arm's collision spheres (blue) and the spheres that
   contain the box (purple).

2. Plan the move
----------------

The start holds the box in the middle compartment and the goal holds it in the top compartment,
as in the MoveIt demo. ``PlanSettings(iterations=1500, seed=1)`` gives the planner 1500
iterations and a fixed seed. Passing the scene as ``collision`` checks every configuration and
every edge with VAMP, the box included.

.. code-pair:: robots/manipulation/arm_ke plan

It prints ``solved=True cost=1.143 waypoints=1597``. The cost is the length of the path under the
kinetic-energy metric. On a laptop CPU (Intel Core i7-10875H), the planner finds a first path
in 0.4 ms, after 7 iterations. The 1500 iterations take 69 ms, and smoothing takes 51 ms.
``result.first_solution_ms``, ``result.time_ms`` and ``result.smooth_ms`` hold these times.

3. Measure the path
-------------------

The returned waypoints are joined by straight lines in joint coordinates. ``measure`` adds up
their length in joint space and their length under the kinetic-energy metric,
:math:`\int \sqrt{\dot q^\top M(q)\, \dot q}\, dt`, by the midpoint rule on each line.

.. code-pair:: robots/manipulation/arm_ke measure

It prints ``joint length=4.918 rad kinetic-energy length=1.143``.

4. Try it: plan with the Euclidean metric
-----------------------------------------

Pass ``metric="euclidean"`` and plan again. The Euclidean metric measures a joint velocity by
:math:`\dot q^\top \dot q`, and the kinetic-energy metric by
:math:`\dot q^\top M(q)\, \dot q`.

.. code-pair:: robots/manipulation/arm_ke try-it

It prints ``joint length=1.688 rad kinetic-energy length=1.263``. The Euclidean path turns the
joints less in total, and the kinetic-energy path is 10 percent shorter under the kinetic-energy
metric. The Euclidean plan finds a first path in 0.2 ms and takes 38 ms with smoothing.

.. robot-scene:: robots-fr3-euclidean
   :alt: The same FR3 move under the Euclidean metric, with the gripper trace in blue.

   The Euclidean plan of the same move, drawn the same way with the trace in blue.

.. plotly-figure:: robots-fr3-joint-travel
   :alt: Grouped bars of the total rotation of each FR3 joint along the two plans.

   How far each joint turns along the two paths. The kinetic-energy plan turns the wrist joints
   6 and 7 by 2.09 and 4.16 rad, against 0.63 and 0.82 rad, and joint 2 by 0.70 rad, against
   0.12 rad.

The mass matrix weighs the joints very differently. At the start, :math:`M(q)` has 1.31, 1.96,
1.50 and 1.16 on the diagonal for joints 1 to 4, and 0.034, 0.066 and 0.001 for the wrist
joints 5 to 7. Turning joint 7 alone at a given speed takes about 1/2000 of the kinetic energy
of turning joint 2 alone. The kinetic-energy plan moves the box with the wrist.

Where to go next
----------------

- Every fixed-base robot of the :doc:`catalog <index>` takes the same calls, for example
  ``geodex.robots.UR5()``. ``geodex.load_scene`` reads a scene of boxes, spheres and cylinders
  from a MotionBenchMaker :footcite:`chamzas2022mbm` YAML file instead.
- :doc:`/ros2/moveit` plans the same move through MoveIt 2 with the geodex planner plugin.
- :doc:`/concepts/metrics` covers the kinetic-energy metric and its lower bound.
- :doc:`/concepts/smoothing` covers the smoother that shortens the planner's path.
- :doc:`mobile-manipulation` puts an arm on a mobile base.
- :ref:`robot-collision-checking` covers how geodex checks a robot against the scene.

References
----------

.. footbibliography::
