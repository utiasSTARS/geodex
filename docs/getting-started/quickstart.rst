Quickstart
==========

On this page, we plan a first motion for a Franka Panda. The arm swings from one side of a post
to the other, and geodex returns a collision-free path that is short under the arm's kinetic
energy. Planning needs the Linux or macOS wheel (see :doc:`installation`).

.. robot-scene:: quickstart-panda
   :hero:
   :alt: A Panda arm swinging around a grey post, with the path of its grasp point drawn in
         blue.

   The finished plan. The blue curve traces the grasp point between the fingers from the
   start (green) to the goal (orange). Drag to orbit, and use the controls to pause the arm or
   move it along the path.

The full script is ``examples/getting_started/quickstart.py``, and the C++ version is
``examples/getting_started/quickstart.cpp``.

1. Load the robot
-----------------

``geodex.robots.Panda()`` is the arm's configuration space, its seven joint angles within
their limits. Its metric is the arm's **kinetic energy**, built from the Panda's mass matrix.

.. code-pair:: getting_started/quickstart load

It prints ``panda joints: 7``.

2. Build the scene
------------------

A ``Scene`` holds the obstacles. Here, it is a post in front of the arm, a box with side lengths
of 10 cm, 10 cm and 80 cm.

.. code-pair:: getting_started/quickstart scene

3. Plan the motion
------------------

The start and the goal differ only in the first joint, the one at the base. The straight line
between them in joint space would swing the arm through the post. ``plan`` checks every
configuration and every edge against the scene with VAMP, a collision checker that tests the
arm's collision spheres. ``PlanSettings(iterations=1500, seed=1)`` gives the planner 1500
iterations and a fixed seed. Run the script again and you get the same path.

.. code-pair:: getting_started/quickstart plan

It prints ``solved=True cost=1.5543 waypoints=105``.

4. Read the result
------------------

``result.path`` is an array of joint configurations, one row per waypoint, joined by straight
lines in joint space. ``result.cost`` is its length under the kinetic-energy metric, and
``result.raw_path`` holds the planner's waypoints before the smoother shortened them.

.. code-pair:: getting_started/quickstart result

It prints ``path: 105 x 7, raw path: 5 x 7``.

5. Try it: plan with the Euclidean metric
-----------------------------------------

Pass ``metric="euclidean"`` and plan again. The Euclidean metric measures a motion by the joint
angles alone.

.. code-pair:: getting_started/quickstart try-it

It prints

.. code-block:: text

   kinetic energy: the joints turn 2.396 rad in total
   euclidean: the joints turn 2.205 rad in total

The Euclidean path turns the joints less in total. See :doc:`/robots/manipulation` for an FR3
arm planned under each metric, with each path measured in joint space and under the
kinetic-energy metric.

Where to go next
----------------

- ``geodex.robots.available()`` lists every built-in robot, and the same calls plan for each.
  See :doc:`/robots/index` for a guide per robot.
- :doc:`/concepts/metrics` covers the kinetic-energy metric and the other metrics geodex provides.
- :doc:`/concepts/planning` and :doc:`/concepts/smoothing` cover the planner and the smoother.
- :doc:`reproducibility` covers seeds and budgets.
- ``pixi run quickstart`` runs this example in a checkout of the repository. See
  ``examples/README.md`` for the other examples.
