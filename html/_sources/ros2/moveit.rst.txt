MoveIt 2 Planner
================

``geodex_moveit`` is a planner plugin for `MoveIt 2 <https://moveit.picknik.ai>`_. It plans joint
paths for fixed-base arms under a Euclidean or a kinetic-energy metric. In this guide, an FR3
moves a box from the middle compartment of a shelf to the top compartment, and we plan the move
under both metrics.

.. video-figure:: ros2-moveit-kinetic-energy
   :width: 75%
   :alt: RViz view of a white FR3 arm with a black Robotiq gripper moving a purple box from the
         middle compartment of a translucent green shelf to the top compartment.

   The FR3 moves the box under the kinetic-energy metric.

1. Build the plugin
-------------------

.. code-block:: bash

   git clone https://github.com/utiasSTARS/geodex_moveit.git
   cd geodex_moveit
   pixi run -e jazzy build

See :doc:`index` for the build in a colcon workspace.

2. Launch the demo
------------------

The launch starts ``move_group`` with the geodex pipeline, the FR3 on mock hardware with the box
in its gripper, the shelf, and RViz.

.. code-block:: bash

   pixi shell -e jazzy
   source install/jazzy/setup.bash
   ros2 launch geodex_moveit_demos fr3_shelf.launch.py mock_hardware:=true rviz:=true

3. Plan and execute
-------------------

In a second shell of the same environment, plan the move under the kinetic-energy metric and
execute it.

.. code-block:: bash

   ros2 run geodex_moveit_demos fr3_shelf_plan.py

It prints the result, and RViz shows the motion of the video at the top.

.. code-block:: text

   kinetic_energy: planning_time=54 ms points=65 duration=3.186 s

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: 54 ms
      :class-card: geodex-stat

      planning time

   .. grid-item-card:: 0.2 ms
      :class-card: geodex-stat

      first solution

   .. grid-item-card:: 1500
      :class-card: geodex-stat

      iterations

The times are from an Intel Core i7-10875H. The script sets a seed and the iteration budget.
Run it again and you get the same trajectory.

4. Try it: plan with the Euclidean metric
-----------------------------------------

.. code-block:: bash

   ros2 run geodex_moveit_demos fr3_shelf_plan.py --metric euclidean

.. code-block:: text

   euclidean: planning_time=27 ms points=42 duration=2.047 s

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: 27 ms
      :class-card: geodex-stat

      planning time

   .. grid-item-card:: 0.1 ms
      :class-card: geodex-stat

      first solution

   .. grid-item-card:: 1500
      :class-card: geodex-stat

      iterations

The Euclidean path is shorter in joint space, 2.45 rad against 5.18 rad, and turns joint 1 at the
base by 1.53 rad. The kinetic-energy path turns joint 1 by 0.11 rad and moves the light wrist
joints instead.

.. video-figure:: ros2-moveit-euclidean
   :width: 75%
   :alt: RViz view of the FR3 moving the purple box from the middle compartment of the green
         shelf to the top compartment along a shorter joint path.

   The same move under the Euclidean metric.

On a real FR3, the joints at the base stay almost still under the kinetic-energy metric, and
they swing the arm under the Euclidean metric.

.. video-figure:: real-fr3-shelf-metrics
   :alt: Two recordings of a real FR3 arm moving a cracker box from the middle compartment of a
         white shelf to the top compartment. On the left, the lower arm stays in place and the
         wrist turns the box. On the right, the whole arm swings from its base.

   The move on a real FR3 under the kinetic-energy metric (left) and the Euclidean metric
   (right).

Switch the metric per robot
---------------------------

``robot`` names a geodex robot model, for example ``panda`` or ``fr3_arm_gripper``, and
``metric`` selects ``euclidean`` or ``kinetic_energy`` for it. With ``robot`` empty, set
``checker: moveit``. The plugin then plans under the Euclidean metric for any group of revolute
and prismatic joints. The FR3 of the demo sets these parameters.

.. literalinclude:: sources/geodex_moveit/geodex_moveit_demos/config/fr3_shelf_geodex.yaml
   :language: yaml
   :start-at: robot:
   :caption: geodex_moveit_demos/config/fr3_shelf_geodex.yaml

The plugin supports only fixed-base arms at the moment. Plan a mobile manipulator with geodex
directly, as in :doc:`/robots/mobile-manipulation`.

Attached objects
----------------

The plugin plans with every object attached in the planning scene. ``end_effector_link`` names
the link that the objects attach to. The vamp checker models an attached object in one of two
ways.

.. plotly-figure:: ros2-moveit-attached-object
   :alt: Two 3D views of the demo's thin orange box. On the left, one translucent blue sphere
         encloses it. On the right, 16 smaller blue spheres cover it closely.

   The demo's 0.04 x 0.24 x 0.16 m box in its enclosing sphere (left) and in a cover of 16
   spheres (right).

.. tab-set::

   .. tab-item:: Enclosing sphere

      The checker encloses the attached shapes in one sphere. This is the default and does not
      need a parameter. Use it for a compact object, or where the motion leaves room around the
      object, as in the demo.

      .. code-block:: yaml

         end_effector_link: 2f85_tcp

   .. tab-item:: Sphere cover

      ``attached_object.spheres`` lists x, y, z and radius per sphere in the frame of the link
      that holds the object. Use a cover where the object passes close to obstacles.

      .. code-block:: yaml

         end_effector_link: 2f85_tcp
         attached_object:
           spheres: [
               -0.000002, -0.100003, -0.002372, 0.036221, -0.000002, -0.100001, 0.106248, 0.040332,
               ...]
         attachment_padding: 0.0012

      `foam <https://github.com/CoMMALab/foam>`_ fits the spheres to a mesh of the object, and
      ``attached_object_spheres.py`` of ``geodex_moveit_demos`` prints this block from foam's
      output.

Collision checking
------------------

``checker: vamp`` checks geodex's sphere model of the robot against the obstacles of the
planning scene. ``checker: moveit`` uses the planning scene's own collision checking, for any
robot. ``obstacle_padding`` is the clearance that planned configurations keep from every
obstacle. Raise it when MoveIt's ``ValidateSolution`` rejects a trajectory.

Parameters
----------

.. include:: generated/moveit_parameters.inc

Where to go next
----------------

- :doc:`/robots/manipulation` plans the box's move between the shelf's compartments with geodex
  directly.
- :doc:`/robots/add-a-robot` adds a robot model the plugin can then serve.
- :doc:`/concepts/smoothing` covers the smoother and what it checks.
- :doc:`nav2` plans base paths through Nav2.
