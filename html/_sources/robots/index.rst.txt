Robot Guides
============

geodex includes nine robots, from fixed-base arms to mobile manipulators. Each robot comes as a
configuration space with its joint limits, a kinetic-energy metric from its mass matrix, a
lower bound of that metric for the planner's heuristic, and a VAMP collision model.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Manipulation
      :link: manipulation
      :link-type: doc

      Plan an FR3 that moves a held box between two shelf compartments, under the Euclidean and the
      kinetic-energy metric, with VAMP collision checking.

   .. grid-item-card:: Navigation
      :link: navigation
      :link-type: doc

      Plan three Clearpath mobile robots between the desks of an office, where the drive changes only
      the weights of the :math:`\mathrm{SE}(2)` metric.

   .. grid-item-card:: Whole-body planning
      :link: mobile-manipulation
      :link-type: doc

      Plan the base and the arm together for a Stretch 3, a Stretch 4, and a UR5e on a Husky
      and on a Ridgeback.

   .. grid-item-card:: Add a robot
      :link: add-a-robot
      :link-type: doc

      Turn a URDF into collision spheres, a VAMP model, a generated mass matrix and its
      lower bound.

The catalog
-----------

Every robot is a Python class in ``geodex.robots`` and a value of the
:cpp:enum:`geodex::robots::Robot` enumeration in C++. The configuration of a mobile robot is its
base pose :math:`(x, y, \theta)` followed by its joint values. Its metric adds a left-invariant
metric on the base pose, holonomic or differential drive, to the kinetic-energy metric of its
joints. The mass matrix of every robot is C++ code generated from its description with
Pinocchio's Composite Rigid Body Algorithm (CRBA), and planning does not need a dynamics
library.

The snippet lists the robots and the number of coordinates of each.

.. code-pair:: robots/catalog catalog

.. list-table::
   :header-rows: 1
   :widths: 34 22 14 18 12

   * - Robot
     - Python class
     - Coordinates
     - Base
     - VAMP spheres
   * - Franka Panda
     - ``Panda``
     - 7
     - fixed
     - 59
   * - Franka FR3 with a Robotiq 2F-85
     - ``Fr3Gripper``
     - 7
     - fixed
     - 60
   * - Universal Robots UR5
     - ``UR5``
     - 6
     - fixed
     - 40
   * - Rethink Baxter, both arms
     - ``Baxter``
     - 14
     - fixed
     - 75
   * - Willow Garage PR2, both arms
     - ``PR2``
     - 14
     - fixed
     - 77
   * - Hello Robot Stretch 3
     - ``Stretch3``
     - 8
     - differential drive
     - 148
   * - Hello Robot Stretch 4
     - ``Stretch4``
     - 8
     - holonomic, three omniwheels
     - 160
   * - Clearpath Ridgeback with a UR5e
     - ``RidgebackUR5e``
     - 9
     - holonomic, mecanum
     - 114
   * - Clearpath Husky A200 with a UR5e
     - ``HuskyUR5e``
     - 9
     - differential drive, skid steer
     - 130

.. _robot-collision-checking:

Collision checking
------------------

The planner and the smoother check the robot as a set of spheres, its VAMP collision model,
against the obstacles of the scene and against the robot itself. The spheres of a body attached
with ``geodex.vamp.attach_spheres`` move with the robot and are checked the same way.

.. robot-scene:: robots-gallery-panda
   :spheres:
   :aspect: 6/5
   :alt: A Franka Panda arm with the 59 collision spheres of its VAMP model drawn over its
         meshes.

   The Panda with the 59 spheres its VAMP model checks. The Spheres button hides them.

Gallery
-------

Each robot below is drawn with the meshes of its description and repeats a short motion of
its joints. Drag to orbit, and use the controls to pause the robot or move it along the
motion. The Spheres button shows the spheres of the robot's VAMP collision model over its
meshes.

.. tab-set::

   .. tab-item:: Panda

      .. robot-scene:: robots-gallery-panda
         :spheres: hidden
         :alt: A Franka Panda arm with its hand.
         :aspect: 6/5

   .. tab-item:: FR3

      .. robot-scene:: robots-gallery-fr3-arm-gripper
         :spheres: hidden
         :alt: A Franka FR3 arm with a Robotiq 2F-85 gripper.
         :aspect: 6/5

   .. tab-item:: UR5

      .. robot-scene:: robots-gallery-ur5
         :spheres: hidden
         :alt: A Universal Robots UR5 arm with a Robotiq 2F-85 gripper.
         :aspect: 6/5

   .. tab-item:: Baxter

      .. robot-scene:: robots-gallery-baxter
         :spheres: hidden
         :alt: A Rethink Baxter with two arms on its pedestal.
         :aspect: 6/5

   .. tab-item:: PR2

      .. robot-scene:: robots-gallery-pr2
         :spheres: hidden
         :alt: A Willow Garage PR2 with two arms.
         :aspect: 6/5

   .. tab-item:: Stretch 3

      .. robot-scene:: robots-gallery-stretch3
         :spheres: hidden
         :alt: A Hello Robot Stretch 3.
         :aspect: 6/5

   .. tab-item:: Stretch 4

      .. robot-scene:: robots-gallery-stretch4
         :spheres: hidden
         :alt: A Hello Robot Stretch 4.
         :aspect: 6/5

   .. tab-item:: Ridgeback

      .. robot-scene:: robots-gallery-ridgeback-ur5e
         :spheres: hidden
         :alt: A Clearpath Ridgeback with a UR5e arm.
         :aspect: 6/5

   .. tab-item:: Husky

      .. robot-scene:: robots-gallery-husky-ur5e
         :spheres: hidden
         :alt: A Clearpath Husky with a UR5e arm.
         :aspect: 6/5

.. toctree::
   :hidden:

   manipulation
   navigation
   mobile-manipulation
   add-a-robot
