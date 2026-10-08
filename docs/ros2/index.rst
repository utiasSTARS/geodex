ROS 2
=====

Two ROS 2 planner plugins run geodex inside Nav2 and MoveIt 2. ``geodex_nav2_planner`` is a
global planner for the Nav2 planner server, and ``geodex_moveit`` is a planner for MoveIt 2
planning pipelines. Both load through pluginlib next to the stock planners, take their settings
from ROS parameters, and plan with geodex's G-RRT\* and smoother.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Nav2 global planner
      :img-top: figs/nav2/rviz_jackal.png
      :link: nav2
      :link-type: doc

      Plan a path on :math:`\mathrm{SE}(2)` with ``planner_server`` for a mobile base between
      the desks of an office, and show it in RViz with the global costmap and the robot's
      footprint.

   .. grid-item-card:: MoveIt 2 planner
      :img-top: figs/moveit/moveit_card.gif
      :link: moveit
      :link-type: doc

      Plan a joint path for a fixed-base arm with ``move_group``, under a Euclidean or
      kinetic-energy metric, and execute it on mock hardware.

What each package provides
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - ``geodex_nav2_planner``
     - ``geodex_moveit``
   * - Plugin
     - ``geodex_nav2_planner::GeodexSE2Planner``
     - ``geodex_moveit/GeodexPlanner``
   * - Base class
     - ``nav2_core::GlobalPlanner``
     - ``planning_interface::PlannerManager``
   * - Plans for
     - a base pose :math:`(x, y, \theta)`
     - the active joints of a fixed-base arm
   * - Metric
     - left-invariant metric on :math:`\mathrm{SE}(2)` times a clearance factor
     - Euclidean, or the kinetic energy of a geodex robot model
   * - Collision checking
     - the footprint polygon against an exact distance transform of the costmap
     - geodex's VAMP sphere model of the robot, or the planning scene
   * - Repository
     - `geodex_nav2_planner <https://github.com/utiasSTARS/geodex_nav2_planner>`_,
       Apache-2.0
     - `geodex_moveit <https://github.com/utiasSTARS/geodex_moveit>`_, Apache-2.0

Supported distributions
-----------------------

Both plugins build for the two long-term-support releases of ROS 2.

.. list-table::
   :header-rows: 1
   :widths: 30 25 20 25

   * - Distribution
     - Ubuntu
     - Nav2
     - MoveIt 2
   * - Jazzy Jalisco
     - 24.04
     - 1.3
     - 2.12
   * - Lyrical Luth
     - 26.04
     - 1.5
     - 2.15

Install with pixi
-----------------

`pixi <https://pixi.sh>`_ installs ROS 2 from `RoboStack <https://robostack.github.io>`_,
builds geodex and the OMPL fork that holds G-RRT\*, builds the plugin with colcon, and runs its
tests. Each repository has one pixi environment per distribution, ``jazzy`` and ``lyrical``, for
Linux x86-64 and macOS arm64.

1. Clone the repository and run the tests.

   .. tab-set::

      .. tab-item:: Nav2 planner

         .. literalinclude:: sources/geodex_nav2_planner/README.md
            :language: bash
            :start-at: git clone https://github.com/utiasSTARS/geodex_nav2_planner
            :end-at: cd geodex_nav2_planner
            :dedent: 3

         .. literalinclude:: sources/geodex_nav2_planner/README.md
            :language: bash
            :start-at: pixi run -e jazzy test
            :end-at: pixi run -e jazzy test
            :dedent: 3

      .. tab-item:: MoveIt planner

         .. literalinclude:: sources/geodex_moveit/README.md
            :language: bash
            :start-at: git clone https://github.com/utiasSTARS/geodex_moveit
            :end-at: cd geodex_moveit
            :dedent: 3

         .. literalinclude:: sources/geodex_moveit/README.md
            :language: bash
            :start-at: pixi run -e jazzy test
            :end-at: pixi run -e jazzy test
            :dedent: 3

2. Open a shell in the environment and source the install space.

   .. code-block:: bash

      pixi shell -e jazzy
      source install/jazzy/setup.bash

``GEODEX_SOURCE_DIR=/path/to/geodex`` in front of a ``pixi run`` command builds against a local
geodex clone instead of the geodex release the plugin uses by default. A pixi build of the
plugin runs only inside its pixi environment.

Install into a colcon workspace
-------------------------------

On a machine with ROS 2 from apt, build geodex, the OMPL fork and, for MoveIt, VAMP into a
prefix of their own, then build the plugin in your workspace against that prefix. Run the
commands from the plugin's clone, for example under ``~/ros2_ws/src``.

.. tab-set::

   .. tab-item:: Nav2 planner

      .. literalinclude:: sources/geodex_nav2_planner/README.md
         :language: bash
         :start-at: bash scripts/build_deps.sh
         :end-at: bash scripts/build_deps.sh
         :dedent: 3

      .. literalinclude:: sources/geodex_nav2_planner/README.md
         :language: bash
         :start-at: rosdep install --from-paths
         :end-at: rosdep install --from-paths
         :dedent: 3

      .. literalinclude:: sources/geodex_nav2_planner/README.md
         :language: bash
         :start-at: colcon build --packages-select
         :end-at: -DCMAKE_PREFIX_PATH=/opt/geodex_deps
         :dedent: 3

   .. tab-item:: MoveIt planner

      .. literalinclude:: sources/geodex_moveit/README.md
         :language: bash
         :start-at: bash scripts/build_deps.sh
         :end-at: bash scripts/build_deps.sh
         :dedent: 3

      .. literalinclude:: sources/geodex_moveit/README.md
         :language: bash
         :start-at: rosdep install --from-paths
         :end-at: rosdep install --from-paths
         :dedent: 3

      .. literalinclude:: sources/geodex_moveit/README.md
         :language: bash
         :start-at: colcon build --packages-select
         :end-at: -DCMAKE_PREFIX_PATH=/opt/geodex_deps
         :dedent: 3

.. toctree::
   :hidden:

   nav2
   moveit
