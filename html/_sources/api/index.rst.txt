API Reference
=============

The C++ core is header-only apart from the robot library and the integrations. The table below
gives the Python and C++ entry points of each capability. See :doc:`python` for every public
name of the module and :doc:`cpp` for every public header.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Capability
     - Python
     - C++
   * - Manifolds
     - ``geodex.Sphere``, ``geodex.SE2``, ...
     - ``geodex::Sphere``, ``geodex::SE2``, ...
   * - Custom metric on a manifold
     - ``geodex.ConfigurationSpace``
     - ``geodex::ConfigurationSpace``
   * - Clearance metric
     - ``geodex.ClearanceMetric``
     - ``geodex::SDFConformalMetric``
   * - Planning
     - ``geodex.plan``, ``geodex.PlanSettings``
     - ``geodex::planning::plan``, ``geodex::planning::PlanSettings``
   * - Smoothing
     - ``geodex.smooth_path``
     - ``geodex::algorithm::smooth_path``
   * - Planner log level
     - ``geodex.set_log_level``
     - ``geodex::planning::set_log_level``
   * - Built-in robots
     - ``geodex.robots.Panda()``, ...
     - ``geodex::robots::Robot::Panda``, ...
   * - Planning for a robot in a scene
     - ``geodex.plan(robot, q0, q1, collision=scene)``
     - ``geodex::robots::plan<R>(q0, q1, env)``
   * - Whole-body spaces
     - ``geodex.robots.Stretch4(base=...)``
     - ``geodex::make_product(SE2, geodex::robots::joint_space<R>())``
   * - Heuristic of a product
     - ``geodex.heuristics.product_lower_bound``
     - ``geodex::heuristics::product_lower_bound``
   * - Collision scenes
     - ``geodex.load_scene``, ``geodex.Scene``
     - ``geodex::integration::vamp::load_scene``
   * - Planar collision
     - ``geodex.collision``
     - ``geodex::collision``

.. toctree::
   :maxdepth: 2

   python
   cpp
