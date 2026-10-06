Python
======

In the Python module ``geodex``, points, tangent vectors and paths are NumPy ``float64``
arrays. See :doc:`/tutorials/geodex-basics` for a walk through the main functions, and
:doc:`index` for the C++ name of each entry point.

.. note::

   On Linux and macOS, the wheel includes everything on this page. On Windows and on Linux
   with musl, it includes the geometry core but not planning (:py:func:`geodex.plan`), the
   built-in robots (``geodex.robots``) or the VAMP collision scenes
   (:py:class:`geodex.Scene`, ``geodex.vamp``). See :doc:`/getting-started/installation`.

.. py:currentmodule:: geodex

.. contents:: On this page
   :local:
   :depth: 1

Manifolds
---------

A manifold bundles a topology (how points and tangent vectors are represented and how
``exp`` and ``log`` move between them) with a Riemannian metric. Points are arrays of the
manifold's ambient representation, which can exceed its intrinsic dimension, as on the
sphere. Every manifold offers the same core methods.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Description
   * - ``dim()``
     - Intrinsic dimension.
   * - ``random_point()``
     - A point from the manifold's sampler.
   * - ``exp(p, v)``, ``log(p, q)``
     - Exponential map (or retraction) at ``p`` and its inverse.
   * - ``inner(p, u, v)``, ``norm(p, v)``
     - Riemannian inner product and norm at ``p``.
   * - ``distance(p, q)``
     - Geodesic distance.
   * - ``geodesic(p, q, t)``
     - The point a fraction ``t`` of the way along the geodesic from ``p`` to ``q``.
   * - ``seed(seed)``, ``set_sampler(name)``
     - Reseed the sampler, or switch it to ``'scrambled'``, ``'halton'`` or ``'random'``.

.. autoclass:: geodex.Euclidean
   :members:
.. autoclass:: geodex.Sphere
   :members:
.. autoclass:: geodex.SphereN
   :members:
.. autoclass:: geodex.Torus
   :members:
.. autoclass:: geodex.SO2
   :members:
.. autoclass:: geodex.SO3
   :members:
.. autoclass:: geodex.SE2
   :members:
.. autoclass:: geodex.SE3
   :members:
.. autoclass:: geodex.Product
   :members:
.. autoclass:: geodex.ConfigurationSpace
   :members:

Metrics
-------

A metric supplies the geometry a :py:class:`ConfigurationSpace` combines with a base
manifold's topology. The mass-matrix and pullback metrics take Python callables that return
the relevant matrix at a configuration. See :doc:`/concepts/metrics`.

.. autoclass:: geodex.ConstantSPDMetric
   :members:
.. autoclass:: geodex.SE2LeftInvariantMetric
   :members:
.. autoclass:: geodex.KineticEnergyMetric
   :members:
.. autoclass:: geodex.JacobiMetric
   :members:
.. autoclass:: geodex.PullbackMetric
   :members:
.. autoclass:: geodex.WeightedMetric
   :members:
.. autoclass:: geodex.AffineCombinedMetric
   :members:
.. autoclass:: geodex.ClearanceMetric
   :members:

Sampling
--------

Every manifold samples its random points with a sampler, scrambled Halton by default.
:py:func:`seed` reseeds the global source of the samplers of manifolds built afterwards. A
plan's own seed is :py:attr:`PlanSettings.seed`. See :doc:`/concepts/sampling`.

.. autofunction:: geodex.seed
.. autoclass:: geodex.ScrambledHaltonSampler
   :members:
.. autoclass:: geodex.HaltonSampler
   :members:
.. autoclass:: geodex.PseudoRandomSampler
   :members:

Geodesics and distance
----------------------

These functions work on any manifold, a :py:class:`ConfigurationSpace` included, and return
the path with convergence diagnostics. See :doc:`/concepts/discrete-geodesic-interpolation`.

.. autofunction:: geodex.distance_midpoint
.. autofunction:: geodex.discrete_geodesic
.. autoclass:: geodex.InterpolationSettings
   :members:
.. autoclass:: geodex.InterpolationResult
   :members:
.. autoclass:: geodex.InterpolationStatus
   :members:
   :undoc-members:

Planning
--------

:py:func:`plan` plans a collision-free path that is short under the space's metric. It needs a
manifold, a start, a goal and optionally a validity function or a scene. A nonzero
:py:attr:`PlanSettings.seed` with an ``iterations`` budget makes the plan reproducible.
:py:attr:`PlanSettings.limits` sets the physical limits of the coordinates. See
:doc:`/concepts/planning`.

.. autofunction:: geodex.plan
.. autoclass:: geodex.PlanSettings
   :members:
.. autoclass:: geodex.PlanResult
   :members:
.. autoclass:: geodex.planners.GreedyRRTstar
   :members:
.. autoclass:: geodex.DirectionalMotionValidator
   :members:

The planners print through OMPL's logger. :py:func:`set_log_level` sets how much they print,
warnings and errors by default.

.. autofunction:: geodex.set_log_level
.. autofunction:: geodex.log_level
.. autoclass:: geodex.LogLevel
   :members:
   :undoc-members:

Smoothing
---------

:py:func:`smooth_path` shortens a valid path under the manifold's metric, rounds its corners
into :math:`C^2` curves and checks every waypoint and edge it returns against the validity
function.
:py:func:`plan` runs it on every path unless :py:attr:`PlanSettings.smooth` is off. See
:doc:`/concepts/smoothing`.

.. autofunction:: geodex.smooth_path
.. autoclass:: geodex.PathSmoothingSettings
   :members:
.. autoclass:: geodex.PathSmoothingResult
   :members:
.. autoclass:: geodex.PathSmoothingProfile
   :members:

Heuristics
----------

The heuristics below are admissible lower bounds on the geodesic distance between two
configurations. :py:func:`plan` takes one through ``heuristic``.
:py:func:`precompute_matrix_lower_bound` computes a constant matrix below a metric at every
point of a box, the bound :py:class:`geodex.heuristics.MatrixLowerBound` takes.
:py:func:`geodex.heuristics.product_lower_bound` combines the lower bounds of the factors of a
:py:class:`Product` into one heuristic. :py:meth:`SE2LeftInvariantMetric.coordinate_lower_bound`
gives the bound of an :math:`\mathrm{SE}(2)` base.

.. autoclass:: geodex.heuristics.Zero
   :members:
.. autoclass:: geodex.heuristics.Euclidean
   :members:
.. autoclass:: geodex.heuristics.EigenvalueLowerBound
   :members:
.. autoclass:: geodex.heuristics.MatrixLowerBound
   :members:
.. autofunction:: geodex.heuristics.product_lower_bound
.. autofunction:: geodex.precompute_matrix_lower_bound
.. autoclass:: geodex.PrecomputeMatrixLowerBoundSettings
   :members:
.. autoclass:: geodex.PrecomputeMatrixLowerBoundResult
   :members:

Robots
------

The built-in robots are configuration spaces with their joint limits, a kinetic-energy or
Euclidean metric on the arm, a generated mass matrix and a lower bound of it. A mobile
robot plans on :math:`\mathrm{SE}(2) \times \mathbb{R}^n` and takes its base metric and the
region its base samples. Passing a robot and a scene to :py:func:`plan` checks collisions
with the robot's VAMP model. See :doc:`/robots/index`.

.. autofunction:: geodex.robots.available
.. autoclass:: geodex.robots.RobotModel
   :members:
.. autoclass:: geodex.robots.Panda
.. autoclass:: geodex.robots.UR5
.. autoclass:: geodex.robots.Baxter
.. autoclass:: geodex.robots.PR2
.. autoclass:: geodex.robots.Fr3Gripper
.. autoclass:: geodex.robots.Stretch3
.. autoclass:: geodex.robots.Stretch4
.. autoclass:: geodex.robots.RidgebackUR5e
.. autoclass:: geodex.robots.HuskyUR5e

Collision scenes
----------------

A scene holds the obstacles VAMP checks a robot against, loaded from a MotionBenchMaker YAML
file or built in memory from boxes, spheres and cylinders. Sizes are full extents and
orientations are scalar-last quaternions ``[qx, qy, qz, qw]``. ``geodex.vamp`` gives direct
access to the per-robot checkers.

.. autofunction:: geodex.load_scene
.. autoclass:: geodex.Scene
   :members:
.. autofunction:: geodex.vamp.load_scene
.. autofunction:: geodex.vamp.make_vamp_checker
.. autoclass:: geodex.vamp.CollisionChecker
   :members:
.. autoclass:: geodex.vamp.EnvHandle
   :members:
.. autofunction:: geodex.vamp.attach_spheres
.. autofunction:: geodex.vamp.pad_scene
.. autofunction:: geodex.vamp.sphere_speed
.. autofunction:: geodex.vamp.registered_robots
.. autofunction:: geodex.vamp.robot_dimension
.. autofunction:: geodex.vamp.robot_joint_names
.. autofunction:: geodex.vamp.robot_end_effector
.. autofunction:: geodex.vamp.robot_spheres

Planar collision
----------------

``geodex.collision`` holds distance grids, footprints and signed distance functions for planar
robots. A :py:class:`~geodex.collision.FootprintGridChecker` is both a validity
test and the distance field of a :py:class:`ClearanceMetric`. See :doc:`/robots/navigation`.

.. autoclass:: geodex.collision.DistanceGrid
   :members:
.. autoclass:: geodex.collision.GridSDF
   :members:
.. autoclass:: geodex.collision.InflatedSDF
   :members:
.. autoclass:: geodex.collision.MemoizedSDF
   :members:
.. autoclass:: geodex.collision.PolygonFootprint
   :members:
.. autoclass:: geodex.collision.FootprintGridChecker
   :members:
.. autoclass:: geodex.collision.CircleSDF
   :members:
.. autoclass:: geodex.collision.CircleSmoothSDF
   :members:
.. autoclass:: geodex.collision.RectObstacle
   :members:
.. autoclass:: geodex.collision.RectSmoothSDF
   :members:
.. autofunction:: geodex.collision.rects_overlap

See also
--------

- :doc:`cpp` for the C++ names of everything on this page.
- :doc:`/getting-started/installation` for what the wheel of each platform includes.
