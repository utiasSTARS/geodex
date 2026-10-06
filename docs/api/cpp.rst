C++
===

The C++ reference has one section per public header, with every class, concept, function,
type and constant the header declares and its documentation. The umbrella header
``geodex/geodex.hpp`` includes the header-only core (manifolds, metrics, algorithms,
heuristics and collision helpers). Include the planning, robot and integration headers one
by one. Each needs the libraries named in its section.

Namespaces follow the layout of the headers. The core lives in ``geodex``, the algorithms in
``geodex::algorithm``, the planar collision helpers in ``geodex::collision``, the heuristics in
``geodex::heuristics``, planning in ``geodex::planning``, the built-in robots in
``geodex::robots``, and the integrations in ``geodex::integration::{ompl, vamp, pinocchio}``.
Names in a ``detail`` namespace are implementation details and are left out.

.. contents:: On this page
   :local:
   :depth: 1

Core concepts
-------------

Every manifold, metric, retraction and sampler satisfies the concepts below. The algorithms
use the geodesic distance and interpolation interfaces of this section.

Manifold concepts
^^^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/core/concepts.hpp.inc

Metric concepts and frozen metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/core/metric.hpp.inc

Retractions
^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/core/retraction.hpp.inc

Distance
^^^^^^^^

.. include:: ../../build/docs/api/cpp/core/distance.hpp.inc

Interpolation
^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/core/interpolation.hpp.inc

Samplers
^^^^^^^^

.. include:: ../../build/docs/api/cpp/core/sampler.hpp.inc

Manifolds
---------

Each manifold is a class template over its metric, retraction and sampler policies.

Euclidean space
^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/manifold/euclidean.hpp.inc

Sphere
^^^^^^

.. include:: ../../build/docs/api/cpp/manifold/sphere.hpp.inc

Torus
^^^^^

.. include:: ../../build/docs/api/cpp/manifold/torus.hpp.inc

SO(2)
^^^^^

.. include:: ../../build/docs/api/cpp/manifold/so2.hpp.inc

SO(3)
^^^^^

.. include:: ../../build/docs/api/cpp/manifold/so3.hpp.inc

SE(2)
^^^^^

.. include:: ../../build/docs/api/cpp/manifold/se2.hpp.inc

SE(3)
^^^^^

.. include:: ../../build/docs/api/cpp/manifold/se3.hpp.inc

Product manifold
^^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/manifold/product.hpp.inc

Configuration space
^^^^^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/manifold/configuration_space.hpp.inc

Metrics
-------

Identity
^^^^^^^^

.. include:: ../../build/docs/api/cpp/metrics/identity.hpp.inc

Constant SPD
^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/metrics/constant_spd.hpp.inc

Weighted
^^^^^^^^

.. include:: ../../build/docs/api/cpp/metrics/weighted.hpp.inc

Affine combination
^^^^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/metrics/affine_combined.hpp.inc

Kinetic energy
^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/metrics/kinetic_energy.hpp.inc

Jacobi
^^^^^^

.. include:: ../../build/docs/api/cpp/metrics/jacobi.hpp.inc

Pullback
^^^^^^^^

.. include:: ../../build/docs/api/cpp/metrics/pullback.hpp.inc

Clearance
^^^^^^^^^

``SDFConformalMetric`` is the metric Python calls ``ClearanceMetric``.

.. include:: ../../build/docs/api/cpp/metrics/clearance.hpp.inc

SE(2) left-invariant
^^^^^^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/metrics/se2_left_invariant.hpp.inc

SE(3) invariant
^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/metrics/se3_invariant.hpp.inc

SO(2) canonical
^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/metrics/so2_canonical.hpp.inc

SO(3) canonical
^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/metrics/so3_canonical.hpp.inc

Algorithms
----------

Geodesic distance
^^^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/algorithm/distance.hpp.inc

Discrete geodesic interpolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/algorithm/interpolation.hpp.inc

Path smoothing
^^^^^^^^^^^^^^

The smoother runs on its own or inside ``planning::plan``. See :doc:`/concepts/smoothing`.

.. include:: ../../build/docs/api/cpp/algorithm/path_smoothing.hpp.inc

Matrix lower bound
^^^^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/algorithm/precompute_matrix_lower_bound.hpp.inc

Heuristics
----------

The heuristics below are admissible lower bounds on the geodesic distance between two
configurations. ``planning::plan`` takes one as its ``heuristic`` argument, and
``heuristics/heuristics.hpp`` includes all of them.

Traits
^^^^^^

.. include:: ../../build/docs/api/cpp/heuristics/traits.hpp.inc

Zero
^^^^

.. include:: ../../build/docs/api/cpp/heuristics/zero.hpp.inc

Euclidean
^^^^^^^^^

.. include:: ../../build/docs/api/cpp/heuristics/euclidean.hpp.inc

Eigenvalue lower bound
^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/heuristics/eigenvalue_lower_bound.hpp.inc

Matrix lower bound
^^^^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/heuristics/matrix_lower_bound.hpp.inc

Product of factor bounds
^^^^^^^^^^^^^^^^^^^^^^^^

``product_lower_bound`` combines the bounds of the factors of a ``ProductManifold``, such as an
SE(2) base and a robot's joint space.

.. include:: ../../build/docs/api/cpp/heuristics/product_lower_bound.hpp.inc

Planning
--------

``planning::plan`` turns a manifold, a start, a goal and a validity function into a smoothed,
collision-free path in a single planning call. The header also declares the settings, the
result and the log level of the planners. It needs a build with the OMPL fork that holds G-RRT\*
(``GEODEX_OMPL``). See :doc:`/concepts/planning`.

.. include:: ../../build/docs/api/cpp/planning/plan.hpp.inc

Robots
------

The robot headers declare the built-in robots, their generated mass matrices and lower
bounds, their joint spaces, and ``robots::plan``, which plans in a VAMP scene with a single
call. They need the robot library (``GEODEX_ROBOTS``), and ``robots/planning.hpp`` also needs
the OMPL fork and VAMP. See :doc:`/robots/index`.

Robot registry and mass matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/robots/mass_matrix.hpp.inc

Mass lower bounds
^^^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/robots/mass_lower_bound.hpp.inc

Joint spaces
^^^^^^^^^^^^

A mobile manipulator plans on ``make_product`` of an ``SE2`` base and ``joint_space<R>()``.

.. include:: ../../build/docs/api/cpp/robots/joint_space.hpp.inc

Robot planning
^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/robots/planning.hpp.inc

Planar collision
----------------

The collision headers declare distance grids, footprints and signed distance functions for
planar robots. ``collision/collision.hpp`` includes all of them.

Distance grid
^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/collision/distance_grid.hpp.inc

Polygon footprint
^^^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/collision/polygon_footprint.hpp.inc

Footprint checker
^^^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/collision/footprint_grid_checker.hpp.inc

Circle obstacles
^^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/collision/circle_sdf.hpp.inc

Rectangle obstacles
^^^^^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/collision/rectangle_sdf.hpp.inc

OMPL integration
----------------

These headers declare the state space, sampler, objective and validators that
``planning::plan`` assembles, for callers that run OMPL themselves, as the Nav2 and MoveIt
plugins do. They need the OMPL fork.

State space
^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/integration/ompl/geodex_state_space.hpp.inc

Informed sampler
^^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/integration/ompl/geodex_informed_sampler.hpp.inc

Optimization objective
^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/integration/ompl/geodex_optimization_objective.hpp.inc

Cost bound feedback
^^^^^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/integration/ompl/cost_bound_feedback.hpp.inc

Validity checker
^^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/integration/ompl/validity_checker.hpp.inc

Directional motion validator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/integration/ompl/directional_motion_validator.hpp.inc

VAMP integration
----------------

These headers declare scenes, per-robot collision checkers and motion validators, and the
batched validity check that the smoother uses. They need a build with VAMP (``GEODEX_VAMP``).

Registry and scenes
^^^^^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/integration/vamp/registry.hpp.inc

Batched validity
^^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/integration/vamp/validity.hpp.inc

Pinocchio integration
---------------------

These headers declare the mass matrices, Jacobians and pullback metrics that Pinocchio computes
at run time from a URDF, for robots outside the built-in catalog. They need a build with Pinocchio
(``GEODEX_PINOCCHIO``).

Mass matrix
^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/integration/pinocchio/mass_matrix.hpp.inc

Jacobian
^^^^^^^^

.. include:: ../../build/docs/api/cpp/integration/pinocchio/jacobian.hpp.inc

Pullback
^^^^^^^^

.. include:: ../../build/docs/api/cpp/integration/pinocchio/pullback.hpp.inc

Utilities
---------

Angles
^^^^^^

.. include:: ../../build/docs/api/cpp/utils/angle.hpp.inc

Lie group helpers
^^^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/utils/lie.hpp.inc

Numerics
^^^^^^^^

.. include:: ../../build/docs/api/cpp/utils/math.hpp.inc

Normal quantile
^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/utils/normal.hpp.inc

Portable random values
^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/utils/random.hpp.inc

Ordered sums
^^^^^^^^^^^^

.. include:: ../../build/docs/api/cpp/utils/ordered_sum.hpp.inc

See also
--------

- :doc:`/api/python` for the same capabilities from Python.
- :doc:`/concepts/architecture` for how the pieces fit together.
