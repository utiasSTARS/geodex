Metrics as Robot Models
=======================

A planner in geodex looks for the shortest path, and the **Riemannian metric** defines its
length. At each configuration :math:`q`, the metric assigns a velocity :math:`\dot q` the cost
:math:`\|\dot q\|_q = \sqrt{\dot q^\top G(q)\, \dot q}`, and the length of a path is the
integral of that cost along it. The metric encodes which motions are expensive for the robot,
such as moving sideways, swinging a heavy link or passing close to a wall. A new metric models a
new robot, while the planner, the collision checks and the smoother stay the same. This page
covers the metrics geodex provides and the robots they model.

In C++, a metric is a policy of the manifold type. In Python, a metric is an object that
``ConfigurationSpace`` combines with a base manifold. Both evaluate ``inner(q, u, v)`` and
``norm(q, v)``, and every algorithm in geodex measures lengths through them.

Mobile bases on SE(2)
---------------------

A planar robot's pose is :math:`(x, y, \theta)`, a point of :math:`\mathrm{SE}(2)`, and its
velocity in the robot's own frame is :math:`(v_x, v_y, \omega)`, the forward, sideways and
turning rates. A **left-invariant metric** weighs these three body-frame rates,

.. math::

   \|\dot q\|^2 = w_x v_x^2 + w_y v_y^2 + w_\theta\, \omega^2,

and gives the same cost to the same maneuver wherever and in whichever direction the robot
starts. Each weight multiplies a squared rate, and a unit of motion at a weight :math:`w` costs
:math:`\sqrt{w}`.

.. list-table::
   :header-rows: 1
   :widths: 10 10 10 30 40

   * - :math:`w_x`
     - :math:`w_y`
     - :math:`w_\theta`
     - Drive
     - Reading
   * - 1
     - 1
     - 0.5
     - Holonomic (mecanum, omni wheels)
     - Sideways motion costs as much as forward motion.
   * - 1
     - 100
     - 1
     - Differential drive, skid steer
     - A meter of sliding costs as much as ten meters of driving forward.
   * - 1
     - 20
     - :math:`r^2`
     - Car-like, turning radius :math:`r`
     - A turn on the spot through an angle :math:`\varphi` costs :math:`r \varphi`, as much
       as driving :math:`r \varphi` straight ahead.

.. code-pair:: concepts/metrics se2-weights

It prints the cost of a unit forward, sideways and turning speed under each metric, for
example ``differential drive forward 1.000  sideways 10.000  turn 1.000``.

The explorer below colors every position of a 6 m square by the metric length of the
constant-twist motion from the robot to that position. This length is the norm of the
:math:`\mathrm{SE}(2)` logarithm, the cost that ``plan`` gives an edge between the two poses.
Each position takes its cheapest heading on arrival. Under the holonomic weights, the cheap
region is round. A larger :math:`w_y` stretches it along the robot's heading and pinches it
sideways, where a turn, a drive and a turn back become cheaper than sliding.

.. raw:: html
   :file: ../_static/se2-metric-explorer.html

A large lateral weight approximates a nonholonomic constraint without imposing it. Shortest
paths avoid sideways motion but may still slide a little where sliding is cheaper than a
detour. A robot that must never drive backward adds that constraint as a hard edge check with
``DirectionalMotionValidator`` (see :doc:`planning`).

Arms and the kinetic-energy metric
----------------------------------

For an articulated robot with the joint-space **mass matrix** :math:`M(q)`, the
kinetic-energy metric gives a joint velocity the cost

.. math::

   \|\dot q\|_q^2 = \dot q^\top M(q)\, \dot q,

twice the kinetic energy of the motion. A shortest path under this metric, traversed at
constant speed, has the smallest integral of kinetic energy among motions of the same
duration. The same joint speed costs more with the arm stretched out, where the shoulder swings
every link at full radius.

.. code-pair:: concepts/metrics kinetic-energy

It prints ``stretched: shoulder speed costs 1.633`` and ``folded: shoulder speed costs 0.851``.

.. video-figure:: metrics-arm-shoulder
   :width: 90%
   :alt: Two planar two-link arms turn their shoulders back and forth at the same rate. The
         stretched arm on the left sweeps its second link through a wide arc, and the folded
         arm on the right keeps its second link close to the shoulder.

   The shoulder turns at 1 rad/s in both poses, and the arrows are the velocities of the two
   link centers. Stretched, the center of the second link moves at 1.5 m/s and the motion
   costs 1.633. Folded, it moves at 0.55 m/s and the same shoulder speed costs 0.851.

The built-in robots of ``geodex.robots`` evaluate their mass matrix with C++ code of the
composite rigid body algorithm (CRBA) that Pinocchio generated ahead of time from the robot
description. Each robot also ships a precomputed constant lower bound of the mass matrix, and
the planner uses this bound in its heuristic.

.. code-pair:: concepts/metrics robot

The **Jacobi metric** :math:`2(H - P(q))\, M(q)` adds gravity. For a total energy :math:`H`
above the potential :math:`P(q)` everywhere, its geodesics are the free motions of the arm at
that energy :footcite:`Arnold1989`. The factor :math:`2(H - P)` is twice the kinetic energy
the arm has left at :math:`q`. It is small where the potential is high and the arm moves
slowly. Geodesics bend toward high potential, the way the path of a thrown ball arcs upward.
See :doc:`/tutorials/minimum-energy-planning` for plans under both metrics.

.. code-pair:: concepts/metrics jacobi

Clearance
---------

A **conformal metric** scales a base metric by a factor that depends on the configuration.
The clearance metric uses the signed distance :math:`\mathrm{sdf}(q)` to the nearest obstacle,

.. math::

   c(q) = 1 + \kappa\, e^{-\beta\, \mathrm{sdf}(q)}, \qquad
   \langle u, v \rangle_q = c(q)\, \langle u, v \rangle^{\text{base}}_q .

A motion at :math:`q` is :math:`\sqrt{c(q)}` times as long as under the base metric. On the
surface of an obstacle, where :math:`\mathrm{sdf}(q) = 0`, it is :math:`\sqrt{1 + \kappa}`
times as long (1.58 for :math:`\kappa = 1.5`). The factor grows further inside an obstacle,
where the signed distance is negative, and fades to 1 in open space. Shortest paths keep away
from obstacles where a detour is cheap. :math:`\kappa` sets the strength, and :math:`\beta`
sets how fast the factor fades. The metric is ``geodex.ClearanceMetric`` in Python and
``geodex::SDFConformalMetric`` in C++, and it accepts any signed distance function, including
the footprint checker of :doc:`/tutorials/se2-planning`.

.. code-pair:: concepts/metrics clearance

Task-space metrics
------------------

A **pullback metric** measures joint motion by the motion it causes in task space. With the
Jacobian :math:`J(q)` of a task map, such as the end-effector position, and a task-space
metric :math:`G_X`,

.. math::

   \langle u, v \rangle_q = u^\top J(q)^\top G_X\, J(q)\, v + \lambda\, u^\top v .

The small :math:`\lambda` keeps the metric positive definite at singularities, where some
joint motion does not move the task point. With :math:`G_X = I`, the cost of a joint velocity
is the speed of the end effector.

.. code-pair:: concepts/metrics pullback

Composing metrics
-----------------

``ConstantSPDMetric`` weights coordinates by a fixed matrix, ``WeightedMetric`` scales a
metric by a function of the configuration, and ``AffineCombinedMetric`` adds metrics with
fixed coefficients, for example a kinetic-energy term and a clearance term. The configuration
space of a mobile manipulator is the product :math:`\mathrm{SE}(2) \times \mathbb{R}^n` of its
base and its arm, and its metric is a block metric with a left-invariant block for the base and
a kinetic-energy block for the arm. A holonomic and a differential-drive base differ in the base
block alone (see :doc:`/robots/mobile-manipulation`).

See also
--------

- :doc:`riemannian-geometry` for the definitions behind inner products and geodesics.
- :doc:`planning` for how the planner uses the metric, including admissible heuristics.
- :doc:`/tutorials/geodex-basics` for swapping metrics on the built-in manifolds.

References
----------

.. footbibliography::
