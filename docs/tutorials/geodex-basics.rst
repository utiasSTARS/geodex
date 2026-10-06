geodex Basics
=============

In this tutorial, we create manifolds, measure tangent vectors, move along the manifold with
the exponential and logarithmic maps, measure distances, interpolate along geodesics, sample
random points, and then change the metric and the retraction of a manifold. Every step prints
a value you can check by hand.

The full script is ``examples/tutorials/geodex_basics.py``, and the C++ version is
``examples/tutorials/geodex_basics.cpp``. The Python snippets import NumPy next to geodex,
and the C++ snippets include the umbrella header.

.. code-pair:: tutorials/geodex_basics setup

See :doc:`/concepts/riemannian-geometry` for the terms this tutorial uses (manifold, tangent
space, metric, exponential map, geodesic).

1. Create a manifold
--------------------

The 2-sphere :math:`\mathbb{S}^2` is the set of unit vectors in :math:`\mathbb{R}^3`. In
Python, ``geodex.Sphere()`` is the 2-sphere and ``geodex.SphereN(n)`` the
:math:`n`-sphere. In C++, the class template ``Sphere<Dim>`` models :math:`\mathbb{S}^n`,
and ``Sphere<>`` is ``Sphere<2>``.

.. code-pair:: tutorials/geodex_basics first-manifold

It prints ``dim = 2``. Other manifolds follow the same pattern.

.. code-pair:: tutorials/geodex_basics create-manifolds

Every manifold is built from **policies**, a metric (what "length" means), a retraction (how
to move along the manifold) and a sampler (how to sample points). The defaults are the round
metric, the exponential map and a scrambled Halton sampler. In C++, they are template
parameters, and ``Sphere<>`` is the same type as the fully qualified one below.

.. code-pair:: tutorials/geodex_basics fully-qualified
   :only: cpp

geodex provides these manifolds, each with its canonical metric by default (the round metric on
spheres, the flat metric on tori, the bi-invariant metric on the rotation groups and a
left-invariant metric on the rigid-body groups).

.. list-table::
   :header-rows: 1
   :widths: 14 25 22 29 10

   * - Manifold
     - Python
     - C++
     - Point
     - dim
   * - :math:`\mathbb{S}^N`
     - ``Sphere()``, ``SphereN(n)``
     - ``Sphere<N>``
     - unit vector of :math:`\mathbb{R}^{N+1}`
     - N
   * - :math:`\mathbb{R}^N`
     - ``Euclidean(n)``
     - ``Euclidean<N>``
     - :math:`N` coordinates
     - N
   * - :math:`\mathbb{T}^N`
     - ``Torus(n)``
     - ``Torus<N>``
     - :math:`N` angles
     - N
   * - :math:`\mathrm{SO}(2)`
     - ``SO2()``
     - ``SO2<>``
     - one angle
     - 1
   * - :math:`\mathrm{SO}(3)`
     - ``SO3()``
     - ``SO3<>``
     - unit quaternion :math:`[x, y, z, w]`
     - 3
   * - :math:`\mathrm{SE}(2)`
     - ``SE2()``
     - ``SE2<>``
     - :math:`(x, y, \theta)`
     - 3
   * - :math:`\mathrm{SE}(3)`
     - ``SE3()``
     - ``SE3<>``
     - translation and quaternion, 7 numbers
     - 6
   * - :math:`\prod_i \mathcal{M}_i`
     - ``Product([...])``
     - ``make_product(...)``
     - the blocks stacked
     - sum

2. Measure tangent vectors
--------------------------

A tangent vector at a point :math:`p` is a velocity through :math:`p`. The metric gives two
tangent vectors an inner product :math:`\langle u, v \rangle_p` and a tangent vector a norm
:math:`\|v\|_p`. Both functions take the base point first. On the sphere with the round
metric, the inner product is the dot product of the ambient vectors.

.. code-pair:: tutorials/geodex_basics inner-sphere

It prints ``ip = 0.0`` and ``n = 1.0``. At the north pole, the two vectors are orthogonal and
of unit length. Euclidean space gives the same numbers for the same vectors.

.. code-pair:: tutorials/geodex_basics inner-euclidean

On a manifold with a position-dependent metric, such as the kinetic-energy metric of a robot
arm, the same two vectors give a different inner product at every point.

3. Move with exp and log
------------------------

The exponential map :math:`\exp_p(v)` starts at :math:`p` and follows the geodesic in the
direction of :math:`v` for a length :math:`\|v\|_p`. The logarithmic map :math:`\log_p(q)`
is its inverse, the tangent vector at :math:`p` that reaches :math:`q`.

.. code-pair:: tutorials/geodex_basics exp-log-sphere

It prints ``v = [1.5708 0 0]`` and ``q_recovered = [1 0 0]`` up to rounding. The direction
of :math:`v` is the initial heading of the great circle from the pole to the equator, and its
norm, :math:`\pi/2`, is the distance between them. On Euclidean space, the two maps are
addition and subtraction.

.. code-pair:: tutorials/geodex_basics exp-log-euclidean

It prints ``v = [-1 1 0]`` and ``q2 = [0 1 0]``.

.. note::

   ``exp(p, log(p, q))`` recovers :math:`q` within the injectivity radius of :math:`p`,
   :math:`\pi` on the sphere. Antipodal points lie at distance :math:`\pi`, where the
   geodesic is not unique, and ``log`` returns a zero vector with a warning.

4. Measure distances
--------------------

``distance(p, q)`` is the length of the shortest path, the norm of :math:`\log_p(q)`. The
north pole and a point on the equator are a quarter of a great circle apart.

.. code-pair:: tutorials/geodex_basics distance-sphere

It prints ``d = 1.5707963267948966``, :math:`\pi/2`. On Euclidean space, the distance is the
norm of the difference.

.. code-pair:: tutorials/geodex_basics distance-euclidean

It prints ``d = 1.4142135623730951``, :math:`\sqrt{2}`. On a circle, the torus
:math:`\mathbb{T}^1`, the angles 0.1 and 6.0 are 5.9 apart in coordinates, and the shortest
path wraps through :math:`2\pi`.

.. code-pair:: tutorials/geodex_basics distance-torus

It prints ``d = 0.3831853071795859``, :math:`2\pi - 5.9`.

.. note::

   ``distance()`` uses the midpoint method, which evaluates the logarithm at the geodesic
   midpoint :footcite:`kyaw2026geometry`. With the exact exponential and logarithmic maps, as
   on the round sphere, it gives the exact geodesic distance.

5. Interpolate along geodesics
------------------------------

``geodesic(p, q, t)`` is the point a fraction :math:`t` of the way along the geodesic from
:math:`p` to :math:`q`, :math:`\exp_p(t \log_p(q))`. On the sphere, it traces an arc of a
great circle.

.. code-pair:: tutorials/geodex_basics geodesic-sphere

It prints ``mid = [0.7071 0 0.7071]``, the unit vector at 45 degrees between the pole and
the equator, then eleven points along the arc. The same call works on every manifold, and on
Euclidean space it is linear interpolation.

.. code-pair:: tutorials/geodex_basics geodesic-euclidean

It prints ``mid = [1 2 3]``.

6. Sample random points
-----------------------

``random_point()`` samples a point uniformly over the manifold. The samples come from a
low-discrepancy sequence (scrambled Halton), which covers the space evenly with few samples.
The 2-sphere maps the unit square onto the sphere with the cylindrical equal-area map, the torus
maps each coordinate onto :math:`[0, 2\pi)`, and Euclidean space fills a box,
:math:`[-1, 1]` per coordinate by default. ``set_sampling_bounds(lo, hi)`` resizes that box.

.. code-pair:: tutorials/geodex_basics random-points

geodex provides three samplers, ``ScrambledHaltonSampler`` (the default), ``HaltonSampler`` (a
fixed deterministic sequence) and ``PseudoRandomSampler`` (independent ``mt19937`` samples).
In C++, the sampler is a template parameter, and in Python a constructor keyword together
with ``set_sampler``.

.. code-pair:: tutorials/geodex_basics sampler-choice

``seed`` on a manifold reseeds its sampler, and ``geodex.seed`` (``geodex::set_default_seed``
in C++) reseeds the source of the samplers constructed afterwards.

.. code-pair:: tutorials/geodex_basics seeding

See :doc:`/concepts/sampling` for the samplers and the maps in detail.

7. Wrap angles on the torus
---------------------------

The :math:`n`-torus :math:`\mathbb{T}^n` is the product of :math:`n` circles, with points
stored as angles in :math:`[0, 2\pi)^n`. The logarithm takes the shortest signed difference
of each angle, wrapped into :math:`[-\pi, \pi)`, and the exponential adds the tangent and
wraps the result back into :math:`[0, 2\pi)`.

.. code-pair:: tutorials/geodex_basics torus-wrap

It prints ``v = [-0.2832 0.3]`` and ``q_recovered = [6.1 0.5]``. The first angle moves back
across zero instead of forward by 6.0.

.. _rotations-rigid-bodies-products:

8. Rotate and move rigid bodies
-------------------------------

geodex provides four Lie groups. The rotation groups :math:`\mathrm{SO}(2)` and
:math:`\mathrm{SO}(3)` rotate without translating, and the special Euclidean groups
:math:`\mathrm{SE}(2)` and :math:`\mathrm{SE}(3)` add a translation to describe rigid-body
poses. Their ``exp`` is the group exponential, and the group composition couples the
coordinates.

:math:`\mathrm{SO}(2)` is the circle group, one angle in :math:`[-\pi, \pi)`, with a
bi-invariant canonical metric. :math:`\mathrm{SO}(3)` is the group of spatial rotations. A
point is a unit quaternion :math:`[x, y, z, w]`, a tangent vector is a body angular velocity,
and geodesics are quaternion SLERP.

.. code-pair:: tutorials/geodex_basics so3

An :math:`\mathrm{SE}(2)` pose is :math:`(x, y, \theta)`, and a tangent vector is a
body-frame velocity :math:`(v_x, v_y, \omega)`. The default metric is the left-invariant
metric with weights :math:`(w_x, w_y, w_\theta)`,

.. math::

   \langle u, v \rangle = w_x u_x v_x + w_y u_y v_y + w_\theta u_\theta v_\theta ,

with unit weights by default.

.. code-pair:: tutorials/geodex_basics se2

It prints the logarithm, the pose it recovers, ``[3 4 1.57]``, and ``distance:
3.5117456798791093``. A large sideways weight :math:`w_y` makes sideways motion expensive, as
for a wheeled base that drives only forward and backward.

.. code-pair:: tutorials/geodex_basics se2-car

It prints ``distance: 20.09975124224178``. The weight multiplies the squared sideways speed,
and with :math:`w_y = 100`, a sideways displacement is ten times as long as the same
displacement forward. A shortest path between two poses turns and drives instead of sliding,
and it may reverse, as the Reeds-Shepp paths of a car do
:footcite:`kyaw2026geometry,belta2002euclidean`.

``random_point()`` on :math:`\mathrm{SE}(2)` samples the position uniformly within the sampling
bounds (default :math:`[0, 10]^2`) and the heading uniformly from :math:`[-\pi, \pi)`.

An :math:`\mathrm{SE}(3)` pose is a translation followed by a scalar-last unit quaternion,
:math:`[t_x, t_y, t_z, q_x, q_y, q_z, q_w]`, and a tangent vector is a body twist
:math:`[v;\, \omega]`. ``geodesic`` follows the screw motion of the constant twist between
two poses.

.. code-pair:: tutorials/geodex_basics se3

.. _body-and-world-frames:

Body and world frames
^^^^^^^^^^^^^^^^^^^^^

A rigid-body velocity can be written in two frames. For a trajectory :math:`X(t)` on a Lie
group, the body velocity is :math:`X^{-1}\dot X` and the spatial velocity is
:math:`\dot X X^{-1}`. Inner products of body velocities give a **left-invariant** metric,
unchanged by a change of the world frame, and inner products of spatial velocities give a
**right-invariant** metric, unchanged by a change of the body frame
:footcite:`park1995distance`.

The ``frame`` argument of :math:`\mathrm{SO}(3)`, :math:`\mathrm{SE}(2)` and
:math:`\mathrm{SE}(3)` picks one. ``frame="body"``, the default, steps with
:math:`\exp_p(\xi) = p \cdot \mathrm{Exp}(\xi)`, and ``frame="world"`` with
:math:`\exp_p(\xi) = \mathrm{Exp}(\xi) \cdot p`. In C++, the choice is a retraction policy,
``SO3LeftExponentialMap`` or ``SO3RightExponentialMap``, and the same pairs for
:math:`\mathrm{SE}(2)` and :math:`\mathrm{SE}(3)`.

The canonical metric of :math:`\mathrm{SO}(3)` is bi-invariant :footcite:`park1995distance`,
and both frames report the same distance between two rotations.

.. code-pair:: tutorials/geodex_basics so3-frames

:math:`\mathrm{SE}(3)` does not have a bi-invariant metric :footcite:`park1995distance`, and
the two frames report different distances between the same poses. :math:`\mathrm{SE}(2)`
does not have one either.

.. code-pair:: tutorials/geodex_basics se3-frames

The ``w_rot`` and ``w_trans`` weights of :math:`\mathrm{SE}(3)` set how a radian of rotation
trades against a unit of translation.

.. list-table::
   :header-rows: 1
   :widths: 22 26 52

   * - Manifold
     - ``frame`` argument
     - Metric
   * - :math:`\mathrm{SO}(2)`
     - none
     - Canonical, bi-invariant
   * - :math:`\mathrm{SO}(3)`
     - ``body`` or ``world``
     - Canonical, bi-invariant, the same distance in both frames
   * - :math:`\mathrm{SE}(2)`
     - ``body`` or ``world``
     - Left- or right-invariant
   * - :math:`\mathrm{SE}(3)`
     - ``body`` or ``world``
     - Left- or right-invariant

9. Combine manifolds
--------------------

A **product manifold** joins several manifolds into one configuration space. Points and
tangent vectors stack the blocks, exp, log and geodesic act block by block, and the distance
is the L2 combination of the block distances. A point in space and a planar base pose, such as
an end-effector position and the pose of a mobile base, form a point of
:math:`\mathbb{R}^3 \times \mathrm{SE}(2)`.

.. code-pair:: tutorials/geodex_basics product

It prints ``dim = 6``. The product samples from one low-discrepancy sequence over the combined
coordinates (see :doc:`/concepts/sampling`).

10. Change the metric
---------------------

``ConstantSPDMetric`` defines a point-independent inner product from a symmetric positive
definite matrix :math:`A`,

.. math::

   \langle u, v \rangle = u^\top A\, v .

With :math:`A = \mathrm{diag}(4, 1, 1)`, a unit step along the first coordinate has length
:math:`\sqrt{4} = 2`, twice the length of a unit step along the others. In
Python, ``ConfigurationSpace(base, metric)`` puts a metric on a base manifold. In C++, the
metric is a template parameter, as in ``Euclidean<3, ConstantSPDMetric<3>>``.

.. code-pair:: tutorials/geodex_basics constant-spd

It prints ``d = 2.449489742783178``, :math:`\sqrt{6}`, against :math:`\sqrt{3}` under the
standard metric. The same metric works on any manifold with 3-vector tangents, the sphere
among them.

.. code-pair:: tutorials/geodex_basics spd-sphere

It prints ``n = 2.0``. The metric decides what we measure (inner products, norms,
distances), and the retraction decides how we move.

11. Try it: swap the retraction
-------------------------------

A **retraction** is a cheaper map that agrees with the exponential map near the base point.
On the sphere, the projection retraction normalizes :math:`p + v` instead of following a great
circle. It agrees with the exponential map to second order in :math:`\|v\|` and does not
evaluate trigonometric functions.

.. code-pair:: tutorials/geodex_basics projection-retraction

It prints ``q_approx = [0.7071 0 0.7071]``. The projection retraction reaches only the open
hemisphere around :math:`p`, and a target 90 degrees away comes back at 45 degrees. Use it for
short steps, and the exponential map for long ones. The metric stays the round metric.

:math:`\mathrm{SE}(2)` also has a cheap Euler retraction.

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Retraction
     - Accuracy
     - Description
   * - ``SE2LeftExponentialMap``
     - exact
     - The group exponential through the :math:`V(\omega)` matrix (default)
   * - ``SE2EulerRetraction``
     - first order, at heading 0 only
     - Adds the tangent to the coordinates and wraps the angle, which treats SE(2) as
       :math:`\mathbb{R}^2 \times \mathbb{S}^1` and reads the tangent as world-frame rates

.. code-pair:: tutorials/geodex_basics se2-euler

``distance()`` and ``geodesic()`` build on exp and log, and a cheaper retraction makes them
less accurate. Use the exponential map where accuracy matters.

Where to go next
----------------

- :doc:`minimum-energy-planning` plans on configuration spaces with position-dependent
  metrics.
- :doc:`se2-planning` plans for mobile robots on :math:`\mathrm{SE}(2)`.
- :doc:`/concepts/riemannian-geometry` defines the geometry behind each step.
- :doc:`/api/index` covers every class and function.

References
----------

.. footbibliography::
