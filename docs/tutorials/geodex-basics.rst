geodex Basics
=============

This tutorial introduces the core operations of geodex: creating manifolds, computing exponential and logarithmic maps, measuring distances, and interpolating along geodesics.
We will work through each concept step by step starting with the underlying mathematics and then show how geodex expresses it in the code.

By the end, you will be comfortable with the four built-in manifolds in geodex (the sphere, Euclidean space, the flat torus, and SE(2)) and understand how to swap metrics and retractions to change the geometry.

.. note::

   This tutorial assumes you have :doc:`installed geodex </getting-started/index>` and have a basic familiarity with the geometric vocabulary (manifolds, tangent spaces, metrics).
   If terms like "exponential map" or "Riemannian metric" are unfamiliar, we recommend reading :doc:`/concepts/index` first.

All C++ examples in this tutorial use the main geodex header:

.. tabs::

   .. code-tab:: c++

      #include <geodex/geodex.hpp>

   .. code-tab:: py

      import geodex
      import numpy as np


Your First Manifold
-------------------

The simplest way to get started is with the 2-sphere, defined as the set of unit vectors in :math:`\mathbb{R}^3`:

.. math::

   \mathbb{S}^2 = \bigl\{ p \in \mathbb{R}^3 : \|p\| = 1 \bigr\}.

In geodex, the ``Sphere<Dim>`` class template models :math:`\mathbb{S}^n` for any
:math:`n \geq 1`. ``Sphere<>`` is shorthand for ``Sphere<2>`` — the default 2-sphere.
Creating one is straightforward:

.. tabs::

   .. code-tab:: c++

      geodex::Sphere<> sphere;
      std::cout << "dim = " << sphere.dim() << "\n";  // 2

   .. code-tab:: py

      sphere = geodex.Sphere()
      print("dim =", sphere.dim())  # 2

The empty angle brackets ``<>`` mean we are using the default intrinsic dimension
(``2``), the default **metric** (``SphereRoundMetric``), and the default **retraction**
(``SphereExponentialMap``). Writing ``Sphere<>`` is equivalent to writing the
fully-qualified type:

.. code-block:: cpp

   geodex::Sphere<2, geodex::SphereRoundMetric, geodex::SphereExponentialMap> sphere;

The default names ``SphereRoundMetric``, ``EuclideanStandardMetric<N>``, and
``TorusFlatMetric<N>`` are type aliases for ``ConstantSPDMetric<K>`` with an identity
matrix — the ambient identity inner product.

This is the **policy-based design** at the heart of geodex: the manifold is parameterized by a metric policy (what "length" means) and a retraction policy (how to move along the manifold).
We will explore both in later sections.

Points on :math:`\mathbb{S}^n` are ``Eigen::Vector<double, n+1>`` unit vectors, and
tangent vectors share the same type (orthogonal to the base point). For the default
``Sphere<>`` (:math:`\mathbb{S}^2`), both are ``Eigen::Vector3d``.

Manifolds at a Glance
^^^^^^^^^^^^^^^^^^^^^

geodex ships with four manifold families.
The table below summarises their essential properties:

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Manifold
     - Symbol
     - Point type
     - dim
     - Default metric
   * - ``Sphere<N>``
     - :math:`\mathbb{S}^N`
     - ``Eigen::Vector<double, N+1>``
     - N
     - ``SphereRoundMetric``
   * - ``Euclidean<N>``
     - :math:`\mathbb{R}^N`
     - ``Eigen::Vector<double, N>``
     - N
     - ``EuclideanStandardMetric<N>``
   * - ``Torus<N>``
     - :math:`\mathbb{T}^N`
     - ``Eigen::Vector<double, N>``
     - N
     - ``TorusFlatMetric<N>``
   * - ``SE2<>``
     - :math:`\mathrm{SE}(2)`
     - ``Eigen::Vector3d``
     - 3
     - ``SE2LeftInvariantMetric``

Creating instances of these manifolds follows the same pattern:

.. tabs::

   .. code-tab:: c++

      geodex::Euclidean<3> R3;  // R^3 with standard metric
      geodex::Torus<2>     T2;  // 2-torus with flat metric
      geodex::SE2<>        se2; // SE(2) with left-invariant metric

   .. code-tab:: py

      euclidean = geodex.Euclidean(3)  # R^3 with standard metric
      torus     = geodex.Torus(2)      # 2-torus with flat metric
      se2       = geodex.SE2()         # SE(2) with left-invariant metric


The Riemannian Inner Product
----------------------------

A Riemannian metric assigns an inner product to each tangent space.
Given two tangent vectors :math:`u, v \in \mathcal{T}_p\mathcal{M}` at a point :math:`p`, the inner product is written

.. math::

   \langle u, v \rangle_p,

and the induced norm is

.. math::

   \|v\|_p = \sqrt{\langle v, v \rangle_p}.

On the sphere with the round metric, the inner product is simply the Euclidean dot product of the ambient vectors:

.. math::

   \langle u, v \rangle_p = u \cdot v.

In code, we call ``inner()`` and ``norm()``:

.. tabs::

   .. code-tab:: c++

      geodex::Sphere<> sphere;

      Eigen::Vector3d p{0, 0, 1};          // north pole
      Eigen::Vector3d u{1, 0, 0};          // tangent vector pointing east
      Eigen::Vector3d v{0, 1, 0};          // tangent vector pointing south

      double ip = sphere.inner(p, u, v);   
      std::cout << "ip = " << ip << "\n";  // 0.0 (orthogonal)
      double n  = sphere.norm(p, u);
      std::cout << "n = " << n << "\n";  // 1.0

   .. code-tab:: py

      sphere = geodex.Sphere()

      p = np.array([0., 0., 1.])  # north pole
      u = np.array([1., 0., 0.])  # tangent vector pointing east
      v = np.array([0., 1., 0.])  # tangent vector pointing south

      ip = sphere.inner(p, u, v)  # 0.0 (orthogonal)
      print("ip =", ip)  # 0.0 (orthogonal)
      n  = sphere.norm(p, u)
      print("n =", n)  # 1.0

Notice that both functions take the base point :math:`p` as their first argument.
For the round metric, the inner product does not actually depend on :math:`p`, but this is a special case.
On manifolds with position-dependent metrics (such as the kinetic energy metric used in robotics), the base point changes the inner product at every configuration.

.. tip::

   The :doc:`minimum-energy-planning` tutorial shows how position-dependent metrics arise naturally when modelling robot inertia.

On Euclidean space, the inner product and norm behave identically to the standard dot product:

.. tabs::

   .. code-tab:: c++

      geodex::Euclidean<3> R3;

      Eigen::Vector3d p{0, 0, 0};
      Eigen::Vector3d u{1, 0, 0};
      Eigen::Vector3d v{0, 1, 0};

      double ip = R3.inner(p, u, v);
      std::cout << "ip = " << ip << "\n";  // 0.0
      double n  = R3.norm(p, u);
      std::cout << "n = " << n << "\n";  // 1.0

   .. code-tab:: py

      euclidean = geodex.Euclidean(3)

      p = np.array([0., 0., 0.])
      u = np.array([1., 0., 0.])
      v = np.array([0., 1., 0.])

      ip = euclidean.inner(p, u, v)
      print("ip =", ip)  # 0.0
      n  = euclidean.norm(p, u)
      print("n =", n)  # 1.0


Exponential and Logarithmic Maps
---------------------------------

The exponential map :math:`\exp_p(v)` starts at a point :math:`p` and "walks" along the manifold in the direction of a tangent vector :math:`v` for a distance equal to :math:`\|v\|_p`.
The result is a new point on the manifold.
The logarithmic map :math:`\log_p(q)` is its inverse: given two points :math:`p` and :math:`q`, it returns the tangent vector at :math:`p` that points toward :math:`q`.

On the sphere, the exponential map has a particularly elegant closed form:

.. math::

   \exp_p(v) = \cos(\|v\|)\, p + \sin(\|v\|)\, \frac{v}{\|v\|}.

This traces a great circle starting at :math:`p` in the direction of :math:`v`.
The logarithmic map inverts the relationship, that is, it finds the initial velocity that, followed for unit time, arrives at :math:`q`:

.. tabs::

   .. code-tab:: c++

      geodex::Sphere<> sphere;

      Eigen::Vector3d p{0, 0, 1};  // north pole
      Eigen::Vector3d q{1, 0, 0};  // point on equator
      
      Eigen::IOFormat fmt(4, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");

      // log: tangent vector at p pointing toward q
      Eigen::Vector3d v = sphere.log(p, q);
      std::cout << "v = " << v.transpose().format(fmt) << "\n"; // ≈ [π/2, 0, 0] - points along the great circle from pole to equator

      // exp: follow that tangent vector to recover q
      Eigen::Vector3d q_recovered = sphere.exp(p, v);
      std::cout << "q_recovered = " << q_recovered.transpose().format(fmt) << "\n"; // ≈ [1, 0, 0]

   .. code-tab:: py

      sphere = geodex.Sphere()

      p = np.array([0., 0., 1.])  # north pole
      q = np.array([1., 0., 0.])  # point on equator

      # log: tangent vector at p pointing toward q
      v = sphere.log(p, q)
      print("v =", v)  # ≈ [π/2, 0, 0] - points along the great circle from pole to equator

      # exp: follow that tangent vector to recover q
      q_recovered = sphere.exp(p, v)
      print("q_recovered =", q_recovered)  # ≈ [1, 0, 0]

The tangent vector :math:`v = \log_p(q)` encodes two things: its *direction* is the initial heading of the geodesic from :math:`p` to :math:`q`, and its *norm* :math:`\|v\|_p` is the geodesic distance between them.
The round trip ``exp(p, log(p, q))`` always recovers :math:`q` (up to numerical precision), as long as :math:`q` lies within the **injectivity radius** of :math:`p`.

.. note::

   On the sphere, the injectivity radius is :math:`\pi`.
   Antipodal points (e.g. the north and south poles) lie exactly at distance :math:`\pi`, and the geodesic between them is not unique, that is, there exist infinitely many great circles passing through both poles.
   In geodex, calling ``log`` on antipodal points returns a zero vector with a warning.

On Euclidean space, the exponential and logarithmic maps reduce to simple addition and subtraction:

.. math::

   \exp_p(v) = p + v, \qquad \log_p(q) = q - p.

.. tabs::

   .. code-tab:: c++

      geodex::Euclidean<3> R3;

      Eigen::Vector3d p{1, 0, 0};
      Eigen::Vector3d q{0, 1, 0};
      
      Eigen::IOFormat fmt(4, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");

      auto v  = R3.log(p, q);
      std::cout << "v = " << v.transpose().format(fmt) << "\n";  // [-1, 1, 0]
      auto q2 = R3.exp(p, v);
      std::cout << "q2 = " << q2.transpose().format(fmt) << "\n";  // [0, 1, 0] == q

   .. code-tab:: py

      euclidean = geodex.Euclidean(3)

      p = np.array([1., 0., 0.])
      q = np.array([0., 1., 0.])

      v  = euclidean.log(p, q)  
      print("v =", v)  # [-1, 1, 0]
      q2 = euclidean.exp(p, v)
      print("q2 =", q2)  # [0, 1, 0] == q


Geodesic Distance
-----------------

The geodesic distance between two points is the length of the shortest path connecting them.
It can be expressed in terms of the logarithmic map and the Riemannian norm:

.. math::

   d(p, q) = \bigl\|\log_p(q)\bigr\|_p.

On the sphere with the round metric, this simplifies to the well-known arc-length formula:

.. math::

   d(p, q) = \arccos(p \cdot q).

We can verify this in the code.
The north pole :math:`(0,0,1)` and a point on the equator :math:`(1,0,0)` are separated by a quarter of a great circle, so the distance should be :math:`\pi/2`:

.. tabs::

   .. code-tab:: c++

      geodex::Sphere<> sphere;

      Eigen::Vector3d p{0, 0, 1};       // north pole
      Eigen::Vector3d q{1, 0, 0};       // equator

      auto d = sphere.distance(p, q);
      std::cout << "d = " << d << "\n";           // 1.5708 ≈ π/2

   .. code-tab:: py

      sphere = geodex.Sphere()

      p = np.array([0., 0., 1.])  # north pole
      q = np.array([1., 0., 0.])  # equator

      d = sphere.distance(p, q)
      print("d =", d)  # 1.5708 ≈ π/2

.. note::

   Internally, ``distance()`` uses the ``distance_midpoint`` algorithm from :cite:`kyaw2026geometry`, which evaluates log maps at the geodesic midpoint to obtain a third-order accurate approximation.
   For manifolds that provide true (exact) exponential and logarithmic maps, like the sphere with ``SphereExponentialMap``, this formula yields the exact geodesic distance.

On Euclidean space, the geodesic distance is just the standard Euclidean norm of the difference:

.. tabs::

   .. code-tab:: c++

      geodex::Euclidean<3> R3;

      Eigen::Vector3d p{1, 0, 0};
      Eigen::Vector3d q{0, 1, 0};

      double d = R3.distance(p, q);
      std::cout << "d = " << d << "\n";  // 1.41421 ≈ √2

   .. code-tab:: py

      euclidean = geodex.Euclidean(3)

      p = np.array([1., 0., 0.])
      q = np.array([0., 1., 0.])

      d = euclidean.distance(p, q)
      print("d =", d)  # 1.41421 ≈ √2

The **torus** is where distance becomes more interesting, because coordinates wrap around.
Consider two angles on a 1-torus (a circle) at :math:`\theta_1 = 0.1` and :math:`\theta_2 = 6.0`.
The naive difference is :math:`5.9`, but the shortest path wraps around through :math:`2\pi`, giving a distance of approximately :math:`0.38`:

.. tabs::

   .. code-tab:: c++

      geodex::Torus<1> S1;

      Eigen::Vector<double, 1> p{0.1};
      Eigen::Vector<double, 1> q{6.0};

      double d = S1.distance(p, q);
      std::cout << "d = " << d << "\n";  // 0.383 ≈ 2π - 5.9

   .. code-tab:: py

      S1 = geodex.Torus(1)

      p = np.array([0.1])
      q = np.array([6.0])

      d = S1.distance(p, q)
      print("d =", d)  # 0.383 ≈ 2π - 5.9


Geodesic Interpolation
----------------------

Given two points :math:`p` and :math:`q`, we often want the point a fraction :math:`t` of the way along the shortest geodesic between them.
The formula is a direct composition of the logarithmic and exponential maps:

.. math::

   \gamma(t) = \exp_p\,\bigl(t \cdot \log_p(q)\bigr), \quad t \in [0, 1].

At :math:`t = 0` we recover :math:`p`, at :math:`t = 1` we recover :math:`q`, and at :math:`t = 0.5` we get the geodesic midpoint.

On the sphere, this traces an arc of the great circle.
The midpoint between the north pole :math:`(0,0,1)` and the equator :math:`(1,0,0)` lies at 45° latitude:

.. tabs::

   .. code-tab:: c++

      geodex::Sphere<> sphere;

      Eigen::Vector3d p{0, 0, 1};  // north pole
      Eigen::Vector3d q{1, 0, 0};  // equator
      
      Eigen::IOFormat fmt(4, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");

      Eigen::Vector3d mid = sphere.geodesic(p, q, 0.5);
      std::cout << "mid = " << mid.transpose().format(fmt) << "\n";  // ≈ [0.707, 0, 0.707]

      // Trace the full geodesic
      for (int i = 0; i <= 10; ++i) {
         double t = i / 10.0;
         Eigen::Vector3d pt = sphere.geodesic(p, q, t);
         std::cout << "t=" << t << ": " << pt.transpose().format(fmt) << "\n";
      }

   .. code-tab:: py

      sphere = geodex.Sphere()

      p = np.array([0., 0., 1.])  # north pole
      q = np.array([1., 0., 0.])  # equator

      mid = sphere.geodesic(p, q, 0.5)
      print("mid =", mid)  # ≈ [0.707, 0, 0.707]

      # Trace the full geodesic
      for i in range(11):
          t = i / 10.0
          pt = sphere.geodesic(p, q, t)
          print(f"t={t:.1f}: {pt}")

The midpoint has coordinates approximately :math:`(1/\sqrt{2},\, 0,\, 1/\sqrt{2})`, which is the unit vector at 45° between the pole and the equator — exactly what we expect.

.. tip::

   The formula :math:`\gamma(t) = \exp_p(t \cdot \log_p(q))` is *universal*: it works on any ``RiemannianManifold``.
   Whether you are interpolating on a sphere, a torus, or SE(2), the same ``geodesic(p, q, t)`` call does the right thing.

On Euclidean space, geodesic interpolation reduces to linear interpolation:

.. math::

   \gamma(t) = (1 - t)\, p + t\, q.

.. tabs::

   .. code-tab:: c++

      geodex::Euclidean<3> R3;

      Eigen::Vector3d p{0, 0, 0};
      Eigen::Vector3d q{2, 4, 6};
      
      Eigen::IOFormat fmt(4, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");

      Eigen::Vector3d mid = R3.geodesic(p, q, 0.5);
      std::cout << "mid = " << mid.transpose().format(fmt) << "\n";  // [1, 2, 3]

   .. code-tab:: py

      euclidean = geodex.Euclidean(3)

      p = np.array([0., 0., 0.])
      q = np.array([2., 4., 6.])

      mid = euclidean.geodesic(p, q, 0.5)
      print("mid =", mid)  # [1, 2, 3]


Random Sampling
---------------

Every manifold in geodex provides a ``random_point()`` method for sampling.
The distribution depends on the manifold:

.. tabs::

   .. code-tab:: c++

      geodex::Sphere<>     sphere;
      geodex::Euclidean<3> R3;
      geodex::Torus<2>     T2;
      geodex::SE2<>        se2;

      auto p1 = sphere.random_point();  // uniform on S²
      auto p2 = R3.random_point();      // uniform in [-1, 1]³
      auto p3 = T2.random_point();      // uniform in [0, 2π)²
      auto p4 = se2.random_point();     // uniform in sampling bounds

   .. code-tab:: py

      sphere    = geodex.Sphere()
      euclidean = geodex.Euclidean(3)
      torus     = geodex.Torus(2)
      se2       = geodex.SE2()

      p1 = sphere.random_point()     # uniform on S²
      p2 = euclidean.random_point()  # uniform in [-1, 1]³
      p3 = torus.random_point()      # uniform in [0, 2π)²
      p4 = se2.random_point()        # uniform in sampling bounds

The sphere uses the standard technique of normalizing a Gaussian vector to obtain a
uniform distribution on :math:`\mathbb{S}^n`. Euclidean space draws each coordinate
uniformly from :math:`[-1, 1]`; call ``set_sampling_bounds(lo, hi)`` to change the box.
The torus samples each angle uniformly from :math:`[0, 2\pi)`.

.. note::

   ``Euclidean``, ``Torus``, and ``SE2`` take a ``SamplerT`` policy template parameter
   that drives ``random_point()``. The default is ``StochasticSampler`` (``mt19937``).
   Swap in ``HaltonSampler`` for deterministic low-discrepancy quasi-random sampling —
   useful for reproducible benchmarks and for planners that benefit from better
   space coverage than pseudo-random draws.

   .. code-block:: cpp

      geodex::Euclidean<3, geodex::EuclideanStandardMetric<3>, geodex::HaltonSampler> R3_halton;

.. note::

   For SE(2), ``random_point()`` samples the position :math:`(x, y)` uniformly within the configured sampling bounds (default: :math:`[0, 10]^2`) and the heading :math:`\theta` uniformly from :math:`[-\pi, \pi)`.
   The bounds can be configured via the constructor.


The Torus and SE(2)
-------------------

The sphere and Euclidean space are the simplest manifolds to work with.
The torus and SE(2) introduce two important complications: **periodic coordinates** and **Lie group structure**.

The Flat Torus
^^^^^^^^^^^^^^

The :math:`n`-torus :math:`\mathbb{T}^n` is the product of :math:`n` circles.
Points are represented as angle vectors in :math:`[0, 2\pi)^n`.
The key difference from Euclidean space is that coordinates **wrap around**: the angles :math:`0` and :math:`2\pi` represent the same point.

The exponential map adds the tangent vector and wraps the result:

.. math::

   \exp_p(v) = (p + v) \bmod 2\pi.

The logarithmic map computes the shortest signed difference for each coordinate:

.. math::

   \bigl(\log_p(q)\bigr)_i = \mathrm{wrap}_{[-\pi,\pi)}\!\bigl(q_i - p_i\bigr).

This wrapping ensures that the log always returns the *shortest* path around each circle.
Consider two points on the 2-torus that are close together when wrapping is accounted for but far apart in raw coordinates:

.. tabs::

   .. code-tab:: c++

      geodex::Torus<2> T2;

      Eigen::Vector2d p{0.1, 0.2};
      Eigen::Vector2d q{6.1, 0.5};
      
      Eigen::IOFormat fmt(4, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");

      // log wraps to the shortest path
      Eigen::Vector2d v = T2.log(p, q);
      std::cout << "v = " << v.transpose().format(fmt) << "\n";  // ≈ [-0.283, 0.3]
      // The first component wraps: 6.1 - 0.1 = 6.0 → wrap to ≈ -0.283

      // exp follows the tangent vector and wraps back to [0, 2π)
      Eigen::Vector2d q_recovered = T2.exp(p, v);
      std::cout << "q_recovered = " << q_recovered.transpose().format(fmt) << "\n";  // ≈ [6.1, 0.5]

   .. code-tab:: py

      T2 = geodex.Torus(2)

      p = np.array([0.1, 0.2])
      q = np.array([6.1, 0.5])

      # log wraps to the shortest path
      v = T2.log(p, q)
      print("v =", v)  # ≈ [-0.283, 0.3]
      # The first component wraps: 6.1 - 0.1 = 6.0 → wrap to ≈ -0.283

      # exp follows the tangent vector and wraps back to [0, 2π)
      q_recovered = T2.exp(p, v)
      print("q_recovered =", q_recovered)  # ≈ [6.1, 0.5]

SE(2) — Rigid-Body Poses
^^^^^^^^^^^^^^^^^^^^^^^^^

The special Euclidean group :math:`\mathrm{SE}(2) = \mathbb{R}^2 \rtimes \mathrm{SO}(2)` describes rigid-body poses in the plane.
A pose is represented as :math:`(x, y, \theta)` where :math:`(x, y)` is the position and :math:`\theta \in [-\pi, \pi)` is the heading.

Unlike the torus, SE(2) is a **Lie group**: the coupling between translation and rotation means that moving "forward" depends on your current heading.
The exponential map on SE(2) uses the Lie group exponential, which accounts for this coupling:

.. math::

   \exp_p(v) = p \cdot \mathrm{Exp}(v),

where :math:`\mathrm{Exp}` is the matrix exponential at the identity and :math:`\cdot` denotes group composition (left translation).
Concretely, the tangent vector :math:`v = (v_x, v_y, \omega)` represents a body-frame velocity, and the matrix :math:`V(\omega)` maps it to a group displacement.

The default metric on SE(2) is the **left-invariant metric** with diagonal weights :math:`(w_x, w_y, w_\theta)`:

.. math::

   \langle u, v \rangle = w_x u_x v_x + w_y u_y v_y + w_\theta u_\theta v_\theta.

With unit weights, the metric treats translational and rotational displacements equally.
In practice, you may want to penalize rotation more heavily (or vice versa) to reflect your robot's kinematic preferences.

.. tabs::

   .. code-tab:: c++

      geodex::SE2<> se2;

      Eigen::Vector3d p{1.0, 2.0, 0.0};   // pose at (1,2), heading east
      Eigen::Vector3d q{3.0, 4.0, 1.57};  // pose at (3,4), heading north
      
      Eigen::IOFormat fmt(4, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");

      Eigen::Vector3d v = se2.log(p, q);
      std::cout << "log: " << v.transpose().format(fmt) << "\n";

      Eigen::Vector3d q_recovered = se2.exp(p, v);
      std::cout << "recovered: " << q_recovered.transpose().format(fmt) << "\n";

      double d = se2.distance(p, q);
      std::cout << "distance: " << d << "\n";

   .. code-tab:: py

      se2 = geodex.SE2()

      p = np.array([1.0, 2.0, 0.0])   # pose at (1,2), heading east
      q = np.array([3.0, 4.0, 1.57])  # pose at (3,4), heading north

      v = se2.log(p, q)
      print("log:", v)

      q_recovered = se2.exp(p, v)
      print("recovered:", q_recovered)

      d = se2.distance(p, q)
      print("distance:", d)

A particularly useful trick is to set a large lateral weight :math:`w_y`.
This heavily penalizes sideways motion in the body frame, mimicking the non-holonomic constraint of a car-like robot that can only drive forward or backward:

.. tabs::

   .. code-tab:: c++

      // Large w_y penalizes lateral (sideways) motion - car-like behavior
      geodex::SE2LeftInvariantMetric car_metric{1.0, 100.0, 1.0};
      geodex::SE2<> se2_car{car_metric};

      Eigen::Vector3d p{0.0, 0.0, 0.0};  // facing east
      Eigen::Vector3d q{2.0, 2.0, 0.0};  // same heading, offset diagonally

      // Geodesics now prefer to turn-and-drive rather than slide sideways
      double d = se2_car.distance(p, q);
      std::cout << "car-like distance: " << d << "\n";

   .. code-tab:: py

      # Large wy penalizes lateral (sideways) motion - car-like behavior
      se2_car = geodex.SE2(wx=1.0, wy=100.0, wtheta=1.0)

      p = np.array([0.0, 0.0, 0.0])  # facing east
      q = np.array([2.0, 2.0, 0.0])  # same heading, offset diagonally

      # Geodesics now prefer to turn-and-drive rather than slide sideways
      d = se2_car.distance(p, q)
      print("car-like distance:", d)

With :math:`w_y = 100`, a direct sideways displacement is 10x more expensive than driving forward.
The geodesic will instead steer and drive, a smooth approximation to Dubins-like paths, without any explicit non-holonomic constraint (see :cite:`kyaw2026geometry,belta2002euclidean`).


Changing the Metric
-------------------

The metric is a swappable policy, i.e., changing it changes what "distance" and "shortest path" mean on the manifold, without touching the underlying manifold structure.
This is one of the most powerful ideas in geodex: the same set of points, the same topology, but a completely different geometry.

ConstantSPDMetric
^^^^^^^^^^^^^^^^^

The simplest way to change the metric is with ``ConstantSPDMetric<N>``, which defines a point-independent inner product using a symmetric positive-definite (SPD) matrix :math:`A`:

.. math::

   \langle u, v \rangle = u^\top A\, v.

For example, setting :math:`A = \mathrm{diag}(4, 1, 1)` makes the first coordinate "count" four times as much as the others when measuring lengths:

.. tabs::

   .. code-tab:: c++

      Eigen::Matrix3d A = Eigen::Vector3d(4, 1, 1).asDiagonal();
      geodex::ConstantSPDMetric<3> weighted{A};

      geodex::Euclidean<3, geodex::ConstantSPDMetric<3>> R3_weighted{weighted};

      Eigen::Vector3d p{0, 0, 0};
      Eigen::Vector3d u{1, 0, 0};
      Eigen::Vector3d v{0, 1, 0};

      double ip = R3_weighted.inner(p, u, v);  // 0.0  (still orthogonal)
      double n  = R3_weighted.norm(p, u);      // 2.0  (scaled by √4)

      Eigen::Vector3d q{1, 1, 1};
      double d = R3_weighted.distance(p, q);
      std::cout << "d = " << d << "\n";  // √(4 + 1 + 1) = √6 ≈ 2.449

   .. code-tab:: py

      A = np.diag([4., 1., 1.])
      weighted = geodex.ConstantSPDMetric(A)

      R3_weighted = geodex.ConfigurationSpace(geodex.Euclidean(3), weighted)

      p = np.array([0., 0., 0.])
      u = np.array([1., 0., 0.])
      v = np.array([0., 1., 0.])

      ip = R3_weighted.inner(p, u, v)  # 0.0  (still orthogonal)
      n  = R3_weighted.norm(p, u)      # 2.0  (scaled by √4)

      q = np.array([1., 1., 1.])
      d = R3_weighted.distance(p, q)
      print("d =", d)  # √(4 + 1 + 1) = √6 ≈ 2.449

Compare this to the standard metric, where the distance from the origin to :math:`(1,1,1)` is :math:`\sqrt{3} \approx 1.732`.
The weighted metric stretches space along the first axis, making that direction "more expensive" to traverse.

.. note::

   In C++, the metric is a compile-time template parameter, that is, ``Euclidean<3, ConstantSPDMetric<3>>``
   constructs a manifold with the custom metric baked in.
   In Python, dimension and metric are runtime values, so use
   ``ConfigurationSpace(base_manifold, metric)`` to compose any base manifold's topology with
   any metric.
   The resulting object supports all the same operations (``inner``, ``norm``, ``distance``,
   ``exp``, ``log``, ``geodesic``); geometry comes from the metric, topology from the base.

Metrics Are Manifold-Agnostic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because metrics are standalone policy types, the same ``ConstantSPDMetric<3>`` can be used on different manifolds.
Here we use it on the sphere:

.. tabs::

   .. code-tab:: c++

      Eigen::Matrix3d A = Eigen::Vector3d(4, 1, 1).asDiagonal();
      geodex::ConstantSPDMetric<3> weighted{A};

      geodex::Sphere<2, geodex::ConstantSPDMetric<3>> sphere_weighted{weighted};

      Eigen::Vector3d p{0, 0, 1};
      Eigen::Vector3d u{1, 0, 0};

      double n = sphere_weighted.norm(p, u);
      std::cout << "n = " << n << "\n";  // 2.0  (same weighting)

   .. code-tab:: py

      A = np.diag([4., 1., 1.])
      weighted = geodex.ConstantSPDMetric(A)

      sphere_weighted = geodex.ConfigurationSpace(geodex.Sphere(), weighted)

      p = np.array([0., 0., 1.])
      u = np.array([1., 0., 0.])

      n = sphere_weighted.norm(p, u)
      print("n =", n)  # 2.0  (same weighting)

The metric changes how we measure lengths on the sphere, but the sphere's topology and retraction (how we step along it) remain unchanged.

.. tip::

   The metric determines *what we measure* (inner products, norms, distances), while the retraction determines *how we move* (exp/log).
   These are independent choices.
   The next section explores swapping the retraction.


Swapping Retractions
--------------------

The exponential and logarithmic maps can be expensive to compute, especially on manifolds where they involve transcendental functions.
A **retraction** is a cheaper approximation that agrees with the true exponential map to first order (or higher).
In geodex, the retraction is a separate policy type that can be swapped independently of the metric.

On the sphere, the **projection retraction** simply normalizes :math:`p + v` instead of tracing a great circle:

.. math::

   R_p(v) = \frac{p + v}{\|p + v\|}.

This is cheaper than the true exponential map (no trigonometric functions) but only a first-order approximation:

.. tabs::

   .. code-tab:: c++

      using ProjSphere = geodex::Sphere<2, geodex::SphereRoundMetric,
                                        geodex::SphereProjectionRetraction>;
      ProjSphere sphere;

      Eigen::Vector3d p{0, 0, 1};
      Eigen::Vector3d q{1, 0, 0};
      
      Eigen::IOFormat fmt(4, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");

      // log + exp with projection retraction: approximate round trip
      auto v       = sphere.log(p, q);
      auto q_approx = sphere.exp(p, v);
      std::cout << "q_approx = " << q_approx.transpose().format(fmt) << "\n";  // ≈ [0.707, 0, 0.707]

   .. code-tab:: py

      sphere = geodex.Sphere(retraction="projection")

      p = np.array([0., 0., 1.])
      q = np.array([1., 0., 0.])

      # log + exp with projection retraction: approximate round trip
      v = sphere.log(p, q)
      q_approx = sphere.exp(p, v)
      print("q_approx =", q_approx)  # ≈ [0.707, 0, 0.707]

The metric is unchanged, only the retraction is swapped.
Calls to ``inner()`` and ``norm()`` still use ``SphereRoundMetric``.

SE(2) Retractions
^^^^^^^^^^^^^^^^^

SE(2) offers two retraction policies:

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Retraction
     - Order
     - Description
   * - ``SE2ExponentialMap``
     - Exact
     - True Lie group exponential using :math:`V(\omega)` matrix (default)
   * - ``SE2EulerRetraction``
     - 1st
     - Component-wise addition with angle wrapping; treats SE(2) as :math:`\mathbb{R}^2 \times S^1`

Creating SE(2) with the Euler retraction follows the same pattern as the sphere:

.. tabs::

   .. code-tab:: c++

      // SE(2) with Euler retraction (1st order, cheapest)
      using SE2Euler = geodex::SE2<geodex::SE2LeftInvariantMetric,
                                   geodex::SE2EulerRetraction>;
      SE2Euler se2_euler;

   .. code-tab:: py

      # SE(2) with Euler retraction (1st order, cheapest)
      se2_euler = geodex.SE2(retraction="euler")

.. note::

   The choice of retraction affects the accuracy of ``distance()`` and ``geodesic()`` because both operations build on exp/log internally.
   When high accuracy matters, use the true exponential map.
   When speed matters more than exactness (e.g. inside a motion planning algorithm that calls ``distance()`` millions of times), a cheaper retraction may be preferable.


Summary and Next Steps
----------------------

We have covered the fundamental operations of geodex:

- **Creating manifolds** — ``Sphere<>``, ``Euclidean<N>``, ``Torus<N>``, and ``SE2<>``
- **Riemannian inner product** — ``inner(p, u, v)`` and ``norm(p, v)``
- **Exponential and logarithmic maps** — ``exp(p, v)`` and ``log(p, q)``
- **Geodesic distance** — ``distance(p, q)``
- **Geodesic interpolation** — ``geodesic(p, q, t)``
- **Random sampling** — ``random_point()``
- **Swapping metrics and retractions** — policy-based template parameters

The table below collects all operations for quick reference:

.. list-table::
   :header-rows: 1
   :widths: 28 38 34

   * - Operation
     - Code
     - Math
   * - Inner product
     - ``m.inner(p, u, v)``
     - :math:`\langle u, v \rangle_p`
   * - Norm
     - ``m.norm(p, v)``
     - :math:`\|v\|_p`
   * - Exponential map
     - ``m.exp(p, v)``
     - :math:`\exp_p(v)`
   * - Logarithmic map
     - ``m.log(p, q)``
     - :math:`\log_p(q)`
   * - Distance
     - ``m.distance(p, q)``
     - :math:`d(p, q)`
   * - Geodesic
     - ``m.geodesic(p, q, t)``
     - :math:`\exp_p(t \cdot \log_p(q))`
   * - Random point
     - ``m.random_point()``
     -
   * - Dimension
     - ``m.dim()``
     - :math:`\dim(\mathcal{M})`

To see these operations in action on a real robotics problem, continue to the :doc:`minimum-energy-planning` tutorial, where we use position-dependent metrics to plan minimum-energy motions for a planar manipulator.

For the full API reference, see :doc:`/api/index`.

References
----------

.. bibliography::
   :filter: docname in docnames
