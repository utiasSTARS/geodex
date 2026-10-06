Path Smoothing
==============

A sampling-based planner returns a feasible path with detours and sharp corners.
:py:func:`geodex.smooth_path` (:cpp:func:`geodex::algorithm::smooth_path` in C++) shortens a
feasible path under the manifold's metric, rounds its corners into smooth curves, checks the
result and returns it at evenly spaced waypoints. By default, ``plan`` runs it on every
solution (see :doc:`planning`), and it also runs on its own on any path.

The smoother takes a manifold, a validity function and the path.

.. code-pair:: concepts/smoothing standalone

It prints ``collision_free=True length=4.3259 waypoints=268``.

.. plotly-figure:: smoothing-example
   :alt: A zig-zag path around two discs and the returned path, which passes over the first
         disc and runs straight to the goal.

   The input path of the snippet (dotted) and the returned path with its 268 waypoints.

The result holds

- ``path``, the returned waypoints,
- ``length``, the length of the path under the metric,
- ``collision_free``, whether the path passed the collision check,
- ``profile``, counts and timings of the work.

The smoother shortens the path with shortcuts, slides the waypoints to pull the path tight,
rounds the corners, checks the result and spaces its waypoints evenly. The sections below
follow these steps on the path above.

.. code-pair:: concepts/smoothing profile

It prints ``6 -> 268 waypoints, 11 shortcuts, 94 waypoint moves, 14639 validity checks,
fallback stage 0``.

Shortcutting
------------

A **shortcut** replaces the part of the path between two waypoints with the geodesic between
them. The smoother takes a shortcut when the geodesic is valid and shorter under the metric.

In its first round on a path of at most 32 waypoints, such as a planner's path, the smoother
tries the shortcuts in order of the length they save. It takes the first valid one, then
starts over on the shorter path, and stops when no valid shortcut is left. These shortcuts do
not use random numbers. In later rounds and on longer paths, the smoother tries shortcuts
between random pairs of waypoints. A pseudo-random generator (``std::mt19937_64``, not the
Halton sampler of the manifold) picks the pairs, and ``seed`` fixes it. The same input and
settings give the same path on every run.

.. video-figure:: smoothing-shortcut
   :width: 90%
   :alt: The zig-zag input path between two discs. Two long shortcuts cross a disc and fail,
         and the third, from the second waypoint to the goal, is valid and replaces the
         waypoints between its ends.

   The first round of shortcuts on the input path. The two shortcuts that save the most
   length cross a disc (orange), and the third is valid (green) and replaces the three
   waypoints between its ends.

Sliding the waypoints
---------------------

After the shortcuts, the path is a chain of edges between waypoints, and it can pass farther
from an obstacle than it needs to. The smoother splits each edge into short pieces, then moves
the waypoints one at a time to lower the path energy

.. math::

   \sum_k \|\log_{q_k} q_{k+1}\|^2_{q_k},

the sum of the squared metric lengths of the edges. A move stays only when the waypoint and
both of its edges stay valid. The waypoints slide along the obstacles and pull the path tight,
and the path bends where the metric makes a bend shorter. The smoother repeats the shortcuts
and the sliding for up to three rounds, and it stops early after a round whose shortcuts
remove no waypoint.

.. plotly-figure:: smoothing-shorten
   :alt: The zig-zag input path and the shortened path, a polyline of twelve waypoints that
         wraps around the top of the first disc and runs straight to the goal.

   The input path (dotted) and the path after shortcutting and sliding, with corner rounding
   off. The path has 12 waypoints.

The sliding needs tangent vectors with one coordinate per dimension of the manifold. On the
sphere, whose tangent vectors have three coordinates, the smoother skips the sliding.

Corner rounding
---------------

The shortened path changes direction at once at each waypoint, and a time parameterization
such as TOTG or TOPP-RA slows the robot down at each of these corners. With ``round_corners``
on (the default), the smoother replaces each corner with a smooth curve.

The rounding curve
^^^^^^^^^^^^^^^^^^

At a corner :math:`c` between the waypoints :math:`a` and :math:`b`, the tangent vectors
:math:`p = \log_c a` and :math:`q = \log_c b` point along the two edges. The curve is

.. math::

   \gamma(t) = \exp_c\Big(\sum_{i=0}^{5} B_i^5(t)\, P_i\Big), \qquad
   P_0, P_1, P_2 = f p,\ \tfrac{2}{3} f p,\ \tfrac{1}{3} f p, \qquad
   P_3, P_4, P_5 = \tfrac{1}{3} g q,\ \tfrac{2}{3} g q,\ g q,

a quintic Bézier curve with Bernstein polynomials :math:`B_i^5` and :math:`t \in [0, 1]`. Its
six control points lie on the two edges. :math:`f` and :math:`g` place the first and the last
control point at the same metric length from :math:`c`, the **size** of the curve. The curve
leaves the incoming edge in the edge's direction and joins the outgoing edge the same way,
with a zero second derivative at both ends.

.. plotly-figure:: smoothing-corners
   :alt: Left, two dotted edges that turn by 60 degrees, six orange control points on them and
         a blue curve from the first to the last. Right, two edges that turn by 150 degrees,
         with the corner marked.

   Left, a turn of 60 degrees, its edges and the rounding curve with its control points.
   Right, a turn of 150 degrees, sharper than ``corner_max_angle``, which stays a corner.

The curve uses only the manifold's ``exp`` and ``log``, and it is :math:`C^2` on the
manifold. On Euclidean spaces, the torus and the joint spaces of the built-in robots, it is a
quintic polynomial in the coordinates.

When a curve stays
^^^^^^^^^^^^^^^^^^

The smoother first tries the largest curve the two edges allow. A curve can use the whole of
an edge whose other end is the start, the goal or a cusp. Two neighboring curves share the
edge between them in proportion to their turning angles. The smoother keeps the curve when it
passes three tests.

1. **Collision.** The curve passes the same checks as the rest of the path (see
   `The check`_).
2. **Length.** The curve is at most 0.1 percent longer under the metric than the part of the
   corner it replaces.
3. **Plane.** The curve stays in the plane of the two edges, within 0.2 percent of its
   motion. On a differential-drive base, a corner between two edges without sideways motion
   gets a curve without sideways motion. This test applies where the tangent vectors have one
   coordinate per dimension.

A curve that fails a test shrinks by a factor of two and is tested again, up to five times.
Once a smaller size passes, the smoother tests two sizes between it and the last size that
failed, and it keeps the largest size that passes. When every size fails, the corner stays.

.. plotly-figure:: smoothing-shrink
   :alt: Two edges that turn by 70 degrees with a small disc inside the corner. The full curve
         and the curve of half its size cross the disc. The curve of a quarter of the full size
         passes, and the kept curve, a little larger, passes between the disc and the corner.

   A corner next to an obstacle. The full curve and the curve of half its size enter the
   obstacle. The curve of a quarter of the full size passes, and of the two sizes tested
   between a quarter and a half, the larger also passes and stays, at about 0.42 of the full
   size.

``profile.rounded_corners`` counts the corners with a curve, ``profile.kept_corners`` the
corners without one, and ``profile.rounding_retries`` the curve sizes that failed.

.. code-pair:: concepts/smoothing corners

It prints ``rounded=8 kept=0 cusps=0 points=268, without rounding 12``.

Rounding settings
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 24 13 63

   * - Setting
     - Default
     - Meaning
   * - ``round_corners``
     - ``True``
     - Round the corners. Turn it off when the controller rounds corners itself.
   * - ``corner_max_angle``
     - :math:`\pi/2`
     - Largest turning angle of a rounded corner, in radians under the metric. A sharper
       corner is a **cusp** and stays a corner (``profile.cusps``). A base that drives forward
       and then backs up along a line turns by :math:`\pi` there.
   * - ``corner_tolerance``
     - :math:`10^{-4}`
     - The smoother approximates a curve with short edges between points on it. No edge
       strays farther from the curve than this distance, in the coordinates of the manifold
       (meters or radians). A corner whose curve would be smaller than this distance stays.
       The same distance sets the step of the even spacing (see `Even spacing`_).
   * - ``sharp_coordinates``
     - 0
     - Number of leading tangent coordinates that may keep a corner while the others follow a
       curve. See below.

``sharp_coordinates`` is the number of leading tangent coordinates that may keep a corner
while the other coordinates follow a curve. On a mobile manipulator, for example, the base
pose can turn in place at a corner while the arm moves along a curve. Where the full curve has
to shrink, and at a cusp, the smoother also tries a curve on which the leading coordinates run
along the two edges and turn at the corner. It keeps the larger curve, and
``profile.split_corners`` counts these curves. The manifold's ``exp`` and ``log`` must act on
the leading coordinates apart from the others, as on a product space. A value of at least the
number of coordinates rounds every coordinate together.

.. note::

   A mobile base can reverse, driving forward and then backing up. The smoother keeps a
   reversal. After the sliding step, a reversal can become several small corners. Rounding
   them gives a curve on which the base switches from driving forward to backing up without
   stopping. To forbid reverse driving, bound it with a motion validator or a path predicate
   (see `The check`_).

The check
---------

``collision_free`` is true when every waypoint of the smoothed path passes the validity
function and every edge passes the edge check. The default edge check tests points along the
geodesic between two waypoints, no more than ``collision_check_resolution`` apart in the
coordinates. A resolution of 0 uses one hundredth of the coordinate length of the input path.

.. plotly-figure:: smoothing-check
   :alt: A close-up of the shortened path next to the edge of the first disc, with blue
         waypoints and small gray crosses between them.

   Edges of the shortened path next to the first disc, with rounding off. The edge check tests
   the waypoints and the points between them (crosses), no more than
   ``collision_check_resolution`` apart, 0.01 here.

When the smoothed path fails the check, the smoother returns an earlier stage that passes, and
``profile.fallback`` says which one.

- 0, the smoothed path.
- 1, the path before the even spacing of ``output_spacing``.
- 2, the path after the first round of shortcuts.
- 3, the input path.

The check tests points and not the space between them. A path can touch an obstacle between
two tested points, and a smaller ``collision_check_resolution`` narrows that gap. The returned
waypoints lie on the checked path, and the edges between them stay within
``corner_tolerance`` of it (see `Even spacing`_). To keep those edges as clear as the checked
path, grow the obstacles in the validity function by a small margin, the largest displacement
of any point of the robot over a change of ``corner_tolerance`` in its configuration. For a
mobile base, that is a footprint a fraction of a millimeter larger. For a robot in a
VAMP scene, ``plan`` adds this margin itself (see `Inside plan`_).

Four optional settings change the edge check.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Setting
     - Meaning
   * - ``edge_provably_clear``
     - A function ``(a, b) -> bool``. ``True`` proves the whole edge valid and skips its
       points, and ``False`` leaves the edge to the point check. A clearance bound from a
       distance field is a typical proof.
   * - ``edge_validator``
     - A function ``(a, b) -> bool`` that replaces the point check, such as a planner's motion
       validator. The smoother calls it with the two ends of each edge in path order, and an
       asymmetric check, such as a bound on reverse driving, sees the direction of travel.
       The evenly spaced edges do not pass through it. Give an asymmetric check also as
       ``path_predicate``.
   * - ``path_predicate``
     - A function ``(waypoints) -> bool`` on the whole path. The smoother rejects every change
       that fails it, such as one that exceeds a budget of reverse driving. Every returned
       path except the unchanged input passes it.
   * - ``edge_travel``
     - A function ``(a, b) -> float``, a bound on how far the checked geometry moves along the
       edge, in the units of ``collision_check_resolution``. The smoother spaces the tested
       points by this bound instead of the coordinate length of the edge. The bound must
       hold for every part of the edge in proportion to its share, as a bound on the speed
       does. A ``collision_check_resolution`` of 0 ignores it.

A correct proof skips only points that would pass, and the smoother returns the same path
with or without it.

.. code-pair:: concepts/smoothing proof

It prints ``same path=True edges settled by the proof=665``.

In C++, a validity functor that also has ``batch(points, n)`` and ``batch_size()`` (the
:cpp:concept:`geodex::algorithm::BatchValidity` concept) receives the edge points in blocks,
the form SIMD collision checkers such as VAMP take.

Even spacing
------------

The smoother returns the path as waypoints at equal steps along it. The start, the goal and
every corner that stays keep their place, and the other waypoints lie at equal steps between
them. The step is the longest one for which the edge between two neighboring waypoints stays
within ``corner_tolerance`` of the smoothed path, and the tightest curve of the path sets it.
``output_spacing`` caps the step when a controller needs waypoints closer together. The
smoother does not check the evenly spaced path again, and its waypoints lie on the checked
path.

.. plotly-figure:: smoothing-spacing
   :alt: Three close-ups of the path over the first disc. Left, the default spacing with many
         waypoints. Middle, fewer waypoints with a larger corner tolerance. Right, the most
         waypoints with an output spacing of 0.01.

   The waypoints over the first disc. Left, the default settings. Middle, a corner tolerance
   of 0.001. Right, an ``output_spacing`` of 0.01.

Around the two discs, the default ``corner_tolerance`` of :math:`10^{-4}` gives a step of
0.016 and 268 waypoints. A tolerance of :math:`10^{-3}` gives a step of 0.052 and 85
waypoints, and an ``output_spacing`` of 0.01 gives a step of 0.01 and 434 waypoints. A larger
tolerance also needs a larger margin around the obstacles (see `The check`_).

.. code-pair:: concepts/smoothing spacing

It prints

.. code-block:: text

   corner_tolerance=0.001 output_spacing=0: waypoints=85 step=0.0515
   corner_tolerance=0.0001 output_spacing=0.01: waypoints=434 step=0.0100

Every default of the smoother except ``corner_tolerance`` is unitless. The default
``corner_tolerance`` of :math:`10^{-4}` works for paths in meters and in radians.

Settings
--------

.. list-table::
   :header-rows: 1
   :widths: 27 14 59

   * - Field
     - Default
     - Meaning
   * - ``collision_check_resolution``
     - 0
     - Largest spacing of the tested points along an edge, in the units of ``edge_travel``
       when that is set. 0 uses one hundredth of the coordinate length of the input path.
   * - ``output_spacing``
     - 0
     - Longest step between the returned waypoints. 0 leaves the step without a limit.
   * - ``seed``
     - 42
     - Seed of the random shortcuts.
   * - ``round_corners``
     - ``True``
     - Round the corners. See `Rounding settings`_.
   * - ``corner_tolerance``
     - :math:`10^{-4}`
     - Largest distance between a curve and its approximating edges, and between the
       checked path and the evenly spaced edges. See `Rounding settings`_.
   * - ``corner_max_angle``
     - :math:`\pi/2`
     - Largest turning angle of a rounded corner. See `Rounding settings`_.
   * - ``sharp_coordinates``
     - 0
     - Leading coordinates that may keep a corner. See `Rounding settings`_.
   * - ``edge_provably_clear``
     - ``None``
     - Proof that a whole edge is valid. See `The check`_.
   * - ``edge_validator``
     - ``None``
     - Edge check that replaces the point check. See `The check`_.
   * - ``path_predicate``
     - ``None``
     - Check of the whole path. See `The check`_.
   * - ``edge_travel``
     - ``None``
     - Bound on how far the checked geometry moves along an edge. See `The check`_.

Inside ``plan``
---------------

``plan`` passes the planner's path to the smoother with the same validity function and the
declared limits. The smoother checks edges at ``PlanSettings.collision_check_resolution``.
When that is 0, it uses ``PlanSettings.smoothing.collision_check_resolution``, or a hundredth
of the diagonal of the bounds when both are 0. A motion validator also checks the smoother's
edges, except when the validity function checks blocks of points. With
:py:class:`geodex.DirectionalMotionValidator`, every edge of the returned path also stays within
the reverse budget. When the planner's edges follow curves, such as discrete geodesics of a
custom metric, the smoother receives the planner's path densified along those curves.
``PlanSettings.smoothing`` holds the smoother's settings, and ``PlanSettings.smooth = False``
turns it off.

For a built-in robot in a VAMP scene, ``plan`` grows every obstacle by a small margin, the
largest displacement of a collision sphere over a change of ``corner_tolerance`` in the
configuration. The smoother checks each edge at configurations close enough that every
collision sphere moves at most ``PlanSettings.collision_check_resolution`` from one check to
the next, and 0 means 5 mm. ``plan`` sets ``edge_travel`` to a bound on the sphere travel of
each edge.

The snippet below plans for a differential-drive robot on :math:`\mathrm{SE}(2)` with an
``output_spacing`` of 0.2.

.. code-pair:: concepts/smoothing in-plan

It prints ``smoothed=True raw length=5.054 smoothed length=4.857``.

.. plotly-figure:: smoothing-in-plan
   :alt: The planner's path of a differential-drive robot between the two discs, dotted, and
         the returned path in blue with arrows for the heading.

   The planner's path (dotted) and the returned path, both drawn along
   :math:`\mathrm{SE}(2)` geodesics. The arrows show the heading of the robot.

See also
--------

- :doc:`planning` for the planner that produces the raw path.
- :doc:`metrics` for the metrics the smoother shortens paths under.
- :doc:`/tutorials/se2-planning` for smoothing with footprints, clearance metrics and a
  directional motion validator.
