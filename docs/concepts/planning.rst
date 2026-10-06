Planning
========

:py:func:`geodex.plan` (:cpp:func:`geodex::planning::plan` in C++) turns a manifold, a validity
function, a start and a goal into a collision-free path that is short under the manifold's
metric. It sets up the planning problem, runs the planner and smooths the result. The same
function plans on the sphere, on :math:`\mathrm{SE}(2)` and in the joint space of an arm. This
page covers what ``plan`` does, its settings, and how an admissible heuristic focuses the
search.

A first plan
------------

The sphere below has two cap-shaped obstacles between the start and the goal. The validity
function returns ``True`` outside both caps, and ``plan`` searches for a path with 2000
iterations and seed 7.

.. code-pair:: concepts/planning first-plan

It prints ``solved=True cost=1.3725 waypoints=77``.

.. robot-scene:: planning-sphere
   :alt: A path on the unit sphere from a green start to an orange goal that passes between
         two dark blue caps.

   The path that ``plan`` returns passes between the two caps. Drag to orbit, and use the
   controls to pause the ball or move it along the path.

``result.path`` holds the smoothed path at evenly spaced waypoints. Draw it or hand it to a
controller as it is. ``result.raw_path`` holds the planner's waypoints before smoothing.

.. code-pair:: concepts/planning result

It prints ``raw waypoints=6 smoothed=True`` and the times of the search and of the smoother.

What ``plan`` does
------------------

.. mermaid::

   %%{init: {'theme':'base','themeVariables':{'primaryColor':'#e7f0fa','primaryTextColor':'#1a1a1a','primaryBorderColor':'#2980b9','lineColor':'#2980b9','secondaryColor':'#e7f0fa','tertiaryColor':'#f7fbfe','background':'transparent'}}}%%
   flowchart LR
       A["manifold<br/>start, goal"] --> B["state space<br/>and bounds"]
       B --> C["validity and<br/>edge checks"]
       C --> D["path length<br/>and heuristic"]
       D --> E["G-RRT*"]
       E --> F["planner's<br/>path"]
       F --> G["smoother"]
       G --> H["PlanResult"]

``plan`` wraps the manifold in an OMPL state space. Its bounds are the manifold's sampling
bounds, grown to hold the start and the goal. The planner takes its samples from the
manifold's own sampler and checks every edge (see `Edges and motion validators`_). The cost of
an edge is its length under the metric.

The planner is **G-RRT\*** :footcite:`kyaw2026greedy`, an asymptotically optimal,
bidirectional variant of RRT\*. It finds a first path, then keeps sampling to shorten it, and
uses a heuristic to sample only where a shorter path can pass (see
`Heuristics and informed sampling`_). The search runs for ``iterations`` iterations when that
is set, and for ``time`` seconds otherwise.

The smoother then shortens the planner's path, rounds its corners and checks the result
against the same validity function (see :doc:`smoothing`). ``path`` holds the smoother's output
when it passes the check, and the planner's path otherwise.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``solved``
     - ``True`` when the planner found a path to the goal within its budget.
   * - ``path``
     - The returned path as evenly spaced waypoints.
   * - ``raw_path``
     - The planner's waypoints before smoothing.
   * - ``smoothed``
     - ``True`` when ``path`` is the smoother's output.
   * - ``cost``
     - The length of ``path`` under the metric.
   * - ``time_ms``
     - Time of the search in milliseconds.
   * - ``smooth_ms``
     - Time of the smoother in milliseconds.
   * - ``first_solution_ms``
     - Time until the search found its first path, and -1 when it found none.
   * - ``first_solution_iterations``
     - Iterations until the search found its first path, and 0 when it found none.
   * - ``informed_samples``
     - Samples from the informed set, the samples from the greedy set included.
   * - ``focused_samples``
     - Samples from the greedy set.
   * - ``uniform_samples``
     - Samples from the whole space, taken before the first path or when the informed set is
       empty.

Settings
--------

Every field of :py:class:`geodex.PlanSettings` (:cpp:struct:`geodex::planning::PlanSettings`
in C++) has a default that solves most problems.

.. code-pair:: concepts/planning settings

.. list-table::
   :header-rows: 1
   :widths: 25 17 58

   * - Field
     - Default
     - Meaning
   * - ``iterations``
     - 0
     - Number of planner iterations. A value above 0 replaces the time budget, and a plan
       with a fixed seed then returns the same path on every run.
   * - ``time``
     - 1.0
     - Planning time in seconds. The planner uses it when ``iterations`` is 0.
   * - ``refine_time``
     - 0
     - Time in seconds that the planner keeps improving the path after its first path,
       within ``time``. 0 uses the whole time budget. The planner ignores it when
       ``iterations`` is set.
   * - ``seed``
     - 0
     - Seed of the plan. A fixed seed above 0 makes the plan reproducible, and 0 takes a new
       seed from the manifold's sampler. See :doc:`/getting-started/reproducibility`.
   * - ``planner``
     - ``GreedyRRTstar()``
     - The planner and its parameters (see the table below).
   * - ``collision_check_resolution``
     - 0
     - Largest distance between two validity checks along an edge, in coordinates. 0 uses
       OMPL's default for the planner. The smoother then uses
       ``smoothing.collision_check_resolution``, or a hundredth of the diagonal of the bounds
       when that is 0 too.
   * - ``interp``
     - ``"base_geodesic"``
     - The curve that joins two states. ``"base_geodesic"`` uses the manifold's
       ``geodesic``, and ``"riemannian_geodesic"`` uses the discrete geodesic of the metric
       (see :doc:`discrete-geodesic-interpolation`). ``"auto"`` uses the manifold's
       ``geodesic`` when it is a geodesic of the metric or when a motion validator checks the
       edges, and the discrete geodesic otherwise.
   * - ``goal_tolerance``
     - 0
     - How close to the goal the path must end, as a distance on the manifold. When the start
       is already this close, ``plan`` checks the edge from the start to the goal and returns
       it.
   * - ``limits``
     - ``None``
     - Lower and upper limits of the coordinates, such as joint limits. The planner and the
       smoother keep every state inside them. Without them, the plan uses the bounds that the
       manifold declares.
   * - ``smooth``
     - ``True``
     - Whether to smooth the planner's path.
   * - ``smoothing``
     - ``PathSmoothingSettings()``
     - Settings of the smoother (see :doc:`smoothing`).

``geodex.planners.GreedyRRTstar`` takes these parameters.

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Parameter
     - Default
     - Meaning
   * - ``range``
     - 0
     - Longest step that the tree grows toward a sample. 0 lets OMPL choose it from the size of
       the bounds.
   * - ``greedy_ratio``
     - 0.9
     - Probability that a sample comes from the greedy set (see
       `Heuristics and informed sampling`_).
   * - ``rewire_factor``
     - 1.1
     - Scale of the radius in which a new state looks for cheaper connections.
   * - ``greedy_cost_for_tree_pruning``
     - ``True``
     - Remove the tree states outside the greedy set when ``greedy_ratio`` is above 0.
       Otherwise, or with ``False``, remove those outside the informed set.
   * - ``max_neighbors``
     - 0
     - Largest number of neighbors that a new state connects to. 0 does not limit it.

Heuristics and informed sampling
--------------------------------

After the first solution, G-RRT\* samples only where a state can still improve the path. A
state :math:`x` can improve a solution of cost :math:`c` only if the shortest path through it
is shorter than :math:`c`. A **heuristic** :math:`h` estimates the length of that path as
:math:`h(x_s, x) + h(x, x_g)`. The heuristic must be **admissible**. An admissible heuristic
does not overestimate the true metric distance. The **informed set**
:math:`\{x : h(x_s, x) + h(x, x_g) \le c\}` then contains every state that can improve the
current solution, and the planner samples it directly :footcite:`gammell2014informed`. geodex
lowers the bound :math:`c` to the heuristic length of the current path, the sum of :math:`h`
over its edges, when that is smaller.

G-RRT\* also samples a smaller **greedy set**, the informed set whose bound is the largest
estimate :math:`h(x_s, v) + h(v, x_g)` over the states :math:`v` of the current path. This set
is the smallest informed set that still holds the current path. ``greedy_ratio`` is the
probability that a sample comes from the greedy set, 0.9 by default, and the other samples
come from the informed set.

For a constant metric :math:`G`, the chord length :math:`\sqrt{\Delta^\top G \Delta}` is the
exact distance. For a metric :math:`M(q)` that changes over the space, geodex computes a
constant matrix :math:`L` with :math:`L \preceq M(q)` everywhere in the bounds (a **Loewner
lower bound**) :footcite:`kyaw2026loewner`. Every path is at least as long under :math:`M`
as under :math:`L`. The chord length under :math:`L` is then admissible, and the informed set
is an ellipsoid the planner samples directly.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Space
     - Default heuristic
   * - Built-in robot (``geodex.robots``)
     - The Loewner bound of its mass matrix, computed once and included with the robot.
   * - :math:`\mathrm{SE}(2)`, :math:`\mathrm{SO}(2)`, the torus, and products of them
     - A Loewner bound computed when ``plan`` starts, with the periodic coordinates
       wrapped.
   * - Any other manifold
     - The Euclidean chord, admissible for identity metrics and metrics with
       :math:`M(q) \succeq I`. Pass ``heuristics.Zero()`` or a lower bound for any other
       metric.

``precompute_matrix_lower_bound`` computes the bound of a custom metric, such as the kinetic
energy of an arm. Its ``lambda_min_certificate`` is the smallest eigenvalue of
:math:`L^{-1/2} M(q) L^{-1/2}` it found over the box. ``converged`` is true when this value is
at least one within the tolerance, and the bound is then admissible.

Edges and motion validators
---------------------------

An edge of the planner is valid when every state along it is valid. By default, ``plan``
checks an edge with the validity function at points along its curve, no more than
``collision_check_resolution`` apart. A **motion validator** replaces this check and decides a
whole edge at once. Pass one as ``motion_validator=`` in Python, or as the factory argument of
``plan`` in C++, when an edge needs a check beyond the validity of its points.
:py:class:`geodex.DirectionalMotionValidator` checks the points of an edge and also rejects the
edge when it drives backward by more than a budget, a limit that a Riemannian metric cannot
impose. See :doc:`/tutorials/se2-planning` for an example with a differential-drive robot.

The smoother checks its edges with the same validator, and with a directional validator,
every edge of the returned path stays within the reverse budget.

The built-in robots take a collision scene in place of a validity function, as in
``geodex.plan(robot, q0, q1, scene)`` (``geodex::robots::plan`` in C++). Their motion validator
is VAMP's edge check, and the plan uses the robot's own heuristic. The smoother checks its
edges at points against the scene, with every obstacle grown by a small margin. See
:ref:`robot-collision-checking` for what these checks guarantee along every edge.

See also
--------

- :doc:`metrics` for the metrics that define path length.
- :doc:`smoothing` for what happens to the raw path.
- :doc:`/getting-started/reproducibility` for seeds and budgets.
- :doc:`/tutorials/minimum-energy-planning` and :doc:`/tutorials/se2-planning` for complete
  planning problems.

References
----------

.. footbibliography::
