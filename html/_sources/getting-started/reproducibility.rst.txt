Reproducibility
===============

On this page, we plan one motion several times and see which settings give the same
path on every run. We use the first plan of :doc:`installation`, a differential-drive robot
driving around a disc.

.. plotly-figure:: reproducibility-budgets
   :alt: Path costs for ten seeds, ten runs each, under an iteration budget and a time budget.

   Ten runs for each of ten seeds, one point per distinct cost of a seed. Under a budget of
   2000 iterations, the ten runs of a seed return one cost (filled blue dots). Under a budget of
   0.02 s, the runs complete different numbers of iterations, and a seed can return more than
   one cost (open orange circles). A run that finds no path within the budget has no point.
   Hover a point for its cost and its number of runs.

The full script is ``examples/getting_started/reproducibility.py``, and the C++ version is
``examples/getting_started/reproducibility.cpp``.

1. Plan twice with one seed
---------------------------

:py:class:`geodex.PlanSettings` has a ``seed`` and an ``iterations`` budget. The seed
seeds OMPL's random number generator and the planner's samplers, and the budget stops the
planner after a fixed number of iterations.

.. code-pair:: getting_started/reproducibility same-seed

It prints ``same path: True same cost: True``. The two paths are equal bit for bit.

2. Change the seed
------------------

Another seed gives other samples and returns another path. To measure a planner, loop over
seeds and report the distribution.

.. code-pair:: getting_started/reproducibility other-seeds

It prints

.. code-block:: text

   seed 1: cost 4.7701
   seed 2: cost 4.7425
   seed 3: cost 4.7236

3. Plan with a time budget
--------------------------

With ``iterations`` at 0, the default, the planner stops after ``time`` seconds. The number
of iterations it completes depends on the machine and on whatever else runs on it. Two runs
return different paths when the planner improves its path between the iteration counts of the
two runs. Use a time budget on a robot that must answer in time, and an iteration budget for
tests, figures and comparisons.

.. code-pair:: getting_started/reproducibility time-budget

It prints ``time budget: cost 4.7411`` on one run, and another run can print another cost.

4. Seed the samplers outside the planner
----------------------------------------

A manifold takes its ``random_point`` samples from a low-discrepancy sampler (see
:doc:`/concepts/sampling`). A sampler constructed without a seed takes one from a shared
source, which ``geodex.seed`` (``geodex::set_default_seed`` in C++) reseeds. Every manifold
constructed afterwards samples a reproducible sequence, and manifolds that already exist keep
theirs. ``seed`` on a manifold reseeds its own sampler. In Python, a space built on a
manifold, such as a ``ConfigurationSpace``, shares the manifold's sampler.

.. code-pair:: getting_started/reproducibility sampling

5. Try it: plan without a seed
------------------------------

A ``seed`` of 0, the default, takes the plan's seed from the next sample of the manifold's own
sampler. Two unseeded plans on one space differ, and ``space.seed(3)`` resets the sampler and
repeats the sequence.

.. code-pair:: getting_started/reproducibility try-it

It prints ``costs 4.7880 and 4.8309, after space.seed(3) 4.7880``. Give each thread that
plans without a seed its own copy of the space.

Exact versions
--------------

The same seed gives the same path only with the same code. geodex builds every dependency from
one exact release or commit.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Input
     - Version
   * - Python package
     - ``pip install pygeodex==1.0.0``
   * - Conda and PyPI packages of the workspace
     - ``pixi.lock``, installed with ``pixi install --locked``
   * - OMPL fork with G-RRT*
     - one commit of ``utiasSTARS/ompl``, set in ``third_party/dependencies.cmake``
   * - VAMP and its header dependencies
     - commit ``e3902f1`` of ``KavrakiLab/vamp``, with nigh, pdqsort and SIMDxorshift at the
       commits VAMP uses
   * - Eigen
     - 5.0.0, installed with the geodex headers
   * - C++ projects
     - ``GIT_TAG v1.0.0`` for FetchContent, ``find_package(geodex 1.0 CONFIG REQUIRED)``,
       which accepts any 1.x release

Across machines
---------------

A path is the same bit for bit on one build and one kind of CPU. A different compiler or CPU
can return a different path for the same seed. To compare machines, compare costs and success
rates over many seeds, and report timings as distributions over seeds and repeated runs, with
the machine and its load.

Where to go next
----------------

- :doc:`/concepts/planning` covers every field of ``PlanSettings``.
- :doc:`/concepts/sampling` covers the samplers and their seeds.
