Core Concepts
=============

These pages cover the geometry behind **geodex**, how the library expresses it, and how the
planner and the smoother use it. Readers who know Riemannian geometry can start at
:doc:`architecture`.

.. toctree::
   :maxdepth: 1

   riemannian-geometry
   architecture
   metrics
   sampling
   discrete-geodesic-interpolation
   planning
   smoothing

**See also**

- :doc:`/tutorials/geodex-basics` for a hands-on walk-through with runnable C++ and
  Python snippets.
- :doc:`/tutorials/minimum-energy-planning` for composing ``KineticEnergyMetric`` and
  ``JacobiMetric`` to plan minimum-energy motions.
- :doc:`/api/cpp` for the C++ API reference.
- :doc:`/api/python` for the Python API reference.
