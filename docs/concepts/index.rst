Core Concepts
=============

This section gathers the conceptual background needed to use **geodex** effectively. It
starts with the geometric vocabulary that underpins the library, then describes how
those ideas are encoded as C++20 concepts and policy types, and finally walks through
the main algorithms that consume them. Readers already comfortable with Riemannian
geometry can skip directly to the architecture and algorithm pages.

.. toctree::
   :maxdepth: 1

   riemannian-geometry
   architecture
   discrete-geodesic-interpolation

**See also**

- :doc:`/tutorials/geodex-basics` for a hands-on walk-through with runnable C++ and
  Python snippets.
- :doc:`/tutorials/minimum-energy-planning` for composing ``KineticEnergyMetric`` and
  ``JacobiMetric`` to plan minimum-energy motions.
- :doc:`/api/index` for the full reference of every class and concept named here.
