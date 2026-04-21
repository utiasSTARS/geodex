geodex
======

**geodex** is a general-purpose software framework for motion planning on Riemannian manifolds.

We ship ready-to-use manifolds (:math:`\mathbb{R}^n`, :math:`S^n`, :math:`T^n`, and :math:`\mathrm{SE}(2)`), all built from swappable metric, retraction, and sampler policies, along with efficient algorithms for geodesic distance and interpolation.

The core engine of geodex is written purely in C++20 for performance, with first-class Python support (``pip install geodex``).
We also provide integrations with popular motion planning frameworks (OMPL, VAMP) and robotics stacks (Nav2, MoveIt 2) through ROS 2.

.. raw:: html

   <div class="landing-grid">

     <a class="landing-card" href="getting-started/index.html">
       <div class="landing-card-thumb landing-card-thumb-text">
   <pre><code>cmake -B build \
     -DBUILD_TESTING=ON
   cmake --build build
   ctest --test-dir build</code></pre>
       </div>
       <div class="landing-card-body">
         <h3>Getting started</h3>
         <p>Build the library, run the tests, and wire up the OMPL and Nav2 integrations.</p>
       </div>
     </a>

     <a class="landing-card" href="concepts/index.html">
       <div class="landing-card-thumb">
         <img src="_static/landing/manifold.svg" alt="" />
       </div>
       <div class="landing-card-body">
         <h3>Core concepts</h3>
         <p>Riemannian geometry, design principles, and algorithms.</p>
       </div>
     </a>

     <a class="landing-card" href="tutorials/se2-planning.html">
       <div class="landing-card-thumb">
         <img src="_static/landing/se2_willow_corridor.svg" alt="" />
       </div>
       <div class="landing-card-body">
         <h3>SE(2) planning</h3>
         <p>Plan paths for holonomic and non-holonomic robots on a real costmap with OMPL.</p>
       </div>
     </a>

     <a class="landing-card" href="tutorials/minimum-energy-planning.html">
       <div class="landing-card-thumb">
         <img src="_static/landing/minimum_energy_arm.gif"
              alt="Two-link planar arm sweeping along a minimum-energy geodesic" />
       </div>
       <div class="landing-card-body">
         <h3>Minimum-energy planning</h3>
         <p>Compute minimum-energy motions for a two-link planar manipulator under kinetic energy and Jacobi Riemannian metrics.</p>
       </div>
     </a>

   </div>

Roadmap
-------

.. list-table::
   :header-rows: 1

   * - Integration
     - Description
     - Status
   * - `OMPL <https://ompl.kavrakilab.org/>`_ and
       `VAMP <https://github.com/KavrakiLab/vamp>`_ integrations
     - | Planning on Riemannian manifolds with
       | state-of-the-art sampling-based planners
     - In progress
   * - `Nav2 <https://nav2.org/>`_ and
       `MoveIt 2 <https://moveit.ai/>`_ plugins
     - | Geometry-aware planning for ROS 2
       | mobile robots and manipulators
     - Planned

Citation
--------

   geodex accompanies the paper
   `Geometry-Aware Sampling-Based Motion Planning on Riemannian Manifolds <https://arxiv.org/abs/2602.00992>`_,
   accepted to `WAFR 2026 <https://algorithmic-robotics.org/>`_.

If you use geodex in your research, consider citing:

.. code-block:: bibtex

   @article{kyaw2026geometry,
      title={Geometry-Aware Sampling-Based Motion Planning on Riemannian Manifolds},
      author={Kyaw, Phone Thiha and Kelly, Jonathan},
      journal={arXiv preprint arXiv:2602.00992},
      year={2026}
   }

.. toctree::
   :hidden:

   getting-started/index
   concepts/index
   tutorials/index
   api/index
