geodex
======

A general-purpose software framework for planning on Riemannian manifolds.

geodex provides:

- **C++20 concepts** for manifolds, metrics, retractions, and geodesics
- **Concrete manifold implementations**: sphere :math:`S^2`, Euclidean :math:`\mathbb{R}^n`, flat torus :math:`T^n`, SE(2)
- **Policy-based design**: manifolds parameterized by interchangeable metric and retraction policies
- **Algorithms**: geodesic distance approximation via midpoint method
- **Python bindings**: First-class support for Python (``pip install geodex``)

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
   :maxdepth: 2
   :hidden:

   getting-started/index
   concepts/index
   tutorials/index
   api/index
