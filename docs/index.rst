geodex
======

.. container:: landing-hero

   .. raw:: html

      <ul class="landing-tags" aria-label="Languages, integrations and license">
        <li>C++20</li>
        <li>Python 3.12+</li>
        <li><a href="https://pypi.org/project/pygeodex/">PyPI</a></li>
        <li>ROS 2</li>
        <li>Apache-2.0</li>
      </ul>

   .. container:: landing-lead

      **geodex** is a general-purpose software framework for motion planning on Riemannian
      manifolds.

      We ship ready-to-use manifolds (:math:`\mathbb{R}^n`, :math:`\mathbb{S}^n`,
      :math:`\mathbb{T}^n`, :math:`\mathrm{SO}(2)`, :math:`\mathrm{SO}(3)`, :math:`\mathrm{SE}(2)`,
      :math:`\mathrm{SE}(3)`, and Cartesian products of these), all built from swappable metric,
      retraction, and sampler policies, along with efficient algorithms for geodesic distance and
      interpolation.

      The core engine of geodex is written purely in C++20 for performance, with first-class Python
      support (``pip install pygeodex``). We also provide integrations with popular motion planning
      frameworks (OMPL, VAMP) and robotics stacks (Nav2, MoveIt 2) through ROS 2.

.. grid:: 1 2 2 2
   :gutter: 3
   :class-container: landing-robots

   .. grid-item::

      .. video-figure:: real-fr3-shelf
         :alt: A Franka FR3 arm moves a cracker box from the bottom compartment of a shelf to the
               top compartment and back

         A Franka FR3 moves a box between shelf compartments with the MoveIt 2 plugin under the
         kinetic-energy metric.

   .. grid-item::

      .. video-figure:: real-jackal-office
         :alt: A Clearpath Jackal drives between office desks and through a narrow passage

         A Clearpath Jackal plans on :math:`\mathrm{SE}(2)` with the Nav2 plugin and drives
         through a narrow passage of an office.

.. grid:: 1 2 3 3
   :gutter: 3
   :class-container: landing-cards

   .. grid-item-card:: Getting started
      :link: getting-started/index
      :link-type: doc
      :img-top: _static/landing/getting-started.svg
      :img-alt: Install commands for pip, pixi and CMake

      Install geodex with pip, CMake or pixi, and plan a first motion.

   .. grid-item-card:: Core concepts
      :link: concepts/index
      :link-type: doc
      :img-top: _static/landing/manifold.svg
      :img-alt: A curved surface with a tangent plane and a tangent vector at one point

      Learn the Riemannian geometry behind geodex, its design and its algorithms.

   .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc
      :img-top: _static/landing/tutorials.svg
      :img-alt: A robot path winding between walls from a start to a goal

      Start from a first manifold and plan minimum-energy and :math:`\mathrm{SE}(2)` motions.

   .. grid-item-card:: Robot guides
      :link: robots/index
      :link-type: doc
      :class-card: landing-card-media

      .. landing-video:: robots
         :alt: A Hello Robot Stretch 3 drawn with its meshes, driving a differential-drive path
         :fallback: robots/figs/landing.png
         :fallback-alt: A Stretch 3 with its meshes reaching over a kitchen island

      Plan manipulation, navigation and whole-body motions for nine built-in robots.

   .. grid-item-card:: ROS 2
      :link: ros2/index
      :link-type: doc
      :img-top: _static/landing/ros2.svg
      :img-alt: The two planner plugins, one in a Nav2 planner server and one in a MoveIt
                planning pipeline

      Plan inside Nav2 and MoveIt 2 with the two geodex planner plugins, on ROS 2 Jazzy and
      Lyrical.

   .. grid-item-card:: API reference
      :link: api/index
      :link-type: doc
      :img-top: _static/landing/api.svg
      :img-alt: A C++ and a Python signature of plan

      Look up every class and function of the Python module and the C++ headers.

Citing geodex
-------------

If you use geodex in your research, please cite the geodex paper.

.. code-block:: bibtex

   @article{kyaw2026geodex,
     title   = {geodex: A Library for Motion Planning on {Riemannian} Manifolds},
     author  = {Kyaw, Phone Thiha and Wei, Ben and Samavi, Sepehr and
                {Rogel Garcia}, Miguel Angel and Kelly, Jonathan},
     journal = {arXiv preprint arXiv:26XX.XXXXX},
     year    = {2026},
     url     = {https://arxiv.org/abs/26XX.XXXXX}
   }

geodex implements the methods of the papers below. Please also cite the ones your work uses.

.. dropdown:: Geometry-Aware Sampling-Based Motion Planning on Riemannian Manifolds
   :class-container: landing-paper

   Midpoint geodesic distance and interpolation under a Riemannian metric.

   .. code-block:: bibtex

      @inproceedings{kyaw2026geometry,
        title     = {Geometry-Aware Sampling-Based Motion Planning on {Riemannian} Manifolds},
        author    = {Kyaw, Phone Thiha and Kelly, Jonathan},
        booktitle = {Proceedings of the 17th World Symposium on the Algorithmic Foundations
                     of Robotics (WAFR)},
        address   = {Oulu, Finland},
        month     = jun,
        year      = {2026},
        url       = {https://arxiv.org/abs/2602.00992}
      }

.. dropdown:: Direct Informed Sampling on Riemannian Manifolds via Loewner Order Lower Bounds
   :class-container: landing-paper

   The Loewner lower bounds behind the admissible heuristics and the informed sampling.

   .. code-block:: bibtex

      @article{kyaw2026loewner,
        title   = {Direct Informed Sampling on {Riemannian} Manifolds via {Loewner} Order
                   Lower Bounds},
        author  = {Kyaw, Phone Thiha and Kelly, Jonathan},
        journal = {IEEE Robotics and Automation Letters},
        year    = {2026},
        url     = {https://arxiv.org/abs/2606.02879}
      }

.. dropdown:: Greedy Heuristics for Sampling-Based Motion Planning in High-Dimensional State Spaces
   :class-container: landing-paper

   The G-RRT\* planner.

   .. code-block:: bibtex

      @article{kyaw2026greedy,
        title   = {Greedy Heuristics for Sampling-Based Motion Planning in High-Dimensional
                   State Spaces},
        author  = {Kyaw, Phone Thiha and Le, Anh Vu and Mohan, Rajesh Elara and
                   Kelly, Jonathan},
        journal = {Autonomous Robots},
        year    = {2026},
        url     = {https://arxiv.org/abs/2405.03411}
      }

.. toctree::
   :hidden:

   getting-started/index
   concepts/index
   tutorials/index
   robots/index
   ros2/index
   api/index
