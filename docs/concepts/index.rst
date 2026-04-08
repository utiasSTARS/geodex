Core Concepts
=============

This document starts with the minimal geometric background needed to understand geodex,
then explains how these concepts are encoded in the library's implementation.
Readers already familiar with Riemannian geometry may skip directly to the implementation details.

Mathematical Background
-----------------------

This section provides the minimal vocabulary necessary to follow the rest of the documentation and use geodex effectively.
It is not an exhaustive introduction to differential geometry.
For a rigorous treatment of Riemannian geometry, see :cite:`Lee2018`.
For readers interested in the computational aspects relevant to this library, see :cite:`Boumal2023` and :cite:`Absil2008`.

**Riemannian Geometry**

A **smooth manifold** :math:`\mathcal{M}` is a topological space that locally looks like a familiar Euclidean space :math:`\mathbb{R}^n`.
Each point :math:`p \in \mathcal{M}` has an associated **tangent space** :math:`\mathcal{T}_p\mathcal{M}` — the vector space of all instantaneous velocities passing through :math:`p`.
Tangent vectors :math:`v \in \mathcal{T}_p\mathcal{M}` are the directions in which you can move on the manifold.

.. image:: ../tutorials/figs/manifold.svg
   :class: responsive-img
   :align: center

A **Riemannian metric** :math:`g` equips every tangent space with a smoothly varying inner product:

.. math::

   g_p : \mathcal{T}_p\mathcal{M} \times \mathcal{T}_p\mathcal{M} \to \mathbb{R}

This inner product allows us to define lengths and angles on the manifold.
The **norm** of a tangent vector :math:`v \in \mathcal{T}_p\mathcal{M}` is defined as:

.. math::

   \|v\|_p = \sqrt{g_p(v, v)}

**Geodesics** :math:`\gamma` are the generalization of straight lines in flat spaces to manifolds — they are locally length-minimizing curves with zero acceleration.

The **exponential map** :math:`\exp_p : \mathcal{T}_p\mathcal{M} \to \mathcal{M}` follows the geodesic starting at :math:`p` with initial velocity :math:`v`:

.. math::

   \exp_p(v) = \gamma(1), \quad \dot\gamma(0) = v

The **logarithmic map** :math:`\log_p : \mathcal{M} \to \mathcal{T}_p\mathcal{M}` is the local inverse of exponential map, that is, it returns the tangent vector at :math:`p` pointing toward :math:`q`:

.. math::

   \log_p(q) = v \quad\Longleftrightarrow\quad \exp_p(v) = q

The **geodesic distance** between two points is the length of the shortest connecting geodesic:

.. math::

   d(p, q) = \|\log_p(q)\|_p

(This formula requires exact exp/log. When only approximations are available,
geodex falls back to the midpoint approximation — see :doc:`distance`).

**Retractions** are first- or second-order approximations to the exponential
map. They are cheaper to evaluate and preserve the manifold topology, but
unlike the true exp, they are not isometries. geodex separates retractions
from metrics as independent policy types (see :doc:`retractions`).

Concept Hierarchy
-----------------

**geodex** is built around a small set of C++20 concepts that define what it means to
be a manifold, a metric, a retraction, and so on.
These concepts compose through a *policy-based design*: a manifold class like
``Sphere`` is parameterized by interchangeable metric, retraction, and sampler
policies, and the compiler statically verifies that the assembled type satisfies the
full ``RiemannianManifold`` interface.

The central abstraction is a two-level hierarchy.
``Manifold`` captures the bare topology (points, tangent vectors, dimension,
sampling), while ``RiemannianManifold`` adds the geometric structure needed for
distance and motion planning.
Three smaller trait concepts (``HasMetric``, ``HasDistance``, ``HasGeodesic``) let
algorithms constrain on only the operations they actually use, and
``HasInjectivityRadius`` exposes the local injectivity radius on manifolds that
support it.

.. mermaid::

   %%{init: {'theme':'base','themeVariables':{'primaryColor':'#e7f0fa','primaryTextColor':'#1a1a1a','primaryBorderColor':'#2980b9','lineColor':'#2980b9','secondaryColor':'#e7f0fa','tertiaryColor':'#f7fbfe','background':'transparent'}}}%%
   classDiagram
       direction TB

       class Manifold {
           <<concept>>
           +Scalar
           +Point
           +Tangent
           +dim() int
           +random_point() Point
       }

       class HasMetric {
           <<concept>>
           +inner(p, u, v)
           +norm(p, v)
       }

       class HasDistance {
           <<concept>>
           +distance(p, q)
       }

       class HasGeodesic {
           <<concept>>
           +geodesic(p, q, t)
           +exp(p, v)
           +log(p, q)
       }

       class HasInjectivityRadius {
           <<concept>>
           +injectivity_radius()
       }

       class RiemannianManifold {
           <<concept>>
       }

       Manifold <|-- HasMetric
       Manifold <|-- HasDistance
       Manifold <|-- HasGeodesic
       Manifold <|-- HasInjectivityRadius
       HasMetric <|-- RiemannianManifold
       HasDistance <|-- RiemannianManifold
       HasGeodesic <|-- RiemannianManifold

Two further policy concepts plug into a manifold from the side.
``Retraction<R, Point, Tangent>`` is the contract every retraction policy satisfies,
with just ``retract(p, v)`` and ``inverse_retract(p, q)``.
``Sampler`` and its refinement ``SeedableSampler`` draw uniform samples from the unit
box and drive ``random_point()`` on ``Euclidean``, ``Torus``, and ``SE2``.

.. mermaid::

   %%{init: {'theme':'base','themeVariables':{'primaryColor':'#e7f0fa','primaryTextColor':'#1a1a1a','primaryBorderColor':'#2980b9','lineColor':'#2980b9','secondaryColor':'#e7f0fa','tertiaryColor':'#f7fbfe','background':'transparent'}}}%%
   classDiagram
       direction TB

       class Manifold {
           <<concept>>
       }

       class Retraction {
           <<concept>>
           +retract(p, v)
           +inverse_retract(p, q)
       }

       class Sampler {
           <<concept>>
           +sample_box(d, out)
       }

       class SeedableSampler {
           <<concept>>
           +seed(s)
       }

       Manifold <-- Retraction : exp / log
       Manifold <-- Sampler : random_point
       Sampler <|-- SeedableSampler

How It All Fits Together
------------------------

Three families of policies (metrics, retractions, samplers) feed into the manifold
classes, and algorithms consume those manifolds through the ``RiemannianManifold``
concept.

.. graphviz::

   digraph geodex {
       rankdir=LR;
       bgcolor="transparent";
       compound=true;
       pad="0.3";
       nodesep="0.35";
       ranksep="0.9";
       node [shape=box, style="filled",
             fillcolor="#e7f0fa", color="#2980b9", penwidth="1.2",
             fontname="Helvetica", fontsize=11, fontcolor="#1a1a1a",
             margin="0.18,0.10", height="0.45"];
       edge [color="#2980b9", penwidth="1.1",
             fontname="Helvetica", fontsize=10, fontcolor="#34495e",
             arrowsize="0.8"];
       graph [fontname="Helvetica", fontsize=11, fontcolor="#1a1a1a",
              color="#2980b9", penwidth="1.0",
              style="filled", fillcolor="#f7fbfe"];

       subgraph cluster_metrics {
           label="Metrics";
           ConstantSPDMetric;
           WeightedMetric;
           KineticEnergyMetric;
           JacobiMetric;
           PullbackMetric;
           SE2LeftInvariantMetric;
       }

       subgraph cluster_retractions {
           label="Retractions";
           SphereExponentialMap;
           SphereProjectionRetraction;
           SE2ExponentialMap;
           SE2EulerRetraction;
       }

       subgraph cluster_samplers {
           label="Samplers";
           StochasticSampler;
           HaltonSampler;
       }

       subgraph cluster_manifolds {
           label="Manifolds";
           Sphere       [label="Sphere<Dim>"];
           Euclidean    [label="Euclidean<Dim>"];
           Torus        [label="Torus<Dim>"];
           SE2          [label="SE2"];
           ConfigurationSpace;
       }

       subgraph cluster_algorithms {
           label="Algorithms";
           distance_midpoint;
           discrete_geodesic;
       }

       JacobiMetric          -> Sphere    [ltail=cluster_metrics,
                                            lhead=cluster_manifolds,
                                            label="metric"];
       SE2EulerRetraction    -> SE2       [ltail=cluster_retractions,
                                            lhead=cluster_manifolds,
                                            label="retraction"];
       HaltonSampler         -> Euclidean [ltail=cluster_samplers,
                                            lhead=cluster_manifolds,
                                            label="sampler"];
       ConfigurationSpace    -> distance_midpoint
                                          [ltail=cluster_manifolds,
                                           lhead=cluster_algorithms,
                                           label="RiemannianManifold"];
   }

``Sphere``, ``Euclidean``, ``Torus``, and ``SE2`` carry the topology and a default
choice of metric, retraction, and sampler, while ``ConfigurationSpace<Base, Metric>``
reuses an existing base topology with a custom metric. The metric ranges from a
constant SPD matrix up to configuration-dependent kinetic-energy and Jacobi metrics
built by composing ``WeightedMetric`` over ``KineticEnergyMetric``. The retraction is
either an exact Lie-group exp/log or a cheaper first-order approximation, and the
sampler is the pseudo-random ``StochasticSampler`` or the low-discrepancy
``HaltonSampler``. Algorithms like ``distance_midpoint`` and ``discrete_geodesic``
only require ``RiemannianManifold``, so any combination of the above plugs in without
touching the algorithm itself.

See also
^^^^^^^^

- :doc:`/tutorials/geodex-basics` for a hands-on walk-through with runnable C++ and
  Python snippets.
- :doc:`/tutorials/minimum-energy-planning` for composing ``KineticEnergyMetric`` and
  ``JacobiMetric`` to plan minimum-energy motions.
- :doc:`/api/index` for the full reference of every class and concept named above.

References
----------

.. bibliography::
   :filter: docname in docnames
