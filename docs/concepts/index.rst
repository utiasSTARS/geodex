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
geodex falls back to the midpoint approximation — see :ref:`distance_midpoint <api:Algorithms>` in the API reference).

**Retractions** are first- or second-order approximations to the exponential
map. They are cheaper to evaluate and preserve the manifold topology, but
unlike the true exp, they are not isometries. geodex separates retractions
from metrics as independent policy types (see the :doc:`/tutorials/geodex-basics` tutorial for usage).

Concept Hierarchy
-----------------

**geodex** is built around a small set of C++20 concepts that define what it means to be a manifold, a metric, a retraction, and so on.
These concepts compose through a *policy-based design*; for example, a manifold class like ``Sphere`` is parameterized by interchangeable metric and retraction policies, and the compiler statically verifies that the assembled type satisfies the full ``RiemannianManifold`` interface.

The central abstraction is a two-level concept hierarchy. ``Manifold`` captures the bare topology (points, tangent vectors, etc), while ``RiemannianManifold`` adds the geometric structure needed for distance computation and motion planning.

.. mermaid::

   %%{init: {"theme": "neutral"}}%%
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

       class RiemannianManifold {
           <<concept>>
           +inner(p, u, v) Scalar
           +norm(p, v) Scalar
           +distance(p, q) Scalar
           +geodesic(p, q, t) Point
           +exp(p, v) Point
           +log(p, q) Tangent
       }

       class HasInjectivityRadius {
           <<concept>>
           +injectivity_radius() Scalar
       }

       Manifold <|-- RiemannianManifold
       Manifold <|-- HasInjectivityRadius

``RiemannianManifold`` is deliberately monolithic: any type satisfying it provides the complete interface that algorithms need.
For finer-grained constraints, geodex also defines three orthogonal trait concepts:

.. mermaid::

   %%{init: {"theme": "neutral"}}%%
   classDiagram
       direction LR

       class Manifold {
           <<concept>>
       }

       class HasMetric {
           <<concept>>
           +inner(p, u, v) Scalar
           +norm(p, v) Scalar
       }

       class HasDistance {
           <<concept>>
           +distance(p, q) Scalar
       }

       class HasGeodesic {
           <<concept>>
           +geodesic(p, q, t) Point
           +exp(p, v) Point
           +log(p, q) Tangent
       }

       Manifold <|-- HasMetric
       Manifold <|-- HasDistance
       Manifold <|-- HasGeodesic

These allow algorithms to require only the operations they actually use.
For example, an algorithm that only needs exp/log can constrain on ``HasGeodesic`` without requiring a full metric.

How It All Fits Together
------------------------

The following diagram shows the full picture: concepts define interfaces,
policies provide implementations, and manifold classes compose them into
concrete types that satisfy the concepts.

.. mermaid::

   %%{init: {"theme": "neutral"}}%%
   flowchart TB
       subgraph Concepts["Concepts (compile-time)"]
           Manifold
           RiemannianManifold
           Retraction["Retraction&lt;R, Point, Tangent&gt;"]
       end

       subgraph Policies["Policies (swappable)"]
           Metric["MetricT<br/><i>e.g. SphereRoundMetric</i>"]
           Retr["RetractionT<br/><i>e.g. SphereExponentialMap</i>"]
       end

       subgraph Concrete["Concrete Manifold"]
           Sphere["Sphere&lt;MetricT, RetractionT&gt;"]
       end

       Metric -->|"inner, norm"| Sphere
       Retr -->|"retract, inverse_retract"| Sphere
       Sphere -.->|"satisfies"| RiemannianManifold
       RiemannianManifold -.->|"refines"| Manifold
       Retr -.->|"satisfies"| Retraction

.. The following pages detail the implementation and usage of these core architectural components:

References
----------

.. bibliography::
   :filter: docname in docnames
