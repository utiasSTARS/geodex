Concept Hierarchy and Architecture
==================================

**geodex** is built around a small set of C++20 concepts that define what it means to
be a manifold, a metric, a retraction, and so on. These concepts compose through a
*policy-based design*: a manifold class like ``Sphere`` is parameterized by
interchangeable metric, retraction, and sampler policies, and the compiler statically
verifies that the assembled type satisfies the full ``RiemannianManifold`` concept.

Concept Hierarchy
-----------------

The central abstraction is a two-level hierarchy. ``Manifold`` captures the bare
topology (points, tangent vectors, dimension, sampling), while ``RiemannianManifold``
adds the geometric structure needed for distance computation and motion planning.

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

``RiemannianManifold`` is deliberately monolithic: any type satisfying it provides the
complete interface that algorithms need. For finer-grained constraints, geodex also
defines three orthogonal trait concepts: ``HasMetric``, ``HasDistance``, and
``HasGeodesic``. These allow algorithms to require only the operations they actually
use. For example, an algorithm that only needs exp/log can constrain on
``HasGeodesic`` without requiring a full metric. ``HasInjectivityRadius`` optionally
exposes the local injectivity radius on manifolds that support it.

Two further policy concepts plug into a manifold from the side. ``Retraction<>`` is the
contract every retraction policy satisfies, with just ``retract(p, v)`` and
``inverse_retract(p, q)``. ``Sampler`` and its refinement ``SeedableSampler`` allow
drawing uniform samples on manifolds.

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
           label=<<B>Metrics</B>>;
           ConstantSPDMetric;
           WeightedMetric;
           KineticEnergyMetric;
           JacobiMetric;
           PullbackMetric;
           SE2LeftInvariantMetric;
       }

       subgraph cluster_retractions {
           label=<<B>Retractions</B>>;
           SphereExponentialMap;
           SphereProjectionRetraction;
           SE2ExponentialMap;
           SE2EulerRetraction;
       }

       subgraph cluster_samplers {
           label=<<B>Samplers</B>>;
           StochasticSampler;
           HaltonSampler;
       }

       subgraph cluster_manifolds {
           label=<<B>Manifolds</B>>;
           Sphere       [label="Sphere<Dim>"];
           Euclidean    [label="Euclidean<Dim>"];
           Torus        [label="Torus<Dim>"];
           SE2          [label="SE2"];
           ConfigurationSpace;
       }

       subgraph cluster_algorithms {
           label=<<B>Algorithms</B>>;
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
