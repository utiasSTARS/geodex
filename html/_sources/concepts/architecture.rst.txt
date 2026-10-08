Concept Hierarchy and Architecture
==================================

**geodex** is built around a small set of C++20 concepts that define what it means to
be a manifold, a metric, a retraction, and so on. These concepts compose through a
*policy-based design*. A manifold class like ``Sphere`` is parameterized by
interchangeable metric, retraction, and sampler policies, and the compiler statically
verifies that the assembled type satisfies the full ``RiemannianManifold`` concept.
Algorithms, the planner and the integrations consume manifolds only through these
concepts, and every layer above the geometry works for every manifold below it.

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

       class HasCoordinateMetric {
           <<concept>>
           +coordinate_metric(p)
       }

       class HasPeriods {
           <<concept>>
           +periods()
       }

       class RiemannianManifold {
           <<concept>>
       }

       Manifold <|-- HasMetric
       Manifold <|-- HasDistance
       Manifold <|-- HasGeodesic
       Manifold <|-- HasInjectivityRadius
       Manifold <|-- HasCoordinateMetric
       Manifold <|-- HasPeriods
       HasMetric <|-- RiemannianManifold
       HasDistance <|-- RiemannianManifold
       HasGeodesic <|-- RiemannianManifold

A type satisfying ``RiemannianManifold`` provides the complete interface that algorithms
need. Three trait concepts (``HasMetric``, ``HasDistance``, ``HasGeodesic``) let algorithms
constrain on only the operations they actually use, and ``HasInjectivityRadius`` exposes the
local injectivity radius on manifolds that support it. ``HasCoordinateMetric`` and
``HasPeriods`` describe the coordinates themselves, the metric tensor on coordinate
velocities and the period of each periodic axis. The planner uses both for an admissible
heuristic whose chord wraps around a periodic heading, as on :math:`\mathrm{SE}(2)` (see
:doc:`planning`).

``Retraction<>`` is the contract every retraction policy satisfies, with only
``retract(p, v)`` and ``inverse_retract(p, q)``. ``Sampler`` and its refinement
``SeedableSampler`` give uniform samples in the unit cube, and the default is a scrambled
Halton low-discrepancy sequence. A manifold's measure-preserving map sends that sequence onto
the manifold (see :doc:`sampling`).

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
           +sample(n, out)
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

The library has four layers. The **geometry** layer holds the manifolds and the three
families of policies they are built from (metrics, retractions, samplers). The
**algorithm** layer consumes manifolds through the concepts, with distance and geodesic
interpolation, the smoother and the metric lower bound. The
**planning** layer turns a manifold into a plan through ``plan()``, with admissible
heuristics, the collision helpers and the built-in robots. The **integration** layer
connects geodex to other software (OMPL for search, VAMP for collision checking, Pinocchio
for dynamics, and the Nav2 and MoveIt 2 plugins in their own repositories). The Python
module binds every layer. See :doc:`/api/index` for the Python and C++ names where they
differ.

.. graphviz::

   digraph geodex {
       rankdir=TB;
       bgcolor="transparent";
       pad="0.3";
       ranksep="0.42";
       node [shape=plain, fontname="Helvetica", fontsize=13, fontcolor="#1a1a1a"];
       edge [color="#2980b9", penwidth="1.1", fontname="Helvetica", fontsize=12,
             fontcolor="#34495e", arrowsize="0.8"];

       // Each layer is one node, a bold title over a row of cells. Cell text starts right
       // after its tag.
       policies [label=<
         <TABLE BGCOLOR="#f7fbfe" COLOR="#2980b9" BORDER="1" CELLBORDER="0"
                CELLSPACING="10" CELLPADDING="7">
           <TR><TD COLSPAN="3" CELLPADDING="2"
               ><FONT POINT-SIZE="14"><B>Policies</B></FONT></TD></TR>
           <TR>
             <TD BGCOLOR="#e7f0fa" BORDER="1">Metrics<BR
               />Identity, ConstantSPD, SE2LeftInvariant,<BR/>KineticEnergy, Jacobi, Pullback,<BR
               />SDFConformal (clearance), Weighted</TD>
             <TD BGCOLOR="#e7f0fa" BORDER="1">Retractions<BR
               />exponential maps, projection, Euler</TD>
             <TD BGCOLOR="#e7f0fa" BORDER="1">Samplers<BR
               />ScrambledHalton, Halton, PseudoRandom</TD>
           </TR>
         </TABLE>>];
       manifolds [label=<
         <TABLE BGCOLOR="#f7fbfe" COLOR="#2980b9" BORDER="1" CELLBORDER="0"
                CELLSPACING="10" CELLPADDING="7">
           <TR><TD COLSPAN="2" CELLPADDING="2"
               ><FONT POINT-SIZE="14"><B>Manifolds</B></FONT></TD></TR>
           <TR>
             <TD BGCOLOR="#e7f0fa" BORDER="1">Sphere, Euclidean, Torus,<BR/>SO2, SO3, SE2, SE3</TD>
             <TD BGCOLOR="#e7f0fa" BORDER="1">ConfigurationSpace,<BR/>ProductManifold</TD>
           </TR>
         </TABLE>>];
       algorithms [label=<
         <TABLE BGCOLOR="#f7fbfe" COLOR="#2980b9" BORDER="1" CELLBORDER="0"
                CELLSPACING="10" CELLPADDING="7">
           <TR><TD COLSPAN="3" CELLPADDING="2"
               ><FONT POINT-SIZE="14"><B>Algorithms</B></FONT></TD></TR>
           <TR>
             <TD BGCOLOR="#e7f0fa" BORDER="1">distance_midpoint,<BR/>discrete_geodesic</TD>
             <TD BGCOLOR="#e7f0fa" BORDER="1">smooth_path</TD>
             <TD BGCOLOR="#e7f0fa" BORDER="1">precompute_matrix_lower_bound</TD>
           </TR>
         </TABLE>>];
       planning [label=<
         <TABLE BGCOLOR="#f7fbfe" COLOR="#2980b9" BORDER="1" CELLBORDER="0"
                CELLSPACING="10" CELLPADDING="7">
           <TR><TD COLSPAN="4" CELLPADDING="2"
               ><FONT POINT-SIZE="14"><B>Planning</B></FONT></TD></TR>
           <TR>
             <TD BGCOLOR="#e7f0fa" BORDER="1">plan()</TD>
             <TD BGCOLOR="#e7f0fa" BORDER="1">heuristics<BR/>Zero, Euclidean, MatrixLowerBound</TD>
             <TD BGCOLOR="#e7f0fa" BORDER="1">collision<BR/>SDFs, footprints, distance grids</TD>
             <TD BGCOLOR="#e7f0fa" BORDER="1">robots<BR/>CRBA mass matrices, bounds,<BR
               />joint spaces</TD>
           </TR>
         </TABLE>>];
       integrations [label=<
         <TABLE BGCOLOR="#f7fbfe" COLOR="#2980b9" BORDER="1" CELLBORDER="0"
                CELLSPACING="10" CELLPADDING="7">
           <TR><TD COLSPAN="4" CELLPADDING="2"
               ><FONT POINT-SIZE="14"><B>Integrations</B></FONT></TD></TR>
           <TR>
             <TD BGCOLOR="#e7f0fa" BORDER="1">OMPL fork<BR/>state space, G-RRT*</TD>
             <TD BGCOLOR="#e7f0fa" BORDER="1">VAMP<BR/>SIMD collision checking</TD>
             <TD BGCOLOR="#e7f0fa" BORDER="1">Pinocchio<BR/>mass matrices from URDF</TD>
             <TD BGCOLOR="#e7f0fa" BORDER="1">Nav2 and MoveIt 2<BR/>planner plugins</TD>
           </TR>
         </TABLE>>];

       policies   -> manifolds    [label="  policies"];
       manifolds  -> algorithms   [label="  RiemannianManifold"];
       algorithms -> planning;
       planning   -> integrations;
   }

Each arrow is a dependency through a concept, not through a concrete type. A new metric works
in every manifold that accepts a metric policy, a new manifold works in every algorithm and
in ``plan()`` as soon as it satisfies ``RiemannianManifold``, and the integrations see only
the planner's interface.

See also
--------

- :doc:`metrics` for what each metric models.
- :doc:`planning` and :doc:`smoothing` for the planning layer.
- :doc:`/api/cpp` for the reference of every type in the map.
