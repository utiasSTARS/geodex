API Reference
=============

Core Concepts
-------------

.. doxygenfile:: core/concepts.hpp
   :sections: briefdescription detaileddescription

.. doxygenfile:: core/metric.hpp
   :sections: briefdescription detaileddescription

.. doxygenfile:: core/distance.hpp
   :sections: briefdescription detaileddescription

.. doxygenfile:: core/interpolation.hpp
   :sections: briefdescription detaileddescription

.. doxygenfile:: core/retraction.hpp
   :sections: briefdescription detaileddescription

Manifolds
---------

Sphere
^^^^^^

.. doxygenclass:: geodex::Sphere
   :members:

.. doxygentypedef:: geodex::SphereRoundMetric

.. doxygenstruct:: geodex::SphereExponentialMap
   :members:

.. doxygenstruct:: geodex::SphereProjectionRetraction
   :members:

Euclidean
^^^^^^^^^

.. doxygenclass:: geodex::Euclidean
   :members:

.. doxygentypedef:: geodex::EuclideanStandardMetric


Torus
^^^^^

.. doxygenclass:: geodex::Torus
   :members:

.. doxygentypedef:: geodex::TorusFlatMetric

SE(2)
^^^^^

.. doxygenclass:: geodex::SE2
   :members:

Configuration Space
^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: geodex::ConfigurationSpace
   :members:

Metrics
-------

.. doxygenclass:: geodex::IdentityMetric
   :members:

.. doxygenclass:: geodex::ConstantSPDMetric
   :members:

.. doxygenclass:: geodex::SE2LeftInvariantMetric
   :members:

.. doxygenclass:: geodex::KineticEnergyMetric
   :members:

.. doxygenclass:: geodex::JacobiMetric
   :members:

.. doxygenclass:: geodex::PullbackMetric
   :members:

.. doxygenclass:: geodex::WeightedMetric
   :members:

.. doxygenclass:: geodex::SDFConformalMetric
   :members:

Collision
---------

.. doxygenclass:: geodex::collision::DistanceGrid
   :members:

.. doxygenclass:: geodex::collision::GridSDF
   :members:

.. doxygenclass:: geodex::collision::InflatedSDF
   :members:

.. doxygenclass:: geodex::collision::PolygonFootprint
   :members:

.. doxygenclass:: geodex::collision::FootprintGridChecker
   :members:

.. doxygenclass:: geodex::collision::CircleSDF
   :members:

.. doxygenclass:: geodex::collision::CircleSmoothSDF
   :members:

.. doxygenstruct:: geodex::collision::RectObstacle
   :members:

.. doxygenclass:: geodex::collision::RectSmoothSDF
   :members:

.. doxygenfunction:: geodex::collision::rects_overlap

Sampling
--------

.. doxygenclass:: geodex::StochasticSampler
   :members:

.. doxygenclass:: geodex::HaltonSampler
   :members:

.. doxygenconcept:: geodex::Sampler

.. doxygenconcept:: geodex::SeedableSampler

Heuristics
----------

.. doxygenstruct:: geodex::EuclideanHeuristic
   :members:

Algorithms
----------

.. doxygenfunction:: geodex::distance_midpoint

.. doxygenstruct:: geodex::InterpolationSettings
   :members:

.. doxygenenum:: geodex::InterpolationStatus

.. doxygenstruct:: geodex::InterpolationResult
   :members:

.. doxygenstruct:: geodex::InterpolationCache
   :members:

.. doxygenfunction:: geodex::discrete_geodesic

Batched inner product
~~~~~~~~~~~~~~~~~~~~~

.. doxygenconcept:: geodex::HasBatchInnerMatrix

OMPL Integration
----------------

.. doxygenclass:: geodex::integration::ompl::GeodexStateSpace
   :members:

.. doxygenclass:: geodex::integration::ompl::GeodexState
   :members:

.. doxygenclass:: geodex::integration::ompl::GeodexStateSampler
   :members:
