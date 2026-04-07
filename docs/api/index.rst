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

.. doxygenstruct:: geodex::ConstantSPDMetric
   :members:

.. doxygenstruct:: geodex::SE2LeftInvariantMetric
   :members:

.. doxygenstruct:: geodex::KineticEnergyMetric
   :members:

.. doxygenstruct:: geodex::JacobiMetric
   :members:

.. doxygenstruct:: geodex::PullbackMetric
   :members:

.. doxygenstruct:: geodex::WeightedMetric
   :members:

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

.. doxygenstruct:: geodex::InterpolationWorkspace
   :members:

.. doxygenfunction:: geodex::discrete_geodesic

Batched inner product
~~~~~~~~~~~~~~~~~~~~~

.. doxygenconcept:: geodex::HasBatchInnerMatrix

OMPL Integration
----------------

.. doxygenclass:: geodex::ompl_integration::GeodexStateSpace
   :members:

.. doxygenclass:: geodex::ompl_integration::GeodexState
   :members:

.. doxygenclass:: geodex::ompl_integration::GeodexStateSampler
   :members:
