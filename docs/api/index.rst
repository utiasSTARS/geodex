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

.. doxygenstruct:: geodex::SphereRoundMetric
   :members:

.. doxygenstruct:: geodex::SphereExponentialMap
   :members:

.. doxygenstruct:: geodex::SphereProjectionRetraction
   :members:

Euclidean
^^^^^^^^^

.. doxygenclass:: geodex::Euclidean
   :members:

.. doxygenstruct:: geodex::EuclideanStandardMetric
   :members:


Torus
^^^^^

.. doxygenclass:: geodex::Torus
   :members:

.. doxygenstruct:: geodex::TorusFlatMetric
   :members:

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

.. doxygenstruct:: geodex::IdentityTaskMetric
   :members:

.. doxygenstruct:: geodex::WeightedMetric
   :members:

Algorithms
----------

.. doxygenfunction:: geodex::distance_midpoint
