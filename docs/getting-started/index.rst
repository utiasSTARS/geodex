Getting Started
===============

This document will take you through installing geodex, building the tests and examples, and running your first computations.

Installation
------------

geodex is a header-only C++20 library — there is nothing to compile for the core library itself.
This page covers getting the source, verifying the build system, and running the bundled tests and examples.

**Requirements**

* **C++20 compiler** — GCC 10+, Clang 13+, or MSVC 19.29+
* **CMake 3.20** or later
* **Eigen3** — fetched automatically via CMake FetchContent (no manual install needed)
* **GoogleTest** — fetched automatically when ``-DBUILD_TESTING=ON``

**Getting the source**

.. code-block:: bash

   git clone https://github.com/utiasSTARS/geodex.git
   cd geodex

**Configure and build**

The core library has no compiled output.
CMake is needed only to build tests and examples:

.. code-block:: bash

   cmake -B build -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_TESTING=ON \
         -DBUILD_EXAMPLES=ON
   cmake --build build

.. note::

   On first configure, CMake will download Eigen3 and GoogleTest via FetchContent.
   This requires an internet connection and takes a minute.
   Subsequent builds are instant.

**Run the tests**

.. code-block:: bash

   ctest --test-dir build --output-on-failure

All tests should pass.
If any fail, check that your compiler supports C++20 concepts.
You may need to pass ``-DCMAKE_CXX_STANDARD=20`` explicitly with some toolchains.

**Run the examples**

After building with ``-DBUILD_EXAMPLES=ON``:

.. code-block:: bash

   ./build/examples/sphere_basics
   ./build/examples/sphere_distance

Python equivalents of all examples are provided in the ``examples/`` directory:

.. code-block:: bash

   python examples/sphere_basics.py
   python examples/sphere_distance.py

**Using geodex in your project**

The recommended approach is CMake FetchContent:

.. code-block:: cmake

   include(FetchContent)
   FetchContent_Declare(
     geodex
     GIT_REPOSITORY https://github.com/utiasSTARS/geodex.git
     GIT_TAG main
     GIT_SHALLOW TRUE
   )
   FetchContent_MakeAvailable(geodex)

   add_executable(my_app main.cpp)
   target_link_libraries(my_app PRIVATE geodex)

This pulls in Eigen automatically. Then just include the main geodex header:

.. code-block:: cpp

   #include <geodex/geodex.hpp>

Python Installation
-------------------

geodex also provides Python bindings built with `nanobind <https://nanobind.readthedocs.io>`_.
The bindings expose the same manifold types and algorithms as the C++ library, and use numpy
arrays for all operations.

**Requirements**

* **Python 3.9** or later
* **NumPy 1.21** or later
* A **C++20 compiler** (GCC 12+, Clang 14+)

**Install from source**

Clone the repository and install the Python bindings:

.. code-block:: bash

   git clone https://github.com/utiasSTARS/geodex.git
   cd geodex
   python3 -m venv .venv
   source .venv/bin/activate
   pip install .

**Development install**

For an editable install with test dependencies,
create and activate a virtual environment:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate

Then install in editable mode with test dependencies using **one** of the following:

**Option A** — Without build isolation (faster for repeated installs during development):

.. code-block:: bash

   pip install scikit-build-core>=0.10 nanobind>=2.4
   pip install --no-build-isolation -ve ".[test]"

**Option B** — With build isolation (simpler, pip handles build dependencies automatically):

.. code-block:: bash

   pip install -ve ".[test]"

**Verify the installation**

.. code-block:: python

   import geodex
   import numpy as np

   sphere = geodex.Sphere()
   print(sphere.dim())  # 2
   p = np.array([0., 0., 1.])
   q = np.array([1., 0., 0.])
   print(sphere.distance(p, q))  # 1.5708 ≈ π/2

**Next steps**

- :doc:`/concepts/index` — geometric background and how geodex models it in C++20
- :doc:`/tutorials/geodex-basics` — learn the core operations on built-in manifolds (C++ and Python)
