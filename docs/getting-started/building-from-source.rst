Building from Source
====================

On this page, we build geodex from a checkout with its pixi workspace, run the test suites,
plan an example, and install geodex for another CMake project. The workspace installs the
compilers, CMake, Ninja, Eigen, Boost, yaml-cpp, Python and the docs tools at the exact versions
in ``pixi.lock``, and geodex builds the OMPL fork and VAMP from one exact commit each.

1. Clone and install the environment
------------------------------------

.. code-block:: bash

   git clone https://github.com/utiasSTARS/geodex.git
   cd geodex
   pixi install --locked

The compilers are GCC 14 on Linux and Clang 19 on macOS.

2. Build and test
-----------------

.. code-block:: bash

   pixi run test

``test`` builds the OMPL fork with the G-RRT* planner, VAMP, the C++ library and the Python
module, then runs the C++ tests, the Python tests and the check of the type stubs. Every task
builds what it needs first.

3. Plan an example
------------------

.. code-block:: bash

   pixi run quickstart

``quickstart`` runs the example of the :doc:`quickstart` page, a Franka Panda swinging around
a post. See ``examples/README.md`` for the other examples and how to run them.

4. Install for a C++ project
----------------------------

.. code-block:: bash

   pixi run install
   pixi run check-package

``install`` installs geodex and the OMPL fork into ``.pixi/prefix`` (the variable
``$GEODEX_PREFIX``), and ``check-package`` builds and runs ``examples/cmake_project`` against
it. A C++ project then adds that prefix to ``CMAKE_PREFIX_PATH``.

5. Try it: build these docs
---------------------------

.. code-block:: bash

   pixi run docs

``docs`` builds this site into ``build/docs/sphinx``, and ``test-docs`` runs every snippet on
these pages in Python and in C++. See ``docs/CONTRIBUTING-docs.md`` for adding a page, a
snippet, a figure or a 3D scene.

Tasks
-----

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Task
     - What it does
   * - ``build-ompl``
     - Builds the OMPL fork into ``.pixi/prefix``.
   * - ``fetch-vamp``
     - Downloads the VAMP source into ``.pixi/vamp``.
   * - ``configure``, ``build-cpp``
     - Configures and builds the C++ library, tests and examples in ``build/``.
   * - ``test-cpp``
     - Runs the C++ suite with ctest.
   * - ``build-py``, ``test-py``
     - Builds the Python module into the environment and runs the Python suite.
   * - ``test``
     - Runs ``test-cpp``, ``test-py`` and ``check-stubs``.
   * - ``install``, ``check-package``
     - Installs geodex into ``.pixi/prefix`` and builds ``examples/cmake_project`` against it.
   * - ``quickstart``
     - Runs the example of the :doc:`quickstart` page.
   * - ``test-docs``
     - Runs every documentation example in Python and in C++.
   * - ``docs``
     - Builds this site into ``build/docs/sphinx``.

CMake options
-------------

A build without pixi configures with these options.

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Option
     - Default
     - Effect
   * - ``GEODEX_OMPL``
     - OFF
     - Builds the OMPL integration, ``plan()`` and their tests. Needs the OMPL fork.
   * - ``GEODEX_BUILD_OMPL``
     - OFF
     - Builds the OMPL fork at the commit set in ``third_party/dependencies.cmake`` and
       installs it with geodex.
   * - ``GEODEX_VAMP``
     - OFF
     - Builds VAMP collision checking for the built-in robots.
   * - ``GEODEX_ROBOTS``
     - ON
     - Builds the mass matrices of the built-in robots.
   * - ``GEODEX_PINOCCHIO``
     - OFF
     - Builds the Pinocchio mass matrix integration.
   * - ``BUILD_TESTING``
     - OFF
     - Builds the C++ tests.
   * - ``BUILD_EXAMPLES``
     - OFF
     - Builds the examples behind these docs. Those that plan need ``GEODEX_OMPL``.
   * - ``BUILD_OMPL_EXAMPLES``
     - OFF
     - Builds the examples and turns on ``GEODEX_OMPL``.
   * - ``BUILD_DOCS``
     - OFF
     - Adds the Doxygen target for the C++ API pages.

.. warning::

   Configure with ``-DCMAKE_BUILD_TYPE=Release`` for anything that plans. The robot mass
   matrices and the planner run many times slower in an unoptimized build.

Where to go next
----------------

- :doc:`installation` covers the wheel and the CMake package without a checkout.
- :doc:`/robots/add-a-robot` covers generating the model of a new robot.
