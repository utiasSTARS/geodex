Installation
============

On this page, we install geodex, check the install, and plan a first motion. geodex installs as a
Python wheel, as a CMake package for C++ projects, or as a pixi workspace that builds geodex for
Python and C++ from a clone of the repository. All three install geodex 1.0.0.

.. _install-pip:

1. Install the Python wheel
---------------------------

.. code-block:: bash

   pip install pygeodex==1.0.0

The package is named ``pygeodex`` and imports as ``geodex``. It needs Python 3.12 or newer. Every
wheel includes the geometry core (manifolds, metrics, sampling, distances, interpolation and the
smoother). On Linux and macOS, it also includes planning, collision checking and the built-in
robots.

.. list-table::
   :header-rows: 1
   :widths: 30 14 56

   * - Platform
     - Planning
     - Notes
   * - Linux x86-64
     - yes
     - Needs glibc 2.28 or newer, as in Ubuntu 20.04 and later. Collision checking of the
       built-in robots needs a CPU with AVX2 and FMA. Planning with a Python validity function
       works on any CPU.
   * - Linux aarch64
     - yes
     - Needs glibc 2.28 or newer, as in Ubuntu 20.04 and later.
   * - Linux with musl (Alpine), x86-64 and aarch64
     - no
     - Includes the geometry core.
   * - macOS arm64
     - yes
     - Needs macOS 11 or newer.
   * - macOS x86-64
     - yes
     - Needs macOS 11 or newer. Collision checking of the built-in robots needs a CPU with AVX2
       and FMA. Planning with a Python validity function works on any CPU.
   * - Windows x86-64
     - no
     - Includes the geometry core.

On other platforms, ``pip install --no-binary pygeodex pygeodex==1.0.0`` builds the package
from source. The build needs a C++20 compiler, CMake 3.20 or newer, git and network access.

2. Check the install
--------------------

``python -m geodex`` prints the version and the components of the install.

.. code-block:: bash

   python -m geodex

On Linux and macOS it prints

.. code-block:: text

   geodex 1.0.0
   planning: yes
   built-in robots: yes
   collision checking: yes

A ``no`` marks a component that the platform does not have (see the table above).

3. Plan a first motion
----------------------

We plan a path for a differential-drive robot around a disc in a 4 m by 4 m room. Planning needs
the Linux or macOS wheel. The robot's poses :math:`(x, y, \theta)` form the manifold
:math:`\mathrm{SE}(2)`. The C++ version builds against the CMake package of step 4.

.. code-pair:: getting_started/first_plan first-plan

It prints ``True 4.77 765``. The planner found a path of cost 4.770 with 765 waypoints.

.. plotly-figure:: installation-first-plan
   :alt: A square room with a gray disc in the middle and a blue path that goes around the
         disc from the start on the left to the goal on the right.

   The planned path around the disc. The arrows show the robot's heading. Hover over the path
   to read the pose.

.. _install-cmake:

4. Use geodex from C++
----------------------

To plan from C++, build and install geodex with CMake. Install Ninja, yaml-cpp and Boost 1.68 or
newer with its serialization and program_options libraries first, for example with your system's
package manager. The build also compiles the OMPL fork that holds the G-RRT* planner and installs
it next to geodex.

.. code-block:: bash

   git clone --branch v1.0.0 https://github.com/utiasSTARS/geodex.git
   cmake -S geodex -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_INSTALL_PREFIX=$HOME/geodex-1.0.0 \
     -DGEODEX_OMPL=ON -DGEODEX_BUILD_OMPL=ON -DGEODEX_VAMP=ON
   cmake --build build
   cmake --install build

``scripts/install_geodex.sh <prefix>`` runs the same build and install steps. A C++ project then
adds the prefix to ``CMAKE_PREFIX_PATH`` and requests the components it uses, as in
``examples/cmake_project``.

.. literalinclude:: /../examples/cmake_project/CMakeLists.txt
   :language: cmake
   :lines: 2-

``geodex::geodex`` includes every component of the table below except ``ompl``, and a project
that plans also links ``ompl::ompl``.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Component
     - Provides
   * - (always)
     - The header-only core and the Eigen headers it uses.
   * - ``ompl``
     - Planning with ``geodex::planning::plan``.
   * - ``robots``
     - Mass matrices of the built-in robots and their lower bounds.
   * - ``vamp``
     - Collision checking of the built-in robots and scene files.
   * - ``pinocchio``
     - Mass matrices from a URDF through Pinocchio, when geodex was built with
       ``-DGEODEX_PINOCCHIO=ON``.

A project that only uses the geometry core can add geodex with FetchContent instead of
installing it.

.. code-block:: cmake

   include(FetchContent)
   FetchContent_Declare(geodex
     GIT_REPOSITORY https://github.com/utiasSTARS/geodex.git
     GIT_TAG v1.0.0
     GIT_SHALLOW TRUE)
   set(GEODEX_ROBOTS OFF)  # leave out the built-in robots
   FetchContent_MakeAvailable(geodex)
   target_link_libraries(my_app PRIVATE geodex::geodex)

.. _install-pixi:

5. Build from the pixi workspace
--------------------------------

`pixi <https://pixi.sh>`_ installs the compilers, CMake, Eigen, Boost, Python and the docs tools
at the exact versions of ``pixi.lock``, and its tasks build geodex for Python and C++.

.. code-block:: bash

   git clone --branch v1.0.0 https://github.com/utiasSTARS/geodex.git
   cd geodex
   pixi install --locked     # install the locked environment
   pixi run test             # build everything and run the tests
   pixi run quickstart       # run the example of the Quickstart page

The workspace supports Linux x86-64, macOS arm64 and macOS x86-64. See
:doc:`building-from-source` for the other tasks of the workspace.

Where to go next
----------------

- :doc:`quickstart` plans a first motion for a robot arm.
- :doc:`reproducibility` covers seeds and budgets.
- :doc:`/concepts/index` covers what the planner does with the metric.
