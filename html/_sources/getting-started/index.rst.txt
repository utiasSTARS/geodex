Getting Started
===============

There are three ways to install geodex. On Linux and macOS, the Python wheel installs the full
planning stack with a single ``pip`` command, without any build steps. The CMake package installs
geodex for C++ projects, which use it through ``find_package``. The pixi workspace builds geodex
for Python and C++ from a clone of the repository. See :doc:`installation` for all three.

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item-card:: Python wheel
      :link: install-pip
      :link-type: ref

      Install geodex with ``pip install pygeodex`` on Python 3.12 or newer.

   .. grid-item-card:: CMake package
      :link: install-cmake
      :link-type: ref

      Install geodex for C++ and find it with ``find_package(geodex 1.0 CONFIG REQUIRED)``.

   .. grid-item-card:: pixi workspace
      :link: install-pixi
      :link-type: ref

      Build and test geodex for Python and C++ with ``pixi run test``.

.. toctree::
   :maxdepth: 1

   installation
   quickstart
   reproducibility
   building-from-source
