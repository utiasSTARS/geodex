# This file pins the third-party sources of the full geodex stack.
#
# Each entry is an archive URL with its SHA-256, plus optional patches with their own
# SHA-256. cmake/GeodexFetch.cmake downloads, verifies, extracts and patches them. The
# pixi tasks (scripts/build_ompl.sh, scripts/fetch_vamp.sh), the CI workflows and the
# bundled wheel build (GEODEX_BUNDLE_DEPS) all read this file. Change a pin here only.

# Eigen 5.0.0 from its GitLab tag archive. geodex installs it, and the OMPL fork and VAMP
# build against it.
set(GEODEX_EIGEN_REF "5.0.0")
set(GEODEX_EIGEN_URL
    "https://gitlab.com/libeigen/eigen/-/archive/${GEODEX_EIGEN_REF}/eigen-${GEODEX_EIGEN_REF}.tar.gz")
set(GEODEX_EIGEN_SHA256 "315c881e19e17542a7d428c5aa37d113c89b9500d350c433797b730cd449c056")

# nanobind 2.4.0 from its PyPI sdist, which carries the robin-map submodule and the stub
# generator. The Python module compiles it in.
set(GEODEX_NANOBIND_REF "2.4.0")
set(GEODEX_NANOBIND_URL
    "https://files.pythonhosted.org/packages/1e/01/a28722f6626e5c8a606dee71cb40c0b2ab9f7715b96bd34a9553c79dbf42/nanobind-2.4.0.tar.gz")
set(GEODEX_NANOBIND_SHA256 "a0392dee5f58881085b2ac8bfe8e53f74285aa4868b1472bfaf76cfb414e1c96")

# The OMPL fork (utiasSTARS/ompl) at commit 4468723 with ompl/grrtstar-release.patch.
set(GEODEX_OMPL_REF "4468723257da9a38442063f6454e1636a4eabe33")
set(GEODEX_OMPL_URL "https://github.com/utiasSTARS/ompl/archive/${GEODEX_OMPL_REF}.tar.gz")
set(GEODEX_OMPL_SHA256 "9b036655f940b7054f2deaf474cf950d999fc8b68f4cb29c2f6534d6aafebac0")
set(GEODEX_OMPL_PATCHES "${CMAKE_CURRENT_LIST_DIR}/ompl/grrtstar-release.patch")
set(GEODEX_OMPL_PATCHES_SHA256
    "2a2f8e467d265bb32ebc54c4791c89a0c784f57b239d1db44d46180720066887")

# VAMP (KavrakiLab/vamp). Its CMake fetches nigh, pdqsort and SIMDxorshift through CPM,
# each pinned to a commit in VAMP's cmake/Dependencies.cmake.
set(GEODEX_VAMP_REF "e3902f1b77e504991b4facec536642f1c522c1cc")
set(GEODEX_VAMP_URL "https://github.com/KavrakiLab/vamp/archive/${GEODEX_VAMP_REF}.tar.gz")
set(GEODEX_VAMP_SHA256 "b6012b25e9fe444e21e96a3ad48d2f13ccc39e6e79137468d6aff570ed8ac1a6")

# VAMP's configure step fetches CPM.cmake and three header libraries. geodex downloads them
# from these pins with hash checks and retries. The pins match cmake/FetchInitCPM.cmake and
# cmake/Dependencies.cmake at the VAMP commit above. Update them together with
# GEODEX_VAMP_REF.
set(GEODEX_VAMP_CPM_URL "https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.40.1/CPM.cmake")
set(GEODEX_VAMP_CPM_SHA256 "117cbf2711572f113bab262933eb5187b08cfc06dce0714a1ee94f2183ddc3ec")
set(GEODEX_NIGH_URL
    "https://github.com/KavrakiLab/nigh/archive/97130999440647c204e0265d05a997dbd8da4e70.tar.gz")
set(GEODEX_NIGH_SHA256 "76d9b42da8d14f09d61331d786c1fda5bfd8fe2756cc7144e1c8482c44c7bd41")
set(GEODEX_PDQSORT_URL
    "https://github.com/orlp/pdqsort/archive/b1ef26a55cdb60d236a5cb199c4234c704f46726.tar.gz")
set(GEODEX_PDQSORT_SHA256 "1df2463f94ebd926f402e7bcd92bf4a16f7a35732080a607fe4716888f1edbb5")
set(GEODEX_SIMDXORSHIFT_URL
    "https://github.com/lemire/SIMDxorshift/archive/857c1a01df53cf1ee1ae8db3238f0ef42ef8e490.tar.gz")
set(GEODEX_SIMDXORSHIFT_SHA256 "fa1578e3b89383726765807d0751606a442d62b8f2e7c8eb2aacd76d7dbef618")

# Boost for the bundled wheel build only. That build compiles serialization and
# program_options, the components OMPL requires, as static libraries.
set(GEODEX_BOOST_REF "1.90.0")
set(GEODEX_BOOST_URL
    "https://github.com/boostorg/boost/releases/download/boost-${GEODEX_BOOST_REF}/boost-${GEODEX_BOOST_REF}-b2-nodocs.tar.xz")
set(GEODEX_BOOST_SHA256 "9e6bee9ab529fb2b0733049692d57d10a72202af085e553539a05b4204211a6f")

# GoogleTest for the C++ test suite.
set(GEODEX_GOOGLETEST_REF "1.14.0")
set(GEODEX_GOOGLETEST_URL
    "https://github.com/google/googletest/archive/refs/tags/v${GEODEX_GOOGLETEST_REF}.tar.gz")
set(GEODEX_GOOGLETEST_SHA256 "8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7")

# yaml-cpp for VAMP scene loading, built as a static library for the bundled wheel only.
set(GEODEX_YAML_CPP_REF "0.8.0")
set(GEODEX_YAML_CPP_URL
    "https://github.com/jbeder/yaml-cpp/archive/refs/tags/${GEODEX_YAML_CPP_REF}.tar.gz")
set(GEODEX_YAML_CPP_SHA256 "fbe74bbdcee21d656715688706da3c8becfd946d92cd44705cc6098bb23b3a16")
