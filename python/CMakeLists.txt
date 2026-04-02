find_package(Python 3.9
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Fetch nanobind
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG v2.4.0
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(nanobind)

nanobind_add_module(_geodex_core
  STABLE_ABI
  NB_STATIC
  src/geodex_python.cpp
  src/bind_sphere.cpp
  src/bind_euclidean.cpp
  src/bind_torus.cpp
  src/bind_se2.cpp
  src/bind_metrics.cpp
  src/bind_config_space.cpp
  src/bind_algorithms.cpp
)

target_link_libraries(_geodex_core PRIVATE geodex)
target_include_directories(_geodex_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

install(TARGETS _geodex_core LIBRARY DESTINATION geodex)
