find_package(Python 3.12
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# nanobind, from its pin in third_party/dependencies.cmake.
set(nanobind_SOURCE_DIR "${CMAKE_BINARY_DIR}/_geodex_deps/src/nanobind")
geodex_fetch(nanobind "${nanobind_SOURCE_DIR}")
find_package(nanobind CONFIG REQUIRED PATHS "${nanobind_SOURCE_DIR}/cmake" NO_DEFAULT_PATH)

set(_geodex_python_sources
  src/geodex_python.cpp
  src/bind_samplers.cpp
  src/bind_sphere.cpp
  src/bind_euclidean.cpp
  src/bind_torus.cpp
  src/bind_se2.cpp
  src/bind_so2.cpp
  src/bind_so3.cpp
  src/bind_se3.cpp
  src/bind_product.cpp
  src/bind_metrics.cpp
  src/bind_config_space.cpp
  src/bind_algorithms.cpp
  src/bind_heuristics.cpp
  src/bind_collision.cpp
)

# VAMP selects its instruction set at compile time and needs AVX2 and FMA on x86-64. With
# GEODEX_PYTHON_ISA_VARIANTS the build installs two extension modules there, a baseline
# module without VAMP for any x86-64 CPU and a module with VAMP. geodex/__init__.py loads
# the one the CPU supports. The baseline module does not contain any AVX2 code, including
# inline functions it shares with VAMP.
set(_geodex_x86_64 OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  set(_geodex_x86_64 ON)
endif()
if(GEODEX_BUNDLE_DEPS AND GEODEX_VAMP AND _geodex_x86_64)
  set(_geodex_isa_variants_default ON)
else()
  set(_geodex_isa_variants_default OFF)
endif()
option(GEODEX_PYTHON_ISA_VARIANTS
  "On x86-64, build a baseline module without VAMP and an AVX2+FMA module with VAMP"
  ${_geodex_isa_variants_default})

# Add one extension module named _geodex_core, installed under `destination`.
function(geodex_add_python_module target destination with_vamp)
  nanobind_add_module(${target}
    STABLE_ABI
    NB_STATIC
    ${_geodex_python_sources}
  )
  set_target_properties(${target} PROPERTIES
    OUTPUT_NAME _geodex_core
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${destination})

  if(with_vamp OR NOT TARGET geodex_vamp)
    target_link_libraries(${target} PRIVATE geodex)
  else()
    # The geodex target links the VAMP archive transitively. A module without VAMP does not
    # link that archive. The linker could take AVX2 copies of shared inline functions from it.
    target_include_directories(${target} PRIVATE
      $<TARGET_PROPERTY:geodex,INTERFACE_INCLUDE_DIRECTORIES>)
    target_compile_features(${target} PRIVATE cxx_std_20)
    foreach(_part geodex_robots geodex_pinocchio)
      if(TARGET ${_part})
        target_link_libraries(${target} PRIVATE ${_part})
      endif()
    endforeach()
  endif()
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

  if(GEODEX_PINOCCHIO)
    target_sources(${target} PRIVATE src/bind_pinocchio.cpp)
    target_compile_definitions(${target} PRIVATE GEODEX_PYTHON_HAS_PINOCCHIO)
  endif()

  if(with_vamp)
    target_sources(${target} PRIVATE src/bind_vamp.cpp src/bind_scene.cpp)
    target_compile_definitions(${target} PRIVATE GEODEX_PYTHON_HAS_VAMP)
  endif()

  if(GEODEX_ROBOTS)
    target_sources(${target} PRIVATE src/bind_robots.cpp)
    target_compile_definitions(${target} PRIVATE GEODEX_PYTHON_HAS_ROBOTS)
  endif()

  if(GEODEX_OMPL)
    target_sources(${target} PRIVATE src/bind_planning.cpp src/plan_se2_clearance.cpp)
    target_link_libraries(${target} PRIVATE ompl::ompl)
    target_compile_definitions(${target} PRIVATE GEODEX_PYTHON_HAS_OMPL)
  endif()

  # Export only the module's init function. The statically linked libraries, including the
  # OMPL fork, stay local. Symbols of another extension in the same process, such as the
  # official ompl wheel, do not bind to them in either direction.
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_version_script "${CMAKE_CURRENT_BINARY_DIR}/geodex_python.version")
    file(WRITE "${_version_script}" "{ global: PyInit__geodex_core; local: *; };\n")
    target_link_options(${target} PRIVATE "LINKER:--exclude-libs,ALL" "LINKER:--as-needed"
      "LINKER:--version-script=${_version_script}")
  elseif(APPLE)
    target_link_options(${target} PRIVATE "LINKER:-exported_symbol,_PyInit__geodex_core")
  endif()

  install(TARGETS ${target} LIBRARY DESTINATION ${destination} COMPONENT python)
endfunction()

if(GEODEX_PYTHON_ISA_VARIANTS AND GEODEX_VAMP AND _geodex_x86_64)
  geodex_add_python_module(_geodex_core_x86_64 geodex/_isa/x86_64 OFF)
  geodex_add_python_module(_geodex_core_x86_64_v3 geodex/_isa/x86_64_v3 ON)
  message(STATUS "geodex Python: x86-64 baseline module and AVX2+FMA module with VAMP")
else()
  geodex_add_python_module(_geodex_core geodex ${GEODEX_VAMP})

  # Generate type stubs from the built module with this build's nanobind. scripts/stubs.py
  # syncs the copies under python/geodex to them and checks them.
  option(GEODEX_PYTHON_STUBS "Generate type stubs for the built module" OFF)
  if(GEODEX_PYTHON_STUBS)
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/stubs/_geodex_core.pyi
      COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/stubs.py generate
              --stubgen ${nanobind_SOURCE_DIR}/src/stubgen.py
              --path ${CMAKE_CURRENT_BINARY_DIR}
              --output ${CMAKE_CURRENT_BINARY_DIR}/stubs
      DEPENDS _geodex_core ${PROJECT_SOURCE_DIR}/scripts/stubs.py
      COMMENT "Generating geodex type stubs")
    add_custom_target(geodex_python_stubs ALL
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/stubs/_geodex_core.pyi)
  endif()
endif()
