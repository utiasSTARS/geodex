# Examples behind the robot guides (docs/robots). Each has a Python version.
if(TARGET geodex_vamp)
  geodex_docs_example(robots_catalog catalog.cpp ROBOTS)
endif()
add_subdirectory(manipulation)
add_subdirectory(navigation)
add_subdirectory(mobile_manipulation)
