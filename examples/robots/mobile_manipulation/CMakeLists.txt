# Examples behind the mobile manipulation guide (docs/robots/mobile-manipulation.rst).
if(TARGET geodex_vamp)
  geodex_docs_example(robots_mobile_manipulation_stretch stretch.cpp OMPL ROBOTS)
  geodex_docs_example(robots_mobile_manipulation_clearpath clearpath.cpp OMPL ROBOTS)
endif()
