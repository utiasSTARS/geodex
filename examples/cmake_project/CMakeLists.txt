# A CMake project that plans with an installed geodex.
cmake_minimum_required(VERSION 3.22)
project(my_planner LANGUAGES CXX)

find_package(geodex 1.0 CONFIG REQUIRED COMPONENTS ompl)

add_executable(my_planner main.cpp)
target_compile_features(my_planner PRIVATE cxx_std_20)
target_link_libraries(my_planner PRIVATE geodex::geodex ompl::ompl)
