cmake_minimum_required(VERSION 3.18...3.19)
project(gofit LANGUAGES CXX)

# Set C++ 17 Standard
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find python and pybind11
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Find Eigen 3.4+
find_package(Eigen3 3.4 REQUIRED)

# Add project sources to python module
file(GLOB SOURCES src/*.cpp)
pybind11_add_module(gofit ${SOURCES})
target_include_directories(gofit PRIVATE ${EIGEN3_INCLUDE_DIR})

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(gofit
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
