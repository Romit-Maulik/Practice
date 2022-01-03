cmake_minimum_required(VERSION 3.5.2)

project(directory_test)
set(CMAKE_BUILD_TYPE Release)

#However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES "*.cpp")

#For the shared library:
message("Adding pybind11")
add_subdirectory(pybind11)
include_directories(${PYBIND11_INCLUDE_DIR})
message(${PYBIND11_INCLUDE_DIR})

# Eigen include directories
include_directories(EIGEN-3.3.7)

#Generate the shared library from the sources
add_executable(app ${SOURCES})

# Link to final executable
target_link_libraries(app PRIVATE pybind11::embed)


