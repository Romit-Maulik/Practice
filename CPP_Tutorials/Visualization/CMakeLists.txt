cmake_minimum_required(VERSION 3.5.2)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(directory_test)
set(CMAKE_BUILD_TYPE Release)

#Bring the headers, such as *.h into the project - over here you need to paste *your* path to the python environment of choice
include_directories(/home/rmlans/miniconda3/envs/tf2_env/include/python3.6m)
include_directories(/home/rmlans/miniconda3/envs/tf2_env/lib/python3.6/site-packages/numpy/core/include)
 
#However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES "*.cpp")

#For the shared library:
# This is where the libpython3.6m.so file is
find_library(PYTHONLIB python3.6m HINTS /home/rmlans/miniconda3/envs/tf2_env/lib)
 
#Generate the shared library from the sources
add_executable(main ${SOURCES})

# Link to final executable
target_link_libraries(main ${PYTHONLIB})


