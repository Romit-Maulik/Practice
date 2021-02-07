cmake_minimum_required(VERSION 3.10.2)
set(CMAKE_BUILD_TYPE Release)
project (lapack_test_project)

#Bring the headers, such as *.h into the project
include_directories(/home/rmlans/anaconda3/envs/deephyper_env/include/)
include_directories(/home/rmlans/Desktop/Tutorials/Lapack/)

# Link directories
# set ( PROJECT_LINK_LIBS liblapacke.so )
# link_directories(/home/rmlans/anaconda3/envs/deephyper_env/lib/)

add_library(lapacke SHARED IMPORTED)
set_property(TARGET lapacke PROPERTY IMPORTED_LOCATION "/home/rmlans/anaconda3/envs/deephyper_env/lib/liblapacke.so")

# Add executable from sources - CMAKE requires sources with *.cpp
file(GLOB SOURCES "*.cpp")
add_executable(lapack_test ${SOURCES})

# Link to final executable
target_link_libraries(lapack_test lapacke)