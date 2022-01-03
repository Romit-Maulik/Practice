cmake_minimum_required(VERSION 3.10.2)

project(directory_test)
set(CMAKE_BUILD_TYPE Release)
 
#Bring the headers, such as *.h into the project
include_directories(include)
 
#However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES "src/*.cpp")

#For the shared library:
set ( PROJECT_LINK_LIBS libtensorflow.so )
link_directories(/usr/local/lib)
 
#Generate the shared library from the sources
add_executable(tensorflow_cpp ${SOURCES})

# Link to tensorflow
target_link_libraries(tensorflow_cpp ${PROJECT_LINK_LIBS} )
 
# g++ -Iinclude/ mwe.cpp tf_utils.cpp -o tensorflow_cpp -L/usr/local/lib -ltensorflow
