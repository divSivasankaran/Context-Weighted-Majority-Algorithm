project(CWMA)
cmake_minimum_required(VERSION 2.8.9)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB SOURCES
   include/*.h
   src/*.cpp
)  
   
 MESSAGE("Process file: ${SOURCES}")
add_executable(CWMA ${SOURCES})

