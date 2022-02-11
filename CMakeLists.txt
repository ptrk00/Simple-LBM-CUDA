cmake_minimum_required(VERSION 3.21)
project(LBM_CUDA CUDA CXX)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

find_package(CUDA QUIET REQUIRED)
message(STATUS "Found CUDA version: ${CUDA_VERSION}")

find_package(OpenGL REQUIRED)

include_directories(include)
include_directories(lib/include)

add_executable(LBM_CUDA src/main.cpp src/render.cpp src/kernel.cu include/render.h include/vec.h include/automata.h include/grid.h include/cell.h)

link_directories(lib)

target_link_libraries(LBM_CUDA ${CMAKE_CURRENT_SOURCE_DIR}/lib/freeglut.lib)

#set_target_properties(LBM_CUDA PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)
