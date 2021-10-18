# Version CMake
cmake_minimum_required(VERSION 3.10)

# Project name
project(opencv_example)

# Options
set(CMAKE_CXX_STANDARD 17)

# Compiler-specific options
if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    add_compile_options(
        -Werror
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

# set(WITH_CUDA ON)

if(WITH_CUDA)
    add_definitions(-DCUDA_ENABLE=true)
endif()

set(OPERATIONS_ON_MATRICES "operations_on_matrices")
file(GLOB_RECURSE OPERATIONS_ON_MATRICES_SOURCES
    operations_on_matrices/*.cpp
)

set(SMOOTH_IMAGES "smooth_images")
file(GLOB_RECURSE SMOOTH_IMAGES_SOURCES
    smooth_images/*.cpp
)

# Compiling target
add_executable(${OPERATIONS_ON_MATRICES} ${OPERATIONS_ON_MATRICES_SOURCES})
add_executable(${SMOOTH_IMAGES} ${SMOOTH_IMAGES_SOURCES})

# Find OpenCV
find_package(OpenCV REQUIRED)

if(WITH_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(${OPERATIONS_ON_MATRICES} ${OpenCV_LIBS})
target_link_libraries(${SMOOTH_IMAGES} ${OpenCV_LIBS})

if(WITH_CUDA)
    target_link_libraries(${OPERATIONS_ON_MATRICES} ${CUDA_LIBRARIES})
endif()