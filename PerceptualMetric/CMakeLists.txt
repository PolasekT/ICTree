#################################################################################
# @project PerceptualMetric
# @brief This project contains modules for machine learning and perceptual metric.
# @author Tomas Polasek
#################################################################################

cmake_minimum_required (VERSION 3.9)

set (PROJECT_NAME PerceptualMetric)
project (${PROJECT_NAME})

set (CMAKE_CXX_STANDARD 17)
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # using Clang
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # using GCC
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC --no-gnu-unique")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    # using Intel C++
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # using Visual Studio C++
endif ()
set (default_build_type "Release")

if (WIN32)
    # All of the Visual Studio projects are x86, so build this for 32bit as well.
    #set (CMAKE_GENERATOR_PLATFORM Win32)
    # Use static glew for 64 bit compilation.
    if (BUILD_ARCHITECTURE STREQUAL "x64")
        add_definitions (-DGLEW_STATIC)
    endif ()
endif ()

set_binary_dir (PerceptualMetric)

add_subdirectory (psrc)

if (${PM_CYTHON_ENABLED})
    add_subdirectory (cython)
endif ()

