#################################################################################
# @project ICTree
# @brief The main project file.
# @author Tomas Polasek
#################################################################################

cmake_minimum_required (VERSION 3.9)
cmake_policy (SET CMP0074 NEW)

set (CMAKE_VERBOSE_MAKEFILE No)

set (PROJECT_NAME ICTree)
project (${PROJECT_NAME})

add_subdirectory (cmake)

add_subdirectory (Common)

add_subdirectory (TreeIOLib)
add_subdirectory (TreeIOAcceleration)
add_subdirectory (TreeIO)

add_subdirectory (PerceptualMetric)
