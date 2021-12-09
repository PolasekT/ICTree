#################################################################################
# @project TreeIOAcceleration
# @brief Library containing acceleration code and data structures.
# @author Tomas Polasek
#################################################################################

cmake_minimum_required (VERSION 3.9)

set (PROJECT_NAME TreeIOAcceleration)
project (${PROJECT_NAME})

set (STATIC_LIB_NAME TreeIOAcceleration_static)
set (SHARED_LIB_NAME TreeIOAcceleration)

set (PROJECT_HEADERS
    TreeIO/impl/TreeIOAccelerationUtils.h

    TreeIO/impl/TreeIOAcceleration.h
    TreeIO/impl/TreeIOPointCloud.h
    TreeIO/impl/TreeIOPolygonalModel.h
)

set (PROJECT_SOURCES
    TreeIO/impl/TreeIOAccelerationUtils.cpp

    TreeIO/impl/TreeIOAcceleration.cpp
    TreeIO/impl/TreeIOPointCloud.cpp
    TreeIO/impl/TreeIOPolygonalModel.cpp
)

find_package (Eigen3 REQUIRED)

set (PROJECT_LIBRARIES
    ${COMMON_LIB}::NanoFLANN
    ${COMMON_LIB}::Intersect
    ${COMMON_LIB}::KdTree
    Eigen3::Eigen
    TreeIO::TreeIOLib_static
)

add_library (
    ${STATIC_LIB_NAME} STATIC
    ${PROJECT_HEADERS}
    ${PROJECT_SOURCES}
)

target_include_directories (
    ${STATIC_LIB_NAME}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include/
)

target_link_libraries (
    ${STATIC_LIB_NAME}
    ${PROJECT_LIBRARIES}
)

add_library (
    TreeIO::${STATIC_LIB_NAME}
    ALIAS
    ${STATIC_LIB_NAME}
)

add_library (
    ${SHARED_LIB_NAME} SHARED
    ${PROJECT_HEADERS}
    ${PROJECT_SOURCES}
)

target_include_directories (
    ${SHARED_LIB_NAME}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include/
)

target_link_libraries (
    ${SHARED_LIB_NAME}
    ${PROJECT_LIBRARIES}
)

add_library (
    TreeIO::${SHARED_LIB_NAME}
    ALIAS
    ${SHARED_LIB_NAME}
)

# Prepare variables for the test generation.
set (PROJECT_LIB_NAME ${STATIC_LIB_NAME})
set (PROJECT_TEST_NAME ${PROJECT_NAME}_test)
# Prepare project tests.
include (../test/CMakeLists.txt)
