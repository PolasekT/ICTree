#################################################################################
# @project PerceptualMetric
# @brief This project contains modules for machine learning and perceptual metric.
# @author Tomas Polasek
#################################################################################

cmake_minimum_required (VERSION 3.9)
cmake_policy (SET CMP0074 NEW)

set (PROJECT_NAME PyTreeIO)
project (${PROJECT_NAME})

set (PROJECT_LIB_NAME PyTreeIO)

set (Python_FIND_STRATEGY LOCATION)
find_package(
    Python 
    COMPONENTS Interpreter Development
)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import os,skbuild; print(os.path.dirname(skbuild.__file__).replace('\\\\', '/'))"
    OUTPUT_VARIABLE SKBUILD_BASE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set (PYTHON_LIBRARY ${Python_LIBRARY_DIRS})
set (PYTHON_INCLUDE_DIR ${Python_INCLUDE_DIRS})
set (PYTHON_PREFIX ${Python_ROOT_DIR})
get_filename_component (PYTHON_LIBRARY_DIR ${PYTHON_INCLUDE_DIR} PATH)
set (PYTHON_LIBRARY_DIR "${PYTHON_LIBRARY_DIR}/libs/")

list (APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package (PythonExtensions MODULE REQUIRED)
set (CYTHON_FLAGS "--directive always_allow_keywords=true")
find_package (Cython MODULE REQUIRED)

set (PROJECT_SRC_DIR src)

set (PROJECT_CYTHON_HEADERS
    ${PROJECT_SRC_DIR}/treeio.pyx
)

set (PROJECT_LIBRARIES
    TreeIO::TreeIOLib_static
    TreeIO::TreeIO_static
)

set (PROJECT_DEFINITIONS
)

set (PROJECT_LINK_OPTIONS
)

set (PROJECT_LINK_DIRECTORIES
    ${PYTHON_LIBRARY_DIR}
)

set (PROJECT_SUB_TARGETS "")

foreach (PYX_HEADER ${PROJECT_CYTHON_HEADERS})
    get_filename_component (PYX_PATH ${PYX_HEADER} DIRECTORY)
    get_filename_component (PYX_NAME ${PYX_HEADER} NAME_WE)
    get_filename_component (PYTHON_MODULE_DST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../lib/${PYX_NAME}/lib/ ABSOLUTE)
    add_cython_target (${PYX_HEADER} CXX PY3 OUTPUT_VAR PYX_SOURCE BASE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
    set (PROJECT_SOURCES
        ${PYX_SOURCE}
        ${PYX_PATH}/${PYX_NAME}.cxx
    )
    set (PROJECT_HEADERS "")
    set (PROJECT_SUB_TARGETS ${PROJECT_SUB_TARGETS} ${PYX_NAME})

    set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PYTHON_MODULE_DST_DIR})
    set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PYTHON_MODULE_DST_DIR})
    set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PYTHON_MODULE_DST_DIR})

    # Static Cython library:

    add_library (
        ${PYX_NAME} MODULE
        ${PROJECT_SOURCES}
        ${PROJECT_HEADERS}
    )
    python_extension_module (${PYX_NAME})

    target_include_directories (
        ${PYX_NAME}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/
    )

    target_link_libraries (
        ${PYX_NAME}
        ${PROJECT_LIBRARIES}
    )

    target_compile_definitions (
        ${PYX_NAME} PUBLIC
        ${PROJECT_DEFINITIONS}
    )

    target_link_options (
        ${PYX_NAME} PUBLIC
        ${PROJECT_LINK_OPTIONS}
    )

    target_link_directories (
        ${PYX_NAME} PUBLIC
        ${PROJECT_LINK_DIRECTORIES}
    )

    install (TARGETS ${PYX_NAME} LIBRARY DESTINATION ${PYX_NAME})
endforeach ()

# Aggregate target building all sources:

add_custom_target (${PROJECT_LIB_NAME})
add_dependencies (${PROJECT_LIB_NAME} ${PROJECT_SUB_TARGETS})
