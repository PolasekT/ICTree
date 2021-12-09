#################################################################################
# @project TreeIO
# @brief This project contains utility used for displaying and editing skeletons.
# @author Tomas Polasek
#################################################################################

cmake_minimum_required (VERSION 3.9)

set (PROJECT_NAME TreeIO)
project (${PROJECT_NAME})

set (STATIC_LIB_NAME ${PROJECT_NAME}_static)
set (SHARED_LIB_NAME ${PROJECT_NAME})

set (PROJECT_HEADERS
    TreeIO/impl/TreeMath.h
    TreeIO/impl/TreeTemplateUtils.h
    TreeIO/impl/TreeUtils.h
    TreeIO/impl/TreeGLUtils.h
    TreeIO/impl/TreeRandomEngine.h
    TreeIO/impl/TreeHistory.h
    TreeIO/impl/TreeOperations.h
    TreeIO/impl/TreeRuntimeMetaData.h
    TreeIO/impl/TreeCamera.h
    TreeIO/impl/TreeSceneInstanceNames.h
    TreeIO/impl/TreeScene.h
    TreeIO/impl/TreeReconstruction.h
    TreeIO/impl/TreeShader.h
    TreeIO/impl/TreeBuffer.h
    TreeIO/impl/TreeTextureBuffer.h
    TreeIO/impl/TreeFrameBuffer.h
    TreeIO/impl/TreeRenderer.h
    TreeIO/impl/TreeRendererModality.h
    TreeIO/impl/TreeShaderCompatBlur.h
    TreeIO/impl/TreeShaderCompatFullscreen.h
    TreeIO/impl/TreeShaderCompatMesh.h
    TreeIO/impl/TreeShaderCompatRecon.h
    TreeIO/impl/TreeShaderCompatScreen.h
    TreeIO/impl/TreeRenderSystem.h
    TreeIO/impl/TreeRenderSystemPhoto.h
    TreeIO/impl/TreeRenderSystemRT.h
    TreeIO/impl/TreeRenderHelper.h
    TreeIO/impl/TreeStats.h
    TreeIO/impl/TreeAugmentation.h
)

set (PROJECT_SOURCES
    TreeIO/impl/TreeUtils.cpp
    TreeIO/impl/TreeGLUtils.cpp
    TreeIO/impl/TreeRandomEngine.cpp
    TreeIO/impl/TreeOperations.cpp
    TreeIO/impl/TreeCamera.cpp
    TreeIO/impl/TreeRuntimeMetaData.cpp
    TreeIO/impl/TreeScene.cpp
    TreeIO/impl/TreeReconstruction.cpp
    TreeIO/impl/TreeShader.cpp
    TreeIO/impl/TreeBuffer.cpp
    TreeIO/impl/TreeTextureBuffer.cpp
    TreeIO/impl/TreeFrameBuffer.cpp
    TreeIO/impl/TreeRenderer.cpp
    TreeIO/impl/TreeRendererModality.cpp
    TreeIO/impl/TreeShaderCompatBlur.cpp
    TreeIO/impl/TreeShaderCompatFullscreen.cpp
    TreeIO/impl/TreeShaderCompatMesh.cpp
    TreeIO/impl/TreeShaderCompatRecon.cpp
    TreeIO/impl/TreeShaderCompatScreen.cpp
    TreeIO/impl/TreeRenderSystem.cpp
    TreeIO/impl/TreeRenderSystemPhoto.cpp
    TreeIO/impl/TreeRenderSystemRT.cpp
    TreeIO/impl/TreeRenderHelper.cpp
    TreeIO/impl/TreeStats.cpp
    TreeIO/impl/TreeAugmentation.cpp
)

if (WIN32)
    # Locate the TBB library
    find_package(TBB CONFIG REQUIRED)

    # Add required libraries when building under windows.
    set (ADDITIONAL_LIBS
        TBB::tbb
    )

    # Locate OpenGL
    if (${IO_USE_EGL})
        message(STATUS "Building TreeIO with headless EGL rendering support!")

        find_package (OpenGL REQUIRED COMPONENTS OpenGL EGL GLX)
        set (OPENGL_LIBS
            OpenGL::OpenGL
            OpenGL::EGL
            OpenGL::GLX
        )

        add_definitions (-DIO_USE_EGL)
    else ()
        find_package (OpenGL REQUIRED)
        set (OPENGL_LIBS
            OpenGL::GL
        )
    endif ()
else ()
    # Prefer GLX over legacy GL library.
    set (OpenGL_GL_PREFERENCE GLVND)
    set (IlmBase_Half_FOUND True)

    # Add required libraries when building under linux.
    set (ADDITIONAL_LIBS
        tbb
        dl
    )

    # Locate OpenGL
    if (${IO_USE_EGL})
        message(STATUS "Building TreeIO with headless EGL rendering support!")

        find_package (OpenGL REQUIRED COMPONENTS OpenGL EGL GLX)
        set (OPENGL_LIBS
            OpenGL::OpenGL
            OpenGL::EGL
            OpenGL::GLX
        )

        add_definitions (-DIO_USE_EGL)
    else ()
        find_package (OpenGL REQUIRED)
        set (OPENGL_LIBS
            OpenGL::GL
        )
    endif ()
endif ()

# Find required libraries.
find_package (GLEW REQUIRED)
find_package (GLUT REQUIRED)
find_package (TBB REQUIRED)

set (PROJECT_LIBRARIES
    ${OPENGL_LIBS}
    GLEW::GLEW
    GLUT::GLUT
    TreeIO::TreeIOLib_static
    TreeIO::TreeIOAcceleration_static
    ${ADDITIONAL_LIBS}
)

add_library (
    ${STATIC_LIB_NAME} STATIC
    ${PROJECT_HEADERS}
    ${PROJECT_SOURCES}
)

target_include_directories (${STATIC_LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories (${STATIC_LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include/)
target_include_directories (${STATIC_LIB_NAME} PUBLIC ${INCLUDE_MAIN_DIR})
target_include_directories (${STATIC_LIB_NAME} PUBLIC ${INCLUDE_COMMON_DIR})
target_include_directories (${STATIC_LIB_NAME} PUBLIC ${INCLUDE_CSVPP_DIR})
target_include_directories (${STATIC_LIB_NAME} PUBLIC ${TBB_INCLUDE_DIRS})
target_include_directories (${STATIC_LIB_NAME} PUBLIC ${OpenEXR_INCLUDE_DIRS})

target_link_libraries (
    ${STATIC_LIB_NAME}
    ${PROJECT_LIBRARIES}
)

add_library (
    ${PROJECT_NAME}::${STATIC_LIB_NAME}
    ALIAS
    ${STATIC_LIB_NAME}
)

add_library (
    ${SHARED_LIB_NAME} SHARED
    ${PROJECT_HEADERS}
    ${PROJECT_SOURCES}
)

target_include_directories (${SHARED_LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories (${SHARED_LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include/)
target_include_directories (${SHARED_LIB_NAME} PUBLIC ${INCLUDE_MAIN_DIR})
target_include_directories (${SHARED_LIB_NAME} PUBLIC ${INCLUDE_COMMON_DIR})
target_include_directories (${SHARED_LIB_NAME} PUBLIC ${INCLUDE_CSVPP_DIR})
target_include_directories (${SHARED_LIB_NAME} PUBLIC ${TBB_INCLUDE_DIRS})
target_include_directories (${SHARED_LIB_NAME} PUBLIC ${OpenEXR_INCLUDE_DIRS})

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
