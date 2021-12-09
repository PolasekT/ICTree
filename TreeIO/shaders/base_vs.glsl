/**
 * @author David Hrusa, Tomas Polasek
 * @date 5.19.2020
 * @version 1.0
 * @brief Vertex shader used for basic 3D rendering.
 */

#version 400 core

#extension GL_ARB_explicit_uniform_location : enable
#extension GL_ARB_separate_shader_objects : enable

/// Position of the vertex in model space.
layout(location = 0) in vec4 iPosition;
/// Color of the vertex.
layout(location = 1) in vec4 iColor;

/// Color of the vertex.
layout(location = 0) out vec4 vColor;
/// Unique identifier of the vertex.
layout(location = 1) flat out int vVetexId;
/// Screen-space position of the veretx.
layout(location = 2) out vec4 vPosition;
/// World-space position of the vertex.
layout(location = 3) out vec4 vWSPosition;

/// Color override value.
layout(location = 0) uniform vec4 uColorOverride = vec4(0.0f, 1.0f, 0.0f, 1.0f);
/// Set to true to force use color override.
layout(location = 1) uniform bool uDoOverrideColor = false;
/// Set to true to use override colors from override storage buffer.
layout(location = 2) uniform bool uUseColorStorage = false;
/// Model matrix.
layout(location = 4) uniform mat4 uModel;
/// Model-view-projection matrix.
layout(location = 5) uniform mat4 uMVP;
/// Enable shading calculation (true) or just use flat colors (false).
layout(location = 6) uniform bool uShaded = true;
/// Shadow map used when uShaded == true.
layout(location = 7) uniform sampler2D uShadowMap;
/// Specification of the soft shadow kernel - x = kernel size, y = sampling factor, z = kernel strength, w = bias.
layout(location = 8) uniform vec4 uShadowKernelSpec;
/// Matrix used to transform world-space location to the shadow map.
layout(location = 9) uniform mat4 uLightViewProjection;

void main()
{
    gl_Position = uMVP * iPosition;
    vColor = iColor;
    vPosition = gl_Position;
    vWSPosition = uModel * iPosition;
    vVetexId = gl_VertexID;
}
