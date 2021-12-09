/**
 * @author David Hrusa, Tomas Polasek
 * @date 5.19.2020
 * @version 1.0
 * @brief Vertex shader used for UI and 2D rendering.
 */

#version 400 core

#extension GL_ARB_explicit_uniform_location : enable
#extension GL_ARB_separate_shader_objects : enable

/// Position of the point.
layout(location = 0) in vec4 iPosition;
/// Color of the point.
layout(location = 1) in vec4 iColor;

/// Color of the point.
layout(location = 0) out vec4 vColor;
/// Unique identifier of the vertex.
layout(location = 1) flat out int vVertId;

/// Model-View-Projection matrix.
layout(location = 0) uniform mat4 uMVP;
/// Set to true to interpret color as: RG -> UV and B -> texture ID.
layout(location = 1) uniform bool uTextured = false;
/// List of textures indexed by the blue color channel.
layout(location = 2) uniform sampler2D uTextures[10];

void main()
{
    gl_Position = uMVP * iPosition;

    vColor = iColor;
    vVertId = gl_VertexID;
}
