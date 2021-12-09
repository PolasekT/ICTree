/**
 * @author David Hrusa, Tomas Polasek
 * @date 5.19.2020
 * @version 1.0
 * @brief Fragment shader used for basic 3D rendering.
 */

#version 430

#extension GL_ARB_explicit_uniform_location : enable
#extension GL_ARB_separate_shader_objects : enable

/// Color of the vertex.
layout(location = 0) in vec4 vColor;
/// Unique identifier of the vertex.
layout(location = 1) flat in int vVetexId;
/// Screen-space position of the veretx.
layout(location = 2) in vec4 vPosition;
/// World-space position of the vertex.
layout(location = 3) in vec4 vWSPosition;

/// Output fragment color.
layout(location = 0) out vec4 fFragmentColor;

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

/// @brief Extra information provideable for each vertex.
struct ExtraVertexInfo
{ vec4 color; };

/// @brief Override buffer.
layout(std140, binding = 0) buffer bOverride
{
    /// Extra vertex information used as overrides.
    ExtraVertexInfo bExtraVertexInfo[];
}; // buffer bOverride

#include "modality_common.glsl"
#include "random_common.glsl"
#include "shadow_common.glsl"

/// @brief Calculate full shading for current fragment.
vec4 calculateShading(out float occlusion)
{
    // Calculate shadow-map projected coordinate and perform bias transformation.
    vec4 smCoordBiased = calculateBiasedSMCoord(uLightViewProjection, vWSPosition);

    // Calculate occlusion. If the light does not see this frament keep it lit.
    float fragmentLit = 0.0f;
    if (smCoordBiased.x > 1.0f || smCoordBiased.y > 1.0f ||
        smCoordBiased.x < 0.0f || smCoordBiased.y < 0.0f ||
        smCoordBiased.z < 0.0f || smCoordBiased.z > 1.0f)
    { fragmentLit = 1.0f; }
    else
    { fragmentLit = calcShadow(uShadowMap, uShadowKernelSpec, smCoordBiased, vWSPosition.xyz); }

    occlusion = fragmentLit;
    return vec4(fragmentLit, fragmentLit, fragmentLit, 1.0f);
}

/// @brief Calculate only flat color without advanced shading.
vec4 calculateFlatColor()
{
    vec4 resultColor = vec4(1.0f, 0.0f, 1.0f, 1.0f);

    // Override using the base color specified.
    if (uDoOverrideColor)
    { resultColor = uColorOverride; }
    else
    { resultColor = vColor; }

    // Override from provided extra information storage.
    if (uUseColorStorage)
    { resultColor = bExtraVertexInfo[vVetexId].color; }

    return resultColor;
}

void main()
{
    float occlusion = 1.0f;

    // Calculate shading
    if (uShaded)
    { fFragmentColor = calculateShading(occlusion); }
    else
    { fFragmentColor = calculateFlatColor(); }

    // Perform modality formatting.
    fFragmentColor = formatModality(fFragmentColor, vec4(0.3f, 0.3f, 0.3f, 1.0f), vec3(1.0f, 1.0f, 1.0f),
        occlusion, vec3(0.0f, 0.0f, 0.0f), vPosition.zw);
}
