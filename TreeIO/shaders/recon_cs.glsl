/**
 * @author Tomas Polasek, David Hrusa, Yichen Sheng
 * @date 4.15.2020
 * @version 1.0
 * @brief Tessellation control shader for the photo-mode rendering of trees.
 */

#version 430 core

// Output = 2 vertices as the end-points of the current segment.
layout(vertices = 2) out;

/// Position of the vertex in world space, w is unused.
in flat vec4 vPosition[];
/// Normal of the vertex in world space, w is set to distance from the tree root.
in flat vec4 vNormal[];
/// Vector parallel with the branch in world space, w is set to the radius of the branch.
in flat vec4 vParallel[];
/// Tangent of the branch in world space, w is unused.
in flat vec4 vTangent[];
/// Adjacency indices for this vertex, x = this idx, y = parent idx, z = child idx, w is unused. 0 as invalid idx.
in flat uvec4 vAdjacency[];

/// Position of the vertex in world space.
out flat vec3 cPosition[];
/// Normal of the vertex in world space.
out flat vec3 cNormal[];
/// Distance of this node from tree root.
out flat float cDistanceFromRoot[];
/// Vector parallel with the branch in world space.
out flat vec3 cParallel[];
/// Radius of the branch.
out flat float cBranchRadius[];
/// Tangent of the branch in world space.
out flat vec3 cTangent[];
/// Adjacency indices for this vertex, x = this idx, y = parent idx, z = child idx, w is unused. 0 as invalid idx.
out flat uvec4 cAdjacency[];

#include "recon_common.glsl"

/// Minimal vertical tessellation of the branch:
const float MIN_TESSELLATION = 5.0f;
/// Maximal vertical tessellation of the branch:
const float MAX_TESSELLATION = 100.0f;

void main()
{
    // Fetch input data:
    const vec3 wsPosS = vPosition[0].xyz;
    const vec3 wsPosT = vPosition[1].xyz;

    // Currently disabled.
    const float distS = min(1.0f, max(1.0f, length(uCameraPos - wsPosS)));
    const float distT = min(1.0f, max(1.0f, length(uCameraPos - wsPosT)));

    // Calculate tessellation level based on edge length and distance to camera.
    float edgeLength = length(wsPosT - wsPosS);
    float avgDistance = (distS + distT) / 2.0f;
    float tessellationLevel = clamp(
        // 13.0f is magical constant to keep quality on par with horizontal tessellation.
        uTessellationMultiplier * 13.0f * edgeLength / avgDistance,
        MIN_TESSELLATION, MAX_TESSELLATION
    );

    // Set tessellation levels:
    gl_TessLevelOuter[0] = 1.0f;
    gl_TessLevelOuter[1] = max(1.0f, tessellationLevel);

    // Pass data to the evaluation shader:
    cPosition[gl_InvocationID] = vPosition[gl_InvocationID].xyz;
    cNormal[gl_InvocationID] = vNormal[gl_InvocationID].xyz;
    cDistanceFromRoot[gl_InvocationID] = vNormal[gl_InvocationID].w;
    cParallel[gl_InvocationID] = vParallel[gl_InvocationID].xyz;
    cBranchRadius[gl_InvocationID] = vParallel[gl_InvocationID].w;
    cTangent[gl_InvocationID] = vTangent[gl_InvocationID].xyz;
    cAdjacency[gl_InvocationID] = vAdjacency[gl_InvocationID];
}
