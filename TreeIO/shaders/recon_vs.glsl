/**
 * @author Tomas Polasek, David Hrusa, Yichen Sheng
 * @date 4.15.2020
 * @version 1.0
 * @brief Vertex shader for the photo-mode rendering of trees.
 */

#version 430 core

/// Position of the vertex in model space, w is always set to 1.0f.
layout(location = 0) in vec4 iPosition;
/// Normal of the vertex in model space, w is set to distance from the tree root.
layout(location = 1) in vec4 iNormal;
/// Vector parallel with the branch, w is set to the radius of the branch.
layout(location = 2) in vec4 iParallel;
/// Tangent of the branch, w is unused.
layout(location = 3) in vec4 iTangent;
/// Adjacency indices for this vertex, x = this idx, y = parent idx, z = child idx, w is unused. 0 as invalid idx.
layout(location = 4) in uvec4 iAdjacency;

/// Position of the vertex in world space, w is unused.
out flat vec4 vPosition;
/// Normal of the vertex in world space, w is set to distance from the tree root.
out flat vec4 vNormal;
/// Vector parallel with the branch in world space, w is set to the radius of the branch.
out flat vec4 vParallel;
/// Tangent of the branch in world space, w is unused.
out flat vec4 vTangent;
/// Adjacency indices for this vertex, x = this idx, y = parent idx, z = child idx, w is unused. 0 as invalid idx.
out flat uvec4 vAdjacency;

#include "recon_common.glsl"

void main()
{
    // Transform coordinates from model-space to world-space.
    vPosition = uModel * iPosition;
    // Keep direction vectors in model-space, to be transformed later.
    vNormal = iNormal;
    vParallel = iParallel;
    vTangent = iTangent;
    vAdjacency = iAdjacency;
}
