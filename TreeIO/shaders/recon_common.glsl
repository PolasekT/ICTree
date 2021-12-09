/**
 * @author Tomas Polasek, David Hrusa, Yichen Sheng
 * @date 5.22.2020
 * @version 1.0
 * @brief Common code for all reconstruction shaders.
 */

// List of all photo uniforms:
/// Model matrix of the displayed mesh.
uniform mat4 uModel;
/// View matrix of the current camera.
uniform mat4 uView;
/// Projection matrix of the current camera.
uniform mat4 uProjection;
/// View-projection matrix of the current camera.
uniform mat4 uVP;
/// Model-view-projection matrix of displayed mesh.
uniform mat4 uMVP;
/// Inverse transpose of MV matrix. Used for transforming normals.
uniform mat4 uNT;
/// Light view projection matrix.
uniform mat4 uLightView;
/// Position of the camera.
uniform vec3 uCameraPos;
/// Position of the light.
uniform vec3 uLightPos;
/// Scale of the mesh.
uniform vec3 uMeshScale;
/// Multiplier used to calculate tessellation level. Default value = 500.0f .
uniform float uTessellationMultiplier;
/// Shadow map from the position of the light.
uniform sampler2D uShadowMap;
/// Perform basic shading (0) or photo-mode shading (1)?
uniform int uPhotoShading;
/// Should the shadow be displayed?
uniform int uApplyShadow;
/// Specification of the soft shadow kernel - x = kernel size, y = sampling factor, z = kernel strength, w = bias.
uniform vec4 uShadowKernelSpec;
/// Color used for foreground areas.
uniform vec3 uForegroundColor;
/// Color used for background areas.
uniform vec3 uBackgroundColor;
/// How opaque should the reconstruction be. 1.0f for fully opaque and 0.0f for fully transparent.
uniform float uOpaqueness;
/// Maximum branch radius.
uniform float uMaxBranchRadius;
/// Maximum distance from the root node.
uniform float uMaxDistanceFromRoot;
/// Tension of the interpolated branch curve.
uniform float uBranchTension;
/// Bias of the interpolated branch curve.
uniform float uBranchBias;
/// Scaler used for branch width multiplication.
uniform float uBranchWidthMultiplier;

/// @brief Container for data of one given vertex.
struct VertexData
{
    /// Position of the vertex in model space, w is always set to 1.0f.
    vec4 position;
    /// Normal of the vertex in model space, w is set to distance from the tree root.
    vec4 normal;
    /// Vector parallel with the branch, w is set to the radius of the branch.
    vec4 parallel;
    /// Tangent of the branch, w is unused.
    vec4 tangent;
    /// Adjacency indices for this vertex, x = this idx, y = parent idx, z = child idx, w is unused. -1 as invalid idx.
    uvec4 adjacency;
}; // struct VertexData

// List of all photo buffers:
/// @brief Buffer containing the draw data used for adjacency information.
layout(std140, binding = 0) buffer bData
{
    VertexData vertices[];
}; // buffer bData

/// @brief Minimum distance between two nodes when MRF is calculated.
#define MINIMUM_MRF_DISTANCE 0.00001f
