/**
 * @author David Hrusa, Tomas Polasek
 * @date 4.21.2020
 * @version 1.0
 * @brief Compute shader for point selection.
 */

#version 430 core

layout(local_size_x = 16) in;

/// Information about dispatch group count.
uniform vec3 uGroupCount;
/// Information about total number of items being processed.
uniform vec3 uItemCount;

/// Model-view-projection matrix for the skeleton.
uniform mat4 uMVP;

/// Position of the first corner of the selection rectangle.
uniform vec2 uCursorPosA;
/// Position of the second corner of the selection rectangle.
uniform vec2 uCursorPosB;

/// Positions of the input points with uItemCount elements.
layout(binding = 0) buffer bInData
{ float bPositions[]; };

/// Packed output elements.
struct VertexSelectionFlag
{ float selected[4]; };

/// Resulting selections buffer with uItemCount elements.
layout(std430, binding = 1) buffer bOutData
{ VertexSelectionFlag bSelected[]; };


/// @brief Fetch position of the input point at given index.
vec4 fetchPosition(uint index)
{
    const vec4 position = vec4(
        bPositions[index * 3 + 0],
        bPositions[index * 3 + 1],
        bPositions[index * 3 + 2],
        0.0f
    );
    return vec4(position[0], position[1], position[2], position[3]);
}

/// @brief Set selected flag for point at given index.
void setSelected(uint index, bool value)
{
    const uint packIdx = uint(index / 4);
    const uint elementIdx = uint(mod(index, 4));
    bSelected[packIdx].selected[elementIdx] = float(value);
}

void main()
{
    // Linear indexing:
    const uint myIdx = gl_GlobalInvocationID.x;
    if (myIdx > uItemCount.x)
    { return; }

    // Evaluate selection:
    const vec4 msPosition = fetchPosition(myIdx);
    // Project point to NDC:
    vec4 ndcPos = uMVP * vec4(msPosition.xyz, 1.0f);
    const bool isInFront = ndcPos.w > 0.0f;
    ndcPos /= ndcPos.w;
    // Save selection resolution:
    const bool selected = (
        ndcPos.x > uCursorPosA.x && ndcPos.y > uCursorPosA.y &&
        ndcPos.x < uCursorPosB.x && ndcPos.y < uCursorPosB.y &&
        isInFront);

    setSelected(myIdx, selected);
}