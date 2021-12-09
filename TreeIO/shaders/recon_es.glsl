/**
 * @author Tomas Polasek, David Hrusa, Yichen Sheng
 * @date 4.15.2020
 * @version 1.0
 * @brief Tessellation evaluation shader for the photo-mode rendering of trees.
 */

#version 430 core

// Input = Tessellated lines representing the current segment, which will be correctly positioned.
layout(isolines, equal_spacing) in;

/// Position of the vertex in world space.
in flat vec3 cPosition[];
/// Normal of the vertex in world space.
in flat vec3 cNormal[];
/// Distance of this node from tree root.
in flat float cDistanceFromRoot[];
/// Vector parallel with the branch in world space.
in flat vec3 cParallel[];
/// Radius of the branch.
in flat float cBranchRadius[];
/// Tangent of the branch in world space.
in flat vec3 cTangent[];
/// Adjacency indices for this vertex, x = this idx, y = parent idx, z = child idx, w is unused. 0 as invalid idx.
in flat uvec4 cAdjacency[];

/// Position of the vertex in world space.
out vec3 ePosition;
/// Normal of the vertex in world space.
out vec3 eNormal;
/// Distance of this node from tree root.
out float eDistanceFromRoot;
/// Vector parallel with the branch in world space.
out vec3 eParallel;
/// Texture coordinate of the point.
out vec2 eTextureUV;
/// Radius of the branch.
out float eBranchRadius;
/// Tangent of the branch in world space.
out vec3 eTangent;
/// Total length of the generated branch without tessellation.
out float eEdgeLength;
/// Total number of generated vertices in the generated branch, excluding the last one.
out float eVertexCount;

#include "recon_common.glsl"

/// @brief Fetch vertex data for given index. Defaults to input values if the index is invalid.
void fetchAdjacencyData(in uint index, in vec4 inPosition, in vec4 inNormal,
in vec4 inParallel, in vec4 inTangent, in uvec4 inAdjacency, out VertexData data)
{
    if (index > 0u)
    {
        data = vertices[index - 1u];
        data.position = uModel * data.position;
    }
    else
    {
        data.position = inPosition;
        data.normal = inNormal;
        data.parallel = inParallel;
        data.tangent = inTangent;
        data.adjacency = inAdjacency;
    }
}

/// @brief Fetch data of the parent and child vertices. Defaults to input values.
void fetchAdjacencyData(out vec3 parentPosition, out vec3 childPosition)
{
    // Parent of the first vertex.
    const uint parentIdx = cAdjacency[0].y;
    VertexData parentData;
    fetchAdjacencyData(parentIdx, vec4(cPosition[0], 1.0f),
    vec4(cNormal[0], cDistanceFromRoot[0]), vec4(cParallel[0], cBranchRadius[0]),
    vec4(cTangent[0], 1.0f), cAdjacency[0], parentData);

    // Child of the second vertex.
    const uint childIdx = cAdjacency[1].z;
    VertexData childData;
    fetchAdjacencyData(childIdx, vec4(cPosition[1], 1.0f),
    vec4(cNormal[1], cDistanceFromRoot[1]), vec4(cParallel[1], cBranchRadius[1]),
    vec4(cTangent[1], 1.0f), cAdjacency[1], childData);

    parentPosition = parentData.position.xyz;
    childPosition = childData.position.xyz;
}

/// @brief Perform hermite interpolation over v0, v1, v2, v3.
void hermiteInterpolation(in vec3 v0, in vec3 v1, in vec3 v2, in vec3 v3,
out vec3 position, out vec3 parallel, out vec3 norm, in float t,
in float tension, in float bias)
{
    const float t2 = t * t;
    const float t3 = t2 * t;

    const vec3 m0 = (v1 - v0) * ( 1.0f + bias) * (1.0f - tension) / 2.0f +
    (v2 - v1) * ( 1.0f - bias) * (1.0f - tension) / 2.0f;
    const vec3 m1 = (v2 - v1) * (1.0f + bias) * (1.0f - tension) / 2.0f +
    (v3 - v2) * (1.0f - bias) * (1.0f - tension) / 2.0f;

    const float a0 =  2.0f * t3  - 3.0f * t2 + 1.0f;
    const float a1 =         t3  - 2.0f * t2 + t;
    const float a2 =         t3  -        t2;
    const float a3 = -2.0f * t3  + 3.0f * t2;

    position = vec3(a0 * v1 + a1 * m0 + a2 * m1 + a3 * v2);

    const vec3 d1 = (( 6.0f  * t2 - 6.0f * t) * v1) +
    (( 3.0f  * t2 - 4.0f * t + 1.0f) * m0) +
    (( 3.0f  * t2 - 2.0f * t) * m1) +
    ((-6.0f  * t2 + 6.0f * t) * v2);

    parallel = normalize(d1);
    // TODO - I don't think this helps. If anything, it creates discontinuity between neighboring segments.
    //if(t <= 0.0f) { parallel =  -(v0 - v1); }
    //if(t >= 1.0f) { parallel =   (v3 - v2); }

    const vec3 d2 = (( 12.0f * t  - 6.0f) * v1) +
    (( 6.0f  * t  - 4.0f) * m0) +
    (( 6.0f  * t  - 2.0f) * m1) +
    ((-12.0f * t  + 6.0f) * v2);

    norm = normalize(d2);
}

/// @brief Perform cubic interpolation over v0, v1, v2, v3.
void cubicInterpolation(in vec3 v0, in vec3 v1, in vec3 v2, in vec3 v3,
out vec3 position, out vec3 parallel, out vec3 norm, float t)
{
    const float t2 = t * t;
    const float t3 = t2 * t;

    const vec3 a0 = v3 - v2 - v0 + v1;
    const vec3 a1 = v0 - v1 - a0;
    const vec3 a2 = v2 - v0;
    const vec3 a3 = v1;

    position = vec3(a0 * t3 + a1 * t2 + a2 * t + a3);

    const vec3 d1 = vec3(3.0f * a0 * t2 + 2.0f * a1 * t + a2);
    parallel = normalize(d1);

    const vec3 d2 = vec3(6.0f * a0 * t + 2.0f * a1);
    norm = normalize(d2);
}

/// @brief Perform linear interpolation over v0, v1, v2, v3.
void linearInterpolation(in vec3 v0, in vec3 v1, in vec3 v2, in vec3 v3,
out vec3 position, out vec3 parallel, out vec3 norm, float t)
{
    position = vec3(v1 * t + v2 * (1.0f - t));
    parallel = normalize(v2 - v1);
    norm = parallel;
}

/// @brief Spherical linear interpolation over v0, v1. This version presumes normalized input vectors.
vec3 slerpNormalized(vec3 v0, vec3 v1, float t)
{
    const float DOT_LIMIT = 0.9999;
    const float dot01 = dot(v0, v1);
    if ((dot01 > DOT_LIMIT) || (dot01 < -DOT_LIMIT))
    {
        if (t <= 0.5f)
        { return v0; }
        else
        { return v1; }
    }

    const float theta = acos(dot01);
    const vec3 p = (v0 * sin((1.0f - t) * theta) + v1 * sin(t * theta)) / sin(theta);

    return p;
}

/// @brief Spherical linear interpolation over v0, v1.
vec3 slerp(vec3 v0, vec3 v1, float t)
{
    const vec3 v0Norm = normalize(v0);
    const vec3 v1Norm = normalize(v1);

    return slerpNormalized(v0Norm, v1Norm, t);
}

void main()
{
    // How far are we along the original line.
    const float argT = gl_TessCoord.x;

    // Recover values for the end-points S and T:
    const vec3 wsPosS = cPosition[0].xyz;
    const vec3 wsPosT = cPosition[1].xyz;
    const float wsPosDistance = length(wsPosT - wsPosS);

    const vec3 wsNormalS = cNormal[0].xyz;
    const vec3 wsNormalT = cNormal[1].xyz;
    const vec3 wsNormalST = normalize(slerp(wsNormalS, wsNormalT, argT));

    const vec3 wsParallelS = cParallel[0].xyz;
    const vec3 wsParallelT = cParallel[1].xyz;
    const vec3 wsParallelST = normalize(slerp(wsParallelS, wsParallelT, argT));

    const vec3 wsTangentS = cTangent[0].xyz;
    const vec3 wsTangentT = cTangent[1].xyz;
    const vec3 wsTangentST = normalize(slerp(wsTangentS, wsTangentT, argT));

    const float radiusS = cBranchRadius[0];
    const float radiusT = cBranchRadius[1];

    const float distanceS = cDistanceFromRoot[0];
    const float distanceT = cDistanceFromRoot[1];

    // Calculate value of argument where the branch radius of parent node ends.
    const float diameterArgT = (wsPosDistance - radiusS) / wsPosDistance;

    // Fetch positions of the S's parent and T's child.
    vec3 wsPosSP, wsPosTC;
    fetchAdjacencyData(wsPosSP, wsPosTC);

    // Build 4 anchor points for interpolation.
    const vec3 v0 = wsPosSP;
    const vec3 v1 = wsPosS;
    const vec3 v2 = wsPosT;
    const vec3 v3 = wsPosTC;

    // Interpolate branch radius.
#if 1
    const float radius = uBranchWidthMultiplier * mix(radiusS, radiusT, smoothstep(0.0f, radiusT / radiusS, argT));
#else
    const float radius = uBranchWidthMultiplier * mix(radiusS, radiusT, smoothstep(diameterArgT, 1.0f, argT));
#endif

    // Interpolate using the anchor points.
    vec3 wsPosition, wsParallel, wsNormal, wsTangent;
    //linearInterpolation(v0, v1, v2, v3, wsPosition, wsParallel, wsNormal, argT);
    //cubicInterpolation(v0, v1, v2, v3, wsPosition, wsParallel, wsNormal, argT);
    // Make interpolated curves more straight when two nodes are near each other.
    const float tensionProximityFactor = 1.0f - smoothstep(MINIMUM_MRF_DISTANCE, MINIMUM_MRF_DISTANCE * radius * 1000.0f, wsPosDistance);
    const float tension = clamp(uBranchTension + tensionProximityFactor, 0.0f, 0.995f);
    const float biasProximityFactor = 1.0f - smoothstep(MINIMUM_MRF_DISTANCE, MINIMUM_MRF_DISTANCE * radius * 1000.0f, wsPosDistance);
    const float bias = clamp(uBranchBias + biasProximityFactor * 0.0f, 0.0f, 0.995f);
    hermiteInterpolation(v0, v1, v2, v3, wsPosition, wsParallel, wsNormal, argT, tension, bias);

    // Calculate oriented basis. Keep orientation from calculated RMF but project it onto interpolated spline.
    wsParallel = normalize(wsParallel);
    // TODO - wsNormalST seems to cause singularity rotations?
#if 0
    wsNormal = normalize(wsNormalST - dot(wsNormalST, wsParallel) * wsParallel);
    wsTangent = normalize(cross(wsNormal, wsParallel));
#else
    wsTangent = normalize(wsTangentST - dot(wsTangentST, wsParallel) * wsParallel);
    wsNormal = normalize(cross(wsParallel, wsTangent));
#endif

    // Interpolate other branch properties:
    const float textureCoord = mix(distanceS, distanceT, argT);
    const float distanceFromRoot = mix(distanceS, distanceT, argT);

    // Pass data to geometry shader:
    ePosition = wsPosition;
    eNormal = wsNormal;
    eDistanceFromRoot = distanceFromRoot;
    eParallel = wsParallel;
    eBranchRadius = radius;
    eTextureUV = vec2(0.0f, textureCoord);
    eTangent = wsTangent;
    // TODO - Calculate real edge length after interpolation? Integrate interpolation from 0.0f to 1.0f.
    eEdgeLength = length(wsPosS - wsPosT);
    eVertexCount = gl_TessLevelOuter[1];
}
