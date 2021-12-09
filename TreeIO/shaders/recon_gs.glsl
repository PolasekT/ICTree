/**
 * @author Tomas Polasek, David Hrusa, Yichen Sheng
 * @date 4.15.2020
 * @version 1.0
 * @brief Geometry shader for the photo-mode rendering of trees.
 */

#version 430 core
#extension GL_EXT_geometry_shader4 : enable

// Input = Tessellated lines representing the current segment.
layout(lines, invocations = 1) in;
// Output = Triangles of the branch tube. Limit is (MAX_TESSELLATION + 1u) * 2.
layout(triangle_strip, max_vertices = 50) out;
//layout(triangle_strip, max_vertices = 62) out;

/// Position of the vertex in world space.
in vec3 ePosition[];
/// Normal of the vertex in world space.
in vec3 eNormal[];
/// Distance of this node from tree root.
in float eDistanceFromRoot[];
/// Vector parallel with the branch in world space.
in vec3 eParallel[];
/// Texture coordinate of the point.
in vec2 eTextureUV[];
/// Radius of the branch.
in float eBranchRadius[];
/// Tangent of the branch in world space.
in vec3 eTangent[];
/// Total length of the generated branch without tessellation.
in float eEdgeLength[];
/// Total number of generated vertices in the generated branch, excluding the last one.
in float eVertexCount[];

/// Position at which the shadow should be sampled, w may be non 1.0f!
out vec4 gShadowCoord;
/// Position of the triangle vertex in world space.
out vec3 gPosition;
/// Position on the branch segment in world space.
out vec3 gBranchPosition;
/// Normal of the vertex in world space.
out vec3 gNormal;
/// Radius of the branch.
out float gBranchRadius;
/// Distance from root.
out float gDistanceFromRoot;

#if 0
// Not used:
/// Vector parallel with the branch in world space.
out vec3 gParallel;
/// Tangent of the branch in world space.
out vec3 gTangent;
/// UV texture coordinates on the surface of the branch tube.
out vec2 gTextureUV;
#endif

#include "recon_common.glsl"

/// Minimal horizontal tessellation of the branch:
const uint MIN_TESSELLATION = 5u;
/// Maximal horizontal tessellation of the branch:
const uint MAX_TESSELLATION = 24u;

void main()
{
    const float PI = 3.141592654f;
    const float PI_2 = 2.0f * 3.141592654f;

    for (uint iii = 0u; iii < gl_VerticesIn - 1; ++iii)
    { // Create circular cross-section for each generated line point, stopping before the last one.
        // Recover values for the end-points S and T:
        const vec3 wsPosS = ePosition[iii + 0u];
        const vec3 wsPosT = ePosition[iii + 1u];

        const vec3 wsNormalS = eNormal[iii + 0u];
        const vec3 wsNormalT = eNormal[iii + 1u];
        const vec3 wsParallelS = eParallel[iii + 0u];
        const vec3 wsParallelT = eParallel[iii + 1u];
        const vec3 wsTangentS = eTangent[iii + 0u];
        const vec3 wsTangentT = eTangent[iii + 1u];

        const float radiusS = eBranchRadius[iii + 0u];
        const float radiusT = eBranchRadius[iii + 1u];

        const vec2 textureUVS = eTextureUV[iii + 0u];
        const vec2 textureUVT = eTextureUV[iii + 1u];

        const float distanceS = eDistanceFromRoot[iii + 0u];
        const float distanceT = eDistanceFromRoot[iii + 1u];

        // Calculate branch primary and secondary eccentricity vectors:
        // TODO - Use cross?
        const vec3 evSPrimary = wsNormalS;
        const vec3 evSSecondary = wsTangentS;
        const vec3 evTPrimary = wsNormalT;
        const vec3 evTSecondary = wsTangentT;

        // Calculate real branch radii:
        const float evSRadius = max(MINIMUM_MRF_DISTANCE, radiusS);
        const float evTRadius = max(MINIMUM_MRF_DISTANCE, radiusT);

        // Currently disabled.
        const float camDistanceS = min(1.0f, max(1.0f, length(uCameraPos - wsPosS)));
        const float camDistanceT = min(1.0f, max(1.0f, length(uCameraPos - wsPosT)));
        // Estimate required tessellation in number of vertices per elipsoid cross-section.
        const uint tessellationS = clamp(
            // 37.0f is magical constant to keep quality on par with vertical tessellation.
            uint(max(0.0f, uTessellationMultiplier * 121.0f * evSRadius / camDistanceS)),
            MIN_TESSELLATION, MAX_TESSELLATION
        );
        const uint tessellationT = clamp(
            // 37.0f is magical constant to keep quality on par with vertical tessellation.
            uint(max(0.0f, uTessellationMultiplier * 121.0f * evTRadius / camDistanceT)),
            MIN_TESSELLATION, MAX_TESSELLATION
        );
        const uint maxTessellationLevel = max(tessellationS, tessellationT);

        for(uint jjj = 0u; jjj <= maxTessellationLevel; ++jjj)
        { // Generate the cross-section vertices.
            // Re-calculate indices for specific upper and lower cylinder parts.
            const float tessellationPtg = jjj / float(maxTessellationLevel);
            const uint tessellationSIdx = uint(round((1.0f - tessellationPtg) * tessellationS));
            const uint tessellationTIdx = uint(round((1.0f - tessellationPtg) * tessellationT));
            // Calculate the point on circumference of the elipsoid cross-section:
            const float angleS = (PI_2 / tessellationS) * tessellationSIdx;
            const float angleT = (PI_2 / tessellationT) * tessellationTIdx;

            const vec3 dirS = normalize(evSPrimary * sin(angleS) + evSSecondary * cos(angleS));
            const vec3 dirT = normalize(evTPrimary * sin(angleT) + evTSecondary * cos(angleT));

            const vec3 wsNewPosS = wsPosS + dirS * evSRadius;
            const vec3 wsNewPosT = wsPosT + dirT * evTRadius;

            // Calculate texture coordinates:
            const vec2 newTextureUVS = vec2(angleS / PI_2, textureUVS.y);
            const vec2 newTextureUVT = vec2(angleT / PI_2, textureUVT.y);

            // TODO - Use normal transform?
            //gNormal = normalize((uNT * vec4((wsPosS - wsNewPosS).xyz, 0.0f)).xyz);
            //gParallel = normalize((uNT * vec4((wsPosT - wsPosS).xyz, 0.0f)).xyz);

            { // Source vertex:
                gShadowCoord = uLightView * vec4(wsNewPosS, 1.0f);
                gPosition = wsNewPosS;
                gl_Position = uVP * vec4(gPosition.xyz, 1.0f);
                gBranchPosition = wsPosS;
                gNormal = normalize((wsNewPosS - wsPosS).xyz);
                gBranchRadius = evSRadius;
                gDistanceFromRoot = distanceS;
#if 0
                // Not used:
                gParallel = normalize((wsPosT - wsPosS).xyz);
                gTangent = normalize(cross(gNormal, gParallel));
                gTextureUV = newTextureUVS;
#endif
            } EmitVertex();

            { // Target vertex:
                gShadowCoord = uLightView * vec4(wsNewPosT, 1.0f);
                gPosition = wsNewPosT;
                gl_Position = uVP * vec4(gPosition.xyz, 1.0f);
                gBranchPosition = wsPosT;
                gNormal = normalize((wsNewPosT - wsPosT).xyz);
                gBranchRadius = evTRadius;
                gDistanceFromRoot = distanceT;
#if 0
                // Not used:
                gParallel = normalize((wsPosT - wsPosS).xyz);
                gTangent = normalize(cross(gNormal, gParallel));
                gTextureUV = newTextureUVT;
#endif
            } EmitVertex();
        }
    }

    // End the triangle strip.
    EndPrimitive();
}
