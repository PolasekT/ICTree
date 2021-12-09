/**
 * @author Tomas Polasek, David Hrusa, Yichen Sheng
 * @date 4.15.2020
 * @version 1.0
 * @brief Fragment shader for the photo-mode rendering of trees.
 */

#version 430 core

/// Position at which the shadow should be sampled, w may be non 1.0f!
in vec4 gShadowCoord;
/// Position of the triangle vertex in world space and depth from camera in w.
in vec3 gPosition;
/// Position on the branch segment in world space.
in vec3 gBranchPosition;
/// Normal of the vertex in world space.
in vec3 gNormal;
/// Radius of the branch.
in float gBranchRadius;
/// Distance from root.
in float gDistanceFromRoot;

#if 0
// Not used:
/// Vector parallel with the branch in world space.
in vec3 gParallel;
/// Tangent of the branch in world space.
in vec3 gTangent;
/// UV texture coordinates on the surface of the branch tube.
in vec2 gTextureUV;
#endif

/// Output color for current fragment.
layout(location = 0) out vec4 fColor;

#include "recon_common.glsl"

#include "modality_common.glsl"
#include "random_common.glsl"
#include "shadow_common.glsl"

/// @brief Calculate shading for photo render.
vec4 photoShading(in float occlusion)
{
    vec3 normal = normalize(gPosition.xyz - gBranchPosition);
    vec3 toLight = normalize(uLightPos - gPosition.xyz);
    vec3 toCamera = normalize(uCameraPos - gPosition.xyz);
    float lightNormalDot = max(0.0f, dot(toLight, normal));
    float viewNormalDot = max(0.0f, abs(dot(toCamera, normal)));
    float radiusPtg = gBranchRadius / uMaxBranchRadius;
    float distancePtg = gDistanceFromRoot / uMaxDistanceFromRoot;

    float radiusComponent = smoothstep(-0.5f, 1.5f, radiusPtg) * 0.2f;
    float distanceComponent = smoothstep(-0.4f, 1.5f, 1.0f - distancePtg) * 0.3f;
    float lightComponent = smoothstep(-0.5f, 1.5f, lightNormalDot) * 0.1f;
    float cameraComponent = smoothstep(-0.3f, 1.7f, viewNormalDot);

    float combinedPtg = smoothstep(
        -0.6f, 1.1f,
        (radiusComponent + distanceComponent + lightComponent)
    );

    vec3 finalColor = mix(
        uBackgroundColor,
        uForegroundColor,
        (1.0f - combinedPtg) * cameraComponent
    );

    return vec4(finalColor * 0.8f, uOpaqueness);

    /*
    float radiusComponent = smoothstep(-0.5f, 1.5f, radiusPtg) * 0.1f;
    float distanceComponent = smoothstep(-0.4f, 1.5f, 1.0f - distancePtg) * 0.8f;
    float lightComponent = smoothstep(-0.5f, 1.5f, lightNormalDot) * 0.2f;
    float cameraComponent = smoothstep(-0.3f, 1.7f, viewNormalDot);

    float combinedPtg = smoothstep(
        -0.3f, 1.0f,
        (radiusComponent + distanceComponent + lightComponent)
    );

    vec3 finalColor = mix(
        uBackgroundColor,
        uForegroundColor,
        (1.0f - combinedPtg) * cameraComponent
    );

    return vec4(finalColor, uOpaqueness);
    */
}

/// @brief Calculate shading for edit render.
vec4 editShading(in float occlusion)
{
    vec3 normal = normalize(gPosition.xyz - gBranchPosition);
    vec3 toLight = normalize(uLightPos - gPosition.xyz);
    vec3 toCamera = normalize(uCameraPos - gPosition.xyz);
    float lightNormalDot = max(0.0f, dot(toLight, normal));
    float viewNormalDot = max(0.0f, abs(dot(toCamera, normal)));
    float distancePtg = gDistanceFromRoot / uMaxDistanceFromRoot;

    float distanceComponent = smoothstep(-0.4f, 1.5f, 1.0f - distancePtg) * 1.0f;
    float lightComponent = smoothstep(-0.5f, 1.5f, lightNormalDot) * 0.2f;
    float cameraComponent = smoothstep(-0.3f, 1.7f, viewNormalDot);

    vec3 finalColor = mix(
        uBackgroundColor,
        uForegroundColor,
        cameraComponent
    );

    return vec4(finalColor, uOpaqueness);
}

void main()
{
    // Calculate percentage of occlusion - 1.0f not occluded, 0.0f fully occluded:
    const float occlusion = ((uApplyShadow > 0) ?
        1.0f - calcShadow(uShadowMap, uShadowKernelSpec, gShadowCoord, gPosition.xyz) :
        1.0f
    );

    // Calculate output color:
    fColor = (uPhotoShading > 0) ? photoShading(occlusion) : editShading(occlusion);

    // Perform modality formatting.
    const vec2 depth = (uVP * vec4(gPosition.xyz, 1.0f)).zw;
    fColor = formatModality(fColor, vec4(0.54f, 0.2f, 0.14f, 1.0f), vec3(1.0f, 1.0f, 1.0f),
        occlusion, gNormal, depth);
}
