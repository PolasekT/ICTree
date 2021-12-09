/**
 * @author Tomas Polasek
 * @date 7.14.2020
 * @version 1.0
 * @brief Common parameters used for modality saving.
 */

/// Selector switch used for modality selection.
uniform int uModalitySelector = 0;
/// Information about the camera - near plane, far plane and fov. Last element is unused.
uniform vec4 uCamera = vec4(0.0001f, 100.0f, 45.0f, 0.0f);

/// @brief Display completely shaded output.
#define MODALITY_SHADED 0
/// @brief Display albedo only.
#define MODALITY_ALBEDO 1
/// @brief Display light only.
#define MODALITY_LIGHT 2
/// @brief Display shadows.
#define MODALITY_SHADOW 3
/// @brief Display normals.
#define MODALITY_NORMAL 4
/// @brief Display depths.
#define MODALITY_DEPTH 5

/**
 * @brief Perform modality output formatting using input values and return the result.
 * @param shadedInput Fully shaded input.
 * @param albedoInput Albedo only.
 * @param lightInput Light only.
 * @param shadowInput Shadow only - 1.0f for fully unoccluded.
 * @param normalInput World-space normals.
 * @param depthInput Depth from the camera - z and w elements.
 */
vec4 formatModality(in vec4 shadedInput, in vec4 albedoInput, in vec3 lightInput,
    in float shadowInput, in vec3 normalInput, in vec2 depthInput)
{
    switch (uModalitySelector)
    {
        default:
        case 0:
        { // Standard shaded output.
            return shadedInput;
        }
        case 1:
        { // Albedo only output.
            return albedoInput;
        }
        case 2:
        { // Light only output.
            return vec4(lightInput, 1.0f);
        }
        case 3:
        { // Shadow only output.
            bool inShadow = shadowInput < 0.7f && shadowInput > 0.1f;
            return vec4(inShadow, inShadow, inShadow, 1.0f);
        }
        case 4:
        { // Normal output.
            return vec4((normalize(normalInput) * 0.5f) + 0.5f, 1.0f);
        }
        case 5:
        { // Depth output.
            const float linearDepth = (2.0f * depthInput.x - uCamera.x - uCamera.y) / (uCamera.y - uCamera.x);
            const float clipDepth = linearDepth / depthInput.y;
            const float viewDepth = (clipDepth * 0.5f) + 0.5f;
            return vec4(viewDepth, viewDepth, viewDepth, 1.0f);
        }
    }
}
