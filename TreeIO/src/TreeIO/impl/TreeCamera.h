/**
 * @author Tomas Polasek
 * @date 1.13.2020
 * @version 1.0
 * @brief Camera state.
 */

#include <cinttypes>
#include <cstring>
#include <sstream>
#include <list>

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "TreeUtils.h"

#ifndef TREE_CAMERA_H
#define TREE_CAMERA_H

namespace treescene
{

/// @brief Specification of camera projection type.
enum class CameraProjection
{
    /// Camera uses perspective projection.
    Perspective,
    /// Camera uses orthographic projection.
    Orthographic
}; // enum class CameraProjection

/// @brief Camera mode specification.
enum class CameraMode
{
    /// Orbital camera allowing free movement.
    Orbital,
    /// Free roam locked to XY plane.
    XY,
    /// Free roam locked to XZ plane.
    XZ,
    /// Free roam locked to YZ plane.
    YZ,
    /// Controlled by the photobooth.
    Scripted
}; // enum class CameraMode

/// @brief Control schemes for the main viewport camera.
enum class CameraInputScheme
{
    /// Focuses on rotating around a given target.
    Orbital,
    /// Similar to orbital, but separates movement along XZ plane and Y axis. Should make navigation easier.
    Cartographic,
    /// Spectator style free camera with no target.
    FPS
}; //enum class CameraInputScheme

/// @brief Current state of camera and monitor.
struct CameraState
{
    /// Horizontal field of view of the camera in degrees.
    static constexpr float DEFAULT_CAMERA_FOV{ 45.0f };
    /// Default zoom used to modify the camera FOV.
    static constexpr float DEFAULT_CAMERA_ZOOM{ 1.0f };
    /// Distance of the near plane from the perspective camera.
    static constexpr float PERSPECTIVE_CAMERA_NEAR_PLANE{ 0.0001f };
    /// Distance of the far plane from the perspective camera.
    static constexpr float PERSPECTIVE_CAMERA_FAR_PLANE{ 100.0f };
    /// Distance of the near plane from the orthographic camera.
    static constexpr float ORTHOGRAPHIC_CAMERA_NEAR_PLANE{ 0.0001f };
    /// Distance of the far plane from the orthographic camera.
    static constexpr float ORTHOGRAPHIC_CAMERA_FAR_PLANE{ 100.0f };

    /// Default camera position.
    static constexpr glm::vec3 DEFAULT_CAMERA_POS{ 0.0f, 1.0f, 0.0f };
    /// Default camera target position.
    static constexpr glm::vec3 DEFAULT_TARGET_POS{ 0.0f, 0.5f, 0.0f };
    /// Default phi angle for the spherical camera in degrees.
    static constexpr float DEFAULT_CAMERA_PHI_ANGLE{ std::numeric_limits<float>::epsilon() };
    /// Default theta angle for the spherical camera in degrees.
    static constexpr float DEFAULT_CAMERA_THETA_ANGLE{ 90.0f + std::numeric_limits<float>::epsilon() };
    /// Default camera distance.
    static constexpr float DEFAULT_CAMERA_DISTANCE{ 3.0f };
    /// Default camera roll.
    static constexpr float DEFAULT_CAMERA_ROLL{ 0.0f };
    /// Default camera pitch.
    static constexpr float DEFAULT_CAMERA_PITCH{ 0.0f };
    /// Default camera yaw.
    static constexpr float DEFAULT_CAMERA_YAW{ 0.0f };

    /// @brief Convert projection enum to ordinal number.
    static int cameraProjectionToOrd(CameraProjection proj);
    /// @brief Convert ordinal number to projection enum.
    static CameraProjection ordToCameraProjection(int proj);

    /// @brief Convert mode enum to ordinal number.
    static int cameraModeToOrd(CameraMode mode);
    /// @brief Convert ordinal number to mode enum.
    static CameraMode ordToCameraMode(int mode);

    /// @brief Convert scheme enum to ordinal number.
    static int cameraInputSchemeToOrd(CameraInputScheme scheme);
    /// @brief Convert ordinal number to scheme enum.
    static CameraInputScheme ordToCameraInputScheme(int scheme);

    /// @brief Transform input position in the model space of skeleton points to the world-space.
    inline glm::vec3 pointSpaceToWorldSpace(const glm::vec3 &psPos) const;

    /// @brief Convert screen-space position to world-space.
    inline glm::vec3 screenSpaceToWorldSpace(const glm::vec2 &ssPos, float distance = 0.1f) const;

    /// @brief Convert screen space coordinate vector to world space coordinates.
    inline glm::vec3 screenSpaceToWorldSpace(const glm::vec3 &ssPos) const;

    /// @brief Calculate ray origin for given screen space position.
    inline glm::vec3 worldSpaceRayOrigin(const glm::vec2 &ssPos) const;

    /// @brief Calculate ray direction for given screen space position.
    inline glm::vec3 worldSpaceRayDirection(const glm::vec2 &ssPos) const;

    /// @brief Convert world-space position to screen-space.
    inline glm::vec3 worldSpaceToScreenSpace(const glm::vec3 &wsPos) const;

    /// @brief Convert model-space to world-space.
    inline glm::vec3 modelSpaceToWorldSpace(const glm::vec3 &msPos) const;

    /// @brief Convert world-space to model-space.
    inline glm::vec3 worldSpaceToModelSpace(const glm::vec3 &msPos) const;

    /// @brief Get scale correction for given world-space coordinate.
    inline float distanceScaleCorrection(const glm::vec3 &wsPos, float unitPixels = 100.0f) const;

    /// @brief Calculate camera position based on current settings.
    inline glm::vec3 calculateCameraPos() const;

    /// @brief Calculate right direction from the current settings.
    inline glm::vec3 calculateRightDiretion() const;
    /// @brief Calculate forward direction from the current settings.
    inline glm::vec3 calculateForwardDiretion() const;
    /// @brief Calculate up direction from the current settings.
    inline glm::vec3 calculateUpDiretion() const;

    /// @brief classic glm look at, but also allows for yaw, pitch and roll to be adjusted afterwards.
    inline glm::mat4 lookAtYPR(const glm::vec3 &eye, const glm::vec3 &tgt,
        const glm::vec3 &up, const glm::vec3 &yawPitchRoll) const;

    /// @brief Calculate view matrix based on current settings.
    inline glm::mat4 calculateViewMatrix(const glm::vec3 &cameraPos) const;

    /// @brief Calculate view matrix based on current settings.
    inline glm::mat4 calculateViewMatrix() const;

    /// @brief Calculate viewport ratio of width to height.
    inline float calculateAspectRatio() const;

    /// @brief Calculate projection matrix based on current settings.
    inline glm::mat4 calculateProjectionMatrix() const;

    /// @brief Get near plane for the current camera.
    inline float nearPlane() const;

    /// @brief Get far plane for the current camera.
    inline float farPlane() const;

    /// @brief Reset camera settings to default values.
    void resetCameraDefaults();

    /// Model matrix of the skeleton points model.
    glm::mat4 model{ };
    /// Inverse model matrix of the skeleton points model.
    glm::mat4 modeli{ };
    /// View matrix of the primary camera.
    glm::mat4 view{ };
    /// Projection matrix of the primary camera.
    glm::mat4 projection{ };

    /// Model-View-Projection matrix, combination of three above matrices.
    glm::mat4 mvp{ };
    /// Inverse Model-View-Projection matrix, inverse of the above.
    glm::mat4 mvpi{ };

    /// Width of the main viewport in pixels.
    std::size_t viewportWidth{ 1024u };
    /// Height of the main viewport in pixels.
    std::size_t viewportHeight{ 768u };

    /// Horizontal field of view of the camera in degrees.
    float cameraFov{ DEFAULT_CAMERA_FOV };
    /// Modifier used for zooming the camera.
    float cameraZoom{ 1.0f };
    /// Distance of the near plane from the perspective camera.
    float cameraPerspectiveNearPlane{ PERSPECTIVE_CAMERA_NEAR_PLANE };
    /// Distance of the far plane from the perspective camera.
    float cameraPerspectiveFarPlane{ PERSPECTIVE_CAMERA_FAR_PLANE };
    /// Distance of the near plane from the orthographic camera.
    float cameraOrthographicNearPlane{ ORTHOGRAPHIC_CAMERA_NEAR_PLANE };
    /// Distance of the far plane from the orthographic camera.
    float cameraOrthographicFarPlane{ ORTHOGRAPHIC_CAMERA_FAR_PLANE };

    /// Position of the camera in world-space.
    glm::vec3 cameraPos{ DEFAULT_CAMERA_POS };
    /// Backup of camera position, used when performing view operations.
    glm::vec3 bckCameraPos{ DEFAULT_CAMERA_POS };
    /// Position of the point the camera is looking at.
    glm::vec3 cameraTargetPos{ DEFAULT_TARGET_POS };

    /// Spherical angle (around y axis) of the camera in degrees.
    float cameraPhiAngle{ DEFAULT_CAMERA_PHI_ANGLE };
    /// Spherical angle (around x axis) of the camera in degrees.
    float cameraThetaAngle{ DEFAULT_CAMERA_THETA_ANGLE };
    /// Distance of the camera from orbited point.
    float cameraDistance{ DEFAULT_CAMERA_DISTANCE };
    /// Roll of the camera in its current position (not around the target).
    float cameraRoll{ DEFAULT_CAMERA_ROLL };
    /// Pitch of the camera in its current position (not around the target).
    float cameraPitch{ DEFAULT_CAMERA_PITCH };
    /// Yaw of the camera in its current position (not around the target).
    float cameraYaw{ DEFAULT_CAMERA_YAW};
    /// Distance used for plane camera modes.
    float cameraPlaneDistance{ DEFAULT_CAMERA_DISTANCE };

    /// Is the camera in perspective or orthographic mode?
    CameraProjection cameraProjection{ CameraProjection::Perspective };

    /// Current camera style.
    CameraMode cameraMode{ CameraMode::Orbital };
    /// Current control scheme.
    CameraInputScheme cameraInputScheme{ CameraInputScheme::Orbital };

    /// Render user interface?
    bool displayUi{ false };

    /// @brief Serialize camera settings.
    std::string exportToString() const;

    /// @brief Deserialize camera settings from given string.
    void importFromString(const std::string &content);
}; // struct CameraState

} // namespace treescene

// Template implementation begin.

namespace treescene
{

inline glm::vec3 CameraState::pointSpaceToWorldSpace(const glm::vec3 &psPos) const
{ return glm::vec4(psPos.x, psPos.y, psPos.z, 1.0f) * model; }

inline glm::vec3 CameraState::screenSpaceToWorldSpace(const glm::vec2 &ssPos, float distance) const
{
    auto viewport{ glm::vec4(0.0f, 0.0f, viewportWidth, viewportHeight) };
    glm::vec2 ssPosCorrected{ ssPos.x, viewport.w - 1.0f - ssPos.y };

    return glm::unProject(
        glm::vec3(ssPosCorrected.x, ssPosCorrected.y, distance),
        view, projection, viewport
    );
}

inline glm::vec3 CameraState::screenSpaceToWorldSpace(const glm::vec3 &ssPos) const
{ return screenSpaceToWorldSpace({ ssPos.x, ssPos.y }, ssPos.z); }

inline glm::vec3 CameraState::worldSpaceRayOrigin(const glm::vec2 &ssPos) const
{
    if (cameraProjection == CameraProjection::Orthographic)
    { // Calculate parallel ray direction.
        const auto wsPos{ screenSpaceToWorldSpace(ssPos, 0.0001f) };

        return wsPos;
    }
    else
    { // Calculate ray from camera origin.
        return calculateCameraPos();
    }
}

inline glm::vec3 CameraState::worldSpaceRayDirection(const glm::vec2 &ssPos) const
{
    if (cameraProjection == CameraProjection::Orthographic)
    { // Calculate parallel ray direction.
        const auto wsPos{ screenSpaceToWorldSpace(ssPos, 0.1f) };
        const auto rayOrigin{ worldSpaceRayOrigin(ssPos) };

        return glm::normalize(wsPos - rayOrigin);
    }
    else
    { // Calculate ray from camera origin.
        const auto wsPos{ screenSpaceToWorldSpace(ssPos)};
        const auto rayOrigin{ worldSpaceRayOrigin(ssPos) };

        return glm::normalize(wsPos - rayOrigin);
    }
}

inline glm::vec3 CameraState::worldSpaceToScreenSpace(const glm::vec3 &wsPos) const
{
    auto viewport{ glm::vec4(0.0f, 0.0f, viewportWidth, viewportHeight) };

    const auto ssPos{ glm::project(
        wsPos, view, projection, viewport
    ) };
    glm::vec3 ssPosCorrected{ ssPos.x, viewport.w - 1.0f - ssPos.y, ssPos.z };

    return ssPosCorrected;
}

inline glm::vec3 CameraState::modelSpaceToWorldSpace(const glm::vec3 &msPos) const
{ return glm::vec3{ model * glm::vec4{ msPos, 1.0f } }; }

inline glm::vec3 CameraState::worldSpaceToModelSpace(const glm::vec3 &msPos) const
{ return glm::vec3{ modeli * glm::vec4{ msPos, 1.0f } }; }

inline float CameraState::distanceScaleCorrection(const glm::vec3 &wsPos, float unitPixels) const
{
    // Create unit length screen-space vector.
    const auto ssPos1{ worldSpaceToScreenSpace(wsPos) };
    const auto ssPos2{ ssPos1 + glm::vec3(1.0f * unitPixels, 0.0f, 0.0f) };

    // Project it back to world-space.
    const auto wsPos1{ wsPos };
    const auto wsPos2{ screenSpaceToWorldSpace(ssPos2) };

    // How long is unit vector in world-space?
    const auto scaleCorrection{ glm::distance(wsPos1, wsPos2) };

    return scaleCorrection;
};

inline glm::vec3 CameraState::calculateCameraPos() const
{
    if (cameraMode == CameraMode::Orbital || cameraMode == CameraMode::Scripted)
    { // Calculate orbital camera movement.
        return cameraTargetPos + glm::vec3{
            std::sin(glm::radians(cameraThetaAngle)) * std::cos(glm::radians(cameraPhiAngle)) * cameraDistance,
            std::cos(glm::radians(cameraThetaAngle)) * cameraDistance,
            std::sin(glm::radians(cameraThetaAngle)) * std::sin(glm::radians(cameraPhiAngle)) * cameraDistance
        };
    }
    else
    { // Free-move camera.
        return cameraPos;
    }
}

inline glm::vec3 CameraState::calculateRightDiretion() const
{
    switch (cameraMode)
    {
        default:
        case CameraMode::Scripted:
        case CameraMode::Orbital:
        {
            /*
             * v00 v01 v02
             * v10 v11 v12
             * v20 v21 v22
             *  r   u  -f
             */
            return glm::vec3{ view[0][0], view[1][0], view[2][0] };
        }
        case CameraMode::XY:
        case CameraMode::YZ:
        case CameraMode::XZ:
        { return glm::cross(calculateForwardDiretion(), calculateUpDiretion()); }
    }
}

inline glm::vec3 CameraState::calculateForwardDiretion() const
{
    switch (cameraMode)
    {
        default:
        case CameraMode::Scripted:
        case CameraMode::Orbital:
        {
            /*
             * v00 v01 v02
             * v10 v11 v12
             * v20 v21 v22
             *  r   u  -f
             */
            return glm::vec3{ -view[0][2], -view[1][2], -view[2][2] };
        }
        case CameraMode::XY:
        { return glm::vec3{ 0.0f, 0.0f, 1.0f }; }
        case CameraMode::YZ:
        { return glm::vec3{ -1.0f, 0.0f, 0.0f }; }
        case CameraMode::XZ:
        { return glm::vec3{ 1.0f, -1.0f, 0.0f }; }
    }
}

inline glm::vec3 CameraState::calculateUpDiretion() const
{
    switch (cameraMode)
    {
        default:
        case CameraMode::Orbital:
        case CameraMode::Scripted:
        case CameraMode::XY:
        case CameraMode::YZ:
        { return glm::vec3{ 0.0f, 1.0f, 0.0f }; }
        case CameraMode::XZ:
        { return glm::vec3{ 1.0f, 0.0f, 0.0f }; }
    }
}

inline glm::mat4 CameraState::lookAtYPR(const glm::vec3 &eye, const glm::vec3 &tgt,
    const glm::vec3 &up, const glm::vec3 &yawPitchRoll) const
{
    auto z{ eye - tgt };
    auto nz{ glm::normalize(z) };
    auto y{ up };
    auto x{ glm::cross(y,nz) };
    y = glm::cross(nz,x);
    //
    auto modMatrix =
            //glm::mat4(1.0);
            glm::rotate(glm::mat4(1), glm::radians(cameraRoll), nz) *
            glm::rotate(glm::mat4(1), glm::radians(cameraPitch), x) *
            glm::rotate(glm::mat4(1), glm::radians(cameraYaw), y);
    // apply the rotations to axes:
    x = glm::vec3(modMatrix * glm::vec4(glm::normalize(x),0));
    y = glm::vec3(modMatrix * glm::vec4(glm::normalize(y),0));
    z = glm::vec3(modMatrix * glm::vec4(nz,0));
    // just normalize in case of rounding errors.
    x = glm::normalize(x);
    y = glm::normalize(y);
    z = glm::normalize(z);
    return {
            x.x,y.x,z.x,0,
            x.y,y.y,z.y,0,
            x.z,y.z,z.z,0,
            glm::dot(-x, eye),glm::dot(-y, eye),glm::dot(-z, eye),1
    };
}

inline glm::mat4 CameraState::calculateViewMatrix(const glm::vec3 &cameraPos) const
{
    auto targetPos{ cameraTargetPos };

    return lookAtYPR(
            cameraPos,
            targetPos,
            calculateUpDiretion(),
            {cameraYaw,cameraPitch,cameraRoll}
            );

}

inline glm::mat4 CameraState::calculateViewMatrix() const
{ return calculateViewMatrix(calculateCameraPos()); }

inline float CameraState::calculateAspectRatio() const
{ return static_cast<float>(viewportWidth) / viewportHeight; }

inline glm::mat4 CameraState::calculateProjectionMatrix() const
{
    switch (cameraProjection)
    {
        default:
        case CameraProjection::Perspective:
        {
            return glm::perspective(
                cameraFov * cameraZoom, calculateAspectRatio(),
                cameraPerspectiveNearPlane, cameraPerspectiveFarPlane
            );
        }
        case CameraProjection::Orthographic:
        {
            const auto horizontalSide{ glm::tan(
                glm::radians(cameraFov * cameraZoom)) *
                                       (cameraDistance + cameraOrthographicNearPlane)
            };
            const auto verticalSide{ horizontalSide / calculateAspectRatio()};
            return glm::ortho(
                -horizontalSide, horizontalSide,
                -verticalSide, verticalSide,
                cameraOrthographicNearPlane, cameraOrthographicFarPlane
            );
        }
    }
}

inline float CameraState::nearPlane() const
{
    switch (cameraProjection)
    {
        default:
        case CameraProjection::Perspective:
        { return cameraPerspectiveNearPlane; }
        case CameraProjection::Orthographic:
        { return cameraOrthographicNearPlane; }
    }
}

inline float CameraState::farPlane() const
{
    switch (cameraProjection)
    {
        default:
        case CameraProjection::Perspective:
        { return cameraPerspectiveFarPlane; }
        case CameraProjection::Orthographic:
        { return cameraOrthographicFarPlane; }
    }
}

} // namespace treescene

// Template implementation end.

#endif // TREE_CAMERA_H