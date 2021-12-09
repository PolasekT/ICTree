/**
 * @author Tomas Polasek
 * @date 1.13.2020
 * @version 1.0
 * @brief Camera state.
 */

#include "TreeCamera.h"

#include "TreeGLUtils.h"

namespace treescene
{

int CameraState::cameraProjectionToOrd(CameraProjection proj)
{
    switch (proj)
    {
        default:
        case CameraProjection::Perspective: { return 0; }
        case CameraProjection::Orthographic: { return 1; }
    }
}

CameraProjection CameraState::ordToCameraProjection(int proj)
{
    switch (proj)
    {
        default:
        case 0: { return CameraProjection::Perspective; }
        case 1: { return CameraProjection::Orthographic; }
    }
}

int CameraState::cameraModeToOrd(CameraMode mode)
{
    switch (mode)
    {
        default:
        case CameraMode::Orbital: { return 0; }
        case CameraMode::XY: { return 1; }
        case CameraMode::XZ: { return 2; }
        case CameraMode::YZ: { return 3; }
        case CameraMode::Scripted: { return 4; }
    }
}

CameraMode CameraState::ordToCameraMode(int mode)
{
    switch (mode)
    {
        default:
        case 0: { return CameraMode::Orbital; }
        case 1: { return CameraMode::XY; }
        case 2: { return CameraMode::XZ; }
        case 3: { return CameraMode::YZ; }
        case 4: { return CameraMode::Scripted; }
    }
}

int CameraState::cameraInputSchemeToOrd(CameraInputScheme scheme)
{
    switch (scheme)
    {
        default:
        case CameraInputScheme::Orbital: { return 0; }
        case CameraInputScheme::Cartographic: { return 1; }
        case CameraInputScheme::FPS: { return 2; }
    }
}

CameraInputScheme CameraState::ordToCameraInputScheme(int scheme)
{
    switch (scheme)
    {
        default:
        case 0: { return CameraInputScheme::Orbital; }
        case 1: { return CameraInputScheme::Cartographic; }
        case 2: { return CameraInputScheme::FPS; }
    }
}

void CameraState::resetCameraDefaults()
{
    cameraPos = DEFAULT_CAMERA_POS;
    bckCameraPos = DEFAULT_CAMERA_POS;
    cameraTargetPos = DEFAULT_TARGET_POS;
    cameraPhiAngle = DEFAULT_CAMERA_PHI_ANGLE;
    cameraThetaAngle = DEFAULT_CAMERA_THETA_ANGLE;
    cameraDistance = DEFAULT_CAMERA_DISTANCE;
    cameraPlaneDistance = DEFAULT_CAMERA_DISTANCE;
    cameraRoll = DEFAULT_CAMERA_ROLL;
    cameraYaw = DEFAULT_CAMERA_YAW;
    cameraPitch = DEFAULT_CAMERA_PITCH;

    cameraPos = calculateCameraPos();
}

std::string CameraState::exportToString() const
{
    std::stringstream ss{ "" };
    ss << "cameraPos," << cameraPos.x << "," << cameraPos.y << "," << cameraPos.z << "\n";
    ss << "bckCameraPos," << bckCameraPos.x << "," << bckCameraPos.y << "," << bckCameraPos.z << "\n";
    ss << "cameraTargetPos," << cameraTargetPos.x << "," << cameraTargetPos.y << "," << cameraTargetPos.z << "\n";
    ss << "cameraPhiAngle," << cameraPhiAngle << "\n";
    ss << "cameraThetaAngle," << cameraThetaAngle << "\n";
    ss << "cameraDistance," << cameraDistance << "\n";
    ss << "cameraPlaneDistance," << cameraPlaneDistance << "\n";
    ss << "cameraMode," << cameraModeToOrd(cameraMode) << "\n";
    ss << "cameraRoll," << cameraRoll << "\n";
    ss << "cameraYaw," << cameraYaw << "\n";
    ss << "cameraPitch," << cameraPitch << "\n";
    ss << "displayUi," << (displayUi ? "true" : "false") << "\n";
    return ss.str();
}

void CameraState::importFromString(const std::string &content)
{
    std::stringstream in{ content };
    std::string line;
    while (std::getline(in,line))
    {
        if (line.length() <= 0)
        { continue; }
        std::vector<std::string> fields;
        strsplit(line.c_str(),fields,',');
        std::string &key = fields[0];

        if (key.compare("cameraPos") == 0)
        {
            cameraPos.x = std::stof(fields[1]);
            cameraPos.y = std::stof(fields[2]);
            cameraPos.z = std::stof(fields[3]);
        }
        else if (key.compare("bckCameraPos") == 0)
        {
            bckCameraPos.x = std::stof(fields[1]);
            bckCameraPos.y = std::stof(fields[2]);
            bckCameraPos.z = std::stof(fields[3]);
        }
        else  if(key.compare("cameraTargetPos") == 0)
        {
            cameraTargetPos.x = std::stof(fields[1]);
            cameraTargetPos.y = std::stof(fields[2]);
            cameraTargetPos.z = std::stof(fields[3]);
        }
        else if (key.compare("cameraPhiAngle") == 0)
        { cameraPhiAngle = std::stof(fields[1]); }
        else if (key.compare("cameraThetaAngle") == 0)
        { cameraThetaAngle = std::stof(fields[1]); }
        else if (key.compare("cameraDistance") == 0)
        { cameraDistance = std::stof(fields[1]); }
        else if (key.compare("cameraPlaneDistance") == 0)
        { cameraPlaneDistance = std::stof(fields[1]); }
        else if (key.compare("cameraMode") == 0)
        { cameraMode = ordToCameraMode(std::stoi(fields[1])); }
        else if (key.compare("cameraRoll") == 0)
        { cameraRoll = std::stof(fields[1]); }
        else if (key.compare("cameraYaw") == 0)
        { cameraYaw = std::stof(fields[1]); }
        else if (key.compare("cameraPitch") == 0)
        { cameraPitch = std::stof(fields[1]); }
        else if (key.compare("displayUi") == 0)
        { displayUi = fields[1] == "true"; }
    }
}

} // namespace treescene
