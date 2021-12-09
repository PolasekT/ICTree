/**
 * @author Tomas Polasek
 * @date 8.14.2021
 * @version 1.0
 * @brief Helper class for automated tree rendering.
 */

#include "TreeRenderHelper.h"

#include "TreeRandomEngine.h"
#include "TreeCamera.h"
#include "TreeRenderer.h"
#include "TreeRenderSystemPhoto.h"
#include "TreeRenderSystemRT.h"
#include "TreeScene.h"

namespace treerndr
{

namespace impl
{

/// @brief Subsection of view renderer state. Takes care of camera specific options.
struct RenderCameraState
{
    float camHeight{ 1.8f };
    float camDistance{ 30.0f };
    float camDistanceVar{ 0.0f };

    float camYaw{ 0.0f };
    float camYawVar{ 0.0f };
    float camYawDither{ 0.0f };
    float camYawDitherLow{ -1.0f };
    float camYawDitherHigh{ 1.0f };

    float camPitch{ 0.0f };
    float camPitchVar{ 0.0f };
    float camPitchDither{ 0.0f };
    float camPitchDitherLow{ -1.0f };
    float camPitchDitherHigh{ 1.0f };

    float camRoll{ 0.0f };
    float camRollVar{ 0.0f };
    float camRollDither{ 0.0f };
    float camRollDitherLow{ -1.0f };
    float camRollDitherHigh{ 1.0f };

    float camShake{ 1.0f };

    float camBaseAngle{ 0.0f };
    std::size_t viewCount{ 0 };
}; // struct RenderCameraState

/// @brief Subsection of renderer tree state. Takes care of tree specific options.
struct RenderTreeState
{
    float treeNodeDither{ 0.0f };
    float treeNodeDitherLow{ -1.0f };
    float treeNodeDitherHigh{ 1.0f };

    float treeBranchDither{ 0.0f };
    float treeBranchDitherLow{ -1.0f };
    float treeBranchDitherHigh{ 1.0f };
}; // struct RenderTreeState

/// @brief Storage for camera offset information.
struct CameraOffsets
{
    float distanceOffset{ 0.0f };
    float yawOffset{ 0.0f };
    float pitchOffset{ 0.0f };
    float rollOffset{ 0.0f };

    float phiTrueOffset{ 0.0f };
    float thetaTrueOffset{ 0.0f };
    float rollTrueOffset{ 0.0f };
}; // struct CameraOffsets

/// @brief Internal implementation of the RenderHelper class.
class RenderHelperImpl
{
public:
    /// @brief Initialize the render helper.
    RenderHelperImpl();
    /// @brief Free resources and destroy.
    ~RenderHelperImpl();

    /// @brief Setup an OpenGL context with given properties. Throws std::runtime_error on error.
    void initializeGlContext(std::size_t width, std::size_t height);

    /// @brief Destroy the currently initialized context and clean up. Null operation if no context is initialized.
    void destroyGlContext();

    /// @brief Initialize a fresh internal rendering state.
    void setupInternalState(const treerndr::RenderConfig &config);

    /// @brief Save meta-data about the current view into given output json file.
    void saveViewMetadata(treerndr::TreeRenderer &renderer,
        treescene::TreeScene &scene, treescene::CameraState &camera,
        const treerndr::RenderConfig &config,
        const std::string &viewPath, const std::string &viewTag, std::size_t viewIdx,
        const std::string &outputPath);

    /// @brief Render given tree and produce the requested views.
    void renderTree(const treeio::ArrayTree &tree,
        const treerndr::RenderConfig &config);

    /// @brief Render dithered views for given tree and produce the requested views.
    void renderDitheredTree(const treeio::ArrayTree &tree,
        const treerndr::RenderConfig &config,
        const treerndr::DitherConfig &dither);
private:
    /// @brief Information about an OpenGL context.
    struct ContextInfo
    {
#ifdef IO_USE_EGL
        /// Target display used by EGL.
        EGLDisplay display{ };
#else // !IO_USE_EGL
#endif // !IO_USE_EGL
    }; // struct ContextInfo

    /// @brief Wrapper around internal state for the rendering.
    struct InternalState
    {
        /// RNG used for repeatable data generation with variance.
        treeutil::RandomEngine randomEngine{ };
        /// Meta-data for the current tree state.
        RenderTreeState treeState{ };
        /// Meta-data for the current camera state.
        RenderCameraState cameraState{ };
        /// Camera offset information.
        CameraOffsets cameraOffsets{ };
        /// Backup of the main camera used for restores.
        treescene::CameraState backupCamera{ };
        /// Offset information for the backup camera.
        CameraOffsets backupCameraOffsets{ };
    }; // struct InternalState

    /// Information about the current OpenGL context.
    std::shared_ptr<ContextInfo> mCtx{ };

    /// Internal state for the current rendering job.
    std::shared_ptr<InternalState> mInternalState{ };
protected:
}; // class RenderHelperImpl

} // namespace impl

} // namespace treerndr

namespace treerndr
{

namespace impl
{

RenderHelperImpl::RenderHelperImpl()
{ setupInternalState({ }); }

RenderHelperImpl::~RenderHelperImpl()
{ /* Automatic */ }

void RenderHelperImpl::initializeGlContext(std::size_t width, std::size_t height)
{
    // Clean up any previous context.
    destroyGlContext();

    // Initialize context information structure.
    auto ctx{ std::make_shared<ContextInfo>() };

#ifdef IO_USE_EGL
    // EGL error handling helper.
    auto checkEglError{ [] (const std::string &msg) {
        const auto code{ eglGetError() };
        if (code != EGL_SUCCESS)
        { throw std::runtime_error(std::string("EGL [") + treeutil::formatIntHex(code) + "] : " + msg); }
    } };

    // Get a list of available EGL devices.
    static constexpr std::size_t MAX_DEVICES{ 32u };
    EGLDeviceEXT eglDevices[MAX_DEVICES]{ };
    EGLint numDevices{ };
    const auto eglQueryDevicesEXT{ reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(
        eglGetProcAddress("eglQueryDevicesEXT")
    ) };
    eglQueryDevicesEXT(MAX_DEVICES, eglDevices, &numDevices);

    // Go through the devices and search for the first compatible display.
    EGLDisplay eglDisplay{ EGL_NO_DISPLAY };
    EGLint eglDisplayCode{ EGL_SUCCESS };
    const auto eglGetPlatformDisplayEXT{ reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
        eglGetProcAddress("eglGetPlatformDisplayEXT")
    ) };
    for (std::size_t deviceIdx = 0u; deviceIdx < numDevices; ++deviceIdx)
    {
        eglDisplay = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, eglDevices[deviceIdx], nullptr);
        eglDisplayCode = eglGetError();
        if (eglDisplay != EGL_NO_DISPLAY && eglDisplayCode == EGL_SUCCESS)
        { break; }
    }
    if (eglDisplay == EGL_NO_DISPLAY || eglDisplayCode != EGL_SUCCESS)
    {
        // Try fallback to default EGL display.
        eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        checkEglError("Failed to eglGetPlatformDisplay and eglGetDisplay");
    }
    ctx->display = eglDisplay;

    // Initialize EGL.
    EGLint eglMinorVer{ }; EGLint eglMajorVer{ };
    eglInitialize(eglDisplay, &eglMajorVer, &eglMinorVer);
    checkEglError("Failed to eglInitialize");

    // Configure EGL.
    static const EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_BLUE_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_RED_SIZE, 8,
        EGL_DEPTH_SIZE, 8,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE,
    };
    EGLint numConfigs{ };
    EGLConfig eglConfig{ };
    eglChooseConfig(eglDisplay, configAttribs, &eglConfig, 1, &numConfigs);
    checkEglError("Failed to eglChooseConfig");

    // Create EGL surface
    static const EGLint surfaceAttribs[] = {
        EGL_WIDTH, static_cast<EGLint>(width),
        EGL_HEIGHT, static_cast<EGLint>(height),
        EGL_NONE,
    };
    const auto eglSurface{ eglCreatePbufferSurface(eglDisplay, eglConfig, surfaceAttribs) };
    checkEglError("Failed to eglCreatePbufferSurface");

    // Bind the EGL API.
    eglBindAPI(EGL_OPENGL_API);
    checkEglError("Failed to eglBindAPI");

    // Create and configure EGL context.
    static const EGLint contextAttribs[] = {
        EGL_CONTEXT_MAJOR_VERSION, 4,
        EGL_CONTEXT_MINOR_VERSION, 3,
        EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
        EGL_NONE,
    };
    const auto eglCtx{ eglCreateContext(eglDisplay, eglConfig, EGL_NO_CONTEXT, contextAttribs) };
    checkEglError("Failed to eglCreateContext");

    // Set the context as current.
    eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglCtx);
    checkEglError("Failed to eglMakeCurrent");
#else // !IO_USE_EGL
    // Initialize GLUT display and window.
    char name[]{ 'T', 'r', 'e', 'e', 'I', 'O', '\0' };
    int argc{ 1 }; char *argv[1]{ name };
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_MULTISAMPLE);

    // Initialize the window position and size.
    int sw{ glutGet(GLUT_SCREEN_WIDTH) - glutGet(GLUT_WINDOW_BORDER_WIDTH) };
    int sh{ glutGet(GLUT_SCREEN_HEIGHT) - glutGet(GLUT_WINDOW_BORDER_HEIGHT) };
    glutInitWindowPosition(50, 50);
    glutInitWindowSize(sw - 100, sh - 100);

    // Create the window.
    int win{ glutCreateWindow ("TreeIO") }; TREE_UNUSED(win);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_EXIT);
#endif // !IO_USE_EGL

    // Initialize OpenGL method wrapper, the GLEW library.
    const GLenum glewErr{ glewInit() };
    if (glewErr != GLEW_OK && glewErr != GLEW_ERROR_NO_GLX_DISPLAY)
    {
        throw std::runtime_error(
            std::string("GLEW [") + treeutil::formatIntHex(glewErr) +
            "] : Failed to glewInit"
        );
    }

    // Context is valid, use collected information.
    mCtx = ctx;
}

void RenderHelperImpl::destroyGlContext()
{
    // Only destroy if we have previously initialized a context.
    if (!mCtx)
    { return; }

#ifdef IO_USE_EGL
    // Destroy the EGL display, cleaning up the context and any associated data.
    eglTerminate(mCtx->display);
#else // !IO_USE_EGL
#endif // !IO_USE_EGL

    // Successfully destroyed the context, remove the information structure.
    mCtx = { };
}

void RenderHelperImpl::setupInternalState(const treerndr::RenderConfig &config)
{
    mInternalState = std::make_shared<InternalState>();

    mInternalState->cameraState.camHeight = config.cameraHeight;
    mInternalState->cameraState.camDistance = config.cameraDistance;
    mInternalState->cameraState.camYaw = 0.0f;
    mInternalState->cameraState.camPitch = 0.0f;
    mInternalState->cameraState.camRoll = 0.0f;
    mInternalState->cameraState.camBaseAngle = 0.0f;
    mInternalState->cameraState.viewCount = config.viewCount;
}

/// @brief Helper function for camera state serialization.
treeio::json generateCameraJson(const treescene::CameraState &camera)
{
    const auto viewPos{ camera.calculateViewMatrix() * glm::vec4{ 0.0f, 0.0f, 0.0f, 1.0f } };
    return treeio::json{
        { "pos", { camera.cameraPos.x, camera.cameraPos.y, camera.cameraPos.z } },
        { "view_pos", { viewPos.x, viewPos.y, viewPos.z } },
        { "target", { camera.cameraTargetPos.x, camera.cameraTargetPos.y, camera.cameraTargetPos.z } },
        { "distance", camera.cameraDistance },
        { "theta", camera.cameraThetaAngle },
        { "phi", camera.cameraPhiAngle },
        { "yaw", camera.cameraYaw },
        { "pitch", camera.cameraPitch },
        { "roll", camera.cameraRoll },
    };
}

/// @brief Helper function for camera offsets serialization.
treeio::json generateCameraOffsetsJson(const CameraOffsets &offsets)
{
    return treeio::json{
        { "distance", offsets.distanceOffset },
        { "yaw", offsets.yawOffset },
        { "pitch", offsets.pitchOffset },
        { "roll", offsets.rollOffset },
        { "phiTrue", offsets.phiTrueOffset },
        { "thetaTrue", offsets.thetaTrueOffset },
        { "rollTrue", offsets.rollTrueOffset },
    };
}

void RenderHelperImpl::saveViewMetadata(treerndr::TreeRenderer &renderer,
    treescene::TreeScene &scene, treescene::CameraState &camera,
    const treerndr::RenderConfig &config,
    const std::string &viewPath, const std::string &viewTag, std::size_t viewIdx,
    const std::string &outputPath)
{
    auto &currentTree{ scene.currentTree()->currentTree() };
    auto &meta{ currentTree.metaData() };
    const auto runtime{ meta.getRuntimeMetaData<treeio::RuntimeMetaData>() };
    const auto &runtimeProps{ runtime->runtimeTreeProperties };
    const auto recon{ renderer.getInstance(treescene::instances::RECONSTRUCTION_NAME) };
    const auto activeRenderer{ renderer.activeRenderer() };

    const treeio::json jsonData{
        {
            "tree",
            {
                { "dir", config.outputPath },
                { "path", runtime->loadPathPath },
                { "name", config.baseName },
                { "rotation", { recon->rotate.x, recon->rotate.y, recon->rotate.z } },
                { "translation", { recon->translate.x, recon->translate.y, recon->translate.z } },
                { "scale", { recon->scale, recon->scale, recon->scale } },
                { "scaleBase", runtimeProps.scaleBase },
                {
                    "graph",
                    {
                        { "scale", runtimeProps.scaleGraph },
                        { "offset", {
                            runtimeProps.offsetGraph[0],
                            runtimeProps.offsetGraph[1],
                            runtimeProps.offsetGraph[2]
                        } },
                        { "showPoints", runtimeProps.showPoints },
                        { "showSegments", runtimeProps.showSegments },
                    }
                },
                {
                    "reconstruction",
                    {
                        { "scale", runtimeProps.scaleReconstruction },
                        { "offset", {
                            runtimeProps.offsetReconstruction[0],
                            runtimeProps.offsetReconstruction[1],
                            runtimeProps.offsetReconstruction[2]
                        } },
                    }
                },
                {
                    "reference",
                    {
                        { "scale", runtimeProps.scaleReference },
                        { "offset", {
                            runtimeProps.offsetReference[0],
                            runtimeProps.offsetReference[1],
                            runtimeProps.offsetReference[2]
                        } },
                    }
                },
            },
        },
        { "camera", generateCameraJson(camera) },
        {
            "render",
            {
                { "name", activeRenderer->identifier() },
                { "modality", activeRenderer->parameters().modality.name() },
                { "clear", {
                    activeRenderer->parameters().modality.getClearColor().r,
                    activeRenderer->parameters().modality.getClearColor().g,
                    activeRenderer->parameters().modality.getClearColor().b,
                    activeRenderer->parameters().modality.getClearColor().a,
                } },
                { "resolution", { config.width, config.height } },
                { "sampling", config.samples },
            }
        },
        {
            "state",
            {
                { "seed", mInternalState->randomEngine.lastSeed() },
                { "distribution", mInternalState->randomEngine.getDistributionName() },
                { "output_path", viewPath },
                { "file_tag", viewTag },
                { "screen_id", viewIdx },
                { "counter", {
                    viewIdx, 0, 0, 0,
                } },
                {
                    "tree",
                    {
                        {
                            "node",
                            {
                                { "dither", mInternalState->treeState.treeNodeDither },
                                { "dither_low", mInternalState->treeState.treeNodeDitherLow },
                                { "dither_high", mInternalState->treeState.treeNodeDitherHigh },
                            }
                        },
                        {
                            "branch",
                            {
                                { "dither", mInternalState->treeState.treeBranchDither },
                                { "dither_low", mInternalState->treeState.treeBranchDitherLow },
                                { "dither_high", mInternalState->treeState.treeBranchDitherHigh },
                            }
                        },
                    }
                },
                {
                    "camera",
                    {
                        { "offsets", generateCameraOffsetsJson(mInternalState->cameraOffsets) },
                        { "backup", generateCameraJson(mInternalState->backupCamera) },
                        { "backup_offsets", generateCameraOffsetsJson(mInternalState->backupCameraOffsets) },
                        {
                            "height",
                            {
                                { "base", mInternalState->cameraState.camHeight },
                            }
                        },
                        {
                            "distance",
                            {
                                { "base", mInternalState->cameraState.camDistance },
                                { "var", mInternalState->cameraState.camDistanceVar },
                            }
                        },
                        {
                            "yaw",
                            {
                                { "base", mInternalState->cameraState.camYaw },
                                { "var", mInternalState->cameraState.camYawVar },
                                { "dither", mInternalState->cameraState.camYawDither },
                                { "dither_low", mInternalState->cameraState.camYawDitherLow },
                                { "dither_high", mInternalState->cameraState.camYawDitherHigh },
                            }
                        },
                        {
                            "pitch",
                            {
                                { "base", mInternalState->cameraState.camPitch },
                                { "var", mInternalState->cameraState.camPitchVar },
                                { "dither", mInternalState->cameraState.camPitchDither },
                                { "dither_low", mInternalState->cameraState.camPitchDitherLow },
                                { "dither_high", mInternalState->cameraState.camPitchDitherHigh },
                            }
                        },
                        {
                            "roll",
                            {
                                { "base", mInternalState->cameraState.camRoll },
                                { "var", mInternalState->cameraState.camRollVar },
                                { "dither", mInternalState->cameraState.camRollDither },
                                { "dither_low", mInternalState->cameraState.camRollDitherLow },
                                { "dither_high", mInternalState->cameraState.camRollDitherHigh },
                            }
                        },
                        { "shake", mInternalState->cameraState.camShake },
                        { "base_angle", mInternalState->cameraState.camBaseAngle },
                        { "view_count", mInternalState->cameraState.viewCount },
                    }
                }
            }
        }
    };

    std::ofstream file(outputPath);
    file << std::setw(4) << jsonData;
}

void RenderHelperImpl::renderTree(const treeio::ArrayTree &tree,
    const treerndr::RenderConfig &config)
{
    treerndr::DitherConfig dither{ };
    dither.ditherCount = 0u;

    renderDitheredTree(tree, config, dither);
}

void RenderHelperImpl::renderDitheredTree(const treeio::ArrayTree &tree,
    const treerndr::RenderConfig &config, const treerndr::DitherConfig &dither)
{
    // Setup fresh internal state.
    setupInternalState(config);

    // Initialize the OpenGL rendering context.
    initializeGlContext(config.width, config.height);

    // Prepare the output paths.
    std::filesystem::create_directories(std::filesystem::path(config.outputPath));

    // Setup renderer.
    auto renderer{ treerndr::TreeRenderer::instantiate() };
    renderer->registerRenderer<treerndr::RenderSystemPhoto>();
    renderer->registerRenderer<treerndr::RenderSystemRT>();
    renderer->initialize();

    // Use the "Photo" mode.
    renderer->useRenderer<treerndr::RenderSystemPhoto>();
    const auto &activeRenderer{ renderer->activeRenderer() };

    // Prepare a scene containing the target tree.
    auto scene{ treescene::TreeScene::instantiate() };
    scene->initialize(renderer);
    scene->reloadTree(tree, true, false);

    // Get handle to the tree instance so we can modify it for each view.
    auto sceneTree{ scene->currentTree() };
    auto treeInstance{ renderer->getInstance(treescene::instances::RECONSTRUCTION_NAME) };

    // Switch to photo mode, using the correct visual style.
    scene->setupPhotoMode();

    // Normalize the tree if requested.
    if (config.treeNormalize)
    {
        auto &metadata{ sceneTree->currentTree().metaData() };
        auto &runtime{ metadata.getRuntimeMetaData<treeio::RuntimeMetaData>()->runtimeTreeProperties };

        runtime.scaleBase = 1.0f;
        runtime.changedScaleBase();

        treeop::normalizeTreeScale(sceneTree->currentTree(),
            runtime.scaleGraph * runtime.scaleBase, true);

        runtime.scaleBase = config.treeScale;
        runtime.changedScaleBase();

        runtime.applyScales(treeInstance);
    }

    // Prepare the camera.
    auto camera{ std::make_shared<treescene::CameraState>() };
    camera->lookAtYPR(
        glm::vec3{ 0.0f, config.cameraHeight, config.cameraDistance },
        glm::vec3{ 0.0f, config.cameraHeight, 0.0f },
        glm::vec3{ 0.0f, 1.0f, 0.0f },
        glm::vec3{ 0.0f, 0.0f, 0.0f }
    );
    mInternalState->backupCamera = *camera;

    // Start rendering.
    for (std::size_t viewIdx = 0u; viewIdx < config.viewCount; ++viewIdx)
    { // Render the views.
        // Rotate the tree for the current view to keep the shadow constant.
        const auto rotation{ 2.0f * treeutil::PI<float> * viewIdx / config.viewCount };
        treeInstance->rotate = { 0.0f, rotation, 0.0f };

        // First: Render the normal view.
        for (std::size_t modalityIdx = 0u; modalityIdx < treerndr::RendererModality::MODALITY_COUNT; ++modalityIdx)
        { // Render modalities.
            // Set the modality.
            activeRenderer->parameters().modality.fromIdx(modalityIdx);
            // Generate save path for current view/modality.
            const auto savePath{
                config.outputPath + "/" + config.baseName +
                "_screen_" + std::to_string(viewIdx) +
                "_" + activeRenderer->parameters().modality.name()
            };
            const auto viewSavePath{ savePath + ".png" };
            const auto metaSavePath{ savePath + ".json" };
            // Render the modality and save.
            renderer->renderScreenshot(
                *camera, *scene, viewSavePath,
                config.width, config.height, config.samples,
                false, false
            );
            // Save metadata for the current view.
            saveViewMetadata(
                *renderer, *scene, *camera,
                config, viewSavePath,
                activeRenderer->parameters().modality.name(),
                viewIdx, metaSavePath
            );
        }

        // Second: Render the dithered variants.
        for (std::size_t variantIdx = 0u; variantIdx < dither.ditherCount; ++variantIdx)
        { // Render dithered views.
            // Reset to the base camera.
            *camera = mInternalState->backupCamera;
            // Dither the camera position.
            const auto distanceOffset{ mInternalState->randomEngine.randomFloat(
                -1.0, 1.0, dither.camDistanceVar)
            };
            const auto yawOffset{ mInternalState->randomEngine.randomFloat(
                dither.camYawDitherLow,
                dither.camYawDitherHigh,
                dither.camYawDither
            ) };
            const auto pitchOffset{ mInternalState->randomEngine.randomFloat(
                dither.camPitchDitherLow,
                dither.camPitchDitherHigh,
                dither.camPitchDither
            ) };
            const auto rollOffset{ mInternalState->randomEngine.randomFloat(
                dither.camRollDitherLow,
                dither.camRollDitherHigh,
                dither.camRollDither
            ) };

            mInternalState->cameraOffsets.distanceOffset = distanceOffset;
            mInternalState->cameraOffsets.yawOffset = yawOffset;
            mInternalState->cameraOffsets.pitchOffset = pitchOffset;
            mInternalState->cameraOffsets.rollOffset = rollOffset;

            const auto originalPhi{ camera->cameraPhiAngle };
            const auto originalTheta{ camera->cameraThetaAngle };
            const auto originalRoll{ camera->cameraRoll };

            camera->cameraDistance += distanceOffset;
            camera->cameraPhiAngle = treeutil::wrapValue(camera->cameraPhiAngle + yawOffset, 0.0f, 360.0f);
            camera->cameraThetaAngle = treeutil::wrapValue(camera->cameraThetaAngle + pitchOffset, 1.0f, 179.0f);
            camera->cameraRoll = treeutil::wrapValue(camera->cameraRoll + rollOffset, 0.0f, 360.0f);

            mInternalState->cameraOffsets.phiTrueOffset = camera->cameraPhiAngle - originalPhi;
            mInternalState->cameraOffsets.thetaTrueOffset = camera->cameraThetaAngle - originalTheta;
            mInternalState->cameraOffsets.rollTrueOffset = camera->cameraRoll - originalRoll;

            for (std::size_t modalityIdx = 0u; modalityIdx < treerndr::RendererModality::MODALITY_COUNT; ++modalityIdx)
            { // Render modalities.
                // Set the modality.
                activeRenderer->parameters().modality.fromIdx(modalityIdx);
                // Generate save path for current view/modality.
                const auto savePath{
                    dither.outputPath + "/" + config.baseName +
                    "_screen_" + std::to_string(viewIdx) +
                    "_variant_" + std::to_string(variantIdx) +
                    "_" + activeRenderer->parameters().modality.name()
                };
                const auto viewSavePath{ savePath + ".png" };
                const auto metaSavePath{ savePath + ".json" };
                // Render the modality and save.
                renderer->renderScreenshot(
                    *camera, *scene, viewSavePath,
                    config.width, config.height, config.samples,
                    false, false
                );
                // Save metadata for the current view.
                saveViewMetadata(
                    *renderer, *scene, *camera,
                    config, viewSavePath,
                    activeRenderer->parameters().modality.name(),
                    viewIdx, metaSavePath
                );
            }
        }
    }

    destroyGlContext();
}

} // namespace impl

RenderHelper::RenderHelper() :
    mImpl{ std::make_shared<impl::RenderHelperImpl>() }
{ }

RenderHelper::~RenderHelper()
{ /* Automatic */ }

void RenderHelper::renderTree(const treeio::ArrayTree &tree, const RenderConfig &config)
{ return mImpl->renderTree(tree, config); }

void RenderHelper::renderDitheredTree(const treeio::ArrayTree &tree,
    const RenderConfig &config, const DitherConfig &dither)
{ return mImpl->renderDitheredTree(tree, config, dither); }

} // namespace treerndr

