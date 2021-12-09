/**
 * @author Tomas Polasek, David Hrusa
 * @date 4.15.2020
 * @version 1.0
 * @brief Concrete renderer systems used for rendering in photo modes.
 */

#include "TreeRenderSystemPhoto.h"

#include "TreeRenderer.h"
#include "TreeBuffer.h"
#include "TreeTextureBuffer.h"
#include "TreeFrameBuffer.h"
#include "TreeShaderCompatRecon.h"
#include "TreeShaderCompatMesh.h"
#include "TreeShaderCompatScreen.h"
#include "TreeShaderCompatBlur.h"
#include "TreeScene.h"

// Internal declarations begin.

namespace treerndr
{

namespace impl
{

/// @brief Internal implementation of the RenderSystemPhoto.
struct RenderSystemPhotoImpl
{
    /// @brief Initialize the internal data structures.
    void initialize();

    /// @brief Reload all of the shaders.
    void reloadShaders();

    /// @brief Render current scene using given input values.
    void render(treescene::CameraState &camera, treescene::TreeScene &scene,
        treerndr::TreeRenderer &renderer);

    /// Shader compatibility structure for main photo rendering shader.
    ShaderCompatRecon::Ptr shaderCompatPhoto{ };

    /// @brief Container for rendering state.
    struct RenderState
    {
        /// Shader used for tree reconstruction rendering.
        treeutil::WrapperPtrT<ShaderCompatRecon> reconShader{ };
        /// Shader used for 3D mesh rendering.
        treeutil::WrapperPtrT<ShaderCompatMesh> meshShader{ };
        /// Shader used for 2D and UI rendering.
        treeutil::WrapperPtrT<ShaderCompatScreen> screenShader{ };
        /// Shader used for blurring textures.
        treeutil::WrapperPtrT<ShaderCompatBlur> blurShader{ };

        /// Shadow map used as primary output and input for blur.
        TextureBufferPtr shadowMapOut{ };
        /// Shadow map used as input/output for blur.
        TextureBufferPtr shadowMapBlur{ };
        /// Frame-buffer used as output for shadow generation.
        FrameBufferPtr shadowFB{ };

        /// Frame-buffer used as output for standard rendering.
        FrameBufferPtr defaultOutputFB{ };

        /// Frame-buffer used as output for standard rendering.
        FrameBufferPtr outputFB{ };
    }; // struct RenderState

    /// Clear color used when everything is working correctly.
    static constexpr treeutil::Color CLEAR_COLOR_DEFAULT{ 1.0f, 1.0f, 1.0f, 0.0f };
    /// Clear color signifying that error occurred in the main shader.
    static constexpr treeutil::Color CLEAR_COLOR_ERROR{ 1.0f, 0.0f, 1.0f, 0.0f };
    /// Clear color signifying that error occurred in the reconstruction shader.
    static constexpr treeutil::Color CLEAR_COLOR_ERROR_RECON{ 0.0f, 1.0f, 1.0f, 0.0f };
    /// Clear color used for depth buffer color output.
    static constexpr treeutil::Color CLEAR_COLOR_DEPTH{ 0.0f, 0.0f, 0.0f, 0.0f };

    /// @brief Prepare given context and rendering state for rendering of shadow pass.
    void prepareShadowCtx(treescene::CameraState &camera, treescene::TreeScene &scene,
        treerndr::TreeRenderer &renderer, RenderState &state, RenderContext &ctx) const;

    /// @brief Prepare given context and rendering state for standard rendering.
    void prepareDrawCtx(treescene::CameraState &camera, treescene::TreeScene &scene,
        treerndr::TreeRenderer &renderer, RenderState &state, RenderContext &ctx) const;

    /// @brief Render given list of instances using current state and rendering context.
    void renderInstances(treescene::CameraState &camera, treescene::TreeScene &scene,
        treerndr::TreeRenderer &renderer, const RenderState &state,
        const RenderContext &ctx, const InstanceList &instances) const;

    /// @brief Blur shadow textures in given rendering state.
    void blurShadows(treescene::CameraState &camera, treescene::TreeScene &scene,
        treerndr::TreeRenderer &renderer, const RenderState &state,
        const RenderContext &ctx) const;

    /// Current rendering context.
    RenderContext mCtx{ };
    /// Container for rendering state and resources.
    RenderState mState{ };
    /// Currently used visualization parameters.
    RenderSystemPhoto::VisualizationParameters mParameters{ };
}; // struct RenderSystemPhotoImpl

} // namespace impl

} // namespace treerndr

// Internal declarations end.

namespace treerndr
{

namespace impl
{

void RenderSystemPhotoImpl::initialize()
{ reloadShaders(); }

void RenderSystemPhotoImpl::reloadShaders()
{
    if (!mState.reconShader)
    { mState.reconShader = ShaderCompatRecon::instantiate(); }
    else
    { mState.reconShader->reloadShaders(); }

    if (!mState.meshShader)
    { mState.meshShader = ShaderCompatMesh::instantiate(); }
    else
    { mState.meshShader->reloadShaders(); }

    if (!mState.screenShader)
    { mState.screenShader = ShaderCompatScreen::instantiate(); }
    else
    { mState.screenShader->reloadShaders(); }

    if (!mState.blurShader)
    { mState.blurShader = ShaderCompatBlur::instantiate(); }
    else
    { mState.blurShader->reloadShaders(); }
}

void RenderSystemPhotoImpl::render(treescene::CameraState &camera,
    treescene::TreeScene &scene, treerndr::TreeRenderer &renderer)
{
    const auto sortedInstances{ renderer.sortInstancesForRendering() };

    // Shadow pass:
    prepareShadowCtx(camera, scene, renderer, mState, mCtx);
    if (mParameters.useShadows)
    {
        renderInstances(camera, scene, renderer, mState, mCtx, sortedInstances);
        blurShadows(camera, scene, renderer, mState, mCtx);
    }

    // Full shading pass:
    prepareDrawCtx(camera, scene, renderer, mState, mCtx);
    renderInstances(camera, scene, renderer, mState, mCtx, sortedInstances);

    // Update current state of the scene:
    // The main model - skeleton points - is used as reference.
    camera.model = renderer.getInstance(scene.INST_POINTS_NAME)->model;
    camera.modeli = glm::inverse(camera.model);
    camera.view = mCtx.view;
    camera.projection = mCtx.projection;
    camera.mvp = camera.projection * camera.view * camera.model;
    camera.mvpi = glm::inverse(camera.mvp);
    camera.viewportWidth = mCtx.frameBufferWidth;
    camera.viewportHeight = mCtx.frameBufferHeight;
}

void RenderSystemPhotoImpl::prepareShadowCtx(treescene::CameraState &camera,
    treescene::TreeScene &scene, treerndr::TreeRenderer &renderer,
    RenderState &state, RenderContext &ctx) const
{
    const auto &lightInstance{ *scene.sceneLight() };
    const auto &lightSpec{ *lightInstance.attachedLight };

    // Check shadow frame-buffer and re-create if necessary:
    if (!state.shadowFB ||
        state.shadowFB->getDepthAttachment()->width() != lightSpec.shadowMapWidth ||
        state.shadowFB->getDepthAttachment()->height() != lightSpec.shadowMapHeight)
    { // Incompatible shadow-map textures -> Create new ones.
        state.shadowFB = FrameBuffer::createFrameBuffer(
            lightSpec.shadowMapWidth, lightSpec.shadowMapHeight,
            false, true, false
        );
        state.shadowMapOut = state.shadowFB->getDepthAttachment();
        state.shadowMapBlur = state.shadowMapOut->deepCopy();

        // Fix corners of the shadow map.
        state.shadowMapOut->wrapWidth(treeutil::TextureWrap::ClampEdge);
        state.shadowMapOut->wrapHeight(treeutil::TextureWrap::ClampEdge);
        state.shadowMapBlur->wrapWidth(treeutil::TextureWrap::ClampEdge);
        state.shadowMapBlur->wrapHeight(treeutil::TextureWrap::ClampEdge);
    }

    // Setup rendering context:
    ctx.frameBufferWidth = lightSpec.shadowMapWidth;
    ctx.frameBufferHeight = lightSpec.shadowMapHeight;
    ctx.cameraPosition = lightInstance.calculateLightPosition();
    ctx.lightPosition = ctx.cameraPosition;
    ctx.cameraNear = camera.nearPlane();
    ctx.cameraFar = camera.farPlane();
    ctx.cameraFov = camera.cameraFov;
    ctx.view = lightInstance.calculateLightViewMatrix();
    ctx.projection = lightInstance.calculateLightProjectionMatrix();
    ctx.viewProjection = lightInstance.calculateLightViewProjectionMatrix(false);
    ctx.lightViewProjection = ctx.viewProjection;
    ctx.modality = mParameters.modality;
    ctx.renderingShadows = true;
    ctx.renderingPhoto = false;

    ctx.inputFrameBuffer = { };
    ctx.outputFrameBuffer = state.shadowFB;
    ctx.inputShadowMap = { };
    ctx.outputShadowMap = state.shadowMapOut;

    state.shadowFB->clearDepth(1.0f);
}

void RenderSystemPhotoImpl::prepareDrawCtx(treescene::CameraState &camera,
    treescene::TreeScene &scene, treerndr::TreeRenderer &renderer,
    RenderState &state, RenderContext &ctx) const
{
    const auto &lightInstance{ *scene.sceneLight() };
    const auto &lightSpec{ *lightInstance.attachedLight }; TREE_UNUSED(lightSpec);
    camera.cameraPos = camera.calculateCameraPos();

    // Create the default frame-buffer.
    if (!state.defaultOutputFB)
    { state.defaultOutputFB = FrameBuffer::instantiate(); }

    // Use requested output FB or just the default one for the window.
    if (mParameters.outputFrameBufferOverride)
    { state.outputFB = mParameters.outputFrameBufferOverride; }
    else
    { state.outputFB = state.defaultOutputFB; }

    // Setup rendering context:
    ctx.frameBufferWidth = camera.viewportWidth;
    ctx.frameBufferHeight = camera.viewportHeight;
    ctx.cameraPosition = camera.cameraPos;
    ctx.lightPosition = lightInstance.calculateLightPosition();
    ctx.cameraNear = camera.nearPlane();
    ctx.cameraFar = camera.farPlane();
    ctx.cameraFov = camera.cameraFov;
    ctx.view = camera.calculateViewMatrix();
    ctx.projection = camera.calculateProjectionMatrix();
    ctx.viewProjection = ctx.projection * ctx.view;
    ctx.lightViewProjection = lightInstance.calculateLightViewProjectionMatrix(false);
    ctx.modality = mParameters.modality;
    ctx.renderingShadows = false;
    ctx.renderingPhoto = true;

    ctx.inputFrameBuffer = { };
    ctx.outputFrameBuffer = state.outputFB;
    ctx.inputShadowMap = state.shadowMapOut;
    ctx.outputShadowMap = { };

    state.outputFB->setViewport(0u, 0u, ctx.frameBufferWidth, ctx.frameBufferHeight);
    state.outputFB->clearColor(
        ctx.modality.modality() == DisplayModality::Shaded ?
        CLEAR_COLOR_DEFAULT :
        ctx.modality.getClearColor()
    );
    state.outputFB->clearDepth(1.0f);
}

void RenderSystemPhotoImpl::renderInstances(treescene::CameraState &camera,
    treescene::TreeScene &scene, treerndr::TreeRenderer &renderer,
    const RenderState &state, const RenderContext &ctx,
    const InstanceList &instances) const
{
    for (const auto &instancePtr : instances)
    { // Render each instantiated element of the scene.
        if (!instancePtr)
        { continue; }

        switch (instancePtr->mesh->shaderType)
        {
            case ShaderType::Screen:
            { /* Skip */ break; }
            case ShaderType::Mesh:
            { state.meshShader->render(instancePtr, ctx, camera); break; }
            case ShaderType::Reconstruction:
            { state.reconShader->render(instancePtr, scene.currentTreeReconstruction(), ctx); break; }
            default:
            {
                Error << "Unknown shader type used for instanced mesh ("
                      << static_cast<std::size_t>(instancePtr->mesh->shaderType)
                      << ")" << std::endl;
            }
        }
    }
}

void RenderSystemPhotoImpl::blurShadows(treescene::CameraState &camera,
    treescene::TreeScene &scene, treerndr::TreeRenderer &renderer,
    const RenderState &state, const RenderContext &ctx) const
{
    state.blurShader->blurTexture(state.shadowMapOut, state.shadowMapBlur,
        false, treeutil::FrameBufferAttachmentType::Depth);
    state.blurShader->blurTexture(state.shadowMapBlur, state.shadowMapOut,
        true, treeutil::FrameBufferAttachmentType::Depth);
}

} // namespace impl

RenderSystemPhoto::RenderSystemPhoto() :
    RenderSystem(RENDERER_NAME),
    mImpl{ std::make_shared<impl::RenderSystemPhotoImpl>() }
{ }
RenderSystemPhoto::~RenderSystemPhoto()
{ /* Automatic */ }

void RenderSystemPhoto::initialize()
{ mImpl->initialize(); }

void RenderSystemPhoto::render(treescene::CameraState &camera, treescene::TreeScene &scene)
{ mImpl->render(camera, scene, *scene.renderer()); }

void RenderSystemPhoto::reloadShaders()
{ mImpl->reloadShaders(); }

RenderSystemPhoto::VisualizationParameters &RenderSystemPhoto::parameters()
{ return mImpl->mParameters; }

#if 0
GLuint RenderSystemPhoto::shadowTexture1()
{ return mImpl->mState.shadowMapOut->id(); }
GLuint RenderSystemPhoto::shadowTexture2()
{ return mImpl->mState.shadowMapBlur->id(); }
#endif

} // namespace treerndr

