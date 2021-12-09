/**
 * @author Tomas Polasek, David Hrusa
 * @date 1.14.2020
 * @version 1.0
 * @brief Tree statistics wrapper.
 */

#include "TreeRenderSystemRT.h"

#include <thread>

#include <TreeIO/TreeIOAcceleration.h>

#include "TreeScene.h"
#include "TreeBuffer.h"
#include "TreeTextureBuffer.h"
#include "TreeFrameBuffer.h"
#include "TreeShaderCompatFullscreen.h"
#include "TreeShaderCompatRecon.h"

namespace treerndr
{

namespace impl
{

/// @brief Internal implementation of the RenderSystemRT.
struct RenderSystemRTImpl
{
    /// @brief Initialize the internal data structures.
    void initialize();

    /// @brief Reset and prepare a new ray-tracer.
    void reloadRayTracer();

    /// @brief Render current scene using given input values.
    void render(treescene::CameraState &camera, treescene::TreeScene &scene,
        treerndr::TreeRenderer &renderer);

    /// @brief Container for rendering state.
    struct RenderState
    {
        /// Pointer to the ray tracer used for scene rendering.
        treert::RayTracer::Ptr rayTracer{ };

        /// Shader used for rendering of ray tracer outputs.
        treeutil::WrapperPtrT<ShaderCompatFullscreen> fullscreenShader{ };

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

    /// @brief Prepare given context and rendering state for standard rendering.
    void prepareDrawCtx(treescene::CameraState &camera, treescene::TreeScene &scene,
        treerndr::TreeRenderer &renderer, RenderState &state, RenderContext &ctx) const;
    /// @brief Set all necessary members of given rendering context.
    static void setupDrawCtx(treescene::CameraState &camera, treescene::TreeScene &scene,
        treerndr::TreeRenderer &renderer, RenderContext &ctx);

    /// @brief Render given list of instances using current state and rendering context.
    void renderInstances(const treescene::CameraState &camera, treescene::TreeScene &scene,
        treerndr::TreeRenderer &renderer, const RenderState &state,
        const RenderContext &ctx, const InstanceList &instances) const;

    /// Current rendering context.
    RenderContext mCtx{ };
    /// Container for rendering state and resources.
    RenderState mState{ };
    /// Currently used visualization parameters.
    RenderSystemRT::VisualizationParameters mParameters{ };
}; // struct RenderSystemRTImpl

} // namespace impl

} // namespace treerndr

namespace treert
{

namespace impl
{

/// @brief Storage for all types of modality buffers.
struct ModalityStorage
{
    /// @brief Pointer to any modality buffer.
    using ModalityPtr = std::shared_ptr<RayTracer::OutputModalityBase>;

    /// @brief Initialize all modality types to default (empty) state.
    ModalityStorage();

    /// @brief Clean up and destroy.
    ~ModalityStorage();

    /// @brief Initialize all modality types to default (empty) state.
    void reset();

    /// @brief Initialize given modalities to requested size.
    void initializeModalities(const std::set<TracingModality> &types,
        std::size_t width, std::size_t height, std::size_t samples,
        bool resetOthers);

    /// @brief Modality storage of given type.
    RayTracer::OutputModalityBase &modality(TracingModality type, bool resolved);

    /// @brief Modality storage of given type.
    ModalityPtr &modalityPtr(TracingModality type, bool resolved);

    /// @brief Shadow modality storage.
    RayTracer::ShadowModality &shadowModality(bool resolved);
    /// @brief Normal modality storage.
    RayTracer::NormalModality &normalModality(bool resolved);
    /// @brief Depth modality storage.
    RayTracer::DepthModality &depthModality(bool resolved);
    /// @brief Triangle count modality storage.
    RayTracer::TriangleCountModality &triangleCountModality(bool resolved);
    /// @brief Volume modality storage.
    RayTracer::VolumeModality &volumeModality(bool resolved);

    /// @brief Generate resolve buffer for given modality.
    template <typename BT>
    RayTracer::OutputModality<BT> generateResolveBuffer(
        const RayTracer::OutputModality<BT> &modality);

    /// @brief Resolve given modality using mean value.
    template <typename BT, typename AT = double>
    void resolveModalityMean(
        const RayTracer::OutputModality<BT> &input,
        RayTracer::OutputModality<BT> &output);

    /// @brief Resolve given modality using median value. This is destructive to the original modality!
    template <typename BT>
    void resolveModalityMedian(
        RayTracer::OutputModality<BT> &input,
        RayTracer::OutputModality<BT> &output);

    /// @brief Get list of all currently valid modalities.
    std::vector<TracingModality> activeModalities();

    /// Storage for unresolved modalities.
    std::map<TracingModality, ModalityPtr> modalities{ };
    /// Storage for resolved modalities.
    std::map<TracingModality, ModalityPtr> resolvedModalities{ };
}; // struct ModalityStorage

/// @brief Triangle buffer generator and cache.
class TriangleCache
{
public:
    /// @brief Information for a single instance.
    struct InstanceRecord
    {
        /// List of triangles for this instance.
        std::vector<treeacc::Triangle> triangles{ };
        /// Pointer to the instance itself.
        RayTracerScene::InstancePtr instance{ };
        /// Version of tree we got the triangles for. Only valid for reconstructions.
        treeop::TreeVersion treeVersion{ };
    }; // struct InstanceRecord

    /// @brief Pointer to an instance record.
    using InstanceRecordPtr = std::shared_ptr<InstanceRecord>;

    /// @brief Initialize empty triangle cache.
    TriangleCache();
    /// @brief Clean up and destroy.
    ~TriangleCache();

    /// @brief Reset the triangle cache to default (empty) state.
    void reset();

    /// @brief Perform triangle caching for given instance.
    void cacheInstance(const RayTracerScene::InstancePtr &instance,
        const treerndr::RenderContext &ctx);

    /// @brief Remove given instance from this cache.
    void removeInstance(const std::string &instanceName);

    /// @brief Generate a triangle buffer containing triangle data from currently cached instances.
    const std::vector<treeacc::Triangle> &generateTriangleBuffer();

    /// @brief Recover triangles for given instance and export it to given path as OBJ file.
    void exportInstance(const std::string &path,
        const RayTracerScene::InstancePtr &instance,
        const treerndr::RenderContext &ctx);
private:
    /// @brief Recover vertex from given vertex buffer and index.
    Vector3D getVertex(const std::vector<char> &buffer, int index, const glm::mat4 &model);

    /// @brief Recover vertex from given vertex buffer and index.
    Vector3D getVertex(const std::vector<glm::vec4> &buffer, int index, const glm::mat4 &model);

    /// @brief Load GL_TRIANGLES mesh data.
    std::vector<treeacc::Triangle> loadTriangleMesh(
        const RayTracerScene::InstancePtr &instance,
        const treerndr::RenderContext &ctx);

    /// @brief Load GL_TRIANGLE_STRIP mesh data.
    std::vector<treeacc::Triangle> loadTriangleStripMesh(
        const RayTracerScene::InstancePtr &instance,
        const treerndr::RenderContext &ctx);

    /// @brief Load GL_TRIANGLES mesh data from RawGeometry buffer.
    std::vector<treeacc::Triangle> loadRawGeometry(
        const RayTracerScene::InstancePtr &instance,
        const treerndr::RawGeometryPtr &geometryPtr,
        const treerndr::RenderContext &ctx,
        bool skipModelTransform);

    /// @brief Cache simple mesh instance.
    bool cacheMesh(const RayTracerScene::InstancePtr &instance,
        const treerndr::RenderContext &ctx);

    /// @brief Cache tree reconstruction.
    bool cacheReconstruction(const RayTracerScene::InstancePtr &instance,
        const treerndr::RenderContext &ctx);

    /// Cache containing instances and their current triangle data.
    std::map<std::string, InstanceRecordPtr> mCache{ };
    /// Dirty flag for the triangle buffer generator.
    bool mDirty{ false };
    /// List of triangles from cached instances.
    std::vector<treeacc::Triangle> mTriangleBuffer{ };

    /// Shader used for tree reconstruction generation.
    treerndr::ShaderCompatRecon::Ptr mReconShader{ };
protected:
}; // class TriangleCache

/// @brief Internal implementation of the RayTracer.
class RayTracerImpl
{
public:
    /// @brief Pixel block used in ray tracing parallelization.
    struct PixelBlock
    {
        /// Starting x coordinate.
        std::size_t xBegin{ 0u };
        /// Ending x coordinate.
        std::size_t xEnd{ 0u };
        /// Starting y coordinate.
        std::size_t yBegin{ 0u };
        /// Ending y coordinate.
        std::size_t yEnd{ 0u };
    }; // struct PixelBlock

    /// Minimal size of a block side for a single pixel block.
    static constexpr std::size_t PARALLEL_MIN_BLOCK_SIZE{ 64u };
    /// Maximal size of a block side for a single pixel block.
    static constexpr std::size_t PARALLEL_MAX_BLOCK_SIZE{ 128u };
    /// Small delta used to prevent self occlusion.
    static constexpr auto RAY_DELTA{ 0.0001f };
    /// Maximal number of secondary rays per pixel.
    static constexpr std::size_t RAY_SECONDARY_LIMIT{ 100u };

    /// @brief Stage 1 : Update current configuration.
    void updateConfiguration(const treerndr::RenderContext &ctx);

    /// @brief Stage 2 : Build the acceleration structures.
    void buildAcceleration(const treerndr::RenderContext &ctx);

    /// @brief Stage 3 : Render into prepared modality buffers.
    void renderModalities(const treerndr::RenderContext &ctx);

    /// @brief Stage 4 : Resolve rendered modality buffers.
    void resolveModalities(const treerndr::RenderContext &ctx);

    /// @brief Generate pixel blocks for given raster size.
    std::vector<PixelBlock> generatePixelBlocks(std::size_t width, std::size_t height);

    /// List of currently requested modalities.
    std::set<TracingModality> requestedModalities{ };
    /// Reset all other modalities not in the requested list?
    bool resetOtherModalities{ false };

    /// Storage of all tracing modality buffers.
    ModalityStorage modalities{ };
    /// Scene being ray traced.
    RayTracerScene scene{ };
    /// Caching for scene object triangles.
    TriangleCache triangleCache{ };
    /// Intersector used for ray-triangle calculations.
    treeacc::TriangleIntersector intersector{ };

    /// Verbosity for progress reporting.
    bool verbose{ false };

    /// Number of samples per pixel.
    std::size_t samples{ 1u };
private:
protected:
}; // class RayTracerImpl

} // namespace impl

} // namespace treert

namespace treerndr
{

namespace impl
{

void RenderSystemRTImpl::initialize()
{ reloadRayTracer(); }

void RenderSystemRTImpl::reloadRayTracer()
{
    mState.rayTracer = treert::RayTracer::instantiate();

    if (!mState.fullscreenShader)
    { mState.fullscreenShader = ShaderCompatFullscreen::instantiate(); }
    else
    { mState.fullscreenShader->reloadShaders(); }
}

void RenderSystemRTImpl::render(treescene::CameraState &camera,
    treescene::TreeScene &scene, treerndr::TreeRenderer &renderer)
{
    const auto sortedInstances{ renderer.sortInstancesForRendering() };

    // RayTracing pass:
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

void RenderSystemRTImpl::prepareDrawCtx(treescene::CameraState &camera,
    treescene::TreeScene &scene, treerndr::TreeRenderer &renderer,
    RenderState &state, RenderContext &ctx) const
{
    // Initialize the ray tracer if necessary.
    if (!state.rayTracer)
    { state.rayTracer = treert::RayTracer::instantiate(); }

    // Create the default frame-buffer.
    if (!state.defaultOutputFB)
    { state.defaultOutputFB = FrameBuffer::instantiate(); }

    // Use requested output FB or just the default one for the window.
    if (mParameters.outputFrameBufferOverride)
    { state.outputFB = mParameters.outputFrameBufferOverride; }
    else
    { state.outputFB = state.defaultOutputFB; }

    // Setup rendering context:
    setupDrawCtx(camera, scene, renderer, ctx);

    ctx.modality = mParameters.modality;
    ctx.inputFrameBuffer = { };
    ctx.outputFrameBuffer = state.outputFB;
    ctx.inputShadowMap = { };
    ctx.outputShadowMap = { };

    state.outputFB->setViewport(0u, 0u, ctx.frameBufferWidth, ctx.frameBufferHeight);
    state.outputFB->clearColor(
        ctx.modality.modality() == DisplayModality::Shaded ?
        CLEAR_COLOR_DEFAULT :
        ctx.modality.getClearColor()
    );
    state.outputFB->clearDepth(1.0f);
}

void RenderSystemRTImpl::setupDrawCtx(treescene::CameraState &camera,
    treescene::TreeScene &scene, treerndr::TreeRenderer &renderer,
    RenderContext &ctx)
{
    const auto &lightInstance{ scene.sceneLight() };
    const auto &lightSpec{ *lightInstance->attachedLight };

    camera.cameraPos = camera.calculateCameraPos();

    ctx.frameBufferWidth = camera.viewportWidth;
    ctx.frameBufferHeight = camera.viewportHeight;
    ctx.cameraPosition = camera.cameraPos;
    ctx.lightPosition = lightInstance->calculateLightPosition();
    ctx.cameraNear = camera.nearPlane();
    ctx.cameraFar = camera.farPlane();
    ctx.cameraFov = camera.cameraFov;
    ctx.view = camera.calculateViewMatrix();
    ctx.projection = camera.calculateProjectionMatrix();
    ctx.viewProjection = ctx.projection * ctx.view;
    ctx.lightViewProjection = lightInstance->calculateLightViewProjectionMatrix(false);
    ctx.renderingShadows = false;
    ctx.renderingPhoto = true;
}

void RenderSystemRTImpl::renderInstances(const treescene::CameraState &camera,
    treescene::TreeScene &scene, treerndr::TreeRenderer &renderer,
    const RenderState &state, const RenderContext &ctx,
    const InstanceList &instances) const
{
    // Setup display for the requested modality.
    const auto requestedModality{ treert::displayModalityToTracing(ctx.modality) };
    state.rayTracer->traceModalities(requestedModality, true);

    // Add requested scene instances to the ray-tracer.
    for (const auto &instancePtr : instances)
    { state.rayTracer->scene().addInstance(instancePtr); }

    // Trace rays, generating the output modalities.
    state.rayTracer->traceRays(ctx);

    // Generate OpenGL texture and send it to the GPU.
    const auto modalityTexture{ state.rayTracer->modality(requestedModality).glTexture() };

    // Draw the resulting texture using a fullscreen quad.
    state.fullscreenShader->drawTexture(modalityTexture);
}

void loadReconBundleInto(VaoPackPtr vao, BufferBundlePtr bundle, GLenum mode)
{
#ifndef NDEBUG
    const auto totalSize{
        sizeof(bundle->vertex[0]) * bundle->vertex.size() +
        sizeof(bundle->color[0]) * bundle->color.size() +
        sizeof(bundle->element[0]) * bundle->element.size()
    };
    Info << "Loading reconstruction data onto the GPU (" << totalSize << " bytes)..." << std::endl;
#endif // NDEBUG

    // Store it in the target VAO:
    if (mode != GL_NONE)
    {
        vao->renderMode = mode;
    }
    glBindVertexArray(vao->vao);

    // Cleanup any old data.
    if (vao->vbo)
    {
        glDeleteBuffers(1u, &vao->vbo);
    }
    if (vao->vbocol)
    {
        glDeleteBuffers(1u, &vao->vbocol);
    }
    if (vao->ebo)
    {
        glDeleteBuffers(1u, &vao->ebo);
    }

    if (bundle->vertex.size() % (20u * sizeof(GLfloat)) != 0)
    { Error << "Unable to load bundle: Vertices should always have 16 components!" << std::endl; return; }

    // Upload reconstruction data:
    glGenBuffers(1u, &vao->vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vao->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(bundle->vertex[0]) * bundle->vertex.size(), bundle->vertex.data(), GL_STATIC_DRAW);
    /// Position of the vertex in model space, w is always set to 1.0f.
    glEnableVertexAttribArray(0u);
    glVertexAttribPointer(0u, 4u, GL_FLOAT, GL_FALSE, (4u * 5u * sizeof(GLfloat)), (void*)0u);
    /// Normal of the vertex in model space, w is set to distance from the tree root.
    glEnableVertexAttribArray(1u);
    glVertexAttribPointer(1u, 4u, GL_FLOAT, GL_FALSE, (4u * 5u * sizeof(GLfloat)), (void*)(1u * 4u * sizeof(GLfloat)));
    /// Vector parallel with the branch, w is set to the radius of the branch.
    glEnableVertexAttribArray(2u);
    glVertexAttribPointer(2u, 4u, GL_FLOAT, GL_FALSE, (4u * 5u * sizeof(GLfloat)), (void*)(2u * 4u * sizeof(GLfloat)));
    /// Tangent of the branch, w is unused.
    glEnableVertexAttribArray(3u);
    glVertexAttribPointer(3u, 4u, GL_FLOAT, GL_FALSE, (4u * 5u * sizeof(GLfloat)), (void*)(3u * 4u * sizeof(GLfloat)));
    /// Adjacency indices for this vertex, x = this idx, y = parent idx, z = child idx, w is unused. 0 as invalid idx.
    glEnableVertexAttribArray(4u);
    glVertexAttribIPointer(4u, 4u, GL_UNSIGNED_INT, (4u * 5u * sizeof(GLfloat)), (void*)(4u * 4u * sizeof(GLfloat)));
    glBindBuffer(GL_ARRAY_BUFFER, 0u);

    vao->shaderType = ShaderType::Reconstruction;
    vao->renderMode = GL_PATCHES;
    vao->elementCount = bundle->vertex.size() / (20u * sizeof(GLfloat));
    vao->bundle = bundle;

    glBindVertexArray(0u);
}

glm::mat4 calculateModelMatrixNoAntiProjection(const MeshInstance &instance)
{
    // No re-calculation:
    if(instance.overrideModelMatrix)
    { return instance.model; }

    const auto translate{
        glm::translate(glm::mat4(1.0f),
            instance.translate
        )
    };
    const auto rotate{
        glm::eulerAngleXYZ(
            instance.rotate.x, instance.rotate.y, instance.rotate.z
        )
    };
    const auto scale{
        glm::scale(glm::mat4(1.0f),
            glm::vec3{ instance.scale, instance.scale, instance.scale }
        )
    };
    const auto model{
        translate * rotate * scale
    };

    return model;
}

} // namespace impl

RenderSystemRT::RenderSystemRT() :
    RenderSystem(RENDERER_NAME),
    mImpl{ std::make_shared<impl::RenderSystemRTImpl>() }
{ }
RenderSystemRT::~RenderSystemRT()
{ /* Automatic */ }

void RenderSystemRT::initialize()
{ mImpl->initialize(); }

void RenderSystemRT::render(treescene::CameraState &camera,
    treescene::TreeScene &scene)
{ mImpl->render(camera, scene, *scene.renderer()); }

void RenderSystemRT::reloadShaders()
{ mImpl->reloadRayTracer(); }

RenderSystemRT::VisualizationParameters &RenderSystemRT::parameters()
{ return mImpl->mParameters; }

RenderContext RenderSystemRT::generateRayTracingContext(
    treescene::CameraState &camera, treescene::TreeScene &scene)
{ RenderContext ctx{ }; impl::RenderSystemRTImpl::setupDrawCtx(camera, scene, *scene.renderer(), ctx); return ctx; }

treert::RayTracerPtr RenderSystemRT::prepareTreeRayTracer(
    const treeio::ArrayTree &inputTree, float treeScale)
{
    // Create a temporary copy of the tree and normalize.
    auto treeHolder{ treeop::ArrayTreeHolder::instantiate(inputTree) };
    auto &tree{ treeHolder->currentTree() };
    auto &metadata{ tree.metaData() };
    auto &runtime{ metadata.getRuntimeMetaData<treeio::RuntimeMetaData>()->runtimeTreeProperties };
    runtime.scaleBase = 1.0f;
    runtime.changedScaleBase();
    treeop::normalizeTreeScale(tree, runtime.scaleGraph * runtime.scaleBase, true);
    runtime.scaleBase = treeScale;
    runtime.changedScaleBase();

    // Construct a tree reconstruction.
    auto reconstruction{ treeutil::TreeReconstruction::instantiate() };

    // Create ad-hoc reconstruction instance.
    auto reconMesh{ treerndr::VaoPack::instantiate() };
    reconMesh->shaderType = treerndr::ShaderType::Reconstruction;
    reconMesh->tree = treeHolder;
    reconMesh->reconstruction = reconstruction;
    auto instance{ treerndr::MeshInstance::instantiate() };
    instance->name = treescene::instances::RECONSTRUCTION_NAME;
    instance->mesh = reconMesh;
    runtime.applyScales(instance);

    // Generate reconstruction geometry and load it.
    treeop::TreeVersion reconVersion{ };
    auto reconGeometry{ treescene::TreeScene::treeReconstructionMesh(treeHolder, reconstruction, reconVersion) };
    auto reconBundle{ treerndr::BufferBundle::instantiate() };
    *reconBundle = std::move(reconGeometry);
    TreeRenderer::loadReconBundleInto(reconMesh, reconBundle, GL_NONE);

    // Prepare ray tracing renderer.
    auto rayTracer{ treert::RayTracer::instantiate() };
    rayTracer->scene().addInstance(instance);

    return rayTracer;
}

} // namespace treerndr

namespace treert
{

RayTracerScene::RayTracerScene()
{ reset(); }
RayTracerScene::~RayTracerScene()
{ reset(); }

bool RayTracerScene::addInstance(const InstancePtr &instance)
{
    if (!instance)
    { return false; }

    const auto findIt{ mInstances.find(instance->name) };

    if (findIt != mInstances.end())
    { return false; }

    mInstances.emplace(instance->name, instance);
    mAdded.emplace(instance->name);
    mRemoved.erase(instance->name);

    mDirty = true;
    return true;
}

bool RayTracerScene::removeInstance(const InstancePtr &instance)
{
    if (!instance)
    { return false; }

    const auto removed{ mInstances.erase(instance->name) != 0 };

    if (!removed)
    { return false; }

    mAdded.erase(instance->name);
    mRemoved.emplace(instance->name);

    mDirty = true;
    return true;
}

void RayTracerScene::reset()
{ mInstances = { }; mDirty = true; mAdded = { }; mRemoved = { }; }

const RayTracerScene::InstanceStorage &RayTracerScene::instances() const
{ return mInstances; }

void RayTracerScene::setDirty()
{ mDirty = true; }

void RayTracerScene::resetDirty()
{ mDirty = false; mAdded.clear(); mRemoved.clear(); }

std::size_t RayTracerScene::exportInstances(const std::string &basePath,
    const treerndr::RenderContext &ctx) const
{
    impl::TriangleCache tc{ };
    for (const auto &instance : mInstances)
    {
        const auto outPath{ basePath + "/" + instance.second->name + ".obj" };
        tc.exportInstance(outPath, instance.second, ctx);
    }

    return mInstances.size();
}

namespace impl
{

ModalityStorage::ModalityStorage()
{ reset(); }

ModalityStorage::~ModalityStorage()
{ /* Automatic */ }

void ModalityStorage::reset()
{
    modalities.clear();
    modalities.emplace(TracingModality::Shadow, std::make_shared<RayTracer::ShadowModality>());
    modalities.emplace(TracingModality::Normal, std::make_shared<RayTracer::NormalModality>());
    modalities.emplace(TracingModality::Depth, std::make_shared<RayTracer::DepthModality>());
    modalities.emplace(TracingModality::TriangleCount, std::make_shared<RayTracer::TriangleCountModality>());
    modalities.emplace(TracingModality::Volume, std::make_shared<RayTracer::VolumeModality>());

    resolvedModalities.clear();
    resolvedModalities.emplace(TracingModality::Shadow, std::make_shared<RayTracer::ShadowModality>());
    resolvedModalities.emplace(TracingModality::Normal, std::make_shared<RayTracer::NormalModality>());
    resolvedModalities.emplace(TracingModality::Depth, std::make_shared<RayTracer::DepthModality>());
    resolvedModalities.emplace(TracingModality::TriangleCount, std::make_shared<RayTracer::TriangleCountModality>());
    resolvedModalities.emplace(TracingModality::Volume, std::make_shared<RayTracer::VolumeModality>());
}

void ModalityStorage::initializeModalities(
    const std::set<TracingModality> &types,
    std::size_t width, std::size_t height,
    std::size_t samples, bool resetOthers)
{
    for (const auto &modalityRec : modalities)
    {
        auto &mod{ *modalityRec.second };
        const auto &modName{ modalityRec.first };
        if (types.find(modName) != types.end())
        {
            if (mod.width != width || mod.height != height || mod.samples != samples)
            { mod.initialize(width, height, samples); }
        }
        else if (resetOthers)
        { mod.reset(); }
    }
}

RayTracer::OutputModalityBase &ModalityStorage::modality(TracingModality type, bool resolved)
{ return *modalityPtr(type, resolved); }

ModalityStorage::ModalityPtr &ModalityStorage::modalityPtr(TracingModality type, bool resolved)
{ return resolved ? resolvedModalities[type] : modalities[type]; }

RayTracer::ShadowModality &ModalityStorage::shadowModality(bool resolved)
{ return *std::dynamic_pointer_cast<RayTracer::ShadowModality>(modalityPtr(TracingModality::Shadow, resolved)); }
RayTracer::NormalModality &ModalityStorage::normalModality(bool resolved)
{ return *std::dynamic_pointer_cast<RayTracer::NormalModality>(modalityPtr(TracingModality::Normal, resolved)); }
RayTracer::DepthModality &ModalityStorage::depthModality(bool resolved)
{ return *std::dynamic_pointer_cast<RayTracer::DepthModality>(modalityPtr(TracingModality::Depth, resolved)); }
RayTracer::TriangleCountModality &ModalityStorage::triangleCountModality(bool resolved)
{ return *std::dynamic_pointer_cast<RayTracer::TriangleCountModality>(modalityPtr(TracingModality::TriangleCount, resolved)); }
RayTracer::VolumeModality &ModalityStorage::volumeModality(bool resolved)
{ return *std::dynamic_pointer_cast<RayTracer::VolumeModality>(modalityPtr(TracingModality::Volume, resolved)); }

template <typename BT>
RayTracer::OutputModality<BT> ModalityStorage::generateResolveBuffer(
    const RayTracer::OutputModality<BT> &modality)
{ return RayTracer::OutputModality<BT>{ modality.width, modality.height, 1u }; }

template <typename BT, typename AT>
void ModalityStorage::resolveModalityMean(
    const RayTracer::OutputModality<BT> &input,
    RayTracer::OutputModality<BT> &output)
{
    if (!input.valid())
    { return; }

    if (input.samples == 1u)
    { output = input; return; }

    if (output.width != input.width || output.height != output.height || output.samples != 1u)
    { output = generateResolveBuffer(input); }

    for (std::size_t yPos = 0u; yPos < input.height; ++yPos)
    {
        for (std::size_t xPos = 0u; xPos < input.width; ++xPos)
        {
            AT accumulator{ };
            for (std::size_t sample = 0u; sample < input.samples; ++sample)
            { accumulator += static_cast<AT>(input.element(xPos, yPos, sample)); }
            output.element(xPos, yPos) = static_cast<BT>(accumulator / input.samples);
        }
    }
}

template <>
void ModalityStorage::resolveModalityMean<Vector3D, Vector3D>(
    const RayTracer::OutputModality<Vector3D> &input,
    RayTracer::OutputModality<Vector3D> &output)
{
    if (!input.valid())
    { return; }

    if (input.samples == 1u)
    { output = input; return; }

    if (output.width != input.width || output.height != output.height || output.samples != 1u)
    { output = generateResolveBuffer(input); }

    for (std::size_t yPos = 0u; yPos < input.height; ++yPos)
    {
        for (std::size_t xPos = 0u; xPos < input.width; ++xPos)
        {
            Vector3D accumulator{ };
            for (std::size_t sample = 0u; sample < input.samples; ++sample)
            { accumulator += input.element(xPos, yPos, sample); }
            output.element(xPos, yPos) = (accumulator / input.samples).normalized();
        }
    }
}

template <typename BT>
void ModalityStorage::resolveModalityMedian(
    RayTracer::OutputModality<BT> &input,
    RayTracer::OutputModality<BT> &output)
{
    if (!input.valid())
    { return; }

    if (input.samples == 1u)
    { output = input; return; }

    if (output.width != input.width || output.height != output.height || output.samples != 1u)
    { output = generateResolveBuffer(input); }

    const auto dualMedian{ input.samples % 2u == 0u };
    const auto sampleHalf{ input.samples / 2u - 1u };
    const auto sampleHalfPlusOne{ sampleHalf + 1u };

    for (std::size_t yPos = 0u; yPos < input.height; ++yPos)
    {
        for (std::size_t xPos = 0u; xPos < input.width; ++xPos)
        {
            std::sort(
                input.element(xPos, yPos, 0u),
                input.element(xPos, yPos, input.samples - 1u)
            );

            if (dualMedian)
            {
                output.element(xPos, yPos) = (
                    input.element(xPos, yPos, sampleHalf) +
                    input.element(xPos, yPos, sampleHalfPlusOne) ) /
                        BT{ 2 };
            }
            else
            { output.element(xPos, yPos) = input.element(xPos, yPos, sampleHalf); }
        }
    }
}

std::vector<TracingModality> ModalityStorage::activeModalities()
{
    std::vector<TracingModality> result{ };

    for (const auto &rec : modalities)
    { if (rec.second->valid()) { result.push_back(rec.first); } }

    return result;
}

TriangleCache::TriangleCache()
{ reset(); }
TriangleCache::~TriangleCache()
{ /* Automatic */ }

void TriangleCache::reset()
{ mCache = { }; mDirty = false; mTriangleBuffer = { }; mReconShader = treerndr::ShaderCompatRecon::instantiate(); }

void TriangleCache::cacheInstance(const RayTracerScene::InstancePtr &instance,
    const treerndr::RenderContext &ctx)
{
    auto cacheChanged{ false };

    switch (instance->mesh->shaderType)
    {
        case treerndr::ShaderType::Screen:
        { /* Skip */ break; }
        case treerndr::ShaderType::Mesh:
        { cacheChanged = cacheMesh(instance, ctx); break; }
        case treerndr::ShaderType::Reconstruction:
        { cacheChanged = cacheReconstruction(instance, ctx); break; }
        default:
        {
            Warning << "Unknown shader type used for instanced mesh ("
                    << static_cast<std::size_t>(instance->mesh->shaderType)
                    << ")" << std::endl;
            break;
        }
    }

    mDirty |= cacheChanged;
}

void TriangleCache::removeInstance(const std::string &instanceName)
{ const auto removed{ mCache.erase(instanceName) }; mDirty = (removed != 0); }

const std::vector<treeacc::Triangle> &TriangleCache::generateTriangleBuffer()
{
    if (!mDirty)
    { return mTriangleBuffer; }

    // Accumulate required buffer size.
    std::size_t triangleCount{ 0u };
    for (const auto &record : mCache)
    { triangleCount += record.second->triangles.size(); }

    // Pre-allocate the buffer and copy all data into it.
    mTriangleBuffer.resize(triangleCount);
    std::size_t bufferOffset{ 0u };
    for (const auto &record : mCache)
    {
        const auto &triangles{ record.second->triangles };
        if (!triangles.empty())
        {
            std::memcpy(
                &mTriangleBuffer[bufferOffset], triangles.data(),
                sizeof(mTriangleBuffer[0u]) * triangles.size()
            );
            bufferOffset += triangles.size();
        }
    }

    mDirty = false;
    return mTriangleBuffer;
}

std::tuple<treeio::ObjImporter::Vertex, treeio::ObjImporter::Vertex, treeio::ObjImporter::Vertex>
    convertTriangle(const treeacc::Triangle &triangle)
{
    treeio::ObjImporter::Vertex v1{ };
    v1.position[0u] = triangle.v1[0u]; v1.position[1u] = triangle.v1[1u]; v1.position[2u] = triangle.v1[2u];
    v1.normal[0u] = triangle.n1[0u]; v1.normal[1u] = triangle.n1[1u]; v1.normal[2u] = triangle.n1[2u];

    treeio::ObjImporter::Vertex v2{ };
    v2.position[0u] = triangle.v2[0u]; v2.position[1u] = triangle.v2[1u]; v2.position[2u] = triangle.v2[2u];
    v2.normal[0u] = triangle.n2[0u]; v2.normal[1u] = triangle.n2[1u]; v2.normal[2u] = triangle.n2[2u];

    treeio::ObjImporter::Vertex v3{ };
    v3.position[0u] = triangle.v3[0u]; v3.position[1u] = triangle.v3[1u]; v3.position[2u] = triangle.v3[2u];
    v3.normal[0u] = triangle.n3[0u]; v3.normal[1u] = triangle.n3[1u]; v3.normal[2u] = triangle.n3[2u];

    return { v1, v2, v3 };
}

void TriangleCache::exportInstance(const std::string &path,
    const RayTracerScene::InstancePtr &instance,
    const treerndr::RenderContext &ctx)
{
    cacheInstance(instance, ctx);
    const auto &record{ mCache[instance->name] };

    std::vector<treeio::ObjImporter::Vertex> triangles{ };
    for (const auto &triangle : record->triangles)
    {
        const auto [ v1, v2, v3 ]{ convertTriangle(triangle) };
        triangles.push_back(v1); triangles.push_back(v2); triangles.push_back(v3);
    }

    treeio::ObjImporter importer{ };
    importer.importVertices(triangles);
    importer.exportTo(path);
}

Vector3D TriangleCache::getVertex(const std::vector<char> &buffer, int index, const glm::mat4 &model)
{
    // TODO - We presume we got 3 element float vectors.
    const auto v3{ reinterpret_cast<const glm::vec3*>(buffer.data())[index] };
    const auto v4{ model * glm::vec4{ v3.x, v3.y, v3.z, 1.0f } };
    return { v4.x, v4.y, v4.z };
}

Vector3D TriangleCache::getVertex(const std::vector<glm::vec4> &buffer, int index, const glm::mat4 &model)
{
    const auto v3{ buffer[index] };
    const auto v4{ model * glm::vec4{ v3.x, v3.y, v3.z, 1.0f } };
    return { v4.x, v4.y, v4.z };
}

std::vector<treeacc::Triangle> TriangleCache::loadTriangleMesh(
    const RayTracerScene::InstancePtr &instance, const treerndr::RenderContext &ctx)
{
    // Prepare for copying over triangle data.
    const auto &bundle{ *instance->mesh->bundle };
    const auto &mesh{ *instance->mesh };
    const auto triangleCount{ mesh.elementCount / 3u };

    // Check we have correct number of indices for GL_TRIANGLES.
    if (triangleCount * 3u != mesh.elementCount)
    { return { }; }

    // Calculate model matrix used for vertex transformation.
    const auto model{ treerndr::TreeRenderer::calculateModelMatrixNoAntiProjection(*instance) };

    // Pre-allocate triangle buffer and create the triangles.
    std::vector<treeacc::Triangle> triangles(triangleCount);
    for (std::size_t triangleIdx = 0u; triangleIdx < triangleCount; ++triangleIdx)
    { // Create triangles from the input buffer.
        auto &triangle{ triangles[triangleIdx] };

        // Triangle index:
        triangle.tIdx = triangleIdx;
        // Element indices.
        triangle.i1 = bundle.element[triangleIdx * 3u + 0u];
        triangle.i2 = bundle.element[triangleIdx * 3u + 1u];
        triangle.i3 = bundle.element[triangleIdx * 3u + 2u];
        // Vertex positions.
        triangle.v1 = getVertex(bundle.vertex, triangle.i1, model);
        triangle.v2 = getVertex(bundle.vertex, triangle.i2, model);
        triangle.v3 = getVertex(bundle.vertex, triangle.i3, model);
        /// Lowest vertex positions.
        triangle.lv = Vector3D{
            std::min<float>(std::min<float>(triangle.v1.x, triangle.v2.x), triangle.v3.x),
            std::min<float>(std::min<float>(triangle.v1.y, triangle.v2.y), triangle.v3.y),
            std::min<float>(std::min<float>(triangle.v1.z, triangle.v2.z), triangle.v3.z),
        };
        /// Highest vertex positions.
        triangle.hv = Vector3D{
            std::max<float>(std::max<float>(triangle.v1.x, triangle.v2.x), triangle.v3.x),
            std::max<float>(std::max<float>(triangle.v1.y, triangle.v2.y), triangle.v3.y),
            std::max<float>(std::max<float>(triangle.v1.z, triangle.v2.z), triangle.v3.z),
        };
        /// Centroid.
        triangle.c = treeacc::middleSampleTriangle(triangle.v1, triangle.v2, triangle.v3);
        // Plane normals.
        const auto normal{ treeacc::triangleNormal(triangle.v1, triangle.v2, triangle.v3) };
        triangle.pn1 = normal;
        triangle.pn2 = normal;
        triangle.pn3 = normal;
        // Vertex normals.
        triangle.n1 = normal;
        triangle.n2 = normal;
        triangle.n3 = normal;
    }

    return triangles;
}

std::vector<treeacc::Triangle> TriangleCache::loadTriangleStripMesh(
    const RayTracerScene::InstancePtr &instance, const treerndr::RenderContext &ctx)
{
    // Prepare for copying over triangle data.
    const auto &bundle{ *instance->mesh->bundle };
    const auto &mesh{ *instance->mesh };
    const auto triangleCount{ mesh.elementCount - 2u };

    // Check we have correct number of indices for GL_TRIANGLE_STRIP.
    if (mesh.elementCount <= 2u)
    { return { }; }

    // Calculate model matrix used for vertex transformation.
    const auto model{ treerndr::TreeRenderer::calculateModelMatrixNoAntiProjection(*instance) };

    // Pre-allocate triangle buffer and create the triangles.
    std::vector<treeacc::Triangle> triangles(triangleCount);
    for (std::size_t triangleIdx = 0u; triangleIdx < triangleCount; ++triangleIdx)
    { // Create triangles from the input buffer.
        auto &triangle{ triangles[triangleIdx] };
        const auto reverseWinding{ triangleIdx % 2u == 1u};

        // Triangle index:
        triangle.tIdx = triangleIdx;
        // Element indices.
        if (reverseWinding)
        { // Reverse winding -> Swap first and second elements.
            triangle.i1 = bundle.element[triangleIdx + 1u];
            triangle.i2 = bundle.element[triangleIdx + 0u];
            triangle.i3 = bundle.element[triangleIdx + 2u];
        }
        else
        { // Normal triangle -> Standard triangle strip.
            triangle.i1 = bundle.element[triangleIdx + 0u];
            triangle.i2 = bundle.element[triangleIdx + 1u];
            triangle.i3 = bundle.element[triangleIdx + 2u];
        }
        // Vertex positions.
        triangle.v1 = getVertex(bundle.vertex, triangle.i1, model);
        triangle.v2 = getVertex(bundle.vertex, triangle.i2, model);
        triangle.v3 = getVertex(bundle.vertex, triangle.i3, model);
        /// Lowest vertex positions.
        triangle.lv = Vector3D{
            std::min<float>(std::min<float>(triangle.v1.x, triangle.v2.x), triangle.v3.x),
            std::min<float>(std::min<float>(triangle.v1.y, triangle.v2.y), triangle.v3.y),
            std::min<float>(std::min<float>(triangle.v1.z, triangle.v2.z), triangle.v3.z),
        };
        /// Highest vertex positions.
        triangle.hv = Vector3D{
            std::max<float>(std::max<float>(triangle.v1.x, triangle.v2.x), triangle.v3.x),
            std::max<float>(std::max<float>(triangle.v1.y, triangle.v2.y), triangle.v3.y),
            std::max<float>(std::max<float>(triangle.v1.z, triangle.v2.z), triangle.v3.z),
        };
        /// Centroid.
        triangle.c = treeacc::middleSampleTriangle(triangle.v1, triangle.v2, triangle.v3);
        // Plane normals.
        const auto normal{ treeacc::triangleNormal(triangle.v1, triangle.v2, triangle.v3) };
        triangle.pn1 = normal;
        triangle.pn2 = normal;
        triangle.pn3 = normal;
        // Vertex normals.
        triangle.n1 = normal;
        triangle.n2 = normal;
        triangle.n3 = normal;
    }

    return triangles;
}

std::vector<treeacc::Triangle> TriangleCache::loadRawGeometry(
    const RayTracerScene::InstancePtr &instance,
    const treerndr::RawGeometryPtr &geometryPtr,
    const treerndr::RenderContext &ctx,
    bool skipModelTransform)
{
    // Prepare for copying over triangle data.
    const auto &geometry{ *geometryPtr };
    const auto triangleCount{ geometry.indices.size() / 3u };

    // Check we have correct number of indices for GL_TRIANGLES.
    if (triangleCount * 3u != geometry.indices.size())
    { return { }; }

    // Calculate model matrix used for vertex transformation.
    const auto model{
        skipModelTransform ?
            glm::mat4{ 1.0f } :
            treerndr::impl::calculateModelMatrixNoAntiProjection(*instance)
    };

    // Pre-allocate triangle buffer and create the triangles.
    std::vector<treeacc::Triangle> triangles(triangleCount);
    for (std::size_t triangleIdx = 0u; triangleIdx < triangleCount; ++triangleIdx)
    { // Create triangles from the input buffer.
        auto &triangle{ triangles[triangleIdx] };

        // Triangle index:
        triangle.tIdx = triangleIdx;
        // Element indices.
        triangle.i1 = geometry.indices[triangleIdx * 3u + 0u];
        triangle.i2 = geometry.indices[triangleIdx * 3u + 1u];
        triangle.i3 = geometry.indices[triangleIdx * 3u + 2u];
        // Vertex positions.
        triangle.v1 = getVertex(geometry.vertices, triangle.i1, model);
        triangle.v2 = getVertex(geometry.vertices, triangle.i2, model);
        triangle.v3 = getVertex(geometry.vertices, triangle.i3, model);
        /// Lowest vertex positions.
        triangle.lv = Vector3D{
            std::min<float>(std::min<float>(triangle.v1.x, triangle.v2.x), triangle.v3.x),
            std::min<float>(std::min<float>(triangle.v1.y, triangle.v2.y), triangle.v3.y),
            std::min<float>(std::min<float>(triangle.v1.z, triangle.v2.z), triangle.v3.z),
        };
        /// Highest vertex positions.
        triangle.hv = Vector3D{
            std::max<float>(std::max<float>(triangle.v1.x, triangle.v2.x), triangle.v3.x),
            std::max<float>(std::max<float>(triangle.v1.y, triangle.v2.y), triangle.v3.y),
            std::max<float>(std::max<float>(triangle.v1.z, triangle.v2.z), triangle.v3.z),
        };
        /// Centroid.
        triangle.c = treeacc::middleSampleTriangle(triangle.v1, triangle.v2, triangle.v3);
        // Plane normals.
        const auto normal{ treeacc::triangleNormal(triangle.v1, triangle.v2, triangle.v3) };
        triangle.pn1 = normal;
        triangle.pn2 = normal;
        triangle.pn3 = normal;
        // Vertex normals.
        triangle.n1 = normal;
        triangle.n2 = normal;
        triangle.n3 = normal;
    }

    return triangles;
}

bool TriangleCache::cacheMesh(const RayTracerScene::InstancePtr &instance,
    const treerndr::RenderContext &ctx)
{
    // Presume that meshes are static and we have no need to update them once loaded.
    const auto findIt{ mCache.find(instance->name) };
    if (findIt != mCache.end())
    { return false; }

    // TODO - Make this more robust for different types of meshes.
    // TODO - Support more render modes apart from GL_TRIANGLES.

    const auto bundlePtr{ instance->mesh->bundle };
    if (!bundlePtr)
    { return false; }

    // Copy triangles from the buffer.
    std::vector<treeacc::Triangle> triangles{ };
    switch (instance->mesh->renderMode)
    {
        case GL_TRIANGLES:
        { triangles = loadTriangleMesh(instance, ctx); break; }
        case GL_TRIANGLE_STRIP:
        { triangles = loadTriangleStripMesh(instance, ctx); break; }
        default:
        {
            Warning << "Unknown mesh renderMode type used for instanced mesh ("
                    << static_cast<std::size_t>(instance->mesh->renderMode)
                    << ")" << std::endl;
            return false;
        }
    }

    // Save the results for later.
    auto record{ std::make_shared<InstanceRecord>() };
    record->instance = instance;
    record->triangles = std::move(triangles);
    mCache[instance->name] = record;

    // We have changed the cache.
    return true;
}

bool TriangleCache::cacheReconstruction(const RayTracerScene::InstancePtr &instance,
    const treerndr::RenderContext &ctx)
{
    // Make sure we have valid reconstruction instance.
    if (!instance->mesh->tree || !instance->mesh->reconstruction)
    { return false; }

    // Check for existing cache record or older reconstruction version.
    const auto findIt{ mCache.find(instance->name) };
    const auto currentVersion{ instance->mesh->tree->version() };
    if (findIt != mCache.end() &&
        (findIt->second->treeVersion >= currentVersion ||
         findIt->second->treeVersion == treeop::TreeVersion{ }))
    { return false; }

    // Generate the tree geometry using reconstruction shader.
    const auto &reconPtr{ instance->mesh->reconstruction };
    const auto reconSnapshot{ mReconShader->generateModel(instance, reconPtr, ctx) };

    // Create triangle buffer, skipping model transform since the triangles are already in world space.
    const auto triangles{ loadRawGeometry(instance, reconSnapshot, ctx, true) };

    // Save the results for later.
    auto record{ std::make_shared<InstanceRecord>() };
    record->instance = instance;
    record->triangles = std::move(triangles);
    record->treeVersion = currentVersion;
    mCache[instance->name] = record;

    // We have changed the cache.
    return true;
}

void RayTracerImpl::updateConfiguration(const treerndr::RenderContext &ctx)
{
    if (verbose) Info << "Updating RayTracer configuration..." << std::endl;

    modalities.initializeModalities(requestedModalities,
        ctx.frameBufferWidth, ctx.frameBufferHeight,
        samples, resetOtherModalities);

    if (verbose) Info << "\tInitialized " << requestedModalities.size() <<" modalities to "
        << ctx.frameBufferWidth << "x" << ctx.frameBufferHeight << "..." << std::endl;
}

void RayTracerImpl::buildAcceleration(const treerndr::RenderContext &ctx)
{
    if (verbose) Info << "Building RayTracer acceleration structures..." << std::endl;

    if (!scene.mDirty)
    { if (verbose) { Info << "\tNo update necessary!" << std::endl; } return; }

    if (verbose) Info << "\tAdding " << scene.mAdded.size() << " new instances to the scene." << std::endl;
    for (const auto &instanceName : scene.mAdded)
    { triangleCache.cacheInstance(scene.mInstances[instanceName], ctx); }
    if (verbose) Info << "\t\tDone" << std::endl;

    if (verbose) Info << "\tRemoving " << scene.mRemoved.size() << " instances from the scene." << std::endl;
    for (const auto &instanceName : scene.mRemoved)
    { triangleCache.removeInstance(instanceName); }
    if (verbose) Info << "\t\tDone" << std::endl;

    if (verbose) Info << "\tGenerating triangle buffer..." << std::endl;
    const auto &triangleBuffer{ triangleCache.generateTriangleBuffer() };
    if (verbose) Info << "\t\tDone, triangle buffer has " << triangleBuffer.size() << " triangles!" << std::endl;

    if (verbose) Info << "\tBuilding intersector acceleration structure..." << std::endl;
    intersector.create(triangleBuffer);
    if (verbose) Info << "\t\tDone" << std::endl;
}

void RayTracerImpl::renderModalities(const treerndr::RenderContext &ctx)
{
    if (verbose)
    {
        Info << "Rendering " << requestedModalities.size() << " modalities using ray tracing..." << std::endl;
        Info << "\t[ ";
        for (const auto &modality : requestedModalities)
        { Info << tracingModalityName(modality) << " , "; }
        Info << " ]" << std::endl;
    }

    // Prepare properties of the output buffers.
    const auto width{ ctx.frameBufferWidth };
    const auto height{ ctx.frameBufferHeight };
    const auto totalPixels{ width * height };
    if (verbose) Info << "\tRendering pixels in " << width << "x" << height << " grid!" << std::endl;
    if (verbose) Info << "\tTotal " << totalPixels << " pixels, " << samples << " each." << std::endl;

    // Prepare output buffers.
    auto &shadowModality{ modalities.shadowModality(false) };
    const auto doShadowModality{ shadowModality.valid() };
    auto &normalModality{ modalities.normalModality(false) };
    const auto doNormalModality{ normalModality.valid() };
    auto &depthModality{ modalities.depthModality(false) };
    const auto doDepthModality{ depthModality.valid() };
    auto &triangleCountModality{ modalities.triangleCountModality(false) };
    const auto doTriangleCountModality{ triangleCountModality.valid() };
    auto &volumeModality{ modalities.volumeModality(false) };
    const auto doVolumeModality{ volumeModality.valid() };

    // Do we need to secondary rays?
    const auto secondaryRaysRequired{
        doTriangleCountModality || doVolumeModality
    };

    // Prepare camera projection matrices.
    const auto &projection{ ctx.projection };
    const auto &view{ ctx.view };
    const auto vp{ projection * view };
    const auto vpi{ glm::inverse(vp) };
    const auto wsLightPosition{ Vector3D{ ctx.lightPosition } };
    const auto orthoCamera{ projection[3u][3u] > 0.0f };

    // Progress reporting.
    treeutil::ProgressBar progressBar{ "Rendering " };
    treeutil::ProgressPrinter progressPrinter{ progressBar, totalPixels, 10u };
    std::size_t currentlyFinished{ 0u };
    std::mutex mtx{ };

    // Parallelization preparation.
    std::vector<PixelBlock> pixelBlocks{ generatePixelBlocks(width, height) };

    // Run ray-tracing in parallel over pre-generated blocks.
    std::for_each(
        std::execution::par,
        pixelBlocks.begin(),
        pixelBlocks.end(),
        [&](const auto &block)
        {
            std::default_random_engine generator{ };
            std::normal_distribution<float> distribution(0.0f, 0.3f);
            std::vector<float> jitterMatrix{ };
            jitterMatrix.resize(samples * 2u);

            for (std::size_t iii = 0u; iii < jitterMatrix.size(); ++iii)
            { jitterMatrix[iii] = distribution(generator); }

            for (std::size_t yPixel = block.yBegin; yPixel < block.yEnd; ++yPixel)
            {
                for (std::size_t xPixel = block.xBegin; xPixel < block.xEnd; ++xPixel)
                {
                    for (std::size_t sample = 0u; sample < samples; ++sample)
                    {
                        // Calculate ray properties for world-space ray casting.
                        const auto ndcRayOrigin{
                            glm::vec4{
                                2.0f * (xPixel + jitterMatrix[sample * 2u + 0u]) / static_cast<float>(width - 1u) - 1.0f,
                                2.0f * (yPixel + jitterMatrix[sample * 2u + 1u]) / static_cast<float>(height - 1u) - 1.0f,
                                orthoCamera * -1.0f + 0.0001f,
                                1.0f
                            }
                        };
                        const auto ndcRayDelta{ ndcRayOrigin + glm::vec4{ 0.0f, 0.0f, 0.1f, 0.0f } };
                        const auto wswRayOrigin{ vpi * ndcRayOrigin };
                        const auto wswRayDelta{ vpi * ndcRayDelta };
                        const auto wsRayOrigin{ orthoCamera ?
                            Vector3D{ wswRayOrigin, true } :
                            Vector3D{ ctx.cameraPosition }
                        };
                        const auto wsRayDelta{ Vector3D{ wswRayDelta, true } };
                        const auto wsRayDirection{ (wsRayDelta - wsRayOrigin).normalized() };

                        auto result{ intersector.queryRayNearest(wsRayOrigin, wsRayDirection) };
                        auto hitGeometry{ result.first != intersector.InvalidPrimitiveIdx };
                        auto wsHitLocation{ wsRayOrigin + wsRayDirection * result.second };

                        if (doDepthModality)
                        { // Record depth of the first hit.
                            if (hitGeometry)
                            { depthModality.element(xPixel, yPixel, sample) = result.second; }
                            else
                            { depthModality.element(xPixel, yPixel, sample) = 0.0f; }
                        }

                        if (doNormalModality)
                        { // Record depth of the first hit.
                            if (hitGeometry)
                            { // Recover the hit triangle and use its normal.
                                const auto &hitTriangle{ intersector.primitives()[result.first] };
                                normalModality.element(xPixel, yPixel, sample) = hitTriangle.centralPlaneNormal();
                            }
                            else
                            { normalModality.element(xPixel, yPixel, sample) = { }; }
                        }

                        if (doShadowModality)
                        { // Cast secondary shadow ray and record result.
                            const auto wsToLight{ (wsLightPosition - wsHitLocation).normalized() };
                            // Offset the origin by a small amount in order to prevent self occlusion.
                            const auto shadowResult{
                                intersector.queryRayNearest(
                                    wsHitLocation + wsToLight * RAY_DELTA,
                                    wsToLight
                                )
                            };
                            const auto shadowHitGeometry{ shadowResult.first != intersector.InvalidPrimitiveIdx };
                            if (shadowHitGeometry)
                            { shadowModality.element(xPixel, yPixel, sample) = 1.0f; }
                            else
                            { shadowModality.element(xPixel, yPixel, sample) = 0.0f; }
                        }

                        if (secondaryRaysRequired)
                        { // Advanced modalities requested -> Cast secondary rays.
                            // Prepare modality calculation.
                            auto secondaryRaysCast{ 0ull };
                            auto hitCount{ hitGeometry ? 1u : 0u };
                            auto volume{ 0.0f };
                            auto volumeLastHit{ result.second };
                            auto insideVolume{ hitGeometry };

                            while (hitGeometry && secondaryRaysCast < RAY_SECONDARY_LIMIT)
                            { // Until the ray escapes traced geometry.
                                // Offset the origin by a small amount in order to prevent self occlusion.
                                wsHitLocation += wsRayDirection * RAY_DELTA;
                                // Trace the recurrent ray.
                                ++secondaryRaysCast;
                                result = intersector.queryRayNearest(
                                    wsHitLocation,
                                    wsRayDirection
                                );
                                hitGeometry = result.first != intersector.InvalidPrimitiveIdx;
                                wsHitLocation = wsHitLocation + wsRayDirection * result.second;

                                if (hitGeometry)
                                {
                                    hitCount++;
                                    if (insideVolume)
                                    {
                                        volume += result.second - volumeLastHit;
                                        volumeLastHit = result.second; insideVolume = false;
                                    }
                                    else
                                    { volumeLastHit = result.second; insideVolume = true; }
                                }
                            }

                            // Update requested modalities:
                            if (doTriangleCountModality)
                            { triangleCountModality.element(xPixel, yPixel, sample) = hitCount; }
                            if (doVolumeModality)
                            { volumeModality.element(xPixel, yPixel, sample) = volume; }
                        }
                    }
                }
            }

            if (verbose)
            { std::unique_lock<std::mutex> lock{ mtx };
                const auto blockPixelCount{
                    (block.yEnd - block.yBegin) *
                    (block.xEnd - block.xBegin)
                };
                currentlyFinished += blockPixelCount;
                progressPrinter.printProgress(Info, currentlyFinished);
            }
        }
    );

    if (verbose) Info << "\tDone!" << std::endl;
}

void RayTracerImpl::resolveModalities(const treerndr::RenderContext &ctx)
{
    if (verbose)
    { Info << "Resolving " << requestedModalities.size() << " modalities..." << std::endl; }

    // Resolve shadows:
    modalities.resolveModalityMean(modalities.shadowModality(false), modalities.shadowModality(true));
    // Resolve normals:
    modalities.resolveModalityMean<Vector3D, Vector3D>(modalities.normalModality(false), modalities.normalModality(true));
    // Resolve depth:
    modalities.resolveModalityMean(modalities.depthModality(false), modalities.depthModality(true));
    // Resolve triangle count:
    // TODO - Use resolveModalityMedian instead?
    modalities.resolveModalityMean(modalities.triangleCountModality(false), modalities.triangleCountModality(true));
    // Resolve volume:
    modalities.resolveModalityMean(modalities.volumeModality(false), modalities.volumeModality(true));

    if (verbose) Info << "\tDone!" << std::endl;
}

std::vector<RayTracerImpl::PixelBlock> RayTracerImpl::generatePixelBlocks(
    std::size_t width, std::size_t height)
{
    // Determine optimal division.
    const auto concurrency{ std::thread::hardware_concurrency() };

    const auto xBlockSize{ std::clamp<std::size_t>(
        width / concurrency,
        PARALLEL_MIN_BLOCK_SIZE,
        PARALLEL_MAX_BLOCK_SIZE
    ) };
    const auto yBlockSize{ std::clamp<std::size_t>(
        height / concurrency,
        PARALLEL_MIN_BLOCK_SIZE,
        PARALLEL_MAX_BLOCK_SIZE
    ) };

    const auto xBlockCount{ static_cast<std::size_t>(
        std::ceil(width / static_cast<float>(xBlockSize))
    ) };
    const auto yBlockCount{ static_cast<std::size_t>(
        std::ceil(height / static_cast<float>(yBlockSize))
    ) };

    // Divide input raster into pixel blocks.
    std::vector<PixelBlock> pixelBlocks{ };
    for (std::size_t yBlock = 0u; yBlock < yBlockCount; ++yBlock)
    {
        for (std::size_t xBlock = 0u; xBlock < xBlockCount; ++xBlock)
        {
            pixelBlocks.emplace_back(PixelBlock{
                xBlock * xBlockSize,
                std::min<std::size_t>((xBlock + 1u) * xBlockSize, width),
                yBlock * yBlockSize,
                std::min<std::size_t>((yBlock + 1u) * yBlockSize, height),
            });
        }
    }

    return pixelBlocks;
}

} // namespace impl

TracingModality tracingModalityFromIdx(std::size_t idx)
{
    switch (idx)
    {
        default:
        case 0u:
        { return TracingModality::Shadow; }
        case 1u:
        { return TracingModality::Normal; }
        case 2u:
        { return TracingModality::Depth; }
        case 3u:
        { return TracingModality::TriangleCount; }
        case 4u:
        { return TracingModality::Volume; }
    }
}

std::string tracingModalityName(TracingModality modality)
{
    switch (modality)
    {
        default:
        case TracingModality::Shadow:
        { return "Shadow"; }
        case TracingModality::Normal:
        { return "Normal"; }
        case TracingModality::Depth:
        { return "Depth"; }
        case TracingModality::TriangleCount:
        { return "TriangleCount"; }
        case TracingModality::Volume:
        { return "Volume"; }
    }
}

std::size_t tracingModalityIdx(TracingModality modality)
{
    switch (modality)
    {
        default:
        case TracingModality::Shadow:
        { return 0u; }
        case TracingModality::Normal:
        { return 1u; }
        case TracingModality::Depth:
        { return 2u; }
        case TracingModality::TriangleCount:
        { return 3u; }
        case TracingModality::Volume:
        { return 4u; }
    }
}

std::size_t tracingModalityCount()
{ return 5u; }

TracingModality displayModalityToTracing(const treerndr::RendererModality &modality)
{
    switch (modality.modality())
    {
        default:
        case treerndr::DisplayModality::Shaded:
        case treerndr::DisplayModality::Albedo:
        case treerndr::DisplayModality::Depth:
        { return TracingModality::Depth; }
        case treerndr::DisplayModality::Light:
        case treerndr::DisplayModality::Shadow:
        { return TracingModality::Shadow; }
        case treerndr::DisplayModality::Normal:
        { return TracingModality::Normal; }
    }
}

RayTracer::RayTracer()
{ reset(); }
RayTracer::~RayTracer()
{ /* Automatic */ }

void RayTracer::reset()
{ mImpl = std::make_shared<impl::RayTracerImpl>(); }

RayTracerScene &RayTracer::scene()
{ return mImpl->scene; }

void RayTracer::traceModalities(const std::initializer_list<TracingModality> &types, bool resetOthers)
{ mImpl->requestedModalities = types; mImpl->resetOtherModalities = resetOthers; }

void RayTracer::traceModalities(const std::vector<TracingModality> &types, bool resetOthers)
{ mImpl->requestedModalities = { types.begin(), types.end() }; mImpl->resetOtherModalities = resetOthers; }

void RayTracer::traceModalities(TracingModality type, bool resetOthers)
{ mImpl->requestedModalities = { type }; mImpl->resetOtherModalities = resetOthers; }

void RayTracer::setVerbose(bool verbose)
{ mImpl->verbose = verbose; }

void RayTracer::setSampling(std::size_t samples)
{ mImpl->samples = samples; }

treerndr::RenderContext RayTracer::generatePerspectiveContext(
    std::size_t viewportWidth, std::size_t viewportHeight,
    const Vector3D &cameraPosition, const Vector3D &cameraTarget,
    float cameraFov, float cameraNear, float cameraFar,
    const Vector3D &lightPosition)
{
    treerndr::RenderContext ctx{ };

    ctx.frameBufferWidth = viewportWidth;
    ctx.frameBufferHeight = viewportHeight;
    ctx.cameraPosition = cameraPosition.toGlm();
    ctx.cameraNear = cameraNear;
    ctx.cameraFar = cameraFar;
    ctx.cameraFov = cameraFov;
    ctx.lightPosition = lightPosition.toGlm();

    treescene::CameraState cameraState{ };
    cameraState.viewportWidth = viewportWidth;
    cameraState.viewportHeight = viewportHeight;
    cameraState.cameraPos = cameraPosition.toGlm();
    cameraState.cameraTargetPos = cameraTarget.toGlm();
    cameraState.cameraFov = cameraFov;
    cameraState.cameraPerspectiveNearPlane = cameraNear;
    cameraState.cameraPerspectiveFarPlane = cameraFar;
    cameraState.cameraProjection = treescene::CameraProjection::Perspective;

    ctx.view = cameraState.calculateViewMatrix(cameraPosition.toGlm());
    ctx.projection = cameraState.calculateProjectionMatrix();

    return ctx;
}

treerndr::RenderContext RayTracer::generateOrthoContext(
    std::size_t viewportWidth, std::size_t viewportHeight,
    const Vector3D &cameraPosition, const Vector3D &cameraTarget,
    float cameraFov, float cameraNear, float cameraFar,
    const Vector3D &lightPosition)
{
    treerndr::RenderContext ctx{ };

    ctx.frameBufferWidth = viewportWidth;
    ctx.frameBufferHeight = viewportHeight;
    ctx.cameraPosition = cameraPosition.toGlm();
    ctx.cameraNear = cameraNear;
    ctx.cameraFar = cameraFar;
    ctx.cameraFov = cameraFov;
    ctx.lightPosition = lightPosition.toGlm();

    treescene::CameraState cameraState{ };
    cameraState.viewportWidth = viewportWidth;
    cameraState.viewportHeight = viewportHeight;
    cameraState.cameraPos = cameraPosition.toGlm();
    cameraState.cameraTargetPos = cameraTarget.toGlm();
    cameraState.cameraFov = cameraFov;
    cameraState.cameraOrthographicNearPlane = cameraNear;
    cameraState.cameraOrthographicFarPlane = cameraFar;
    cameraState.cameraProjection = treescene::CameraProjection::Orthographic;

    ctx.view = cameraState.calculateViewMatrix(cameraPosition.toGlm());
    ctx.projection = cameraState.calculateProjectionMatrix();

    return ctx;
}

void RayTracer::traceRays(const treerndr::RenderContext &ctx)
{
    /* Stage 1 : Update current configuration. */
    mImpl->updateConfiguration(ctx);

    /* Stage 2 : Build the acceleration structures. */
    mImpl->buildAcceleration(ctx);

    /* Stage 3 : Render into prepared modality buffers. */
    mImpl->renderModalities(ctx);

    /* Stage 4 : Resolve rendered modality buffers. */
    mImpl->resolveModalities(ctx);
}

const RayTracer::ShadowModality &RayTracer::shadowModality() const
{ return mImpl->modalities.shadowModality(true); }
const RayTracer::NormalModality &RayTracer::normalModality() const
{ return mImpl->modalities.normalModality(true); }
const RayTracer::DepthModality &RayTracer::depthModality() const
{ return mImpl->modalities.depthModality(true); }
const RayTracer::TriangleCountModality &RayTracer::triangleCountModality() const
{ return mImpl->modalities.triangleCountModality(true); }
const RayTracer::VolumeModality &RayTracer::volumeModality() const
{ return mImpl->modalities.volumeModality(true); }

const RayTracer::OutputModalityBase &RayTracer::modality(TracingModality type) const
{ return mImpl->modalities.modality(type, true); }

std::vector<TracingModality> RayTracer::activeModalities() const
{ return mImpl->modalities.activeModalities(); }

}
