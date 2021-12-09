/**
 * @author Tomas Polasek
 * @date 4.16.2020
 * @version 1.0
 * @brief Compatibility header for reconstruction shaders.
 */

#include "TreeShaderCompatRecon.h"

#include "TreeShaderCompatBlob.h"
#include "TreeRenderer.h"

namespace treerndr
{

ShaderCompatRecon::ShaderCompatRecon()
{ initializeUniforms(); reloadShaders(); }

void ShaderCompatRecon::initializeUniforms()
{
    mUniforms = UniformSet(
        false,
        uModel, uView, uProjection, uVP, uMVP, uNT, uLightView, uCameraPos, uLightPos,
        uMeshScale, uTessellationMultiplier, uShadowMap, uApplyShadow, uPhotoShading,
        uShadowKernelSpec, uForegroundColor, uBackgroundColor, uOpaqueness,
        uMaxBranchRadius, uMaxDistanceFromRoot, uBranchTension, uBranchBias,
        uBranchWidthMultiplier, uModalitySelector, uCamera, bData
    );
}

void ShaderCompatRecon::reloadShaders()
{
    mProgram = ShaderProgramFactory::renderProgram()
        .addVertexFallback("shaders/recon_vs.glsl", glsl::RECON_VS_GLSL)
        .addControlFallback("shaders/recon_cs.glsl", glsl::RECON_CS_GLSL)
        .addEvaluationFallback("shaders/recon_es.glsl", glsl::RECON_ES_GLSL)
        .addGeometryFallback("shaders/recon_gs.glsl", glsl::RECON_GS_GLSL)
        .addFragmentFallback("shaders/recon_fs.glsl", glsl::RECON_FS_GLSL)
        .checkUniforms(mUniforms.getUniformIdentifiers())
        .transformFeedback({ "gPosition" })
        .finalize();
}

void ShaderCompatRecon::activate() const
{ mProgram.use(); }

void ShaderCompatRecon::setUniforms() const
{ mUniforms.setUniforms(mProgram); }

void ShaderCompatRecon::render(const MeshInstancePtr &instancePtr,
    const treeutil::TreeReconstruction::Ptr &reconPtr,
    const RenderContext &ctx, bool transformFeedback)
{
    if (!instancePtr || !instancePtr->show || !reconPtr)
    { return; }

    // Prepare reconstruction data:
    auto &instance{ *instancePtr };
    const auto &mesh{ *instance.mesh };
    const auto model{ TreeRenderer::calculateModelMatrixNoAntiProjection(instance) };
    const auto &reconstruction{ *reconPtr };
    const auto &visParameters{ reconstruction.parameters() };

    if (ctx.renderingShadows && !instance.castsShadows)
    { return; }

    const auto &projection{ ctx.projection };
    const auto &view{ ctx.view };
    const auto viewProjection{ projection * view };

    const auto normalTransform{ glm::transpose(glm::inverse(view * model)) };
    const auto mvp{ viewProjection * model };
    instance.model = model;

    const glm::vec3 meshScale{ visParameters.meshScale, visParameters.meshScale, visParameters.meshScale };

    const auto applyShadow{ !ctx.renderingShadows && ctx.renderingPhoto };
    const auto photoShading{ !ctx.renderingShadows && ctx.renderingPhoto };
    const auto shadowKernelSpec{ glm::vec4{
        ctx.shadowKernelSize,
        ctx.shadowSamplingFactor,
        ctx.shadowStrength,
        ctx.shadowBias
    } };

    const auto vertexDataBuffer{ Buffer::createWrapBuffer<GLfloat>(
        mesh.vbo, mesh.elementCount, false,
        Buffer::UsageFrequency::Static,
        Buffer::UsageAccess::Draw,
        Buffer::Target::ShaderStorage
    ) };

    // Set shader uniforms:
    uModel = model;
    uView = view;
    uProjection = projection;
    uVP = viewProjection;
    uMVP = mvp;
    uNT = normalTransform;
    uLightView = ctx.lightViewProjection;
    uCameraPos = ctx.cameraPosition;
    uLightPos = ctx.lightPosition;
    uMeshScale = meshScale;
    uTessellationMultiplier = visParameters.tessellationMultiplier;
    uShadowMap = ctx.inputShadowMap;
    uApplyShadow = applyShadow;
    uPhotoShading = photoShading;
    uShadowKernelSpec = shadowKernelSpec;
    uForegroundColor = visParameters.foregroundColor;
    uBackgroundColor = visParameters.backgroundColor;
    uOpaqueness = visParameters.opaqueness;
    uMaxBranchRadius = reconPtr->maxBranchRadius();
    uMaxDistanceFromRoot = reconPtr->maxDistanceFromRoot();
    uBranchTension = visParameters.branchTension;
    uBranchBias = visParameters.branchBias;
    uBranchWidthMultiplier = visParameters.branchWidthMultiplier;
    uModalitySelector = ctx.modality.idx();
    uCamera = { ctx.cameraNear, ctx.cameraFar, ctx.cameraFov, 0.0f };
    bData = vertexDataBuffer;

    // Render the reconstruction:
    if (!transformFeedback)
    { activate(); }
    setUniforms();
    if (ctx.outputFrameBuffer) { ctx.outputFrameBuffer->bind(); }
    { glBindVertexArray(mesh.vao);
        glEnable(GL_DEPTH_TEST);
        if (instance.alwaysVisible)
        { glDepthFunc(GL_ALWAYS); }
        else
        { glDepthFunc(GL_LESS); }

        if (instance.cullFaces)
        { glEnable(GL_CULL_FACE); }
        else
        { glDisable(GL_CULL_FACE); }

        if (instance.transparent)
        { glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); }
        else
        { glDisable(GL_BLEND); }

        glPatchParameteri(GL_PATCH_VERTICES, 2);
        glPolygonMode(GL_FRONT_AND_BACK, instance.wireframe ? GL_LINE : GL_FILL);

        glDrawArrays(mesh.renderMode, 0, static_cast<GLsizei>(mesh.elementCount));

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0u);
    } glBindVertexArray(0u);
    if (ctx.outputFrameBuffer) { ctx.outputFrameBuffer->unbind(); }
}

std::size_t ShaderCompatRecon::transformFeedback(
    const MeshInstancePtr &instancePtr,
    const treeutil::TreeReconstruction::Ptr &reconPtr,
    const RenderContext &ctx, Buffer::Ptr &outputBuffer)
{
    // Prepare query for determining number of primitives generated.
    GLuint primitiveCountQuery[2u]{ 0u, 0u };
    glGenQueries(2u, primitiveCountQuery);

    // Perform dummy rendering without actually writing to frame-buffer.
    glEnable(GL_RASTERIZER_DISCARD);

    // Get total number of primitives written so we can prepare appropriate sized buffer.
    { glBeginQuery(GL_PRIMITIVES_GENERATED, primitiveCountQuery[0u]);

        {
            render(instancePtr, reconPtr, ctx);
        }

        // Make sure we wait until all of the operations are finished.
        glFlush();

    } glEndQuery(GL_PRIMITIVES_GENERATED);

    // Calculate resulting vertex count.
    GLuint primitiveCount{ 0u };
    glGetQueryObjectuiv(primitiveCountQuery[0u], GL_QUERY_RESULT, &primitiveCount);
    const auto totalTriangleCount{ primitiveCount };
    const auto totalVertexCount{ VERTICES_PER_PRIMITIVE * primitiveCount };

    // Create appropriately sized buffer:
    if (!outputBuffer ||
        outputBuffer->usesType<glm::vec3>() ||
        outputBuffer->elementCount() < totalVertexCount)
    {
        outputBuffer = Buffer::instantiate(
            totalVertexCount, glm::vec3{ },
            Buffer::UsageFrequency::Static,
            Buffer::UsageAccess::Read,
            Buffer::Target::TransformFeedback);
    }

    // Perform real geometry generation and save it to the prepared transform feedback buffer.
    { glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, primitiveCountQuery[1u]);

        activate();

        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, outputBuffer->id());
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0u, outputBuffer->id());

        { glBeginTransformFeedback(PRIMITIVE_USED);
            render(instancePtr, reconPtr, ctx, true);
        } glEndTransformFeedback();

        // Make sure we wait until all of the operations are finished.
        glFlush();

    } glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);

    // Check we actually received announced number of vertices.
    glGetQueryObjectuiv(primitiveCountQuery[1u], GL_QUERY_RESULT, &primitiveCount);
    const auto resultTriangleCount{ primitiveCount };
    if (totalTriangleCount != resultTriangleCount)
    {
        Error << "Transform feedback generated different amount of triangles than announced ("
              << resultTriangleCount << " vs " << totalTriangleCount << std::endl;
    }

    // Return context state to original settings:
    glDisable(GL_RASTERIZER_DISCARD);

    // Free created queries.
    glDeleteQueries(sizeof(primitiveCountQuery) / sizeof(primitiveCountQuery[0]), primitiveCountQuery);

    return resultTriangleCount;
}

RawGeometryPtr ShaderCompatRecon::generateModel(
    const MeshInstancePtr &instancePtr,
    const treeutil::TreeReconstruction::Ptr &reconPtr,
    const RenderContext &ctx)
{
    treeutil::Timer profilingTimer{ };

    // Generate the model and store it in GPU buffer.
    const auto trianglesGenerated{ transformFeedback(
        instancePtr, reconPtr, ctx,
        mTransformFeedbackBuffer
    )};

    Info << "[Prof] timeReconstructionGpu: " << profilingTimer.reset() << std::endl;

    // Create container and store generated triangles in it.
    const auto result{ RawGeometry::instantiate() };

    // Download data from the GPU.
    mTransformFeedbackBuffer->download();
    const auto &data{ mTransformFeedbackBuffer->asVector<glm::vec3>() };

    // Copy triangle mesh geometry.
    for (std::size_t tIdx = 0u; tIdx < trianglesGenerated; ++tIdx)
    { result->insertTriangle(data[tIdx * 3u + 0u], data[tIdx * 3u + 1u], data[tIdx * 3u + 2u]); }

    return result;
}

void ShaderCompatRecon::describe(std::ostream &out, const std::string &indent) const
{
    out << "[ ShaderCompatRecon: \n"
        << indent << "\tShader Program = ";
    mProgram.describe(out, indent + "\t\t");
    out << "\n" << indent << "\tShader Uniforms = ";
    mUniforms.describe(out, indent + "\t\t");
    out << "\n" << indent << " ]";
}

} // namespace treerndr
