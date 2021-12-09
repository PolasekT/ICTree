/**
 * @author Tomas Polasek, David Hrusa
 * @date 5.19.2020
 * @version 1.0
 * @brief Compatibility header for edit rendering shaders.
 */

#include "TreeShaderCompatMesh.h"

#include "TreeShaderCompatBlob.h"
#include "TreeRenderer.h"

namespace treerndr
{

ShaderCompatMesh::ShaderCompatMesh()
{ initializeUniforms(); reloadShaders(); }

void ShaderCompatMesh::initializeUniforms()
{
    mUniforms = UniformSet(
        false,
        uColorOverride, uDoOverrideColor, uUseColorStorage, uModel, uMVP, uShaded,
        uShadowMap, uShadowKernelSpec, uLightViewProjection, uModalitySelector, uCamera,
        bOverride
    );
}

void ShaderCompatMesh::reloadShaders()
{
    mProgram = ShaderProgramFactory::renderProgram()
        .addVertexFallback("shaders/base_vs.glsl", glsl::BASE_VS_GLSL)
        .addFragmentFallback("shaders/base_fs.glsl", glsl::BASE_FS_GLSL)
        .checkUniforms(mUniforms.getUniformIdentifiers())
        .finalize();
}

void ShaderCompatMesh::activate() const
{ mProgram.use(); }

void ShaderCompatMesh::setUniforms() const
{ mUniforms.setUniforms(mProgram); }

void ShaderCompatMesh::render(const MeshInstancePtr &instancePtr,
    const RenderContext &ctx, const treescene::CameraState &camera)
{
    if (!instancePtr || !instancePtr->show)
    { return; }

    // Prepare instance data:
    auto &instance{ *instancePtr };
    const auto &mesh{ *instance.mesh };
    const auto model{ TreeRenderer::calculateModelMatrix(instance, camera) };

    if (ctx.renderingShadows && !instance.castsShadows)
    { return; }

    const auto &projection{ ctx.projection };
    const auto &view{ ctx.view };

    const auto mvp{ projection * view * model };
    instance.model = model;

    const auto useShading{ !ctx.renderingShadows && instance.shadows };
    const auto shadowKernelSpec{ glm::vec4{
        ctx.shadowKernelSize,
        ctx.shadowSamplingFactor,
        ctx.shadowStrength,
        ctx.shadowBias
    } };

    // Set shader uniforms:
    uColorOverride = instance.overrideColor;
    uDoOverrideColor = instance.overrideColorEnabled;
    uUseColorStorage = instance.overrideColorStorageEnabled;
    uModel = model;
    uMVP = mvp;
    uShaded = useShading;
    uShadowMap = ctx.inputShadowMap;
    uShadowKernelSpec = shadowKernelSpec;
    uLightViewProjection = ctx.lightViewProjection;
    uModalitySelector = ctx.modality.idx();
    uCamera = { camera.nearPlane(), camera.farPlane(), camera.cameraFov, 0.0f };

    if (mesh.sbo)
    { // Wrap original storage buffer object in a wrapper.
        bOverride = Buffer::createWrapBuffer<glm::vec4>(
            mesh.sbo, mesh.elementCount, false,
            Buffer::UsageFrequency::Static,
            Buffer::UsageAccess::Draw,
            Buffer::Target::ShaderStorage
        );
    }
    else
    { bOverride = nullptr; }

    // Render the mesh:
    activate();
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

        glPointSize(instance.pointSize);
        glLineWidth(instance.lineWidth);
        glPolygonMode(GL_FRONT_AND_BACK, instance.wireframe ? GL_LINE : GL_FILL);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ebo);

        glDrawElements(mesh.renderMode, static_cast<GLsizei>(mesh.elementCount), GL_UNSIGNED_INT, nullptr);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0u);
    } glBindVertexArray(0u);
    if (ctx.outputFrameBuffer) { ctx.outputFrameBuffer->unbind(); }
}

void ShaderCompatMesh::describe(std::ostream &out, const std::string &indent) const
{
    out << "[ ShaderCompatMesh: \n"
        << indent << "\tShader Program = ";
    mProgram.describe(out, indent + "\t\t");
    out << "\n" << indent << "\tShader Uniforms = ";
    mUniforms.describe(out, indent + "\t\t");
    out << "\n" << indent << " ]";
}

} // namespace treerndr
