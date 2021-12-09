/**
 * @author Tomas Polasek, David Hrusa
 * @date 5.19.2020
 * @version 1.0
 * @brief Compatibility header for edit rendering shaders.
 */

#include "TreeShaderCompatScreen.h"

#include "TreeShaderCompatBlob.h"
#include "TreeRenderer.h"

namespace treerndr
{

ShaderCompatScreen::ShaderCompatScreen()
{ initializeUniforms(); reloadShaders(); }

void ShaderCompatScreen::initializeUniforms()
{
    mUniforms = UniformSet(
        false,
        uMVP, uTextured, uTextures
    );
}

void ShaderCompatScreen::reloadShaders()
{
    mProgram = ShaderProgramFactory::renderProgram()
        .addVertexFallback("shaders/screen_vs.glsl", glsl::SCREEN_VS_GLSL)
        .addFragmentFallback("shaders/screen_fs.glsl", glsl::SCREEN_FS_GLSL)
        .checkUniforms(mUniforms.getUniformIdentifiers())
        .finalize();
}

void ShaderCompatScreen::activate() const
{ mProgram.use(); }

void ShaderCompatScreen::setUniforms() const
{ mUniforms.setUniforms(mProgram); }

void ShaderCompatScreen::render(const MeshInstancePtr &instancePtr,
    const RenderContext &ctx, const treescene::CameraState &camera)
{
    if (!instancePtr || !instancePtr->show)
    { return; }

    // Prepare instance data:
    auto &instance{ *instancePtr };
    const auto &mesh{ *instance.mesh };
    const auto model{ TreeRenderer::calculateModelMatrix(instance, camera) };

    const auto mvp{ glm::ortho<float>(
        0.0f, ctx.frameBufferWidth,
        0.0f, ctx.frameBufferHeight,
        -1.0f, 1.0f
    ) };
    instance.model = model;

    // Set shader uniforms:
    uMVP = mvp;
    uTextured = false;
    uTextures = { };

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

        glPolygonMode(GL_FRONT_AND_BACK, instance.wireframe ? GL_LINE : GL_FILL);

        glDrawArrays(mesh.renderMode, 0, static_cast<GLsizei>(mesh.elementCount));
    } glBindVertexArray(0u);
    if (ctx.outputFrameBuffer) { ctx.outputFrameBuffer->unbind(); }
}

void ShaderCompatScreen::describe(std::ostream &out, const std::string &indent) const
{
    out << "[ ShaderCompatScreen: \n"
        << indent << "\tShader Program = ";
    mProgram.describe(out, indent + "\t\t");
    out << "\n" << indent << "\tShader Uniforms = ";
    mUniforms.describe(out, indent + "\t\t");
    out << "\n" << indent << " ]";
}

} // namespace treerndr
