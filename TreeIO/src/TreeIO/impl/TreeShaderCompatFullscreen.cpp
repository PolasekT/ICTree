/**
 * @author Tomas Polasek, David Hrusa
 * @date 5.20.2020
 * @version 1.0
 * @brief Compatibility header for blur shader.
 */

#include "TreeShaderCompatFullscreen.h"

#include "TreeShaderCompatBlob.h"

namespace treerndr
{

ShaderCompatFullscreen::ShaderCompatFullscreen()
{ initializeUniforms(); reloadShaders(); }

void ShaderCompatFullscreen::initializeUniforms()
{
    mUniforms = UniformSet(
        false,
        uInput
    );
}

void ShaderCompatFullscreen::reloadShaders()
{
    mProgram = ShaderProgramFactory::renderProgram()
        .addVertexFallback("shaders/fullscreen_vs.glsl", glsl::FULLSCREEN_VS_GLSL)
        .addFragmentFallback("shaders/fullscreen_fs.glsl", glsl::FULLSCREEN_FS_GLSL)
        .checkUniforms(mUniforms.getUniformIdentifiers())
        .finalize();
}

void ShaderCompatFullscreen::activate() const
{ mProgram.use(); }

void ShaderCompatFullscreen::setUniforms() const
{ mUniforms.setUniforms(mProgram); }

void ShaderCompatFullscreen::drawTexture(const TextureBuffer::Ptr &input)
{ uInput = input; drawFullScreen(); }

void ShaderCompatFullscreen::drawFullScreen()
{
    activate();
    setUniforms();

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_ALWAYS);
    glDisable(GL_CULL_FACE);

    glBindBuffer(GL_ARRAY_BUFFER, 0u);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

void ShaderCompatFullscreen::describe(std::ostream &out, const std::string &indent) const
{
    out << "[ ShaderCompatFullscreen: \n"
        << indent << "\tShader Program = ";
    mProgram.describe(out, indent + "\t\t");
    out << "\n" << indent << "\tShader Uniforms = ";
    mUniforms.describe(out, indent + "\t\t");
    out << "\n" << indent << " ]";
}

} // namespace treerndr
