/**
 * @author Tomas Polasek, David Hrusa
 * @date 5.20.2020
 * @version 1.0
 * @brief Compatibility header for blur shader.
 */

#include "TreeShaderCompatBlur.h"

#include "TreeShaderCompatBlob.h"

namespace treerndr
{

ShaderCompatBlur::ShaderCompatBlur()
{ initializeUniforms(); reloadShaders(); }

void ShaderCompatBlur::initializeUniforms()
{
    mUniforms = UniformSet(
        false,
        uInput, uBlurHorizontal
    );
    mFrameBuffer = FrameBuffer::instantiate();
}

void ShaderCompatBlur::reloadShaders()
{
    mProgram = ShaderProgramFactory::renderProgram()
        .addVertexFallback("shaders/blur_vs.glsl", glsl::BLUR_VS_GLSL)
        .addFragmentFallback("shaders/blur_fs.glsl", glsl::BLUR_FS_GLSL)
        .checkUniforms(mUniforms.getUniformIdentifiers())
        .finalize();
}

void ShaderCompatBlur::activate() const
{ mProgram.use(); }

void ShaderCompatBlur::setUniforms() const
{ mUniforms.setUniforms(mProgram); }

void ShaderCompatBlur::blurTexture(const TextureBuffer::Ptr &input, const TextureBuffer::Ptr &output,
    bool horizontal, treeutil::FrameBufferAttachmentType type)
{
    // Prepare frame-buffer with output redirected to provided texture.
    mFrameBuffer->clearAttachments();
    mFrameBuffer->addAttachment(output, type);
    mFrameBuffer->bind();

    uInput = input;
    uBlurHorizontal = horizontal;
    drawFullScreen();
}

void ShaderCompatBlur::drawFullScreen()
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

void ShaderCompatBlur::describe(std::ostream &out, const std::string &indent) const
{
    out << "[ ShaderCompatBlur: \n"
        << indent << "\tShader Program = ";
    mProgram.describe(out, indent + "\t\t");
    out << "\n" << indent << "\tShader Uniforms = ";
    mUniforms.describe(out, indent + "\t\t");
    out << "\n" << indent << " ]";
}

} // namespace treerndr
