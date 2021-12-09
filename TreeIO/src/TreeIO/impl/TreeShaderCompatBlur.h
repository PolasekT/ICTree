/**
 * @author Tomas Polasek, David Hrusa
 * @date 5.20.2020
 * @version 1.0
 * @brief Compatibility header for blur shader.
 */

#include "TreeUtils.h"
#include "TreeGLUtils.h"

#include "TreeShader.h"
#include "TreeTextureBuffer.h"
#include "TreeFrameBuffer.h"

#ifndef TREE_SHADER_COMPAT_BLUR_H
#define TREE_SHADER_COMPAT_BLUR_H

namespace treerndr
{

/// @brief Compatibility helper for blur shaders.
class ShaderCompatBlur : public treeutil::PointerWrapper<ShaderCompatBlur>
{
public:
    /// @brief Initialize the uniform storage and compile shaders.
    ShaderCompatBlur();

    /// @brief Initialize uniform storage.
    void initializeUniforms();

    /// @brief Reload all of the shaders and recompile shader programs.
    void reloadShaders();

    /// @brief Activate the shader program.
    void activate() const;

    /// @brief Set current uniform values for given program.
    void setUniforms() const;

    /// @brief Blur input texture and store results in the output texture.
    void blurTexture(const TextureBuffer::Ptr &input, const TextureBuffer::Ptr &output,
        bool horizontal, treeutil::FrameBufferAttachmentType type = treeutil::FrameBufferAttachmentType::Color);

    /// @brief Set uniforms, activate shader and run it with full-screen triangle.
    void drawFullScreen();

    /// @brief Print description of this shader compatibility module.
    void describe(std::ostream &out, const std::string &indent = "") const;

    // Uniforms:
    uniform(sampler2D, uInput, "Input texture to be blurred.");
    uniform(int, uBlurHorizontal, "Whether to perform vertical (0) or horizontal (1) blurring pass.");
private:
    /// Set of active uniforms.
    UniformSet mUniforms{ };
    /// Shader program containing photo shaders.
    ShaderProgram mProgram{ };
    /// Frame-buffer used for blurring of textures.
    FrameBuffer::Ptr mFrameBuffer{ };
protected:
}; // struct ShaderCompatBlur

} // namespace treerndr

/// @brief Print shader compatibility description.
inline std::ostream &operator<<(std::ostream &out, const treerndr::ShaderCompatBlur &shaderCompat);

// Template implementation begin.

namespace treerndr
{

} // namespace treerndr

inline std::ostream &operator<<(std::ostream &out, const treerndr::ShaderCompatBlur &shaderCompat)
{ shaderCompat.describe(out); return out; }

// Template implementation end.

#endif // TREE_SHADER_COMPAT_BLUR_H
