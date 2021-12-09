/**
 * @author Tomas Polasek
 * @date 14.10.2020
 * @version 1.0
 * @brief Compatibility header for fullscreen shader.
 */

#include "TreeUtils.h"
#include "TreeGLUtils.h"

#include "TreeShader.h"
#include "TreeTextureBuffer.h"
#include "TreeFrameBuffer.h"

#ifndef TREE_SHADER_COMPAT_FULLSCREEN_H
#define TREE_SHADER_COMPAT_FULLSCREEN_H

namespace treerndr
{

/// @brief Compatibility helper for fullscreen shaders.
class ShaderCompatFullscreen : public treeutil::PointerWrapper<ShaderCompatFullscreen>
{
public:
    /// @brief Initialize the uniform storage and compile shaders.
    ShaderCompatFullscreen();

    /// @brief Initialize uniform storage.
    void initializeUniforms();

    /// @brief Reload all of the shaders and recompile shader programs.
    void reloadShaders();

    /// @brief Activate the shader program.
    void activate() const;

    /// @brief Set current uniform values for given program.
    void setUniforms() const;

    /// @brief Draw input texture using a fullscreen quad.
    void drawTexture(const TextureBuffer::Ptr &input);

    /// @brief Set uniforms, activate shader and run it with full-screen triangle.
    void drawFullScreen();

    /// @brief Print description of this shader compatibility module.
    void describe(std::ostream &out, const std::string &indent = "") const;

    // Uniforms:
    uniform(sampler2D, uInput, "Input texture to draw.");
private:
    /// Set of active uniforms.
    UniformSet mUniforms{ };
    /// Shader program containing photo shaders.
    ShaderProgram mProgram{ };
protected:
}; // struct ShaderCompatFullscreen

} // namespace treerndr

/// @brief Print shader compatibility description.
inline std::ostream &operator<<(std::ostream &out, const treerndr::ShaderCompatFullscreen &shaderCompat);

// Template implementation begin.

namespace treerndr
{

} // namespace treerndr

inline std::ostream &operator<<(std::ostream &out, const treerndr::ShaderCompatFullscreen &shaderCompat)
{ shaderCompat.describe(out); return out; }

// Template implementation end.

#endif // TREE_SHADER_COMPAT_FULLSCREEN_H
