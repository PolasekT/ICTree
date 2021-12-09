/**
 * @author Tomas Polasek, David Hrusa
 * @date 5.19.2020
 * @version 1.0
 * @brief Compatibility header for screen rendering shaders.
 */

#include "TreeUtils.h"
#include "TreeGLUtils.h"

#include "TreeShader.h"
#include "TreeTextureBuffer.h"
#include "TreeBuffer.h"
#include "TreeRenderSystem.h"

#ifndef TREE_SHADER_COMPAT_SCREEN_H
#define TREE_SHADER_COMPAT_SCREEN_H

namespace treerndr
{

/// @brief Compatibility helper for screen shaders.
class ShaderCompatScreen : public treeutil::PointerWrapper<ShaderCompatScreen>
{
public:
    /// @brief Initialize the uniform storage and compile shaders.
    ShaderCompatScreen();

    /// @brief Initialize uniform storage.
    void initializeUniforms();

    /// @brief Reload all of the shaders and recompile shader programs.
    void reloadShaders();

    /// @brief Activate the shader program.
    void activate() const;

    /// @brief Set current uniform values for given program.
    void setUniforms() const;

    /// @brief Render given instance using settings in provided rendering context.
    void render(const MeshInstancePtr &instancePtr,
        const RenderContext &ctx, const treescene::CameraState &camera);

    /// @brief Print description of this shader compatibility module.
    void describe(std::ostream &out, const std::string &indent = "") const;

    // Uniforms:
    uniform(mat4, uMVP, "Model-View-Projection matrix.");
    uniform(bool, uTextured, "Set to true to interpret color as: RG -> UV and B -> texture ID.");
    uniformArr(sampler2D, 10, uTextures, "List of textures indexed by the blue color channel.");
private:
    /// Set of active uniforms.
    UniformSet mUniforms{ };
    /// Shader program containing photo shaders.
    ShaderProgram mProgram{ };
protected:
}; // struct ShaderCompatScreen

} // namespace treerndr

/// @brief Print shader compatibility description.
inline std::ostream &operator<<(std::ostream &out, const treerndr::ShaderCompatScreen &shaderCompat);

// Template implementation begin.

namespace treerndr
{

} // namespace treerndr

inline std::ostream &operator<<(std::ostream &out, const treerndr::ShaderCompatScreen &shaderCompat)
{ shaderCompat.describe(out); return out; }

// Template implementation end.

#endif // TREE_SHADER_COMPAT_SCREEN_H
