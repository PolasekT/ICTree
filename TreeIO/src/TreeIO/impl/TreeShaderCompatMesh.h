/**
 * @author Tomas Polasek, David Hrusa
 * @date 5.19.2020
 * @version 1.0
 * @brief Compatibility header for mesh rendering shaders.
 */

#include "TreeUtils.h"
#include "TreeGLUtils.h"

#include "TreeShader.h"
#include "TreeTextureBuffer.h"
#include "TreeBuffer.h"
#include "TreeRenderSystem.h"

#ifndef TREE_SHADER_COMPAT_MESH_H
#define TREE_SHADER_COMPAT_MESH_H

namespace treerndr
{

/// @brief Compatibility helper for mesh shaders.
class ShaderCompatMesh : public treeutil::PointerWrapper<ShaderCompatMesh>
{
public:
    /// @brief Initialize the uniform storage and compile shaders.
    ShaderCompatMesh();

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
    uniform(vec4, uColorOverride, "Color override value.");
    uniform(bool, uDoOverrideColor, "Set to true to force use color override.");
    uniform(bool, uUseColorStorage, "Set to true to use override colors from override storage buffer.");
    uniform(mat4, uModel, "Model matrix.");
    uniform(mat4, uMVP, "Model-view-projection matrix.");
    uniform(bool, uShaded, "Enable shading calculation (true) or just use flat colors (false).");
    uniform(sampler2D, uShadowMap, "Shadow map used when uShaded == true.", /*unit = */ 0u);
    uniform(vec4, uShadowKernelSpec, "Specification of the soft shadow kernel - x = kernel size, y = sampling factor, z = kernel strength, w = bias.");
    uniform(mat4, uLightViewProjection, "Matrix used to transform world-space location to the shadow map.");
    uniform(int, uModalitySelector, "Selector switch used for modality selection.", 0);
    uniform(float, uFar, "Location of the far plane.", 100.0f);
    uniform(vec4, uCamera, "Information about the camera - near plane, far plane and fov. Last element is unused.");
    uniform(buffer, bOverride, "Override buffer", /*binding = */ 0u);
private:
    /// Set of active uniforms.
    UniformSet mUniforms{ };
    /// Shader program containing photo shaders.
    ShaderProgram mProgram{ };
protected:
}; // struct ShaderCompatMesh

} // namespace treerndr

/// @brief Print shader compatibility description.
inline std::ostream &operator<<(std::ostream &out, const treerndr::ShaderCompatMesh &shaderCompat);

// Template implementation begin.

namespace treerndr
{

} // namespace treerndr

inline std::ostream &operator<<(std::ostream &out, const treerndr::ShaderCompatMesh &shaderCompat)
{ shaderCompat.describe(out); return out; }

// Template implementation end.

#endif // TREE_SHADER_COMPAT_MESH_H
