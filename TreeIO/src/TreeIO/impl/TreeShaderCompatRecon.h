/**
 * @author Tomas Polasek
 * @date 4.16.2020
 * @version 1.0
 * @brief Compatibility header for reconstruction shaders.
 */

#include "TreeUtils.h"
#include "TreeGLUtils.h"

#include "TreeShader.h"
#include "TreeTextureBuffer.h"
#include "TreeBuffer.h"
#include "TreeRenderSystem.h"
#include "TreeReconstruction.h"

#ifndef TREE_SHADER_COMPAT_RECON_H
#define TREE_SHADER_COMPAT_RECON_H

namespace treerndr
{

/// @brief Compatibility helper for reconstruction shaders.
class ShaderCompatRecon : public treeutil::PointerWrapper<ShaderCompatRecon>
{
public:
    /// @brief Initialize the uniform storage and compile shaders.
    ShaderCompatRecon();

    /// @brief Initialize uniform storage.
    void initializeUniforms();

    /// @brief Reload all of the shaders and recompile shader programs.
    void reloadShaders();

    /// @brief Activate the shader program.
    void activate() const;

    /// @brief Set current uniform values for given program.
    void setUniforms() const;

    /// @brief Render reconstruction of given tree using rendering context.
    void render(const MeshInstancePtr &instancePtr,
        const treeutil::TreeReconstruction::Ptr &reconPtr,
        const RenderContext &ctx, bool transformFeedback = false);

    /**
     * @brief Save reconstruction. Output buffer is automatically created should be
     *   re-used for multiple uses. Resulting buffer will contain glm::vec4. Returns
     *   triangles generated.
     */
    std::size_t transformFeedback(const MeshInstancePtr &instancePtr,
        const treeutil::TreeReconstruction::Ptr &reconPtr,
        const RenderContext &ctx, Buffer::Ptr &outputBuffer);

    /// @brief Automatically generate model using transform feedback and return the resulting geometry.
    RawGeometryPtr generateModel(const MeshInstancePtr &instancePtr,
        const treeutil::TreeReconstruction::Ptr &reconPtr,
        const RenderContext &ctx);

    /// @brief Print description of this shader compatibility module.
    void describe(std::ostream &out, const std::string &indent = "") const;

    // Uniforms:
    uniform(mat4, uModel, "Model matrix of the displayed mesh.");
    uniform(mat4, uView, "View matrix of the current camera");
    uniform(mat4, uProjection, "Projection matrix of the current camera");
    uniform(mat4, uVP, "View-projection matrix of the current camera.");
    uniform(mat4, uMVP, "Model-view-projection matrix of displayed mesh.");
    uniform(mat4, uNT, "Inverse transpose of MV matrix. Used for transforming normals.");
    uniform(mat4, uLightView, "Light view projection matrix.");
    uniform(vec3, uCameraPos, "Position of the camera.");
    uniform(vec3, uLightPos, "Position of the light.");
    uniform(vec3, uMeshScale, "Scale of the mesh.");
    uniform(float, uTessellationMultiplier, "Multiplier used to calculate tessellation level. Default value = 500.0f .", 500.0f);
    uniform(sampler2D, uShadowMap, "Shadow map from the position of the light.", 0u);
    uniform(int, uApplyShadow, "Should the shadow be displayed?");
    uniform(int, uPhotoShading, "Perform basic shading (0) or photo-mode shading (1)?");
    uniform(vec4, uShadowKernelSpec, "Specification of the soft shadow kernel - x = kernel size, y = sampling factor, z = kernel strength, w = bias.");
    uniform(vec3, uForegroundColor, "Color used for foreground areas.");
    uniform(vec3, uBackgroundColor, "Color used for background areas.");
    uniform(float, uOpaqueness, "How opaque should the reconstruction be. 1.0f for fully opaque and 0.0f for fully transparent.");
    uniform(float, uMaxBranchRadius, "Maximum branch radius.");
    uniform(float, uMaxDistanceFromRoot, "Maximum distance from the root node.");
    uniform(float, uBranchTension, "Tension of the interpolated branch curve.");
    uniform(float, uBranchBias, "Bias of the interpolated branch curve.");
    uniform(float, uBranchWidthMultiplier, "Scaler used for branch width multiplication.");
    uniform(int, uModalitySelector, "Selector switch used for modality selection.", 0);
    uniform(vec4, uCamera, "Information about the camera - near plane, far plane and fov. Last element is unused.");
    uniform(buffer, bData, "Buffer containing the draw data used for adjacency information.", /*binding = */ 0u);
private:
    /// Primitive generated in geometry shader. Used for transform feedback.
    static constexpr auto PRIMITIVE_USED{ GL_TRIANGLES };
    /// Number of vertices per one primitive generated in the geometry shader. Used for transform feedback.
    static constexpr auto VERTICES_PER_PRIMITIVE{ 3u };

    /// Set of active uniforms.
    UniformSet mUniforms{ };
    /// Shader program containing photo shaders.
    ShaderProgram mProgram{ };
    /// Buffer used for automatic transform feedback model saving.
    Buffer::Ptr mTransformFeedbackBuffer{ };
protected:
}; // struct ShaderCompatRecon

} // namespace treerndr

/// @brief Print shader compatibility description.
inline std::ostream &operator<<(std::ostream &out, const treerndr::ShaderCompatRecon &shaderCompat);

// Template implementation begin.

namespace treerndr
{

} // namespace treerndr

inline std::ostream &operator<<(std::ostream &out, const treerndr::ShaderCompatRecon &shaderCompat)
{ shaderCompat.describe(out); return out; }

// Template implementation end.

#endif // TREE_SHADER_COMPAT_RECON_H
