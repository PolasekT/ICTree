/**
 * @author Tomas Polasek, David Hrusa
 * @date 4.14.2020
 * @version 1.0
 * @brief Base renderer system used for rendering the current scene.
 */

#include <vector>
#include <map>
#include <memory>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtx/euler_angles.hpp>

#include "TreeUtils.h"
#include "TreeGLUtils.h"
#include "TreeCamera.h"
#include "TreeShader.h"
#include "TreeRendererModality.h"

#ifndef TREE_RENDER_SYSTEM_H
#define TREE_RENDER_SYSTEM_H

// Forward declarations:
namespace treescene
{

class TreeScene;

} // namespace treescene

namespace treerndr
{

// Forward declarations:
class TreeRenderer;
class Buffer;
using BufferPtr = treeutil::WrapperPtrT<FrameBuffer>;
class TextureBuffer;
using TextureBufferPtr = treeutil::WrapperPtrT<TextureBuffer>;
class FrameBuffer;
using FrameBufferPtr = treeutil::WrapperPtrT<FrameBuffer>;
struct RawGeometry;
using RawGeometryPtr = treeutil::WrapperPtrT<RawGeometry>;
struct VaoPack;
using VaoPackPtr = treeutil::WrapperPtrT<VaoPack>;
struct MeshInstance;
using MeshInstancePtr = treeutil::WrapperPtrT<MeshInstance>;
using InstanceList = std::vector<MeshInstancePtr>;
struct Texture;
using TexturePtr = treeutil::WrapperPtrT<Texture>;
struct BufferBundle;
using BufferBundlePtr = treeutil::WrapperPtrT<BufferBundle>;
struct FrameBufferObject;
using FrameBufferObjectPtr = treeutil::WrapperPtrT<FrameBufferObject>;
struct Light;
using LightPtr = treeutil::WrapperPtrT<Light>;

/// @brief Structure containing current state of rendering and the common parameters.
struct RenderContext
{
    /// Width of the output viewport in pixels.
    std::size_t frameBufferWidth{ };
    /// Height of the output viewport in pixels.
    std::size_t frameBufferHeight{ };

    /// Position of the current camera.
    glm::vec3 cameraPosition{ };
    /// Position of the primary light.
    glm::vec3 lightPosition{ };

    /// Distance of the camera near plane.
    float cameraNear{ };
    /// Distance of the camera far plane.
    float cameraFar{ };
    /// Field of view of the camera - orthographic or perspective.
    float cameraFov{ };

    /// View matrix for the render camera.
    glm::mat4 view{ };
    /// Projection matrix for the render camera.
    glm::mat4 projection{ };
    /// View-projection matrix for the render camera.
    glm::mat4 viewProjection{ };
    /// View-projection matrix of the light corresponding to the inputShadowMap;
    glm::mat4 lightViewProjection{ };

    /// Hint that shadows are being draw and all shading may be disabled.
    bool renderingShadows{ false };
    /// Size of the kernel in one dimension - e.g. (2.0f, 2.0f) results in 5x5 filter.
    float shadowKernelSize{ 2.0f };
    /// Multiplier for shadow map resolution used in calculation of one pixel increments.
    float shadowSamplingFactor{ 1.3f };
    /// Strength of displayed shadows.
    float shadowStrength{ 0.5f };
    /// Bias used for reducing shadow artifacts.
    float shadowBias{ 0.001f };
    /// Are we rendering simplified (false) or realistic (true) image?
    bool renderingPhoto{ false };

    /// Modality selector for rendering different modalities.
    RendererModality modality{ DisplayModality::Shaded };

    /// Input frame-buffer used for transform operations.
    treeutil::WrapperPtrT<FrameBuffer> inputFrameBuffer{ };
    /// Output frame-buffer used for transform operations and rendering.
    treeutil::WrapperPtrT<FrameBuffer> outputFrameBuffer{ };

    /// Input shadow-map texture used when calculating shading.
    treeutil::WrapperPtrT<TextureBuffer> inputShadowMap{ };
    /// output shadow-map texture used as output when rendering shadows.
    treeutil::WrapperPtrT<TextureBuffer> outputShadowMap{ };
}; // struct RenderContext

/// @brief Base class for all TreeRenderSystems, which can be used to render the current scene.
class RenderSystem : public treeutil::PointerWrapper<RenderSystem>
{
public:
    /// @brief Options for visualization using this renderer.
    struct VisualizationParameters
    {
        /// Display shadows?
        bool useShadows{ true };
        /// Modality selector for rendering different modalities.
        RendererModality modality{ DisplayModality::Shaded };
        /// When set, this frame-buffer will be used as output instead of the current renderer FB.
        FrameBufferPtr outputFrameBufferOverride{ nullptr };
    }; // struct VisualizationParameters

    /// @brief Initialize the render system, giving it a unique name.
    RenderSystem(const std::string &name);
    /// @brief Clean-up and destroy.
    virtual ~RenderSystem();

    /// @brief Initialize the renderer and create any required resources.
    virtual void initialize() = 0;

    /// @brief Render current configuration of the scene.
    virtual void render(treescene::CameraState &camera,
        treescene::TreeScene &scene) = 0;
    /// @brief Reload all shaders for this renderer.
    virtual void reloadShaders() = 0;

    /// @brief Print description of this renderer.
    virtual void describe(std::ostream &out, const std::string &indent = "") const;

    /// @brief Access visualization parameters.
    virtual VisualizationParameters &parameters() = 0;

    /// @brief Get unique identifier of this renderer.
    const std::string &identifier() const;
private:
    /// Unique name of this rendering system.
    std::string mName{ };
protected:
}; // class RenderSystem

} // namespace treerndr

/// @brief Print renderer description.
inline std::ostream &operator<<(std::ostream &out, const treerndr::RenderSystem &renderer);

// Template implementation begin.

namespace treerndr
{

} // namespace treerndr

inline std::ostream &operator<<(std::ostream &out, const treerndr::RenderSystem &renderer)
{ renderer.describe(out); return out; }

// Template implementation end.

#endif // TREE_RENDER_SYSTEM_H
