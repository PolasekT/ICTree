/**
 * @author Tomas Polasek, David Hrusa
 * @date 4.15.2020
 * @version 1.0
 * @brief Concrete renderer systems used for rendering in photo modes.
 */

#include "TreeRenderSystem.h"

#include "TreeShader.h"
#include "TreeBuffer.h"
#include "TreeTextureBuffer.h"
#include "TreeFrameBuffer.h"
#include "TreeRuntimeMetaData.h"

#ifndef TREE_RENDER_SYSTEM_PHOTO_H
#define TREE_RENDER_SYSTEM_PHOTO_H

namespace treerndr
{

// Forward declaration of internal implementation:
namespace impl
{ struct RenderSystemPhotoImpl; }

/// @brief Renderer used for photo modes.
class RenderSystemPhoto : public RenderSystem
{
public:
    /// Name of this renderer.
    static constexpr auto RENDERER_NAME{ "Photo" };

    /// @brief Initialize the render system.
    RenderSystemPhoto();
    /// @brief Clean-up and destroy.
    virtual ~RenderSystemPhoto();

    /// @brief Initialize the renderer and create any required resources.
    virtual void initialize() override final;
    /// @brief Render current configuration of the scene.
    virtual void render(treescene::CameraState &camera,
        treescene::TreeScene &scene) override final;
    /// @brief Reload all shaders for this renderer.
    virtual void reloadShaders() override final;

    /// @brief Access visualization parameters.
    virtual VisualizationParameters &parameters() override final;
private:
    /// Internal implementation.
    std::shared_ptr<impl::RenderSystemPhotoImpl> mImpl{ };
protected:
}; // class RenderSystemPhoto

} // namespace treerndr

// Template implementation begin.

namespace treerndr
{

} // namespace treerndr

// Template implementation end.

#endif // TREE_RENDER_SYSTEM_PHOTO_H
