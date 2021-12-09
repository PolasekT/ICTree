/**
 * @author Tomas Polasek
 * @date 8.14.2021
 * @version 1.0
 * @brief Helper class for simplified tree rendering.
 */

#ifndef ICTREE_TREE_RENDER_HELPER_H
#define ICTREE_TREE_RENDER_HELPER_H

#include "TreeUtils.h"

namespace treerndr
{

// Forward declaration.
namespace impl
{ class RenderHelperImpl; }

/// @brief Wrapper around rendering settings.
struct RenderConfig
{
    /// Path to save the images to.
    std::string outputPath{ "./" };
    /**
     * Base name for the saved files. For each view several
     *   modalities will be produced. Their names use the following naming
     *   scheme: <BASE>_<VIEW_IDX>_<MODALITY>.png
     * Additionally, meta-data file will be created, sharing the same name
     *   but using ".json" extension instead.
    */
    std::string baseName{ "tree" };
    /// Width of the rendered image.
    std::size_t width{ 1024u };
    /// Height of the rendered image.
    std::size_t height{ 1024u };
    /// Number of samples to render per pixel.
    std::size_t samples{ 4u };
    /// Number of views to render.
    std::size_t viewCount{ 5u };
    /// Distance of the camera from the origin.
    float cameraDistance{ 5.0f };
    /// Height of the camera from the ground plane.
    float cameraHeight{ 1.8f };
    /// Normalize the tree size to treeScale?
    bool treeNormalize{ false };
    /// Scale to normalize the tree to, if treeNormalize is enabled.
    float treeScale{ 1.0f };
}; // struct RenderConfig

/// @brief Wrapper around dithered rendering settings.
struct DitherConfig
{
    /// Path to save the images to.
    std::string outputPath{ "./" };
    /// Seed used for repeatable dithered view generation.
    int seed{ 0 };
    /// Number of dithered variants to generate.
    std::size_t ditherCount{ 16 };

    /// Variance of the camera distance.
    float camDistanceVar{ 0.0f };

    /// Strength of yaw dithering.
    float camYawDither{ 0.0f };
    /// Minimal yaw dithering.
    float camYawDitherLow{ -1.0f };
    /// Maximal yaw dithering.
    float camYawDitherHigh{ 1.0f };

    /// Strength of pitch dithering.
    float camPitchDither{ 0.0f };
    /// Minimal roll dithering.
    float camPitchDitherLow{ -1.0f };
    /// Maximal pitch dithering.
    float camPitchDitherHigh{ 1.0f };

    /// Strength of roll dithering.
    float camRollDither{ 0.0f };
    /// Minimal roll dithering.
    float camRollDitherLow{ -1.0f };
    /// Maximal roll dithering.
    float camRollDitherHigh{ 1.0f };
}; // struct DitherConfig

/// @brief Helper class for simplified tree rendering.
class RenderHelper
{
public:

    /// @brief Initialize the helper.
    RenderHelper();
    /// @brief Free resources and destruct.
    ~RenderHelper();

    /**
     * @brief Render a tree from several view-points and save to given path.
     * @param tree Tree to render.
     * @param config Settings used for the rendering.
     */
    void renderTree(const treeio::ArrayTree &tree, const RenderConfig &config);

    /**
     * @brief Render dithered views for given tree.
     * @param tree Tree to render.
     * @param config Settings used for the rendering.
     * @param dither Dithering configuration used.
     */
    void renderDitheredTree(const treeio::ArrayTree &tree,
        const RenderConfig &config, const DitherConfig &dither);
private:
    /// Internal implementation.
    std::shared_ptr<impl::RenderHelperImpl> mImpl{ };
protected:
}; // class RenderHelper

} // namespace treerndr

#endif //ICTREE_TREE_RENDER_HELPER_H
