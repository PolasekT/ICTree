/**
 * @author Tomas Polasek
 * @date 7.14.2020
 * @version 1.0
 * @brief Support for displaying different modalities.
 */

#include "TreeUtils.h"
#include "TreeGLUtils.h"

#ifndef TREE_RENDERER_MODALITY_H
#define TREE_RENDERER_MODALITY_H

namespace treerndr
{

/// @brief Modalities displayable by the rendering shaders.
enum class DisplayModality
{
    /// @brief Display completely shaded output.
    Shaded,
    /// @brief Display albedo only.
    Albedo,
    /// @brief Display light only.
    Light,
    /// @brief Display shadows.
    Shadow,
    /// @brief Display normals.
    Normal,
    /// @brief Display depths.
    Depth,
    /// @brief Stopper used to determine modality count.
    Sentinel,
}; // enum class DisplayModality

/// @brief Support for displaying different modalities.
class RendererModality
{
public:
    /// Total number of modalities.
    static constexpr auto MODALITY_COUNT{ static_cast<std::size_t>(DisplayModality::Sentinel) };

    /// @brief Initialize modality rendering for given display modality.
    RendererModality(DisplayModality modality = DisplayModality::Shaded);
    /// @brief Initialize modality rendering for given display modality.
    RendererModality(const std::string &name = "Shaded");
    /// @brief Initialize modality rendering for given display modality.
    RendererModality(std::size_t idx = 0u);

    /// @brief Clean-up and destroy.
    ~RendererModality() = default;

    /// @brief Get current modality.
    DisplayModality modality() const;
    /// @brief Convert modality to its name.
    std::string name() const;
    /// @brief Convert modality to its index.
    std::size_t idx() const;

    /// @brief Set internal modality.
    void fromModality(DisplayModality modality);
    /// @brief Set internal modality by its name.
    void fromName(const std::string &name);
    /// @brief Set internal modality by its index.
    void fromIdx(std::size_t idx);

    /// @brief Get clear color for current modality.
    treeutil::Color getClearColor() const;
private:
    /// Currently used modality.
    DisplayModality mModality{ DisplayModality::Shaded };
protected:
}; // class RendererModality

} // namespace treerndr

#endif // TREE_RENDERER_MODALITY_H
