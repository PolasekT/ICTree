/**
 * @author Tomas Polasek, David Hrusa
 * @date 5.7.2020
 * @version 1.0
 * @brief Tree reconstruction utilities.
 */

#include <vector>

#include "TreeUtils.h"

#ifndef TREE_RECONSTRUCTION_H
#define TREE_RECONSTRUCTION_H

namespace treeutil
{

// Forward declaration.
class TreeChains;

/// @brief Helper class used for generating ArrayTree reconstructions.
class TreeReconstruction : public treeutil::PointerWrapper<TreeReconstruction>
{
public:
    /// @brief Container for data required by one reconstructed vertex.
    struct VertexData
    {
        /// Total number of float elements this structure takes in size.
        static constexpr std::size_t FLOAT_ELEMENTS{ 4u + 4u + 4u + 4u + 4u };

        /// Position of the vertex in world space, w is unused.
        glm::vec4 position{ };
        /// Normal of the vertex in world space, w is set to distance from the tree root.
        glm::vec4 normal{ };
        /// Vector parallel with the branch in world space, w is set to the radius of the branch.
        glm::vec4 parallel{ };
        /// Tangent of the branch in world space, w is unused.
        glm::vec4 tangent{ };
        /// Adjacency indices for this vertex, x = this idx, y = parent idx, z = child idx, w is unused. 0 as invalid idx.
        glm::uvec4 adjacency{ };
    }; // struct VertexData

    /// @brief Reconstruction visualization parameters.
    struct VisualParameters
    {
        /// Scaling of the reconstruction without adjusting branch thickness ratios.
        float meshScale{ 1.0f };
        /// Multiplication constant used in determining tessellation level of the generated mesh.
        float tessellationMultiplier{ 1.0f };
        /// Color used to represent foreground.
        glm::vec3 foregroundColor{ 1.0f, 1.0f, 1.0f };
        /// Color used to represent background.
        glm::vec3 backgroundColor{ 0.0f, 0.0f, 0.0f };
        /// How opaque should the reconstruction be. 1.0f for fully opaque and 0.0f for fully transparent.
        float opaqueness{ 0.75f };
        /// Tension of branches connecting skeleton nodes.
        float branchTension{ 0.0f };
        /// Bias of branches connecting skeleton nodes.
        float branchBias{ 0.0f };
        /// Multiplier used for branch thickness.
        float branchWidthMultiplier{ 1.0f };
    }; // struct VisualParameters

    /// @brief Initialize helper structures.
    TreeReconstruction();
    /// @brief Initialize helper structures and reconstruct given tree.
    TreeReconstruction(treeio::ArrayTree &tree, bool copyResults);
    /// @brief Cleanup and destroy.
    ~TreeReconstruction();

    // Builders:

    /// @brief Reconstruct given tree and fill internal data structures. Optionally copy resulting widths to the input.
    bool reconstructTree(treeio::ArrayTree &tree, bool copyResults);

    // Accessors:

    /// @brief Get vertex data of the current reconstruction.
    const std::vector<VertexData> &vertexData() const;
    /// @brief Access the visualization parameters of this reconstruction.
    VisualParameters &parameters();
    /// @brief Access the visualization parameters of this reconstruction.
    const VisualParameters &parameters() const;
    /// @brief Get maximum value of distance from root.
    float maxDistanceFromRoot() const;
    /// @brief Get maximum value of branch radius.
    float maxBranchRadius() const;

    /// Minimal branch width before forced recalculation is triggered.
    static constexpr auto MIN_BRANCH_THICKNESS{ 0.00001f };
private:
    /// @brief Prepare input tree for processing.
    WrapperPtrT<TreeChains> prepareTree(const treeio::ArrayTree &tree) const;
    /// @brief Calculate missing radii for given tree.
    void calculateRadii(const WrapperPtrT<TreeChains> &tree) const;
    /// @brief Calculate vertex data for provided tree.
    std::vector<VertexData> prepareVertexData(const WrapperPtrT<TreeChains> &tree,
        float &maxDistanceFromRoot, float &maxBranchRadius) const;
    /// @brief Calculate vertex data for given node. Does NOT fill in adjacency information!
    VertexData prepareNodeData(const WrapperPtrT<TreeChains> &tree, std::size_t nodeId,
        float &maxDistanceFromRoot, float &maxBranchRadius) const;

    /// @brief Copy calculated values from inputTree into the outputTree.
    void copyResultsTo(const WrapperPtrT<TreeChains> &inputTree, treeio::ArrayTree &outputTree);

    /// Vertex data of the currently reconstructed tree.
    std::vector<VertexData> mVertexData{ };
    /// Parameters used for reconstruction visualization.
    VisualParameters mParameters{ };
    /// Maximum distance from root.
    float mMaxDistanceFromRoot{ 0.0f };
    /// Maximum branch radius.
    float mMaxBranchRadius{ 0.0f };
protected:
}; // class TreeReconstruction

} // namespace treeutil

#endif // TREE_RECONSTRUCTION_H
