/**
 * @author Tomas Polasek, David Hrusa
 * @date 1.15.2020
 * @version 1.0
 * @brief Scene representation for the main viewport.
 */

#ifndef TREE_SCENE_H
#define TREE_SCENE_H

#include "TreeUtils.h"
#include "TreeSceneInstanceNames.h"
#include "TreeRenderer.h"
#include "TreeRuntimeMetaData.h"

// Forward declarations:

namespace treescene
{

/// @brief Scene representation for the main viewport.
class TreeScene : public treeutil::PointerWrapper<TreeScene>
{
public:
    /// Name of the object containing tree skeleton points.
    static constexpr const char *INST_POINTS_NAME{ instances::POINTS_NAME };
    /// Name of the object containing tree skeleton segments.
    static constexpr const char *INST_SEGMENTS_NAME{ instances::SEGMENTS_NAME };
    /// Name of the object containing the reference model.
    static constexpr const char *INST_REFERENCE_NAME{ instances::REFERENCE_NAME };
    /// Name of the object containing tree skeleton segments.
    static constexpr const char* INST_RECONSTRUCTION_NAME{ instances::RECONSTRUCTION_NAME };
    /// Name of the object containing the floor grid.
    static constexpr const char *INST_GRID_NAME{ instances::GRID_NAME };
    /// Name of the object containing the floor ground plane.
    static constexpr const char *INST_PLANE_NAME{ instances::PLANE_NAME };
    /// Name of the object representing the light.
    static constexpr const char *INST_LIGHT_NAME{ instances::LIGHT_NAME };
    /// Default position of the scene light.
    static constexpr glm::vec3 LIGHT_DEFAULT_POS{ -4.5f, 10.0f, -4.5f };
    /// Maximum number of nodes to run reconstruction for.
    static constexpr std::size_t RECONSTRUCTION_NODE_LIMIT{ 1000u };
    /// Number of lines used for the grid.
    static constexpr std::size_t GRID_SIZE{ 120u };
    /// Spacing between 2 lines.
    static constexpr float GRID_SPACING{ 0.25f };
    /// Length of the plane side.
    static constexpr float PLANE_SIZE{ 30.0f };
    /// Delete all history states when reloading a tree?
    static constexpr bool DELETE_HISTORY_ON_RELOAD{ false };

    /// @brief Initialize scene structures. Must be further initialized!
    TreeScene();
    /// @brief Cleanup resources and deinitialize.
    ~TreeScene();

    /// @brief Initialize the scene.
    void initialize(treerndr::TreeRenderer::Ptr renderer);

    /// @brief Replace currently loaded tree skeleton with contents of given file. This resets all selections!
    void reloadTree(const std::string &path, bool deleteHistory = DELETE_HISTORY_ON_RELOAD,
        bool loadReference = true);

    /**
     * @brief Replace currently loaded tree skeleton with provided serialized representation.
     *
     * @param serialized Serialized representation of the tree (.tree format).
     * @param geometryOnly Set to true to change only tree geometry and keep all other meta-data.
     * @warning This resets all selections!
     */
    void reloadTreeSerialized(const std::string &serialized, bool geometryOnly = true,
        bool deleteHistory = DELETE_HISTORY_ON_RELOAD, bool loadReference = true);

    /// @brief Replace currently loaded tree skeleton with the provided one. This resets all selections!
    void reloadTree(const treeio::ArrayTree &tree, bool deleteHistory = DELETE_HISTORY_ON_RELOAD,
        bool loadReference = true);

    /// @brief Update rendered tree geometry from the currently loaded tree skeleton.
    void reloadTreeGeometry();

    /// @brief Update colors of the rendered tree skeleton. Faster than than full reload.
    void reloadTreePointColors();

    /// @brief Replace currently loaded reference model with contents of given file. Returns reference bounding box.
    treeutil::BoundingBox reloadReference(const std::string &path, bool includeSkeleton,
        bool updateSkeletonPath = true, bool forceReload = false);

    /// @brief Replace currently loaded reference model with contents of given file. Returns reference bounding box.
    treeutil::BoundingBox reloadObjReference(const std::string &path,
        bool updateSkeletonPath = true, bool forceReload = false);

    /// @brief Replace currently loaded reference model with contents of given file. Returns reference bounding box.
    treeutil::BoundingBox reloadFbxReference(const std::string &path, bool includeSkeleton,
        bool updateSkeletonPath = true);

    /// @brief Clear all dirty flags from tree skeletons. Returns original dirty state.
    bool clearDirtyFlags();

    /// @brief Get the currently used tree skeleton.
    treeop::ArrayTreeHolder::Ptr currentTree();

    /// @brief Get reconstruction of the currently used tree skeleton. Automatically re-calculates if not current.
    treeutil::TreeReconstruction::Ptr currentTreeReconstruction();

    /// @brief Access runtime properties of the current tree.
    treeio::RuntimeTreeProperties &currentRuntimeProperties();

    /// @brief Perform reconstruction re-calculation next time it is requested.
    void recalculateReconstruction();

    /// @brief Set visibility of instance by name to be visible (true) or invisible (false).
    void displayInstance(const std::string &name, bool visible);

    /// @brief Get object representing the main scene light.
    treerndr::MeshInstance::Ptr sceneLight();

    /// @brief Set scene parameters for photo mode.
    void setupPhotoMode();

    /// @brief Set scene parameters for edit mode.
    void setupEditMode();

    /// @brief Get currently used renderer.
    treerndr::TreeRenderer::Ptr renderer();

    /// @brief Convert segments in given tree skeleton into a mesh.
    static treerndr::BufferBundle treeSegmentsMesh(const treeop::ArrayTreeHolder::Ptr &treeHolder);

    /// @brief Convert points in given tree skeleton into a mesh.
    static treerndr::BufferBundle treePointsMesh(const treeop::ArrayTreeHolder::Ptr &treeHolder,
        bool onlyColors = false);

    /// @brief Convert a given tree skeleton into a reconstruction mesh.
    static treerndr::BufferBundle treeReconstructionMesh(const treeop::ArrayTreeHolder::Ptr &treeHolder,
        const treeutil::TreeReconstruction::Ptr &reconstruction, treeop::TreeVersion &reconVersion);
private:
    /// Size used for skeleton nodes.
    static constexpr float NODE_SIZE{ 8.5f };
    /// Width used for skeleton segments.
    static constexpr float SEGMENT_WIDTH{ 2.5f };
    /// Width used for grid lines.
    static constexpr float GRID_WIDTH{ 1.5f };

    /// @brief Update the tree model from current tree. Returns whether update occurred.
    bool updateTreeModel();
    /// @brief Update the current selection visualization. Returns whether update occurred.
    bool updateTreeModelSelection(treeop::ArrayTreeSelection &selection);

    /// @brief Update the tree reconstruction with current data. Returns immediately if current.
    static treeop::TreeVersion updateReconstruction(const treeop::ArrayTreeHolder::Ptr &tree,
        const treeutil::TreeReconstruction::Ptr &reconstruction, const treeop::TreeVersion &reconVersion);

    /// Renderer being used to render this scene.
    treerndr::TreeRenderer::Ptr mRenderer{ };
    /// Currently loaded tree skeleton.
    treeop::ArrayTreeHolder::Ptr mTree;
    /// Tree reconstruction, which may not be current.
    treeutil::TreeReconstruction::Ptr mReconstruction{ };
    /// Version of the currently reconstructed tree.
    treeop::TreeVersion mReconstructionVersion{ };

    /// Version of the currently displayed tree skeleton.
    treeop::TreeVersion mTreeVersion{ };
    /// Version of the currently displayed tree selection.
    treeop::TreeSelectionVersion mSelectionVersion{ };
protected:
}; // class TreeScene

} // namespace treescene

#endif // TREE_SCENE_H