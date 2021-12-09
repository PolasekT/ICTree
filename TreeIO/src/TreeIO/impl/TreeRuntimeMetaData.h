/**
 * @author Tomas Polasek, David Hrusa
 * @date 5.8.2021
 * @version 1.0
 * @brief Implementation of tree runtime metadata.
 */

#ifndef TREE_RUNTIME_METADATA_H
#define TREE_RUNTIME_METADATA_H

#include <vector>
#include <string>

#include "TreeUtils.h"
#include "TreeRenderer.h"
#include "TreeSceneInstanceNames.h"

namespace treeio
{

/// @brief Container for tree scaling information.
struct RuntimeTreeProperties
{
    /// @brief Base scale of the tree used to multiply all other scales.
    float scaleBase{ 1.0f };
    /// @brief How much scale was applied by the scaleBase.
    float appliedScaleBase{ 1.0f };
    /// @brief Scale of the line graph of the tree.
    float scaleGraph{ 1.0f };
    /// @brief Scale of the reconstructed mesh.
    float scaleReconstruction{ 1.0f };
    /// @brief Scale of the reference obj model, if applicable
    float scaleReference{ 1.0f };
    /// @brief How much scale was applied to the reference. Used for branch width scaling.
    float appliedScaleReference{ 1.0f };

    /// @brief Scale of the points of the tree.
    bool showPoints{ true };
    /// @brief Scale of the line graph of the tree.
    bool showSegments{ true };
    /// @brief Scale of the reconstructed mesh.
    bool showReconstruction{ true };
    /// @brief Scale of the reference obj model, if applicable.
    bool showReference{ true };

    /// @brief Color of the reconstructed mesh.
    float colorReconstruction[4]{ 0.6f, 0.6f, 0.6f, 1.0f };
    /// @brief Color of the reference obj model, if applicable
    float colorReference[4]{ 0.2f, 0.6f, 0.8f, 0.4f };

    /// @brief Position of the line graph of the tree.
    float offsetGraph[3]{ 0.0f, 0.0f, 0.0f };
    /// @brief Position of the reconstructed mesh.
    float offsetReconstruction[3]{ 0.0f, 0.0f, 0.0f };
    /// @brief Position of the reference obj model, if applicable.
    float offsetReference[3]{ 0.0f, 0.0f, 0.0f };

    /// @brief Name of the points instance model.
    std::vector<std::string> instancePointNames{ treescene::instances::POINTS_NAME };
    /// @brief Name of the line instance model.
    std::vector<std::string> instanceSegmentNames{ treescene::instances::SEGMENTS_NAME };
    /// @brief Name of the reference model instance model.
    std::vector<std::string> instanceReferenceNames { treescene::instances::REFERENCE_NAME };
    /// @brief Name of the reconstruction instance model.
    std::vector<std::string> instanceReconstructionNames { treescene::instances::RECONSTRUCTION_NAME };

    /// @brief Whether changes to these values should get saved into the original tree file.
    bool saveChanges{ true };

    /// @brief Set scales for provided instance.
    void applyScales(const treerndr::MeshInstancePtr &instance,
        bool setScales = true, bool setPositions = true,
        bool setVisibility = true, bool setColors = true);

    /// @brief Recalculate based on new scale base.
    void changedScaleBase();
private:
protected:
}; // struct RuntimeTreeProperties

/// @brief Runtime only variables.
struct RuntimeMetaData : TreeRuntimeMetaData
{
    /// @brief Create a duplicate of this runtime meta-data structure.
    virtual Ptr duplicate() const override final;

    /// @brief called during loading of the tree to refresh the runtime metadata after deserialization.
    virtual void onLoad(TreeMetaData &metaData) override final;

    /// @brief called during saving of the tree to store any runtime variables into the permanent meta data prior to serialization.
    virtual void onSave(TreeMetaData &metaData) override final;

    /// Whether these metadata keep track of where the tree got loaded from.
    bool loadPathExists{ false };
    /// Can be used by the loader/saver to keep track of where this particular file got loaded from.
    std::string loadPathPath{ "stored.tree" };
    /// Keeps track of all the tree sizes without overwriting the metadata of the tree.
    RuntimeTreeProperties runtimeTreeProperties{ };
}; // struct RuntimeMetaData

} // namespace treeio

#endif // TREE_RUNTIME_METADATA_H
