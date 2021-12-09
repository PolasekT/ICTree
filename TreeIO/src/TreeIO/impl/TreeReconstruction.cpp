/**
 * @author Tomas Polasek, David Hrusa
 * @date 5.7.2020
 * @version 1.0
 * @brief Tree reconstruction utilities.
 */

#include "TreeReconstruction.h"

#include "TreeGLUtils.h"
#include "TreeUtils.h"

namespace treeutil
{

TreeReconstruction::TreeReconstruction()
{ /* Automatic */ }
TreeReconstruction::TreeReconstruction(treeio::ArrayTree &tree, bool copyResults)
{ reconstructTree(tree, copyResults); }
TreeReconstruction::~TreeReconstruction()
{ /* Automatic */ }

bool TreeReconstruction::reconstructTree(treeio::ArrayTree &tree, bool copyResults)
{
    const auto treeChains{ prepareTree(tree) };
    if (!treeChains)
    { Error << "Failed to generate chains for tree reconstruction!" << std::endl; return false; }

    calculateRadii(treeChains);

    auto maxDistanceFromRoot{ 0.0f };
    auto maxBranchRadius{ 0.0f };
    auto vertexData{ prepareVertexData(treeChains, maxDistanceFromRoot, maxBranchRadius) };
    if (vertexData.empty())
    { Error << "Failed to generate vertex data for tree reconstruction!" << std::endl; }
    mVertexData = std::move(vertexData);
    mMaxDistanceFromRoot = maxDistanceFromRoot;
    mMaxBranchRadius = maxBranchRadius;

    if (copyResults)
    { copyResultsTo(treeChains, tree); }

    return !mVertexData.empty();
}

const std::vector<TreeReconstruction::VertexData> &TreeReconstruction::vertexData() const
{ return mVertexData; }

TreeReconstruction::VisualParameters &TreeReconstruction::parameters()
{ return mParameters; }

const TreeReconstruction::VisualParameters &TreeReconstruction::parameters() const
{ return mParameters; }

float TreeReconstruction::maxDistanceFromRoot() const
{ return mMaxDistanceFromRoot; }

float TreeReconstruction::maxBranchRadius() const
{ return mMaxBranchRadius; }

WrapperPtrT<TreeChains> TreeReconstruction::prepareTree(const treeio::ArrayTree &tree) const
{
    if (tree.nodeCount() == 0u)
    { return nullptr; }

    // Initialize and calculate tree properties.
    return TreeChains::instantiate(tree);
}

void TreeReconstruction::calculateRadii(const WrapperPtrT<TreeChains> &tree) const
{
    const auto &metaData{ tree->internalTree().metaData() };

    if (!treeutil::aboveEpsilon(metaData.thicknessFactor))
    {
        Warning << "Unable to calculate radii for tree reconstruction (" << tree->internalTree().filePath()
                << "), since thicknessFactor = " << metaData.thicknessFactor << std::endl;
        return;
    }

    // Cascade calculation from leaves towards the root node.
    tree->cascadeDownwards([&] (auto &internalTree, const auto &currentNode) {
        // Check if we need to re-calculate current nodes branch thickness.
        auto &currentData{ internalTree.getNode(currentNode).data() };
        if (!metaData.recalculateRadius &&
            currentData.calculatedThickness >= MIN_BRANCH_THICKNESS &&
            !currentData.recalculatedRadius)
        { return false; }

        // Leaf nodes have predetermined branch width.
        const auto &children{ internalTree.getNodeChildren(currentNode) };
        if (children.empty())
        { currentData.calculatedThickness = metaData.startingThickness; }
        else
        {
            // Find highest child order.
            auto highestChildOrder{ std::numeric_limits<std::size_t>::min() };
            for (const auto &cId : children)
            { highestChildOrder = std::max<std::size_t>(highestChildOrder, internalTree.getNode(cId).data().graveliusOrder); }

            // Accumulate widths of children of highest order.
            auto childWidthSum{ 0.0f };
            for (const auto &cId : children)
            { childWidthSum += std::pow(internalTree.getNode(cId).data().calculatedThickness, metaData.thicknessFactor); }

            // Save current branch width.
            currentData.calculatedThickness = std::max<float>(
                MIN_BRANCH_THICKNESS,
                std::pow(childWidthSum, 1.0f / metaData.thicknessFactor)
            );
        }

        if (currentData.freezeThickness)
        { /* Keep currentData.thickness value the same */ }
        else
        { currentData.thickness = currentData.calculatedThickness; }

        currentData.recalculatedRadius = true;

        // Continue with downwards cascade.
        return false;
    });
}

std::vector<TreeReconstruction::VertexData> TreeReconstruction::prepareVertexData(
    const WrapperPtrT<TreeChains> &tree, float &maxDistanceFromRoot, float &maxBranchRadius) const
{
    // Prepare for accumulation of vertex data.
    const auto &chains{ tree->chains() }; TREE_UNUSED(chains);
    const auto &propTree{ tree->internalTree() };
    using NodeIdT = TreeChains::NodeIdT;
    static constexpr auto INVALID_NODE_ID{ TreeChains::INVALID_NODE_ID };
    static constexpr auto INVALID_IDX{ std::numeric_limits<std::size_t>::max() };

    // Prepare result array.
    std::vector<VertexData> vertexData{ };

    // Reserve space for all of the branches - one record for each end of the branch.
    const auto totalVertexData{ (propTree.nodeCount() - 1u) * 2u };
    vertexData.reserve(totalVertexData);

    /// @brief Helper structure for iteration.
    struct IterHelper
    {
        /// The vertex itself.
        NodeIdT vertex{ };
        /// Index of this vertex within the resulting vertex data array.
        std::size_t vertexIdx{ };
        /// Parent vertex.
        NodeIdT parent{ };
        /// Index of the parent within the resulting vertex data array.
        std::size_t parentIdx{ };
        /// Node used as primary child by the parent.
        NodeIdT parentPrimaryChild{ };
    }; // struct IterHelper

    // Initialize the algorithm with root vertex.
    std::stack<IterHelper> vertexStack{ };
    vertexStack.emplace(IterHelper{ propTree.getRootId(), INVALID_IDX, INVALID_NODE_ID, INVALID_IDX , INVALID_NODE_ID });
    maxDistanceFromRoot = 0.0f;
    maxBranchRadius = 0.0f;

    while (!vertexStack.empty())
    { // Iterate through all of the vertices in the input tree skeleton - Root to leaves.
        const auto currentHelper{ vertexStack.top() }; vertexStack.pop();
        const auto parentId{ currentHelper.parent }; TREE_UNUSED(parentId);
        const auto parentIdx{ currentHelper.parentIdx };
        const auto currentId{ currentHelper.vertex };
        const auto currentIdx{ currentHelper.vertexIdx }; TREE_UNUSED(currentIdx);

        // Prepare common data for the source node.
        auto currentDataS{ prepareNodeData(tree, currentId, maxDistanceFromRoot, maxBranchRadius) };

        // Create child segments.
        for (const auto &cId : propTree.getNodeChildren(currentHelper.vertex))
        { // Iterate through all child vertices.
            // Find primary child for this node:
            auto primaryChild{ INVALID_IDX };
            auto primaryChildTotalLength{ 0.0f };
            for (const auto &cId : propTree.getNodeChildren(currentHelper.vertex))
            { // Iterate through all child vertices.
                const auto childTotalLength{ propTree.getNode(cId).data().totalChildLength };
                if (primaryChild == INVALID_IDX || childTotalLength > primaryChildTotalLength)
                { primaryChild = cId; primaryChildTotalLength = childTotalLength; }
            }

            // Allocate vertex data for the child segment, one for each end of the branch.
            const auto cIdx{ vertexData.size() };
            vertexData.push_back({ }); auto &dataS{ vertexData.back() };
            vertexData.push_back({ }); auto &dataT{ vertexData.back() };
            vertexStack.emplace(IterHelper{
                cId, cIdx,
                currentHelper.vertex, cIdx,
                primaryChild
            });

            if (cId == primaryChild)
            { // We are processing primary child of our parent -> Fill in the childs child index.
                vertexData[parentIdx + 1u].adjacency.z = cIdx + 1u + 1u;
            }

            // Fill data:
            dataS = currentDataS;
            // Adjacency indices for this vertex, x = this idx, y = parent idx, z = child idx, w is unused. 0 as invalid idx.
            dataS.adjacency = {
                // Source node is first:
                cIdx + 1u,
                parentIdx == INVALID_IDX ? 0u : parentIdx + 1u,
                // Target node is second:
                cIdx + 1u + 1u,
                0u
            };

            // Adjacency indices for this vertex, x = this idx, y = parent idx, z = child idx, w is unused. 0 as invalid idx.
            dataT = prepareNodeData(tree, cId, maxDistanceFromRoot, maxBranchRadius);
            dataT.adjacency = {
                // Target node is second:
                cIdx + 1u + 1u,
                // Source node is first:
                cIdx + 1u,
                // Primary child will fill this later:
                0u,
                0u
            };
        }
    }

    return vertexData;
}

TreeReconstruction::VertexData TreeReconstruction::prepareNodeData(
    const WrapperPtrT<TreeChains> &tree, TreeChains::NodeIdT nodeId,
    float &maxDistanceFromRoot, float &maxBranchRadius) const
{
    VertexData data{ };
    const auto &propTree{ tree->internalTree() };

    // Position of the vertex in world space, w is unused.
    const auto currentPos{ propTree.getNode(nodeId).data().pos };
    data.position = glm::vec4{ currentPos.x, currentPos.y, currentPos.z, 1.0f };

    // Normal of the vertex in world space, w is set to distance from the tree root.
    const auto currentNormal{ propTree.getNode(nodeId).data().basis.bitangent };
    const auto currentRootDistance{ propTree.getNode(nodeId).data().distance };
    maxDistanceFromRoot = std::max<float>(maxDistanceFromRoot, currentRootDistance);
    data.normal = glm::vec4{ currentNormal.x, currentNormal.y, currentNormal.z, currentRootDistance };

    // Vector parallel with the branch in world space, w is set to the radius of the branch.
    const auto currentParallel{ propTree.getNode(nodeId).data().basis.direction };
    const auto currentRadius{ propTree.getNode(nodeId).data().thickness };
    maxBranchRadius = std::max<float>(maxBranchRadius, currentRadius);
    data.parallel = glm::vec4{ currentParallel.x, currentParallel.y, currentParallel.z, currentRadius };

    // Tangent of the branch in world space, w is unused.
    const auto currentTangent{ propTree.getNode(nodeId).data().basis.tangent };
    data.tangent = glm::vec4{ currentTangent.x, currentTangent.y, currentTangent.z, 1.0f };

    return data;
}

void TreeReconstruction::copyResultsTo(const WrapperPtrT<TreeChains> &inputTree,
    treeio::ArrayTree &outputTree)
{
    inputTree->cascadeDownwards([&outputTree] (auto &inputTree, const auto &currentNode) -> bool {
        const auto &inputData{ inputTree.getNode(currentNode).data() };
        auto &outputData{ outputTree.getNode(currentNode).data() };

        outputData.thickness = inputData.thickness;

        // Continue with downwards cascade.
        return false;
    });
}

} // namespace treeutil
