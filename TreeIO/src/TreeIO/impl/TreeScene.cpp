/**
 * @author Tomas Polasek, David Hrusa
 * @date 1.15.2020
 * @version 1.0
 * @brief Scene representation for the main viewport.
 */

#include "TreeScene.h"

#include "TreeOperations.h"
#include "TreeReconstruction.h"

namespace treescene
{

TreeScene::TreeScene()
{ /* Automatic */ }

TreeScene::~TreeScene()
{ /* TODO - Cleanup scene objects? */ }

void TreeScene::reloadTree(const std::string &path, bool deleteHistory, bool loadReference)
{
    auto newTree{ treeio::ArrayTree::fromPath<treeio::RuntimeMetaData>(path) };

    auto &meta{ newTree.metaData() };
    const auto runtimeMeta{ meta.getRuntimeMetaData<treeio::RuntimeMetaData>() };
    runtimeMeta->loadPathExists = true;
    runtimeMeta->loadPathPath = path;

    reloadTree(newTree, deleteHistory, loadReference);
}

void TreeScene::reloadTreeSerialized(const std::string &serialized,
    bool geometryOnly, bool deleteHistory, bool loadReference)
{
    auto newTree{ treeio::ArrayTree::fromString<treeio::RuntimeMetaData>(serialized) };

    if (geometryOnly)
    {
        const auto &currentTree{ mTree->currentTree() };
        newTree.setFilePath(currentTree.filePath());
        newTree.metaData() = currentTree.metaData();
    }

    reloadTree(newTree, deleteHistory, loadReference);
    Info << "Reloaded tree from serialized skeleton!" << std::endl;
}

void TreeScene::reloadTree(const treeio::ArrayTree &tree, bool deleteHistory, bool loadReference)
{
    if (!tree.isRootNodeValid())
    { Error << "Failed to load invalid tree!" << std::endl; return; }

    // TODO - Delete all history states, or keep the old tree?
    mTree->changeTree(tree, deleteHistory);
    // Reload reference model if possible
    if (loadReference && !mTree->currentTree().metaData().treeReference.empty())
    { reloadReference(mTree->currentTree().metaData().treeReference, false); }
    reloadTreeGeometry();
}

void TreeScene::reloadTreeGeometry()
{
    if (!mTree->currentTree().isRootNodeValid())
    { return; }

#ifndef NDEBUG
    Info << "Reloading whole tree geometry to the GPU..." << std::endl;
#endif // NDEBUG

    // Create mesh data for points and segments from the source tree.
    auto pointsData{ treePointsMesh(mTree) };
    auto segmentsData{ treeSegmentsMesh(mTree) };
    auto reconData{ treeReconstructionMesh(mTree, mReconstruction, mReconstructionVersion) };

    treeutil::checkGLError("reloadTreeGeom: gen data");

    // Recover the currently loaded meshes and bundle.
    const auto pointsMesh{ mRenderer->createGetVao(INST_POINTS_NAME) };
    const auto pointsBundle{ mRenderer->createGetBundle(INST_POINTS_NAME) };
    const auto segmentsMesh{ mRenderer->createGetVao(INST_SEGMENTS_NAME) };
    const auto segmentsBundle{ mRenderer->createGetBundle(INST_SEGMENTS_NAME) };
    const auto reconMesh{ mRenderer->createGetVao(INST_RECONSTRUCTION_NAME) };
    const auto reconBundle{ mRenderer->createGetBundle(INST_RECONSTRUCTION_NAME) };

    treeutil::checkGLError("reloadTreeGeom: createVAOs");

    // Replace meshes and reload.
    *pointsBundle = std::move(pointsData);
    mRenderer->loadBundleInto(pointsMesh, pointsBundle, GL_NONE);
    *segmentsBundle = std::move(segmentsData);
    mRenderer->loadBundleInto(segmentsMesh, segmentsBundle, GL_NONE);
    *reconBundle = std::move(reconData);
    mRenderer->loadReconBundleInto(reconMesh, reconBundle, GL_NONE);

    // Update tree instances being displayed by the mesh.
    reconMesh->tree = currentTree();
    reconMesh->reconstruction = currentTreeReconstruction();

    // Update point colors using the storage buffer.
    mRenderer->addStorageBufferTo(pointsMesh, pointsBundle->color, GL_DYNAMIC_DRAW);

    treeutil::checkGLError("reloadTreeGeom: add buffers ");
}

void TreeScene::reloadTreePointColors()
{
    if (!mTree->currentTree().isRootNodeValid())
    { return; }

#ifndef NDEBUG
    Info << "Reloading tree point colors to the GPU..." << std::endl;
#endif // NDEBUG

    /// Generate point colors array.
    const auto pointColors{ treePointsMesh(mTree, true) };

    // Recover currently loaded mesh and bundle.
    const auto pointsMesh{ mRenderer->createGetVao(INST_POINTS_NAME) };
    const auto pointsBundle{ mRenderer->createGetBundle(INST_POINTS_NAME) };

    // Replace colors and reload using the storage buffer.
    pointsBundle->color = std::move(pointColors.color);
    mRenderer->addStorageBufferTo(pointsMesh, pointsBundle->color, GL_DYNAMIC_DRAW);
}

treeutil::BoundingBox TreeScene::reloadReference(const std::string &path,
    bool includeSkeleton, bool updateSkeletonPath, bool forceReload)
{
    if (treeutil::equalCaseInsensitive(treeutil::fileExtension(path), ".obj"))
    { return reloadObjReference(path, updateSkeletonPath, forceReload); }
    else if (treeutil::equalCaseInsensitive(treeutil::fileExtension(path), ".fbx"))
    { return reloadFbxReference(path, includeSkeleton, updateSkeletonPath); }

    return { };
}

treeutil::BoundingBox TreeScene::reloadObjReference(const std::string &path,
    bool updateSkeletonPath, bool forceReload)
{
    if (!treeutil::fileExists(path))
    { Error << "Failed to load reference model from: " << path << std::endl; return { }; }
    const auto referenceMesh{ mRenderer->getVao(INST_REFERENCE_NAME) };
    const auto referenceBundle{ mRenderer->loadObjInto(referenceMesh, path, GL_TRIANGLES,forceReload) };

    currentTree()->currentTree().metaData().treeReference = path;
    // Default save path for the skeleton should use the same base path as the input model.
    if(updateSkeletonPath)
    { currentTree()->currentTree().setFilePath(treeutil::replaceExtension(path, ".tree")); }

    return referenceBundle->boundingBox();
}

treeutil::BoundingBox TreeScene::reloadFbxReference(const std::string &path,
    bool includeSkeleton, bool updateSkeletonPath)
{
    if (!treeutil::fileExists(path))
    { Error << "Failed to load reference model from: " << path << std::endl; return { }; }

    const auto referenceMesh{ mRenderer->getVao(INST_REFERENCE_NAME) };

    treeio::ArrayTree skeleton{ };
    const auto referenceBundle{
        mRenderer->loadFbxInto(referenceMesh, path, GL_TRIANGLES,
            includeSkeleton ? &skeleton : nullptr)
    };

    if (includeSkeleton && skeleton.nodeCount())
    { // Got a new skeleton -> replace the old one.
        skeleton.metaData().treeReference = path;
        skeleton.setLoaded(true);
        currentTree()->changeTree(std::move(skeleton), DELETE_HISTORY_ON_RELOAD);
    }
    else
    { // No skeleton included, keep the old one.
        currentTree()->currentTree().metaData().treeReference = path;
    }

    // Default save path for the skeleton should use the same base path as the input model.
    if(updateSkeletonPath)
    { currentTree()->currentTree().setFilePath(treeutil::replaceExtension(path, ".tree")); }

    return referenceBundle->boundingBox();
}

bool TreeScene::clearDirtyFlags()
{ return mTree->resetDirty(); }

treeop::ArrayTreeHolder::Ptr TreeScene::currentTree()
{ return mTree; }

treeutil::TreeReconstruction::Ptr TreeScene::currentTreeReconstruction()
{ updateReconstruction(mTree, mReconstruction, mReconstructionVersion); return mReconstruction; }

treeio::RuntimeTreeProperties &TreeScene::currentRuntimeProperties()
{ return currentTree()->currentTree().metaData().getRuntimeMetaData<treeio::RuntimeMetaData>()->runtimeTreeProperties; }

void TreeScene::recalculateReconstruction()
{ mReconstructionVersion = { }; }

void TreeScene::displayInstance(const std::string &name, bool visible)
{
    const auto instance{ mRenderer->getInstance(name) };
    if (instance)
    { instance->show = visible; }
}

treerndr::MeshInstance::Ptr TreeScene::sceneLight()
{ return mRenderer->getInstance(INST_LIGHT_NAME); }

void TreeScene::setupPhotoMode()
{
    const auto pointsInstance{ mRenderer->getInstance(INST_POINTS_NAME) };
    pointsInstance->show = false;

    const auto segmentsInstance{ mRenderer->getInstance(INST_SEGMENTS_NAME) };
    segmentsInstance->show = false;

    const auto referenceInstance{ mRenderer->getInstance(INST_REFERENCE_NAME) };
    referenceInstance->show = false;

    const auto reconInstance = mRenderer->getInstance(INST_RECONSTRUCTION_NAME);
    reconInstance->shadows = true;
    reconInstance->castsShadows = true;
    reconInstance->transparent = false;
    reconInstance->show = true;

    const auto gridInstance{ mRenderer->getInstance(INST_GRID_NAME) };
    gridInstance->show = false;
    currentTree()->currentTree().metaData().getRuntimeMetaData<
        treeio::RuntimeMetaData>()->runtimeTreeProperties.showPoints = false;
    currentTree()->currentTree().metaData().getRuntimeMetaData<
        treeio::RuntimeMetaData>()->runtimeTreeProperties.showSegments = false;

    const auto planeInstance{ mRenderer->getInstance(INST_PLANE_NAME) };
    planeInstance->orderPriority = -5;
    planeInstance->shadows = true;
    planeInstance->castsShadows = true;
    planeInstance->show = true;

    const auto lightInstance{ mRenderer->getInstance(INST_LIGHT_NAME) };
    lightInstance->show = false;
}

void TreeScene::setupEditMode()
{
    const auto pointsInstance{ mRenderer->getInstance(INST_POINTS_NAME) };
    pointsInstance->pointSize = NODE_SIZE;
    pointsInstance->alwaysVisible = false;
    pointsInstance->show = true;
    // Setup points to use color override storage buffer.
    pointsInstance->overrideColorStorageEnabled = true;

    const auto segmentsInstance{ mRenderer->getInstance(INST_SEGMENTS_NAME) };
    segmentsInstance->lineWidth = SEGMENT_WIDTH;
    segmentsInstance->alwaysVisible = false;
    segmentsInstance->show = true;

    const auto referenceInstance{ mRenderer->getInstance(INST_REFERENCE_NAME) };
    referenceInstance->transparent = true;
    referenceInstance->show = true;

    const auto reconInstance = mRenderer->getInstance(INST_RECONSTRUCTION_NAME);
    reconInstance->shadows = true;
    reconInstance->castsShadows = true;
    reconInstance->transparent = true;
    //reconInstance->cullFaces = false;
    reconInstance->show = true;

    const auto gridInstance{ mRenderer->getInstance(INST_GRID_NAME) };
    gridInstance->lineWidth = GRID_WIDTH;
    gridInstance->alwaysVisible = true;
    gridInstance->orderPriority = -4;
    gridInstance->show = true;

    const auto planeInstance{ mRenderer->getInstance(INST_PLANE_NAME) };
    planeInstance->orderPriority = -5;
    planeInstance->shadows = true;
    planeInstance->castsShadows = true;
    planeInstance->show = true;

    const auto lightInstance{ mRenderer->getInstance(INST_LIGHT_NAME) };
    lightInstance->translate = LIGHT_DEFAULT_POS;
    lightInstance->show = true;
}

treerndr::TreeRenderer::Ptr TreeScene::renderer()
{ return mRenderer; }

void TreeScene::initialize(treerndr::TreeRenderer::Ptr renderer)
{
    // Set the renderers.
    mRenderer = renderer;

    // Create an empty placeholder tree skeleton.
    mTree = treeop::ArrayTreeHolder::instantiate();
    mReconstruction = treeutil::TreeReconstruction::instantiate();
    mReconstructionVersion = { };

    // Initialize buffers and instances holding the skeleton and reference model.
    const auto pointsMesh{ mRenderer->createGetVao(INST_POINTS_NAME) };
    mRenderer->createGetBundle(INST_POINTS_NAME);
    pointsMesh->renderMode = GL_POINTS;
    const auto pointsInstance{ mRenderer->createInstance(INST_POINTS_NAME, pointsMesh) };
    pointsInstance->pointSize = NODE_SIZE;
    pointsInstance->alwaysVisible = false;

    const auto segmentsMesh{ mRenderer->createGetVao(INST_SEGMENTS_NAME) };
    mRenderer->createGetBundle(INST_SEGMENTS_NAME);
    segmentsMesh->renderMode = GL_LINES;
    const auto segmentsInstance{ mRenderer->createInstance(INST_SEGMENTS_NAME, segmentsMesh) };
    segmentsInstance->lineWidth = SEGMENT_WIDTH;
    segmentsInstance->alwaysVisible = false;

    const auto referenceMesh{ mRenderer->createGetVao(INST_REFERENCE_NAME) };
    const auto referenceInstance{ mRenderer->createInstance(INST_REFERENCE_NAME, referenceMesh) };
    referenceInstance->transparent = true;

    const auto reconMesh{ mRenderer->createGetVao(INST_RECONSTRUCTION_NAME) };
    reconMesh->renderMode = GL_PATCHES;
    const auto reconInstance = mRenderer->createInstance(INST_RECONSTRUCTION_NAME, reconMesh);
    reconInstance->shadows = true;
    reconInstance->castsShadows = true;
    reconInstance->transparent = true;
    //reconInstance->cullFaces = false;

    // Setup points to use color override storage buffer.
    pointsInstance->overrideColorStorageEnabled = true;

    // Load grid pattern and instantiate.
    const auto gridMesh{ mRenderer->createGetVao(INST_GRID_NAME) };
    const auto bundleGrid{ mRenderer->createGetBundle(INST_GRID_NAME) };
    *bundleGrid = treerndr::PatternFactory::createGrid(GRID_SIZE, GRID_SPACING);
    gridMesh->renderMode = GL_LINES;
    mRenderer->loadBundleInto(gridMesh, bundleGrid, GL_NONE);
    const auto gridInstance{ mRenderer->createInstance(INST_GRID_NAME, gridMesh) };
    gridInstance->lineWidth = GRID_WIDTH;
    gridInstance->alwaysVisible = true;
    gridInstance->orderPriority = -4;

    // Load plane pattern and instantiate.
    const auto planeMesh{ mRenderer->createGetVao(INST_PLANE_NAME) };
    const auto bundlePlane{ mRenderer->createGetBundle(INST_PLANE_NAME) };
    *bundlePlane = treerndr::PatternFactory::createPlane(PLANE_SIZE);
    planeMesh->renderMode = GL_TRIANGLE_STRIP;
    mRenderer->loadBundleInto(planeMesh, bundlePlane, GL_NONE);
    auto planeInstance = mRenderer->createInstance(INST_PLANE_NAME, planeMesh);
    planeInstance->orderPriority = -5;
    planeInstance->shadows = true;
    planeInstance->castsShadows = true;

    // Load light pattern and instantiate.
    const auto lightMesh{ mRenderer->createGetVao(INST_LIGHT_NAME) };
    mRenderer->createGetBundle(INST_LIGHT_NAME);
    lightMesh->renderMode = GL_TRIANGLES;
    auto lightInstance{ mRenderer->createInstance(INST_LIGHT_NAME, lightMesh) };
    lightInstance->translate = LIGHT_DEFAULT_POS;
    lightInstance->show = true;
    const auto lightLight = mRenderer->createGetLight(INST_LIGHT_NAME);
    lightLight->type = treerndr::Light::LightType::Directional;
    lightLight->shadowMapWidth = 4096.0f;
    lightLight->shadowMapHeight = 4096.0f;
    lightLight->nearPlane = 0.1f;
    lightLight->farPlane = 40.0f;
    lightLight->fieldOfView = 768.0f;
    mRenderer->attachLight(lightInstance, lightLight);
}

treerndr::BufferBundle TreeScene::treeSegmentsMesh(const treeop::ArrayTreeHolder::Ptr &treeHolder)
{
    treerndr::BufferBundle bundle{ };
    const auto &tree{ treeHolder->currentTree() };

    GLuint idx{ 0u };
    GLint previous{ -1 };
    std::vector<treeio::ArrayTree::NodeIdT> diveStore{ };
    std::vector<treeio::ArrayTree::NodeIdT> diveStoreIdx{ };
    std::vector<treeio::ArrayTree::NodeIdT> idxStoreChild{ };
    auto current{ tree.getRootId() };

    while (true) 
    {
        for (std::size_t iii = 1u; iii < tree.getNodeChildren(current).size(); ++iii) 
        {
            // Backup: current location, element index, next expanded child.
            diveStore.push_back(current);
            diveStoreIdx.push_back(idx);
            idxStoreChild.push_back(iii);
        }

        const auto &currentNode{ tree.getNode(current) };
        bundle.pushVertexData(currentNode.data().pos.x);
        bundle.pushVertexData(currentNode.data().pos.y);
        bundle.pushVertexData(currentNode.data().pos.z);
        bundle.color.push_back(currentNode.data().lineColor.r);
        bundle.color.push_back(currentNode.data().lineColor.g);
        bundle.color.push_back(currentNode.data().lineColor.b);
        bundle.color.push_back(currentNode.data().lineColor.a);

        if (previous >= 0)
        {
            bundle.element.push_back(static_cast<GLuint>(previous));
            bundle.element.push_back(idx);
        }
        if (tree.getNodeChildren(current).size() > 0)
        {
            // Continue down the first child if possible.
            current = tree.getNodeChildren(current)[0];
            previous = static_cast<GLint>(idx);
        }
        else
        {
            // Reached a dead end. Either bounce back or terminate.
            if (!diveStore.empty())
            {
                current = tree.getNodeChildren(diveStore.back())[idxStoreChild.back()];
                previous = (GLint)diveStoreIdx.back();
                diveStore.pop_back();
                diveStoreIdx.pop_back();
                idxStoreChild.pop_back();
            }
            else 
            { return bundle; }
        }
        idx++;
    }

    return bundle;
}

treerndr::BufferBundle TreeScene::treePointsMesh(const treeop::ArrayTreeHolder::Ptr &treeHolder, bool onlyColors)
{
    treerndr::BufferBundle bundle;
    const auto &tree{ treeHolder->currentTree() };

    GLuint idx{ 0u };
    std::vector<treeio::ArrayTree::NodeIdT> diveStore{ };
    std::vector<treeio::ArrayTree::NodeIdT> idxStoreChild{ };
    auto current{ tree.getRootId() };
    glm::vec4 color { 1.0f, 1.0f, 0.0f, 1.0f };
    
    while (tree.isNodeIdValid(current))
    {
        for (std::size_t iii = 1u; iii < tree.getNodeChildren(current).size(); ++iii) 
        {
            // Backup: current location and the next expanded child.
            diveStore.push_back(current);
            idxStoreChild.push_back(iii);
        }

        const auto &currentNode{ tree.getNode(current) };
        if (!onlyColors)
        {
            bundle.pushVertexData(currentNode.data().pos.x);
            bundle.pushVertexData(currentNode.data().pos.y);
            bundle.pushVertexData(currentNode.data().pos.z);
        }
        bundle.color.push_back(currentNode.data().pointColor.r);
        bundle.color.push_back(currentNode.data().pointColor.g);
        bundle.color.push_back(currentNode.data().pointColor.b);
        bundle.color.push_back(currentNode.data().pointColor.a);
        if (!onlyColors)
        { bundle.element.push_back(idx); }

        if (tree.getNodeChildren(current).size() > 0u) 
        {
            // Continue down the first child if possible.
            current = tree.getNodeChildren(current)[0u];
        }
        else 
        {
            // Reached a dead end. Either bounce back or terminate.
            if (diveStore.size() > 0u) 
            {
                current = tree.getNodeChildren(diveStore.back())[idxStoreChild.back()];
                diveStore.pop_back();
                idxStoreChild.pop_back();
            }
            else 
            { return bundle; }
        }
        idx++;
    }
    return bundle;
}

treerndr::BufferBundle TreeScene::treeReconstructionMesh(
    const treeop::ArrayTreeHolder::Ptr &treeHolder,
    const treeutil::TreeReconstruction::Ptr &reconstruction,
    treeop::TreeVersion &reconVersion)
{
    // Update the reconstruction if necessary.
    reconVersion = updateReconstruction(treeHolder, reconstruction, reconVersion);

    const auto &vertexData{ reconstruction->vertexData() };

    // Store vertex data:
    treerndr::BufferBundle bundle{ };
    static constexpr auto ELEMENT_SIZE{ sizeof(GLfloat) };
    static constexpr auto STRIDE{ treeutil::TreeReconstruction::VertexData::FLOAT_ELEMENTS * ELEMENT_SIZE };
    bundle.vertex.resize(vertexData.size() * STRIDE);
    for (std::size_t iii = 0u; iii < vertexData.size(); ++iii)
    {
        auto &data{ vertexData[iii] };

        // TODO - Create a proper VAO wrapper without improper casting.
        // Position of the vertex in world space, w is unused.
        bundle.setVertexData(data.position, iii, STRIDE, ELEMENT_SIZE * 0u);
        // Normal of the vertex in world space, w is set to distance from the tree root.
        bundle.setVertexData(data.normal, iii, STRIDE, ELEMENT_SIZE * 4u);
        // Vector parallel with the branch in world space, w is set to the radius of the branch.
        bundle.setVertexData(data.parallel, iii, STRIDE, ELEMENT_SIZE * 8u);
        // Tangent of the branch in world space, w is unused.
        bundle.setVertexData(data.tangent, iii, STRIDE, ELEMENT_SIZE * 12u);
        // Adjacency indices for this vertex, x = this idx, y = parent idx, z = child idx, w is unused. 0 as invalid idx.
        bundle.setVertexData(data.adjacency, iii, STRIDE, ELEMENT_SIZE * 16u);
    }

    return bundle;
}

treeop::TreeVersion TreeScene::updateReconstruction(const treeop::ArrayTreeHolder::Ptr &tree,
    const treeutil::TreeReconstruction::Ptr &reconstruction, const treeop::TreeVersion &reconVersion)
{
    reconstruction->parameters().meshScale = tree->currentTree().metaData().calcSkeletonScale();

    if (reconVersion == tree->version())
    { return reconVersion; }

    treeutil::Timer profilingTimer{ };
    reconstruction->reconstructTree(tree->currentTree(), true);
    treeutil::Info << "[Prof] timeReconstructCpu: " << profilingTimer.reset() << std::endl;

    return tree->version();
}

bool TreeScene::updateTreeModel()
{
    if (mTreeVersion < mTree->version())
    {
        const auto cleanedTree{ mTree->cleanupTree() };
        reloadTreeGeometry();
        mTreeVersion = mTree->version();

        return true;
    }
    else
    { return false; }
}

bool TreeScene::updateTreeModelSelection(treeop::ArrayTreeSelection &selection)
{
    if (mSelectionVersion < selection.version())
    { reloadTreePointColors(); mSelectionVersion = selection.version(); return true; }
    else
    { return false; }
}

} // namespace treescene
