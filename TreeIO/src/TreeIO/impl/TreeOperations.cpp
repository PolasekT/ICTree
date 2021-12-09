/**
 * @author Tomas Polasek
 * @date 11.20.2019
 * @version 1.0
 * @brief Operations and helpers on tree skeletons.
 */

#include "TreeOperations.h"

#include <algorithm>
#include <memory>

#include "TreeRuntimeMetaData.h"

namespace treeop
{

ArrayTreeHolder::ArrayTreeHolder(const treeio::ArrayTree &tree) : 
    mDirty{ false }, mTopologyChanged{ false }, mTree{ tree }, mHistory{ tree }, mVersion{ *this }
{ }

ArrayTreeHolder::~ArrayTreeHolder()
{ /* Automatic */}

void ArrayTreeHolder::changeTree(const treeio::ArrayTree &tree, bool resetHistory)
{
    auto treeCopy{ tree };
    changeTree(std::move(treeCopy), resetHistory);
}

void ArrayTreeHolder::changeTree(treeio::ArrayTree &&tree, bool resetHistory)
{
    mTree = std::move(tree);

    // Fill in missing runtime meta-data.
    if (!mTree.metaData().getRuntimeMetaData<treeio::RuntimeMetaData>())
    { mTree.metaData().setRuntimeMetaData(std::make_shared<treeio::RuntimeMetaData>()); }

    if (resetHistory || mHistory.stateCount() == 0u || !mHistory.history().front().state.isRootNodeValid())
    {
        markDirty();
        mHistory = OperationHistory<treeio::ArrayTree>{ mTree };
        mLastSavedVersion = { mVersion };
    }
    else
    {
        markDirty();
        snapshot();
    }
}

bool ArrayTreeHolder::snapshot()
{ 
    if (mLastSavedVersion < mVersion)
    { mHistory.pushState(mTree); mLastSavedVersion = mVersion; return true; }
    else
    { return false; }
}

void ArrayTreeHolder::popSnapshot()
{ mHistory.popState(); }

TreeVersion ArrayTreeHolder::lastSnapshotVersion() const
{ return mLastSavedVersion; }

bool ArrayTreeHolder::undo()
{
    // Make a snapshot.
    snapshot();

    // Go to the past state.
    const auto result{ mHistory.undo() };
    if (result)
    {
        mTree = mHistory.currentState();
        mTree.translationMap().clear();
        markDirty();
        mLastSavedVersion = mVersion;
    }
    return result;
}

bool ArrayTreeHolder::redo()
{
    const auto result{ mHistory.redo() };
    if (result)
    {
        mTree = mHistory.currentState();
        mTree.translationMap().clear();
        markDirty();
        mLastSavedVersion = mVersion;
    }
    return result;
}

bool ArrayTreeHolder::moveToState(std::size_t idx)
{ return mHistory.moveToState(idx); }

const ArrayTreeHolder::OperationHistoryList &ArrayTreeHolder::history() const
{ return mHistory.history(); }

treeio::ArrayTree &ArrayTreeHolder::currentTree()
{ return mTree; }

const treeio::ArrayTree &ArrayTreeHolder::currentTree() const
{ return mTree; }

bool ArrayTreeHolder::markDirty()
{
    const auto oldDirty{ mDirty };

    mDirty = true;
    mVersion++;

    return oldDirty;
}

bool ArrayTreeHolder::isDirty() const
{ return mDirty; }

bool ArrayTreeHolder::resetDirty()
{ return mDirty = false; }

void ArrayTreeHolder::setTopologyChanged()
{ mTopologyChanged = true; }

bool ArrayTreeHolder::hasTopologyChanged()
{ return mTopologyChanged; }

bool ArrayTreeHolder::cleanupTree()
{
    if (!mTopologyChanged)
    { return false; }

#ifndef NDEBUG
    Info << "Cleaning up and updating the current tree model!" << std::endl;
#endif // NDEBUG

    const auto savedPreviously{ mLastSavedVersion == mVersion };

    auto cleanTree{ currentTree().cleanup() };
    changeTree(std::move(cleanTree), false);

    mTopologyChanged = false;
    if (savedPreviously)
    { mLastSavedVersion = mVersion; }
    return true;
}

ArrayTreeHolder::TreeVersion ArrayTreeHolder::version() const
{ return mVersion; }

PointSet::PointSet() : 
    mPoints{ }
{ }

PointSet::PointSet(const InnerSetT &source) : 
    mPoints{ source }
{ }

PointSet::~PointSet()
{ /* Automatic */ }

PointSet PointSet::fromVector(const std::vector<PointT> &input)
{
    PointSet result{ };

    for (const auto &p : input)
    { result.addPoint(p); }

    return result;
}

PointSet::InnerSetT &PointSet::data()
{ return mPoints; }

const PointSet::InnerSetT &PointSet::data() const
{ return mPoints; }

bool PointSet::hasPoint(const PointT &point) const 
{ return hasPoint(mPoints, point); }

bool PointSet::removePoint(const PointT &point)
{ return removePoint(mPoints, point); }

bool PointSet::addPoint(const PointT &point)
{ return addPoint(mPoints, point); }

bool PointSet::clear()
{ 
    if (mPoints.empty())
    { return false; } 
    else
    { mPoints.clear(); return true; }
}

bool PointSet::removePoints(const PointSet &points)
{
    auto changed{ false };
    for (const auto &point : points.data())
    { changed |= removePoint(point); }
    return changed;
}

bool PointSet::addPoints(const PointSet& points)
{
    auto changed{ false };
    for (const auto& point : points.data())
    { changed |= addPoint(point); }
    return changed;
}

void PointSet::translatePoints(const std::map<PointT, PointT> &translationMap)
{
    decltype(mPoints) newSelection{ };

    for (const auto &pIdx : mPoints)
    { // Translate all points.
        const auto findIt{ translationMap.find(pIdx) };
        // If we find it -> put translation into the output set.
        if (findIt != translationMap.end())
        { newSelection.push_back(findIt->second); }
        // Else remove it by omission.
    }

    mPoints = newSelection;
}

std::size_t PointSet::size() const
{ return mPoints.size(); }

bool PointSet::empty() const
{ return mPoints.empty(); }

std::vector<PointSet> PointSet::computeComponents(const treeio::ArrayTree &tree) const 
{
    /*
     * Very simple and stupid algorithm, but it works: 
     * 1) Choose a seed point.
     * 2) Find all connected points to that seed.
     * 3) Create a component for them and remove them all from 
     *      unused list.
     */
    auto unusedPoints{ mPoints };
    std::vector<PointSet> components;

    while (!unusedPoints.empty())
    {
        // Start a new component.
        PointSet component{ };

        // From a chosen seed point.
        auto currentPoint{ unusedPoints.back() };

        // Find all connected points: 

        // 1) Find lowest parent, remembering it.
        while (hasPoint(mPoints, tree.getNodeParent(currentPoint)))
        { currentPoint = tree.getNodeParent(currentPoint); }

        // 2) Perform depth-first search, adding to the component.
        std::deque<PointT> queue{ currentPoint };
        while (!queue.empty())
        { // Add all children, recursively.
            currentPoint = queue.back();
            queue.pop_back();
            if (hasPoint(unusedPoints, currentPoint))
            { // Only process new points.
                component.addPoint(currentPoint);
                removePoint(unusedPoints, currentPoint);
                const auto children{ tree.getNodeChildren(currentPoint) };
                for (const auto &child : children)
                { // Add only children which are within the original set.
                    if (hasPoint(mPoints, child))
                    { queue.push_back(child); }
                }
            }
        }

        components.push_back(component);
    }

    return components;
}

PointSet PointSet::pointSupremum(const treeio::ArrayTree &tree) const 
{
    PointSet supremum{ };

    for (const auto &point : mPoints)
    {
        const auto parent{ tree.getNodeParent(point) };
        // Root, is also counted as supremum.
        if (!hasPoint(parent))
        { supremum.addPoint(point); }
    }

    return supremum;
}

PointSet PointSet::pointInfimum(const treeio::ArrayTree &tree) const 
{
    PointSet infimum{ };

    for (const auto &point : mPoints)
    {
        const auto children{ tree.getNodeChildren(point) };
        // Any point whose children are not part of the set is infimum.
        for (const auto &child : children)
        {
            if (pointValid(tree, child) && !hasPoint(child))
            { infimum.addPoint(point); break; }
        }
        // Branch ends are also counted as infimum.
        if (children.empty())
        { infimum.addPoint(point); }
    }

    return infimum;
}

PointSet PointSet::pointBorderSupremum(const treeio::ArrayTree &tree, const PointSet &supremum) const 
{
    PointSet borderSupremum{ };

    for (const auto &point : supremum.data())
    {
        const auto parent{ tree.getNodeParent(point) };
        if (pointValid(tree, parent) && !hasPoint(parent))
        { borderSupremum.addPoint(parent); }
    }

    return borderSupremum;
}

PointSet PointSet::pointBorderInfimum(const treeio::ArrayTree &tree, const PointSet &infimum) const 
{
    PointSet borderInfimum{ };

    for (const auto &point : infimum.data())
    {
        const auto children{ tree.getNodeChildren(point) };
        for (const auto& child : children)
        {
            if (pointValid(tree, child) && !hasPoint(child))
            { borderInfimum.addPoint(child); }
        }
    }

    return borderInfimum;
}

bool PointSet::hasPoint(const InnerSetT &points, const PointT &point)
{
    const auto findIt{ std::find(points.begin(), points.end(), point) };
    return findIt != points.end();
}

bool PointSet::removePoint(InnerSetT &points, const PointT &point)
{
    const auto findIt{ std::find(points.begin(), points.end(), point) };

    if (findIt == points.end())
    { return false; }
    else
    { points.erase(findIt); return true; }
}

bool PointSet::addPoint(InnerSetT &points, const PointT &point)
{
    const auto findIt{ std::find(points.begin(), points.end(), point) };

    if (findIt == points.end())
    { points.push_back(point); return true; }
    else
    { return false; }
}

bool PointSet::pointValid(const treeio::ArrayTree &tree, const PointT &point)
{ return point >= treeio::ArrayTree::INVALID_NODE_ID && tree.isNodeIdValid(point); }

ArrayTreeSelection::ArrayTreeSelection(ArrayTreeHolderPtr tree) :
    mDirty{ false }, mTree{ tree }, mSelectedPoints{ }, mHistory{ { } },
    mVersion{ *this }
{ }

ArrayTreeSelection::~ArrayTreeSelection()
{ /* Automatic */ }

void ArrayTreeSelection::changeTree(ArrayTreeHolderPtr tree)
{
    mTree = tree;
    mSelectedPoints.clear();
    mHistory = OperationHistory<PointSet>{ mSelectedPoints };

    // Update dirty status.
    markDirty();
}

void ArrayTreeSelection::selectPoints(const std::vector<PointT> &points)
{
    // TODO - Remove for performance?
    auto changed{ false };

    // Make a snapshot.
    const auto createdSnapshot{ snapshot() };

    for (const auto &point : points)
    { changed |= mSelectedPoints.addPoint(point); }

    if (!changed)
    { // Remove snapshot when no changed occurred.
        if (createdSnapshot)
        { popSnapshot(); }
    }
    else
    { markDirty(); }
}

void ArrayTreeSelection::unselectPoints(bool noCheckpoint)
{ 
    // Make a snapshot.
    auto createdSnapshot{ false };
    if (!noCheckpoint)
    { createdSnapshot = snapshot(); }

    const auto changed{ mSelectedPoints.clear() }; 

    if (!changed)
    { // Remove snapshot when no changed occurred.
        if (!noCheckpoint && createdSnapshot)
        { mHistory.popState(); }
    }
    else
    { markDirty(); }
}

void ArrayTreeSelection::unselectPoints(const std::vector<PointT> &points)
{
    // TODO - Remove for performance?
    auto changed{ false };

    // Make a snapshot.
    const auto createdSnapshot{ snapshot() };

    for (const auto &point : points)
    { changed |= mSelectedPoints.removePoint(point); }

    if (!changed)
    { // Remove snapshot when no changed occurred.
        if (createdSnapshot)
        { mHistory.popState(); }
    }
    else
    { markDirty(); }
}

bool ArrayTreeSelection::pointSelected(PointT point) const
{ return mSelectedPoints.hasPoint(point); }

bool ArrayTreeSelection::markDirty()
{
    const auto oldDirty{ mDirty };

    mDirty = true;
    mVersion++;

    return oldDirty;
}

bool ArrayTreeSelection::isDirty() const
{ return mDirty; }

bool ArrayTreeSelection::resetDirty()
{ return mDirty = false; }

ArrayTreeSelection::SelectionVersion ArrayTreeSelection::version() const
{ return mVersion; }

void ArrayTreeSelection::treeReloaded()
{
    const auto &translationMap{ mTree->currentTree().translationMap() };

    // Perform translation, if possible. Else just remove all points.
    if (!translationMap.empty())
    { mSelectedPoints.translatePoints(translationMap); }
    else
    { unselectPoints(); }
}

bool ArrayTreeSelection::undo()
{
    // Make a snapshot.
    snapshot();

    // Go to the past state.
    const auto result{ mHistory.undo() };
    if (result)
    { mSelectedPoints = mHistory.currentState(); markDirty(); mLastSavedVersion = mVersion; }

    return result;
}

bool ArrayTreeSelection::redo()
{
    const auto result{ mHistory.redo() };
    if (result)
    { mSelectedPoints = mHistory.currentState(); markDirty(); mLastSavedVersion = mVersion; }
    return result;
}

bool ArrayTreeSelection::moveToState(std::size_t idx)
{ return mHistory.moveToState(idx); }

const ArrayTreeSelection::OperationHistoryList &ArrayTreeSelection::history() const
{ return mHistory.history(); }

bool ArrayTreeSelection::snapshot()
{
    if (mLastSavedVersion < mVersion)
    { mHistory.pushState(mSelectedPoints); mLastSavedVersion = mVersion; return true; }
    else
    { return false; }
}

void ArrayTreeSelection::popSnapshot()
{ mHistory.popState(); }

ArrayTreeHolder &ArrayTreeSelection::currentTreeHolder()
{ return *mTree; }

const treeio::ArrayTree &ArrayTreeSelection::currentTree() const
{ return mTree->currentTree(); }

const PointSet &ArrayTreeSelection::currentlySelected() const
{ return mSelectedPoints; }

PointPosition ArrayTreeSelection::calculateMedian() const
{
    // Sum all of the points.
    PointPosition sum{ };
    for (const auto &pointIdx : mSelectedPoints.data())
    { sum += currentTree().getNode(pointIdx).data().pos; }

    // Return resulting median point.
    if (!mSelectedPoints.empty())
    { return sum / static_cast<float>(mSelectedPoints.size()); }
    else
    { return { }; }
}

void ArrayTreeSelection::colorizeSelected(const treeutil::Color &selected,
    const treeutil::Color &nonSelected, bool dirtifyTree)
{
    auto &currentTree{ mTree->currentTree() };
    if (dirtifyTree)
    { mTree->markDirty(); }

    for (auto iii = currentTree.beginNodeId(); iii < currentTree.endNodeId(); ++iii)
    {
        if (pointSelected(iii))
        { currentTree.getNode(iii).data().pointColor = selected; }
        else
        { currentTree.getNode(iii).data().pointColor = nonSelected; }
    }
}

void ArrayTreeSelection::deltaTranslateSelected(const PointPosition &delta)
{
    auto &currentTree{ mTree->currentTree() };
    mTree->markDirty();

    for (const auto &nodeIdx : mSelectedPoints.data())
    {
        auto &node{ currentTree.getNode(nodeIdx) };
        node.data().pos.x += delta.x;
        node.data().pos.y += delta.y;
        node.data().pos.z += delta.z;
    }
}

void ArrayTreeSelection::updateWidthSelected(float changePercentage)
{
    auto &currentTree{ mTree->currentTree() };
    mTree->markDirty();

    const auto percentageScaler{ 1.0f + changePercentage };

    for (const auto &nodeIdx : mSelectedPoints.data())
    {
        auto &node{ currentTree.getNode(nodeIdx) };
        node.data().thickness *= percentageScaler;
        node.data().freezeThickness = true;
    }
}

void ArrayTreeSelection::unfreezeWidthSelected()
{
    auto &currentTree{ mTree->currentTree() };
    mTree->markDirty();

    for (const auto &nodeIdx : mSelectedPoints.data())
    {
        auto &node{ currentTree.getNode(nodeIdx) };
        node.data().freezeThickness = false;
    }
}

float normalizeTreeScale(treeio::ArrayTree &tree, float currentScale, bool setBaseScale)
{
    const auto bb{ tree.getBoundingBox() };

    const auto maxLength{ bb.size.max() };
    const auto scaleToOne{ 1.0f / (maxLength * currentScale) };

    // Fill in missing runtime meta-data.
    if (!tree.metaData().getRuntimeMetaData<treeio::RuntimeMetaData>())
    { tree.metaData().setRuntimeMetaData(std::make_shared<treeio::RuntimeMetaData>()); }

    auto &props{ tree.metaData().getRuntimeMetaData<treeio::RuntimeMetaData>()->runtimeTreeProperties };
    // TODO *= vs = ?
    props.scaleReconstruction = std::abs(props.scaleReconstruction) * scaleToOne;
    props.scaleGraph = std::abs(props.scaleGraph) * scaleToOne;
    props.scaleReference = std::abs(props.scaleReference) * scaleToOne;

    if (setBaseScale)
    { props.scaleBase = maxLength; }
    else
    { props.scaleBase = 1.0f; }

    return maxLength;
}

}
