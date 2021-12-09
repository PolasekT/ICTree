/**
 * @author Tomas Polasek
 * @date 11.20.2019
 * @version 1.0
 * @brief Operations and helpers on tree skeletons.
 */

#include <cassert>
#include <list>
#include <set>
#include <vector>
#include <stdexcept>

#include "TreeUtils.h"
#include "TreeHistory.h"

#ifndef TREE_OPERATIONS_H
#define TREE_OPERATIONS_H

// Forward declaration.
namespace treerndr
{ struct VaoPack; }

namespace treeop
{

/// @brief Wrapper around an ArrayTree, which allows undo and redo operations.
class ArrayTreeHolder : public treeutil::PointerWrapper<ArrayTreeHolder> 
{
public:
    /// Checkpointing type for the tree changes.
    using TreeVersion = treeutil::VersionCheckpoint<ArrayTreeHolder>;
    /// List of history states.
    using OperationHistoryList = typename OperationHistory<treeio::ArrayTree>::HistoryList;

    /// @brief Initialize wrapper with given tree skeleton.
    ArrayTreeHolder(const treeio::ArrayTree &tree = { });
    /// @brief Free all data.
    ~ArrayTreeHolder();

    // Movable and copyable.
    ArrayTreeHolder(const ArrayTreeHolder &other) = default;
    ArrayTreeHolder(ArrayTreeHolder &&other) = default;
    ArrayTreeHolder &operator=(const ArrayTreeHolder &other) = default;
    ArrayTreeHolder &operator=(ArrayTreeHolder &&other) = default;

    /// @brief Change the active tree to the given tree. Reset history removes all history.
    void changeTree(const treeio::ArrayTree &tree, bool resetHistory = true);

    /// @brief Change the active tree to the given tree. Reset history removes all history.
    void changeTree(treeio::ArrayTree &&tree, bool resetHistory = true);

    /// @brief Create a snapshot of the current skeleton state. Returns success.
    bool snapshot();
    /// @brief Remove the last added snapshot.
    void popSnapshot();

    /// @brief Version of latest history snapshot made for this tree.
    TreeVersion lastSnapshotVersion() const;

    /// @brief Undo and switch to skeleton before the current snapshot. Returns whether tree changed.
    bool undo();
    /// @brief Redo and switch to skeleton after the current snapshot. Returns whether tree changed.
    bool redo();
    /// @brief Try to move to indexed history state. Returns whether tree changed.
    bool moveToState(std::size_t idx);
    /// @brief Access list of history states.
    const OperationHistoryList &history() const;

    /// @brief Access the current state of the tree skeleton.
    treeio::ArrayTree &currentTree();
    /// @brief Access the current state of the tree skeleton.
    const treeio::ArrayTree &currentTree() const;

    /// @brief Mark this skeleton as dirty - a change has been made to it. Returns the old value.
    bool markDirty();

    /// @brief Has this tree skeleton changed its content since last resetDirty()?
    bool isDirty() const;

    /// @brief Reset the dirty status and return the old value.
    bool resetDirty();

    /// @brief Set flag indicating topology of the tree has changed and needs cleaning.
    void setTopologyChanged();

    /// @brief Did the topology change and the tree needs cleaning?
    bool hasTopologyChanged();

    /// @brief Perform tree cleaning and reset topology change flag. This includes reloading the tree! Returns success.
    bool cleanupTree();

    /// @brief Get version of the currently loaded tree.
    TreeVersion version() const;
private:
    /// Dirty flag for the change of the tree skeleton.
    bool mDirty;
    /// Flag indicating topology of the tree has changed and needs cleaning.
    bool mTopologyChanged;
    /// Currently used tree skeleton.
    treeio::ArrayTree mTree;
    /// History states of the skeleton.
    OperationHistory<treeio::ArrayTree> mHistory{ };
    /// Version of the currently loaded tree.
    TreeVersion mVersion{ };
    /// Last version which has been saved to the history.
    TreeVersion mLastSavedVersion{ };
protected:
}; // class ArrayTreeHolder

/// @brief Type used for versioning trees within the ArrayTreeHolder.
using TreeVersion = ArrayTreeHolder::TreeVersion;

/// @brief Set of unique points.
class PointSet
{
public:
    /// Type representing a single point.
    using PointT = treeio::ArrayTree::NodeIdT;
    /// Type of internal representation of the set.
    using InnerSetT = std::vector<PointT>;

    /// @brief Initialize empty point set.
    PointSet();
    /// @brief Initialize from given list of points. Must be unique!
    PointSet(const InnerSetT &source);
    /// @brief Free the point set structures.
    ~PointSet();

    /// @brief Create PointSet from vector, which may contain duplicates.
    static PointSet fromVector(const std::vector<PointT> &input);

    // Copyable and movable.
    PointSet(const PointSet &other) = default;
    PointSet(PointSet &&other) = default;
    PointSet &operator=(const PointSet &other) = default;
    PointSet &operator=(PointSet &&other) = default;

    /// @brief Raw access to the unerlying representation. Change on your own peril!
    InnerSetT &data();
    /// @brief Raw access to the unerlying representation.
    const InnerSetT &data() const;

    /// @brief Does this set contain given point?
    bool hasPoint(const PointT &point) const;

    /// @brief Remove point from this set. Returns true if removed.
    bool removePoint(const PointT &point);

    /// @brief Add point to this set. Returns true if added.
    bool addPoint(const PointT &point);

    /// @brief Remove points from this set. Returns true if removed.
    bool removePoints(const PointSet &points);

    /// @brief Add points to this set. Returns true if added.
    bool addPoints(const PointSet &points);

    /// @brief Translate all points from one indexation to the other. Missing points are removed.
    void translatePoints(const std::map<PointT, PointT> &translationMap);

    /// @brief Clear all selected points. Return true if changes occurred.
    bool clear();

    /// @brief Number of points within the set.
    std::size_t size() const;

    /// @brief Is the point set empty?
    bool empty() const;

    /// @brief Divide this set into independent components.
    std::vector<PointSet> computeComponents(const treeio::ArrayTree &tree) const;

    /// @brief Find supremum of this set - towards the parents.
    PointSet pointSupremum(const treeio::ArrayTree &tree) const;

    /// @brief Find supremum of this set - towards the children.
    PointSet pointInfimum(const treeio::ArrayTree &tree) const;

    /**
     * @brief Find supremum border points not in this set - towards the parents.
     * @param tree Tree to search through.
     * @param supremum Supremum of the points set.
     * @return Returns border supremum set.
     */
    PointSet pointBorderSupremum(const treeio::ArrayTree &tree, const PointSet &supremum) const; 

    /**
     * @brief Find infimum border points not in this set - towards the childern.
     * @param tree Tree to search through.
     * @param infimum Infimum of the points set.
     * @return Returns border infimum list.
     */
    PointSet pointBorderInfimum(const treeio::ArrayTree &tree, const PointSet &infimum) const; 

    /// @brief Is given point a valid point in the target tree?
    static bool pointValid(const treeio::ArrayTree &tree, const PointT &point);
private:
    /// @brief Does given set contain given point?
    static bool hasPoint(const InnerSetT &points, const PointT &point);

    /// @brief Remove point from given set. Returns true if removed.
    static bool removePoint(InnerSetT &points, const PointT &point);

    /// @brief Add point to given set. Returns true if added.
    static bool addPoint(InnerSetT &points, const PointT &point);

    /// List of uqnie points within this set.
    InnerSetT mPoints{ };
protected:
}; // class PointSet

/// @brief Wrapper around position of a single point.
using PointPosition = treeutil::Vector3D;

/**
 * @brief Wrapper around an ArrayTree, which allows selection of
 * points and various operations.
 */
class ArrayTreeSelection : public treeutil::PointerWrapper<ArrayTreeSelection> 
{
public:
    /// Checkpointing type for the tree changes.
    using SelectionVersion = treeutil::VersionCheckpoint<ArrayTreeSelection>;
    /// List of history states.
    using OperationHistoryList = typename OperationHistory<PointSet>::HistoryList;

    /// Type representing a single point.
    using PointT = PointSet::PointT;
    /// Pointer to tree skeleton used when performing selection.
    using ArrayTreeHolderPtr = ArrayTreeHolder::Ptr;

    /// @brief Initialize wrapper with given tree skeleton.
    ArrayTreeSelection(ArrayTreeHolderPtr tree);
    /// @brief Free all data.
    ~ArrayTreeSelection();

    /// @brief Change the active tree to the given tree. This resets the selection.
    void changeTree(ArrayTreeHolderPtr tree);

    /**
     * @brief Select points by their indices.
     * @param points List of points to select.
     */
    void selectPoints(const std::vector<PointT> &points);

    /// @brief Unselect all points.
    void unselectPoints(bool noCheckpoint = false);

    /**
     * @brief Unselect points by their indices.
     * @param points List of points to unselect.
     */
    void unselectPoints(const std::vector<PointT> &points);

    /// @brief Is given point currently selected?
    bool pointSelected(PointT point) const;

    /// @brief Mark this point selection as dirty - it's content has been changed. Returns the old value.
    bool markDirty();

    /// @brief Has this point set changed its content since last resetDirty()?
    bool isDirty() const;

    /// @brief Reset the dirty status and return the old value.
    bool resetDirty();

    /// @brief Get version of the currently loaded tree.
    SelectionVersion version() const;

    /// @brief Perform necessary actions if the current tree has been reloaded.
    void treeReloaded(); 

    /**
     * @brief Undo the last operation.
     * @return Returns true if the selection is changed.
     */
    bool undo();

    /**
     * @brief Redo the last operation.
     * @return Returns true if the selection is changed.
     */
    bool redo();
    /// @brief Try to move to indexed history state. Returns whether selection changed.
    bool moveToState(std::size_t idx);
    /// @brief Access list of history states.
    const OperationHistoryList &history() const;

    /// @brief Create a snapshot of the current selection state. Returns success.
    bool snapshot();
    /// @brief Remove the last added snapshot.
    void popSnapshot();

    /// @brief Get current tree holder.
    ArrayTreeHolder &currentTreeHolder();

    /// @brief Get current state of the tree.
    const treeio::ArrayTree &currentTree() const;

    /// @brief Get current set of selected points.
    const PointSet &currentlySelected() const;

    /// @brief Calculate median point for the currently selected points. Returns origin when selection is empty.
    PointPosition calculateMedian() const;

    /**
     * @brief Colorize points on the tree based on whether
     * they are selected.
     * @param selected Color used for selected points.
     * @param nonSelected Color used for non-selected points.
     * @param dirtifyTree Mark the inside tree as dirty.
     */
    void colorizeSelected(const treeutil::Color &selected,
        const treeutil::Color &nonSelected, bool dirtifyTree = false);

    /// @brief Translate all selected points by given delta.
    void deltaTranslateSelected(const PointPosition &delta);

    /// @brief Update widths of all selected points by given percentage.
    void updateWidthSelected(float changePercentage);

    /// @brief Unfreeze widths of all selected points.
    void unfreezeWidthSelected();
private:
    /// Dirty flag for the change of the selected points.
    bool mDirty;
    /// Currently used tree skeleton.
    ArrayTreeHolderPtr mTree;
    /// List of unique point which are currently selected.
    PointSet mSelectedPoints;
    /// History of applied operations.
    OperationHistory<PointSet> mHistory;
    /// Version of the current tree selection.
    SelectionVersion mVersion;
    /// Last version which has been saved to the history.
    SelectionVersion mLastSavedVersion{ };
protected:
}; // class ArrayTreeSelection

/// @brief Type used for versioning selections within the ArrayTreeSelection.
using TreeSelectionVersion = ArrayTreeSelection::SelectionVersion;

/// @brief Normalize scales of given tree and return its original base scale.
float normalizeTreeScale(treeio::ArrayTree &tree, float currentScale = 1.0f, bool setBaseScale = true);

} // namespace treeop

// Template implementation begin.

// Template implementation end.

#endif // TREE_OPERATIONS_H
