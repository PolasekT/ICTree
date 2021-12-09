/**
 * @author Tomas Polasek
 * @date 11.20.2019
 * @version 1.0
 * @brief Helper for keeping history states of a given instance.
 */

#include <cassert>
#include <list>
#include <set>
#include <vector>
#include <stdexcept>

#ifndef TREE_HISTORY_H
#define TREE_HISTORY_H

namespace treeop
{

/**
 * @brief Wrapper around history states for a given tree.
 * @tparam T Type of object being backed up.
 */
template <typename T>
class OperationHistory
{
public:
    /// @brief Wrapper around a state.
    struct HistoryState
    {
        /// State of the tree at this point.
        T state;
    }; // struct HistoryState
    using HistoryList = std::list<HistoryState>;

    /// Default number of history states kept in the history.
    static constexpr std::size_t DEFAULT_KEPT_HISTORY{ 32u };

    /// @brief Initialize empty history. Current state is the given instance.
    explicit OperationHistory(const T &current = { }, std::size_t keptHistory = DEFAULT_KEPT_HISTORY);
    /// @brief Free all history states.
    ~OperationHistory() = default;

    // Copy operators: 
    OperationHistory(const OperationHistory &other);
    OperationHistory &operator=(const OperationHistory &other);

    /**
     * @brief Push a new history state overwriting any
     * future state.
     * @param tree Tree state to push.
     */
    void pushState(const T &tree);

    /// @brief Pop the last added history state.
    void popState();

    /**
     * @brief Access the current state without modification.
     * @return Returns the current state.
     * @warning Returned reference is valid until this state
     * is overwritten.
     */
    const T &currentState() const;

    /**
     * @brief Return one state to the past, keeping all of
     * the future state intact. Pushing a new state overwrites
     * all of the future state!
     * @return Returns true if the operation changed the current 
     * state. If no movement has been made the returned value is
     * false.
     */
    bool undo();

    /**
     * @brief Move one state to the future, keeping all of
     * the future state intact.
     * @return Returns true if the operation changed the current 
     * state. If no movement has been made the returned value is
     * false.
     */
    bool redo();

    /**
     * @brief Try to move to state with given index, returning success.
     * @param idx Index of history state to return to.
     * @return Returns whether change occurred.
     */
    bool moveToState(std::size_t idx);

    /// @brief Get current state index in the history list.
    std::size_t currentStateIdx() const;

    /// @brief Get number of remembered states.
    std::size_t stateCount() const;

    /// @brief Access the list of history state.
    const HistoryList &history() const;
private:
    using HistoryListIt = typename HistoryList::const_iterator;

    /// History of the tree states.
    HistoryList mHistory;
    /// Iterator pointing to the current history state.
    HistoryListIt mCurrentState;
    /// Maximum Number of states held at one moment.
    std::size_t mKeptHistory;
protected:
}; // class OperationHistory

} // namespace treeop

// Template implementation begin.
namespace treeop
{

template <typename T>
OperationHistory<T>::OperationHistory(const T &current, std::size_t keptHistory) :
    mHistory{ }, mCurrentState{ }, mKeptHistory{ keptHistory }
{
    mHistory.push_back({ current });
    mCurrentState = mHistory.begin();
}

template <typename T>
OperationHistory<T>::OperationHistory(const OperationHistory<T> &other) :
    OperationHistory<T>()
{ *this = other; }

template <typename T>
OperationHistory<T> &OperationHistory<T>::operator=(const OperationHistory<T> &other)
{
    mHistory = other.mHistory;
    mCurrentState = mHistory.begin();
    std::advance(mCurrentState, std::distance(other.mHistory.begin(), other.mCurrentState));

    return *this;
}

template <typename T>
void OperationHistory<T>::pushState(const T &tree)
{
    if (mCurrentState != std::prev(mHistory.end()))
    { // Delete all future state.
        const auto deleteBeginIt{ std::next(mCurrentState) };
        mHistory.erase(deleteBeginIt, mHistory.end());
    }

    // Push the new state and move to it.
    mHistory.push_back({ tree });
    mCurrentState = std::prev(mHistory.end());

    // Keep history within specified size.
    if (mHistory.size() > mKeptHistory)
    { mHistory.pop_front(); }
}

template <typename T>
void OperationHistory<T>::popState()
{
    // Pop the last state and move to the new last.
    mHistory.pop_back();
    mCurrentState = std::prev(mHistory.end());
}

template <typename T>
const T &OperationHistory<T>::currentState() const
{ return mCurrentState->state; }

template <typename T>
bool OperationHistory<T>::undo()
{
    if (mCurrentState == mHistory.begin() || mHistory.empty())
    { return false; }
    else
    { mCurrentState = std::prev(mCurrentState); return true; }
}

template <typename T>
bool OperationHistory<T>::redo()
{
    if (mCurrentState == std::prev(mHistory.end()))
    { return false; }
    else
    { mCurrentState = std::next(mCurrentState); return true; }
}

template <typename T>
bool OperationHistory<T>::moveToState(std::size_t idx)
{
    if (idx >= stateCount())
    { return false; }
    else
    { mCurrentState = std::next(mHistory.begin(), idx); return true; }
}

template <typename T>
std::size_t OperationHistory<T>::currentStateIdx() const
{ return std::distance(mHistory.begin(), mCurrentState); }

template <typename T>
std::size_t OperationHistory<T>::stateCount() const
{ return mHistory.size(); }

template <typename T>
const typename OperationHistory<T>::HistoryList &OperationHistory<T>::history() const
{ return mHistory; }

} // namespace treeop

// Template implementation end.

#endif // TREE_HISTORY_H