/**
 * @author Tomas Polasek
 * @date 4.21.2020
 * @version 1.0
 * @brief Template helpers and utilities.
 */

#ifndef TREE_TEMPLATE_UTILS_H
#define TREE_TEMPLATE_UTILS_H

#include <utility>
#include <type_traits>
#include <iterator>

namespace treeutil
{

namespace impl
{

// Source: https://stackoverflow.com/a/29634934
using std::begin;
using std::end;

// Positive case:
template <typename T>
auto is_iterable_impl(int) -> decltype(
        // Test begin, end and operator!= :
        begin(std::declval<T&>()) != end(std::declval<T&>()),
        // Handle overloaded operator, :
        void(),
        // Text operator++ :
        ++std::declval<decltype(begin(std::declval<T&>()))&>(),
        // Text operator* :
        void(*begin(std::declval<T&>())),
        // Resulting type -> It is iterable!
        std::true_type{ }
    );

// Negative case:
template <typename T>
std::false_type is_iterable_impl(...);

} // namespace impl

/// @brief Is given type iterable? Iterable types must have begin, end, operator!= and operator*.
template <typename T>
using is_iterable = decltype(impl::is_iterable_impl<T>(0));

} // namespace treeutil

#endif // TREE_TEMPLATE_UTILS_H

// Template implementation begin.

namespace treeutil
{

} // namespace treeutil

// Template implementation end.
