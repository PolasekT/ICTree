/**
 * @author Tomas Polasek
 * @date 26.3.2020
 * @version 1.0
 * @brief Mathematic utilities and helpers.
 */

#ifndef TREE_MATH_H
#define TREE_MATH_H

#include <cmath>
#include <random>

namespace treeutil
{

/// @brief Compare 2 variables and return the larger one.
template <typename T1, typename T2>
const T1 &max(const T1 &first, const T2 &second);

/// @brief Calculate ceil and return value of requested type.
template <typename T, typename InT>
T ceil(const InT &val)
{ return static_cast<T>(std::ceil(val)); }

/// @brief Calculate floor and return value of requested type.
template <typename T, typename InT>
T floor(const InT &val)
{ return static_cast<T>(std::floor(val)); }

/**
 * Wrap value between lower and higher bounds.
 * @tparam T Type of the value.
 * @param val Input value.
 * @param low Lower bound, inclusive.
 * @param high Higher bound, inclusive.
 * @return Returns wrapped value.
 */
template <typename T, typename E>
E wrapValue(const E &val, const E &low, const E &high);

} // namespace treeutil

// Template implementation.

namespace treeutil
{

template <typename T1, typename T2>
const T1 &max(const T1 &first, const T2 &second)
{ return (first > second) ? first : second; }

// Specialization for integers.
template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type \
    wrapValue(const T &val, const T &low, const T &high)
{
    T v{ (val - low) % (high - low) };
    if (v < T(0)) { v = high + v; }
    return low + v;
}

// Specialization for floats.
template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type \
    wrapValue(const T &val, const T &low, const T &high)
{
    T v{ std::fmod((val - low), (high - low)) };
    if (v < T(0)) { v = high + v; }
    return low + v;
}

} // namespace treeutil

// Template implementation end.

#endif // TREE_MATH_H
