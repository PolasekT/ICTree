/**
 * @author Tomas Polasek
 * @date 26.4.2021
 * @version 1.0
 * @brief Python wrapper for TreeIO.
 */

#ifndef PERCEPTUALMETRIC_TREEIO_H
#define PERCEPTUALMETRIC_TREEIO_H

#include <string>

#include <TreeIO/TreeIO.h>
#include <TreeIO/Tree.h>
#include <TreeIO/Render.h>
#include <TreeIO/Augment.h>

/// @brief Simple testing function used for testing Cython bindings.
int testIntegration(const std::string &s);

/// @brief Create a new ArrayTree.
treeio::ArrayTree *treeConstruct();
/// @brief Destroy an ArrayTree.
void treeDestroy(treeio::ArrayTree *tree);
/// @brief Parse ArrayTree from string in the .tree format.
treeio::ArrayTree *treeFromString(const std::string &serialized);
/// @brief Parse ArrayTree from file path using the .tree format.
treeio::ArrayTree *treeFromPath(const std::string &path);
/// @brief Save ArrayTree to file path using the .tree format.
bool treeSave(treeio::ArrayTree *tree, const std::string &path);
/// @brief Copy tree data from source to the destination.
void treeCopy(treeio::ArrayTree *dst, treeio::ArrayTree *src);
/// @brief Augment input tree model by dithering its nodes. Returns true in case of changes.
bool treeAugment(treeio::ArrayTree *tree,
    treeaug::TreeAugmenter *augmenter,
    int seed, bool normal,
    float nLow, float nHigh, float nStrength,
    float bLow, float bHigh, float bStrength,
    bool skipLeaves);

#endif // PERCEPTUALMETRIC_TREEIO_H
