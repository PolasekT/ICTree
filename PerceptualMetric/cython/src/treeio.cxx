/**
 * @author Tomas Polasek
 * @date 26.4.2021
 * @version 1.0
 * @brief Python wrapper for TreeIO.
 */

#include "treeio.h"

#include <iostream>

int testIntegration(const std::string &s)
{ std::cout << "Hello World: " << s << std::endl; return 42; }

treeio::ArrayTree *treeConstruct()
{ return new treeio::ArrayTree; }

void treeDestroy(treeio::ArrayTree *tree)
{ /* No actions necessary */ }

treeio::ArrayTree *treeFromString(const std::string &serialized)
{
    auto tree{ treeConstruct() };
    *tree = treeio::ArrayTree::fromString<treeio::RuntimeMetaData>(serialized);
    return tree;
}

treeio::ArrayTree *treeFromPath(const std::string &path)
{
    auto tree{ treeConstruct() };
    *tree = treeio::ArrayTree::fromPath<treeio::RuntimeMetaData>(path);
    return tree;
}

bool treeSave(treeio::ArrayTree *tree, const std::string &path)
{ return tree->saveTree(path); }

void treeCopy(treeio::ArrayTree *dst, treeio::ArrayTree *src)
{ *dst = *src; }

bool treeAugment(treeio::ArrayTree *tree,
    treeaug::TreeAugmenter *augmenter,
    int seed, bool normal,
    float nLow, float nHigh, float nStrength,
    float bLow, float bHigh, float bStrength,
    bool skipLeaves)
{
    treeaug::AugmenterConfig cfg{ };
    cfg.seed = seed;
    cfg.distribution = normal ? treeutil::RandomDistribution::Normal : treeutil::RandomDistribution::Uniform;
    cfg.rng = nullptr;
    cfg.nodeDither = { nLow, nHigh, nStrength };
    cfg.branchDither = { bLow, bHigh, bStrength };
    cfg.branchDitherSkipLeaves = skipLeaves;

    return augmenter->augment(*tree, cfg);
}
