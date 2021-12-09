/**
 * @author Tomas Polasek
 * @date 5.11.2021
 * @version 1.0
 * @brief Tree augmentation utilities.
 */

#include "TreeAugmentation.h"

#include "TreeReconstruction.h"

namespace treeaug
{

TreeAugmenter::TreeAugmenter()
{ /* Automatic */ }

TreeAugmenter::~TreeAugmenter()
{ /* Automatic */ }

bool TreeAugmenter::augment(const treeop::ArrayTreeHolder::Ptr &treeHolder, const AugmenterConfig &cfg)
{
    if (!prepareRng(cfg))
    { treeop::Warning << "TreeAugmenter failed RNG init, ending prematurely!" << std::endl; return false; }

    auto treeModified{ false };

    const auto nodeDitherResult{ augmentNodeDither(treeHolder, cfg) };
    treeModified |= nodeDitherResult;
    if (cfg.nodeDither.strength > 0.0 && !nodeDitherResult)
    { treeop::Warning << "TreeAugmenter failed node dithering, ending prematurely!" << std::endl; }

    const auto branchDitherResult{ augmentBranchDither(treeHolder, cfg) };
    treeModified |= branchDitherResult;
    if (cfg.branchDither.strength > 0.0 && !branchDitherResult)
    { treeop::Warning << "TreeAugmenter failed branch dithering, ending prematurely!" << std::endl; }

    return treeModified;
}

bool TreeAugmenter::augment(treeio::ArrayTree &tree, const AugmenterConfig &cfg)
{
    if (!prepareRng(cfg))
    { treeop::Warning << "TreeAugmenter failed RNG init, ending prematurely!" << std::endl; return false; }

    auto treeModified{ false };

    const auto nodeDitherResult{ augmentNodeDither(tree, cfg) };
    treeModified |= nodeDitherResult;
    if (cfg.nodeDither.strength > 0.0 && !nodeDitherResult)
    { treeop::Warning << "TreeAugmenter failed node dithering, ending prematurely!" << std::endl; }

    const auto branchDitherResult{ augmentNodeDither(tree, cfg) };
    treeModified |= branchDitherResult;
    if (cfg.branchDither.strength > 0.0 && !branchDitherResult)
    { treeop::Warning << "TreeAugmenter failed branch dithering, ending prematurely!" << std::endl; }

    return treeModified;
}

bool TreeAugmenter::prepareRng(const AugmenterConfig &cfg)
{
    mInternalRng.setDistribution(cfg.distribution);

    if (cfg.seed == 0)
    { mInternalRng.resetSeed(); }
    else
    { mInternalRng.resetSeed(cfg.seed); }

    mExternalRng = cfg.rng;

    return true;
}

bool TreeAugmenter::augmentNodeDither(const treeop::ArrayTreeHolder::Ptr &treeHolder, const AugmenterConfig &cfg)
{
    // Use the current tree.
    const auto result{ augmentNodeDither(treeHolder->currentTree(), cfg) };

    // Note the changes.
    if (result)
    { treeHolder->markDirty(); }

    return result;
}

bool TreeAugmenter::augmentNodeDither(treeio::ArrayTree &tree, const AugmenterConfig &cfg)
{
    if (cfg.nodeDither.strength == 0.0f)
    { return false; }

    // Determine base offset length.
    const auto bb{ tree.getBoundingBox() };
    const auto baseOffset{ bb.diameter() * cfg.nodeDither.strength };

    // Offset nodes:
    for (auto nodeId = tree.beginNodeId(); nodeId < tree.endNodeId(); ++nodeId)
    {
        auto &node{ tree.getNode(nodeId) };
        auto &nData{ node.data() };

        nData.pos += treeutil::Vector3D{
            rng().randomFloat(cfg.nodeDither.low, cfg.nodeDither.high, baseOffset),
            rng().randomFloat(cfg.nodeDither.low, cfg.nodeDither.high, baseOffset),
            rng().randomFloat(cfg.nodeDither.low, cfg.nodeDither.high, baseOffset),
        };
    }

    return true;
}

bool TreeAugmenter::augmentBranchDither(const treeop::ArrayTreeHolder::Ptr &treeHolder, const AugmenterConfig &cfg)
{
    // Use the current tree.
    const auto result{ augmentBranchDither(treeHolder->currentTree(), cfg) };

    // Note the changes.
    if (result)
    { treeHolder->markDirty(); }

    return result;
}

bool TreeAugmenter::augmentBranchDither(treeio::ArrayTree &tree, const AugmenterConfig &cfg)
{
    if (cfg.branchDither.strength == 0.0f)
    { return false; }

    // Determine base offset scale.
    const auto baseScale{ 1.0f + cfg.branchDither.strength };

    // Scale branch width:
    for (auto nodeId = tree.beginNodeId(); nodeId < tree.endNodeId(); ++nodeId)
    {
        auto &parentNode{ tree.getNode(nodeId) };

        if (!cfg.branchDitherSkipLeaves || std::as_const(parentNode).children().size() >= 1)
        {
            const auto powerFactor{ rng().randomFloat(cfg.branchDither.low, cfg.branchDither.high, baseScale) };

            if (parentNode.data().thickness <= treeutil::TreeReconstruction::MIN_BRANCH_THICKNESS)
            { parentNode.data().thickness = treeutil::TreeReconstruction::MIN_BRANCH_THICKNESS; }

            const auto thickness{ std::pow<double>(parentNode.data().thickness, powerFactor) };
            if (thickness <= treeutil::TreeReconstruction::MIN_BRANCH_THICKNESS)
            { parentNode.data().thickness = treeutil::TreeReconstruction::MIN_BRANCH_THICKNESS; }
            else
            { parentNode.data().thickness = static_cast<float>(thickness); }
        }
    }

    return true;
}

treeutil::RandomEngine &TreeAugmenter::rng()
{
    if (mExternalRng)
    { return *mExternalRng; }
    else
    { return mInternalRng; }
}

}
