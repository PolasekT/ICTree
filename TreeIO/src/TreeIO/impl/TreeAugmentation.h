/**
 * @author Tomas Polasek
 * @date 5.11.2021
 * @version 1.0
 * @brief Tree augmentation utilities.
 */

#ifndef TREE_AUGMENTATION_H
#define TREE_AUGMENTATION_H

#include <TreeIO/TreeIO.h>

#include "TreeUtils.h"
#include "TreeOperations.h"
#include "TreeRandomEngine.h"

namespace treeaug
{

/// @brief Specification of random offset.
struct RandomOffsetSpec
{
    /// Floor value.
    float low{ -1.0f };
    /// Ceiling value.
    float high{ 1.0f };
    /// Strength of operation. Use 0.0f to disable.
    float strength{ 0.0f };
}; // struct RandomOffsetSpec

/// @brief Container for augmentation configuration.
struct AugmenterConfig
{
    /// Seed used for random operations. Set to zero for automatic seed generation.
    int seed{ 0 };
    /// Distribution used for random operations.
    treeutil::RandomDistribution distribution{ treeutil::RandomDistribution::Uniform };
    /// Specify pointer to exiting RNG to use it instead of the internal one.
    treeutil::RandomEngine *rng{ nullptr };

    /// Randomize node offsets within a small surrounding area.
    RandomOffsetSpec nodeDither{ };
    /// Randomized branch width scale.
    RandomOffsetSpec branchDither{ };
    /// Skip leaves for branch dithering?
    bool branchDitherSkipLeaves{ false };
}; // struct AugmenterConfig

/// @brief Container for tree augmentation operations.
class TreeAugmenter
{
public:
    /// @brief Initialize empty TreeAugmenter.
    TreeAugmenter();
    /// @brief Clean up and destroy.
    ~TreeAugmenter();

    /// @brief Augment provided tree using given configuration.
    bool augment(const treeop::ArrayTreeHolder::Ptr &treeHolder, const AugmenterConfig &cfg = { });

    /// @brief Augment provided tree using given configuration.
    bool augment(treeio::ArrayTree &tree, const AugmenterConfig &cfg = { });
private:
    /// @brief Initialize random number generator.
    bool prepareRng(const AugmenterConfig &cfg);

    /// @brief Perform node dithering augmentation. Returns true if any change was performed.
    bool augmentNodeDither(const treeop::ArrayTreeHolder::Ptr &treeHolder, const AugmenterConfig &cfg);
    /// @brief Perform node dithering augmentation. Returns true if any change was performed.
    bool augmentNodeDither(treeio::ArrayTree &tree, const AugmenterConfig &cfg);

    /// @brief Perform branch dithering augmentation. Returns true if any change was performed.
    bool augmentBranchDither(const treeop::ArrayTreeHolder::Ptr &treeHolder, const AugmenterConfig &cfg);
    /// @brief Perform branch dithering augmentation. Returns true if any change was performed.
    bool augmentBranchDither(treeio::ArrayTree &tree, const AugmenterConfig &cfg);

    /// @brief Access the current RNG.
    treeutil::RandomEngine &rng();

    /// Internal randomness generator.
    treeutil::RandomEngine mInternalRng{ };
    /// External randomness generator.
    treeutil::RandomEngine *mExternalRng{ nullptr };
protected:
}; // class TreeAugmenter

}

#endif // TREE_AUGMENTATION_H
