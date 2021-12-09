/**
 * @author Tomas Polasek, David Hrusa
 * @date 1.14.2020
 * @version 1.0
 * @brief gui rendering and helper classes.
 */

#include "TreeStats.h"

#include <queue>

#include <boost/random.hpp>
#include <boost/math/distributions.hpp>

#include "TreeScene.h"
#include "TreeRenderSystemRT.h"

namespace treestat
{

namespace impl
{

/// @brief Internal implementation of the DistributionEngine.
struct DistributionEngineImpl
{
    /// @brief Randomness engine used.
    using EngineT = boost::mt19937;

    /// @brief Type used for variate generators.
    template <typename DistT>
    using VariateGenerator = boost::variate_generator<EngineT, DistT>;

    /// Instance of the engine.
    EngineT engine{ };
}; // struct DistributionEngineImpl

/// @brief Normal Gaussian distribution.
struct NormalDistribution : public DistributionProperties
{
    /// Unique identifier of this distribution.
    static constexpr auto IDENTIFIER{ "Normal" };
    /// @brief Type used for sampling the distribution.
    using DistT = boost::math::normal_distribution<RealType>;

    /// @brief Initialize normal distribution with mean and standard deviation.
    NormalDistribution(DistributionEngineImpl &engine,
        RealType mean = 0, RealType sigma = 1);

    /// @brief Initialize normal distribution from serialized representation.
    NormalDistribution(DistributionEngineImpl &engine,
        const std::string &serialized);

    // Implement interface:
    virtual std::string serialize() const override final;
    virtual void deserialize(const std::string &serialized) override final;
    virtual void setEngine(DistributionEngineImpl &engine) override final;
    virtual RealType cdf(const RealType &x) const override final;
    virtual RealType sample() override final;

    /// @brief Intitialize the internal structure using provided parameters.
    void initialize(RealType mean, RealType sigma);

    /// Distribution implementation.
    DistT distribution{ };
    /// Sample generator for the distribution.
    DistributionEngineImpl::EngineT generator{ };
}; // struct NormalDistribution

NormalDistribution::NormalDistribution(DistributionEngineImpl &engine,
    RealType mean, RealType sigma) :
    generator{ engine.engine }
{ initialize(mean, sigma); }

NormalDistribution::NormalDistribution(DistributionEngineImpl &engine,
    const std::string &serialized) :
    generator{ engine.engine }
{ deserialize(serialized); }

std::string NormalDistribution::serialize() const
{
    return treeio::json{
        { "distribution", IDENTIFIER },
        { "mean", distribution.mean() },
        { "sigma", distribution.standard_deviation() },
    }.dump();
}

void NormalDistribution::deserialize(const std::string &serialized)
{
    const auto data{ treeio::json::parse(serialized) };

    if (data.at("distribution").get<std::string>() != IDENTIFIER)
    { throw StochasticException("Deserializing with unknown distribution identifier!"); }

    initialize(
        data.at("mean").get<RealType>(),
        data.at("sigma").get<RealType>()
    );
}

void NormalDistribution::setEngine(DistributionEngineImpl &engine)
{ generator = engine.engine; }

NormalDistribution::RealType NormalDistribution::cdf(const RealType &x) const
{ return boost::math::cdf<RealType>(distribution, x); }

NormalDistribution::RealType NormalDistribution::sample()
{ return boost::math::quantile<RealType>(distribution, generator()); }

void NormalDistribution::initialize(RealType mean, RealType sigma)
{ distribution = DistT{ mean, sigma }; }



std::shared_ptr<DistributionProperties>
    DistributionProperties::normalDistribution(DistributionEngineImpl &engine,
        RealType mean, RealType sigma)
{ return std::make_shared<NormalDistribution>(engine, mean, sigma); }

std::shared_ptr<DistributionProperties>
    DistributionProperties::deserializeDistribution(
        DistributionEngineImpl &engine, const std::string &serialized)
{
    const auto data{ treeio::json::parse(serialized) };
    const auto name{ data.at("distribution").get<std::string>() };

    if (name == NormalDistribution::IDENTIFIER)
    { return std::make_shared<NormalDistribution>(engine, serialized); }
    else
    { throw StochasticException("Unknown distribution is being deserialized!"); }
}

} // namespace impl

StochasticEngine::StochasticEngine(uint32_t seed)
{ reset(seed); }
StochasticEngine::~StochasticEngine()
{ /* Automatic */ }

void StochasticEngine::reset(uint32_t seed)
{
    if (!mImpl)
    { mImpl = std::make_shared<impl::DistributionEngineImpl>(); }

    if (seed)
    { mImpl->engine.seed(seed); }
    else
    { mImpl->engine.seed(); }
}

std::string ImageData::valueTypeToStr(const ValueType &type)
{
    switch (type)
    {
        default:
        case ValueType::UInt:
        { return "UInt"; }
        case ValueType::Float:
        { return "Float"; }
    }
}

void ImageData::saveTo(treeio::json &out) const
{
    out["image"] = {
        { "width", width },
        { "height", height },
        { "channels", channels },
        { "valueType", valueTypeToStr(valueType) },
        { "data", data.empty() ? std::string{ } : treeutil::encodeBinaryJSON(data) },
    };
}

std::size_t branchingToOrdinal(const Branching &branching)
{
    switch (branching)
    {
        default:
        case Branching::Monopodial:
        { return 0u; }
        case Branching::SympodialMonochasial:
        { return 1u; }
        case Branching::SympodialDichasial:
        { return 2u; }
    }
}

Branching ordinalToBranching(std::size_t ordinal)
{
    switch (ordinal)
    {
        default:
        case 0u:
        { return Branching::Monopodial; }
        case 1u:
        { return Branching::SympodialMonochasial; }
        case 2u:
        { return Branching::SympodialDichasial; }
    }
}

std::size_t ramificationToOrdinal(const Ramification &ramification)
{
    switch (ramification)
    {
        default:
        case Ramification::Continuous:
        { return 0u; }
        case Ramification::Rhythmic:
        { return 1u; }
        case Ramification::Diffuse:
        { return 2u; }
    }
}

Ramification ordinalToRamification(std::size_t ordinal)
{
    switch (ordinal)
    {
        default:
        case 0u:
        { return Ramification::Continuous; }
        case 1u:
        { return Ramification::Rhythmic; }
        case 2u:
        { return Ramification::Diffuse; }
    }
}

void TreeStatValues::saveTo(treeio::json &out) const
{
    forEach([&out] (const auto &var)
    {
        auto &varOut{ out[var.name] };
        varOut["name"] = var.name;
        varOut["description"] = var.description;

        var.var.saveTo(varOut);
        var.hist.saveTo(varOut);
    });

    auto &commonOut{ out["common"] };
    commonOut = {
        { "treeAge", treeAge },
        { "trunkLength", trunkLength },
        { "interNodeLength", interNodeLengthEstimate },
        { "interNodeVariance", interNodeLengthVariance },
        { "leafCount", leafCount },
        { "branching", branchingToOrdinal(branching) },
        { "ramification", ramificationToOrdinal(ramification) },
    };

    auto &visualOut{ out["visual"] };
    shadowImprint.saveTo(visualOut["shadowImprint"]);
}

TreeStats::TreeStats()
{ /* Automatic */ }

TreeStats::TreeStats(const treeio::ArrayTree &tree)
{ calculateStatistics(tree); }

TreeStats::~TreeStats()
{ /* Automatic */ }

void printChains(std::size_t currentChainIdx, const treeutil::TreeChains &chains, std::size_t maxDepth,
    const std::string &indent = "", std::size_t currentDepth = 0u)
{
    const auto &currentChain{ chains.chains()[currentChainIdx] };

    const auto newline{ std::string{ "\n" } + indent + "\t" };
    Info << indent << "[ Chain " << currentChainIdx
        << newline << "depth: " << currentDepth
        << "; nodeCount: " << currentChain.nodes.size()
        << "; gOrder: " << currentChain.graveliusOrder
        << "; gDepth: " << currentChain.graveliusDepth
        << newline << "children (" << currentChain.childChains.size() << "): [ \n";

    if (currentDepth + 1u < maxDepth)
    {
        if (currentChain.childChains.empty())
        { Info << indent << "\tNo Child Chains"; }
        else
        {
            for (const auto &childChainIdx: currentChain.childChains)
            { printChains(childChainIdx, chains, maxDepth, indent + "\t", currentDepth + 1u); Info << "\n"; }
        }
    }
    else
    { Info << indent << "\tMaximum Depth Reached"; }

    Info << newline << " ]"
        << "\n" << indent << " ]";

    if (currentDepth == 0u)
    { Info << std::endl; }
}

void TreeStats::calculateStatistics(const treeio::ArrayTree &tree)
{
    treeutil::Timer profilingTimer{ };

    /* Phase 0 Initialize structures */
    mStats = { };
    if (tree.empty()) { return; }

    /* Phase 1 Prepare helper tree data */
    const auto [chains, chainDataStorage]{ prepareTree(mStats, mSettings, tree) };
    if (chainDataStorage.empty()) { return; }

    const auto timePreparation{ profilingTimer.reset() };

    /* Phase 2 Calculate basic statistics */
    calculateBasicStats(mStats, mSettings, chains, chainDataStorage);

    const auto timeBasicStats{ profilingTimer.reset() };

    /* Phase 3 Calculate visual statistics */
    calculateVisualStats(mStats, mSettings, tree);

    const auto timeVisualStats{ profilingTimer.reset() };

    /* Phase 4 Calculate derived statistics */
    calculateDerivedStats(mStats, mSettings, chains, chainDataStorage);

    const auto timeDerivedStats{ profilingTimer.reset() };

    /* Phase 5 Calculate growth statistics */
    calculateGrowthStats(mStats, mSettings, chains, chainDataStorage);

    const auto timeGrowthStats{ profilingTimer.reset() };

    /* Phase 6 Finalize and clean up */
    finalizeStats(mStats, mSettings);

    const auto timeFinalizeStats{ profilingTimer.reset() };

    const auto timeTotalNoVisual{
        timePreparation + timeBasicStats +
        timeDerivedStats + timeGrowthStats +
        timeFinalizeStats
    };
    const auto timeTotal{ timeTotalNoVisual + timeVisualStats };

    Info << "[Prof] timePreparation: " << timePreparation << std::endl;
    Info << "[Prof] timeBasicStats: " << timeBasicStats << std::endl;
    Info << "[Prof] timeVisualStats: " << timeVisualStats << std::endl;
    Info << "[Prof] timeDerivedStats: " << timeDerivedStats << std::endl;
    Info << "[Prof] timeGrowthStats: " << timeGrowthStats << std::endl;
    Info << "[Prof] timeFinalizeStats: " << timeFinalizeStats << std::endl;
    Info << "[Prof] timeTotalNoVisual: " << timeTotalNoVisual << std::endl;
    Info << "[Prof] timeTotal: " << timeTotal << std::endl;
}

void TreeStats::saveStatisticsToDynamic(treeio::ArrayTree &tree)
{
    auto &dynamic{ tree.metaData().dynamicData() };

    auto &statsOut{ dynamic["stats"] };
    mStats.saveTo(statsOut);
}

std::pair<treeutil::TreeChains, std::vector<TreeStats::ChainData>>
    TreeStats::prepareTree(TreeStatValues &stats, const StatsSettings &settings,
        const treeio::ArrayTree &tree)
{
    treeutil::TreeChains chains{ tree };

    //printChains(0u, chains, 3u);

    if (chains.chains().empty())
    { return { }; }

    std::vector<ChainData> chainDataStorage{ };
    chainDataStorage.resize(chains.chains().size());

    // Calculate forward pass information - root -> leaves.
    prepareTreeForwardPass(chains, chainDataStorage, stats, settings);

    // Calculate backward pass information - leaves -> root.
    prepareTreeBackwardPass(chains, chainDataStorage, stats, settings);

    return { chains, chainDataStorage };
}

void TreeStats::prepareTreeForwardPass(treeutil::TreeChains &chains, std::vector<ChainData> &chainDataStorage,
    TreeStatValues &stats, const StatsSettings &settings)
{
    const auto &chainsTree{ chains.internalTree() };

    for (std::size_t chainIdx = 0u; chainIdx < chains.chains().size(); ++chainIdx)
    { // Go through chains and pre-calculate data.
        const auto &chain{ chains.chains()[chainIdx] };
        auto &chainData{ chainDataStorage[chainIdx] };

        float length{ 0.0f };
        float volume{ 0.0f };
        float angleSum{ 0.0f };
        float angleAbsSum{ 0.0f };
        float lengthAverageThickness{ 0.0f };
        Vector3D startToEnd{ };
        if (chain.nodes.size() > 1u)
        {
            const auto &firstNodeData{ chainsTree.getNode(chain.nodes[0u]).data() };
            const auto &secondNodeData{ chainsTree.getNode(chain.nodes[1u]).data() };
            const auto &lastNodeData{ chainsTree.getNode(chain.nodes.back()).data() };

            chainData.segmentData.reserve(chain.nodes.size() - 1u);
            auto lastNodePos{ firstNodeData.pos };
            auto lastNodeThickness{ firstNodeData.thickness };
            startToEnd = (lastNodeData.pos - lastNodePos);
            auto lastToCurrent{ (secondNodeData.pos - lastNodePos).normalized() };

            chainData.minThickness = std::min<float>(
                chainData.minThickness, firstNodeData.thickness);
            chainData.maxThickness = std::max<float>(
                chainData.maxThickness, firstNodeData.thickness);
            chainData.firstDirection = lastToCurrent;

            for (auto it = chain.nodes.begin() + 1u; it != chain.nodes.end(); ++it)
            {
                const auto currentNodePos{ chainsTree.getNode(*it).data().pos };
                const auto currentDirection{ (currentNodePos - lastNodePos).normalized() };
                const auto lastToCurrentAngle{
                    treeutil::angleBetweenNormVectorsRad<float>(lastToCurrent, currentDirection)
                };

                const auto segmentLength{ lastNodePos.distanceTo(currentNodePos) };
                length += segmentLength;
                angleSum += lastToCurrentAngle;
                angleAbsSum += std::abs(lastToCurrentAngle);

                auto thickness{ chainsTree.getNode(*it).data().thickness };
                if (std::isnan(thickness) || std::isinf(thickness))
                { thickness = 0.0f; }

                volume += treeutil::circularConeFrustumVolume(
                    segmentLength, lastNodeThickness, thickness);

                lengthAverageThickness += segmentLength * thickness;
                stats.segmentThickness.observeSample(thickness);
                chainData.minThickness = std::min<float>(
                    chainData.minThickness, thickness);
                chainData.maxThickness = std::max<float>(
                    chainData.maxThickness, thickness);

                chainData.segmentData.emplace_back(SegmentData{
                    segmentLength, lastNodeThickness, thickness
                });

                lastNodePos = currentNodePos;
                lastNodeThickness = thickness;
                lastToCurrent = currentDirection;

                // Try to find a better first direction, when first few nodes are at the exact same position.
                if (!treeutil::aboveEpsilon(chainData.firstDirection.length()))
                { chainData.firstDirection = lastToCurrent; }
            }

            chainData.lastDirection = lastToCurrent;
        }

        chainData.length = length;
        chainData.angleSum = angleSum;
        chainData.angleAbsSum = angleAbsSum;
        chainData.lengthAverageThickness = lengthAverageThickness;
        chainData.volume = volume;
        chainData.startToEnd = startToEnd;
    }
}

void TreeStats::prepareTreeBackwardPass(treeutil::TreeChains &chains, std::vector<ChainData> &chainDataStorage,
    TreeStatValues &stats, const StatsSettings &settings)
{
    using ChainIdxT = treeutil::TreeChains::NodeChain::ChainIdxT;
    static constexpr auto INVALID_CHAIN_IDX{ treeutil::TreeChains::NodeChain::INVALID_CHAIN_IDX };

    std::vector<std::size_t> finalizedChildChains{ };
    finalizedChildChains.resize(chains.chains().size(), 0u);
    std::queue<ChainIdxT> toProcess{ };
    for (const auto &leafChainIdx : chains.leafChains())
    { toProcess.push(leafChainIdx); }

    while (!toProcess.empty())
    { // Process all chains from leaves to root.
        const auto currentChainIdx{ toProcess.front() }; toProcess.pop();
        auto &currentChain{ chains.chains()[currentChainIdx] };
        auto &currentChainData{ chainDataStorage[currentChainIdx] };

        for (const auto &ccIdx : currentChain.childChains)
        { // Aggregate data from child chains.
            const auto &childChain{ chains.chains()[ccIdx] };
            const auto &childChainData{ chainDataStorage[ccIdx] };

            currentChainData.subtreeLength += childChainData.subtreeLength;
            currentChainData.subtreeVolume += childChainData.subtreeVolume;
            currentChainData.subtreeLeafCount += childChainData.subtreeLeafCount;
        }

        // Set fixed value for leaf chains.
        if (currentChain.childChains.empty())
        {
            currentChainData.subtreeLeafCount = 1u;
            currentChainData.subtreeLength = currentChainData.length;
            currentChainData.subtreeVolume = currentChainData.volume;
        }

        // Check parent node finalization..
        const auto parentChainIdx{ currentChain.parentChain };
        if (parentChainIdx != INVALID_CHAIN_IDX)
        { // We have parent -> Not currently at the root node.
            const auto &parentChain{ chains.chains()[parentChainIdx] };
            finalizedChildChains[parentChainIdx]++;

            // Check if all of the parents children have necessary calculations finished.
            if (finalizedChildChains[parentChainIdx] >= parentChain.childChains.size())
            { // All children are finished -> Add parent for processing.
                toProcess.push(parentChainIdx);
            }
        }
    }
}

void TreeStats::calculateBasicStats(TreeStatValues &stats, const StatsSettings &settings,
    const treeutil::TreeChains &chains, const std::vector<ChainData> &chainDataStorage)
{
    for (std::size_t chainIdx = 0u; chainIdx < chains.chains().size(); ++chainIdx)
    { // Go through the tree using non-branching sequences of segments -> chains.
        const auto &chain{ chains.chains()[chainIdx] };
        const auto &chainData{ chainDataStorage[chainIdx] };

        const auto segmentCount{ static_cast<uint32_t>(chain.nodes.size() - 1u) };
        const auto chainDepth{ static_cast<uint32_t>(chain.chainDepth) };

        stats.segmentsPerChain.observeSample(segmentCount);
        stats.chainsPerDepth.observeSample(chainDepth);
        if (treeutil::aboveEpsilon(chainData.length))
        { stats.chainLength.observeSample(chainData.length); }
        if (treeutil::aboveEpsilon(chainData.startToEnd.length()))
        { stats.chainTotalLength.observeSample(chainData.startToEnd.length()); }
        stats.chainDeformation.observeSample(chainData.angleAbsSum);

        for (const auto &segment : chainData.segmentData)
        { // Approximate volume of each segment with circular cone frustum volume.
            const auto volume{
                treeutil::circularConeFrustumVolume(
                    segment.length, segment.startThickness, segment.endThickness)
            };
            stats.segmentVolume.observeSample({ segment.startThickness, volume });
            stats.segmentVolume.observeSample({ segment.endThickness, volume });
        }

        if (chain.nodes.size() > 2u && treeutil::aboveEpsilon(chainData.startToEnd.length()) &&
            treeutil::aboveEpsilon(chainData.length))
        { stats.chainStraightness.observeSample(chainData.startToEnd.length() / chainData.length); }
        if (treeutil::aboveEpsilon(chainData.startToEnd.length()))
        {
            stats.chainSlope.observeSample(
                treeutil::angleBetweenVectorsRad<float>(
                    chainData.startToEnd, Vector3D{ 0.0f, 1.0f, 0.0f }
                )
            );
        }

        if (chainData.minThickness < std::numeric_limits<float>::max())
        { stats.chainMinThickness.observeSample(chainData.minThickness); }
        if (chainData.maxThickness > std::numeric_limits<float>::min())
        { stats.chainMaxThickness.observeSample(chainData.maxThickness); }
        if (chainData.minThickness < std::numeric_limits<float>::max() &&
            chainData.maxThickness > std::numeric_limits<float>::min())
        { stats.chainMinMaxThicknessRatio.observeSample(chainData.minThickness / chainData.maxThickness); }

        if (chain.parentChain != chain.INVALID_CHAIN_IDX)
        { // Calculate child-parent statistics.
            const auto &parentChain{ chains.chains()[chain.parentChain] };
            const auto &parentChainData{ chainDataStorage[chain.parentChain] };

            const auto parentChildAngle{ treeutil::angleBetweenVectorsRad<float>(
                parentChainData.startToEnd, chainData.startToEnd
            ) };

            if (treeutil::aboveEpsilon(chainData.length) && treeutil::aboveEpsilon(parentChainData.length))
            { stats.chainLengthRatio.observeSample(chainData.length / parentChainData.length); }
            if (treeutil::aboveEpsilon(chainData.angleSum) && treeutil::aboveEpsilon(parentChainData.angleSum))
            { stats.chainAngleSumDelta.observeSample(chainData.angleSum - parentChainData.angleSum); }
            stats.chainParentChildAngle.observeSample(parentChildAngle);

            const auto tropismBranch{ Vector3D::dotProduct(
                (chainData.startToEnd - chainData.firstDirection).normalized(),
                StatsSettings::DEFAULT_HORIZON_UP_DIRECTION.normalized()
            ) };
            stats.chainTropismBranch.observeSample(tropismBranch);

            const auto tropismHorizon{ Vector3D::dotProduct(
                chainData.firstDirection.normalized(),
                StatsSettings::DEFAULT_HORIZON_UP_DIRECTION.normalized()
            ) };
            stats.chainTropismHorizon.observeSample(tropismHorizon);

            const auto tropismParent{ Vector3D::dotProduct(
                    parentChainData.lastDirection.normalized(),
                    chainData.startToEnd.normalized()
            ) };
            stats.chainTropismParent.observeSample(tropismParent);
        }
        if (chain.childChains.size() > 1u)
        { // Calculate child statistics.
            auto minChildAngle{ std::numeric_limits<float>::max() };
            auto maxChildAngle{ std::numeric_limits<float>::lowest() };
            auto minChildAngleTotal{ std::numeric_limits<float>::max() };
            auto maxChildAngleTotal{ std::numeric_limits<float>::lowest() };

            auto minParentAngle{ std::numeric_limits<float>::max() };
            auto maxParentAngle{ std::numeric_limits<float>::lowest() };
            auto minParentAngleTotal{ std::numeric_limits<float>::max() };
            auto maxParentAngleTotal{ std::numeric_limits<float>::lowest() };

            for (std::size_t idx1 = 0u; idx1 < chain.childChains.size(); ++idx1)
            {
                const auto &cChainIdx1{ chain.childChains[idx1] };
                const auto &cChainData1{ chainDataStorage[cChainIdx1] };

                auto minSiblingAngle{ std::numeric_limits<float>::max() };
                auto maxSiblingAngle{ std::numeric_limits<float>::lowest() };
                auto minSiblingAngleTotal{ std::numeric_limits<float>::max() };
                auto maxSiblingAngleTotal{ std::numeric_limits<float>::lowest() };

                const auto parentChainAngle{ treeutil::angleBetweenVectorsRad<float>(
                    chainData.lastDirection, cChainData1.firstDirection)};
                const auto parentChainAngleTotal{ treeutil::angleBetweenVectorsRad<float>(
                    chainData.startToEnd, cChainData1.startToEnd)};

                if (treeutil::aboveEpsilon(cChainData1.firstDirection.length()) &&
                    treeutil::aboveEpsilon(chainData.lastDirection.length()))
                {
                    stats.chainParentAngle.observeSample(parentChainAngle);
                    minParentAngle = std::min<float>(minParentAngle, parentChainAngle);
                    maxParentAngle = std::max<float>(maxChildAngle, parentChainAngle);
                }

                if (treeutil::aboveEpsilon(cChainData1.startToEnd.length()) &&
                    treeutil::aboveEpsilon(chainData.startToEnd.length()))
                {
                    stats.chainParentAngleTotal.observeSample(parentChainAngleTotal);
                    minParentAngleTotal = std::min<float>(minParentAngleTotal, parentChainAngleTotal);
                    maxParentAngleTotal = std::max<float>(maxParentAngleTotal, parentChainAngleTotal);
                }

                for (std::size_t idx2 = 0u; idx2 < chain.childChains.size(); ++idx2)
                {
                    const auto &cChainIdx2{ chain.childChains[idx2] };
                    const auto &cChainData2{ chainDataStorage[cChainIdx2] };

                    const auto childChainAngle{ treeutil::angleBetweenVectorsRad<float>(
                        cChainData1.firstDirection, cChainData2.firstDirection)};
                    const auto childChainAngleTotal{ treeutil::angleBetweenVectorsRad<float>(
                        cChainData1.startToEnd, cChainData2.startToEnd)};

                    if (treeutil::aboveEpsilon(cChainData1.firstDirection.length()) &&
                        treeutil::aboveEpsilon(cChainData2.firstDirection.length()))
                    {
                        stats.chainSiblingAngle.observeSample(childChainAngle);
                        minSiblingAngle = std::min<float>(minSiblingAngle, childChainAngle);
                        maxSiblingAngle = std::max<float>(maxSiblingAngle, childChainAngle);
                    }

                    if (treeutil::aboveEpsilon(cChainData1.startToEnd.length()) &&
                        treeutil::aboveEpsilon(cChainData2.startToEnd.length()))
                    {
                        stats.chainSiblingAngleTotal.observeSample(childChainAngleTotal);
                        minSiblingAngleTotal = std::min<float>(minSiblingAngleTotal, childChainAngleTotal);
                        maxSiblingAngleTotal = std::max<float>(maxSiblingAngleTotal, childChainAngleTotal);
                    }

                    if (treeutil::aboveEpsilon(chainData.subtreeLeafCount))
                    {
                        const auto leafAsymmetry{
                            (cChainData1.subtreeLeafCount - cChainData2.subtreeLeafCount) /
                            static_cast<float>(chainData.subtreeLeafCount)
                        };
                        stats.chainAsymmetryLeaf.observeSample(leafAsymmetry);
                    }

                    if (treeutil::aboveEpsilon(chainData.subtreeLength))
                    {
                        const auto lengthAsymmetry{
                            (cChainData1.subtreeLength - cChainData2.subtreeLength) /
                            static_cast<float>(chainData.subtreeLength)
                        };
                        stats.chainAsymmetryLength.observeSample(lengthAsymmetry);
                    }

                    if (treeutil::aboveEpsilon(chainData.subtreeVolume))
                    {
                        const auto volumeAsymmetry{
                            (cChainData1.subtreeVolume - cChainData2.subtreeVolume) /
                            static_cast<float>(chainData.subtreeVolume)
                        };
                        stats.chainAsymmetryVolume.observeSample(volumeAsymmetry);
                    }
                }
                if (minSiblingAngle < std::numeric_limits<float>::max())
                {
                    stats.chainMinSiblingAngle.observeSample(minSiblingAngle);
                    minChildAngle = std::min<float>(minChildAngle, minSiblingAngle);
                }
                if (maxSiblingAngle > std::numeric_limits<float>::lowest())
                {
                    stats.chainMaxSiblingAngle.observeSample(maxSiblingAngle);
                    maxChildAngle = std::max<float>(maxChildAngle, maxSiblingAngle);
                }
                if (minSiblingAngleTotal < std::numeric_limits<float>::max())
                {
                    stats.chainMinSiblingAngleTotal.observeSample(minSiblingAngleTotal);
                    minChildAngleTotal = std::min<float>(minChildAngleTotal, minSiblingAngleTotal);
                }
                if (maxSiblingAngleTotal > std::numeric_limits<float>::lowest())
                {
                    stats.chainMaxSiblingAngleTotal.observeSample(maxSiblingAngleTotal);
                    maxChildAngleTotal = std::max<float>(maxChildAngleTotal, maxSiblingAngleTotal);
                }
            }

            if (minChildAngle < std::numeric_limits<float>::max())
            { stats.chainMinChildAngle.observeSample(minChildAngle); }
            if (maxChildAngle > std::numeric_limits<float>::lowest())
            { stats.chainMaxChildAngle.observeSample(maxChildAngle); }
            if (minChildAngleTotal < std::numeric_limits<float>::max())
            { stats.chainMinChildAngleTotal.observeSample(minChildAngleTotal); }
            if (maxChildAngleTotal > std::numeric_limits<float>::lowest())
            { stats.chainMaxChildAngleTotal.observeSample(maxChildAngleTotal); }

            if (minParentAngle < std::numeric_limits<float>::max())
            { stats.chainMinParentAngle.observeSample(minParentAngle); }
            if (maxParentAngle > std::numeric_limits<float>::lowest())
            { stats.chainMaxParentAngle.observeSample(maxParentAngle); }
            if (minParentAngleTotal < std::numeric_limits<float>::max())
            { stats.chainMinParentAngleTotal.observeSample(minParentAngleTotal); }
            if (maxParentAngleTotal > std::numeric_limits<float>::lowest())
            { stats.chainMaxParentAngleTotal.observeSample(maxParentAngleTotal); }
        }
    }
}

void TreeStats::calculateGrowthStats(TreeStatValues &stats, const StatsSettings &settings,
    const treeutil::TreeChains &chains, const std::vector<ChainData> &chainDataStorage)
{
    const auto branchingPointDelta{ stats.interNodeLengthEstimate * settings.growthBranchingPointDelta };
    const auto compactChains{ chains.generateCompactChains(branchingPointDelta) };
    const auto &chainsTree{ chains.internalTree() };
    const auto &fullChains{ chains.chains() };

    for (const auto &compactChain : compactChains)
    { // Pre-calculate statistics on the compact chains, skipping leaves.
        if (compactChain.childChains.empty())
        { continue; }

        const auto maxCompactedChainLength{ calculateMaxCompactChainLength(
            fullChains, chainsTree, compactChain) };
        const auto internodesInChain{ maxCompactedChainLength / stats.interNodeLengthEstimate };
        stats.chainInternodeCount.observeSample(internodesInChain);
    }

    // Calculate required statistics from the first run.
    const auto averageInternodeStats{ stats.chainInternodeCount.var.properties() };

    for (const auto &compactChain : compactChains)
    { // Go through all compacted chains, skipping leaves.
        if (compactChain.childChains.empty())
        { continue; }
        if (compactChain.childChains.size() == 1u)
        { Warning << "Found compact chain with one child chain!" << std::endl; }

        std::vector<float> branchContinuationWeights{ };
        std::vector<float> branchSizeWeights{ };

        for (const auto &childChainRec : compactChain.childChains)
        { // Calculate weights for child chains.
            const auto parentChainInfo{ compactChain.compactedChains[childChainRec.originIdx] };
            const auto parentChainIdx{ parentChainInfo.chainIdx };
            const auto &parentChain{ fullChains[parentChainIdx] };
            const auto &parentChainData{ chainDataStorage[parentChainIdx] };

            const auto childChainIdx{ childChainRec.chainIdx };
            const auto &childChain{ fullChains[childChainIdx] };
            const auto &childChainData{ chainDataStorage[childChainIdx] };

            const auto childContinuationWeight{ Vector3D::dotProduct(
                parentChainData.lastDirection, childChainData.firstDirection) };
            branchContinuationWeights.push_back(childContinuationWeight);

            const auto childSizeWeight{ childChainData.subtreeLeafCount };
            branchSizeWeights.push_back(childSizeWeight);
        }

        // Determine branching categories.
        const auto [ mpFitness, mpSameAxis, mpHigherAxis ]{
            calculateMonopodialFitness(branchContinuationWeights, branchSizeWeights) };
        const auto [ smFitness, smSameAxis, smHigherAxis ]{
            calculateSympodialMonochasialFitness(branchContinuationWeights, branchSizeWeights) };
        const auto [ sdFitness, sdSameAxis, sdHigherAxis ]{
            calculateSympodialDichasialFitness(branchContinuationWeights, branchSizeWeights) };

        // Observe branching results.
        stats.chainMonopodialBranchingFitness.observeSample(mpFitness);
        stats.aggregateBranchingFitness.observeSample(
            { branchingToOrdinal(Branching::Monopodial), mpFitness });
        stats.chainSympodialMonochasialBranchingFitness.observeSample(smFitness);
        stats.aggregateBranchingFitness.observeSample(
            { branchingToOrdinal(Branching::SympodialMonochasial), smFitness });
        stats.chainSympodialDichasialBranchingFitness.observeSample(sdFitness);
        stats.aggregateBranchingFitness.observeSample(
            { branchingToOrdinal(Branching::SympodialDichasial), sdFitness });

        // Calculate how many internodes does this chain contain before branching.
        const auto maxCompactedChainLength{ calculateMaxCompactChainLength(
            fullChains, chainsTree, compactChain) };
        const auto internodesInChain{ maxCompactedChainLength / stats.interNodeLengthEstimate };

        // Determine ramification categories.
        const auto continuousFitness{ calculateContinuousFitness(internodesInChain,
            averageInternodeStats.mean, averageInternodeStats.variance,
            averageInternodeStats.min, averageInternodeStats.max) };
        const auto rhythmicFitness{ calculateRhythmicFitness(internodesInChain,
            averageInternodeStats.mean, averageInternodeStats.variance,
            averageInternodeStats.min, averageInternodeStats.max) };
        const auto diffuseFitness{ calculateDiffuseFitness(internodesInChain,
            averageInternodeStats.mean, averageInternodeStats.variance,
            averageInternodeStats.min, averageInternodeStats.max) };

        // Observe ramification results.
        stats.chainContinuousRamificationFitness.observeSample(continuousFitness);
        stats.aggregateRamificationFitness.observeSample(
            { ramificationToOrdinal(Ramification::Continuous), continuousFitness });
        stats.chainRhythmicRamificationFitness.observeSample(rhythmicFitness);
        stats.aggregateRamificationFitness.observeSample(
            { ramificationToOrdinal(Ramification::Rhythmic), rhythmicFitness });
        stats.chainDiffuseRamificationFitness.observeSample(diffuseFitness);
        stats.aggregateRamificationFitness.observeSample(
            { ramificationToOrdinal(Ramification::Diffuse), diffuseFitness });
    }
}

/*(
 * prumer kmene
 * krivost retezu
 * oproti horizontu
 * Minimalni pocet featur ktere koreluji
 * vetvici uhel pro ruzne urovne by mel byt konstantni
 *   vypocitat rozdil oproti prumerne hodnote -> variance
 *   vynechat prvni uroven
 * Vypocitat featury pouze pro uroven 1 a 2
 * Vaha - tluste vetve pod kmenem jsou dulezitejsi.
 *      - vahovat pomoci objemu.
 * Apicalni dominance explicitni feature
 *
 * Pomery bb v shadow obrazu a ve 3D
 *
 * Overleaf:
 *  Tabulka
 *  Delky/tloustky, uhly, krivosti, objemy.
 *  Globalni, lokalni
 */

float TreeStats::calculateMaxCompactChainLength(
    const treeutil::TreeChains::ChainStorage &fullChains,
    const treeutil::TreeChains::InternalArrayTree &chainsTree,
    const treeutil::TreeChains::CompactNodeChain &compactChain)
{
    auto maxCompactedChainLength{ std::numeric_limits<float>::lowest() };

    /// @brief Helper for calculating lengths of compacted chains.
    struct ChainLengthHelper
    {
        /// Next index to be processed.
        std::size_t nextIdx{ 0u };
        /// Currently accumulated length.
        float accumulatedLength{ 0.0f };
    }; // struct ChainLengthHelper

    std::queue<ChainLengthHelper> compactedChainQueue{ };
    compactedChainQueue.push(ChainLengthHelper{ 0u, 0.0f });
    while (!compactedChainQueue.empty())
    {
        const auto chainLengthHelper{ compactedChainQueue.front() }; compactedChainQueue.pop();
        const auto subChainRec{ compactChain.compactedChains[chainLengthHelper.nextIdx] };
        const auto &subChain{ fullChains[subChainRec.chainIdx] };

        const auto chainLength{
            chainLengthHelper.accumulatedLength +
            subChain.calculateChainLength(chainsTree)
        };

        auto foundChild{ false };
        for (std::size_t iii = 0u; iii < compactChain.compactedChains.size(); ++iii)
        {
            const auto &subChainChildRec{ compactChain.compactedChains[iii] };
            if (subChainChildRec.originIdx == chainLengthHelper.nextIdx)
            { foundChild = true; compactedChainQueue.push(ChainLengthHelper{ iii, chainLength }); }
        }

        if (!foundChild)
        { maxCompactedChainLength = std::max(maxCompactedChainLength, chainLength); }
    }

    return maxCompactedChainLength;
}

std::tuple<float, TreeStats::BranchIdxList, TreeStats::BranchIdxList>
TreeStats::calculateMonopodialFitness(
        const std::vector<float> &branchContinuationWeights,
        const std::vector<float> &branchSizeWeights)
{
    /*
     * Monopodial:
     *
     *  \ | /
     *   \|/
     *    |
     *
     *  * One primary branch with high size/continuation - same axis.
     *  * Two or more child branches with the same size weights - higher axis.
     */

    BranchIdxList sameAxis{ };
    BranchIdxList higherAxis{ };

    const auto [minPrimary, argMinPrimary, maxPrimary, argMaxPrimary]{
        treeutil::argMinMax(branchSizeWeights) };
    const auto [minContinuation, argMinContinuation, maxContinuation, argMaxContinuation]{
        treeutil::argMinMax(branchContinuationWeights) };
    const auto allSame{ !treeutil::aboveEpsilon(maxPrimary - minPrimary) };
    const auto primaryBranchIdx{ allSame ? argMaxContinuation : argMaxPrimary };
    const auto maxDifference{
        allSame ?
        1.0f :
        (maxPrimary - minPrimary)
    };

    const auto secondaryBranchCount{ branchSizeWeights.size() - 1u };
    auto meanSecondarySizeWeight{ 0.0f };
    auto secondMaxPrimary{ minPrimary };
    for (std::size_t iii = 0u; iii < branchSizeWeights.size(); ++iii)
    {
        if (iii == primaryBranchIdx) { continue; }
        meanSecondarySizeWeight += branchSizeWeights[iii];
        secondMaxPrimary = std::max(secondMaxPrimary, branchSizeWeights[iii]);
    }
    meanSecondarySizeWeight /= static_cast<float>(secondaryBranchCount);

    const auto primaryBranchFitness{
        (allSame ? 1.0f : ((branchSizeWeights[primaryBranchIdx] - secondMaxPrimary) / maxDifference)) *
        ((branchContinuationWeights[primaryBranchIdx] + 1.0f) / 2.0f)
    };
    sameAxis.push_back(primaryBranchIdx);

    auto secondaryBranchFitness{ 0.0f };
    for (std::size_t iii = 0u; iii < branchSizeWeights.size(); ++iii)
    {
        if (iii == primaryBranchIdx) { continue; }
        const auto branchFitness{ branchSizeWeights[iii] - meanSecondarySizeWeight };
        secondaryBranchFitness += 1.0f - (branchFitness * branchFitness) / (maxDifference * maxDifference);
        higherAxis.push_back(iii);
    }
    secondaryBranchFitness /= static_cast<float>(std::max<std::size_t>(2u, secondaryBranchCount));

    const auto monopodialFitness{ primaryBranchFitness * secondaryBranchFitness };

    return { monopodialFitness, sameAxis, higherAxis };
}

std::tuple<float, TreeStats::BranchIdxList, TreeStats::BranchIdxList>
TreeStats::calculateSympodialMonochasialFitness(
    const std::vector<float> &branchContinuationWeights,
    const std::vector<float> &branchSizeWeights)
{
    /*
     * Sympodial Monochasial:
     *
     *  \ |/
     *   \|
     *    |
     *
     *  * One primary branch - same axis.
     *  * One secondary branch - higher axis.
     */

    BranchIdxList sameAxis{ };
    BranchIdxList higherAxis{ };

    const auto [minPrimary, argMinPrimary, maxPrimary, argMaxPrimary]{
        treeutil::argMinMax(branchSizeWeights) };
    const auto [minContinuation, argMinContinuation, maxContinuation, argMaxContinuation]{
        treeutil::argMinMax(branchContinuationWeights) };
    const auto allSame{ !treeutil::aboveEpsilon(maxPrimary - minPrimary) };
    const auto primaryBranchIdx{ allSame ? argMaxContinuation : argMaxPrimary };
    const auto maxDifference{
        allSame ?
        1.0f :
        (maxPrimary - minPrimary)
    };

    const auto secondaryBranchCount{ branchSizeWeights.size() - 1u };
    auto secondMaxPrimary{ minPrimary };
    std::size_t argSecondMaxPrimary{ 1u };
    for (std::size_t iii = 0u; iii < branchSizeWeights.size(); ++iii)
    {
        if (iii == primaryBranchIdx) { continue; }
        const auto sizeValue{ branchSizeWeights[iii] };
        if (secondMaxPrimary >= sizeValue)
        { secondMaxPrimary = sizeValue; argSecondMaxPrimary = iii; }
    }
    const auto secondaryBranchIdx{ argSecondMaxPrimary };

    const auto primaryBranchFitness{
        (allSame ? 1.0f : ((branchSizeWeights[primaryBranchIdx] - secondMaxPrimary) / maxDifference)) *
        ((branchContinuationWeights[primaryBranchIdx] + 1.0f) / 2.0f)
    };
    sameAxis.push_back(primaryBranchIdx);
    higherAxis.push_back(secondaryBranchIdx);

    auto secondaryBranchFitnessPenalization{ 0.0f };
    for (std::size_t iii = 0u; iii < branchSizeWeights.size(); ++iii)
    {
        if (iii == primaryBranchIdx || iii == secondaryBranchIdx) { continue; }
        const auto branchFitness{ branchSizeWeights[iii] };
        secondaryBranchFitnessPenalization += branchFitness / maxDifference;
        higherAxis.push_back(iii);
    }
    const auto secondaryBranchFitness{
        secondaryBranchCount >= 2u ?
        1.0f - secondaryBranchFitnessPenalization / static_cast<float>(secondaryBranchCount - 1u) :
        1.0f
    };

    const auto sympodialMonochasialFitness{ primaryBranchFitness * secondaryBranchFitness };

    return { sympodialMonochasialFitness, sameAxis, higherAxis };
}

std::tuple<float, TreeStats::BranchIdxList, TreeStats::BranchIdxList>
TreeStats::calculateSympodialDichasialFitness(
    const std::vector<float> &branchContinuationWeights,
    const std::vector<float> &branchSizeWeights)
{
    /*
     * Sympodial Dichasial:
     *
     *  \   /
     *   \|/
     *    |
     *
     *  * Two secondary branches with same size weight, low continuation - higher axis.
     *  * Optional one lower size weight branch - same axis.
     */

    BranchIdxList sameAxis{ };
    BranchIdxList higherAxis{ };

    const auto [minPrimary, argMinPrimary, maxPrimary, argMaxPrimary]{
        treeutil::argMinMax(branchSizeWeights) };
    const auto allSame{ !treeutil::aboveEpsilon(maxPrimary - minPrimary) };
    const auto primaryBranchIdx{ argMaxPrimary };
    const auto maxDifference{
        allSame ?
        1.0f :
        (maxPrimary - minPrimary)
    };

    const auto secondaryBranchCount{ branchSizeWeights.size() - 1u };
    auto secondMaxPrimary{ minPrimary };
    std::size_t argSecondMaxPrimary{ 1u };
    for (std::size_t iii = 0u; iii < branchSizeWeights.size(); ++iii)
    {
        if (iii == primaryBranchIdx) { continue; }
        const auto sizeValue{ branchSizeWeights[iii] };
        if (secondMaxPrimary >= sizeValue)
        { secondMaxPrimary = sizeValue; argSecondMaxPrimary = iii; }
    }
    const auto secondaryBranchIdx{ argSecondMaxPrimary };

    const auto primaryBranchFitness{
        (secondMaxPrimary / branchSizeWeights[primaryBranchIdx]) *
        (1.0f - (std::abs(
            branchContinuationWeights[primaryBranchIdx] -
            branchContinuationWeights[secondaryBranchIdx]
        ) / 2.0f))
    };
    higherAxis.push_back(primaryBranchIdx);
    higherAxis.push_back(secondaryBranchIdx);

    auto secondaryBranchFitnessPenalization{ 0.0f };
    for (std::size_t iii = 0u; iii < branchSizeWeights.size(); ++iii)
    {
        if (iii == primaryBranchIdx || iii == secondaryBranchIdx) { continue; }
        const auto branchFitness{ branchSizeWeights[iii] };
        secondaryBranchFitnessPenalization += (branchFitness / maxDifference) *
                                              ((branchContinuationWeights[iii] + 1.0f) / 2.0f);
        sameAxis.push_back(iii);
    }
    const auto secondaryBranchFitness{
        secondaryBranchCount >= 2u ?
        1.0f - secondaryBranchFitnessPenalization / static_cast<float>(secondaryBranchCount - 1u) :
        1.0f
    };

    const auto sympodialDichasialFitness{ primaryBranchFitness * secondaryBranchFitness };

    return { sympodialDichasialFitness, sameAxis, higherAxis };
}

ImageData copyModality(const treert::RayTracer::OutputModality<float> &modality)
{
    ImageData result{ };

    result.width = modality.width;
    result.height = modality.height;
    result.channels = 1u;
    result.valueType = ImageData::ValueType::Float;

    const auto dataSize{ modality.dataBuffer.size() * sizeof(float) };
    result.data.resize(dataSize);
    std::memcpy(result.data.data(), modality.dataBuffer.data(), dataSize);

    return result;
}

ImageData copyModality(const treert::RayTracer::OutputModality<Vector3D> &modality)
{
    ImageData result{ };

    result.width = modality.width;
    result.height = modality.height;
    result.channels = 3u;
    result.valueType = ImageData::ValueType::Float;

    const auto dataSize{ modality.dataBuffer.size() * sizeof(Vector3D) };
    result.data.resize(dataSize);
    std::memcpy(result.data.data(), modality.dataBuffer.data(), dataSize);

    return result;
}

ImageData copyModality(const treert::RayTracer::OutputModality<uint32_t> &modality)
{
    ImageData result{ };

    result.width = modality.width;
    result.height = modality.height;
    result.channels = 1u;
    result.valueType = ImageData::ValueType::UInt;

    const auto dataSize{ modality.dataBuffer.size() * sizeof(uint32_t) };
    result.data.resize(dataSize);
    std::memcpy(result.data.data(), modality.dataBuffer.data(), dataSize);

    return result;
}

template <typename ModalityT>
ImageData createEmpty(std::size_t w, std::size_t h)
{ ModalityT modality(w, h, 1); return copyModality(modality); }

void TreeStats::calculateVisualStats(TreeStatValues &stats,
    const StatsSettings &settings, const treeio::ArrayTree &inputTree)
{
    if (!settings.visualEnabled)
    {
        stats.shadowImprint = createEmpty<treert::RayTracer::VolumeModality>(1u, 1u);
        return;
    }

    // Prepare ray-tracer with tree reconstruction placed in the scene.
    const auto rayTracerPtr{
        treerndr::RenderSystemRT::prepareTreeRayTracer(
            inputTree, settings.visualTreeScale)
    };
    auto &rayTracer{ *rayTracerPtr };

    // Setup ray-tracer.
    rayTracer.setVerbose(settings.visualVerbose);
    rayTracer.setSampling(settings.visualSampleCount);

    // Prepare modalities.
    rayTracer.traceModalities({ treert::TracingModality::Volume });
    treert::RayTracer::VolumeModality tracedAccumulator{ };

    // Prepare progress printing for long jobs.
    treeutil::ProgressBar progressBar{ "Views " };
    static constexpr auto PROGRESS_PRINT_STEP{ 10u };
    treeutil::ProgressPrinter progressPrinter{ progressBar,
        std::max<std::size_t>(PROGRESS_PRINT_STEP, settings.visualViewCount),
        PROGRESS_PRINT_STEP
    };

    // Ray-trace requested views.
    const auto basePos{ settings.visualTreeScale / 2.0f };
    for (std::size_t viewIdx = 0u; viewIdx < settings.visualViewCount; ++viewIdx)
    { // Ray-trace each view independently.
        const auto t{ viewIdx / static_cast<float>(settings.visualViewCount) };

        // Position the camera.
        const auto cameraPosition{
            Vector3D{
                std::sin(t * 2.0f * treeutil::PI<float>) * basePos,
                basePos,
                std::cos(t * 2.0f * treeutil::PI<float>) * basePos
            }
        };
        const auto cameraFocus{ Vector3D{ 0.0f, basePos, 0.0f } };

        // Prepare ray-tracing context for current view.
        const auto ctx{
            rayTracer.generateOrthoContext(
                settings.visualViewWidth, settings.visualViewHeight,
                cameraPosition, cameraFocus
            )
        };

        // Perform ray tracing.
        rayTracer.traceRays(ctx);

        // Accumulate results.
        const auto &tracedData{ rayTracer.volumeModality() };
        if (tracedAccumulator.dataBuffer.empty())
        { tracedAccumulator = tracedData; }
        else
        {
            for (std::size_t jjj = 0u; jjj < tracedAccumulator.dataBuffer.size(); ++jjj)
            { tracedAccumulator.dataBuffer[jjj] += tracedData.dataBuffer[jjj]; }
        }

        // Report on progress.
        if (settings.visualViewCount >= PROGRESS_PRINT_STEP)
        { progressPrinter.printProgress(Info, viewIdx + 1u); }
    }

    // Finalize processing of modalities.
    for (auto &volume : tracedAccumulator.dataBuffer)
    { volume /= static_cast<float>(settings.visualViewCount); }

    // Save results into buffers.
    stats.shadowImprint = copyModality(tracedAccumulator);

    if (settings.visualExportResults)
    { // Export the results.
        tracedAccumulator.exportToFile(settings.visualExportPath + "/shadowImprint.png");
    }
}

void TreeStats::calculateDerivedStats(TreeStatValues &stats, const StatsSettings &settings,
    const treeutil::TreeChains &chains, const std::vector<ChainData> &chainDataStorage)
{
    const auto &chainLengths{ stats.chainLength.var.observations() };
    // Get trunk chain length, which is always the first.
    stats.trunkLength = chainLengths.empty() ? 0.0f : chainLengths.front();

    // Estimate inter-node length, without including the trunk.
    const auto [lengthMean, lengthVariance]{
        chainLengths.empty() ?
            std::make_pair(0.0f, 0.0f) :
            utils::sampleMeanVariance<float>(
                chainLengths.begin() + 1u,
                chainLengths.end()
            )
    };
    stats.interNodeLengthEstimate = lengthMean;
    stats.interNodeLengthVariance = lengthVariance;

    // Find the deepest leaf chain.
    std::size_t deepestLeafChain{ 0u };
    for (const auto &leafChainIdx : chains.leafChains())
    { deepestLeafChain = std::max(deepestLeafChain, chains.chains()[leafChainIdx].chainDepth); }

    // Estimate crown age by using the deepest leaf chain, without the trunk.
    const auto crownAge{ deepestLeafChain - 1u };
    // Estimate trunk age by dividing it into corresponding inter-node sizes.
    const auto trunkAge{ static_cast<std::size_t>(std::ceil(stats.trunkLength / stats.interNodeLengthEstimate)) };

    // Total age is the age of the crown + age of the trunk...
    stats.treeAge = crownAge + trunkAge;

    // Determine number of leaves.
    stats.leafCount = chains.leafChains().size();
}

float TreeStats::calculateContinuousFitness(float internodeCount, float mean, float var, float min, float max)
{
    /*
     * Continuous ramification is defined by regular branching, specifically
     * 1 internode per branch.
     */

    // Calculate fitness as distance from optimal value - 1 internode per branching.
    const auto diff{ std::abs(1.0f - internodeCount) };
    const auto fitness{ treeutil::smoothstep<float>(1.0f - diff) };

    return fitness;
}

float TreeStats::calculateRhythmicFitness(float internodeCount, float mean, float var, float min, float max)
{
    /*
     * Rhythmic ramification is defined by regular branching, which is other
     * than 1 internode per branch.
     */

    // Calculate regularity as distance from optimal value - average internodes per branching.
    const auto regularityDiff{ std::abs(mean - internodeCount) };
    const auto regularityFitness{ treeutil::smoothstep<float>(1.0f - regularityDiff) };

    // Calculate rhythmic nature as distance from 1 internode per branch.
    const auto rhythmicDiff{ std::abs(1.0f - internodeCount) };
    const auto rhythmicFitness{ treeutil::smoothstep<float>(rhythmicDiff) };

    // Resulting fitness is combination of the two attributes.
    const auto fitness{ regularityFitness * rhythmicFitness };
    return fitness;
}

float TreeStats::calculateDiffuseFitness(float internodeCount, float mean, float var, float min, float max)
{
    /*
     * Diffuse ramification is defined by irregular branching.
     */

    // Calculate irregularity as distance from optimal value - average internodes per branching.
    const auto diff{ std::abs(mean - internodeCount) };
    const auto fitness{ treeutil::smoothstep<float>(diff) };

    return fitness;
}

void TreeStats::finalizeStats(TreeStatValues &stats, const StatsSettings &settings)
{
    // Calculate the rest of the histograms.
    stats.forEach([] (auto &var)
    { var.var.properties(); if (!var.histPrepared) { var.calculateHistogram(MIN_BUCKETS); } });

    // Calculate properties and clean up unnecessary data:
    stats.forEach([] (auto &var)
    // TODO - Clear observations after saving?
    { /* var.var.clearObservations(true); */ });

    // Determine aggregate stats.
    stats.branching = ordinalToBranching(stats.aggregateBranchingFitness.hist.maxBucket().first);
    stats.ramification = ordinalToRamification(stats.aggregateRamificationFitness.hist.maxBucket().first);
}

TreeStats::StatsSettings &TreeStats::settings()
{ return mSettings; }
const TreeStats::StatsSettings &TreeStats::settings() const
{ return mSettings; }

} // namespace treestat
