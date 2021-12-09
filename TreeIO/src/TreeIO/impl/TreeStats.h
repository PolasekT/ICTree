/**
 * @author Tomas Polasek, David Hrusa
 * @date 1.14.2020
 * @version 1.0
 * @brief Tree statistics wrapper.
 */

#ifndef TREE_STAT_H
#define TREE_STAT_H

#include "TreeUtils.h"

namespace treestat
{

/// @brief Wrapper around histogram data.
template <typename CT = uint32_t, typename BT = uint32_t>
class TreeHistogram
{
public:
    /// @brief Initialize histogram while setting only the step size.
    TreeHistogram(const BT &step = 1);

    /// @brief Initialize histogram with given data.
    TreeHistogram(const BT &min, const BT &max,
        const BT &step = 1,
        std::size_t minBuckets = 8u, std::size_t maxBuckets = 32u,
        bool moveMin = false, bool moveMax = true);

    /// @brief Initialize histogram with given data.
    template <typename ForwardItT>
    TreeHistogram(const ForwardItT &first, const ForwardItT &last,
        const BT &min = { }, const BT &max = { },
        const BT &step = 1,
        std::size_t minBuckets = 8u, std::size_t maxBuckets = 32u,
        bool moveMin = false, bool moveMax = true);

    /// @brief Clean up and destroy.
    ~TreeHistogram();

    /// @brief Clear this histogram and reset it to default state.
    void clear(const BT &step = 1);

    /// @brief Clear only histogram values, without resetting the buckets.
    void clearValues();

    /**
     * @brief Initialize buckets for given data.
     *
     * This process is optional. Using this method pre-initializes buckets, but they
     * may still grow, when getBucket() is called for values outside of <min, max>
     * interval.
     *
     * If given step does not divide interval into integral number of buckets, then
     * the max/min will be moved to the first larger value satisfying this property.
     *
     * @param min Minimum expected value.
     * @param max Maximum expected value.
     * @param step Step per bucket.
     * @param minBuckets Minimum number of buckets.
     * @param maxBuckets Maximum number of buckets.
     * @param moveMin Set to true to move min value to satisfy bucket requirements.
     * @param moveMax Set to true to move max value to satisfy bucket requirements.
     * @warning This operation resets all buckets and clears the histogram!
     * @throws runtime_error Thrown when moveMin == moveMax == false and the
     * requirements are not satisfied.
     */
    void initializeBuckets(const BT &min, const BT &max,
        const BT &step = 1,
        std::size_t minBuckets = 8u, std::size_t maxBuckets = 32u,
        bool moveMin = false, bool moveMax = true);

    /// @brief Get bucket for given value. Initialization of buckets is optional.
    CT &getBucket(const BT &value, bool create = true);

    /// @brief Get bucket for given value. Initialization of buckets is optional.
    CT &getBucket(const BT &value) const;

    /// @brief Count all data in the input interval.
    template <typename ForwardItT>
    void countData(const ForwardItT &first, const ForwardItT &last);

    /// @brief Get list of current buckets, <a, b).
    const std::vector<BT> buckets() const;
    /// @brief Get list of current counts.
    const std::vector<CT> &histogram() const;

    /// @brief Get bucket <a, b) containing the minimum. When multiple exist, the last will be returned.
    std::pair<BT, BT> minBucket() const;
    /// @brief Get bucket <a, b) containing the maximum. When multiple exist, the last will be returned.
    std::pair<BT, BT> maxBucket() const;

    /// @brief Print description of this histogram.
    void describe(std::ostream &out, const std::string &indent = "") const;

    /// @brief Serialize content of this histogram into given json.
    void saveTo(treeio::json &out) const;
private:
    /// @brief Get float value using the histogram data.
    static float floatGetter(void *data, int index)
    { return static_cast<float>(reinterpret_cast<CT*>(data)[index]); }

    /// @brief Get bucket index for given value. Creates new bucket if not available.
    std::size_t getCreateBucketIdx(const BT &value);

    /// @brief Get bucket index for given value or histogram.size() if the bucket does not exist.
    std::size_t getBucketIdx(const BT &value) const;

    /// @brief Calculated moved interval end-points using given requirements.
    static std::pair<BT, BT> calculateMovedInterval(
        const BT &min, const BT &max, bool moveMin, bool moveMax,
        const BT &intervalDelta);

    /// Currently used step per bucket.
    BT mStep{ 1 };
    /// List of bucket borders.
    std::vector<BT> mBuckets{ };
    /// Current histogram data.
    std::vector<CT> mHistogram{ };
protected:
}; // class TreeHistogram

/// @brief Automated observing and statistic calculation of a single variable.
template <typename VT = float>
class VariableObserver
{
public:
    /// @brief Helper for extracting true value type.
    template <typename ValueT, bool IsPair>
    struct ValueExtractor;

    /// @brief Extract value type from a paired value.
    template <typename ValueT>
    struct ValueExtractor<ValueT, true>
    {
        static constexpr auto is_pair{ true };
        using type = typename ValueT::first_type;
        static const type &value(const ValueT &v)
        { return v.first; }
    }; // struct ValueExtractor<ValueT, true>
    /// @brief Extract value type from a simple value.
    template <typename ValueT>
    struct ValueExtractor<ValueT, false>
    {
        static constexpr auto is_pair{ false };
        using type = ValueT;
        static const type &value(const ValueT &v)
        { return v; }
    }; // struct ValueExtractor<ValueT, false>

    /// @brief Helper used for value extraction.
    using ValueExtractorT = ValueExtractor<VT, treeutil::is_specialization_v<VT, std::pair>>;
    /// @brief Real value type of the stochastic variable.
    using ValueT = typename ValueExtractorT::type;

    /// @brief Initialize empty observer.
    VariableObserver();
    /// @brief Initialize empty observer with explicit observation keeping.
    VariableObserver(bool keepObservations);
    /// @brief Clean up and destroy.
    ~VariableObserver();

    // Observation:
    /**
     * @brief Add external sample to data accumulator of this variable.
     * @param value New observed sample.
     */
    void observeSample(const VT &value);

    /// @brief Add all external samples in the provided interval.
    template <typename ForwardIteratorT>
    void observeSamples(const ForwardIteratorT &first, const ForwardIteratorT &last);

    // Utilities:
    /// @brief Clear all observations and reset to default state.
    void clear();

    /// @brief Only clear observations, keeping statistics.
    void clearObservations();

    /// @brief Set to true to keep observations from observeSample*() in the accumulator.
    void setKeepObservations(bool keepObservations);

    // Statistics:
    /// @brief Total number of observed samples.
    std::size_t count() const;
    /// @brief Total minimum of all observed values.
    const VT &min() const;
    /// @brief Total maximum of all observed values.
    const VT &max() const;
    /// @brief List of kept observation values.
    const std::vector<VT> &values() const;

    /// @brief Print description of this observer.
    void describe(std::ostream &out, const std::string &indent = "") const;

    /// @brief Serialize content of this observer into given json.
    void saveTo(treeio::json &out) const;
private:
    /// @brief Container for observed statistics.
    struct Statistics
    {
        /// @brief Process new value statistics.
        void newValue(const VT &value);

        /// Current observation count.
        std::size_t count{ 0u };
        /// Total minimum of all observed values.
        VT min{ std::numeric_limits<VT>::max() };
        /// Total maximum of all observed values.
        VT max{ std::numeric_limits<VT>::min() };
    }; // struct Statistics

    /// Statistics for the observed variable.
    Statistics mStats{ };
    /// List of past observed values.
    std::vector<VT> mValues{ };
    /// Keep observations in the accumulator?
    bool mKeepObservations{ false };
protected:
}; // struct VariableObserver

/// @brief Exception thrown when error occurs in stochastic classes.
struct StochasticException : public std::runtime_error
{
    StochasticException(const std::string &msg):
        std::runtime_error{ msg } { }
}; // struct StochasticException

/// @brief Enumeration of available probability distribution types
enum class DistributionType
{
    // Normal Gaussian distribution.
    Normal
}; // DistributionType

namespace impl
{

/// @brief Internal implementation of the DistributionEngine.
struct DistributionEngineImpl;

/// @brief Properties of a stochastic distribution
struct DistributionProperties
{
    /// @brief Type used for real typed distribution values.
    using RealType = double;

    // Interface:

    /// @brief Serialize properties of this distribution.
    virtual std::string serialize() const = 0;
    /// @brief Deserialize properties and setup this distribution.
    virtual void deserialize(const std::string &serialized) = 0;

    /// @brief Set given randomness engine as current.
    virtual void setEngine(DistributionEngineImpl &engine) = 0;

    /// @brief Get cumulative distribution function value at x.
    virtual RealType cdf(const RealType &x) const = 0;

    /// @brief Sample a value from the distribution.
    virtual RealType sample() = 0;

    // Factory methods:

    /// @brief Construct distribution using provided parameters.
    template <DistributionType Type, typename... CArgTs>
    static std::shared_ptr<DistributionProperties> constructDistribution(
        DistributionEngineImpl &engine, CArgTs... cArgs);
    /// @brief Construct distribution using provided parameters.
    template <typename... CArgTs>
    static std::shared_ptr<DistributionProperties> constructDistribution(
        DistributionEngineImpl &engine, DistributionType type, CArgTs... cArgs);
    /// @brief Automatically detect distribution and de-serialize.
    static std::shared_ptr<DistributionProperties> deserializeDistribution(
        DistributionEngineImpl &engine, const std::string &serialized);

    /// @brief Construct distribution with given properties.
    static std::shared_ptr<DistributionProperties> normalDistribution(
        DistributionEngineImpl &engine, RealType mean = 0, RealType sigma = 1);
};

} // namespace impl

/// @brief Wrapper around a uniform random number generator.
class StochasticEngine
{
public:
    /// @brief Initialize the distribution engine. Use seed = 0u for auto-detect.
    StochasticEngine(uint32_t seed = 0u);
    /// @brief Clean up and destroy.
    ~StochasticEngine();

    /// @brief Reset the engine with given seed. Use seed = 0u for auto-detect.
    void reset(uint32_t seed = 0u);
private:
    template <typename VT>
    friend class StochasticDistribution;

    /// @brief Internal implementation.
    std::shared_ptr<impl::DistributionEngineImpl> mImpl{ };
protected:
}; // class StochasticEngine

/// @brief Wrapper around a probability distribution function.
template <typename VT = float>
class StochasticDistribution
{
public:
    /// @brief Initialize invalid stochastic distribution.
    StochasticDistribution();
    /**
     * @brief Construct distribution with given properties.
     *
     * Distributions have following parameters, in order:
     *  * Normal:
     *    * Mean - Mean mu.
     *    * Sigma - Standard deviation sigma.
     *
     * @param cArgs Arguments passed to the distribution constructor.
     */
    template <typename... CArgTs>
    StochasticDistribution(DistributionType type, CArgTs... cArgs);

    /// @brief Clean up and destroy.
    ~StochasticDistribution();

    // Engine manipulation:
    /// @brief Reset the engine with given seed. Use value = 0u for auto-detection.
    void seed(uint32_t value = 0u);

    // Distribution manipulation:
    /// @brief Set distribution according to provided parameters. For details see the constructor.
    template <typename... CArgTs>
    void setDistribution(DistributionType type, CArgTs... cArgs);

    /// Save current distribution settings to string.
    std::string saveDistribution(const std::string &distribution);
    /// Load distribution settings from string.
    void loadDistribution(const std::string &distribution);

    // Operators:
    /// @brief Get cumulative distribution function value at x.
    VT cdf(const VT &x) const;

    /// @brief Sample a value from the distribution.
    VT sample();
private:
    /// @brief Chec we have a valid distribution else throw error.
    void checkDistributionValidThrow() const;

    /// Engine used for randomness generation.
    StochasticEngine mEngine{ };
    /// Currently used distribution
    std::shared_ptr<impl::DistributionProperties> mDistribution{ };
protected:
}; // class StochasticDistribution

namespace utils
{

/// @brief Calculate mean for given data.
template <typename ReturnT, typename ForwardIteratorT>
ReturnT mean(const ForwardIteratorT &first, const ForwardIteratorT &last);

/// @brief Calculate variance for given data.
template <typename ReturnT, typename ForwardIteratorT>
ReturnT variance(const ForwardIteratorT &first, const ForwardIteratorT &last);

/// @brief Calculate standard deviation for given data.
template <typename ReturnT, typename ForwardIteratorT>
ReturnT stddev(const ForwardIteratorT &first, const ForwardIteratorT &last);

/// @brief Calculate mean and variance for given data.
template <typename ReturnT, typename ForwardIteratorT>
std::pair<ReturnT, ReturnT> meanVariance(const ForwardIteratorT &first, const ForwardIteratorT &last);

/// @brief Calculate sample mean for given data.
template <typename ReturnT, typename ForwardIteratorT>
ReturnT sampleMean(const ForwardIteratorT &first, const ForwardIteratorT &last);

/// @brief Calculate sample variance for given data.
template <typename ReturnT, typename ForwardIteratorT>
ReturnT sampleVariance(const ForwardIteratorT &first, const ForwardIteratorT &last);

/// @brief Calculate sample standard deviation for given data.
template <typename ReturnT, typename ForwardIteratorT>
ReturnT sampleStddev(const ForwardIteratorT &first, const ForwardIteratorT &last);

/// @brief Calculate sample mean and sample variance for given data.
template <typename ReturnT, typename ForwardIteratorT>
std::pair<ReturnT, ReturnT> sampleMeanVariance(const ForwardIteratorT &first, const ForwardIteratorT &last);

/// @brief Get min and max elements in given data.
template <typename ReturnT, typename ForwardIteratorT>
std::pair<ReturnT, ReturnT> minMax(const ForwardIteratorT &first, const ForwardIteratorT &last);

} // namespace utils

/// @brief Wrapper around a stochastic variable.
template <typename VT = float>
class StochasticVariable
{
public:
    /// @brief Real value type of the stochastic variable.
    using ValueT = typename VariableObserver<VT>::ValueT;
    /// @brief Base value type used for higher precision operations.
    using BasePreciseValueT = double;
    /// Is this a pair value variable?
    static constexpr auto PairVariable{ VariableObserver<VT>::ValueExtractorT::is_pair };
    /// @brief Value type used for higher precision operations.
    using PreciseValueT = typename std::conditional<
        PairVariable,
        std::pair<BasePreciseValueT, BasePreciseValueT>,
        BasePreciseValueT
    >::type;

    /// @brief Holder of statistics calculated for this stochastic variable.
    struct StatisticProperties
    {
        /// @brief Type used for storing statistics, which require floating point precision.
        using ST = PreciseValueT;

        /// @brief Get simplified version of properties, taking first element in case of pairs.
        typename StochasticVariable<ValueT>::StatisticProperties simpleProperties() const;

        /// Sample mean of the observations.
        ST mean{ };
        /// Sample variance of the observations.
        ST variance{ };

        /// Total observation count.
        std::size_t count{ 0u };
        /// Total minimum of all observed values.
        VT min{ std::numeric_limits<VT>::max() };
        /// Total maximum of all observed values.
        VT max{ std::numeric_limits<VT>::min() };
    }; // struct StatisticProperties

    // Constructors:
    /// @brief Initialize default stochastic variable.
    StochasticVariable();
    /// @brief Clean up and destroy.
    ~StochasticVariable();

    // Factory functions:

    // Observations:
    /**
     * @brief Add external sample to data accumulator of this variable.
     * @param value New observed sample.
     */
    void observeSample(const VT &value);

    /// @brief Add all external samples in the provided interval.
    template <typename ForwardIteratorT>
    void observeSamples(const ForwardIteratorT &first, const ForwardIteratorT &last);

    // Utilities:
    /// @brief Clear all data and reset to default state.
    void clear();
    /**
     * @brief Clear accumulated observations, keeping all pre-calculated statistics.
     *
     * @param calculateProperties Optionally re-calculate properties before clearing.
     */
    void clearObservations(bool calculateProperties = false);

    // Operations:
    /// @brief Calculate properties from the currently accumulated observations. Automatic caching.
    const StatisticProperties &properties();
    /// @brief Calculate properties from the currently accumulated observations. Throwns when dirty!
    const StatisticProperties &properties() const;

    /// @brief Get list of currently accumulated observations.
    const std::vector<VT> &observations() const;

    /// @brief Prepare empty histogram for observed values. Default values for auto-detection.
    template <typename CountT>
    TreeHistogram<CountT, ValueT> prepareHistogram(const ValueT &step = { },
        std::size_t minBuckets = 0u, std::size_t maxBuckets = 0u,
        const ValueT &automaticStepsPerVariance = 20);

    /// @brief Calculate histogram from currently accumulated observations. Default values for auto-detection.
    template <typename CountT>
    TreeHistogram<CountT, ValueT> calculateHistogram(const ValueT &step = { },
        std::size_t minBuckets = 0u, std::size_t maxBuckets = 0u,
        const ValueT &automaticStepsPerVariance = 20);

    /// @brief Print description of this variable.
    void describe(std::ostream &out, const std::string &indent = "") const;

    /// @brief Serialize content of this variable into given json.
    void saveTo(treeio::json &out) const;
private:
    template <typename T>
    friend class StochasticDistribution;

    /// @brief Calculate properties from current data.
    StatisticProperties calculateProperties() const;

    /// Observer for values of the stochastic variable.
    VariableObserver<VT> mObserver{ };
    /// Are the statistics dirty and need recalculation?
    bool mPropertiesDirty{ true };
    /// Statistics for the currently accumulated observations.
    StatisticProperties mProperties{ };
protected:
}; // class StochasticVariable

/// @brief Helper structure covering both variable and histogram fuctionality.
template <typename VT, typename CT>
struct StochasticVariableHistogram
{
    /// @brief Base type used for values.
    using ValueT = VT;
    /// @brief Simple type used for buckets.
    using BucketT = typename StochasticVariable<ValueT>::ValueT;
    /// @brief Type used for counting occurrences in the histogram.
    using CountT = CT;

    /// @brief Initialize the wrapper.
    StochasticVariableHistogram(const std::string &shortName,
        const std::string &longDescription);

    /// @brief Add sample to the stochastic variable.
    void observeSample(const ValueT &sample);

    /// @brief Calculate histogram from the stochastic variable.
    void calculateHistogram(std::size_t minBuckets);

    /// Short name of the variable.
    std::string name{ };
    /// Longer description.
    std::string description{ };

    /// Stochastic variable.
    StochasticVariable<ValueT> var{ };
    /// Corresponding histogram.
    TreeHistogram<CountT, BucketT> hist{ };
    /// Is the histogram prepared or calculated?
    bool histPrepared{ false };
}; // struct StochasticVariableHistogram

/// @brief Wrapper around image data.
struct ImageData
{
    /// Internal types for each channel value.
    enum class ValueType
    {
        UInt,
        Float
    }; // enum class ValueType

    /// @brief Convert ValueType to string
    static std::string valueTypeToStr(const ValueType &type);

    /// @brief Serialize content of this image into given json.
    void saveTo(treeio::json &out) const;

    /// Width of the image.
    std::size_t width{ 0u };
    /// Height of the image.
    std::size_t height{ 0u };
    /// Number of channels per pixel.
    std::size_t channels{ 0u };
    /// Type of each value.
    ValueType valueType{ ValueType::UInt };

    /// Image data.
    std::vector<uint8_t> data{ };
}; // struct ImageData

/// @brief Branching styles at tree branching points.
enum class Branching
{
    /// Single primary branch always continues.
    Monopodial,
    /// One primary, one secondary.
    SympodialMonochasial,
    /// Two secondary, optionally one stunt.
    SympodialDichasial
}; // enum class Branching

/// @brief Convert branching enumeration to ordinal.
std::size_t branchingToOrdinal(const Branching &branching);
/// @brief Convert ordinal to branching enumeration.
Branching ordinalToBranching(std::size_t ordinal);

/// @brief How are secondary branches generated.
enum class Ramification
{
    /// Every branching node starts a new axis.
    Continuous,
    /// Some nodes start a new axis.
    Rhythmic,
    /// Nodes starting a new axis are randomly distributed.
    Diffuse
}; // enum class Ramification

/// @brief Convert ramification enumeration to ordinal.
std::size_t ramificationToOrdinal(const Ramification &ramification);
/// @brief Convert ordinal to branching enumeration.
Ramification ordinalToRamification(std::size_t ordinal);

/// @brief Container for tree statistics.
struct TreeStatValues
{
    /// @brief Run given function on each stochastic variable.
    template <typename FuncT>
    void forEach(const FuncT &func) const;

    /// @brief Run given function on each stochastic variable.
    template <typename FuncT>
    void forEach(const FuncT &func);

    /// @brief Serialize content of this variable into given json.
    void saveTo(treeio::json &out) const;

    /// Segment thickness.
    StochasticVariableHistogram<float, float> segmentThickness{
        "segmentThickness", "Segment Thickness" };
    /// Segment volume per thickness.
    StochasticVariableHistogram<std::pair<float, float>, float> segmentVolume{
        "segmentVolume", "Segment Volume" };
    /// Segment counts per chain.
    StochasticVariableHistogram<uint32_t, uint32_t> segmentsPerChain{
        "segmentsPerChain", "Segments Per Chain" };
    /// Chain depths.
    StochasticVariableHistogram<uint32_t, uint32_t> chainsPerDepth{
        "chainsPerDepth", "Chains Per Depth" };
    /// Chain segment-wise lengths.
    StochasticVariableHistogram<float, uint32_t> chainLength{
        "chainLength", "Chain Lengths" };
    /// Chain lengths from start to end.
    StochasticVariableHistogram<float, uint32_t> chainTotalLength{
        "chainTotalLength", "Total Chain Lengths" };
    /// Sums of segment angles.
    StochasticVariableHistogram<float, uint32_t> chainDeformation{
        "chainDeformation", "Deformation" };
    /// Length ratios.
    StochasticVariableHistogram<float, uint32_t> chainLengthRatio{
        "chainLengthRatio", "Length Ratios" };
    /// Angle sum deltas.
    StochasticVariableHistogram<float, uint32_t> chainAngleSumDelta{
        "chainAngleSumDelta", "Angle Sum Deltas" };
    /// Parent-child angles.
    StochasticVariableHistogram<float, uint32_t> chainParentChildAngle{
        "chainParentChildAngle", "Parent Child Angle" };
    /// Chain straightness.
    StochasticVariableHistogram<float, uint32_t> chainStraightness{
        "chainStraightness", "Straightness" };
    /// Chain slope.
    StochasticVariableHistogram<float, uint32_t> chainSlope{
        "chainSlope", "Slope" };
    /// Minimal chain thickness.
    StochasticVariableHistogram<float, uint32_t> chainMinThickness{
        "chainMinThickness", "Minimal Thickness" };
    /// Maximal chain thickness.
    StochasticVariableHistogram<float, uint32_t> chainMaxThickness{
        "chainMaxThickness", "Maximal Thickness" };
    /// Ratio of minimal to maximal chain thickness.
    StochasticVariableHistogram<float, uint32_t> chainMinMaxThicknessRatio{
        "chainMinMaxThicknessRatio", "Minimal/Maximal Thickness Ratio" };
    /// Minimal child angle per parent.
    StochasticVariableHistogram<float, uint32_t> chainMinChildAngle{
        "chainMinChildAngle", "Minimal Child Angle Per Parent" };
    /// Maximal child angle per parent.
    StochasticVariableHistogram<float, uint32_t> chainMaxChildAngle{
        "chainMaxChildAngle", "Maximal Child Angle Per Parent" };
    /// Total minimal child angle per parent.
    StochasticVariableHistogram<float, uint32_t> chainMinChildAngleTotal{
        "chainMinChildAngleTotal", "Total Minimal Child Angle Per Parent" };
    /// Total maximal child angle per parent.
    StochasticVariableHistogram<float, uint32_t> chainMaxChildAngleTotal{
        "chainMaxChildAngleTotal", "Total Maximal Child Angle Per Parent" };
    /// Sibling angle.
    StochasticVariableHistogram<float, uint32_t> chainSiblingAngle{
        "chainSiblingAngle", "Sibling Angles" };
    /// Minimal sibling chain angle per parent.
    StochasticVariableHistogram<float, uint32_t> chainMinSiblingAngle{
        "chainMinSiblingAngle", "Minimal Sibling Angles Per Parent" };
    /// Maximal sibling chain angle per parent.
    StochasticVariableHistogram<float, uint32_t> chainMaxSiblingAngle{
        "chainMaxSiblingAngle", "Maximal Sibling Angles Per Parent" };
    /// Total sibling angle.
    StochasticVariableHistogram<float, uint32_t> chainSiblingAngleTotal{
        "chainSiblingAngleTotal", "Sibling Total Angles" };
    /// Minimal total sibling chain angle per parent.
    StochasticVariableHistogram<float, uint32_t> chainMinSiblingAngleTotal{
        "chainMinSiblingAngleTotal", "Minimal Total Sibling Angles Per Parent" };
    /// Maximal total sibling chain angle per parent.
    StochasticVariableHistogram<float, uint32_t> chainMaxSiblingAngleTotal{
        "chainMaxSiblingAngleTotal", "Maximal Total Sibling Angles Per Parent" };
    /// Parent to child angle.
    StochasticVariableHistogram<float, uint32_t> chainParentAngle{
        "chainParentAngle", "Parent To Child Angle" };
    /// Minimal parent to child angle per parent.
    StochasticVariableHistogram<float, uint32_t> chainMinParentAngle{
        "chainMinParentAngle", "Minimal Parent To Child Angle Per Parent" };
    /// Maximal parent to child angle per parent.
    StochasticVariableHistogram<float, uint32_t> chainMaxParentAngle{
        "chainMaxParentAngle", "Maximal Parent To Child Angle Per Parent" };
    /// Total parent to child angle.
    StochasticVariableHistogram<float, uint32_t> chainParentAngleTotal{
        "chainParentAngleTotal", "Parent To Child Total Angle" };
    /// Minimal total parent to child angle per parent.
    StochasticVariableHistogram<float, uint32_t> chainMinParentAngleTotal{
        "chainMinParentAngleTotal", "Minimal Parent To Child Total Angle Per Parent" };
    /// Maximal total parent to child angle per parent.
    StochasticVariableHistogram<float, uint32_t> chainMaxParentAngleTotal{
        "chainMaxParentAngleTotal", "Maximal Parent To Child Total Angle Per Parent" };

    /// Tropism estimate of a given chain with respect to the branch.
    StochasticVariableHistogram<float, uint32_t> chainTropismBranch{
        "chainTropismBranch", "Tropism With Respect To Branch Itself" };
    /// Tropism estimate of a given chain with respect to the horizon.
    StochasticVariableHistogram<float, uint32_t> chainTropismHorizon{
        "chainTropismHorizon", "Tropism With Respect To Horizon" };
    /// Tropism estimate of a given chain with respect to the parent.
    StochasticVariableHistogram<float, uint32_t> chainTropismParent{
        "chainTropismParent", "Tropism With Respect To Parent" };

    /// Asymmetry calculated using leaf nodes.
    StochasticVariableHistogram<float, uint32_t> chainAsymmetryLeaf{
        "chainAsymmetryLeaf", "Asymmetry Of Children Using Leaf" };
    /// Asymmetry calculated using sub-tree length.
    StochasticVariableHistogram<float, uint32_t> chainAsymmetryLength{
        "chainAsymmetryLength", "Asymmetry Of Children Using Sub-Tree Length" };
    /// Asymmetry calculated using sub-tree volume.
    StochasticVariableHistogram<float, uint32_t> chainAsymmetryVolume{
        "chainAsymmetryVolume", "Asymmetry Of Children Using Sub-Tree Volume" };

    /// Monopodial branching fitness per chain.
    StochasticVariableHistogram<float, float> chainMonopodialBranchingFitness{
        "chainMonopodialBranchingFitness", "Monopodial Branching Fitness Per Chain" };
    /// Sympodial Monochasial branching fitness per chain.
    StochasticVariableHistogram<float, float> chainSympodialMonochasialBranchingFitness{
        "chainSympodialMonochasialBranchingFitness", "Sympodial Monochasial Branching Fitness Per Chain" };
    /// Sympodial Dichasial branching fitness per chain.
    StochasticVariableHistogram<float, float> chainSympodialDichasialBranchingFitness{
        "chainSympodialDichasialBranchingFitness", "Sympodial Dichasial Branching Fitness Per Chain" };
    /// Aggregate branching fitnesses for the whole tree.
    StochasticVariableHistogram<std::pair<uint32_t, float>, float> aggregateBranchingFitness{
        "aggregateBranchingFitness", "Aggregate Branching Fitness For The Whole Tree" };

    /// Continuous ramification fitness per chain.
    StochasticVariableHistogram<float, float> chainContinuousRamificationFitness{
        "chainContinuousRamificationFitness", "Continuous Ramification Fitness Per Chain" };
    /// Rhythmic ramification growth fitness per chain.
    StochasticVariableHistogram<float, float> chainRhythmicRamificationFitness{
        "chainRhythmicRamificationFitness", "Rhythmic Ramification Fitness Per Chain" };
    /// Diffuse ramification growth fitness per chain.
    StochasticVariableHistogram<float, float> chainDiffuseRamificationFitness{
        "chainDiffuseRamificationFitness", "Diffuse Ramification Fitness Per Chain" };
    /// Aggregate ramification fitnesses for the whole tree.
    StochasticVariableHistogram<std::pair<uint32_t, float>, float> aggregateRamificationFitness{
        "aggregateRamificationFitness", "Aggregate Ramification Fitness For The Whole Tree" };

    /// Number of internodes per chain.
    StochasticVariableHistogram<float, uint32_t> chainInternodeCount{
        "chainInternodeCount", "Number Of Internodes Per Chain" };

    /// Estimated tree age.
    std::size_t treeAge{ 0u };
    /// Length of the trunk chain.
    float trunkLength{ 0.0f };
    /// Estimation of inter-node length.
    float interNodeLengthEstimate{ };
    /// Variance of the internode length estimation.
    float interNodeLengthVariance{ };

    /// Number of leaf nodes.
    std::size_t leafCount{ };

    /// Cardinal branching structure of the tree.
    Branching branching{ Branching::Monopodial };
    /// Cardinal ramification of the tree.
    Ramification ramification{ Ramification::Continuous };

    /// Shadow imprint of the tree.
    ImageData shadowImprint{ };
}; // struct TreeStatValues

/// @brief Tree statistics wrapper.
class TreeStats
{
public:
    /// @brief Container for settings used when generating tree statistics.
    struct StatsSettings
    {
        /// Default tree scale used for visual feature extraction.
        static constexpr auto DEFAULT_VISUAL_TREE_SCALE{ 5.0f };
        /// Default view count for visual feature extraction.
        static constexpr auto DEFAULT_VISUAL_VIEW_COUNT{ 90u }; //360u
        /// Default resolution of the views for visual feature extraction.
        static constexpr auto DEFAULT_VISUAL_VIEW_RESOLUTION{ 128u }; //256u
        /// Default number of samples for visual feature extraction.
        static constexpr auto DEFAULT_VISUAL_SAMPLE_COUNT{ 8u };
        /// Default enable state for the visual features.
        static constexpr auto DEFAULT_VISUAL_ENABLED{ false };
        /// Default verboseness of ray-tracer for visual feature extraction.
        static constexpr auto DEFAULT_VISUAL_VERBOSE{ false };
        /// Default display setting for visual feature extraction.
        static constexpr auto DEFAULT_VISUAL_DISPLAY_RESULTS{ false };
        /// Default setting for visual feature exporting.
        static constexpr auto DEFAULT_VISUAL_EXPORT_RESULTS{ false };
        /// Default multiplier used for branching point detection.
        static constexpr auto DEFAULT_GROWTH_BRANCHING_POINT_DELTA{ 0.001f };
        /// Up vector used for the horizon plane.
        static constexpr Vector3D DEFAULT_HORIZON_UP_DIRECTION{ 0.0f, 1.0f, 0.0f };

        /// Tree scale used for visual feature extraction.
        float visualTreeScale{ DEFAULT_VISUAL_TREE_SCALE };
        /// Number of views for visual feature extraction.
        std::size_t visualViewCount{ DEFAULT_VISUAL_VIEW_COUNT };
        /// Resolution of the views for visual feature extraction.
        std::size_t visualViewWidth{ DEFAULT_VISUAL_VIEW_RESOLUTION };
        /// Resolution of the views for visual feature extraction.
        std::size_t visualViewHeight{ DEFAULT_VISUAL_VIEW_RESOLUTION };
        /// Sampling of the ray-tracer for visual feature extraction.
        std::size_t visualSampleCount{ DEFAULT_VISUAL_SAMPLE_COUNT };
        /// Enable for the visual features..
        bool visualEnabled{ DEFAULT_VISUAL_ENABLED };
        /// Verboseness of the ray-tracer for visual feature extraction.
        bool visualVerbose{ DEFAULT_VISUAL_VERBOSE };
        /// Set export path in order to export visual statistics into files.
        bool visualExportResults{ DEFAULT_VISUAL_EXPORT_RESULTS };
        /// Export the visual statistics to this path.
        std::string visualExportPath{ "data/output/" };

        /// Delta multiplier used for compact branching point detection.
        float growthBranchingPointDelta{ DEFAULT_GROWTH_BRANCHING_POINT_DELTA };
    }; // struct StatsSettings!

    /// @brief Initialize empty wrapper with no statistics calculated.
    TreeStats();
    /// @brief Initialize wrapper and calculate statistics for given tree.
    TreeStats(const treeio::ArrayTree &tree);

    /// @brief Clean up and destroy.
    ~TreeStats();

    /// @brief Calculate statistics for given tree.
    void calculateStatistics(const treeio::ArrayTree &tree);

    /// @brief Save currently calculated statistics to the dynamic meta-data of given tree.
    void saveStatisticsToDynamic(treeio::ArrayTree &tree);

    /// @brief Access the settings for statistics generation.
    StatsSettings &settings();
    /// @brief Access the settings for statistics generation.
    const StatsSettings &settings() const;
private:
    /// Information about a single segment.
    struct SegmentData
    {
        /// Length of the segment.
        float length{ 0.0f };
        /// Starting thickness of this segment.
        float startThickness{ 0.0f };
        /// Ending thickness of this segment.
        float endThickness{ 0.0f };
    }; // struct SegmentData

    /// @brief Additional pre-computed chain data.
    struct ChainData
    {
        /// Total length of the chain.
        float length{ 0.0f };
        /// Sum of radian angle differences between segments.
        float angleSum{ 0.0f };
        /// Sum of absolute radian angle differences between segments.
        float angleAbsSum{ 0.0f };
        /// Thickness averaged by segment lengths.
        float lengthAverageThickness{ 0.0f };
        /// Volume of the branches estimated using circular frustum cones.
        float volume{ 0.0f };

        /// Sum of chain lengths from this chain toward the leaves.
        float subtreeLength{ 0.0f };
        /// Sum of chain volumes from this chain toward the leaves.
        float subtreeVolume{ 0.0f };
        /// Number of leaves from this chain onward.
        float subtreeLeafCount{ 0.0f };

        /// Order of the current axis.
        std::size_t axisOrder{ 0u };

        /// Minimum thickness of this chain.
        float minThickness{ std::numeric_limits<float>::max() };
        /// Maximum thickness of this chain.
        float maxThickness{ std::numeric_limits<float>::min() };

        /// Start to end vector for this chain.
        Vector3D startToEnd{ };
        /// First segment direction in this chain.
        Vector3D firstDirection{ };
        /// Last segment direction in this chain.
        Vector3D lastDirection{ };

        /// Storage for per-segment data.
        std::vector<SegmentData> segmentData{ };
    }; // struct ChainData

    /// @brief List of branch indices.
    using BranchIdxList = std::vector<std::size_t>;

    /// Minimal bucket count of histograms.
    static constexpr auto MIN_BUCKETS{ 10u };

    /// @brief Phase 1 Prepare helper tree data.
    static std::pair<treeutil::TreeChains, std::vector<ChainData>> prepareTree(
        TreeStatValues &stats, const StatsSettings &settings, const treeio::ArrayTree &tree);

    /// @brief Calculate forward pass information.
    static void prepareTreeForwardPass(treeutil::TreeChains &chains, std::vector<ChainData> &chainDataStorage,
        TreeStatValues &stats, const StatsSettings &settings);

    /// @brief Calculate backward pass information.
    static void prepareTreeBackwardPass(treeutil::TreeChains &chains, std::vector<ChainData> &chainDataStorage,
        TreeStatValues &stats, const StatsSettings &settings);

    /// @brief Phase 2 Calculate basic statistics.
    static void calculateBasicStats(TreeStatValues &stats, const StatsSettings &settings,
        const treeutil::TreeChains &chains, const std::vector<ChainData> &chainDataStorage);

    /// @brief Phase 3 Calculate visual statistics.
    static void calculateVisualStats(TreeStatValues &stats, const StatsSettings &settings,
        const treeio::ArrayTree &inputTree);

    /// @brief Phase 4 Calculate derived statistics.
    static void calculateDerivedStats(TreeStatValues &stats, const StatsSettings &settings,
        const treeutil::TreeChains &chains, const std::vector<ChainData> &chainDataStorage);

    /// @brief Phase 5 Calculate growth statistics.
    static void calculateGrowthStats(TreeStatValues &stats, const StatsSettings &settings,
        const treeutil::TreeChains &chains, const std::vector<ChainData> &chainDataStorage);

    /// @brief Calculate maximum length from all of the sub-chains in given compacted chains.
    static float calculateMaxCompactChainLength(
        const treeutil::TreeChains::ChainStorage &fullChains,
        const treeutil::TreeChains::InternalArrayTree &chainsTree,
        const treeutil::TreeChains::CompactNodeChain &compactChain);

    /// @brief Calculate monopodial fitness of given branches and return same axis and higher axis indices.
    static std::tuple<float, BranchIdxList, BranchIdxList>
        calculateMonopodialFitness(
            const std::vector<float> &branchContinuationWeights,
            const std::vector<float> &branchSizeWeights);

    /// @brief Calculate sympodial monochasial fitness of given branches and return same axis and higher axis indices.
    static std::tuple<float, BranchIdxList, BranchIdxList>
        calculateSympodialMonochasialFitness(
            const std::vector<float> &branchContinuationWeights,
            const std::vector<float> &branchSizeWeights);

    /// @brief Calculate sympodial dichasial fitness of given branches and return same axis and higher axis indices.
    static std::tuple<float, BranchIdxList, BranchIdxList>
        calculateSympodialDichasialFitness(
            const std::vector<float> &branchContinuationWeights,
            const std::vector<float> &branchSizeWeights);

    /// @brief Calculate continuous ramification fitness using given internode information.
    static float calculateContinuousFitness(float internodeCount, float mean, float var, float min, float max);
    /// @brief Calculate rhythmic ramification fitness using given internode information.
    static float calculateRhythmicFitness(float internodeCount, float mean, float var, float min, float max);
    /// @brief Calculate diffuse ramification fitness using given internode information.
    static float calculateDiffuseFitness(float internodeCount, float mean, float var, float min, float max);

    /// @brief Phase 6 Finalize and clean up.
    static void finalizeStats(TreeStatValues &stats, const StatsSettings &settings);

    /// Calculated statistic values for the last loaded tree.
    TreeStatValues mStats{ };
    /// Settings for statistics calculation.
    StatsSettings mSettings{ };
protected:
}; // class TreeStats



#if 0

/// Combination of final branch node and its depth within the tree.
struct BranchDepth
{
    /// Last node of the branch.
    treeio::TreeNode *node{ nullptr };
    /// Number of segments leading to this node.
    uint32_t depth{ 0u };
}; // struct BranchDepth

/// Combination of two segment nodes and its depth within the tree.
struct SegmentDepth
{
    /// Segments first node - lower in the hierarchy.
    treeio::TreeNode *node1{ nullptr };
    /// Segments second node - higher in the hierarchy.
    treeio::TreeNode *node2{ nullptr };
    /// Number of segments leading to the first node.
    uint32_t depth{ 0u };
    /// Length of the segment
    float length{ 0.0f };
}; // struct SegmentDepth

/// Histogram containing information about branching.
struct BranchHistogram
{
    BranchHistogram() = default;
    ~BranchHistogram() = default;

    /**
     * @brief Calculate the segment counts histogram for
     * given vector of branches.
     *
     * Resulting histogram will contain segment counts.
     * Each bucket on the x-axis represents all
     * branches with given amount of segments - e.g.
     * index 0 contains all branches of length 0,
     * index x contains all branches of length x.
     *
     * @param branches List of branches to take into
     * account.
     */
    void calculateSegmentCounts(const std::vector<BranchDepth> &branches);

    /**
     * @brief Calculate the segmen lengths histogram for
     * given tree.
     *
     * Resulting histogram will contain segment lengths.
     * Each bucket on the x-axis represents all
     * segments with given length.
     *
     * @param segments List of branches to take into
     * account.
     * @param minBuckets Minimum number of buckets in the
     * histogram.
     * @param maxBuckets Maximum number of buckets in the
     * histogram.
     */
    void calculateSegmentLengths(const std::vector<SegmentDepth> &segments,
        uint32_t minBuckets = 8u, uint32_t maxBuckets = 32u);

    /// Simple getter, which should be passed pointer to uint32_t array.
    static float floatGetter(void *data, int index)
    { return static_cast<float>(reinterpret_cast<uint32_t*>(data)[index]); }

    /// Histogram holder.
    std::vector<uint32_t> histogram;
}; // struct BranchHistogram

/// Container for statistics about a tree.
class TreeStatistics
{
public:
    TreeStatistics() = default;
    ~TreeStatistics() = default;

    TreeStatistics(const TreeStatistics &other) = default;
    TreeStatistics &operator=(const TreeStatistics &other) = default;
    TreeStatistics(TreeStatistics &&other) = default;
    TreeStatistics &operator=(TreeStatistics &&other) = default;

    /**
     * @brief Calculate statistics for given tree.
     *
     * @param targetTree Tree to calculate statistics for.
     * @param colorize Colorize the longest branch and segment?
     */
    TreeStatistics(const treeio::Tree &targetTree, bool colorize = true);

    /**
     * @brief Calculate statistics for given tree.
     *
     * @param targetTree Tree to calculate statistics for.
     * @param colorize Colorize the longest branch and segment?
     */
    void calculate(const treeio::Tree &targetTree, bool colorize = true);

    /// Print the statistics to stdout.
    void printInfo();

    /// Histogram of segment counts per each branch.
    treeutil::BranchHistogram segmentCounts{ };
    /// Histogram of segment lengths per each branch.
    treeutil::BranchHistogram segmentLengths{ };

    /// List of branches with depth given by number of segments.
    std::vector<BranchDepth> branches{ };
    /// List of segments with depth given by number of segments.
    std::vector<SegmentDepth> segments{ };

    /// List of branches with depth given by number of branching points.
    std::vector<BranchDepth> realDepthBranches{ };
    /// List of segments with depth given by number of branching points.
    std::vector<SegmentDepth> realDepthSegments{ };

    /// Branch with most segments.
    BranchDepth mostSegmentedBranch{ };
    /// Branch with most branching points.
    BranchDepth mostBranchingBranch{ };
    /// Longest segment.
    SegmentDepth longestSegment{ };

    /// Estimated tree age.
    uint32_t treeAge{ 0u };
    /// Mean of the internode length.
    float internodeLengthMean{ 0.0f };
    /// Variance of the internode length.
    float internodeLengthVar{ 0.0f };
    /// Length of the trunk.
    float trunkLength{ 0.0f };
private:
    /// Target tree.
    treeio::Tree mTree{ };
}; // struct TreeStatistics

/**
 * @brief Fix parent attributes so they point to the
 * correct memory.
 *
 * @param tree Tree to fix up.
 */
void fixTreeParentNodes(treeio::Tree &tree);

/**
 * @brief Fix backwards branches on given tree skeleton.
 *
 * @param tree Tree to fix up.
 */
void fixTreeBackwardBranches(treeio::Tree &tree);

/**
 * @brief Find all branches in the given tree. The length
 * is taken in segments or number of branching points.
 *
 * @param tree Input tree object.
 * @param countBranches When true the distance is a number
 * of branching points all the way to the root. When false
 * distance is number of segments instead.
 *
 * @return Returns array of branch points with
 * their associated depthts.
 */
std::vector<BranchDepth> findBranches(treeio::Tree &tree,
    bool countBranches = false);

/**
 * @brief Find all segments in the given tree. The length
 * is taken in segments or number of branching points.
 *
 * @param tree Input tree object.
 * @param countBranches When true the distance is a number
 * of branching points all the way to the root. When false
 * distance is number of segments instead.
 *
 * @return Returns array of segments with their
 * associated depthts.
 */
std::vector<SegmentDepth> findSegments(treeio::Tree &tree,
    bool countBranches = false);

/**
 * @brief Find all segments and branches in the given tree.
 * The length is taken in segments or number of branching
 * points.
 *
 * @param tree Input tree object.
 * @param countBranches When true the distance is a number
 * of branching points all the way to the root. When false
 * distance is number of segments instead.
 *
 * @return Returns ararys of branches and segments, with
 * their associated depths.
 */
std::tuple<std::vector<BranchDepth>, std::vector<SegmentDepth>>
findBranchesSegments(treeio::Tree &tree, bool countBranches = false);

/**
 * @brief Find the longest branch in given tree.
 *
 * @param branches List of branches to search through.
 */
BranchDepth findLongestBranch(const std::vector<BranchDepth> &branches);

/**
 * @brief Find the longest segment in given tree.
 *
 * @param segments List of segments to search through.
 */
SegmentDepth findLongestSegment(const std::vector<SegmentDepth> &segments);

/**
 * @brief Estimate age of a trees crown from its longest branch.
 *
 * @param longestBranch Longest branch from the target tree.
 * @return Returns estimated age of given tree.
 */
uint32_t estimateCrownAge(const BranchDepth &longestBranch);

/**
 * @brief Estimate age of a trees trunk.
 *
 * @param trunkLength Length of the target trunk.
 * @param internodelength Length of a single internode.
 * @return Returns estimated age of the trunk.
 */
uint32_t estimateTrunkAge(float trunkLength, float internodeLength);

/// Is given node a branching node?
bool isBranchingNode(const treeio::TreeNode *node);

/**
 * @brief Calculate internode lengths for a given branch.
 * Internode is taken as all segments between 2 branching nodes.
 *
 * @param branch Input branch.
 * @return Returns list of internode lengths, where the last value
 * is the lenght of the "trunk".
 */
std::vector<float> calculateInternodeLengths(const BranchDepth &branch);

/**
 * @brief Estimate mean and variance of internode length for
 * a tree using its longest branch.
 *
 * @param longestBranch Longest branch from the target tree.
 * @return Returns mean and variance of the internode length.
 * The third returned value contained estimate of trunk length.
 */
std::tuple<float, float, float> estimateInternodeLength(const BranchDepth &longestBranch);

/**
 * @brief Estimate mean and variance of nternode length for
 * a tree using all of its branches.
 *
 * @param branches List of branches in the target tree.
 * @return Returns mean and variance of the internode length.
 */
std::tuple<float, float> estimateInternodeLengthFromAll(const std::vector<BranchDepth> &branches);

/**
 * @brief Estimate internode length for the last tier of branching
 * on given branch path.
 *
 * @param branch Branch to estimate the internode length from.
 * @return Returns estimated internode length.
 */
float estimateLastInternodeLength(const BranchDepth &branch);

/**
 * @brief Estimate mean and variance of nternode length for
 * a tree using all of its youngest branches.
 *
 * @param branches List of branches in the target tree.
 * @return Returns mean and variance of the internode length.
 */
std::tuple<float, float> estimateInternodeLengthFromYoung(const std::vector<BranchDepth> &branches);

/**
 * @brief Colorize the branch and its points all the
 * way to the trunk.
 *
 * @param branch Branch to colorize.
 * @param pointColor New color of the nodes.
 * @param lineColor New color of the segments.
 */
void colorizeBranch(const BranchDepth &branch,
    treeio::Color pointColor = { 0.3f, 0.9f, 0.3f },
    treeio::Color lineColor = { 0.1f, 0.7f, 0.1f });

/**
 * @brief Colorize the segment.
 *
 * @param segment Segment to colorize.
 * @param lineColor New color of the segments.
 */
void colorizeSegment(const SegmentDepth &segment,
    treeio::Color lineColor = { 0.1f, 0.1f, 0.7f });

/// @brief Get distance between 2 tree nodes.
float nodeDistance(treeio::TreeNode *n1, treeio::TreeNode *n2);

/// @brief Swaps the values between dira and dirb (x - 1, y - 2, z - 3) Negative number means opposite direction
void swapNodeCoords(treeio::TreeNode* node, int dira, int dirb);

/// @brief Swaps the values between dira and dirb (x - 1, y - 2, z - 3) Negative number means opposite direction
void swapTreeCoords(treeio::TreeNode* tree, int dira, int dirb);

/// @brief Swaps the values between dira and dirb (x - 1, y - 2, z - 3) Negative number means opposite direction
void swapArrayTreeCoords(treeio::ArrayTree* tree, int current, int dira, int dirb);

#endif



} // namespace treestat

/// @brief Print histogram description.
template <typename CT, typename BT>
inline std::ostream &operator<<(std::ostream &out, const treestat::TreeHistogram<CT, BT> &histogram);

// Template implementation begin.

namespace treestat
{

template <typename CT, typename BT>
TreeHistogram<CT, BT>::TreeHistogram(const BT &step)
{ clear(step); }

template <typename CT, typename BT>
TreeHistogram<CT, BT>::TreeHistogram(const BT &min, const BT &max,
    const BT &step, std::size_t minBuckets, std::size_t maxBuckets,
    bool moveMin, bool moveMax)
{ initializeBuckets(min, max, step, minBuckets, maxBuckets, moveMin, moveMax); }

template <typename CT, typename BT>
template <typename ForwardItT>
TreeHistogram<CT, BT>::TreeHistogram(const ForwardItT &first, const ForwardItT &last,
    const BT &min, const BT &max, const BT &step,
    std::size_t minBuckets, std::size_t maxBuckets,
    bool moveMin, bool moveMax)
{
    if (min != BT{ } || max != BT{ })
    { initializeBuckets(min, max, step, minBuckets, maxBuckets, moveMin, moveMax); }
    countData(first, last);
}

template <typename CT, typename BT>
TreeHistogram<CT, BT>::~TreeHistogram()
{ /* Automatic */ }

template <typename CT, typename BT>
void TreeHistogram<CT, BT>::clear(const BT &step)
{ mStep = step; mBuckets.clear(); mHistogram.clear(); }

template <typename CT, typename BT>
void TreeHistogram<CT, BT>::clearValues()
{ std::fill(mHistogram.begin(), mHistogram.end(), 0); }

template <typename CT, typename BT>
void TreeHistogram<CT, BT>::initializeBuckets(const BT &min, const BT &max,
    const BT &step, std::size_t minBuckets, std::size_t maxBuckets,
    bool moveMin, bool moveMax)
{
    clear(step);

    // Calculate number of buckets to cover the whole interval.
    auto bucketCount{ static_cast<std::size_t>(std::ceil((max - min) / static_cast<double>(mStep))) };

    if (bucketCount < minBuckets && minBuckets)
    {
        bucketCount = minBuckets;
        const auto optimalStep{ (max - min) / static_cast<double>(bucketCount) };
        mStep = static_cast<BT>(std::numeric_limits<BT>::is_exact ? std::ceil(optimalStep) : optimalStep);
    }
    else if (bucketCount > maxBuckets && maxBuckets)
    {
        bucketCount = maxBuckets;
        const auto optimalStep{ (max - min) / static_cast<double>(bucketCount) };
        mStep = static_cast<BT>(std::numeric_limits<BT>::is_exact ? std::ceil(optimalStep) : optimalStep);
    }

    // Calculate how much we need to change the interval to satisfy bucket sizing.
    auto intervalDelta{ (bucketCount * mStep) - (max - min) };
    // Move interval end-points to create a integral number of buckets.
    const auto [ cMin, cMax ]{ calculateMovedInterval(min, max, moveMin, moveMax, intervalDelta) };

    // Create the buckets.
    mBuckets.resize(bucketCount + 1u);
    mHistogram.resize(bucketCount, 0);

    // Initialize buckets.
    for (std::size_t bucketIdx = 0u; bucketIdx < bucketCount + 1u; ++bucketIdx)
    { mBuckets[bucketIdx] = cMin + bucketIdx * mStep; }
}

template <typename CT, typename BT>
CT &TreeHistogram<CT, BT>::getBucket(const BT &value, bool create)
{
    const auto bucketIdx{
        create ?
            getCreateBucketIdx(value) :
            getBucketIdx(value)
    };

    if (bucketIdx >= mHistogram.size())
    { throw std::runtime_error("Failed to find corresponding bucket and create == false!"); }

    return mHistogram[bucketIdx];
}

template <typename CT, typename BT>
CT &TreeHistogram<CT, BT>::getBucket(const BT &value) const
{
    const auto bucketIdx{ getBucketIdx(value) };

    if (bucketIdx >= mHistogram.size())
    { throw std::runtime_error("Failed to find corresponding bucket!"); }

    return mHistogram[bucketIdx];
}

template <typename CT, typename BT>
template <typename ForwardItT>
void TreeHistogram<CT, BT>::countData(const ForwardItT &first, const ForwardItT &last)
{
    using ValueT = treeutil::remove_const_reference_t<decltype(*first)>;
    if constexpr (treeutil::is_specialization_v<ValueT, std::pair>)
    { // Pair representing <value, increment>.
        for (auto it = first; it != last; ++it) { getBucket(it->first) += it->second; }
    }
    else
    { // Single value with implicit increment of 1.
        for (auto it = first; it != last; ++it) { ++getBucket(*it); }
    }
}

template <typename CT, typename BT>
const std::vector<BT> TreeHistogram<CT, BT>::buckets() const
{ return mBuckets; }

template <typename CT, typename BT>
const std::vector<CT> &TreeHistogram<CT, BT>::histogram() const
{ return mHistogram; }

template <typename CT, typename BT>
std::pair<BT, BT> TreeHistogram<CT, BT>::minBucket() const
{
    const auto [minValue, minBucketIdx]{ treeutil::argMin(mHistogram) };

    if (mBuckets.empty() || minBucketIdx >= mBuckets.size() - 1u)
    { return { std::numeric_limits<BT>::max(), std::numeric_limits<BT>::max() }; }

    return { mBuckets[minBucketIdx], mBuckets[minBucketIdx + 1u] };
}

template <typename CT, typename BT>
std::pair<BT, BT> TreeHistogram<CT, BT>::maxBucket() const
{
    const auto [maxValue, maxBucketIdx]{ treeutil::argMax(mHistogram) };

    if (mBuckets.empty() || maxBucketIdx >= mBuckets.size() - 1u)
    { return { std::numeric_limits<BT>::min(), std::numeric_limits<BT>::min() }; }

    return { mBuckets[maxBucketIdx], mBuckets[maxBucketIdx + 1u] };
}

template <typename CT, typename BT>
void TreeHistogram<CT, BT>::describe(std::ostream &out, const std::string &indent) const
{
    out << "[ Histogram: \n";
    for (std::size_t iii = 0u; iii < mHistogram.size(); ++iii)
    { out << indent << "<" << mBuckets[iii] << " ; " << mBuckets[iii + 1u] << ") => " << mHistogram[iii] << ", \n"; }
    out << indent << " ]";
}

template <typename CT, typename BT>
void TreeHistogram<CT, BT>::saveTo(treeio::json &out) const
{
    auto &outHistogram{ out["histogram"] };

    outHistogram = {
        { "min", mBuckets.empty() ? BT{ } : mBuckets.front() },
        { "max", mBuckets.empty() ? BT{ } : mBuckets.back() },
        { "buckets", mBuckets.size() },
    };

    auto &outHistogramData{ outHistogram["data"] };

    for (std::size_t iii = 0u; iii < mHistogram.size(); ++iii)
    {
        // Optimize out empty buckets.
        if (!treeutil::aboveEpsilon(mHistogram[iii]))
        { continue; }

        outHistogramData.push_back({
            { "start", mBuckets[iii] },
            { "end", mBuckets[iii + 1u] },
            { "count", mHistogram[iii] }
        });
    }
}

template <typename CT, typename BT>
std::size_t TreeHistogram<CT, BT>::getCreateBucketIdx(const BT &value)
{
    if (mBuckets.empty())
    { // No buckets -> Create a single initial bucket.
        mBuckets.resize(2u);
        mHistogram.resize(1u, 0);

        mBuckets[0] = static_cast<std::size_t>(value / mStep) * mStep;
        mBuckets[1] = mBuckets[0] + mStep;

        return 0u;
    }
    else if (value < mBuckets.front())
    { // Value is before the first bucket -> Create new buckets.
        const auto currentMin{ mBuckets.front() };
        const auto newBucketsRequired{
            static_cast<std::size_t>(std::ceil((currentMin - value) / mStep))
        };

        // Make space for the new buckets.
        mBuckets.resize(mBuckets.size() + newBucketsRequired);
        mHistogram.resize(mHistogram.size() + newBucketsRequired, 0);

        // Initialize bucket end-points and the histogram.
        std::move_backward(mBuckets.begin(),
            mBuckets.begin() + mBuckets.size() - newBucketsRequired,
            mBuckets.begin() + mBuckets.size());
        for (std::size_t bucketIdx = 0u; bucketIdx < newBucketsRequired; ++bucketIdx)
        { mBuckets[bucketIdx] = currentMin - (newBucketsRequired - bucketIdx) * mStep; }
        std::move_backward(mHistogram.begin(),
            mHistogram.begin() + mHistogram.size() - newBucketsRequired,
            mHistogram.begin() + mHistogram.size());
        std::fill(mHistogram.begin(), mHistogram.begin() + newBucketsRequired, 0);

        // The first bucket should now contain the new value.
        assert(mBuckets[0] <= value && mBuckets[1] >= value);
        return 0u;
    }
    else if (value >= mBuckets.back())
    { // Value is after the last bucket -> Create new buckets.
        const auto currentMax{ mBuckets.back() };
        // Add one more bucket, because intervals are <a, b).
        const auto newBucketsRequired{
            static_cast<std::size_t>(std::ceil((value - currentMax) / mStep) + 1)
        };
        const auto currentBucketCount{ mBuckets.size() };

        // Make space for the new buckets.
        mBuckets.resize(mBuckets.size() + newBucketsRequired);
        mHistogram.resize(mHistogram.size() + newBucketsRequired, 0);

        // Initialize bucket end-points and the histogram.
        for (std::size_t bucketIdx = 0u; bucketIdx < newBucketsRequired; ++bucketIdx)
        { mBuckets[currentBucketCount + bucketIdx] = currentMax + (bucketIdx + 1u) * mStep; }

        // The last bucket should now contain the new value.
        assert(mBuckets[mBuckets.size() - 2u] <= value && mBuckets[mBuckets.size() - 1u] >= value);
        return mHistogram.size() - 1u;
    }
    else
    { // Value already has a bucket -> Get its index.
        const auto lb{ std::lower_bound(mBuckets.begin(), mBuckets.end(), value) };
        assert(lb != mBuckets.end());
        const auto bucketIdx{ std::distance(mBuckets.begin(), lb) };
        assert(bucketIdx <= mHistogram.size());
        return static_cast<std::size_t>(std::max<decltype(bucketIdx)>(0, bucketIdx - (mBuckets[bucketIdx] != value)));
    }
}

template <typename CT, typename BT>
std::size_t TreeHistogram<CT, BT>::getBucketIdx(const BT &value) const
{
    if (mBuckets.empty() || value < mBuckets.front() || value >= mBuckets.back())
    { // Bucket is not created -> Return invalid index.
        return mHistogram.size();
    }
    else
    { // Value already has a bucket -> Get its index.
        const auto lb{ std::lower_bound(mBuckets.begin(), mBuckets.end(), value) };
        assert(lb != mBuckets.end());
        const auto bucketIdx{ std::distance(mBuckets.begin(), lb) };
        assert(bucketIdx <= mHistogram.size());
        return static_cast<std::size_t>(std::max<decltype(bucketIdx)>(0, bucketIdx - (mBuckets[bucketIdx] != value)));
    }
}

template <typename CT, typename BT>
std::pair<BT, BT> TreeHistogram<CT, BT>::calculateMovedInterval(
    const BT &min, const BT &max, bool moveMin, bool moveMax,
    const BT &intervalDelta)
{
    if (intervalDelta > std::numeric_limits<BT>::epsilon() && !moveMin && !moveMax)
    {
        throw std::runtime_error("Initializing buckets with disabled interval moving, "
                                 "but interval does not satisfy the requirements!");
    }

    // Start with original interval.
    BT cMin{ min };
    BT cMax{ max };
    BT delta{ intervalDelta };

    // Move interval end-points to create a integral number of buckets.
    if (moveMin && moveMax)
    { // Moving both interval end-points, making sure we don't overflow / underflow.
        const auto minDelta{
            std::min<BT>(
                treeutil::maximumNegativeDelta(cMin),
                delta / BT(2)
            )
        };
        cMin -= minDelta; delta -= minDelta;
        const auto maxDelta{
            std::min<BT>(
                treeutil::maximumPositiveDelta(cMin),
                delta
            )
        };
        cMax += maxDelta; delta -= maxDelta;
    }
    else
    { // Moving one or none of the interval end-points, making sure we don't overflow / underlow.
        cMin -= std::min<BT>(
            treeutil::maximumNegativeDelta(cMin),
            moveMin * delta
        );
        cMax += std::min<BT>(
            treeutil::maximumPositiveDelta(cMax),
            moveMax * delta
        );
    }

    return { cMin, cMax };
}

template <typename VT>
VariableObserver<VT>::VariableObserver()
{ clear(); }
template <typename VT>
VariableObserver<VT>::VariableObserver(bool keepObservations)
{ clear(); setKeepObservations(keepObservations); }
template <typename VT>
VariableObserver<VT>::~VariableObserver()
{ /* Automatic */ }

template <typename VT>
void VariableObserver<VT>::observeSample(const VT &value)
{ mStats.newValue(value); if (mKeepObservations) { mValues.emplace_back(value); } }

template <typename VT>
template <typename ForwardIteratorT>
void VariableObserver<VT>::observeSamples(const ForwardIteratorT &first, const ForwardIteratorT &last)
{ for (auto it = first; it != last; ++it) { observeSample(*it); } }

template <typename VT>
void VariableObserver<VT>::clear()
{ mStats = { }; mValues.clear(); mKeepObservations = false; }

template <typename VT>
void VariableObserver<VT>::clearObservations()
{ mValues.clear(); }

template <typename VT>
void VariableObserver<VT>::setKeepObservations(bool keepObservations)
{ mKeepObservations = keepObservations; }

template <typename VT>
std::size_t VariableObserver<VT>::count() const
{ return mStats.count; }
template <typename VT>
const VT &VariableObserver<VT>::min() const
{ return mStats.min; }
template <typename VT>
const VT &VariableObserver<VT>::max() const
{ return mStats.max; }
template <typename VT>
const std::vector<VT> &VariableObserver<VT>::values() const
{ return mValues; }

template <typename VT>
void VariableObserver<VT>::describe(std::ostream &out, const std::string &indent) const
{
    out << "[ VariableObserver: \n"
        << indent << "\tCount: " << mStats.count << "\n"
        << indent << "\tMin: " << mStats.min << "\n"
        << indent << "\tMax: " << mStats.max << "\n";

    out << indent << "\tValues: [ ";
    for (const auto &val : mValues)
    { out << val << " "; }
    out << " ]\n";

    out << indent << " ]";
}

template <typename VT>
void VariableObserver<VT>::saveTo(treeio::json &out) const
{
    out["variable"] = {
        { "count", mStats.count },
        { "min", mStats.min },
        { "max", mStats.max },
        { "values", treeutil::encodeBinaryJSON(treeutil::hdf5Compress(mValues)) }
    };
}

template <typename VT>
void VariableObserver<VT>::Statistics::newValue(const VT &value)
{
    count++;
    min = std::min<VT>(min, value);
    max = std::max<VT>(max, value);
}

namespace impl
{

template <typename... CArgTs>
std::shared_ptr<DistributionProperties>
    DistributionProperties::constructDistribution
        (DistributionEngineImpl &engine, DistributionType type, CArgTs... cArgs)
{
    switch (type)
    {
        case DistributionType::Normal:
        { return constructDistribution<DistributionType::Normal>(engine, std::forward<CArgTs>(cArgs)...); }
        default:
        { throw std::runtime_error("Constructing unknown type of distribution!"); }
    }
}

template <DistributionType Type, typename... CArgTs>
std::shared_ptr<DistributionProperties>
DistributionProperties::constructDistribution(DistributionEngineImpl &engine, CArgTs... cArgs)
{ return constructDistribution(engine, Type, std::forward<CArgTs>(cArgs)...); }

} // namespace impl

template <typename VT>
StochasticDistribution<VT>::StochasticDistribution()
{ /* Automatic */ }

template <typename VT>
template <typename... CArgTs>
StochasticDistribution<VT>::StochasticDistribution(DistributionType type, CArgTs... cArgs)
{ setDistribution(type, std::forward<CArgTs>(cArgs)...); }

template <typename VT>
StochasticDistribution<VT>::~StochasticDistribution()
{ /* Automatic */ }

template <typename VT>
void StochasticDistribution<VT>::seed(uint32_t value)
{ mEngine.reset(value); if (mDistribution) { mDistribution->setEngine(*mEngine.mImpl); } }

template <typename VT>
template <typename... CArgTs>
void StochasticDistribution<VT>::setDistribution(DistributionType type, CArgTs... cArgs)
{
    mDistribution = impl::DistributionProperties::constructDistribution(
        *mEngine.mImpl, type, std::forward<CArgTs>(cArgs)...);
}

template <typename VT>
std::string StochasticDistribution<VT>::saveDistribution(const std::string &distribution)
{ return mDistribution ? mDistribution->serialize() : std::string{ }; }

template <typename VT>
void StochasticDistribution<VT>::loadDistribution(const std::string &distribution)
{ mDistribution = impl::DistributionProperties::deserializeDistribution(*mEngine.mImpl, distribution); }

template <typename VT>
VT StochasticDistribution<VT>::cdf(const VT &x) const
{ checkDistributionValidThrow(); return mDistribution->cdf(x); }

template <typename VT>
VT StochasticDistribution<VT>::sample()
{ checkDistributionValidThrow(); return mDistribution->sample(); }

template <typename VT>
void StochasticDistribution<VT>::checkDistributionValidThrow() const
{ if (!mDistribution) { throw StochasticException("Stochastic distribution has no distribution selected!"); } }

namespace utils
{

template <typename ReturnT, typename ForwardIteratorT>
ReturnT mean(const ForwardIteratorT &first, const ForwardIteratorT &last)
{
    const auto sum{ std::reduce(first, last, ReturnT{ }, std::plus<>()) };

    return sum / static_cast<ReturnT>(std::distance(first, last));
}

template <typename ReturnT, typename ForwardIteratorT>
ReturnT variance(const ForwardIteratorT &first, const ForwardIteratorT &last)
{
    const auto m{ mean<ForwardIteratorT, ReturnT>(first, last) };
    const auto sum{ std::accumulate(first, last, ReturnT{ },
        [&] (const auto &s, const auto &v) {
        return s + (v - m) * (v - m);
    })};

    return sum / static_cast<ReturnT>(std::distance(first, last));
}

template <typename ReturnT, typename ForwardIteratorT>
ReturnT stddev(const ForwardIteratorT &first, const ForwardIteratorT &last)
{ return static_cast<ReturnT>(std::sqrt(variance<ForwardIteratorT, ReturnT>(first, last))); }

template <typename ReturnT, typename ForwardIteratorT>
std::pair<ReturnT, ReturnT> meanVariance(const ForwardIteratorT &first, const ForwardIteratorT &last)
{
    const auto sum{ std::reduce(first, last, ReturnT{ }, std::plus<>()) };
    const auto varSum{ std::accumulate(first, last, ReturnT{ },
        [&] (const auto &s, const auto &v) {
            return s + (v - sum) * (v - sum);
        })};
    const auto count{ static_cast<ReturnT>(std::distance(first, last)) };

    return { sum / count, varSum / count};
}

template <typename ReturnT, typename ForwardIteratorT>
ReturnT sampleMean(const ForwardIteratorT &first, const ForwardIteratorT &last)
{
    const auto sum{ std::reduce(first, last, ReturnT{ }, std::plus<>()) };

    return sum / static_cast<ReturnT>(std::distance(first, last) + 1);
}

template <typename ReturnT, typename ForwardIteratorT>
ReturnT sampleVariance(const ForwardIteratorT &first, const ForwardIteratorT &last)
{
    if (first == last)
    { return { 0, 0 }; }

    const auto m{ sampleMean<ForwardIteratorT, ReturnT>(first, last) };
    const auto sum{ std::accumulate(first, last, ReturnT{ },
        [&] (const auto &s, const auto &v) {
            return s + (v - m) * (v - m);
        })};

    return sum / static_cast<ReturnT>(std::distance(first, last) + 1);
}

template <typename ReturnT, typename ForwardIteratorT>
ReturnT sampleStddev(const ForwardIteratorT &first, const ForwardIteratorT &last)
{ return static_cast<ReturnT>(std::sqrt(sampleVariance<ForwardIteratorT, ReturnT>(first, last))); }

template <typename ReturnT, typename ForwardIteratorT>
std::pair<ReturnT, ReturnT> sampleMeanVariance(const ForwardIteratorT &first, const ForwardIteratorT &last)
{
    if (first == last)
    { return { { }, { } }; }

    const auto count{ static_cast<ReturnT>(std::distance(first, last) + 1) };
    const auto sum{ std::reduce(first, last, ReturnT{ }, std::plus<>()) };
    const auto mean{ sum / count };
    const auto varSum{ std::accumulate(first, last, ReturnT{ },
        [&] (const auto &s, const auto &v) {
            const auto vs{ static_cast<ReturnT>(v - mean) };
            return static_cast<ReturnT>(s) + vs * vs;
        })};
    const auto var{ varSum / count };

    return { mean, var };
}

template <typename ReturnT, typename ForwardIteratorT>
std::pair<ReturnT, ReturnT> minMax(const ForwardIteratorT &first, const ForwardIteratorT &last)
{
    auto min{ std::numeric_limits<ReturnT>::max() };
    auto max{ std::numeric_limits<ReturnT>::min() };
    for (auto it = first; it != last; ++it)
    { min = std::min<ReturnT>(min, *it); max = std::max<ReturnT>(max, *it); }

    return { min, max };
}

}

template <typename VT>
typename StochasticVariable<typename StochasticVariable<VT>::ValueT>::StatisticProperties
    StochasticVariable<VT>::StatisticProperties::simpleProperties() const
{
    if constexpr (PairVariable)
    { // Paired variable -> Convert to simple form using the first elements.
        return {
            mean.first, variance.first,
            count,
            min.first, max.first
        };
    }
    else
    { return *this; }
}

template <typename VT>
StochasticVariable<VT>::StochasticVariable()
{ clear(); }
template <typename VT>
StochasticVariable<VT>::~StochasticVariable()
{ /* Automatic */ }

template <typename VT>
void StochasticVariable<VT>::observeSample(const VT &value)
{ mObserver.observeSample(value); mPropertiesDirty = true; }

template <typename VT>
template <typename ForwardIteratorT>
void StochasticVariable<VT>::observeSamples(const ForwardIteratorT &first, const ForwardIteratorT &last)
{ mObserver.observeSamples(first, last); mPropertiesDirty = true; }

template <typename VT>
void StochasticVariable<VT>::clear()
{ mObserver.clear(); mObserver.setKeepObservations(true); mPropertiesDirty = true; }

template <typename VT>
void StochasticVariable<VT>::clearObservations(bool calculateProperties)
{ if (calculateProperties) { properties(); } mObserver.clearObservations(); }

template <typename VT>
const typename StochasticVariable<VT>::StatisticProperties &StochasticVariable<VT>::properties()
{
    if (!mPropertiesDirty)
    { return mProperties; }

    mPropertiesDirty = false;

    if (mObserver.values().empty())
    { mProperties = { }; }
    else
    { mProperties = calculateProperties(); }

    return mProperties;
}

template <typename VT>
const typename StochasticVariable<VT>::StatisticProperties &StochasticVariable<VT>::properties() const
{
    if (mPropertiesDirty)
    { throw StochasticException("Calling properties() on const object with dirty properties!"); }
    return mProperties;
}

template <typename VT>
const std::vector<VT> &StochasticVariable<VT>::observations() const
{ return mObserver.values(); }

template <typename VT>
template <typename CountT>
TreeHistogram<CountT, typename StochasticVariable<VT>::ValueT>
    StochasticVariable<VT>::prepareHistogram(
        const ValueT &step, std::size_t minBuckets, std::size_t maxBuckets,
        const ValueT &automaticStepsPerVariance)
{
    if (mObserver.values().empty())
    { return TreeHistogram<CountT, ValueT>{ }; }

    const auto props{ properties().simpleProperties() };
    const auto interval{ props.max - props.min };

    auto cStep{
        step == ValueT{ } ?
        (std::numeric_limits<ValueT>::is_exact ?
         static_cast<ValueT>(std::ceil(props.variance / static_cast<double>(automaticStepsPerVariance))) :
         static_cast<ValueT>(props.variance / static_cast<double>(automaticStepsPerVariance))
        ) :
        step
    };
    cStep = !treeutil::aboveEpsilon<ValueT>(cStep) ? ValueT{ 1 } : cStep;

    static constexpr auto MIN_BUCKETS{ 10u };
    const auto cMinBuckets{ std::max<std::size_t>(minBuckets ? minBuckets : MIN_BUCKETS, MIN_BUCKETS) };
    static constexpr auto MAX_BUCKETS{ 100u };
    const auto cMaxBuckets{ std::min<std::size_t>(maxBuckets ? maxBuckets : MAX_BUCKETS, MAX_BUCKETS) };

    cStep = treeutil::aboveEpsilon<ValueT>(interval / cMinBuckets) ?
        (((interval / cStep) < cMinBuckets) ? interval / cMinBuckets : cStep) : cStep;
    cStep = treeutil::aboveEpsilon<ValueT>(interval / cMaxBuckets) ?
        (((interval / cStep) > cMaxBuckets) ? interval / cMaxBuckets : cStep) : cStep;

    static constexpr auto INTERVAL_EXTENSION{ 5u };
    const auto cMinExtension{
        std::min<ValueT>(
            (props.min - std::numeric_limits<ValueT>::min()) / cStep,
            INTERVAL_EXTENSION
        )
    };
    const auto cMaxExtension{
        std::min<ValueT>(
            (std::numeric_limits<ValueT>::max() - props.max) / cStep,
            INTERVAL_EXTENSION
        )
    };

    const auto cMin{ props.min - cMinExtension * cStep};
    const auto cMax{ props.max + cMaxExtension * cStep};

    const auto &values{ mObserver.values() };
    TreeHistogram<CountT, ValueT> histogram{
        cMin, cMax, cStep,
        cMinBuckets, cMaxBuckets, true, true
    };

    return histogram;
}

template <typename VT>
template <typename CountT>
TreeHistogram<CountT, typename StochasticVariable<VT>::ValueT>
    StochasticVariable<VT>::calculateHistogram(
        const ValueT &step, std::size_t minBuckets, std::size_t maxBuckets,
        const ValueT &automaticStepsPerVariance)
{
    if (mObserver.values().empty())
    { return TreeHistogram<CountT, ValueT>{ }; }

    auto histogram{ prepareHistogram<CountT>(step, minBuckets, maxBuckets, automaticStepsPerVariance) };

    const auto &values{ mObserver.values() };
    histogram.countData(values.begin(), values.end());

    return histogram;
}

template <typename VT>
void StochasticVariable<VT>::describe(std::ostream &out, const std::string &indent) const
{
    out << "[ StochasticVariable: \n"
        << indent << "\tObserver: ";
    mObserver.describe(out, indent + "\t");

    const auto properties{ mPropertiesDirty ? calculateProperties() : mProperties };
    out << indent << "\tMean: " << properties.mean << "\n"
        << indent << "\tVar: " << properties.variance << "\n"
        << indent << "\tCount: " << properties.count << "\n"
        << indent << "\tMin: " << properties.min << "\n"
        << indent << "\tMax: " << properties.max << "\n";

    out << indent << " ]";
}

template <typename VT>
void StochasticVariable<VT>::saveTo(treeio::json &out) const
{
    const auto &props{ properties() };

    out["stochastic"] = {
        { "mean", props.mean },
        { "variance", props.variance },
    };

    mObserver.saveTo(out);
}

template <typename VT>
typename StochasticVariable<VT>::StatisticProperties
    StochasticVariable<VT>::calculateProperties() const
{
    StatisticProperties properties{ };

    const auto &values{ mObserver.values() };
    const auto [mean, variance]{
        [&] () {
            if constexpr (PairVariable)
            { // Pair variable -> Compute mean and variance per value.
                const auto firstFetcher{ [] (auto &val) { return val.first; } };
                const auto secondFetcher{ [] (auto &val) { return val.second; } };
                return std::make_pair(
                    utils::sampleMeanVariance<BasePreciseValueT>(
                        treeutil::LambdaIterator(values.begin(), firstFetcher),
                        treeutil::LambdaIterator(values.end(), firstFetcher)),
                    utils::sampleMeanVariance<BasePreciseValueT>(
                        treeutil::LambdaIterator(values.begin(), secondFetcher),
                        treeutil::LambdaIterator(values.end(), secondFetcher))
                );
            }
            else
            { // Simple variable -> Compute mean and variance normally.
                return utils::sampleMeanVariance<BasePreciseValueT>(values.begin(), values.end());
            }
        }()
    };
    properties.mean = mean;
    properties.variance = variance;

    properties.count = mObserver.count();
    properties.min = mObserver.min();
    properties.max = mObserver.max();

    return properties;
}

template <typename VT, typename CT>
void StochasticVariableHistogram<VT, CT>::observeSample(const VT &sample)
{ var.observeSample(sample); }

template <typename VT, typename CT>
StochasticVariableHistogram<VT, CT>::StochasticVariableHistogram(
    const std::string &shortName, const std::string &longDescription) :
    name{ shortName }, description{ longDescription }
{ }

template <typename VT, typename CT>
void StochasticVariableHistogram<VT, CT>::calculateHistogram(std::size_t minBuckets)
{ hist = var.template calculateHistogram<CT>({ }, minBuckets); histPrepared = true; }

template <typename FuncT>
void TreeStatValues::forEach(const FuncT &func) const
{ const_cast<TreeStatValues*>(this)->forEach(func); }

template <typename FuncT>
void TreeStatValues::forEach(const FuncT &func)
{
    func(segmentThickness);
    func(segmentVolume);
    func(segmentsPerChain);
    func(chainsPerDepth);
    func(chainLength);
    func(chainTotalLength);
    func(chainDeformation);
    func(chainLengthRatio);
    func(chainAngleSumDelta);
    func(chainParentChildAngle);
    func(chainStraightness);
    func(chainSlope);
    func(chainMinThickness);
    func(chainMaxThickness);
    func(chainMinMaxThicknessRatio);
    func(chainMinChildAngle);
    func(chainMaxChildAngle);
    func(chainMinChildAngleTotal);
    func(chainMaxChildAngleTotal);
    func(chainSiblingAngle);
    func(chainMinSiblingAngle);
    func(chainMaxSiblingAngle);
    func(chainSiblingAngleTotal);
    func(chainMinSiblingAngleTotal);
    func(chainMaxSiblingAngleTotal);
    func(chainParentAngle);
    func(chainMinParentAngle);
    func(chainMaxParentAngle);
    func(chainParentAngleTotal);
    func(chainMinParentAngleTotal);
    func(chainMaxParentAngleTotal);

    func(chainTropismBranch);
    func(chainTropismHorizon);
    func(chainTropismParent);

    func(chainAsymmetryLeaf);
    func(chainAsymmetryLength);
    func(chainAsymmetryVolume);

    func(chainMonopodialBranchingFitness);
    func(chainSympodialMonochasialBranchingFitness);
    func(chainSympodialDichasialBranchingFitness);
    func(aggregateBranchingFitness);
    func(chainContinuousRamificationFitness);
    func(chainRhythmicRamificationFitness);
    func(chainDiffuseRamificationFitness);
    func(aggregateRamificationFitness);
    func(chainInternodeCount);
}

} // namespace treestat

template <typename CT, typename BT>
inline std::ostream &operator<<(std::ostream &out, const treestat::TreeHistogram<CT, BT> &histogram)
{ histogram.describe(out); return out; }

template <typename VT>
inline std::ostream &operator<<(std::ostream &out, const treestat::VariableObserver<VT> &observer)
{ observer.describe(out); return out; }

template <typename VT>
inline std::ostream &operator<<(std::ostream &out, const treestat::StochasticVariable<VT> &variable)
{ variable.describe(out); return out; }

// Template implementation end.

#endif // TREE_STAT_H
