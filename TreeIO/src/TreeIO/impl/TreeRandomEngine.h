/**
 * @author Tomas Polasek, David Hrusa
 * @date 28.7.2020
 * @version 1.0
 * @brief Generates random numbers the C++ way.
 */

#ifndef CGANTREE_TREERANDOMENGINE_H
#define CGANTREE_TREERANDOMENGINE_H

#include <random>
#include <cmath>
#include <time.h>

namespace treeutil
{

/// @brief Random number distributions.
enum class RandomDistribution
{
    Uniform,
    Normal,
}; // enum class RandomDistribution

/// @brief generic interface for an implemented random engine to be sloted in.
class RandomEngine
{
public:
    /// @brief Initialize default random engine.
    RandomEngine();
    /// @brief Clean up and destroy.
    virtual ~RandomEngine();

    /// @brief Set seed based on current time.
    virtual void resetSeed();

    /// @brief Set seed to the last value.
    virtual void resetLastSeed();

    /// @brief Set seed to given value.
    virtual void resetSeed(int seed);

    /// @brief Get random integer from currently chosen distribution.
    virtual int randomInt(int min = 0, int max = 100, int mult = 1);

    /// @brief Get random float from currently chosen distribution.
    virtual float randomFloat(float min = 0.0f, float max = 1.0f, float mult = 1.0f);

    /// @brief Get the last used seed value.
    virtual int lastSeed() const;

    /// @brief Get currently used distribution.
    virtual RandomDistribution distribution() const;

    /// @brief Set distribution function to use.
    virtual void setDistribution(RandomDistribution distribution);

    /// @brief Get name of the distribution used.
    virtual std::string getDistributionName() const;
private:
    /// Last set seed value.
    int mLastSeed{ 0 };

    /// Randomness generator.
    std::mt19937 mGen{ };
    /// Currently used distribution.
    RandomDistribution mDistribution{ RandomDistribution::Uniform };
protected:
    /// @brief Set seed based on implementation.
    virtual void setSeed(int seed);
}; // class RandomEngine

} // namespace treeutil

#endif //CGANTREE_TREERANDOMENGINE_H
