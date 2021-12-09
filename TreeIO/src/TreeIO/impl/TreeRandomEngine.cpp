/**
 * @author Tomas Polasek, David Hrusa
 * @date 28.7.2020
 * @version 1.0
 * @brief Generates random numbers the C++ way.
 */

#include "TreeRandomEngine.h"

#include <stdexcept>

namespace treeutil
{

RandomEngine::RandomEngine()
{ }

RandomEngine::~RandomEngine()
{ }

void RandomEngine::resetSeed()
{ resetSeed(time(nullptr)); }

void RandomEngine::resetLastSeed()
{ resetSeed(mLastSeed); }

void RandomEngine::resetSeed(int seed)
{ mLastSeed = seed; setSeed(seed); }

int RandomEngine::randomInt(int min, int max, int mult)
{
    switch (mDistribution)
    {
        case RandomDistribution::Uniform:
        { return mult * std::uniform_int_distribution{ min, max }(mGen); }
        case RandomDistribution::Normal:
        {
            // Place it in the middle.
            const auto mean{ (min + max) / 2.0f };
            // 3 sigma.
            const auto stddev{ (max - min) / 6.0f };
            return static_cast<int>(mult * std::normal_distribution<float>{ mean, stddev }(mGen));
        }
        default:
        { throw std::runtime_error("Distribution not implemented!"); }
    }
}

float RandomEngine::randomFloat(float min, float max, float mult)
{
    switch (mDistribution)
    {
        case RandomDistribution::Uniform:
        { return mult * std::uniform_real_distribution{ min, max }(mGen); }
        case RandomDistribution::Normal:
        {
            // Place it in the middle.
            const auto mean{ (min + max) / 2.0f };
            // 3 sigma.
            const auto stddev{ (max - min) / 6.0f };
            return mult * std::normal_distribution<float>{ mean, stddev }(mGen);
        }
        default:
        { throw std::runtime_error("Distribution not implemented!"); }
    }
}

int RandomEngine::lastSeed() const
{ return mLastSeed; }

RandomDistribution RandomEngine::distribution() const
{ return mDistribution; }

void RandomEngine::setDistribution(RandomDistribution distribution)
{ mDistribution = distribution; }

std::string RandomEngine::getDistributionName() const 
{
    switch (mDistribution)
    {
        case RandomDistribution::Uniform:
        { return "uniform"; }
        case RandomDistribution::Normal:
        { return "normal"; }
        default:
        { return "unknown"; }
    }
}

void RandomEngine::setSeed(int seed)
{ mGen.seed(seed); }

} // namespce treeutil
