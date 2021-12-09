/**
 * @author Tomas Polasek
 * @date 2.20.2020
 * @version 1.0
 * @brief Representation of a point cloud in 3D space.
 */

#include "TreeIOPointCloud.h"

#include "TreeIOAcceleration.h"

namespace treeacc
{

std::vector<std::size_t> getConnectedTriangles(const Triangle &triangle,
    const std::multimap<std::size_t, std::size_t> &indexToTriangle)
{
    std::vector<std::size_t> connectedTriangles{ };
    {
        const auto [findIt, endIt]{ indexToTriangle.equal_range(triangle.i1) };
        for (auto it = findIt; it != endIt; ++it)
        { connectedTriangles.push_back(it->second); }
    }
    {
        const auto [findIt, endIt]{ indexToTriangle.equal_range(triangle.i2) };
        for (auto it = findIt; it != endIt; ++it)
        { connectedTriangles.push_back(it->second); }
    }
    {
        const auto [findIt, endIt]{ indexToTriangle.equal_range(triangle.i2) };
        for (auto it = findIt; it != endIt; ++it)
        { connectedTriangles.push_back(it->second); }
    }

    // Include only unique connections.
    std::sort(connectedTriangles.begin(), connectedTriangles.end());
    const auto last{ std::unique(connectedTriangles.begin(), connectedTriangles.end())};
    connectedTriangles.erase(last, connectedTriangles.end());

    return connectedTriangles;
}

std::vector<std::size_t> getConnectedTriangles(const Triangle &triangle,
    const TriangleIntersector &intersector)
{
    std::vector<std::size_t> connectedTriangles{ };
    {
        const auto results{ intersector.queryPoint(triangle.v1) };
        connectedTriangles.insert(connectedTriangles.end(), results.begin(), results.end());
    }
    {
        const auto results{ intersector.queryPoint(triangle.v2) };
        connectedTriangles.insert(connectedTriangles.end(), results.begin(), results.end());
    }
    {
        const auto results{ intersector.queryPoint(triangle.v3) };
        connectedTriangles.insert(connectedTriangles.end(), results.begin(), results.end());
    }

    // Include only unique connections.
    std::sort(connectedTriangles.begin(), connectedTriangles.end());
    const auto last{ std::unique(connectedTriangles.begin(), connectedTriangles.end())};
    connectedTriangles.erase(last, connectedTriangles.end());

    return connectedTriangles;
}

Vector3D calculateAveragePlaneNormal(const Vector3D &point, const std::vector<Triangle> &triangles,
    const TriangleIntersector &intersector)
{
    const auto connectedTriangles{ intersector.queryPoint(point) };

    Vector3D planeNormalSum{ };
    for (const auto &tIdx : connectedTriangles)
    { planeNormalSum += triangles[tIdx].centralPlaneNormal(); }

    auto averagePlaneNormal{ planeNormalSum / connectedTriangles.size() };
    averagePlaneNormal.normalize();

    return averagePlaneNormal;
}

PointCloud::PointCloud() :
    mIntersector{ std::make_shared<TriangleIntersector>() }
{ /* Automatic */ }

PointCloud::~PointCloud()
{ /* Automatic */ }

void PointCloud::sampleModel(PolygonalModel &model,
    bool normalize, float targetPointCount, float perTrianglePointCount,
    bool useFloating, float oversampleBottom, float oversampleFar,
    float adjacencyTriangleMultiplier, float adjacencyIntersectMultiplier)
{
    Info << "Sampling input model..." << std::endl;

    const auto originalRad{ model.radius() };
    const auto originalPos{ model.centerOfMass() };

    static constexpr auto CENTER_NORMALIZE{ false };

    if (normalize)
    {
        std::cout << "\tNormalizing the input model..." << std::endl;
        model.normalize(1.0f, CENTER_NORMALIZE);
        std::cout << "\t\tDone!" << std::endl;
    }

    mOriginalRadius = originalRad;
    if (CENTER_NORMALIZE)
    { mOriginalPosition = originalPos; }
    else
    { mOriginalPosition = Vector3D{ 0.0f, 0.0f, 0.0f }; }

    const auto indexBuffer{ model.indexBuffer() };
    const auto vertexBuffer{ model.vertexBuffer() };
    const auto triangleCount{ static_cast<std::size_t>(model.triangleCount()) };

    Info << "\tThe model has " << triangleCount << " triangles, " << model.vertexCount()
              << " vertices and " << model.indexCount() << " indices." << std::endl;

    Info << "\tPreparing triangle vertex data and creating adjacency list..." << std::endl;

    std::vector<Triangle> triangles{ };
    triangles.resize(triangleCount);
    std::multimap<std::size_t, std::size_t> indexToTriangle{ };

    auto totalArea{ 0.0f };
    auto totalPerimeter{ 0.0f };
    auto totalLength{ 0.0f };
    auto bottom{ std::numeric_limits<float>::max() };
    auto top{ std::numeric_limits<float>::min() };
    auto maxDistance{ 0.0f };
    for (std::size_t tIdx = 0u; tIdx < triangleCount; ++tIdx)
    {
        triangles[tIdx] = Triangle(tIdx, vertexBuffer, indexBuffer);
        totalArea += triangles[tIdx].area();
        totalPerimeter += triangles[tIdx].perimeter();
        totalLength += triangles[tIdx].longestAltitude();

        bottom = std::min<float>(bottom, triangles[tIdx].lv.z);
        top = std::max<float>(top, triangles[tIdx].lv.z);
        maxDistance = std::max<float>(maxDistance, triangles[tIdx].longestSide());

        // Remember which index is shared by which triangles.
        indexToTriangle.insert({ triangles[tIdx].i1, tIdx });
        indexToTriangle.insert({ triangles[tIdx].i2, tIdx });
        indexToTriangle.insert({ triangles[tIdx].i3, tIdx });
    }

    const auto height{ top - bottom };

    Info << "\tModels total area: " << totalArea << " total perimeter: " << totalPerimeter << std::endl;
    Info << "\tModels bottom is at: " << bottom << " top is at: " << top << std::endl;
    Info << "\tModels is " << height << " units high" << std::endl;

    Info << "\tInitializing triangle acceleration structure..." << std::endl;
    mIntersector->create(triangles);
    Info << "\t\tDone!" << std::endl;

    Info << "\tCalculating curvature coefficients..." << std::endl;
    auto totalCurvature{ 0.0f };
    for (auto &t : triangles)
    {
        const auto pn1{ calculateAveragePlaneNormal(t.v1, triangles, *mIntersector) };
        const auto pn2{ calculateAveragePlaneNormal(t.v2, triangles, *mIntersector) };
        const auto pn3{ calculateAveragePlaneNormal(t.v3, triangles, *mIntersector) };

        t.providePlaneNormals(pn1, pn2, pn3);
        totalCurvature += t.curvature();
    }
    Info << "\t\tDone!" << std::endl;

    Info << "\tSampling the model and connecting adjacent triangles..." << std::endl;

    // TODO - Fix this to compensate correctly.
    // We are automatically moving the model to origin.
    if (normalize)
    { mOriginalPosition.z -= mOriginalRadius * bottom; }

    auto sampleAccumulator{ 0.0f };
    for (std::size_t tIdx = 0u; tIdx < triangleCount; ++tIdx)
    {
        const auto &t{ triangles[tIdx] };

        // Base number of samples is based on ratio of the triangles area.
        const auto triangleAreaSamples{ t.area() / totalArea * targetPointCount };
        // The ratio of the triangle perimeter.
        const auto trianglePerimeterSamples{ t.perimeter() / totalPerimeter * targetPointCount };
        // The ratio of the triangle length.
        const auto triangleLengthSamples{ t.longestAltitude() / totalLength * targetPointCount };
        // And the ratio of the triangle curvature.
        const auto triangleCurvatureSamples{ t.curvature() / totalCurvature * targetPointCount };
        // Oversampling the bottom part of the tree.
        const auto oversamplingBottom{ std::max<float>(
                // Add 1 sample at the bottom and linearly lower until 0 at oversampleBottom% .
                (oversampleBottom >= std::numeric_limits<float>::epsilon() ?
                 (-(t.lv.z - bottom) / height + oversampleBottom) * (1.0f / oversampleBottom) * 1.0f :
                 0.0f),
                0.0f
        ) };
        // TODO - Split into far in the z-axis and far in the xy plane?
        // Oversampling branches further from the middle.
        const auto oversamplingFar{ std::max<float>(
                // Add 1 sample at the far ends and linearly lower until 0 at oversampleBottom% .
                (oversampleFar >= std::numeric_limits<float>::epsilon() ?
                 (-(1.0f - t.longestSide() / maxDistance) + oversampleFar) * (1.0f / oversampleFar) * 1.0f :
                 0.0f),
                0.0f
        ) };
        sampleAccumulator += std::max({
                triangleAreaSamples, trianglePerimeterSamples,
                triangleLengthSamples, triangleCurvatureSamples
            }) + oversamplingBottom + oversamplingFar;
        sampleAccumulator += perTrianglePointCount;

        if (useFloating)
        {
            // Get the floating point part of the accumulator and check if we get a real sample.
            const auto floatingAccumulator{ sampleAccumulator - static_cast<int>(sampleAccumulator) };
            sampleAccumulator -= floatingAccumulator;
            auto randomSample{ treeutil::uniformZeroToOne() };
            if (floatingAccumulator > randomSample)
            { sampleAccumulator += 1.0f; }
        }

        const Vertex3D v1{ t.v1, t.n1 };
        const Vertex3D v2{ t.v2, t.n2 };
        const Vertex3D v3{ t.v3, t.n3 };
        // TODO - Use smooth normals calculated from adjacent triangles?
        /*
        const Vertex3D v1{ t.v1, t.pn1 };
        const Vertex3D v2{ t.v2, t.pn2 };
        const Vertex3D v3{ t.v3, t.pn3 };
        */
        const auto curvatureWeights{ t.curvatureWeights() };

        for (; sampleAccumulator >= 1.0f; sampleAccumulator -= 1.0f)
        { // Add whole samples.
            const auto sample{ uniformSampleTriangle(v1, v2, v3, curvatureWeights) };
            // Move samples to start at zero z.
            const Vertex3D movedSample{
                    //Vector3D{ sample.p.x, sample.p.y, sample.p.z - bottom },
                    Vector3D{ sample.p.x, sample.p.y, sample.p.z - bottom },
                    sample.n, tIdx
            };
            mPoints.push_back(movedSample);
        }
    }

    Info << "\t\tDone!" << std::endl;

    Info << "\tGenerated " << mPoints.size() << " samples from given " << model.vertexCount() << " input vertices." << std::endl;
    Debug << "\tRemaininng accumulator: " << sampleAccumulator << std::endl;
}

} // namespace treeacc
