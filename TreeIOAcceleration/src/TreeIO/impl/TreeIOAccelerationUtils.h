/**
 * @author Tomas Polasek
 * @date 10.13.2020
 * @version 1.0
 * @brief Utilities and statistics for the tree acceleration classes.
 */

#ifndef TREEIO_ACCELERATION_UTILS_H
#define TREEIO_ACCELERATION_UTILS_H

#include <array>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include <glm/glm.hpp>
#define HAS_GLM

#include <TreeIO/TreeIO.h>

namespace treeacc
{

// Forward logging utilities:
using treeutil::Debug;
using treeutil::Info;
using treeutil::Warning;
using treeutil::Error;

// Allow simple use of Vector3D and Vector2D.
using treeutil::Vector3D;
using treeutil::Vector2D;

/// @brief Vertex in the point cloud.
struct Vertex3D
{
    /// Position.
    Vector3D p{ };
    /// Normal.
    Vector3D n{ };
    /// Triangle index.
    std::size_t t{ };
}; // struct Vertex3D

/// @brief Wrapper around line information.
struct Line
{
    /// @brief Create a line from origin and direction.
    static Line fromDirection(const Vector3D &origin, const Vector3D &direction);

    /// @brief Initialize invalid line.
    Line() = default;

    /// @brief Create from 2 points on the line.
    Line(const Vector3D &a, const Vector3D &b) :
        v{ a }, w{ b }
    { }

    /// First defining point.
    Vector3D v{ };
    /// Second defining point.
    Vector3D w{ };
}; // struct Line

/// @brief Wrapper around ray information.
struct Ray
{
    /// @brief Initialize invalid ray.
    Ray() = default;

    /// @brief Create ray from its origin and direction.
    Ray(const Vector3D &o, const Vector3D &d) :
        origin{ o }, direction{ d }
    { }

    /// Origin of the ray
    Vector3D origin{ };
    /// Direction of the ray.
    Vector3D direction{ };
}; // struct Ray

/// @brief Axis aligned bounding box in 3D.
struct BoundingBox3D
{
    /// @brief Create empty bounding box.
    BoundingBox3D();
    /// @brief Create tight bounding box around given point.
    BoundingBox3D(const Vector3D &pos);
    /// @brief Create bounding box with given corners - min and max.
    BoundingBox3D(const Vector3D &lv, const Vector3D &hv);

    /// @brief Does given ray intersect this AABB?
    bool intersects(const Ray &ray) const;

    /// @brief Calculate ray intersection with this bounding box. Returns max if no intersections were found.
    float intersection(const Ray &ray) const;

    /// Minimum corner of the bounding box.
    Vector3D min{ };
    /// Minimum corner of the bounding box.
    Vector3D max{ };
}; // struct BoundingBox3D

/// @brief Wrapper around triangle information.
struct Triangle
{
    Triangle() = default;
    ~Triangle() = default;

    /// @brief Initialize triangle from given buffers.
    Triangle(std::size_t idx,
        const treeio::ModelImporterBase::Vertex *vb,
        const treeio::ModelImporterBase::IndexElementT *ib);

    /// @brief Calculate distance from centroid to some other triangle.
    float centroidDistance(const Triangle &other) const;

    /// @brief Calculate squared distance from centroid to some other triangle.
    float centroidSquaredDistance(const Triangle &other) const;

    /// @brief Calculate squared distance from centroid to some point.
    float centroidSquaredDistance(const Vector3D &point) const;

    /// @brief Does this triangle intersect with the other one?
    bool intersects(const Triangle &other) const;

    /// @brief Does given ray intersect this triangle?
    bool intersects(const Ray &ray) const;

    /// @brief Calculate ray intersection with this triangle. Returns max if no intersections were found.
    float intersection(const Ray &ray) const;

    /// @brief Does this triangle contain given point?
    bool contains(const Vector3D &point, float epsilon = std::numeric_limits<float>::epsilon()) const;

    /// @brief Calculate barycentric coordinates for given point with respect to v1, v2, v3.
    Vector3D barycentrics(const Vector3D &point) const;

    /// @brief Compute Axis-Aligned Bounding Box.
    BoundingBox3D aabb() const;

    /// @brief Calculate area of the triangle.
    float area() const;
    /// @brief Calculate perimeter of the triangle.
    float perimeter() const;
    /// @brief Calculate the longest altitude of the triangle.
    float longestAltitude() const;
    /// @brief Get length of the longest side of this triangle.
    float longestSide() const;

    /// @brief Calculate plane of the triangle plane.
    Vector3D centralPlaneNormal() const;

    /// @brief Set plane normals for each vertex.
    void providePlaneNormals(const Vector3D &pNormal1, const Vector3D &pNormal2, const Vector3D &pNormal3);

    /// @brief Calculate smoothed point position using plane normals.
    Vector3D smoothPoint(const Vector3D &point) const;

    /**
     * @brief Calculate curvature of this triangle within the triangle mesh.
     *
     * @return Returns metric of curvature <0.0f, 1.0f> (least -> most).
     * @warning Only available when plane normals are provided.
     */
    float curvature() const;

    /**
     * @brief Calculate curvature weight at v1, v2 and v3. Weights sum to one.
     *
     * @return Returns metric of curvature <0.0f, 1.0f>^3 (least -> most). All weights sum to one.
     * @warning Only available when plane normals are provided.
     */
    Vector3D curvatureWeights() const;

    /// @brief Compare two triangles for equality.
    bool operator==(const Triangle &other) const
    { return i1 == other.i1 && i2 == other.i2 && i3 == other.i3; }

    /// @brief Index of the triangle within the triangle buffer.
    std::size_t tIdx{ 0u };

    // Indices.
    int i1{ -1 };
    int i2{ -1 };
    int i3{ -1 };

    // Vertex positions.
    Vector3D v1{ };
    Vector3D v2{ };
    Vector3D v3{ };
    /// Lowest vertex positions.
    Vector3D lv{ };
    /// Highest vertex positions.
    Vector3D hv{ };
    /// Centroid.
    Vector3D c{ };

    // Vertex normals.
    Vector3D n1{ };
    Vector3D n2{ };
    Vector3D n3{ };

    // Are plane normals available?
    bool providedPlaneNormals{ false };

    // Plane normals. Only set when provided by providePlaneNormals.
    Vector3D pn1{ };
    Vector3D pn2{ };
    Vector3D pn3{ };
    // Plane radii. Only set when provided by providePlaneNormals.
    float pr12{ };
    float pr13{ };
    float pr23{ };
}; // struct Triangle

/// @brief Simple timer class using std::chrono::high_resolution_clock.
class Timer
{
public:
    /// @brief Clock used by this timer.
    using Clock = std::chrono::high_resolution_clock;
    /// @brief Type representing seconds elapsed.
    using SecondsT = double;

    /// @brief Initialize timer and start.
    inline Timer();

    /// @brief Reset the timer and return seconds elapsed since last reset.
    inline SecondsT reset();

    /// @brief Get seconds elapsed since last reset.
    inline SecondsT elapsed() const;
private:
    /// Time at which this timer started counting.
    Clock::time_point mStart{ };
protected:
}; // class Timer

/// @brief Helper method for calculating area of a triangle given by 3 points.
float calculateTriangleArea(const Vector3D &p1, const Vector3D &p2, const Vector3D &p3);

/// @brief Helper method for calculating perimeter of a triangle given by 3 points.
float calculateTrianglePerimeter(const Vector3D &p1, const Vector3D &p2, const Vector3D &p3);

/// @brief Calculate median point of given triangle.
Vector3D calculateTriangleMedianPoint(const Vector3D &p1, const Vector3D &p2, const Vector3D &p3);

/// @brief Helper method for calculating triangles median line from p1p2 to p3.
float calculateTriangleMedian(const Vector3D &p1, const Vector3D &p2, const Vector3D &p3);

/// @brief Helper method for calculating triangles longest median.
float calculateTriangleLongestMedian(const Vector3D &p1, const Vector3D &p2, const Vector3D &p3);

/// @brief Sample triangle using barycentric coordinates.
Vector3D sampleTriangle(const Vector3D &a1, const Vector3D &a2, const Vector3D &a3, const Vector3D &w);

/// @brief Generate barycentric weights for sampling of given triangle with given densities phi.
Vector3D triangleWeights(const Vector3D &p1, const Vector3D &p2, const Vector3D &p3, const Vector3D &phi);

/// @brief Generate barycentric weights for uniform sampling of given triangle.
Vector3D uniformTriangleWeights(const Vector3D &p1, const Vector3D &p2, const Vector3D &p3);

/// @brief Generate barycentric weights for sampling of given triangle using distance from median.
Vector3D medianTriangleWeights(const Vector3D &p1, const Vector3D &p2, const Vector3D &p3);

/// @brief Perform random uniform sampling of a triangles attributes.
Vertex3D uniformSampleTriangle(const Vertex3D &v1, const Vertex3D &v2, const Vertex3D &v3);

/// @brief Perform random uniform sampling of a triangles attributes with vertex weights.
Vertex3D uniformSampleTriangle(const Vertex3D &v1, const Vertex3D &v2, const Vertex3D &v3, const Vector3D &weights);

/// @brief Sample triangles point in the middle.
Vector3D middleSampleTriangle(const Vector3D &a1, const Vector3D &a2, const Vector3D &a3);

/// @brief Calculate triangle normal.
Vector3D triangleNormal(const Vector3D &a1, const Vector3D &a2, const Vector3D &a3);

/// @brief Convert vertex from model to a cloud point.
inline Vector3D modelVertexToCloudPoint(const treeio::ModelImporterBase::Vertex &v);

/// @brief Convert normal from model to a cloud point.
inline Vector3D modelNormalToCloudNormal(const treeio::ModelImporterBase::Vertex &v);

/// @brief Convert vertex from model to a cloud point.
inline Vector3D modelVertexToCloudPoint(const Vector3D &v);

/// @brief Convert vertex from cloud point to a model vertex.
inline treeio::ModelImporterBase::Vertex cloudPointToModelVertex(const Vector3D &v);

/// @brief Convert vertex from cloud point to a model point.
inline Vector3D cloudPointToModelPoint(const Vector3D &v);

/// @brief Calculate least-squares intersection of a set of lines.
Vector3D lsIntersect(const std::vector<Line> &lines);

/// @brief Calculate least-squares intersection of a set of lines.
Vector3D lsIntersect(std::initializer_list<Line> lines);

} // namespace treeacc

// Template implementation begin.

namespace treeacc
{

Timer::Timer()
{ reset(); }

inline Timer::SecondsT Timer::reset()
{ const auto result{ elapsed() }; mStart = Clock::now(); return result; }

inline Timer::SecondsT Timer::elapsed() const
{ return std::chrono::duration_cast<std::chrono::duration<SecondsT>>(Clock::now() - mStart).count(); }

inline Vector3D modelVertexToCloudPoint(const treeio::ModelImporterBase::Vertex &v)
{ return { v.position[0], -v.position[2], v.position[1] }; }

inline Vector3D modelNormalToCloudNormal(const treeio::ModelImporterBase::Vertex &v)
{ return { v.normal[0], -v.normal[2], v.normal[1] }; }
//{ return { -v.normal[0], v.normal[2], -v.normal[1] }; }

inline Vector3D modelVertexToCloudPoint(const Vector3D &v)
{ return { v[0], -v[2], v[1] }; }

inline treeio::ModelImporterBase::Vertex cloudPointToModelVertex(const Vector3D &v)
{ return { v.x, v.z, -v.y }; }

inline Vector3D cloudPointToModelPoint(const Vector3D &v)
{ return { v.x, v.z, -v.y }; }

} // namespace treeacc

// Template implementation end.

#endif // TREEIO_ACCELERATION_UTILS_H
