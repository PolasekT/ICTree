/**
 * @author Tomas Polasek
 * @date 2.20.2020
 * @version 1.0
 * @brief Representation of a point cloud in 3D space.
 */

#ifndef TREEIO_ACCELERATION_POINTCLOUD_H
#define TREEIO_ACCELERATION_POINTCLOUD_H

#include <TreeIO/TreeIO.h>

#include "TreeIOAccelerationUtils.h"
#include "TreeIOPolygonalModel.h"

namespace treeacc
{

// Forward declaration.
class TriangleIntersector;

/// @brief Data for a single vertex in the triangle graph.
struct TriangleGraphVertex
{
    /// Information about the triangle.
    Triangle t;
}; // struct TriangleGraphVertex

/// @brief Data for a single edge in the triangle graph.
struct TriangleGraphEdge
{
    /// Distance between centers of the triangles.
    float distance{ 0.0f };
}; // struct TriangleGraphEdge

/// @brief Point cloud is a collection of vertices.
class PointCloud
{
public:
    /// Shortcut for pointer to a point cloud.
    using Ptr = std::shared_ptr<PointCloud>;

    /// Number of dimensions each point in the point cloud has.
    static constexpr std::size_t DATA_DIMENSIONALITY{ 3u };

    /// @brief Initialize empty point cloud.
    PointCloud();
    /// @brief Free all point cloud data.
    ~PointCloud();

    // Copy and move operators:
    PointCloud(const PointCloud &other) = default;
    PointCloud &operator=(const PointCloud &other) = default;
    PointCloud(PointCloud &&other) = default;
    PointCloud &operator=(PointCloud &&other) = default;

    /**
     * @brief Sample the input model into a point cloud.
     *
     * @param model The input model.
     * @param normalize Should the input model be normalized?
     *  Warning: when set to true the original model will be
     *  changed!
     * @param targetPointCount How many points should we roughly
     *  sample?
     * @param perTrianglePointCount Number of additional points per
     *  each triangle.
     * @param useFloating Should the floating samples be assigned
     *   by chance?
     * @param oversampleBottom Oversample bottom part of the model.
     *  The value specified how many percent of the total height
     *  should be oversampled.
     * @param oversampleFar Oversample parts of the model further
     *  from the middle. This options works best when normalization
     *  is enabled.
     * @param adjacencyTriangleMultiplier Multiplier for adjacency
     *  distance for 2 adjacent triangles.
     * @param adjacencyIntersectMultiplier Multiplier for adjacency
     *  distance for 2 intersecting triangles.
     */
    void sampleModel(PolygonalModel &model, bool normalize,
        float targetPointCount, float perTrianglePointCount = 0.0f,
        bool useFloating = true, float oversampleBottom = 0.05f,
        float oversampleFar = 0.05f,
        float adjacencyTriangleMultiplier = 1.0f,
        float adjacencyIntersectMultiplier = 16.0f);

    // Helper methods for NanoFLANN:

    /// @brief Getter which allows to use this structure with NanoFLANN.
    inline float kdtree_get_pt(const std::size_t idx, const std::size_t dim) const
    { return mPoints[idx].p[dim]; }

    /// @brief Get number of points. Allows to use this structure with NanoFLANN.
    inline std::size_t kdtree_get_point_count() const
    { return mPoints.size(); }

    /// @brief Let NanoFLANN compute the bounding box.
    template <class BBoxT>
    inline bool kdtree_get_bbox(BBoxT&) const
    { return false; }

    // Getters:

    /// @brief Access the original points array.
    const std::vector<Vertex3D> &originalPoints() const
    { return mOriginalPoints; }

    /// @brief Access the original points array.
    std::vector<Vertex3D> &originalPoints()
    { return mOriginalPoints; }

    /// @brief Access the points array.
    const std::vector<Vertex3D> &points() const
    { return mPoints; }

    /// @brief Access the points array.
    std::vector<Vertex3D> &points()
    { return mPoints; }

    /// @brief Access the mapping from points to original points.
    const std::vector<std::set<std::size_t>> &pointToOriginalMapping() const
    { return mPointToOriginalMapping; }

    /// @brief Access the mapping from points to original points.
    std::vector<std::set<std::size_t>> &pointToOriginalMapping()
    { return mPointToOriginalMapping; }

    /// @brief Access the acceleration structure containing triangles from the input model.
    const TriangleIntersector &intersector() const
    { return *mIntersector; }

    /// @brief Get original radius of the model.
    float originalRadius() const
    { return mOriginalRadius; }

    /// @brief Get original position of the model.
    const Vector3D &originalPosition() const
    { return mOriginalPosition; }
private:
    /// Vertices of the point cloud.
    std::vector<Vertex3D> mOriginalPoints{ };
    /// Vertices of the point cloud.
    std::vector<Vertex3D> mPoints{ };
    /// Mapping from points to original points.
    std::vector<std::set<std::size_t>> mPointToOriginalMapping{ };
    /// Original radius of the model. Used for un-normalization.
    float mOriginalRadius{ };
    /// Original center of the model. Used for un-normalization.
    Vector3D mOriginalPosition{ };
    /// Acceleration structure containing triangles from the input model.
    std::shared_ptr<TriangleIntersector> mIntersector{ };
protected:
}; // struct PointCloud

} // namespace treeacc

#endif // TREEIO_ACCELERATION_POINTCLOUD_H
