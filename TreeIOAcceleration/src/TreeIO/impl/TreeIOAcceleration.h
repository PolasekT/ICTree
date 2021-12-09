/**
 * @author Tomas Polasek
 * @date 3.23.2020
 * @version 1.0
 * @brief Acceleration structures and query helpers.
 */

#ifndef TREEIO_ACCELERATION_ACCELERATION_H
#define TREEIO_ACCELERATION_ACCELERATION_H

#include <TreeIO/TreeIO.h>
#include <kd_tree/KdTree.h>
#include <nanoflann/nanoflann.hpp>

#include "TreeIOAccelerationUtils.h"
#include "TreeIOPointCloud.h"

namespace treeacc
{

namespace impl
{

/// @brief Implementation of the triangle intersector.
struct TriangleIntersectorImpl;

}

/// @brief Wrapper around spatial index and its data.
struct PointCloudAcceleration
{
public:
    /// Reference to the old type of KD Tree.
    using LegacyKdTree = kd::KdTree;

    /// NanoFLANN indexing structure.
    using KdTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointCloud, float>,
        PointCloud, PointCloud::DATA_DIMENSIONALITY>;

    /// Pointer to the indexing structure.
    using KdTreePtr = std::shared_ptr<KdTree>;
    /// Pointer to the old type of KD Tree.
    using LegacyKdTreePtr = std::shared_ptr<LegacyKdTree>;
    /// Pointer to the point data.
    using PointCloudPtr = std::shared_ptr<PointCloud>;

    /// @brief Wrapper around results for a NN query.
    struct QueryResult
    {
        /// Type used to represent distance.
        using DistanceT = float;
        /// Type used to point index.
        using IndexT = std::size_t;
        /// Index-distance pair.
        using IndexDistancePairT = std::pair<IndexT, DistanceT>;

        /// @brief Get index of neighbor with given index.
        inline IndexT getNeighborPositionIndex(std::size_t idx) const
        { return points[idx].first; }

        /// @brief Get distance of neighbor with given index.
        inline DistanceT getNeighborPositionDistance(std::size_t idx) const
        { return points[idx].second; }

        /// Indices and distances of the located neighbors.
        std::vector<IndexDistancePairT> points{ };

        /// Are the points and their distances sorted from nearest?
        bool sorted{ };

        /// Source point for the query.
        Vertex3D source{ };
    }; // struct QueryResult

    /// @brief Emulate pointer to index behavior.
    KdTreePtr operator->()
    { return mIndex; }

    /// @brief Access the internal point cloud data structure.
    PointCloud &cloud()
    { return *mData; }

    /// @brief Access the internal point cloud data structure.
    const PointCloud &cloud() const
    { return *mData; }

    /// @brief Create index for given point cloud.
    void fromCloud(const PointCloud &cloud, std::size_t maxLeafSize);
    /// @brief Create index for given point cloud.
    void fromCloud(PointCloud &&cloud, std::size_t maxLeafSize);
    /// @brief Create index for given vertices.
    void fromVertices(const std::vector<Vertex3D> &vertices, std::size_t maxLeafSize);
    /// @brief Create index for given vertices.
    void fromVertices(std::vector<Vertex3D> &&vertices, std::size_t maxLeafSize);
    /// @brief Create index for given vertices.
    void fromVertices(const Vertex3D *vertices, std::size_t vertexCount, std::size_t maxLeafSize);

    /// @brief Perform a search for all nearest points within given range around the source point.
    QueryResult queryRange(const Vector3D &p, float range, std::size_t maxNeighbors = 0, bool sort = true) const;
    /// @brief Perform a search for all a given number of nearest neighbors around the source point.
    QueryResult queryKNN(const Vector3D &p, std::size_t maxNeighbors) const;
    /// @brief Search for a single nearest point from given position.
    QueryResult queryPosition(const Vector3D &p) const;

    /// @brief Get legacy KD Tree from this indexing structure.
    LegacyKdTreePtr toLegacyKdTree() const;
private:
    /// Index for the data.
    KdTreePtr mIndex{ };
    /// Data used by the index.
    PointCloudPtr mData{ };
protected:
}; // class PointCloudAcceleration

/// @brief Wrapper around a set of triangles which allows accelerated query operations using BVH.
class TriangleIntersector
{
public:
    /// @brief Primitive stored in the acceleration structure.
    using PrimitiveT = Triangle;
    /// @brief Scalar used for distances and inner operations.
    using ScalarT = float;
    /// @brief Type used for specifying the set of primitive.
    using PrimitiveSetT = std::vector<Triangle>;
    /// @brief Indexing type used to distinguish primitives.
    using PrimitiveIdxT = decltype(PrimitiveT::tIdx);
    /// @brief Type of result provided by the intersect operations.
    using IntersectResultT = std::vector<PrimitiveIdxT>;
    /// @brief Type of result provided by the nearest intersect operations. Index to primitive and its distance.
    using NearestIntersectResultT = std::pair<PrimitiveIdxT, ScalarT>;
    /// @brief Value used for invalid primitive indices.
    static constexpr auto InvalidPrimitiveIdx{ std::numeric_limits<PrimitiveIdxT>::max() };
    /// @brief Value used for invalid scalar values.
    static constexpr auto InvalidScalar{ std::numeric_limits<ScalarT>::max() };
    /// @brief Value used for invalid index-distance pair.
    static constexpr auto InvalidIdxDistance{ std::pair{ InvalidPrimitiveIdx, InvalidScalar } };

    /// @brief Create empty acceleration structure.
    TriangleIntersector();
    /// @brief Free all acceleration structures.
    ~TriangleIntersector();

    /// @brief Create acceleration structures for given set of triangles.
    TriangleIntersector(const PrimitiveSetT &primitives);

    /// @brief Create acceleration structures for given set of triangles.
    void create(const PrimitiveSetT &primitives);

    /// @brief Get list of primitives currently contained within this acceleration structure.
    const PrimitiveSetT &primitives() const;

    /**
     * @brief Get indices of intersecting triangles for given input triangle.
     *
     * @param primitive Queried primitive, used to detect intersections.
     * @param limit Limit of the maximum number of found intersections. Set to
     *  zero for unlimited.
     *
     * @return Returns list of primitive indices for which an intersection was found.
     *  The indices are taken from the initial list of primitives (i.e. Triangle::tIdx).
     */
    IntersectResultT queryIntersect(const PrimitiveT &primitive, std::size_t limit = 0u) const;

    /**
     * @brief Get triangles which contain given point.
     *
     * @param point Queried point against which will the triangles be tested.
     * @param limit Limit of the maximum number of found intersections. Set to
     *  zero for unlimited.
     *
     * @return Returns list of primitive indices for which an intersection was found.
     *  The indices are taken from the initial list of primitives (i.e. Triangle::tIdx).
     */
    IntersectResultT queryPoint(const Vector3D &point, std::size_t limit = 0u) const;

    /**
     * @brief Get triangles intersecting given ray.
     *
     * @param origin Origin of the ray.
     * @param direction Direction of the ray.
     * @param limit Limit of the maximum number of found intersections. Set to
     *  zero for unlimited.
     *
     * @return Returns list of primitive indices for which an intersection was found.
     *  The indices are taken from the initial list of primitives (i.e. Triangle::tIdx).
     */
    IntersectResultT queryRay(const Vector3D &origin, const Vector3D &direction, std::size_t limit = 0u) const;

    /**
     * @brief Get the nearest triangle index intersecting given ray.
     *
     * @param origin Origin of the ray.
     * @param direction Direction of the ray.
     *
     * @return Returns index of the nearest primitive or max if none were found and its distance.
     */
    NearestIntersectResultT queryRayNearest(const Vector3D &origin, const Vector3D &direction) const;

    /**
     * @brief Check occlusion between 2 points.
     *
     * @param first First point.
     * @param second Second point.
     *
     * @return Returns distance and triangle index of the occlusion from first to second point.
     *  Returns InvalidPrimitiveIdx, InvalidScalar if no such occlusion exists between these 2 points.
     */
     NearestIntersectResultT checkOcclusion(const Vector3D &first, const Vector3D &second) const;
private:
    /// Inner implementation of the intersector.
    std::unique_ptr<impl::TriangleIntersectorImpl> mImpl{ };
protected:
}; // class TriangleIntersector

} // namespace treeacc

#endif // TREEIO_ACCELERATION_ACCELERATION_H
