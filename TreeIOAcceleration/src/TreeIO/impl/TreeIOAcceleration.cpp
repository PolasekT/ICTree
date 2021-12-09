/**
 * @author Tomas Polasek
 * @date 3.23.2020
 * @version 1.0
 * @brief Acceleration structures and query helpers.
 */

#include "TreeIOAcceleration.h"

#include <Eigen/Eigen>
#include <unsupported/Eigen/BVH>

namespace treeacc
{

void PointCloudAcceleration::fromCloud(const PointCloud &cloud, std::size_t maxLeafSize)
{ PointCloud cloudCopy{ cloud }; fromCloud(std::move(cloudCopy), maxLeafSize); }

void PointCloudAcceleration::fromCloud(PointCloud &&cloud, std::size_t maxLeafSize)
{
    mData = std::make_shared<PointCloud>(std::move(cloud));

    mIndex = std::make_shared<KdTree>(PointCloud::DATA_DIMENSIONALITY, *mData,
        nanoflann::KDTreeSingleIndexAdaptorParams(maxLeafSize));
    mIndex->buildIndex();
}

void PointCloudAcceleration::fromVertices(const std::vector<Vertex3D> &vertices, std::size_t maxLeafSize)
{ std::vector<Vertex3D> vertexCopy{ vertices }; fromVertices(std::move(vertexCopy), maxLeafSize); }

void PointCloudAcceleration::fromVertices(std::vector<Vertex3D> &&vertices, std::size_t maxLeafSize)
{
    if (!mData)
    { mData = std::make_shared<PointCloud>( ); }

    mData->points() = std::move(vertices);

    mIndex = std::make_shared<KdTree>(PointCloud::DATA_DIMENSIONALITY, *mData,
        nanoflann::KDTreeSingleIndexAdaptorParams(maxLeafSize));
    mIndex->buildIndex();
}

void PointCloudAcceleration::fromVertices(const Vertex3D *vertices, std::size_t vertexCount,
    std::size_t maxLeafSize)
{
    std::vector<Vertex3D> vertexCopy{ vertices, vertices + vertexCount };
    fromVertices(std::move(vertexCopy), maxLeafSize);
}

PointCloudAcceleration::QueryResult PointCloudAcceleration::queryRange(const Vector3D &p,
    float range, std::size_t maxNeighbors, bool sort) const
{
    QueryResult result{ };
    nanoflann::SearchParams searchParams{ };
    searchParams.sorted = sort;
    const auto found{ mIndex->radiusSearch(&p[0], range * range, result.points, searchParams) };

    if (maxNeighbors && found > maxNeighbors)
    { result.points.resize(maxNeighbors); }

    result.sorted = sort;
    return result;
}

PointCloudAcceleration::QueryResult PointCloudAcceleration::queryKNN(const Vector3D &p,
    std::size_t maxNeighbors) const
{
    QueryResult result{ };

    std::vector<QueryResult::IndexT> indices{ };
    indices.resize(maxNeighbors);
    std::vector<QueryResult::DistanceT> distances{ };
    distances.resize(maxNeighbors);
    const auto found{ mIndex->knnSearch(&p[0], maxNeighbors, indices.data(), distances.data()) };

    result.points.resize(found);
    for (std::size_t iii = 0u; iii < found; ++iii)
    { result.points[iii] = std::make_pair(indices[iii], distances[iii]); }

    // TODO - Check it is actually sorted.
    result.sorted = true;
    return result;
}

PointCloudAcceleration::QueryResult PointCloudAcceleration::queryPosition(const Vector3D &p) const
{ return queryKNN(p, 1u); }

PointCloudAcceleration::LegacyKdTreePtr PointCloudAcceleration::toLegacyKdTree() const
{
    std::vector<kd::Vector3D> positions{ };
    const auto vertexCount{ mData->points().size() };
    positions.resize(vertexCount);
    for (std::size_t iii = 0u; iii < vertexCount; ++iii)
    { const auto &v{ mData->points()[iii] }; positions[iii] = { v.p.x, v.p.y, v.p.z }; }

    return std::make_shared<LegacyKdTree>(positions.data(), positions.size(), mIndex->index_params.leaf_max_size);
}

namespace impl
{

struct TriangleIntersectorImpl
{
    /// @brief Scalar used for distances and inner operations.
    using ScalarT = TriangleIntersector::ScalarT;
    /// @brief Dimensionality of the accelerated space.
    static constexpr auto Dimensionality{ 3u };

    /// @brier Wrapper around acceleration object which includes index of the original primitive.
    class AccelerationObject : public Eigen::AlignedBox<ScalarT, Dimensionality>
    {
    public:
        /// Epsilon used when constructing loose bounding box.
        static constexpr auto EPS{ std::numeric_limits<float>::epsilon() };

        /// @brief Empty acceleration object.
        inline AccelerationObject();

        /// @brief Create the acceleration object with index to original primitive.
        inline AccelerationObject(std::size_t primitiveIdx, const Vector3D &min, const Vector3D &max);

        /// @brief Get reference to the original primitive.
        inline TriangleIntersector::PrimitiveIdxT primitiveIdx() const;
    private:
        /// Index to the original primitive.
        std::size_t mPrimitiveIdx{ };
    protected:
    }; // class AccelerationObject

    /// @brief Object used in the acceleration structure nodes.
    using AccelerationObjectT = AccelerationObject;

    /// @brief Acceleration structure used for the operations.
    using SpatialAcceleratorT = Eigen::KdBVH<ScalarT, Dimensionality, TriangleIntersector::PrimitiveT>;

    /// @brief Type used for storing the list of currently accelerated primitives.
    using PrimitiveStoreT = std::vector<TriangleIntersector::PrimitiveT>;
    /// @brief Type used for storing the list of acceleration objects.
    using AccelerationStoreT = std::vector<AccelerationObjectT>;

    /// @brief Helper used for detecting volume intersections.
    template <typename TargetT, typename VolumeT, typename AccelerationT>
    struct VolumeIntersector;

    /// @brief Helper used for detecting object intersections.
    template <typename TargetT>
    struct ObjectIntersector;

    /// @brief Aggregator used for hit queries.
    template <typename TargetT>
    struct HitQueryAggregator;

    /// @brief Aggregator used for nearest hit queries.
    template <typename TargetT>
    struct NearestQueryAggregator;

    /// @brief Functor used for intersection queries.
    template <typename TargetT, typename AggregatorT>
    class IntersectQuery
    {
    public:
        /// @brief Shortcut for the volume intersector type.
        using VolumeIntersectorT = VolumeIntersector<TargetT,
            SpatialAcceleratorT::Volume, TriangleIntersectorImpl::AccelerationObjectT>;
        /// @brief Shortcut for the object intersector type.
        using ObjectIntersectorT = ObjectIntersector<TargetT>;

        /// @brief Initialize intersection query for given target. Set limit to non-zero to end after finding set amount of hits.
        template <typename... AggCArgTs>
        IntersectQuery(const TargetT &target, AggCArgTs... aggregatorArguments);

        /// @brief Returns true if the volume intersects current query.
        bool intersectVolume(const SpatialAcceleratorT::Volume &volume);
        /// @brief Returns true if the search should end immediately.
        bool intersectObject(const SpatialAcceleratorT::Object &object);

        /// @brief Get result aggregator of the intersection query.
        const AggregatorT &aggregator() const;
        /// @brief Get queried target.
        const TriangleIntersector::PrimitiveT &target() const;
    private:
        /// Target being queried.
        TargetT mTarget{ };
        /// Acceleration object of the target.
        TriangleIntersectorImpl::AccelerationObjectT mAcceleration{ };
        /// Aggregator of results.
        AggregatorT mAggregator{ };
    protected:
    }; // class IntersectQuery

    /// @brief Create acceleration object for given primitive, which is at provided index in the primitive array.
    static AccelerationObjectT createAccelerationObject(const TriangleIntersector::PrimitiveT &primitive,
        std::size_t idx = 0u);

    /// @brief Create acceleration object for given point.
    static AccelerationObjectT createAccelerationObject(const Vector3D &point);

    /// @brief Create acceleration object for given ray.
    static AccelerationObjectT createAccelerationObject(const Ray &ray);

    /// @brief Build the acceleration structure for given list of primitives.
    void buildAccelerator(const TriangleIntersector::PrimitiveSetT &primitives);

    /// @brief Perform intersection query with given primitive against the built acceleration structure.
    TriangleIntersector::IntersectResultT queryIntersect(const TriangleIntersector::PrimitiveT &primitive,
        std::size_t limit = 0u) const;

    /// @brief Perform intersection query for primitives which contain given point.
    TriangleIntersector::IntersectResultT queryPoint(const Vector3D &point, std::size_t limit = 0u) const;

    /// @brief Perform intersection query for primitives having intersection with given ray.
    TriangleIntersector::IntersectResultT queryRay(const Ray &ray, std::size_t limit = 0u) const;

    /// @brief Find nearest primitive intersection with given ray and return its index and distance.
    TriangleIntersector::NearestIntersectResultT queryRayNearest(const Ray &ray) const;

    /// Currently used acceleration structure.
    std::unique_ptr<SpatialAcceleratorT> accelerator{ };

    /// Primitives stored in the acceleration structure.
    PrimitiveStoreT primitives{ };
    /// Acceleration objects to the corresponding primitives.
    AccelerationStoreT accelerations{ };
}; // struct TriangleIntersectorImpl

inline TriangleIntersectorImpl::AccelerationObject::AccelerationObject()
{ /* Automatic */ }

inline TriangleIntersectorImpl::AccelerationObject::AccelerationObject(
    std::size_t primitiveIdx, const Vector3D &min, const Vector3D &max) :
    AlignedBox(
        VectorType{ min.x - EPS, min.y - EPS, min.z - EPS },
        VectorType{ max.x + EPS, max.y + EPS, max.z + EPS }),
    mPrimitiveIdx{ primitiveIdx }
{ }

inline TriangleIntersector::PrimitiveIdxT TriangleIntersectorImpl::AccelerationObject::primitiveIdx() const
{ return mPrimitiveIdx; }

template <typename TargetT, typename VolumeT, typename AccelerationT>
struct TriangleIntersectorImpl::VolumeIntersector
{
    /// @brief Check intersection of given target and its acceleration agains the volume.
    static bool intersectVolume(const TargetT &target, const VolumeT &volume, const AccelerationT &acceleration);
}; // struct VolumeIntersector

template <typename TargetT, typename VolumeT, typename AccelerationT>
bool TriangleIntersectorImpl::VolumeIntersector<TargetT, VolumeT, AccelerationT>::intersectVolume(
    const TargetT &target, const VolumeT &volume, const AccelerationT &acceleration)
{ return volume.intersects(acceleration); }

template <typename VolumeT, typename AccelerationT>
struct TriangleIntersectorImpl::VolumeIntersector<Ray, VolumeT, AccelerationT>
{
    /// @brief Check intersection of given target and its acceleration agains the volume.
    static bool intersectVolume(const Ray &target, const VolumeT &volume, const AccelerationT &acceleration);
}; // struct VolumeIntersector

template <typename VolumeT, typename AccelerationT>
bool TriangleIntersectorImpl::VolumeIntersector<Ray, VolumeT, AccelerationT>::intersectVolume(
    const Ray &target, const VolumeT &volume, const AccelerationT &acceleration)
{
    // Perform ray-aabb intersection test.
    const auto min{ volume.corner(SpatialAcceleratorT::Volume::BottomLeftFloor) };
    const auto max{ volume.corner(SpatialAcceleratorT::Volume::TopRightCeil) };
    BoundingBox3D bb({ min.x(), min.y(), min.z() }, { max.x(), max.y(), max.z() });

    return bb.intersects(target);
}

template <typename TargetT>
struct TriangleIntersectorImpl::ObjectIntersector
{
    /// @brief Perform intersection test and fill members.
    ObjectIntersector(const TargetT &target, const SpatialAcceleratorT::Object &object);

    /// Distance of the intersection point.
    ScalarT distance{ TriangleIntersector::InvalidScalar };
    /// Does the object intersect target?
    bool intersects{ false };
    /// Index of the object.
    TriangleIntersector::PrimitiveIdxT objectIdx{ TriangleIntersector::InvalidPrimitiveIdx };
}; // struct ObjectIntersector

template <>
TriangleIntersectorImpl::ObjectIntersector<TriangleIntersector::PrimitiveT>::ObjectIntersector(
    const TriangleIntersector::PrimitiveT &target, const SpatialAcceleratorT::Object &object)
{ distance = 0.0f; intersects = target.intersects(object); objectIdx = object.tIdx; }

template <>
TriangleIntersectorImpl::ObjectIntersector<Vector3D>::ObjectIntersector(
    const Vector3D &target, const SpatialAcceleratorT::Object &object)
{ distance = 0.0f; intersects = object.contains(target); objectIdx = object.tIdx; }

template <>
TriangleIntersectorImpl::ObjectIntersector<Ray>::ObjectIntersector(
    const Ray &target, const SpatialAcceleratorT::Object &object)
{ distance = object.intersection(target); intersects = distance < TriangleIntersector::InvalidScalar; objectIdx = object.tIdx; }

template <typename TargetT>
struct TriangleIntersectorImpl::HitQueryAggregator
{
    /// @brief Aggregate results up to given limit. Set to zero for unlimited results.
    HitQueryAggregator(std::size_t resultLimit = 0u) :
        limit{ resultLimit }
    { }

    /// @brief Check intersection with given object and return whether to quit.
    bool checkAddFull(const ObjectIntersector<TargetT> &intersector);

    /// @brief Limiti of the number of results.
    std::size_t limit{ 0u };
    /// @brief List of results.
    std::vector<TriangleIntersector::PrimitiveIdxT> results{ };
}; // struct HitQueryAggregator

template <typename TargetT>
bool TriangleIntersectorImpl::HitQueryAggregator<TargetT>::checkAddFull(
    const ObjectIntersector<TargetT> &intersector)
{
    if (intersector.intersects)
    { // Found intersecting triangle -> Add it to the list.
        results.push_back(intersector.objectIdx);
    }

    // Returns true only when we have found requested number of hits.
    return (limit > 0u) ? (results.size() >= limit) : false;
}

/// @brief Aggregator used for nearest hit queries.
template <typename TargetT>
struct TriangleIntersectorImpl::NearestQueryAggregator
{
    /// @brief Aggregate results up to given limit. Set to zero for unlimited results.
    NearestQueryAggregator()
    { }

    /// @brief Check intersection with given object and return whether to quit.
    bool checkAddFull(const ObjectIntersector<TargetT> &intersector);

    /// @brief Distance of the currently nearest result.
    ScalarT distance{ TriangleIntersector::InvalidScalar };
    /// @brief Index of the currently nearest result.
    TriangleIntersector::PrimitiveIdxT nearestIdx{ TriangleIntersector::InvalidPrimitiveIdx };
}; // struct NearestQueryAggregator

template <typename TargetT>
bool TriangleIntersectorImpl::NearestQueryAggregator<TargetT>::checkAddFull(
    const ObjectIntersector<TargetT> &intersector)
{
    const auto intersectionDistance{ intersector.distance };
    if (intersectionDistance < distance && intersectionDistance > ScalarT(0) && intersector.intersects)
    { // Found new nearest -> remember its index and distance.
        nearestIdx = intersector.objectIdx;
        distance = intersectionDistance;
    }

    // Never full, we need to check all possibilities.
    return false;
}

template <typename TargetT, typename AggregatorT>
template <typename... AggCArgTs>
TriangleIntersectorImpl::IntersectQuery<TargetT, AggregatorT>::IntersectQuery(
    const TargetT &target, AggCArgTs... aggregatorArguments) :
    mTarget{ target }, mAcceleration{ createAccelerationObject(target) },
    mAggregator(std::forward<AggCArgTs>(aggregatorArguments)...)
{ }

template <typename TargetT, typename AggregatorT>
bool TriangleIntersectorImpl::IntersectQuery<TargetT, AggregatorT>::intersectVolume(
    const SpatialAcceleratorT::Volume &volume)
{
    // Go down to the leafs and find all potential intersections.
    return VolumeIntersectorT::intersectVolume(mTarget, volume, mAcceleration);
}

template <typename TargetT, typename AggregatorT>
bool TriangleIntersectorImpl::IntersectQuery<TargetT, AggregatorT>::intersectObject(
    const SpatialAcceleratorT::Object &object)
{ return mAggregator.checkAddFull(ObjectIntersector(mTarget, object)); }

template <typename TargetT, typename AggregatorT>
const AggregatorT &TriangleIntersectorImpl::IntersectQuery<TargetT, AggregatorT>::aggregator() const
{ return mAggregator; }
template <typename TargetT, typename AggregatorT>
const TriangleIntersector::PrimitiveT &TriangleIntersectorImpl::IntersectQuery<TargetT, AggregatorT>::target() const
{ return mTarget; }

TriangleIntersectorImpl::AccelerationObjectT TriangleIntersectorImpl::createAccelerationObject(
    const TriangleIntersector::PrimitiveT &primitive, std::size_t idx)
{
    const auto aabb{ primitive.aabb() };
    return AccelerationObjectT(idx, aabb.min, aabb.max);
}

TriangleIntersectorImpl::AccelerationObjectT TriangleIntersectorImpl::createAccelerationObject(const Vector3D &point)
{ return AccelerationObjectT(0u, point, point); }

TriangleIntersectorImpl::AccelerationObjectT TriangleIntersectorImpl::createAccelerationObject(const Ray &ray)
{
    // Create binding box extending to infinity.
    const auto firstCorner{ ray.origin };
    const auto secondCorner{ ray.direction.sgn() * Vector3D::maxVector() };
    return AccelerationObjectT(0u,
        Vector3D::elementMin(firstCorner, secondCorner),
        Vector3D::elementMax(firstCorner, secondCorner));
}

void TriangleIntersectorImpl::buildAccelerator(const TriangleIntersector::PrimitiveSetT &p)
{
    const auto primitiveCount{ p.size() };

    // Switch to new primitives.
    primitives = p;
    accelerations.resize(primitiveCount);

    // Re-calculate acceleration objects.
    for (std::size_t iii = 0u; iii < accelerations.size(); ++iii)
    { accelerations[iii] = createAccelerationObject(primitives[iii], iii); }

    // Create the accelerator.
    accelerator = std::make_unique<SpatialAcceleratorT>(
        primitives.begin(), primitives.end(), accelerations.begin(), accelerations.end());
}

TriangleIntersector::IntersectResultT TriangleIntersectorImpl::queryIntersect(
    const TriangleIntersector::PrimitiveT &primitive, std::size_t limit) const
{
    if (!accelerator)
    { return { }; }

    using TargetT = TriangleIntersector::PrimitiveT;
    IntersectQuery<TargetT, HitQueryAggregator<TargetT>> query(primitive, limit);
    Eigen::BVIntersect(*accelerator, query);

    return query.aggregator().results;
}

TriangleIntersector::IntersectResultT TriangleIntersectorImpl::queryPoint(
    const Vector3D &point, std::size_t limit) const
{
    if (!accelerator)
    { return { }; }

    using TargetT = Vector3D;
    IntersectQuery<TargetT, HitQueryAggregator<TargetT>> query(point, limit);
    Eigen::BVIntersect(*accelerator, query);

    return query.aggregator().results;
}

TriangleIntersector::IntersectResultT TriangleIntersectorImpl::queryRay(
    const Ray &ray, std::size_t limit) const
{
    if (!accelerator)
    { return { }; }

    using TargetT = Ray;
    IntersectQuery<TargetT, HitQueryAggregator<TargetT>> query(ray, limit);
    Eigen::BVIntersect(*accelerator, query);

    return query.aggregator().results;
}

TriangleIntersector::NearestIntersectResultT TriangleIntersectorImpl::queryRayNearest( const Ray &ray) const
{
    if (!accelerator)
    { return { }; }

    using TargetT = Ray;
    IntersectQuery<TargetT, NearestQueryAggregator<TargetT>> query(ray);
    Eigen::BVIntersect(*accelerator, query);

    return { query.aggregator().nearestIdx, query.aggregator().distance };
}

} // namespace impl

TriangleIntersector::TriangleIntersector() :
    TriangleIntersector(PrimitiveSetT{ })
{ }

TriangleIntersector::~TriangleIntersector()
{ /* Automatic */ }

TriangleIntersector::TriangleIntersector(const PrimitiveSetT &primitives)
{ create(primitives); }

void TriangleIntersector::create(const PrimitiveSetT &primitives)
{
    mImpl = std::make_unique<impl::TriangleIntersectorImpl>();
    mImpl->buildAccelerator(primitives);
}

const TriangleIntersector::PrimitiveSetT &TriangleIntersector::primitives() const
{ return mImpl->primitives; }

TriangleIntersector::IntersectResultT TriangleIntersector::queryIntersect(
    const PrimitiveT &primitive, std::size_t limit) const
{ return mImpl ? mImpl->queryIntersect(primitive, limit) : IntersectResultT{ }; }

TriangleIntersector::IntersectResultT TriangleIntersector::queryPoint(
    const Vector3D &point, std::size_t limit) const
{ return mImpl ? mImpl->queryPoint(point, limit) : IntersectResultT{ }; }

TriangleIntersector::IntersectResultT TriangleIntersector::queryRay(
    const Vector3D &origin, const Vector3D &direction, std::size_t limit) const
{ return mImpl ? mImpl->queryRay(Ray{ origin, direction }, limit) : IntersectResultT{ }; }

TriangleIntersector::NearestIntersectResultT TriangleIntersector::queryRayNearest(
    const Vector3D &origin, const Vector3D &direction) const
{ return mImpl ? mImpl->queryRayNearest(Ray{ origin, direction }) : InvalidIdxDistance; }

TriangleIntersector::NearestIntersectResultT TriangleIntersector::checkOcclusion(
    const Vector3D &first, const Vector3D &second) const
{
    const auto rayOrigin{ first };
    const auto rayDirection{ second - first };
    const auto rayDirectionNorm{ rayDirection.normalized() };
    const auto distance{ rayDirection.length() };
    const auto nearestQuery{ queryRayNearest(rayOrigin, rayDirectionNorm) };

    return (nearestQuery.second <= (distance + std::numeric_limits<ScalarT>::epsilon())) ? nearestQuery : InvalidIdxDistance;
}

} // namespace treeacc

