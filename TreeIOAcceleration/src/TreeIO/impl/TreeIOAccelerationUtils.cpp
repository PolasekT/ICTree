/**
 * @author Tomas Polasek
 * @date 11.20.2019
 * @version 1.0
 * @brief Utilities and statistics for the treeio::Tree class.
 */

#include "TreeIOAccelerationUtils.h"

#include <Eigen/Eigen>

#include <intersect/intersect.h>

namespace treeacc
{

Line Line::fromDirection(const Vector3D &origin, const Vector3D &direction)
{ return Line{ origin, origin + direction }; }

BoundingBox3D::BoundingBox3D() :
    BoundingBox3D( Vector3D{ } )
{ }
BoundingBox3D::BoundingBox3D(const Vector3D &pos) :
    BoundingBox3D(pos, pos)
{ }
BoundingBox3D::BoundingBox3D(const Vector3D &lv, const Vector3D &hv)
{ min = lv; max = hv; }

bool BoundingBox3D::intersects(const Ray &ray) const
{ return intersection(ray) < std::numeric_limits<float>::max(); }

float BoundingBox3D::intersection(const Ray &ray) const
{
    const auto tMinCorner{ (min - ray.origin) / ray.direction };
    const auto tMaxCorner{ (max - ray.origin) / ray.direction };
    const auto tMin{ Vector3D::elementMin(tMinCorner, tMaxCorner).max() };
    const auto tMax{ Vector3D::elementMax(tMinCorner, tMaxCorner).min() };

    return (tMax < 0.0f || tMin > tMax) ? std::numeric_limits<float>::max() : tMin;
}

float calculateTriangleArea(const Vector3D &p1, const Vector3D &p2, const Vector3D &p3)
{
    const auto a{ p2 - p1 };
    const auto b{ p3 - p1 };

    const auto prod{ Vector3D::crossProduct(a, b) };

    return prod.length() / 2.0f;
}

float calculateTrianglePerimeter(const Vector3D &p1, const Vector3D &p2, const Vector3D &p3)
{
    const auto a{ p2 - p1 };
    const auto b{ p3 - p1 };
    const auto c{ p2 - p3 };

    return a.length() + b.length() + c.length();
}

Vector3D calculateTriangleMedianPoint(const Vector3D &p1, const Vector3D &p2, const Vector3D &p3)
{ return (p1 + p2 + p3) / 3.0f; }

float calculateTriangleMedian(const Vector3D &p1, const Vector3D &p2, const Vector3D &p3)
{
    const auto p1p2{ p1 + p2 / 2.0f };
    return (p3 - p1p2).length();
}

float calculateTriangleLongestMedian(const Vector3D &p1, const Vector3D &p2, const Vector3D &p3)
{
    const auto m1{ calculateTriangleMedian(p1, p2, p3) };
    const auto m2{ calculateTriangleMedian(p2, p3, p1) };
    const auto m3{ calculateTriangleMedian(p1, p3, p2) };

    return std::max<float>({ m1, m2, m3 });
}

Vector3D sampleTriangle(const Vector3D &a1, const Vector3D &a2, const Vector3D &a3, const Vector3D &w)
{
    // TODO - Vector3D scalar multiplication + addition seems broken?
    return {
        w.x * a1.x + w.y * a2.x + w.z * a3.x,
        w.x * a1.y + w.y * a2.y + w.z * a3.y,
        w.x * a1.z + w.y * a2.z + w.z * a3.z
    };
}

/**
 * Sample barycentric U coordinate using provided random value.
 * Source: Portsmouth, Jamie. "Efficient barycentric point sampling on meshes."
 *
 * @param phiU Normalized relative weight for U-axis.
 * @param phiV Normalized relative weight for V-axis.
 * @param rand Random value used for sampling U coordinate.
 * @param limit Quality limit for inner loop.
 *
 * @return Returns sampled U coordinate.
 */
float sampleBarycentricU(float phiU, float phiV, float rand, float limit = 5.0e-3)
{
    static constexpr auto MAX_ITERATIONS{ 20u };
    static constexpr auto EPSILON{ std::numeric_limits<float>::epsilon() };
    const auto l{ (2.0f * phiU - phiV) / 3.0f };
    auto u{ 0.5f };
    for (std::size_t iter = 0u; iter < MAX_ITERATIONS; ++iter)
    {
        const auto u1{ 1.0f - u };
        const auto P{ u * (2.0f - u) - l * u * u1 * u1 - rand};
        const auto Pd{ std::max<float>(u1 * (2.0f + l * (3.0f * u - 1.0f)), EPSILON)};
        const auto du{ std::max<float>(std::min<float>(P / Pd, 0.25f), -0.25f) };

        u -= du;
        u = std::max<float>(std::min<float>(u, 1.0f - EPSILON), EPSILON);
        if (std::fabs(du) < limit)
        { break; }
    }

    return u;
}

/**
 * Sample barycentric V coordinate using provided random value.
 * Source: Portsmouth, Jamie. "Efficient barycentric point sampling on meshes."
 *
 * @param u Previously sampled u coordinate.
 * @param phiU Normalized relative weight for U-axis.
 * @param phiV Normalized relative weight for V-axis.
 * @param rand Random value used for sampling V coordinate.
 *
 * @return Returns sampled V coordinate.
 */
float sampleBarycentricV(float u, float phiU, float phiV, float rand)
{
    static constexpr auto EPSILON{ std::numeric_limits<float>::epsilon() };

    if (std::fabs(phiV) < EPSILON)
    { return (1.0f - u) * rand; }

    const auto tau{ 1.0f / 3.0f - (1.0f + (u - 1.0f / 3.0f) * phiU) / phiV };
    const auto tmp{ tau + u - 1.0f };
    const auto q{ std::sqrt(tau * tau * (1.0f - rand) + tmp * tmp * rand) };

    return tau <= 0.5f * (1.0f - u) ? tau + q : tau - q;
}

Vector3D triangleWeights(const Vector3D &p1, const Vector3D &p2, const Vector3D &p3, const Vector3D &phi)
{
    /*
     * Source: Portsmouth, Jamie. "Efficient barycentric point sampling on meshes."
     */

    // Recover densities at each of the triangle vertices.
    const auto phi1{ phi.x };
    const auto phi2{ phi.y };
    const auto phi3{ phi.z };
    const auto averagePhi{ (phi1 + phi2 + phi3) / 3.0f};

    // Transform absolute distances to normalized relative weights.
    const auto relativePhi1{ (phi1 - phi3) / averagePhi };
    const auto relativePhi2{ (phi2 - phi3) / averagePhi };

    // Get independent random samples from uniform distribution.
    const auto randU{ treeutil::uniformZeroToOne<float>() };
    const auto randV{ treeutil::uniformZeroToOne<float>() };

    // Sample barycentric coordinates.
    const auto u{ sampleBarycentricU(relativePhi1, relativePhi2, randU) };
    const auto v{ sampleBarycentricV(u, relativePhi1, relativePhi2, randV) };
    const auto w{ 1.0f - u - v };

    return { u, v, w };
}

Vector3D uniformTriangleWeights(const Vector3D &p1, const Vector3D &p2, const Vector3D &p3)
{ return triangleWeights(p1, p2, p3, { 1.0f, 1.0f, 1.0f }); }

Vector3D medianTriangleWeights(const Vector3D &p1, const Vector3D &p2, const Vector3D &p3)
{
    // Median - center of gravity of input triangle.
    const auto median{ calculateTriangleMedianPoint(p1, p2, p3) };

    // Calculate distances median <-> point.
    const auto phi1{ (p1 - median).length() };
    const auto phi2{ (p2 - median).length() };
    const auto phi3{ (p3 - median).length() };

    return triangleWeights(p1, p2, p3, { phi1, phi2, phi3 });
}

Vertex3D uniformSampleTriangle(const Vertex3D &v1, const Vertex3D &v2, const Vertex3D &v3)
{
    const auto barycentricCoordinates{ uniformTriangleWeights(v1.p, v2.p, v3.p) };

    const auto p{ sampleTriangle(v1.p, v2.p, v3.p, barycentricCoordinates) };
    auto n{ sampleTriangle(v1.n, v2.n, v3.n, barycentricCoordinates) };
    n.normalize();

    return { p, n };
}

Vertex3D uniformSampleTriangle(const Vertex3D &v1, const Vertex3D &v2, const Vertex3D &v3, const Vector3D &weights)
{
    const auto barycentricCoordinates{ triangleWeights(v1.p, v2.p, v3.p, weights) };

    const auto p{ sampleTriangle(v1.p, v2.p, v3.p, barycentricCoordinates) };
    auto n{ sampleTriangle(v1.n, v2.n, v3.n, barycentricCoordinates) };
    n.normalize();

    return { p, n };
}

Vector3D middleSampleTriangle(const Vector3D &a1, const Vector3D &a2, const Vector3D &a3)
{ return sampleTriangle(a1, a2, a3, Vector3D{ 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f }); }

Vector3D triangleNormal(const Vector3D &a1, const Vector3D &a2, const Vector3D &a3)
{
    const auto f{ a2 - a1 }; const auto s{ a3 - a1 };
    return Vector3D{
        f.y * s.z - f.z * s.y,
        f.z * s.x - f.x * s.z,
        f.x * s.y - f.y * s.x
    };
}

Triangle::Triangle(std::size_t idx,
    const treeio::ModelImporterBase::Vertex *vb,
    const treeio::ModelImporterBase::IndexElementT *ib):
    tIdx{ idx }
{
    const auto triangleBaseIndex{ idx * 3u };

    i1 = ib[triangleBaseIndex + 0u];
    i2 = ib[triangleBaseIndex + 1u];
    i3 = ib[triangleBaseIndex + 2u];

    v1 = modelVertexToCloudPoint(vb[i1]);
    v2 = modelVertexToCloudPoint(vb[i2]);
    v3 = modelVertexToCloudPoint(vb[i3]);
    lv = Vector3D{
        std::min<float>(std::min<float>(v1.x, v2.x), v3.x),
        std::min<float>(std::min<float>(v1.y, v2.y), v3.y),
        std::min<float>(std::min<float>(v1.z, v2.z), v3.z),
    };
    hv = Vector3D{
        std::max<float>(std::max<float>(v1.x, v2.x), v3.x),
        std::max<float>(std::max<float>(v1.y, v2.y), v3.y),
        std::max<float>(std::max<float>(v1.z, v2.z), v3.z),
    };
    c = middleSampleTriangle(v1, v2, v3);

    n1 = modelNormalToCloudNormal(vb[i1]);
    n2 = modelNormalToCloudNormal(vb[i2]);
    n3 = modelNormalToCloudNormal(vb[i3]);
}

float Triangle::centroidDistance(const Triangle &other) const
{ return Vector3D::distance(c, other.c); }

float Triangle::centroidSquaredDistance(const Triangle &other) const
{ return Vector3D::squaredDistance(c, other.c); }

float Triangle::centroidSquaredDistance(const Vector3D &point) const
{ return Vector3D::squaredDistance(c, point); }

bool Triangle::intersects(const Triangle &other) const
{
    // Vertices of this triangle:
    double v11[3]{ v1.x, v1.y, v1.z };
    double v12[3]{ v2.x, v2.y, v2.z };
    double v13[3]{ v3.x, v3.y, v3.z };

    // Vertices of the other triangle:
    double v21[3]{ other.v1.x, other.v1.y, other.v1.z };
    double v22[3]{ other.v2.x, other.v2.y, other.v2.z };
    double v23[3]{ other.v3.x, other.v3.y, other.v3.z };

    const auto result{ tri_tri_overlap_test_3d(
        v11, v12, v13,
        v21, v22, v23
    ) };

    return result != 0;
}

bool Triangle::intersects(const Ray &ray) const
{ return intersection(ray) < std::numeric_limits<float>::max(); }

float Triangle::intersection(const Ray &ray) const
{
    // Constants:
    static constexpr auto EPSILON{ std::numeric_limits<float>::epsilon() };

    // Moller Trumbore algorithm:
    const auto edge1{ v2 - v1 };
    const auto edge2{ v3 - v1 };
    const auto cross{ Vector3D::crossProduct(ray.direction, edge2) };
    const auto det{ Vector3D::dotProduct(edge1, cross) };

    // Ray is parallel to the triangle.
    /*
    std::cout << "v1: " << v1 << " v2: " << v2 << " v3: " << v2 << std::endl;
    std::cout << "edg1: " << edge1 << std::endl;
    std::cout << "edg2: " << edge2 << std::endl;
    std::cout << "cross: " << cross << std::endl;
    std::cout << "det: " << det << std::endl;
     */
    if (std::abs<float>(det) < EPSILON)
    { return std::numeric_limits<float>::max(); }

    const auto invDet{ 1.0f / det };
    const auto uVec{ ray.origin - v1 };
    const auto vVec{ Vector3D::crossProduct(uVec, edge1) };

    // Check intersection point is within triangle.
    const auto u{ invDet * Vector3D::dotProduct(uVec, cross) };
    //std::cout << "u: " << u << std::endl;
    if (u < -EPSILON || u > 1.0f + EPSILON)
    { return std::numeric_limits<float>::max(); }
    const auto v{ invDet * Vector3D::dotProduct(ray.direction, vVec) };
    //std::cout << "v " << v << std::endl;
    if (v < -EPSILON || u + v > 1.0f + EPSILON)
    { return std::numeric_limits<float>::max(); }

    // Calculate intersection time.
    const auto t{ invDet * Vector3D::dotProduct(edge2, vVec) };
    return t;
}

bool Triangle::contains(const Vector3D &point, float epsilon) const
{
#if 0
    /*
     * Using volume of a tetrahedron v1, v2, v3, point:
     *
     * 1) Calculate volume.
     * 2) Point is inside <-> volume < epsilon
     */
    return std::abs<float>(
        Vector3D::dotProduct(
            v1 - point,
            Vector3D::crossProduct(v2 - point, v3 - point)
        )
    ) < epsilon;

#else

    /*
     * Using barycentric coordinates:
     *
     * 1) Project point onto the triangle plane and get its distance -> distance.
     * 2) Calculate barycentrics on the 2D plane -> alpha, beta.
     * 3) Point is inside <-> distance < epsilon && alpha > -epsilon && beta > -epsilon && alpha + beta <= 1 + epsilon
     */

    const auto pn{ centralPlaneNormal() };
    const auto distance{ Vector3D::dotProduct(pn, point) };
    const auto projected{ pn * distance };

    const auto s2{ v2 - v1 };
    const auto s3{ v3 - v1 };
    const auto sp{ point - v1 };

    const auto d2p{ Vector3D::dotProduct(s2, sp) };
    const auto d3p{ Vector3D::dotProduct(s2, sp) };
    const auto dsh{ Vector3D::dotProduct(s2, s3) };
    const auto s2l{ s2.length() };
    const auto s3l{ s3.length() };
    // Normalize for alpha and beta to be 1.0 when reaching v2 and v3 respectively.
    const auto normalizer{ 1.0f / (d2p * d3p - dsh * dsh) };
    // Alpha represents distance along s2.
    const auto alpha{ (d2p * s3l - dsh * d3p) * normalizer };
    const auto beta{ (d3p * s2l - dsh * d2p) * normalizer };

    return alpha > -epsilon && beta > -epsilon && alpha + beta <= 1.0f + epsilon;
#endif
}

Vector3D Triangle::barycentrics(const Vector3D &point) const
{
    // Source: https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Conversion_between_barycentric_and_Cartesian_coordinates
    const auto &p{ point };
    const auto b1{
        ((v2.y - v3.y) * (p.x - v3.x) + (v3.x - v2.x) * (p.y - v3.y)) /
        ((v2.y - v3.y) * (v1.x - v3.x) + (v3.x - v2.x) * (v1.y - v3.y))
    };
    const auto b2{
        ((v3.y - v1.y) * (p.x - v3.x) + (v1.x - v3.x) * (p.y - v3.y)) /
        ((v2.y - v3.y) * (v1.x - v3.x) + (v3.x - v2.x) * (v1.y - v3.y))
    };
    const auto b3{ 1.0f - b1 - b2 };

    return { b1, b2, b3 };
}

BoundingBox3D Triangle::aabb() const
{ return BoundingBox3D(lv, hv); }

float Triangle::area() const
{ return calculateTriangleArea(v1, v2, v3); }

float Triangle::perimeter() const
{ return calculateTrianglePerimeter(v1, v2, v3); }

float Triangle::longestAltitude() const
{ return calculateTriangleLongestMedian(v1, v2, v3); }

Vector3D Triangle::centralPlaneNormal() const
{
    /*
     * Use cross product of two sides:
     *    3
     *  /   \
     * 1 --- 2
     * Counter clock-wise -> front-face.
     * normal = cross(1->2, 1->3)
     */
    const auto c1{ v2 - v1 };
    const auto c2{ v3 - v1 };
    auto planeNormal{ Vector3D::crossProduct(c1, c2) };
    planeNormal.normalize();

    return planeNormal;
}

void Triangle::providePlaneNormals(const Vector3D &pNormal1, const Vector3D &pNormal2, const Vector3D &pNormal3)
{
    providedPlaneNormals = true;

    pn1 = pNormal1;
    pn2 = pNormal2;
    pn3 = pNormal3;

    const auto l1{ Line::fromDirection(v1, pn1) };
    const auto l2{ Line::fromDirection(v2, pn2) };
    const auto l3{ Line::fromDirection(v3, pn3) };

    const auto i12{ lsIntersect({ l1, l2 }) };
    const auto i13{ lsIntersect({ l1, l3 }) };
    const auto i23{ lsIntersect({ l2, l3 }) };

    //d12 = Vector3D::distance(v1 + (v2 - v1) * Vector3D::dotProduct((v2 - v1), (i12 - v1)), i12);
    //d13 = Vector3D::distance(v1 + (v3 - v1) * Vector3D::dotProduct((v3 - v1), (i13 - v1)), i13);
    //d23 = Vector3D::distance(v2 + (v3 - v2) * Vector3D::dotProduct((v3 - v2), (i23 - v2)), i13);

    pr12 = (Vector3D::distance(i12, v1) + Vector3D::distance(i12, v2)) / 2.0f;
    pr13 = (Vector3D::distance(i13, v1) + Vector3D::distance(i13, v3)) / 2.0f;
    pr23 = (Vector3D::distance(i23, v2) + Vector3D::distance(i23, v3)) / 2.0f;
}

Vector3D Triangle::smoothPoint(const Vector3D &point) const
{
    if (!providedPlaneNormals)
    { return point; }

    const auto bary{ barycentrics(point) };
    const auto shiftedBary{ bary - Vector3D{ 0.5f, 0.5f, 0.5f } };
    const auto radiusWeights{ Vector3D{ 1.0f, 1.0f, 1.0f } - 4.0f * (shiftedBary * shiftedBary) };

    return point + centralPlaneNormal() * radiusWeights * Vector3D{ pr12, pr13, pr23 };
}

float Triangle::curvature() const
{
    // TODO - Better curvature metric?
    return (Vector3D::dotProduct(pn1, pn2) + Vector3D::dotProduct(pn2, pn3) + Vector3D::dotProduct(pn3, pn1)) / 3.0f;
}

Vector3D Triangle::curvatureWeights() const
{
    const auto cpn{ centralPlaneNormal() };

    // Calculate weights as offset from central plane normal.
    const auto w1{ std::abs(Vector3D::dotProduct(cpn, pn1)) };
    const auto w2{ std::abs(Vector3D::dotProduct(cpn, pn2)) };
    const auto w3{ std::abs(Vector3D::dotProduct(cpn, pn3)) };

    // Fallback to uniform if we don't have valid values.
    const auto weightSum{ w1 + w2 + w3};
    if (weightSum <= std::numeric_limits<float>::epsilon() || !std::isfinite(weightSum))
    { return { 1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f }; }

    // Normalize weights to be in <0.0f, 1.0f> and sum to 1.0f .
    const auto normalizer{ 1.0f / (w1 + w2 + w3) };
    return { w1 * normalizer, w2 * normalizer, w3 * normalizer };
}

float Triangle::longestSide() const
{
    return std::max<float>(
        std::max<float>(
            v1.length(),
            v2.length()),
        v3.length()
    );
}

Vector3D lsIntersect(const std::vector<Line> &lines)
{
    // Initialize accumulation variables:

    // Coefficients of the left sides.
    Eigen::Matrix3f A{ Eigen::Matrix3f::Zero() };
    // Coefficients of the right sides.
    Eigen::Vector3f b{ Eigen::Vector3f::Zero() };
    // Constants.
    float c{ 0.0f };

    //std::cout << "Starting lsIntersect..." << std::endl;

    // Populate the coefficient matrices:
    for (const auto &line : lines)
    { // Process each line in turn.
        // Recover points on the line.
        const Eigen::Vector3f v{ line.v.x, line.v.y, line.v.z };
        const Eigen::Vector3f w{ line.w.x, line.w.y, line.w.z };

        //std::cout << "Adding line: " << v.transpose() << " -> " << w.transpose() << std::endl;

        // Calculate line direction.
        const Eigen::Vector3f u{ (v - w).normalized() };
        const Eigen::Vector3f ut{ u.transpose() };

        // Calculate projection to hyper-plane orthogonal to this line.
        Eigen::Matrix3f kroneckerProduct{ };
        kroneckerProduct <<
                         u(0) * ut(0), u(0) * ut(1), u(0) * ut(2),
            u(1) * ut(0), u(1) * ut(1), u(1) * ut(2),
            u(2) * ut(0), u(2) * ut(1), u(2) * ut(2);
        const Eigen::Matrix3f projection{ Eigen::Matrix3f::Identity() - kroneckerProduct };

        // Calculate distance of our line to the projection plane.
        const Eigen::Vector3f p{ projection * w };

        // Populate coefficients of the least-squares system of equations.
        A += projection;
        b += p;
        c += p(0) * p(0) + p(1) * p(1) + p(2) * p(2);
    }

    // Solve least-squares problem.
    //const Eigen::Vector3f result{ A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b) };
    const Eigen::Vector3f result{ A.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(b) };

    //std::cout << "LS intersection result: " << result.transpose() << std::endl;

    return { result(0), result(1), result(2) };
}

Vector3D lsIntersect(std::initializer_list<Line> lines)
{ return lsIntersect({ lines.begin(), lines.end() }); }

} // namespace treeacc
