/**
 * @author Tomas Polasek
 * @date 2.20.2020
 * @version 1.0
 * @brief Wrapper around a polygonal model.
 */

#include "TreeIOPolygonalModel.h"

namespace treeacc
{

namespace impl
{

/// @brief Internal implementation of the PolygonalModel class.
class PolygonalModelImpl
{
public:
    /// @brief Import OBJ file from given file path.
    bool importObj(const std::string &filename, bool rebuildNormals);

    /// @brief Import FBX file from given file path.
    bool importFbx(const std::string &filename, bool rebuildNormals);

    /// @brief Normalize the model and scale it to given size. Optionally center the model as well.
    void normalize(float scaleTo, bool center);

    /// Path to the loaded model file.
    std::string mModelPath{ };
    /// Is model currently loaded?
    bool mLoaded{ false };

    /// Coordinates of the center of the currently loaded model.
    Vector3D mCenterOfMass{ };
    /// Width (x) of the currently loaded model.
    float mWidth{ };
    /// Height (y) of the currently loaded model.
    float mHeight{ };
    /// Length / depth (z) of the currently loaded model.
    float mDepth{ };
    /// Radius of sphere enclosing the whole model.
    float mRadius{ };

    /// Buffer containing index data.
    std::vector<PolygonalModel::Index> mIndexBuffer{ };
    /// Buffer containing vertex data.
    std::vector<PolygonalModel::Vertex> mVertexBuffer{ };
private:
    /// @brief Load data from filled importer.
    bool loadFromImporter(treeio::ModelImporterBase &importer);

    /// @brief Calculate statistics for currently loaded model.
    void calculateStatistics();
protected:
}; // class PolygonalModelImpl

} // namespace impl

} // namespace treeacc

namespace treeacc
{

namespace impl
{

bool PolygonalModelImpl::importObj(const std::string &filename, bool rebuildNormals)
{
    treeio::ObjImporter importer;
    importer.import(filename);
    const auto loadResult{ loadFromImporter(importer) };

    // TODO - Rebuild normals?
    TREE_UNUSED(rebuildNormals);

    mModelPath = filename;
    mLoaded = true;
    calculateStatistics();

    return loadResult;
}

bool PolygonalModelImpl::importFbx(const std::string &filename, bool rebuildNormals)
{
    treeio::FbxImporter importer;
    importer.import(filename);
    const auto loadResult{ loadFromImporter(importer) };

    // TODO - Rebuild normals?
    TREE_UNUSED(rebuildNormals);

    mModelPath = filename;
    mLoaded = true;
    calculateStatistics();

    return loadResult;
}

void PolygonalModelImpl::normalize(float scaleTo, bool center)
{
    // TODO - calculateStatistics() ?

    const auto scale{ scaleTo / mRadius };
    const auto translate{ center ? -mCenterOfMass : Vector3D{ }};

    for (auto &vertex : mVertexBuffer)
    {
        vertex.position[0u] = (vertex.position[0u] + translate.x) * scale;
        vertex.position[1u] = (vertex.position[1u] + translate.y) * scale;
        vertex.position[2u] = (vertex.position[2u] + translate.z) * scale;
    }
}

bool PolygonalModelImpl::loadFromImporter(treeio::ModelImporterBase &importer)
{
    const auto vertexCount{ importer.vertexCount() };
    const auto &indices{ importer.indices() };
    const auto indexCount{ importer.indexCount() };

    // Copy vertex data:
    mVertexBuffer.resize(vertexCount);
    for (std::size_t iii = 0u; iii < vertexCount; ++iii)
    { mVertexBuffer[iii] = importer.getVertex(iii); }

    // Copy index data:
    mIndexBuffer.resize(indexCount);
    for (std::size_t iii = 0u; iii < indexCount; ++iii)
    { mIndexBuffer[iii] = importer.indices()[iii]; }

    return true;
}

void PolygonalModelImpl::calculateStatistics()
{
    auto xMin{ std::numeric_limits<float>::max() };
    auto yMin{ std::numeric_limits<float>::max() };
    auto zMin{ std::numeric_limits<float>::max() };

    auto xMax{ std::numeric_limits<float>::min() };
    auto yMax{ std::numeric_limits<float>::min() };
    auto zMax{ std::numeric_limits<float>::min() };

    for (const auto &vertex : mVertexBuffer)
    {
        const auto position{ vertex.position };

        xMin = std::min<float>(xMin, position[0u]);
        yMin = std::min<float>(yMin, position[1u]);
        zMin = std::min<float>(zMin, position[2u]);

        xMax = std::max<float>(xMax, position[0u]);
        yMax = std::max<float>(yMax, position[1u]);
        zMax = std::max<float>(zMax, position[2u]);
    }

    mCenterOfMass = { (xMin + xMax) / 2.0f, (yMin + yMax) / 2.0f, (zMin + zMax) / 2.0f, };
    mWidth = (xMax - xMin);
    mHeight = (yMax - yMin);
    mDepth = (zMax - zMin);
    mRadius = std::max<float>(std::max<float>(mWidth, mHeight), mDepth);
}

}

PolygonalModel::PolygonalModel()
{ /* Automatic */ }

PolygonalModel::Ptr PolygonalModel::createNormalized(
    const PolygonalModel &other, float scaleTo, bool center)
{
    // Create a copy, including the same model.
    auto normalizedModel{ std::make_shared<PolygonalModel>() };
    *normalizedModel = other;

    // Create the normalized model.
    normalizedModel->mModel = std::make_shared<impl::PolygonalModelImpl>(*other.mModel);
    normalizedModel->normalize(scaleTo, center);

    return normalizedModel;
}

PolygonalModel::~PolygonalModel()
{ /* Automatic */ }

PolygonalModel::PolygonalModel(const PolygonalModel &other)
{ *this = other; }

PolygonalModel &PolygonalModel::operator=(const PolygonalModel &other)
{
    // Deep copy the model.
    mModel = std::make_shared<impl::PolygonalModelImpl>(*other.mModel);

    return *this;
}

PolygonalModel::PolygonalModel(const std::string &filename, bool rebuildNormals)
{
    if (!importFromFile(filename, rebuildNormals))
    { throw LoadFailedException("Unable to load provided model!"); }
}

bool PolygonalModel::importFromFile(const std::string &filename, bool rebuildNormals)
{
    if (treeutil::equalCaseInsensitive(treeutil::fileExtension(filename), ".obj"))
    { return importObj(filename, rebuildNormals); }
    else if (treeutil::equalCaseInsensitive(treeutil::fileExtension(filename), ".fbx"))
    { return importFbx(filename, rebuildNormals); }

    return false;
}

void PolygonalModel::normalize(float scaleTo, bool center)
{ mModel->normalize(scaleTo, center); }

Vector3D PolygonalModel::centerOfMass() const
{ return mModel->mCenterOfMass; }

float PolygonalModel::width() const
{ return mModel->mWidth; }

float PolygonalModel::height() const
{ return mModel->mHeight; }

float PolygonalModel::length() const
{ return mModel->mDepth; }

float PolygonalModel::radius() const
{ return mModel->mRadius; }

const PolygonalModel::Index *PolygonalModel::indexBuffer() const
{ return mModel->mIndexBuffer.data(); }

std::size_t PolygonalModel::indexCount() const
{ return mModel->mIndexBuffer.size(); }

const PolygonalModel::Vertex &PolygonalModel::getVertex(std::size_t idx) const
{ return mModel->mVertexBuffer[idx]; }

const PolygonalModel::Vertex *PolygonalModel::vertexBuffer() const
{ return mModel->mVertexBuffer.data(); }

std::size_t PolygonalModel::vertexCount() const
{ return mModel->mVertexBuffer.size(); }

std::size_t PolygonalModel::triangleCount() const
{ assert(indexCount() % 3u == 0u); return indexCount() / 3u; }

const std::string &PolygonalModel::loadedPath() const
{ return mModel->mModelPath; }

bool PolygonalModel::isLoaded() const
{ return mModel && !loadedPath().empty() && mModel->mLoaded; }

bool PolygonalModel::importObj(const std::string &filename, bool rebuildNormals)
{
    mModel = std::make_shared<impl::PolygonalModelImpl>();

    if (rebuildNormals)
    { treeutil::Info << "Importing OBJ model and rebuilding normals..." << std::endl; }
    else
    { treeutil::Info << "Importing OBJ model..." << std::endl; }

    return mModel->importObj(filename, rebuildNormals);
}

bool PolygonalModel::importFbx(const std::string &filename, bool rebuildNormals)
{
    mModel = std::make_shared<impl::PolygonalModelImpl>();

    if (rebuildNormals)
    { treeutil::Info << "Importing FBX model and rebuilding normals..." << std::endl; }
    else
    { treeutil::Info << "Importing FBX model..." << std::endl; }

    return mModel->importFbx(filename, rebuildNormals);
}

} // namespace treeacc
