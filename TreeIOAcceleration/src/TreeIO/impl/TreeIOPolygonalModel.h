/**
 * @author Tomas Polasek
 * @date 2.20.2020
 * @version 1.0
 * @brief Wrapper around a polygonal model.
 */

#ifndef TREEIO_ACCELERATION_POLYGONAL_MODEL_H
#define TREEIO_ACCELERATION_POLYGONAL_MODEL_H

#include <exception>
#include <memory>

#include <TreeIO/TreeIO.h>

#include "TreeIOAccelerationUtils.h"

namespace treeacc
{

namespace impl
{

/// @brief Internal implementation of the PolygonalModel class.
class PolygonalModelImpl;

} // namespace impl

/// @brief Wrapper around a polygonal model.
class PolygonalModel
{
public:
    /// @brief Vertex representation used by this model.
    using Vertex = treeio::ModelImporterBase::Vertex;

    /// @brief Type used to represent a single index to the vertex data.
    using Index = treeio::ModelImporterBase::IndexElementT;

    /// Shortcut for pointer to a polygonal model.
    using Ptr = std::shared_ptr<PolygonalModel>;

    /// @brief Exception thrown when model loading failed.
    struct LoadFailedException : public std::runtime_error
    {
        LoadFailedException(const char *msg) :
            std::runtime_error( msg ) { }
    }; // struct LoadFailedException

    /// @brief Initialize unloaded polygonal model.
    PolygonalModel();

    /// @brief Return normalized version of this model. Optionally also center it to origin.
    static PolygonalModel::Ptr createNormalized(const PolygonalModel &other, float scaleTo = 1.0f, bool center = true);

    /// @brief Free all model resources.
    ~PolygonalModel();

    // Copy and move operators:
    PolygonalModel(const PolygonalModel &other);
    PolygonalModel &operator=(const PolygonalModel &other);
    PolygonalModel(PolygonalModel &&other) = default;
    PolygonalModel &operator=(PolygonalModel &&other) = default;

    /**
     * @brief Load model from given file and optionally rebuild its normals.
     *
     * @param filename Name of the target OBJ formatted file.
     * @param rebuildNormals Set to true to re-calculate normals based on triangles.
     *
     * @throws LoadFailedException Thrown when loading of given model failed.
     */
    PolygonalModel(const std::string &filename, bool rebuildNormals);

    /**
     * @brief Load model from given file and optionally rebuild its normals.
     *
     * @param filename Name of the target OBJ formatted file.
     * @param rebuildNormals Set to true to re-calculate normals based on triangles.
     *
     * @return Returns true if the loading completed successfully.
     */
    bool importFromFile(const std::string &filename, bool rebuildNormals);

    /// @brief Normalize model to given scale. Optionally also center it to origin.
    void normalize(float scaleTo = 1.0f, bool center = true);

    /// @brief Get coordinates of the center of the currently loaded model.
    Vector3D centerOfMass() const;
    /// @brief Get width of the currently loaded model.
    float width() const;
    /// @brief Get height of the currently loaded model.
    float height() const;
    /// @brief Get length of the currently loaded model.
    float length() const;
    /// @brief Get radius of the currently loaded model.
    float radius() const;

    /// @brief Access the index buffer.
    const Index *indexBuffer() const;
    /// @brief Number of elements in the index buffer.
    std::size_t indexCount() const;

    /// @brief Access vertex with given index.
    const Vertex &getVertex(std::size_t idx) const;
    /// @brief Access the vertex buffer.
    const Vertex *vertexBuffer() const;
    /// @brief Number of vertices in the vertex buffer.
    std::size_t vertexCount() const;

    /// @brief Total number of triangles.
    std::size_t triangleCount() const;

    /// @brief Get path to the currently loaded model.
    const std::string &loadedPath() const;
    /// @brief Is there a model currently loaded?
    bool isLoaded() const;
private:
    /// @brief Import OBJ file from given file path.
    bool importObj(const std::string &filename, bool rebuildNormals);

    /// @brief Import FBX file from given file path.
    bool importFbx(const std::string &filename, bool rebuildNormals);

    /// Pointer to the model implementation.
    using ModelPtr = std::shared_ptr<impl::PolygonalModelImpl>;

    /// Instance of the internal model.
    ModelPtr mModel{ };
protected:
}; // class PolygonalModel

} // namespace treeacc

#endif // TREEIO_ACCELERATION_POLYGONAL_MODEL_H
