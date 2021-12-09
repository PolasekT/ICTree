/**
 * @author Tomas Polasek, David Hrusa
 * @date 1.14.2020
 * @version 1.0
 * @brief Renderer for the main viewport.
 */

#include <vector>
#include <map>
#include <memory>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtx/euler_angles.hpp>

#include "TreeUtils.h"
#include "TreeGLUtils.h"
#include "TreeCamera.h"
#include "TreeShader.h"
#include "TreeBuffer.h"
#include "TreeTextureBuffer.h"
#include "TreeFrameBuffer.h"
#include "TreeRenderSystem.h"
#include "TreeOperations.h"
#include "TreeReconstruction.h"

#ifndef TREE_RENDERER_H
#define TREE_RENDERER_H

namespace treerndr
{

/// Comment prepended to the generated OBJ files.
static constexpr const char* OBJ_FILE_HEADER_TEXT{ "# produced by TreeIO\n" };

/// @brief Wrapper for a set of points and face indices. Ready to be saved as an obj.
struct RawGeometry : public treeutil::PointerWrapper<RawGeometry>
{
    /// List of vertices in the geometry.
    std::vector<glm::vec4> vertices{ };
    /// List of ordered indices into the vertices array comprising the resulting triangles (GL_TRIANGLE).
    std::vector<GLint> indices{ };

    /// @brief Insert triangle to the this container.
    void insertTriangle(const glm::vec4 &v0, const glm::vec4 &v1, const glm::vec4 &v2);
    /// @brief Insert triangle to the this container.
    void insertTriangle(const glm::vec3 &v0, const glm::vec3 &v1, const glm::vec3 &v2);
    /// @brief Saves the raw mesh data as an obj file.
    void exportObjToFile(std::string path = "data/transform_feedback_mesh.obj");
}; // struct RawGeometry

/// @brief Type of shader used for rendering scene instances.
enum class ShaderType
{
    /// 2D instances and UI elements.
    Screen,
    /// All 3D objects except the reconstruction.
    Mesh,
    /// Tree reconstruction.
    Reconstruction
}; // enum class ShaderType

/// @brief Wrapper around one vertex array object.
struct VaoPack : public treeutil::PointerWrapper<VaoPack>
{
    /// Name of the mesh.
    std::string name{ };
    /// Vertex array object describing the VBOs.
    GLuint vao{ };
    /// Vertex buffer object containing the position.
    GLuint vbo{ };
    /// Vertex buffer object containing the colors.
    GLuint vbocol{ };
    /// Vertex buffer object containing the indices.
    GLuint ebo{ };
    /// Storage buffer used to store extra vertex information such as color.
    GLuint sbo{ };
    /// Rendering mode for contained data.
    GLenum renderMode{ GL_LINES };
    /// Number of elements in the EBO.
    std::size_t elementCount{ 0u };

    /// When set to true, the reconstruction shader will treat this mesh as a processed buffer rather than normal geometry.
    ShaderType shaderType{ ShaderType::Mesh };

    /// Source bundle used in creation of this VAO.
    BufferBundlePtr bundle{ };
    /// Tree being represented by this VAO. Only valid for shaderType == Reconstruction.
    treeop::ArrayTreeHolder::Ptr tree{ };
    /// Tree reconstruction being represented by this VAO. Only valid for shaderType == Reconstruction.
    treeutil::TreeReconstruction::Ptr reconstruction{ };
}; // struct VaoPack

// Forward declaration:
struct Light;

/// @brief Wrapper around information about a single mesh instance within the scene..
struct MeshInstance : public treeutil::PointerWrapper<MeshInstance>
{
    /// Calculate position of the attached light if any.
    glm::vec3 calculateLightPosition() const;
    /// Calculate view-projection matrix for the attached light, if any.
    glm::mat4 calculateLightViewProjectionMatrix(bool useBias = false) const;
    /// Calculate view matrix for the attached light, if any.
    glm::mat4 calculateLightViewMatrix() const;
    /// Calculate view matrix for the attached light, if any.
    glm::mat4 calculateLightProjectionMatrix() const;

    /// Name of this instance. May be left empty.
    std::string name{ };
    /// Pointer to the mesh being instantiated.
    VaoPack::Ptr mesh{ };
    /// Model matrix for this object.
    glm::mat4 model{ glm::mat4(1.0f) };
    /// Scale of the mesh.
    float scale{ 1.0f };
    /// Translation of the mesh from origin.
    glm::vec3 translate{ 0.0f };
    /// Euler-angles rotation of the model.
    glm::vec3 rotate{ 0.0f };
    /// When set to true, model matrix will not be recalculated.
    bool overrideModelMatrix{ false };
    /// Display as wireframe model?
    bool wireframe{ false };
    /// Does this model contain alpha transparency?
    bool transparent{ false };
    /// Is this instance visible even when behind other objects?
    bool alwaysVisible{ false };
    /// Priority used when ordering instances for rendering. Instances with lower values will be rendered sooner.
    int8_t orderPriority{ 0 };
    /// Cull non-visible geometry faces.
    bool cullFaces{ true };
    /// Display in the scene?
    bool show{ true };
    /// How thick should the lines should be for this instance.
    float lineWidth{ 1u };
    /// How large should the points be for this instance.
    float pointSize{ 5u };
    /// Whether this mesh uses textures.
    bool textured{ false };
    /// Whether this mesh uses shadows.
    bool shadows{ false };
    /// Whether this mesh casts shadows.
    bool castsShadows{ false };
    /// Color used as override.
    glm::vec4 overrideColor{ 0.0f };
    /// Use provided color instead of the model specified one?
    bool overrideColorEnabled{ false };
    /// Use color provided in the storage buffer instead of the model specified one?
    bool overrideColorStorageEnabled{ false };
    /// Should not scale with projective camera distance.
    bool antiProjectiveScaling{ false };
    /// Pointer to light attached to this object.
    treeutil::WrapperPtrT<Light> attachedLight{ };
}; // struct MeshInstance

/// @brief Wrapper around information about a single loaded texture.
struct Texture : public treeutil::PointerWrapper<Texture>
{
    /// Buffer containing the texture data.
    TextureBuffer::Ptr buffer;
}; // struct Texture

/// @brief Bundle of buffers which are required to render an object.
struct BufferBundle : public treeutil::PointerWrapper<BufferBundle>
{
    // TODO - Rewrite and create VertexArrayObject to hold this.

    /// Number of vector elements per one color.
    static constexpr auto COLOR_COMPONENTS{ 4u };

    /// Vertex buffer memory.
    std::vector<char> vertex{ };
    /// Color data - 4 floats RGBA.
    std::vector<GLfloat> color{ };
    /// Element data - indices into vertex/color data.
    std::vector<GLuint> element{ };

    /// @brief Get bounding box of this mesh. Result is cached.
    const treeutil::BoundingBox &boundingBox();

    /// @brief Load vertex data from given array of values. Returns number of bytes written.
    template <typename T>
    std::size_t loadVertexData(const std::vector<T> &data);

    /// @brief Set one value within the vertex buffer. Returns number of bytes written.
    template <typename T>
    std::size_t setVertexData(const T &value, std::size_t baseIndex,
        std::size_t byteStride = 1u, std::size_t byteOffset = 0u);

    /// @brief Push one value into the vertex buffer. Returns number of bytes written.
    template <typename T>
    std::size_t pushVertexData(const T &value);

    /// @brief Access vertex data by type.
    template <typename T>
    T &getVertexData(std::size_t baseIndex, std::size_t byteOffset = 0u);

    /// @brief Access vertex data by type.
    template <typename T>
    const T &getVertexData(std::size_t baseIndex, std::size_t byteOffset = 0u) const;
private:
    /// Calculated bounding box.
    treeutil::BoundingBox mBoundingBox{ };
}; // struct BufferBundle

/// @brief Wrapper around information about a single FrameBuffer.
struct FrameBufferObject : public treeutil::PointerWrapper<BufferBundle>
{
    /// Internal frame buffer containing the buffer itself.
    FrameBuffer::Ptr buffer{ };
}; // struct FrameBuffer

/// @brief Holds the necessary information for rendering a light and its shadow map.
struct Light : public treeutil::PointerWrapper<Light>
{
    /// @brief Listing of available light types.
    enum class LightType
    {
        Point,
        Directional
    }; // LightType

    /// Offset of the light from the original object.
    glm::vec3 offset{ };

    /// Target the light is "looking" at. For directional lights this is used to calculate direction.
    glm::vec3 target{ };

    /// View matrix transformation used by the light.
    glm::mat4 view{ };
    /// Projection matrix used by the light.
    glm::mat4 projection{ };

    /// Base resolution of the shadow map created by this light.
    std::size_t shadowMapWidth{ 4096u };
    /// Base resolution of the shadow map created by this light.
    std::size_t shadowMapHeight{ 4096u };
    /// Near plane distance for this light.
    float nearPlane{ 10.0f };
    /// Far plane distance for this light.
    float farPlane{ 40.0f };
    /// Field of view used for some types of lights.
    float fieldOfView{ 40.0f };
    /// Type of this light.
    LightType type{ LightType::Point };

    /// Calculate view matrix for this light and return the result.
    glm::mat4 calculateViewMatrix(const glm::vec3 &position = { });

    /// Calculate projection matrix for this light and return the result.
    glm::mat4 calculateProjectionMatrix();

    /// Calculate view-projection matrix for this light and return the result.
    glm::mat4 calculateViewProjectionMatrix(const glm::vec3 &position = { }, bool useBias = false);
}; // struct Light

/// @brief Holder for static methods generating data buffers for various patterns.
struct PatternFactory
{
    // Pure static class, factory methods only.
    PatternFactory() = delete;

    /// @brief Create a grid pattern with provided attributes.
    static BufferBundle createGrid(std::size_t size = 60u, float spacing = 0.5f,
        float yOffset = 0.03f, const glm::vec4 &color = { 0.3f, 0.3f, 0.3f, 1.0f });

    /// @brief Create a grid pattern with provided attributes.
    static BufferBundle createPlane(float size = 30.0f,
        float yOffset = 0.0f, const glm::vec4 &color = { 1.0f, 1.0f, 1.0f, 1.0f });
}; // struct PatternFactory

/// @brief Wrapper around all information required for rendering the main application viewport.
class TreeRenderer : public treeutil::PointerWrapper<TreeRenderer>
{
public:
    /// Pointer to a single object mesh.
    using VaoPackPtr = VaoPack::Ptr;
    /// Pointer to a single mesh instance.
    using MeshInstancePtr = MeshInstance::Ptr;
    /// Pointer to a single bundle containing object data.
    using BufferBundlePtr = BufferBundle::Ptr;
    /// Pointer to a single texture containing its data.
    using TexturePtr = Texture::Ptr;
    /// Pointer to a single frame-buffer containing its data.
    using FrameBufferPtr = FrameBufferObject::Ptr;
    /// Pointer to a single light containing its data.
    using LightPtr = Light::Ptr;

    /// @brief Initialize internal structures. Does NOT initialize anything OpenGL related!
    TreeRenderer();
    /// @brief Free all rendering resources and buffers.
    ~TreeRenderer();

    // No copying or moving.
    TreeRenderer(const TreeRenderer &other) = delete;
    TreeRenderer &operator=(const TreeRenderer &other) = delete;

    /// @brief Initialize the renderer. Specify zero width/height for automatic detection.
    void initialize(std::size_t width = 0u, std::size_t height = 0u);

    /// @brief Should be called on viewport resize.
    void resize(int newWidth, int newHeight);

    /// @brief Create and get VAO by its name. If it already exists it is just returned.
    VaoPackPtr createGetVao(const std::string &name);
    /// @brief Returns VAO with given name. Returns nullptr if not registered.
    VaoPackPtr getVao(const std::string &name);

    /// @brief Create and get bundle by its name. If it already exists it is just returned.
    BufferBundlePtr createGetBundle(const std::string &name);
    /// @brief Returns bundle with given name. Returns nullptr if not registered.
    BufferBundlePtr getBundle(const std::string &name);

    /// @brief Load obj from given path into provided vertex array object. Returns pointer to corresponding buffer data.
    BufferBundlePtr loadObjInto(VaoPackPtr vao, const std::string &path, GLenum mode = GL_NONE, bool forceReload = false);

    /**
     * @brief Load fbx from given path into provided vertex array object. Returns pointer to corresponding buffer data.
     *
     * @param vao Target VAO pack.
     * @param path Path to the FBX file.
     * @param mode Mode of rendering for the laoded model.
     * @param output Optional output skeleton from the loaded fbx.
     *
     * @return Returns pointer to corresponding buffer data.
     */
    BufferBundlePtr loadFbxInto(VaoPackPtr vao, const std::string &path,
        GLenum mode = GL_NONE, treeio::ArrayTree *output = nullptr);

    /// @brief Load given buffer bundle into provided vertex array object, providing attrib pointer mappings for the main shader.
    static void loadBundleInto(VaoPackPtr vao, BufferBundlePtr bundle, GLenum mode = GL_NONE);

    /// @brief Load given buffer bundle into provided vertex array object, providing attrib pointer mappings for the recon shader.
    static void loadReconBundleInto(VaoPackPtr vao, BufferBundlePtr bundle, GLenum mode = GL_NONE);

    /// @brief Copy one mesh into another. Performs deep copy of all existing buffers.
    static void duplicateMeshInto(VaoPackPtr srcVao, VaoPackPtr dstVao);

    /**
     * @brief Create a new storage buffer, fill it with data and assign it to given mesh.
     * @param vao Mesh to assign the storage buffer to.
     * @param data Data uploaded to the buffer.
     * @param mode Use mode of the buffer.
     * @tparam T Type of data saved to the buffer.
     */
    template <typename T>
    static void addStorageBufferTo(VaoPackPtr vao, const std::vector<T> &data, GLenum mode = GL_DYNAMIC_DRAW)
    { addStorageBufferTo(vao, data.data(), sizeof(T) * data.size(), mode); }

    /**
     * @brief Create a new storage buffer, fill it with data and assign it to given mesh.
     * @param vao Mesh to assign the storage buffer to.
     * @param data Pointer to the data. May be nullptr in which case the buffer will be
     *   created with storage for required number of bytes which are left uninitialized.
     * @param mode Use mode of the buffer.
     */
    static void addStorageBufferTo(VaoPackPtr vao, const void *data, std::size_t dataBytes, GLenum mode = GL_DYNAMIC_DRAW);

    /// @brief Load OBJ mesh from file, name it and return a handle to it. Use alias to disable caching.
    VaoPackPtr loadMesh(const std::string &path, const std::string &alias = "");
    /// @brief Duplicate provided mesh under different name.
    VaoPackPtr duplicateMesh(VaoPackPtr mesh, const std::string &name);

    /// @brief Load OBJ mesh from file, name it and instantiate it within the scene.
    MeshInstancePtr loadMeshToInstance(const std::string &name, const std::string &path);
    /// @brief Create instance of given mesh. Returns nullptr if not loaded.
    MeshInstancePtr createInstance(const std::string &name, VaoPackPtr mesh);
    /// @brief Get instance by name. Returns nullptr if not loaded.
    MeshInstancePtr getInstance(const std::string &name);

    /// @brief Access iterable instance map.
    std::map<std::string, MeshInstancePtr> iterableInstances() const;

    /// @brief Load texture from file and name it.
    TexturePtr loadTexture(const std::string &name, const std::string &path);
    /// @brief Load texture from file and name it using its path.
    TexturePtr loadTexture(const std::string &path);
    /// @brief Get loaded texture by name. Returns nullptr if not loaded.
    TexturePtr getTexture(const std::string &name);

    /// @brief Load texture from given path into provided texture object.
    void loadTextureInto(TexturePtr texture, const std::string &path);

    /// @brief Create and get frame-buffer by its name. If it already exists it is just returned.
    BufferBundlePtr createGetFrameBuffer(const std::string &name);
    /// @brief Returns frame-buffer with given name. Returns nullptr if not registered.
    BufferBundlePtr getFrameBuffer(const std::string &name);

    /// @brief Create and get light by its name. If it already exists it is just returned.
    LightPtr createGetLight(const std::string &name);
    /// @brief Returns light with given name. Returns nullptr if not registered.
    LightPtr getLight(const std::string &name);
    /// @brief Attach light to given instance.
    bool attachLight(const MeshInstancePtr &instance, const LightPtr &light);

    /// @brief Make given renderer type available for use. Must be used before calling initialize()!
    template <typename T>
    void registerRenderer();

    /// @brief Use given renderer type.
    template <typename T>
    void useRenderer();

    /// @brief Use given renderer type.
    void useRenderer(const std::string &name);

    /// @brief Access already registered renderer.
    template <typename RendererT>
    RendererT &renderer();

    /// @brief Access already registered renderer.
    template <typename RendererT>
    const RendererT &renderer() const;

    /// @brief Get currently used renderer.
    RenderSystem::Ptr activeRenderer();

    /// @brief Attempt to reload all used shaders.
    void reloadShaders();

    /// @brief Render given configuration of the scene.
    void render(treescene::CameraState &camera, treescene::TreeScene &scene);

    /**
     * @brief Export rendered viewport into a file. Returns true if failed. Use 0, 0, 0, 0 to capture whole screen.
     * @warning This function only copies content of the current output frame-buffer.
     *
     * @param camera Target camera state.
     * @param scene Scene to render.
     * @param path Path to save the output pixels to.
     * @param xStart Starting pixel offset on the x axis.
     * @param yStart Starting pixel offset on the y axis.
     * @param xEnd Ending pixel offset on the x axis. Use zero to continue until viewportWidth.
     * @param yEnd Ending pixel offset on the y axis. Use zero to continue until viewportHeight.
     * @param addTimestamp Add timestamp to the filename?
     * @return Returns false if operation completed successfully. Returns true on error.
     */
    bool screenshot(treescene::CameraState &camera, treescene::TreeScene &scene,
        const std::string &path, std::size_t xStart = 0u, std::size_t yStart = 0u,
        std::size_t xEnd = 0u, std::size_t yEnd = 0u, bool addTimestamp = true);

    /**
     * @brief Render specified viewport into a file. Returns true if failed. Use 0, 0 for current viewport size.
     * @warning This function performs full rendering!
     *
     * @param camera Target camera state.
     * @param scene Scene to render.
     * @param path Path to save the output pixels to.
     * @param xPixels Size of the frame-buffer to render to.
     * @param yPixels Size of the frame-buffer to render to.
     * @param samples Samples per pixel. Values greater than 1 enable multi-sampling.
     * @param addTimestamp Add timestamp to the filename?
     * @param renderUi Render the user interface?
     * @return Returns false if operation completed successfully. Returns true on error.
     */
    bool renderScreenshot(treescene::CameraState &camera, treescene::TreeScene &scene,
        const std::string &path, std::size_t xPixels = 0u, std::size_t yPixels = 0u,
        std::size_t samples = 1u, bool addTimestamp = true, bool renderUi = false);

    /// @brief Calculate model matrix for given object instance.
    static glm::mat4 calculateModelMatrix(const MeshInstance &instance, const treescene::CameraState &camera);

    /// @brief Calculate model matrix for given object instance. This version does not allow for anti-projective scaling.
    static glm::mat4 calculateModelMatrixNoAntiProjection(const MeshInstance &instance);

    /// @brief Sort instances in order in which they should be rendered.
    std::vector<MeshInstancePtr> sortInstancesForRendering(bool includeInvisible = false);
private:
    /// Path used for placeholder OBJ models, when the original is not available.
    static constexpr auto OBJ_PLACEHOLDER_MODEL_PATH{ "obj/placeholder.obj" };

    // Allow access to the internals.
    friend class RenderSystem;

    /// Pointer to a rendering system.
    using RendererPtr = RenderSystem::Ptr;

    /// Mapping from mesh name to its data.
    using MeshMap = std::map<std::string, VaoPackPtr>;
    /// Mesh instances within the scene.
    using InstanceMap = std::map<std::string, MeshInstancePtr>;
    /// Mapping from bundle name to its data.
    using BundleMap = std::map<std::string, BufferBundlePtr>;
    /// Mapping from texture name to its data.
    using TextureMap = std::map<std::string, TexturePtr>;
    /// Mapping from frame-buffer name to its data.
    using FrameBufferMap = std::map<std::string, FrameBufferPtr>;
    /// Mapping from light name to its data.
    using LightMap = std::map<std::string, LightPtr>;
    /// Mapping from renderer name to its implementation.
    using RendererMap = std::map<std::string, RendererPtr>;

    /// @brief Print some information about the current OpenGL context.
    void printGlInfo();

    /// @brief Setup OpenGL error traces.
    void setupDebugReporting();

    /// @brief If source represents valid buffer, then a new destination with same content will be created.
    static void duplicateBufferInto(GLuint source, GLuint &destination);

    /// @brief Sort instances in order in which they should be rendered.
    std::vector<MeshInstancePtr> sortInstancesForRendering(const InstanceMap &instances, bool includeInvisible = false);

    /// @brief Get registered renderer by name.
    RendererPtr getRenderer(const std::string &name);

    /// Mapping from mesh name to its data.
    MeshMap mMeshes{ };
    /// Mesh instances within the scene.
    InstanceMap mInstances{ };
    /// Mapping from bundle name to its data.
    BundleMap mBundles{ };
    /// Mapping from texture name to its data.
    TextureMap mTextures{ };
    /// Mapping from frame-buffer name to its data.
    FrameBufferMap mFrameBuffers{ };
    /// Mapping from light name to its data.
    LightMap mLights{ };
    /// Mapping from renderer name to its implementation.
    RendererMap mRenderers{ };

    /// Currently used renderer.
    RendererPtr mActiveRenderer{ };
    /// Has the renderer benn initialized?
    bool mInitialized{ false };
protected:
}; // class TreeRenderer

} // namespace treerndr

// Template implementation begin.

namespace treerndr
{

template <typename T>
std::size_t BufferBundle::loadVertexData(const std::vector<T> &data)
{
    const auto byteSize{ data.size() * sizeof(data[0]) };
    vertex.resize(byteSize);
    std::memcpy(vertex.data(), data.data(), byteSize);
    return byteSize;
}

template <typename T>
std::size_t BufferBundle::setVertexData(const T &value,
    std::size_t baseIndex, std::size_t byteStride, std::size_t byteOffset)
{
    const auto startIndex{ baseIndex * byteStride + byteOffset };
    const auto *data{ reinterpret_cast<const char*>(&value) };
    for (std::size_t iii = 0u; iii < sizeof(T); ++iii)
    { vertex[startIndex + iii] = data[iii]; }
    return sizeof(T);
}

template <typename T>
std::size_t BufferBundle::pushVertexData(const T &value)
{
    const auto baseIdx{ vertex.size() };
    vertex.resize(baseIdx + sizeof(T));
    return setVertexData(value, baseIdx, 1u, 0u);
}

template <typename T>
T &BufferBundle::getVertexData(std::size_t baseIndex, std::size_t byteOffset)
{
    const auto startIndex{ baseIndex * sizeof(T) + byteOffset };
    return *reinterpret_cast<T*>(&vertex[startIndex]);
}

template <typename T>
const T &BufferBundle::getVertexData(std::size_t baseIndex, std::size_t byteOffset) const
{
    const auto startIndex{ baseIndex * sizeof(T) + byteOffset };
    return *reinterpret_cast<const T*>(&vertex[startIndex]);
}

template <typename T>
void TreeRenderer::registerRenderer()
{
    if (mInitialized)
    {
        Error << "Failed to register renderer \"" << T::RENDERER_NAME
              << "\", unable to add after initialize()!" << std::endl;
    }
    mRenderers.emplace(T::RENDERER_NAME, treeutil::WrapperCtrT<T>());
    if (!mActiveRenderer)
    { mActiveRenderer = mRenderers[T::RENDERER_NAME]; }
}

template <typename T>
void TreeRenderer::useRenderer()
{ useRenderer(T::RENDERER_NAME); }

template <typename RendererT>
RendererT &TreeRenderer::renderer()
{
    const auto rendererPtr{ getRenderer(RendererT::RENDERER_NAME) };
    return *std::dynamic_pointer_cast<RendererT>(rendererPtr);
}

template <typename RendererT>
const RendererT &TreeRenderer::renderer() const
{
    const auto rendererPtr{ getRenderer(RendererT::RENDERER_NAME) };
    return *std::dynamic_pointer_cast<RendererT>(rendererPtr);
}

} // namespace treerndr

// Template implementation end.

#endif // TREE_RENDERER_H
