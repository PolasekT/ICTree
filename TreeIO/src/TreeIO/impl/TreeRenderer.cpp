/**
 * @author Tomas Polasek, David Hrusa
 * @date 1.14.2020
 * @version 1.0
 * @brief Renderer for the main viewport.
 */

#include "TreeRenderer.h"

#include <fstream>

#include <TreeIO/TreeIO.h>

#include "TreeGLUtils.h"
#include "TreeScene.h"

namespace treerndr
{

void RawGeometry::insertTriangle(const glm::vec4 &v0, const glm::vec4 &v1, const glm::vec4 &v2)
{ treeutil::insertTriangle(vertices, indices, v0, v2, v2); }

void RawGeometry::insertTriangle(const glm::vec3 &v0, const glm::vec3 &v1, const glm::vec3 &v2)
{
    treeutil::insertTriangle(vertices, indices,
        glm::vec4{ v0.x, v0.y, v0.z, 1.0f },
        glm::vec4{ v1.x, v1.y, v1.z, 1.0f },
        glm::vec4{ v2.x, v2.y, v2.z, 1.0f });
}

void RawGeometry::exportObjToFile(std::string path)
{
    std::ofstream out(path);
    out << OBJ_FILE_HEADER_TEXT << "# vertices begin here:\n";
    for (std::size_t t = 0; t < vertices.size(); t++)
    { out << "v " << vertices[t].x << " " << vertices[t].y << " " << vertices[t].z << /*" " << vertices[t].w <<*/ "\n"; }
    out << "# faces\n";
    for (std::size_t f = 0; f < indices.size(); f += 3)
    { out << "f " << indices[f] << " " << indices[f + 1] << " " << indices[f + 2] << "\n"; }
    out.close();
    Info << "Raw Mesh written to '"<< path << "'\n";
}

glm::vec3 MeshInstance::calculateLightPosition() const
{ return attachedLight ? translate + attachedLight->offset : translate; }

glm::mat4 MeshInstance::calculateLightViewProjectionMatrix(bool useBias) const
{ return attachedLight ? attachedLight->calculateViewProjectionMatrix(calculateLightPosition(), useBias) : glm::mat4{ }; }

glm::mat4 MeshInstance::calculateLightViewMatrix() const
{ return attachedLight ? attachedLight->calculateViewMatrix(calculateLightPosition()) : glm::mat4{ }; }

glm::mat4 MeshInstance::calculateLightProjectionMatrix() const
{ return attachedLight ? attachedLight->calculateProjectionMatrix() : glm::mat4{ }; }

const treeutil::BoundingBox &BufferBundle::boundingBox()
{
    // Check if we have a cached version ready.
    if (mBoundingBox.filled)
    { return mBoundingBox; }

    // Calculate bounding box.
    auto minX{ std::numeric_limits<float>::max() };
    auto minY{ std::numeric_limits<float>::max() };
    auto minZ{ std::numeric_limits<float>::max() };
    auto maxX{ std::numeric_limits<float>::min() };
    auto maxY{ std::numeric_limits<float>::min() };
    auto maxZ{ std::numeric_limits<float>::min() };

    static constexpr auto VERTEX_ELEMENTS{ 3u };
    const auto vertexCount{ vertex.size() / (VERTEX_ELEMENTS * sizeof(GLfloat)) };
    for (std::size_t vertIdx = 0u; vertIdx < vertexCount; ++vertIdx)
    { // Find extreme values.
        const auto xPos{ getVertexData<GLfloat>(vertIdx * VERTEX_ELEMENTS + 0u) };
        const auto yPos{ getVertexData<GLfloat>(vertIdx * VERTEX_ELEMENTS + 1u) };
        const auto zPos{ getVertexData<GLfloat>(vertIdx * VERTEX_ELEMENTS + 2u) };

        minX = std::min(minX, xPos); minY = std::min(minY, yPos); minZ = std::min(minZ, zPos);
        maxX = std::max(maxX, xPos); maxY = std::max(maxY, yPos); maxZ = std::max(maxZ, zPos);
    }

    // Cache the values for later re-use.
    mBoundingBox.position[0u] = minX;
    mBoundingBox.position[1u] = minY;
    mBoundingBox.position[2u] = minZ;
    mBoundingBox.size[0u] = maxX - minX;
    mBoundingBox.size[1u] = maxY - minY;
    mBoundingBox.size[2u] = maxZ - minZ;
    mBoundingBox.filled = true;

    return mBoundingBox;
}

glm::mat4 Light::calculateViewMatrix(const glm::vec3 &position)
{
    const auto offPosition{ position + offset };
    const auto offTarget{ target + offset };
    const auto positionToTarget{ glm::normalize(offTarget - offPosition) };
    view = glm::lookAt(
        offPosition, offTarget,
        (glm::dot(positionToTarget, glm::vec3{ 1.0f, 0.0f, 0.0f }) >=
            1.0f - std::numeric_limits<float>::epsilon()) ?
                glm::cross(positionToTarget, glm::vec3{ 0.0f, 0.0f, 1.0f }) :
                glm::cross(positionToTarget, glm::vec3{ 1.0f, 0.0f, 0.0f })
    );

    return view;
}

glm::mat4 Light::calculateProjectionMatrix()
{
    if (type == LightType::Directional)
    { // Directional light source -> Use orthographic projection.
        const auto halfWidth{ shadowMapWidth / 2.0f };
        const auto halfHeight{ shadowMapHeight / 2.0f };
        const auto xSide{ halfWidth / fieldOfView };
        const auto ySide{ halfHeight / fieldOfView };
        projection = glm::ortho(
            -xSide, xSide, -ySide, ySide,
            nearPlane, farPlane);
    }
    else
    { // Point light source -> Use perspective projection.
        projection = glm::perspective(glm::radians(fieldOfView),
            shadowMapWidth / static_cast<float>(shadowMapHeight),
            nearPlane, farPlane);
    }

    return projection;
}

glm::mat4 Light::calculateViewProjectionMatrix(const glm::vec3 &position, bool useBias)
{
    glm::mat4 bias{
        0.5f, 0.0f, 0.0f, 0.5,
        0.0f, 0.5, 0.0f, 0.5,
        0.0f, 0.0f, 0.5, 0.5,
        0.0f, 0.0f, 0.0f, 1.0f
    };

    view = calculateProjectionMatrix() * calculateViewMatrix(position);
    if (useBias)
    { view = bias * view; }

    return view;
}

BufferBundle PatternFactory::createGrid(std::size_t size,
    float spacing, float yOffset, const glm::vec4 &color)
{
    const auto GRID_SIZE{ size };
    const auto GRID_LENGTH{ static_cast<float>(size) };

    const auto VERTEX_CHANNELS{ 3u };
    const auto ELEMENT_CHANNELS{ 1u };
    const auto COLOR_CHANNELS{ 4u };

    /*
     * 2 vertices per line, 2 lines (horizontal and vertical)
     * and added one finishing line for the grid.
     */
    const auto VERTEX_COUNT{ 2u * 2u * (GRID_SIZE + 1u) };

    std::vector<GLfloat> verts(VERTEX_COUNT * VERTEX_CHANNELS);
    std::vector<GLuint> eles(VERTEX_COUNT * ELEMENT_CHANNELS);
    std::vector<GLfloat> cols(VERTEX_COUNT * COLOR_CHANNELS);

    for (GLuint iii = 0u; iii <= GRID_SIZE; ++iii)
    {
        // First point:
        eles[iii * 2u] = iii * 2u;
        verts[iii * 6u] = (-GRID_LENGTH / 2.0f + iii) * spacing;
        verts[iii * 6u + 1u] = yOffset;
        verts[iii * 6u + 2u] = (-GRID_LENGTH / 2.0f) * spacing;
        // Second point:
        eles[iii * 2u + 1u] = iii * 2u + 1u;
        verts[iii * 6u + 3] = (-GRID_LENGTH / 2.0f + iii) * spacing;
        verts[iii * 6u + 4] = yOffset;
        verts[iii * 6u + 5] = (GRID_LENGTH / 2.0f) * spacing;

        // First point:
        eles[GRID_SIZE * 2u + iii * 2u] = static_cast<GLuint>(GRID_LENGTH * 2u + iii * 2u);
        verts[(GRID_SIZE + 1u) * 6u + iii * 6u] = (-GRID_LENGTH / 2.0f) * spacing;
        verts[(GRID_SIZE + 1u) * 6u + iii * 6u + 1u] = yOffset;
        verts[(GRID_SIZE + 1u) * 6u + iii * 6u + 2u] = (-GRID_LENGTH / 2.0f + iii) * spacing;
        // Second point:
        eles[GRID_SIZE * 2u + iii * 2u + 1u] = static_cast<GLuint>(GRID_LENGTH * 2u + iii * 2u + 1u);
        verts[(GRID_SIZE + 1u) * 6u + iii * 6u + 3] = (GRID_LENGTH / 2.0f) * spacing;
        verts[(GRID_SIZE + 1u) * 6u + iii * 6u + 4] = yOffset;
        verts[(GRID_SIZE + 1u) * 6u + iii * 6u + 5] = (-GRID_LENGTH / 2.0f + iii) * spacing;
    }

    for (std::size_t iii = 0u; iii < VERTEX_COUNT; ++iii)
    {
        cols[iii * COLOR_CHANNELS + 0u] = color.r;
        cols[iii * COLOR_CHANNELS + 1u] = color.g;
        cols[iii * COLOR_CHANNELS + 2u] = color.b;
        cols[iii * COLOR_CHANNELS + 3u] = color.a;
    }

    BufferBundle result{ };
    result.loadVertexData(verts);
    result.color = std::move(cols);
    result.element = std::move(eles);

    return result;
}

BufferBundle PatternFactory::createPlane(float size,
    float yOffset, const glm::vec4 &color)
{
    const auto GRID_SIZE{ size };
    const auto VERTEX_CHANNELS{ 3u };
    const auto ELEMENT_CHANNELS{ 1u };
    const auto COLOR_CHANNELS{ 4u };

    const auto VERTEX_COUNT{ 4u };

    std::vector<GLfloat> verts(VERTEX_COUNT * VERTEX_CHANNELS);
    std::vector<GLuint> eles(VERTEX_COUNT * ELEMENT_CHANNELS);
    std::vector<GLfloat> cols(VERTEX_COUNT * COLOR_CHANNELS);

    {
        // First point:
        eles[0u * ELEMENT_CHANNELS] = 0u;
        verts[0u * VERTEX_CHANNELS+0u] = (-GRID_SIZE / 2.0f);
        verts[0u * VERTEX_CHANNELS+1u] = yOffset;
        verts[0u * VERTEX_CHANNELS+2u] = (-GRID_SIZE / 2.0f);
        // Second point:
        eles[1u * ELEMENT_CHANNELS] = 1u;
        verts[1u * VERTEX_CHANNELS+0u] = (-GRID_SIZE / 2.0f);
        verts[1u * VERTEX_CHANNELS+1u] = yOffset;
        verts[1u * VERTEX_CHANNELS+2u] = (GRID_SIZE / 2.0f);

        // First point:
        eles[2u * ELEMENT_CHANNELS] = 2u;
        verts[2u * VERTEX_CHANNELS+0u] = (GRID_SIZE / 2.0f);
        verts[2u * VERTEX_CHANNELS+1u] = yOffset;
        verts[2u * VERTEX_CHANNELS+2u] = (-GRID_SIZE / 2.0f);
        // Second point:
        eles[3u * ELEMENT_CHANNELS] = 3u;
        verts[3u * VERTEX_CHANNELS+0u] = (GRID_SIZE / 2.0f);
        verts[3u * VERTEX_CHANNELS+1u] = yOffset;
        verts[3u * VERTEX_CHANNELS+2u] = (GRID_SIZE / 2.0f);
    }

    for (std::size_t iii = 0u; iii < VERTEX_COUNT; ++iii)
    {
        cols[iii * COLOR_CHANNELS + 0u] = color.r;
        cols[iii * COLOR_CHANNELS + 1u] = color.g;
        cols[iii * COLOR_CHANNELS + 2u] = color.b;
        cols[iii * COLOR_CHANNELS + 3u] = color.a;
    }

    BufferBundle result{ };
    result.loadVertexData(verts);
    result.color = std::move(cols);
    result.element = std::move(eles);

    return result;
}

TreeRenderer::TreeRenderer()
{ /* Automatic */ }

TreeRenderer::~TreeRenderer()
{ /* Automatic */ }

void TreeRenderer::initialize(std::size_t width, std::size_t height)
{
    printGlInfo();

    glewInit();

#ifndef NDEBUG
    setupDebugReporting();
#endif // NDEBUG

#ifdef IO_USE_EGL
    // Initialize viewport.
    glEnable(GL_DEPTH_TEST);
    glViewport(
        0, 0,
        width ? width : 1024,
        height ? height : 1024
    );

    // Alpha blending.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Multi-sampling.
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_MULTISAMPLE_ARB);
#else // IO_USE_EGL
    // Initialize viewport.
    glEnable(GL_DEPTH_TEST);
    glViewport(
        0, 0,
        width ? width : glutGet(GLUT_WINDOW_WIDTH),
        height ? height : glutGet(GLUT_WINDOW_HEIGHT)
    );

    // Alpha blending.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Multi-sampling.
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_MULTISAMPLE_ARB);
#endif

    // Initialize all renderers.
    for (const auto &rendererRec : mRenderers)
    {
        rendererRec.second->initialize();
        Info << "Initialized renderer " << rendererRec.first << " : " << *rendererRec.second << std::endl;
    }

    mInitialized = true;
}

void TreeRenderer::resize(int newWidth, int newHeight)
{ glViewport(0, 0, newWidth, newHeight); }

TreeRenderer::VaoPackPtr TreeRenderer::createGetVao(const std::string &name)
{
    auto vao{ getVao(name) };

    if (!vao)
    { // Not created yet -> initialize.
        vao = VaoPack::instantiate();
        vao->name = name;
        glGenVertexArrays(1u, &vao->vao);
        mMeshes.emplace(name, vao);
    }

    return vao;
}

TreeRenderer::VaoPackPtr TreeRenderer::getVao(const std::string &name)
{
    const auto findIt{ mMeshes.find(name) };
    if (findIt != mMeshes.end())
    { return findIt->second; }
    else
    { return { }; }
}

TreeRenderer::BufferBundlePtr TreeRenderer::createGetBundle(const std::string &name)
{
    auto bundle{ getBundle(name) };

    if (!bundle)
    { // Not created yet -> initialize.
        bundle = BufferBundle::instantiate();
        mBundles.emplace(name, bundle);
    }

    return bundle;
}

TreeRenderer::BufferBundlePtr TreeRenderer::getBundle(const std::string &name)
{
    const auto findIt{ mBundles.find(name) };
    if (findIt != mBundles.end())
    { return findIt->second; }
    else
    { return { }; }
}

TreeRenderer::BufferBundlePtr TreeRenderer::loadObjInto(VaoPackPtr vao, const std::string &path, GLenum mode, bool forceReload)
{
    auto bundle{ getBundle(path) };

    if (!bundle || forceReload)
    { // Bundle for this object is not created yet -> initialize.
        treeio::ObjImporter importer;
        if (!importer.import(path))
        {
            Error << "Model import failed, using placeholder model ("
                  << OBJ_PLACEHOLDER_MODEL_PATH << ")!" << std::endl;
            if (!importer.import(OBJ_PLACEHOLDER_MODEL_PATH))
            { Error << "Failed to load placeholder model!" << std::endl; }
        }

        Info << "Loaded obj model (\"" << path << "\") with: " << importer.vertexCount()
             << " vertices and " << importer.indexCount() << " indices" << std::endl;

        // Store the model data in a new bundle.
        bundle = createGetBundle(path);
        // TODO - Use full Vertex structure?
        bundle->loadVertexData(importer.movePositions());
        bundle->color = importer.moveColors();
        bundle->element = importer.moveIndices();
    }

    loadBundleInto(vao, bundle, mode);
    return bundle;
}

TreeRenderer::BufferBundlePtr TreeRenderer::loadFbxInto(VaoPackPtr vao, const std::string &path,
    GLenum mode, treeio::ArrayTree *output)
{
    auto bundle{ getBundle(path) };

    if (!bundle || output)
    { // Bundle for this object is not created yet or we need to re-import -> initialize.
        treeio::FbxImporter importer;
        importer.import(path, output != nullptr);

        Info << "Loaded fbx model (\"" << path << "\") with: " << importer.vertexCount()
             << " vertices and " << importer.indexCount() << " indices" << std::endl;

        if (importer.skeleton().nodeCount())
        { Info << "Fbx model also included skeleton with " << importer.skeleton().nodeCount() << " nodes" << std::endl; }
        else if (output)
        { Info << "Fbx model did not include skeleton" << std::endl; }
        else
        { Info << "Skeleton loading was skipped" << std::endl; }

        // Store the model data in a new bundle.
        bundle = createGetBundle(path);
        // TODO - Use all elements of the Vertex?
        bundle->loadVertexData(importer.movePositions());
        bundle->color = importer.moveColors();
        bundle->element = importer.moveIndices();

        // Optionally also save the skeleton.
        if (output)
        { *output = importer.moveSkeleton(); }
    }

    loadBundleInto(vao, bundle, mode);
    return bundle;
}

void TreeRenderer::loadBundleInto(VaoPackPtr vao, BufferBundlePtr bundle, GLenum mode)
{
#ifndef NDEBUG
    const auto totalSize{
        sizeof(bundle->vertex[0]) * bundle->vertex.size() +
        sizeof(bundle->color[0]) * bundle->color.size() +
        sizeof(bundle->element[0]) * bundle->element.size()
    };
    Info << "Loading bundle data onto the GPU (" << totalSize << " bytes)..." << std::endl;
#endif // NDEBUG

    // Store it in the target VAO:
    if (mode != GL_NONE)
    { vao->renderMode = mode; }
    glBindVertexArray(vao->vao);

    // Cleanup any old data.
    if (vao->vbo)
    { glDeleteBuffers(1u, &vao->vbo); vao->vbo = 0u; }
    if (vao->vbocol)
    { glDeleteBuffers(1u, &vao->vbocol); vao->vbocol = 0u; }
    if (vao->ebo)
    { glDeleteBuffers(1u, &vao->ebo); vao->ebo = 0u; }

    if (bundle->color.size() % 4u != 0u)
    { Error << "Unable to load bundle: Colors should always have 4 components!" << std::endl; return; }

    // Upload position data:
    glGenBuffers(1u, &vao->vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vao->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(bundle->vertex[0]) * bundle->vertex.size(), bundle->vertex.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0u);
    glVertexAttribPointer(0u, 3u, GL_FLOAT, GL_FALSE, 3u * sizeof(GLfloat), nullptr);
    glBindBuffer(GL_ARRAY_BUFFER, 0u);

    // Upload color data:
    if (!bundle->color.empty())
    {
        glGenBuffers(1u, &vao->vbocol);
        glBindBuffer(GL_ARRAY_BUFFER, vao->vbocol);
        glBufferData(GL_ARRAY_BUFFER, sizeof(bundle->color[0]) * bundle->color.size(), bundle->color.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(1u);
        glVertexAttribPointer(1u, 4u, GL_FLOAT, GL_FALSE, 4u * sizeof(GLfloat), nullptr);
        glBindBuffer(GL_ARRAY_BUFFER, 0u);
    }

    // Upload element data:
    vao->elementCount = bundle->element.size();
    glGenBuffers(1u, &vao->ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vao->ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(bundle->element[0]) * bundle->element.size(), bundle->element.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0u);
    vao->bundle = bundle;

    glBindVertexArray(0u);
}

void TreeRenderer::loadReconBundleInto(VaoPackPtr vao, BufferBundlePtr bundle, GLenum mode)
{
#ifndef NDEBUG
    const auto totalSize{
        sizeof(bundle->vertex[0]) * bundle->vertex.size() +
        sizeof(bundle->color[0]) * bundle->color.size() +
        sizeof(bundle->element[0]) * bundle->element.size()
    };
    Info << "Loading reconstruction data onto the GPU (" << totalSize << " bytes)..." << std::endl;
#endif // NDEBUG

    // Store it in the target VAO:
    if (mode != GL_NONE)
    {
        vao->renderMode = mode;
    }
    glBindVertexArray(vao->vao);

    // Cleanup any old data.
    if (vao->vbo)
    {
        glDeleteBuffers(1u, &vao->vbo);
    }
    if (vao->vbocol)
    {
        glDeleteBuffers(1u, &vao->vbocol);
    }
    if (vao->ebo)
    {
        glDeleteBuffers(1u, &vao->ebo);
    }

    if (bundle->vertex.size() % (20u * sizeof(GLfloat)) != 0)
    { Error << "Unable to load bundle: Vertices should always have 16 components!" << std::endl; return; }

    // Upload reconstruction data:
    glGenBuffers(1u, &vao->vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vao->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(bundle->vertex[0]) * bundle->vertex.size(), bundle->vertex.data(), GL_STATIC_DRAW);
    /// Position of the vertex in model space, w is always set to 1.0f.
    glEnableVertexAttribArray(0u);
    glVertexAttribPointer(0u, 4u, GL_FLOAT, GL_FALSE, (4u * 5u * sizeof(GLfloat)), (void*)0u);
    /// Normal of the vertex in model space, w is set to distance from the tree root.
    glEnableVertexAttribArray(1u);
    glVertexAttribPointer(1u, 4u, GL_FLOAT, GL_FALSE, (4u * 5u * sizeof(GLfloat)), (void*)(1u * 4u * sizeof(GLfloat)));
    /// Vector parallel with the branch, w is set to the radius of the branch.
    glEnableVertexAttribArray(2u);
    glVertexAttribPointer(2u, 4u, GL_FLOAT, GL_FALSE, (4u * 5u * sizeof(GLfloat)), (void*)(2u * 4u * sizeof(GLfloat)));
    /// Tangent of the branch, w is unused.
    glEnableVertexAttribArray(3u);
    glVertexAttribPointer(3u, 4u, GL_FLOAT, GL_FALSE, (4u * 5u * sizeof(GLfloat)), (void*)(3u * 4u * sizeof(GLfloat)));
    /// Adjacency indices for this vertex, x = this idx, y = parent idx, z = child idx, w is unused. 0 as invalid idx.
    glEnableVertexAttribArray(4u);
    glVertexAttribIPointer(4u, 4u, GL_UNSIGNED_INT, (4u * 5u * sizeof(GLfloat)), (void*)(4u * 4u * sizeof(GLfloat)));
    glBindBuffer(GL_ARRAY_BUFFER, 0u);

    vao->shaderType = ShaderType::Reconstruction;
    vao->renderMode = GL_PATCHES;
    vao->elementCount = bundle->vertex.size() / (20u * sizeof(GLfloat));
    vao->bundle = bundle;

    glBindVertexArray(0u);
}

void TreeRenderer::duplicateMeshInto(VaoPackPtr srcVao, VaoPackPtr dstVao)
{
    // TODO - Also duplicate the VAO?
    dstVao->vao = dstVao->vao;
    dstVao->bundle = srcVao->bundle;
    duplicateBufferInto(srcVao->vbo, dstVao->vbo);
    duplicateBufferInto(srcVao->vbocol, dstVao->vbocol);
    duplicateBufferInto(srcVao->ebo, dstVao->ebo);
    duplicateBufferInto(srcVao->sbo, dstVao->sbo);
}

void TreeRenderer::addStorageBufferTo(VaoPackPtr vao, const void *data, std::size_t dataBytes, GLenum mode)
{
#ifndef NDEBUG
    Info << "Creating storage buffer on the GPU (" << dataBytes << " bytes)..." << std::endl;
#endif // NDEBUG
    // Cleanup the old storage buffer.
    if (vao->sbo)
    { glDeleteBuffers(1u, &vao->sbo); vao->sbo = 0u; }

    // Create the buffer and fill it with data.
    glGenBuffers(1u, &vao->sbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vao->sbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, dataBytes, data, mode);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0u);
}

TreeRenderer::VaoPackPtr TreeRenderer::loadMesh(const std::string &path, const std::string &alias)
{
    // Check if we are using an alias.
    auto name{ path };
    if (!alias.empty())
    { name = alias; }

    // Check if we loaded the mesh before.
    auto vao{ getVao(name) };

    if (!vao)
    { // Not loaded yet -> initialize.
        vao = createGetVao(name);
        loadObjInto(vao, path, GL_TRIANGLES);
    }

    return vao;
}

TreeRenderer::VaoPackPtr TreeRenderer::duplicateMesh(VaoPackPtr mesh, const std::string &name)
{
    // Check if we duplicated the mesh before.
    auto vao{ getVao(name) };

    if (!vao)
    { // Not created yet -> initialize.
        vao = createGetVao(name);
        duplicateMeshInto(mesh, vao);
    }

    return vao;
}

TreeRenderer::MeshInstancePtr TreeRenderer::loadMeshToInstance(const std::string &name, const std::string &path)
{
    auto instance{ getInstance(name) };

    if (!instance)
    { // Not created yet -> initialize.
        // Load the mesh and create its insatnce.
        auto vao{ loadMesh(path) };
        instance = createInstance(name, vao);
    }

    return instance;
}

TreeRenderer::MeshInstancePtr TreeRenderer::createInstance(const std::string &name, VaoPackPtr mesh)
{
    auto instance{ getInstance(name) };

    if (!instance)
    { // Not created yet -> initialize.
        // Create the instance.
        instance = MeshInstance::instantiate();

        instance->name = name;
        instance->mesh = mesh;
        instance->model = glm::mat4(1.0f);
        instance->scale = 1.0f;
        instance->translate = { 0.0f, 0.0f, 0.0f };
        instance->rotate = { 0.0f, 0.0f, 0.0f };
        instance->wireframe = false;
        instance->show = true;
        instance->overrideColor = { 0.0f, 0.0f, 0.0f, 0.0f };
        instance->overrideColorEnabled = false;
        instance->overrideColorStorageEnabled = false;

        mInstances.emplace(name, instance);
    }

    return instance;
}

TreeRenderer::MeshInstancePtr TreeRenderer::getInstance(const std::string &name)
{
    const auto findIt{ mInstances.find(name) };
    if (findIt != mInstances.end())
    { return findIt->second; }
    else
    { return { }; }
}

std::map<std::string, TreeRenderer::MeshInstancePtr> TreeRenderer::iterableInstances() const
{ return mInstances; }

TreeRenderer::TexturePtr TreeRenderer::loadTexture(const std::string &name, const std::string &path)
{
    auto texture{ getTexture(name) };

    if (!texture)
    { // Not created yet -> initialize.
        texture = Texture::instantiate();
        loadTextureInto(texture, path);
        mTextures.emplace(name, texture);
    }

    return texture;
}

TreeRenderer::TexturePtr TreeRenderer::loadTexture(const std::string &path)
{ return loadTexture(path, path); }

TreeRenderer::TexturePtr TreeRenderer::getTexture(const std::string &name)
{
    const auto findIt{ mTextures.find(name) };
    if (findIt != mTextures.end())
    { return findIt->second; }
    else
    { return { }; }
}

void TreeRenderer::loadTextureInto(TexturePtr texture, const std::string &path)
{
    treeio::ImageImporter importer{ };
    bool loadedProperly{ importer.importImage(path) };

    texture->buffer = TextureBuffer::instantiate(importer);
    texture->buffer->upload();

    if (loadedProperly)
    {
        Info << "File \"" << path.c_str() << "\" loaded with texture id: "
             << texture->buffer->id() << "(" << texture->buffer->width() << "x"
             << texture->buffer->height() << ")" << std::endl;
    }
    else
    {
        Error << "File \"" << path.c_str() << "\" failed loading! Generated dummy with texture id: "
              << texture->buffer->id() << "(" << texture->buffer->width() << "x"
              << texture->buffer->height() << ")" << std::endl;
    }
}

TreeRenderer::BufferBundlePtr TreeRenderer::createGetFrameBuffer(const std::string &name)
{
    auto frameBuffer{ getFrameBuffer(name) };

    if (!frameBuffer)
    { // Not created yet -> initialize.
        frameBuffer = FrameBufferObject::instantiate();
        mFrameBuffers.emplace(name, frameBuffer);
    }

    return frameBuffer;
}

TreeRenderer::BufferBundlePtr TreeRenderer::getFrameBuffer(const std::string &name)
{
    const auto findIt{ mFrameBuffers.find(name) };
    if (findIt != mFrameBuffers.end())
    { return findIt->second; }
    else
    { return { }; }
}

void TreeRenderer::useRenderer(const std::string &name)
{
    const auto findIt{ mRenderers.find(name) };
    if (findIt == mRenderers.end())
    { Error << "Failed to find specified renderer: " << name << std::endl; }
    else
    { mActiveRenderer = findIt->second; }
}

TreeRenderer::LightPtr TreeRenderer::createGetLight(const std::string &name)
{
    auto light{ getLight(name) };

    if (!light)
    { // Not created yet -> initialize.
        light = Light::instantiate();
        mLights.emplace(name, light);
    }

    return light;
}

TreeRenderer::LightPtr TreeRenderer::getLight(const std::string &name)
{
    const auto findIt{ mLights.find(name) };
    if (findIt != mLights.end())
    { return findIt->second; }
    else
    { return { }; }
}

bool TreeRenderer::attachLight(const MeshInstancePtr &instance, const LightPtr &light)
{
    if (!instance || !light)
    { return false; }

    instance->attachedLight = light;

    return true;
}

RenderSystem::Ptr TreeRenderer::activeRenderer()
{ return mActiveRenderer; }

void TreeRenderer::reloadShaders()
{
    if (mActiveRenderer)
    { mActiveRenderer->reloadShaders(); }
}

void TreeRenderer::render(treescene::CameraState &camera, treescene::TreeScene &scene)
{
    if (mActiveRenderer)
    { mActiveRenderer->render(camera, scene); }
}

bool TreeRenderer::screenshot(treescene::CameraState &camera, treescene::TreeScene &scene,
    const std::string &path, std::size_t xStart, std::size_t yStart,
    std::size_t xEnd, std::size_t yEnd, bool addTimestamp)
{
    // Re-calculate requested size:
    const auto xOffset{ xStart };
    const auto yOffset{ yStart };
    const auto width{
        xEnd ?
        xEnd - xOffset :
        camera.viewportWidth - xOffset
    };
    const auto height{
        yEnd ?
        yEnd - yOffset :
        camera.viewportHeight - yOffset
    };

    // Create output file-name:
    const auto basePath{ treeutil::filePath(path) };
    const auto fileBaseName{ treeutil::fileBaseName(path) };
    const auto extension{ treeutil::fileExtension(path) };
    const auto timestampStr{
        addTimestamp ?
        std::string{ "_" } + treeutil::strTimestamp(std::chrono::system_clock::now(), "_", "__") :
        std::string{ "" }
    };
    const auto resultFilePath{ basePath +
                               (basePath.length() > 0u ? treeutil::sysSepStr() : "") +
                               fileBaseName + timestampStr + extension
    };

    // Recover pixels from the current frame-buffer:
    std::vector<GLubyte> pixels{ };
    pixels.resize(3u * width * height);
    glReadPixels(
        static_cast<GLint>(xOffset), static_cast<GLint>(yOffset),
        static_cast<GLsizei>(width), static_cast<GLsizei>(height),
        GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

    // Save the image:
    if(basePath.length() > 0)
        std::filesystem::create_directory(basePath);
    treeio::ImageExporter exporter{ };
    exporter.loadImage(pixels, width, height);
    if (exporter.exportImage(resultFilePath))
    { return false; }
    else
    { Error << "Exception thrown while saving file \"" << path.c_str() << "\"" << std::endl; return true; }
}

bool TreeRenderer::renderScreenshot(treescene::CameraState &camera, treescene::TreeScene &scene,
    const std::string &path, std::size_t xPixels, std::size_t yPixels,
    std::size_t samples, bool addTimestamp, bool renderUi)
{
    if (!mActiveRenderer)
    { return false; }

    // Backup values for later restoration to original state.
    const auto frameBufferOverrideBackup{ mActiveRenderer->parameters().outputFrameBufferOverride };
    const auto originalViewportWidth{ camera.viewportWidth };
    const auto originalViewportHeight{ camera.viewportHeight };
    const auto originalDisplayUi{ camera.displayUi };

    if (xPixels != 0u || yPixels != 0u)
    { // Not using the default frame-buffer -> Setup override.
        mActiveRenderer->parameters().outputFrameBufferOverride =
            FrameBuffer::createMultiSampleFrameBuffer(xPixels, yPixels, true, true, false, samples);
        camera.viewportWidth = xPixels;
        camera.viewportHeight = yPixels;
    }
    // Else -> Use the already setup frame-buffer or the prepared override.

    // Setup the camera settings.
    camera.displayUi = renderUi;

    // Render the scene.
    render(camera, scene);

    // Remember which frame-buffer we used for rendering.
    auto screenshotFrameBuffer{ mActiveRenderer->parameters().outputFrameBufferOverride };

    // Bind requested frame-buffer or just use the default one.
    if (screenshotFrameBuffer)
    {
        if (samples > 1u)
        { // In case of multi-sampling, we need to copy the output into non-multi-sampled FB first.
            const auto originalOutputFrameBuffer{ screenshotFrameBuffer };
            screenshotFrameBuffer = FrameBuffer::createFrameBuffer(xPixels, yPixels, true, true, false);
            FrameBuffer::blitFrameBuffers(
                *originalOutputFrameBuffer, *screenshotFrameBuffer,
                0u, 0u, xPixels, yPixels, 0u, 0u, xPixels, yPixels,
                { FrameBuffer::AttachmentType::Color }, TextureBuffer::Filter::Linear
            );
        }
        screenshotFrameBuffer->bind();
    }
    else
    { FrameBuffer::bindDefault(); }

    // Save the screenshot.
    screenshot(camera, scene, path, 0u, 0u, 0u, 0u, addTimestamp);

    // Unbind target frame-buffer.
    FrameBuffer::unbindAll();

    // Return to the original state.
    camera.displayUi = originalDisplayUi;
    camera.viewportWidth = originalViewportWidth;
    camera.viewportHeight = originalViewportHeight;
    mActiveRenderer->parameters().outputFrameBufferOverride = frameBufferOverrideBackup;

    return true;
}

glm::mat4 TreeRenderer::calculateModelMatrix(const MeshInstance &instance, const treescene::CameraState &camera)
{
    // No re-calculation:
    if(instance.overrideModelMatrix)
    { return instance.model; }

    const auto correctiveScaling{
        (instance.antiProjectiveScaling ?
         camera.distanceScaleCorrection(instance.translate) :
         1.0f
        )
    };
    const auto scale{
        glm::scale(glm::mat4(1.0f),
            glm::vec3{ correctiveScaling, correctiveScaling, correctiveScaling }
        )
    };
    const auto model{
        calculateModelMatrixNoAntiProjection(instance) * scale
    };

    return model;
}

glm::mat4 TreeRenderer::calculateModelMatrixNoAntiProjection(const MeshInstance &instance)
{
    // No re-calculation:
    if(instance.overrideModelMatrix)
    { return instance.model; }

    const auto translate{
        glm::translate(glm::mat4(1.0f),
            instance.translate
        )
    };
    const auto rotate{
        glm::eulerAngleXYZ(
            instance.rotate.x, instance.rotate.y, instance.rotate.z
        )
    };
    const auto scale{
        glm::scale(glm::mat4(1.0f),
            glm::vec3{ instance.scale, instance.scale, instance.scale }
        )
    };
    const auto model{
        translate * rotate * scale
    };

    return model;
}

std::vector<TreeRenderer::MeshInstancePtr> TreeRenderer::sortInstancesForRendering(bool includeInvisible)
{ return sortInstancesForRendering(mInstances, includeInvisible); }

void TreeRenderer::printGlInfo()
{
    Info << "Vendor: "   << treeutil::sanitizeGlString(glGetString(GL_VENDOR))                   << std::endl;
    Info << "Renderer: " << treeutil::sanitizeGlString(glGetString(GL_RENDERER))                 << std::endl;
    Info << "Version: "  << treeutil::sanitizeGlString(glGetString(GL_VERSION))                  << std::endl;
    Info << "GLSL Ver: " << treeutil::sanitizeGlString(glGetString(GL_SHADING_LANGUAGE_VERSION)) << std::endl;
}

void GLAPIENTRY debugGlMessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity,
    GLsizei length, const GLchar* message, const GLvoid* userParam)
{
    TREE_UNUSED(source);
    TREE_UNUSED(id);
    TREE_UNUSED(length);
    TREE_UNUSED(userParam);

    if (type == GL_DEBUG_TYPE_OTHER)
    { return; }

    fprintf(stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
        (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""), type, severity, message);
}

void TreeRenderer::setupDebugReporting()
{
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
#ifdef _WIN32
    // TODO - Crashes on Windows...
    //return;
    // TODO - Change function definition?
    glDebugMessageCallback(reinterpret_cast<GLDEBUGPROC>(debugGlMessageCallback), nullptr);
#else // _WIN32
    glDebugMessageCallback(debugGlMessageCallback, nullptr);
#endif // _WIN32
    //glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_ERROR, GL_DEBUG_SEVERITY_HIGH, 0, nullptr, GL_TRUE);
}

void TreeRenderer::duplicateBufferInto(GLuint source, GLuint &destination)
{
    // Check we got a valid buffer.
    if (!source)
    { return; }

    // If destination already contains a buffer, free it.
    if (destination)
    { glDeleteBuffers(1u, &destination); destination = 0u; }

    // Generate new target buffer.
    glGenBuffers(1u, &destination);

    // Bind the source buffer for copying then get its size and usage.
    glBindBuffer(GL_COPY_READ_BUFFER, source);
    GLint sourceSize{ 0u };
    glGetBufferParameteriv(GL_COPY_READ_BUFFER, GL_BUFFER_SIZE, &sourceSize);
    GLint sourceUsage{ 0u };
    glGetBufferParameteriv(GL_COPY_READ_BUFFER, GL_BUFFER_USAGE, &sourceUsage);

    // Allocate the new buffer to be the same size as source.
    glBindBuffer(GL_COPY_WRITE_BUFFER, destination);
    glBufferData(GL_COPY_WRITE_BUFFER, sourceSize, nullptr, sourceUsage);

    // Copy data from source buffer to the destination buffer.
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, sourceSize);

    // Cleanup binds.
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0u);
    glBindBuffer(GL_COPY_READ_BUFFER, 0u);
}

std::vector<TreeRenderer::MeshInstancePtr> TreeRenderer::sortInstancesForRendering(
    const InstanceMap &instances, bool includeInvisible)
{
    // TODO - Cache this vector? For now the instance count is very low...

    std::vector<TreeRenderer::MeshInstancePtr> sorted{ };

    // Make list of all renderable instances.
    for (const auto &instanceRecord : instances)
    {
        const auto &instancePtr{ instanceRecord.second };
        const auto &instance{ *instancePtr };

        if (instance.show || includeInvisible)
        { sorted.emplace_back(instancePtr); }
    }

    // Sort instances based on their rendering order.
    std::sort(sorted.begin(), sorted.end(), [] (const auto &instance1, const auto &instance2)
    {
        // Sorting by multiple keys in order of priority - orderPriority, alwaysVisible, transparent.
        return instance1->orderPriority < instance2->orderPriority ||
               (instance1->orderPriority == instance2->orderPriority &&
                instance1->alwaysVisible < instance2->alwaysVisible) ||
               (instance1->orderPriority == instance2->orderPriority &&
                instance1->alwaysVisible == instance2->alwaysVisible &&
                instance1->transparent < instance2->transparent);
    });

    return sorted;
}

TreeRenderer::RendererPtr TreeRenderer::getRenderer(const std::string &name)
{
    const auto findIt{ mRenderers.find(name) };
    if (findIt != mRenderers.end())
    { return findIt->second; }
    else
    { return { }; }
}

} // namespace treerndr
