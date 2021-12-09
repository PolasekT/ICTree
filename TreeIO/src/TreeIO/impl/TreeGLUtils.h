/**
 * @author Tomas Polasek, David Hrusa
 * @date 2.26.2020
 * @version 1.0
 * @brief Utilities which require the inclusion of OpenGL. Exists to keep TreeUtils OpenGL independent.
 */

#ifndef TREE_GL_UTILS_H
#define TREE_GL_UTILS_H

#include <iostream>
#include <vector>
#include <utility>

#include <GL/glew.h>
#include <GL/freeglut.h>

#ifdef _WIN32
#   include <windows.h>
#   include <GL/GL.h>
#else
#   include <GL/gl.h>
#endif

#ifdef IO_USE_EGL
#   include <EGL/egl.h>
#   include <EGL/eglext.h>
// Undefine some macros with potential overlap.
#   undef Bool
#   undef CursorShape
#   undef Expose
#   undef KeyPress
#   undef KeyRelease
#   undef FocusIn
#   undef FocusOut
#   undef FontChange
#   undef None
#   undef Status
#   undef Unsorted
#else // !IO_USE_EGL
#endif // !IO_USE_EGL

#include <glm/glm.hpp>

#include "TreeUtils.h"

// Forward declaration:
namespace treerndr
{ class Buffer; class FrameBuffer; class TextureBuffer; }
namespace treeutil
{ struct Color; }

namespace treeutil
{

/// @brief provides a text representation of an OpenGL error code.
const char *getGLErrorStr(GLenum err);

/// @brief prints a message if an OpenGL Error has occurred. Compile with DISABLE_DEBUG to disable this function.
void checkGLError(const char *note);

/// @brief Make sure the input string is valid, else return a default string.
const GLubyte *sanitizeGlString(const GLubyte *str);

/// @brief Make sure the input string is valid, else return a default string.
const char *sanitizeGlString(const char *str);

/// @brief Convert template type into OpenGL enumeration of corresponding value.
template <typename T>
constexpr GLenum typeToGLType();

/// @brief Depth element with 16 bits.
struct Depth16
{
    /// Depth channel value.
    uint16_t depth{ };
}; // struct Depth16

/// @brief Depth element with 32 bits.
struct Depth32
{
    /// Depth channel value.
    uint32_t depth{ };
}; // struct Depth32

/// @brief Depth element with 32 floating point bits.
struct Depth32f
{
    /// Depth channel value.
    float depth{ };
}; // struct Depth32f

/// @brief Depth/stencil with 24/8 bits.
struct Depth24Stencil8
{
    /// 24 bits of depth and 8 bits of stencil values.
    uint32_t depthStencil{ };
}; // struct Depth24Stencil8

/// @brief Stencil element with 8 bits.
struct Stencil8
{
    /// 8 bits of stencil value.
    uint8_t stencil{ };
}; // struct Stencil8

/// @brief OpenGL type tokens.
namespace glt
{

// Bool types:
struct boolT { using type = bool; };
struct bvec2 { };
struct bvec2T { using type = glm::bvec2; };
struct bvec3 { };
struct bvec3T { using type = glm::bvec3; };
struct bvec4 { };
struct bvec4T { using type = glm::bvec4; };

// Float types:
struct floatT { using type = float; };
struct vec2 { };
struct vec2T { using type = glm::vec2; };
struct vec3 { };
struct vec3T { using type = glm::vec3; };
struct vec4 { };
struct vec4T { using type = glm::vec4; };

// Integer types:
struct intT { using type = int; };
struct ivec2 { };
struct ivec2T { using type = glm::ivec2; };
struct ivec3 { };
struct ivec3T { using type = glm::ivec3; };
struct ivec4 { };
struct ivec4T { using type = glm::ivec4; };

// Matrix types:
struct mat4 { };
struct mat4T { using type = glm::mat4; };

// Sampler types:
struct sampler2D { };
struct sampler2DT { using type = treeutil::WrapperPtrT<treerndr::TextureBuffer>; };

// Buffer types:
struct buffer { };
struct bufferT { using type = treeutil::WrapperPtrT<treerndr::Buffer>; };

// Array types:
template <typename ElementT, std::size_t Size>
struct array { };
template <typename ElementT, std::size_t Size>
struct arrayT
{ using type = std::array<ElementT, Size>; };

} // namespace glt

/**
 * @brief Convert OpenGL type (i.e. mat4, vec3, ...) into corresponding C++ type.
 * @usage:
 *  treeutil::glTypeToType<treeutil::glt::mat4>::type myMat4{ };
 */
template <typename GLT>
struct glTypeToType;

/**
 * @brief Helper used for compile-time detection of whether given OpenGL type is a texture.
 * @usage:
 *  treeutil::glTypeIsTexture<treeutil::glt::sampler2D>::value // == true
 *  treeutil::glTypeIsTexture<treeutil::glt::mat4>::value // == false
 */
template <typename GLT>
struct glTypeIsTexture;

/**
 * @brief Helper used for compile-time detection of whether given C++ type is a texture.
 * @usage:
 *  treeutil::typeIsTexture<treerndr::TextureBuffer>::value // == true
 *  treeutil::typeIsTexture<std::size_t>::value // == false
 */
template <typename T>
struct typeIsTexture;

/**
 * @brief Helper used for compile-time detection of whether given OpenGL type is a buffer.
 * @usage:
 *  treeutil::typeIsTexture<treeutil::glt::buffer>::value // == true
 *  treeutil::typeIsTexture<treeutil::glt::sampler2D>::value // == false
 */
template <typename T>
struct glTypeIsBuffer;

/**
 * @brief Helper used for compile-time detection of whether given C++ type is a buffer.
 * @usage:
 *  treeutil::typeIsTexture<treerndr::Buffer>::value // == true
 *  treeutil::typeIsTexture<treerndr::TextureBuffer>::value // == false
 */
template <typename T>
struct typeIsBuffer;

/// @brief Convert template type into corresponding OpenGL texture format.
template <typename T>
constexpr GLenum typeToGLTextureFormat();
/// @brief Convert template type into corresponding OpenGL texture type.
template <typename T>
constexpr GLenum typeToGLTextureType();
/// @brief Convert template type into corresponding OpenGL texture lower precision internal format.
template <typename T>
constexpr GLenum typeToGLTextureInternalFormatLP(bool snorm = false);
/// @brief Convert template type into corresponding OpenGL texture higher precision internal format.
template <typename T>
constexpr GLenum typeToGLTextureInternalFormatHP(bool snorm = false);

/// @brief Convert OpenGL shader type enumeration into readable string.
std::string shaderTypeToStr(GLenum shader);
/// @brief Convert OpenGL type enumeration into readable string.
std::string glTypeToStr(GLenum type);
/// @brief Convert OpenGL texture format enumeration into readable string.
std::string glTextureFormatToStr(GLenum type);
/// @brief Convert OpenGL texture type enumeration into readable string.
std::string glTextureTypeToStr(GLenum type);
/// @brief Convert OpenGL texture internal format enumeration into readable string.
std::string glTextureInternalFormatToStr(GLenum type);

/// @brief Enumeration of texture filtering options.
enum class TextureFilter
{
    /// Returns value nearest to the sample point.
    Nearest,
    /// Returns weighed average based on sample point neighborhood.
    Linear,
}; // enum class TextureFilter

/// @brief Enumeration of mip-map texture filtering options.
enum class TextureMipFilter
{
    /// Do not use mip-maps.
    None,
    /// Uses the nearest mip-map to the sampled point.
    Nearest,
    /// Returns weighed average based on mip-maps layers nearest to the sampled point.
    Linear,
}; // enum class TextureMipFilter

/// @brief Convert enumeration into GLEnum.
GLenum textureFilterToGLEnum(TextureFilter filter, TextureMipFilter mipFilter = TextureMipFilter::None);
/// @brief Convert enumeration into readable string.
std::string textureFilterToStr(TextureFilter filter, TextureMipFilter mipFilter = TextureMipFilter::None);

/// @brief Enumeration of mip-map texture wrapping options.
enum class TextureWrap
{
    /// Wrap coordinates, repeating the input.
    Repeat,
    /// Set value to the pre-set border value.
    ClampBorder,
    /// Set value to the nearest real pixel value from the image.
    ClampEdge,
    /// Same as repeat, but the images are mirrored.
    MirrorRepeat,
    /// Same as clamp to edge, but the other side of the image is used.
    MirrorClampEdge
}; // enum class TextureWrap

/// @brief Convert enumeration into GLEnum.
GLenum textureWrapToGLEnum(TextureWrap wrap);
/// @brief Convert enumeration into readable string.
std::string textureWrapToStr(TextureWrap wrap);

/// @brief Enumeration of buffer targets.
enum class BufferTarget
{
    /// Vertex attributes.
    ArrayBuffer,
    /// Atomic counters.
    AtomicCounter,
    /// Buffer copy source.
    CopyRead,
    /// Buffer copy destination.
    CopyWrite,
    /// Indirect compute dispatch commands.
    DispatchIndirect,
    /// Indirect command arguments.
    DrawIndirect,
    /// Vertex array indices.
    ElementArray,
    /// Pixel read target.
    PixelPack,
    /// Texture data source.
    PixelUnpack,
    /// Query result buffer.
    Query,
    /// Read-write storage for buffers.
    ShaderStorage,
    /// Texture data buffer.
    Texture,
    /// Transform feedback buffer.
    TransformFeedback,
    /// Uniform block storage.
    Uniform
}; // enum class BufferTarget

/// @brief Convert enumeration into GLEnum.
GLenum bufferTargetToGLEnum(BufferTarget target);
/// @brief Convert enumeration into readable string.
std::string bufferTargetToStr(BufferTarget target);

/// @brief Enumeration of buffer usage frequencies.
enum class BufferUsageFrequency
{
    /// Modified once and used at most a few times.
    Stream,
    /// Modified once and used many times.
    Static,
    /// Modified repeatedly and used many times.
    Dynamic
}; // enum class BufferUsageFrequency

/// @brief Enumeration of buffer usage access types.
enum class BufferUsageAccess
{
    /// Modified by application and used by OpenGL drawing and image operations.
    Draw,
    /// Modified by OpenGL and used by application.
    Read,
    /// Modified by OpenGL and used by OpenGL
    Copy
}; // enum class BufferUsageAccess

/// @brief Convert enumeration into GLEnum.
GLenum bufferUsageToGLEnum(BufferUsageFrequency frequency, BufferUsageAccess access);
/// @brief Convert enumeration into readable string.
std::string bufferUsageToStr(BufferUsageFrequency frequency, BufferUsageAccess access);

/// @brief Enumeration of frame-buffer attachment types.
enum class FrameBufferAttachmentType
{
    /// Color render target. Each frame-buffer may have multiple of these.
    Color,
    /// Depth render target
    Depth,
    /// Stencil render target
    Stencil,
    /// Combination of depth and stencil render target.
    DepthStencil
}; // enum class FrameBufferAttachmentType

/// @brief Convert enumeration into GLEnum.
GLenum frameBufferAttachmentToGLEnum(FrameBufferAttachmentType type);
/// @brief Convert enumeration into readable string.
std::string frameBufferAttachmentToStr(FrameBufferAttachmentType type);

/// @brief Enumeration of frame-buffer target.
enum class FrameBufferTarget
{
    /// Standard frame-buffer to be used for rendering or reading.
    FrameBuffer,
    /// Frame-buffer may only be used for reading.
    ReadFrameBuffer,
    /// Frame-buffer may only be used as render target.
    DrawFrameBuffer
}; // enum class FrameBufferTarget

/// @brief Convert enumeration into GLEnum.
GLenum frameBufferTargetToGLEnum(FrameBufferTarget target);
/// @brief Convert enumeration into readable string.
std::string frameBufferTargetToStr(FrameBufferTarget target);

/// @brief Check framebuffer bound to given target for errors. Returns true if no errors occurred. Optionally returns string message.
bool checkFrameBufferStatus(FrameBufferTarget target, std::string *str = nullptr);

/**
 * @brief Helper to efficiently insert triangles into vertex and element buffers.
 * Attempts to index already existing vertices to minimize obj file size, Verts closer than epsilon are merged:
 *
 * @param vertices Vector used for storing the vertices.
 * @param indices Vector used for storing the indices.
 * @param v0 Vec4 corner of the triangle.
 * @param v1 Vec4 corner of the triangle.
 * @param v2 Vec4 corner of the triangle.
 * @param epsilon Minimum difference in coordinates to merge two points.
 * @param window Maximum number of points checked from the end of vertices vector. Duplicate
 *  points are usually close to each other topologically coming from a geometry shader so the
 *  likelihood of 100% optimization is high.
*/
void insertTriangle(std::vector<glm::vec4> &vertices, std::vector<GLint> &indices,
    const glm::vec4 &v0, const glm::vec4 &v1, const glm::vec4 &v2,
    float epsilon = std::numeric_limits<float>::epsilon(),
    std::size_t window = 100u);

/// @brief Print information about shader compilation error.
void printShaderCompileError(std::ostream &out, GLuint shader);

/// @brief Print annotated shader source code.
void printAnnotatedShaderSource(std::ostream &out, const std::string &name, GLenum type, const std::string &source);

/// @brief Print information about shader program linking error.
void printProgramLinkError(std::ostream &out, GLuint program);

#ifdef IO_USE_EGL
/// @brief Print information about all currently available EGL devices.
void printEglDeviceInfo();
#endif // IO_USE_EGL

} // namespace treeutil

// Helpers for writing out glm types:
// Float types:
inline std::ostream &operator<<(std::ostream &stream, const glm::vec2 &vec);
inline std::ostream &operator<<(std::ostream &stream, const glm::vec3 &vec);
inline std::ostream &operator<<(std::ostream &stream, const glm::vec4 &vec);

// Int types:
inline std::ostream &operator<<(std::ostream &stream, const glm::ivec2 &vec);
inline std::ostream &operator<<(std::ostream &stream, const glm::ivec3 &vec);
inline std::ostream &operator<<(std::ostream &stream, const glm::ivec4 &vec);

// Matrix types:
inline std::ostream &operator<<(std::ostream &stream, const glm::mat4 &mat);

namespace treeutil
{

namespace impl
{

template <typename T>
constexpr auto typeToGLType(const std::vector<T>&) -> T;
template <typename T, std::size_t Size>
constexpr auto typeToGLType(const std::array<T, Size>&) -> T;

} // namespace impl

// Bool types:
template <> constexpr GLenum typeToGLType<bool>() { return GL_BOOL; }
template <> constexpr GLenum typeToGLType<glm::bvec1>() { return GL_BOOL; }
template <> constexpr GLenum typeToGLType<glm::bvec2>() { return GL_BOOL_VEC2; }
template <> constexpr GLenum typeToGLType<glm::bvec3>() { return GL_BOOL_VEC3; }
template <> constexpr GLenum typeToGLType<glm::bvec4>() { return GL_BOOL_VEC4; }

// Float types:
template <> constexpr GLenum typeToGLType<float>() { return GL_FLOAT; }
template <> constexpr GLenum typeToGLType<glm::vec1>() { return GL_FLOAT; }
template <> constexpr GLenum typeToGLType<Vector2D>() { return GL_FLOAT_VEC2; }
template <> constexpr GLenum typeToGLType<glm::vec2>() { return GL_FLOAT_VEC2; }
template <> constexpr GLenum typeToGLType<Vector3D>() { return GL_FLOAT_VEC3; }
template <> constexpr GLenum typeToGLType<glm::vec3>() { return GL_FLOAT_VEC3; }
template <> constexpr GLenum typeToGLType<glm::vec4>() { return GL_FLOAT_VEC4; }

// Integer types:
template <> constexpr GLenum typeToGLType<int>() { return GL_INT; }
template <> constexpr GLenum typeToGLType<uint32_t>() { return GL_UNSIGNED_INT; }
template <> constexpr GLenum typeToGLType<glm::ivec1>() { return GL_INT; }
template <> constexpr GLenum typeToGLType<glm::ivec2>() { return GL_INT_VEC2; }
template <> constexpr GLenum typeToGLType<glm::ivec3>() { return GL_INT_VEC3; }
template <> constexpr GLenum typeToGLType<glm::ivec4>() { return GL_INT_VEC4; }

// Matrix types:
template <> constexpr GLenum typeToGLType<glm::mat4>() { return GL_FLOAT_MAT4; }

// Sampler types:
template <> constexpr GLenum typeToGLType<treerndr::TextureBuffer>() { return GL_SAMPLER_2D; }
template <> constexpr GLenum typeToGLType<glt::sampler2DT::type>() { return GL_SAMPLER_2D; }

// Buffer types:
template <> constexpr GLenum typeToGLType<treerndr::Buffer>() { return GL_BUFFER; }
template <> constexpr GLenum typeToGLType<glt::bufferT::type>() { return GL_BUFFER; }

// Array types:
template <typename T> constexpr GLenum typeToGLType()
{ return typeToGLType<typename T::value_type>(); }

// Bool types:
template <> struct glTypeToType<bool> { using type = glt::boolT::type; };
template <> struct glTypeToType<glt::bvec2> { using type = glt::bvec2T::type; };
template <> struct glTypeToType<glt::bvec3> { using type = glt::bvec3T::type; };
template <> struct glTypeToType<glt::bvec4> { using type = glt::bvec4T::type; };

// Float types:
template <> struct glTypeToType<float> { using type = glt::floatT::type; };
template <> struct glTypeToType<glt::vec2> { using type = glt::vec2T::type; };
template <> struct glTypeToType<glt::vec3> { using type = glt::vec3T::type; };
template <> struct glTypeToType<glt::vec4> { using type = glt::vec4T::type; };

// Integer types:
template <> struct glTypeToType<int> { using type = int; };
template <> struct glTypeToType<glt::ivec2> { using type = glt::ivec2T::type; };
template <> struct glTypeToType<glt::ivec3> { using type = glt::ivec3T::type; };
template <> struct glTypeToType<glt::ivec4> { using type = glt::ivec4T::type; };

// Matrix types:
template <> struct glTypeToType<glt::mat4> { using type = glt::mat4T::type; };

// Sampler types:
template <> struct glTypeToType<glt::sampler2D> { using type = glt::sampler2DT::type; };

// Buffer types:
template <> struct glTypeToType<glt::buffer> { using type = glt::bufferT::type; };

// Array types:
template <typename ElementT, std::size_t Size>
struct glTypeToType<glt::array<ElementT, Size>> { using type = typename glt::arrayT<ElementT, Size>::type; };

// Default case is false:
template <typename GLT> struct glTypeIsTexture { static constexpr auto value{ false }; };
// Sampler types:
template <> struct glTypeIsTexture<glt::sampler2D> { static constexpr auto value{ true }; };

// Default case is false:
template <typename T> struct typeIsTexture { static constexpr auto value{ false }; };
// Sampler types:
template <> struct typeIsTexture<treerndr::TextureBuffer> { static constexpr auto value{ true }; };
template <> struct typeIsTexture<treeutil::WrapperPtrT<treerndr::TextureBuffer>> { static constexpr auto value{ true }; };

// Default case is false:
template <typename T> struct glTypeIsBuffer { static constexpr auto value{ false }; };
// Sampler types:
template <> struct glTypeIsBuffer<treeutil::glt::buffer> { static constexpr auto value{ true }; };

// Default case is false:
template <typename T> struct typeIsBuffer { static constexpr auto value{ false }; };
// Sampler types:
template <> struct typeIsBuffer<treerndr::Buffer> { static constexpr auto value{ true }; };
template <> struct typeIsBuffer<treeutil::WrapperPtrT<treerndr::Buffer>> { static constexpr auto value{ true }; };

// Float types:
template <> constexpr GLenum typeToGLTextureFormat<float>() { return GL_RED; }
template <> constexpr GLenum typeToGLTextureType<float>() { return GL_FLOAT; }
template <> constexpr GLenum typeToGLTextureInternalFormatLP<float>(bool snorm) { return snorm ? GL_R8_SNORM : GL_R8; }
template <> constexpr GLenum typeToGLTextureInternalFormatHP<float>(bool snorm) { return snorm ? GL_R16_SNORM : GL_R16; }
template <> constexpr GLenum typeToGLTextureFormat<glm::vec1>() { return GL_RED; }
template <> constexpr GLenum typeToGLTextureType<glm::vec1>() { return GL_FLOAT; }
template <> constexpr GLenum typeToGLTextureInternalFormatLP<glm::vec1>(bool snorm) { return snorm ? GL_R8_SNORM : GL_R8; }
template <> constexpr GLenum typeToGLTextureInternalFormatHP<glm::vec1>(bool snorm) { return snorm ? GL_R16_SNORM : GL_R16; }
template <> constexpr GLenum typeToGLTextureFormat<Vector2D>() { return GL_RG; }
template <> constexpr GLenum typeToGLTextureType<Vector2D>() { return GL_FLOAT; }
template <> constexpr GLenum typeToGLTextureInternalFormatLP<Vector2D>(bool snorm) { return snorm ? GL_RG8_SNORM : GL_RG8; }
template <> constexpr GLenum typeToGLTextureInternalFormatHP<Vector2D>(bool snorm) { return snorm ? GL_RG16_SNORM : GL_RG16; }
template <> constexpr GLenum typeToGLTextureFormat<glm::vec2>() { return GL_RG; }
template <> constexpr GLenum typeToGLTextureType<glm::vec2>() { return GL_FLOAT; }
template <> constexpr GLenum typeToGLTextureInternalFormatLP<glm::vec2>(bool snorm) { return snorm ? GL_RG8_SNORM : GL_RG8; }
template <> constexpr GLenum typeToGLTextureInternalFormatHP<glm::vec2>(bool snorm) { return snorm ? GL_RG16_SNORM : GL_RG16; }
template <> constexpr GLenum typeToGLTextureFormat<Vector3D>() { return GL_RGB; }
template <> constexpr GLenum typeToGLTextureType<Vector3D>() { return GL_FLOAT; }
template <> constexpr GLenum typeToGLTextureInternalFormatLP<Vector3D>(bool snorm) { return snorm ? GL_RGB8_SNORM : GL_RGB8; }
template <> constexpr GLenum typeToGLTextureInternalFormatHP<Vector3D>(bool snorm) { return snorm ? GL_RGB16_SNORM : GL_RGB16; }
template <> constexpr GLenum typeToGLTextureFormat<glm::vec3>() { return GL_RGB; }
template <> constexpr GLenum typeToGLTextureType<glm::vec3>() { return GL_FLOAT; }
template <> constexpr GLenum typeToGLTextureInternalFormatLP<glm::vec3>(bool snorm) { return snorm ? GL_RGB8_SNORM : GL_RGB8; }
template <> constexpr GLenum typeToGLTextureInternalFormatHP<glm::vec3>(bool snorm) { return snorm ? GL_RGB16_SNORM : GL_RGB16; }
template <> constexpr GLenum typeToGLTextureFormat<Color>() { return GL_RGBA; }
template <> constexpr GLenum typeToGLTextureType<Color>() { return GL_FLOAT; }
template <> constexpr GLenum typeToGLTextureInternalFormatLP<Color>(bool snorm) { return snorm ? GL_RGBA8_SNORM : GL_RGBA8; }
template <> constexpr GLenum typeToGLTextureInternalFormatHP<Color>(bool snorm) { return snorm ? GL_RGBA16_SNORM : GL_RGBA16; }
template <> constexpr GLenum typeToGLTextureFormat<glm::vec4>() { return GL_RGBA; }
template <> constexpr GLenum typeToGLTextureType<glm::vec4>() { return GL_FLOAT; }
template <> constexpr GLenum typeToGLTextureInternalFormatLP<glm::vec4>(bool snorm) { return snorm ? GL_RGBA8_SNORM : GL_RGBA8; }
template <> constexpr GLenum typeToGLTextureInternalFormatHP<glm::vec4>(bool snorm) { return snorm ? GL_RGBA16_SNORM : GL_RGBA16; }

// Depth/stencil types:
template <> constexpr GLenum typeToGLTextureFormat<Depth16>() { return GL_DEPTH_COMPONENT; }
template <> constexpr GLenum typeToGLTextureType<Depth16>() { return GL_FLOAT; }
template <> constexpr GLenum typeToGLTextureInternalFormatLP<Depth16>(bool snorm) { return GL_DEPTH_COMPONENT; }
template <> constexpr GLenum typeToGLTextureInternalFormatHP<Depth16>(bool snorm) { return GL_DEPTH_COMPONENT16; }
template <> constexpr GLenum typeToGLTextureFormat<Depth32>() { return GL_DEPTH_COMPONENT; }
template <> constexpr GLenum typeToGLTextureType<Depth32>() { return GL_FLOAT; }
template <> constexpr GLenum typeToGLTextureInternalFormatLP<Depth32>(bool snorm) { return GL_DEPTH_COMPONENT; }
template <> constexpr GLenum typeToGLTextureInternalFormatHP<Depth32>(bool snorm) { return GL_DEPTH_COMPONENT32; }
template <> constexpr GLenum typeToGLTextureFormat<Depth32f>() { return GL_DEPTH_COMPONENT; }
template <> constexpr GLenum typeToGLTextureType<Depth32f>() { return GL_FLOAT; }
template <> constexpr GLenum typeToGLTextureInternalFormatLP<Depth32f>(bool snorm) { return GL_DEPTH_COMPONENT; }
template <> constexpr GLenum typeToGLTextureInternalFormatHP<Depth32f>(bool snorm) { return GL_DEPTH_COMPONENT32F; }
template <> constexpr GLenum typeToGLTextureFormat<Depth24Stencil8>() { return GL_DEPTH_STENCIL; }
template <> constexpr GLenum typeToGLTextureType<Depth24Stencil8>() { return GL_FLOAT; }
template <> constexpr GLenum typeToGLTextureInternalFormatLP<Depth24Stencil8>(bool snorm) { return GL_DEPTH_STENCIL; }
template <> constexpr GLenum typeToGLTextureInternalFormatHP<Depth24Stencil8>(bool snorm) { return GL_DEPTH24_STENCIL8; }
template <> constexpr GLenum typeToGLTextureFormat<Stencil8>() { return GL_STENCIL_INDEX; }
template <> constexpr GLenum typeToGLTextureType<Stencil8>() { return GL_FLOAT; }
template <> constexpr GLenum typeToGLTextureInternalFormatLP<Stencil8>(bool snorm) { return GL_STENCIL; }
template <> constexpr GLenum typeToGLTextureInternalFormatHP<Stencil8>(bool snorm) { return GL_STENCIL_INDEX8; }

// Integer types:
template <> constexpr GLenum typeToGLTextureFormat<int>() { return GL_RED_INTEGER; }
template <> constexpr GLenum typeToGLTextureType<int>() { return GL_INT; }
template <> constexpr GLenum typeToGLTextureInternalFormatLP<int>(bool snorm) { return snorm ? GL_R8_SNORM : GL_R8; }
template <> constexpr GLenum typeToGLTextureInternalFormatHP<int>(bool snorm) { return snorm ? GL_R16_SNORM : GL_R16; }
template <> constexpr GLenum typeToGLTextureFormat<uint32_t>() { return GL_RED_INTEGER; }
template <> constexpr GLenum typeToGLTextureType<uint32_t>() { return GL_UNSIGNED_INT; }
template <> constexpr GLenum typeToGLTextureInternalFormatLP<uint32_t>(bool snorm) { return snorm ? GL_R8_SNORM : GL_R8; }
template <> constexpr GLenum typeToGLTextureInternalFormatHP<uint32_t>(bool snorm) { return snorm ? GL_R16_SNORM : GL_R16; }
template <> constexpr GLenum typeToGLTextureFormat<unsigned char>() { return GL_RED_INTEGER; }
template <> constexpr GLenum typeToGLTextureType<unsigned char>() { return GL_UNSIGNED_BYTE; }
template <> constexpr GLenum typeToGLTextureInternalFormatLP<unsigned char>(bool snorm) { return snorm ? GL_R8_SNORM : GL_R8; }
template <> constexpr GLenum typeToGLTextureInternalFormatHP<unsigned char>(bool snorm) { return snorm ? GL_R16_SNORM : GL_R16; }
template <> constexpr GLenum typeToGLTextureFormat<glm::u8vec1>() { return GL_RED_INTEGER; }
template <> constexpr GLenum typeToGLTextureType<glm::u8vec1>() { return GL_UNSIGNED_BYTE; }
template <> constexpr GLenum typeToGLTextureInternalFormatLP<glm::u8vec1>(bool snorm) { return snorm ? GL_R8_SNORM : GL_R8; }
template <> constexpr GLenum typeToGLTextureInternalFormatHP<glm::u8vec1>(bool snorm) { return snorm ? GL_R16_SNORM : GL_R16; }
template <> constexpr GLenum typeToGLTextureFormat<glm::ivec1>() { return GL_RED_INTEGER; }
template <> constexpr GLenum typeToGLTextureType<glm::ivec1>() { return GL_INT; }
template <> constexpr GLenum typeToGLTextureInternalFormatLP<glm::ivec1>(bool snorm) { return snorm ? GL_R8_SNORM : GL_R8; }
template <> constexpr GLenum typeToGLTextureInternalFormatHP<glm::ivec1>(bool snorm) { return snorm ? GL_R16_SNORM : GL_R16; }
template <> constexpr GLenum typeToGLTextureFormat<glm::u8vec2>() { return GL_RG_INTEGER; }
template <> constexpr GLenum typeToGLTextureType<glm::u8vec2>() { return GL_UNSIGNED_BYTE; }
template <> constexpr GLenum typeToGLTextureInternalFormatLP<glm::u8vec2>(bool snorm) { return snorm ? GL_RG8_SNORM : GL_RG8; }
template <> constexpr GLenum typeToGLTextureInternalFormatHP<glm::u8vec2>(bool snorm) { return snorm ? GL_RG16_SNORM : GL_RG16; }
template <> constexpr GLenum typeToGLTextureFormat<glm::ivec2>() { return GL_RG_INTEGER; }
template <> constexpr GLenum typeToGLTextureType<glm::ivec2>() { return GL_INT; }
template <> constexpr GLenum typeToGLTextureInternalFormatLP<glm::ivec2>(bool snorm) { return snorm ? GL_RG8_SNORM : GL_RG8; }
template <> constexpr GLenum typeToGLTextureInternalFormatHP<glm::ivec2>(bool snorm) { return snorm ? GL_RG16_SNORM : GL_RG16; }
template <> constexpr GLenum typeToGLTextureFormat<glm::u8vec3>() { return GL_RGB_INTEGER; }
template <> constexpr GLenum typeToGLTextureType<glm::u8vec3>() { return GL_UNSIGNED_BYTE; }
template <> constexpr GLenum typeToGLTextureInternalFormatLP<glm::u8vec3>(bool snorm) { return snorm ? GL_RGB8_SNORM : GL_RGB8; }
template <> constexpr GLenum typeToGLTextureInternalFormatHP<glm::u8vec3>(bool snorm) { return snorm ? GL_RGB16_SNORM : GL_RGB16; }
template <> constexpr GLenum typeToGLTextureFormat<glm::ivec3>() { return GL_RGB_INTEGER; }
template <> constexpr GLenum typeToGLTextureType<glm::ivec3>() { return GL_INT; }
template <> constexpr GLenum typeToGLTextureInternalFormatLP<glm::ivec3>(bool snorm) { return snorm ? GL_RGB8_SNORM : GL_RGB8; }
template <> constexpr GLenum typeToGLTextureInternalFormatHP<glm::ivec3>(bool snorm) { return snorm ? GL_RGB16_SNORM : GL_RGB16; }
template <> constexpr GLenum typeToGLTextureFormat<glm::u8vec4>() { return GL_RGBA_INTEGER; }
template <> constexpr GLenum typeToGLTextureType<glm::u8vec4>() { return GL_UNSIGNED_BYTE; }
template <> constexpr GLenum typeToGLTextureInternalFormatLP<glm::u8vec4>(bool snorm) { return snorm ? GL_RGBA8_SNORM : GL_RGBA8; }
template <> constexpr GLenum typeToGLTextureInternalFormatHP<glm::u8vec4>(bool snorm) { return snorm ? GL_RGBA16_SNORM : GL_RGBA16; }
template <> constexpr GLenum typeToGLTextureFormat<glm::ivec4>() { return GL_RGBA_INTEGER; }
template <> constexpr GLenum typeToGLTextureType<glm::ivec4>() { return GL_INT; }
template <> constexpr GLenum typeToGLTextureInternalFormatLP<glm::ivec4>(bool snorm) { return snorm ? GL_RGBA8_SNORM : GL_RGBA8; }
template <> constexpr GLenum typeToGLTextureInternalFormatHP<glm::ivec4>(bool snorm) { return snorm ? GL_RGBA16_SNORM : GL_RGBA16; }

} // namespace treeutil

// Bool types:
inline std::ostream &operator<<(std::ostream &stream, const glm::bvec2 &vec)
{ stream << vec.x << "," << vec.y ; return stream; }
inline std::ostream &operator<<(std::ostream &stream, const glm::bvec3 &vec)
{ stream << vec.x << "," << vec.y << "," << vec.z; return stream; }
inline std::ostream &operator<<(std::ostream &stream, const glm::bvec4 &vec)
{ stream << vec.x << "," << vec.y << "," << vec.z << "," << vec.w; return stream; }

// Float types:
inline std::ostream &operator<<(std::ostream &stream, const glm::vec2 &vec)
{ stream << vec.x << "," << vec.y ; return stream; }
inline std::ostream &operator<<(std::ostream &stream, const glm::vec3 &vec)
{ stream << vec.x << "," << vec.y << "," << vec.z; return stream; }
inline std::ostream &operator<<(std::ostream &stream, const glm::vec4 &vec)
{ stream << vec.x << "," << vec.y << "," << vec.z << "," << vec.w; return stream; }

// Int types:
inline std::ostream &operator<<(std::ostream &stream, const glm::ivec2 &vec)
{ stream << vec.x << "," << vec.y ; return stream; }
inline std::ostream &operator<<(std::ostream &stream, const glm::ivec3 &vec)
{ stream << vec.x << "," << vec.y << "," << vec.z; return stream; }
inline std::ostream &operator<<(std::ostream &stream, const glm::ivec4 &vec)
{ stream << vec.x << "," << vec.y << "," << vec.z << "," << vec.w; return stream; }

// UInt types:
inline std::ostream &operator<<(std::ostream &stream, const glm::uvec2 &vec)
{ stream << vec.x << "," << vec.y ; return stream; }
inline std::ostream &operator<<(std::ostream &stream, const glm::uvec3 &vec)
{ stream << vec.x << "," << vec.y << "," << vec.z; return stream; }
inline std::ostream &operator<<(std::ostream &stream, const glm::uvec4 &vec)
{ stream << vec.x << "," << vec.y << "," << vec.z << "," << vec.w; return stream; }

// Matrix types:
inline std::ostream &operator<<(std::ostream &stream, const glm::mat4 &mat)
{
    stream << "{[" << mat[0][0] << "," << mat[0][1] << "," << mat[0][2] << "," << mat[0][3] << "],"
           << "[" << mat[1][0] << "," << mat[1][1] << "," << mat[1][2] << "," << mat[1][3] << "],"
           << "[" << mat[2][0] << "," << mat[2][1] << "," << mat[2][2] << "," << mat[2][3] << "],"
           << "[" << mat[3][0] << "," << mat[3][1] << "," << mat[3][2] << "," << mat[3][3] << "]}";
    return stream;
}

// Template implementation end.

#endif // TREE_GL_UTILS_H