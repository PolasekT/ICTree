/**
 * @author Tomas Polasek, David Hrusa
 * @date 2.26.2020
 * @version 1.0
 * @brief Utilities which require the inclusion of OpenGL. Existst to keep TreeUtils OpenGL independent.
 */

#include "TreeGLUtils.h"

namespace treeutil
{

const char *getGLErrorStr(GLenum err)
{
    switch (err)
    {
        case GL_NO_ERROR:
        { return "No error"; }
        case GL_INVALID_ENUM:
        { return "Invalid enum"; }
        case GL_INVALID_VALUE:
        { return "Invalid value"; }
        case GL_INVALID_OPERATION:
        { return "Invalid operation"; }
        case GL_STACK_OVERFLOW:
        { return "Stack overflow"; }
        case GL_STACK_UNDERFLOW:
        { return "Stack underflow"; }
        case GL_OUT_OF_MEMORY:
        { return "Out of memory"; }
        default:
        { return "Unknown error"; }
    }
}

void checkGLError(const char *note)
{
    // compile flag to disable all debugging tools:
#ifdef DISABLE_DEBUG
    return;
#endif
    while (true)
    {
        const GLenum err = glGetError();
        if (GL_NO_ERROR == err)
        { break; }

        Error << "\t@: " << note
              << "\n\tGL Error: " << getGLErrorStr(err)
              << "\n\tError Txt: " << glewGetErrorString(err)
              << std::endl;
    }
}

const GLubyte *sanitizeGlString(const GLubyte *str)
{
    static const unsigned char DEFAULT_STR[]{ 'U', 'n', 'k', 'n', 'o', 'w', 'n', '\0' };
    return str != nullptr ? str : DEFAULT_STR;
}

const char *sanitizeGlString(const char *str)
{
    static const char DEFAULT_STR[]{ 'U', 'n', 'k', 'n', 'o', 'w', 'n', '\0' };
    return str != nullptr ? str : DEFAULT_STR;
}

std::string shaderTypeToStr(GLenum shader)
{
    switch (shader)
    {
        case GL_VERTEX_SHADER:
        { return "Vertex"; }
        case GL_TESS_EVALUATION_SHADER:
        { return "Evaluation"; }
        case GL_TESS_CONTROL_SHADER:
        { return "Control"; }
        case GL_GEOMETRY_SHADER:
        { return "Geometry"; }
        case GL_FRAGMENT_SHADER:
        { return "Fragment"; }
        case GL_COMPUTE_SHADER:
        { return "Compute"; }
        default:
        { return "Unknown"; }
    }
}

std::string glTypeToStr(GLenum type)
{
    switch (type)
    {
        // Float types:
        case GL_FLOAT:
        { return "float"; }
        case GL_FLOAT_VEC2:
        { return "vec2"; }
        case GL_FLOAT_VEC3:
        { return "vec3"; }
        case GL_FLOAT_VEC4:
        { return "vec4"; }
        // Int types:
        case GL_INT:
        { return "int"; }
        case GL_INT_VEC2:
        { return "ivec2"; }
        case GL_INT_VEC3:
        { return "ivec3"; }
        case GL_INT_VEC4:
        { return "ivec4"; }
        // Matrix types:
        case GL_FLOAT_MAT4:
        { return "mat4"; }
        // Sampler types:
        case GL_SAMPLER_2D:
        { return "sampler2d"; }
        default:
        { return "unknown"; }
    }
}

std::string glTextureFormatToStr(GLenum type)
{
    switch (type)
    {
        case GL_DEPTH_COMPONENT:
        { return "Depth"; }
        case GL_DEPTH_STENCIL:
        { return "DepthStencil"; }
        case GL_STENCIL_INDEX:
        { return "StencilIndex"; }
        case GL_RED:
        { return "R"; }
        case GL_RG:
        { return "RG"; }
        case GL_RGB:
        { return "RGB"; }
        case GL_RGBA:
        { return "RGBA"; }
        case GL_RED_INTEGER:
        { return "Ri"; }
        case GL_RG_INTEGER:
        { return "RGi"; }
        case GL_RGB_INTEGER:
        { return "RGBi"; }
        case GL_RGBA_INTEGER:
        { return "RGBAi"; }
        default:
        { return "Unknown"; }
    }
}

std::string glTextureTypeToStr(GLenum type)
{ return glTypeToStr(type); }

std::string glTextureInternalFormatToStr(GLenum type)
{
    switch (type)
    {
        case GL_R8:
        { return "R8"; }
        case GL_R8_SNORM:
        { return "R8s"; }
        case GL_RG8:
        { return "RG8"; }
        case GL_RG8_SNORM:
        { return "RG8s"; }
        case GL_RGB8:
        { return "RGB8"; }
        case GL_RGB8_SNORM:
        { return "RGB8s"; }
        case GL_RGBA8:
        { return "RGBA8"; }
        case GL_RGBA8_SNORM:
        { return "RGBA8s"; }
        case GL_R16:
        { return "R16"; }
        case GL_R16_SNORM:
        { return "R16s"; }
        case GL_RG16:
        { return "RG16"; }
        case GL_RG16_SNORM:
        { return "RG16s"; }
        case GL_RGB16:
        { return "RGB16"; }
        case GL_RGB16_SNORM:
        { return "RGB16s"; }
        case GL_RGBA16:
        { return "RGBA16"; }
        case GL_RGBA16_SNORM:
        { return "RGBA16s"; }
        case GL_DEPTH_COMPONENT:
        { return "Depth"; }
        case GL_DEPTH_COMPONENT16:
        { return "Depth16"; }
        case GL_DEPTH_COMPONENT32:
        { return "Depth32"; }
        case GL_DEPTH_COMPONENT32F:
        { return "Depth32f"; }
        case GL_DEPTH_STENCIL:
        { return "DepthStencil"; }
        case GL_DEPTH24_STENCIL8:
        { return "Depth24Stencil8"; }
        case GL_STENCIL:
        { return "Stencil"; }
        case GL_STENCIL_INDEX8:
        { return "Stencil8"; }
        default:
        { return "Unknown"; }
    }
}

GLenum textureFilterToGLEnum(TextureFilter filter, TextureMipFilter mipFilter)
{
    switch (filter)
    {
        case TextureFilter::Nearest:
        {
            switch (mipFilter)
            {
                case TextureMipFilter::None:
                { return GL_NEAREST; }
                case TextureMipFilter::Nearest:
                { return GL_NEAREST_MIPMAP_NEAREST; }
                case TextureMipFilter::Linear:
                { return GL_NEAREST_MIPMAP_LINEAR; }
                default:
                { return GL_NONE; }
            }
        }
        case TextureFilter::Linear:
        {
            switch (mipFilter)
            {
                case TextureMipFilter::None:
                { return GL_LINEAR; }
                case TextureMipFilter::Nearest:
                { return GL_LINEAR_MIPMAP_NEAREST; }
                case TextureMipFilter::Linear:
                { return GL_LINEAR_MIPMAP_LINEAR; }
                default:
                { return GL_NONE; }
            }
        }
        default:
        { return GL_NONE; }
    }
}

std::string textureFilterToStr(TextureFilter filter, TextureMipFilter mipFilter)
{
    switch (filter)
    {
        case TextureFilter::Nearest:
        {
            switch (mipFilter)
            {
                case TextureMipFilter::None:
                { return "Nearest"; }
                case TextureMipFilter::Nearest:
                { return "NearestMipNearest"; }
                case TextureMipFilter::Linear:
                { return "NearestMipLinear"; }
                default:
                { return "Unknown"; }
            }
        }
        case TextureFilter::Linear:
        {
            switch (mipFilter)
            {
                case TextureMipFilter::None:
                { return "Linear"; }
                case TextureMipFilter::Nearest:
                { return "LinearMipNearest"; }
                case TextureMipFilter::Linear:
                { return "LinearMipLinear"; }
                default:
                { return "Unknown"; }
            }
        }
        default:
        { return "Unknown"; }
    }
}

GLenum textureWrapToGLEnum(TextureWrap wrap)
{
    switch (wrap)
    {
        case TextureWrap::Repeat:
        { return GL_REPEAT; }
        case TextureWrap::ClampBorder:
        { return GL_CLAMP_TO_BORDER; }
        case TextureWrap::ClampEdge:
        { return GL_CLAMP_TO_EDGE; }
        case TextureWrap::MirrorRepeat:
        { return GL_MIRRORED_REPEAT; }
        case TextureWrap::MirrorClampEdge:
        { return GL_MIRROR_CLAMP_TO_EDGE; }
        default:
        { return GL_NONE; }
    }
}

std::string textureWrapToStr(TextureWrap wrap)
{
    switch (wrap)
    {
        case TextureWrap::Repeat:
        { return "Repeat"; }
        case TextureWrap::ClampBorder:
        { return "ClampBorder"; }
        case TextureWrap::ClampEdge:
        { return "ClampEdge"; }
        case TextureWrap::MirrorRepeat:
        { return "MirrorRepeat"; }
        case TextureWrap::MirrorClampEdge:
        { return "MirrorClampEdge"; }
        default:
        { return "Unknown"; }
    }
}

GLenum bufferTargetToGLEnum(BufferTarget target)
{
    switch (target)
    {
        case BufferTarget::ArrayBuffer:
        { return GL_ARRAY_BUFFER; }
        case BufferTarget::AtomicCounter:
        { return GL_ATOMIC_COUNTER_BUFFER; }
        case BufferTarget::CopyRead:
        { return GL_COPY_READ_BUFFER; }
        case BufferTarget::CopyWrite:
        { return GL_COPY_WRITE_BUFFER; }
        case BufferTarget::DispatchIndirect:
        { return GL_DISPATCH_INDIRECT_BUFFER; }
        case BufferTarget::DrawIndirect:
        { return GL_DRAW_INDIRECT_BUFFER; }
        case BufferTarget::ElementArray:
        { return GL_ELEMENT_ARRAY_BUFFER; }
        case BufferTarget::PixelPack:
        { return GL_PIXEL_PACK_BUFFER; }
        case BufferTarget::PixelUnpack:
        { return GL_PIXEL_UNPACK_BUFFER; }
        case BufferTarget::Query:
        { return GL_QUERY_BUFFER; }
        case BufferTarget::ShaderStorage:
        { return GL_SHADER_STORAGE_BUFFER; }
        case BufferTarget::Texture:
        { return GL_TEXTURE_BUFFER; }
        case BufferTarget::TransformFeedback:
        { return GL_TRANSFORM_FEEDBACK_BUFFER; }
        case BufferTarget::Uniform:
        { return GL_UNIFORM_BUFFER; }
        default:
        { return GL_NONE; }
    }
}

std::string bufferTargetToStr(BufferTarget target)
{
    switch (target)
    {
        case BufferTarget::ArrayBuffer:
        { return "Array"; }
        case BufferTarget::AtomicCounter:
        { return "AtomicCounter"; }
        case BufferTarget::CopyRead:
        { return "CopyRead"; }
        case BufferTarget::CopyWrite:
        { return "CopyWrite"; }
        case BufferTarget::DispatchIndirect:
        { return "DispatchIndirect"; }
        case BufferTarget::DrawIndirect:
        { return "DrawIndirect"; }
        case BufferTarget::ElementArray:
        { return "ElementArray"; }
        case BufferTarget::PixelPack:
        { return "PixelPack"; }
        case BufferTarget::PixelUnpack:
        { return "PixelUnpack"; }
        case BufferTarget::Query:
        { return "Query"; }
        case BufferTarget::ShaderStorage:
        { return "ShaderStorage"; }
        case BufferTarget::Texture:
        { return "Texture"; }
        case BufferTarget::TransformFeedback:
        { return "TransformFeedback"; }
        case BufferTarget::Uniform:
        { return "Uniform"; }
        default:
        { return "Unknown"; }
    }
}

GLenum bufferUsageToGLEnum(BufferUsageFrequency frequency, BufferUsageAccess access)
{
    switch (frequency)
    {
        case BufferUsageFrequency::Stream:
        {
            switch (access)
            {
                case BufferUsageAccess::Draw:
                { return GL_STREAM_DRAW; }
                case BufferUsageAccess::Read:
                { return GL_STREAM_READ; }
                case BufferUsageAccess::Copy:
                { return GL_STREAM_COPY; }
                default:
                { return GL_NONE; }
            }
        }
        case BufferUsageFrequency::Static:
        {
            switch (access)
            {
                case BufferUsageAccess::Draw:
                { return GL_STATIC_DRAW; }
                case BufferUsageAccess::Read:
                { return GL_STATIC_READ; }
                case BufferUsageAccess::Copy:
                { return GL_STATIC_COPY; }
                default:
                { return GL_NONE; }
            }
        }
        case BufferUsageFrequency::Dynamic:
        {
            switch (access)
            {
                case BufferUsageAccess::Draw:
                { return GL_DYNAMIC_DRAW; }
                case BufferUsageAccess::Read:
                { return GL_DYNAMIC_READ; }
                case BufferUsageAccess::Copy:
                { return GL_DYNAMIC_COPY; }
                default:
                { return GL_NONE; }
            }
        }
        default:
        { return GL_NONE; }
    }
}
std::string bufferUsageToStr(BufferUsageFrequency frequency, BufferUsageAccess access)
{
    switch (frequency)
    {
        case BufferUsageFrequency::Stream:
        {
            switch (access)
            {
                case BufferUsageAccess::Draw:
                { return "StreamDraw"; }
                case BufferUsageAccess::Read:
                { return "StreamRead"; }
                case BufferUsageAccess::Copy:
                { return "StreamCopy"; }
                default:
                { return "Unknown"; }
            }
        }
        case BufferUsageFrequency::Static:
        {
            switch (access)
            {
                case BufferUsageAccess::Draw:
                { return "StaticDraw"; }
                case BufferUsageAccess::Read:
                { return "StaticRead"; }
                case BufferUsageAccess::Copy:
                { return "StaticCopy"; }
                default:
                { return "Unknown"; }
            }
        }
        case BufferUsageFrequency::Dynamic:
        {
            switch (access)
            {
                case BufferUsageAccess::Draw:
                { return "DynamicDraw"; }
                case BufferUsageAccess::Read:
                { return "DynamicRead"; }
                case BufferUsageAccess::Copy:
                { return "DynamicCopy"; }
                default:
                { return "Unknown"; }
            }
        }
        default:
        { return "Unknown"; }
    }
}

GLenum frameBufferAttachmentToGLEnum(FrameBufferAttachmentType type)
{
    switch (type)
    {
        case FrameBufferAttachmentType::Color:
        { return GL_COLOR_ATTACHMENT0; }
        case FrameBufferAttachmentType::Depth:
        { return GL_DEPTH_ATTACHMENT; }
        case FrameBufferAttachmentType::Stencil:
        { return GL_STENCIL_ATTACHMENT; }
        case FrameBufferAttachmentType::DepthStencil:
        { return GL_DEPTH_STENCIL_ATTACHMENT; }
        default:
        { return GL_NONE; }
    }
}

std::string frameBufferAttachmentToStr(FrameBufferAttachmentType type)
{
    switch (type)
    {
        case FrameBufferAttachmentType::Color:
        { return "Color"; }
        case FrameBufferAttachmentType::Depth:
        { return "Depth"; }
        case FrameBufferAttachmentType::Stencil:
        { return "Stencil"; }
        case FrameBufferAttachmentType::DepthStencil:
        { return "DepthStencil"; }
        default:
        { return "Unknown"; }
    }
}

GLenum frameBufferTargetToGLEnum(FrameBufferTarget target)
{
    switch (target)
    {
        case FrameBufferTarget::FrameBuffer:
        { return GL_FRAMEBUFFER; }
        case FrameBufferTarget::ReadFrameBuffer:
        { return GL_READ_FRAMEBUFFER; }
        case FrameBufferTarget::DrawFrameBuffer:
        { return GL_DRAW_FRAMEBUFFER; }
        default:
        { return GL_NONE; }
    }
}

std::string frameBufferTargetToStr(FrameBufferTarget target)
{
    switch (target)
    {
        case FrameBufferTarget::FrameBuffer:
        { return "FrameBuffer"; }
        case FrameBufferTarget::ReadFrameBuffer:
        { return "ReadFrameBuffer"; }
        case FrameBufferTarget::DrawFrameBuffer:
        { return "DrawFrameBuffer"; }
        default:
        { return "Unknown"; }
    }
}

bool checkFrameBufferStatus(FrameBufferTarget target, std::string *str)
{
    const auto frameBufferStatus{ glCheckFramebufferStatus(frameBufferTargetToGLEnum(target)) };

    auto statusOk{ false };
    switch (frameBufferStatus)
    {
        case GL_FRAMEBUFFER_COMPLETE:
        { statusOk = true; if (str) *str = ""; break; }
        case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
        { statusOk = false; if (str) *str = "Frame-buffer status check: Attachments are incomplete!"; break; }
        case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
        { statusOk = false; if (str) *str = "Frame-buffer status check: No attachments!"; break; }
        case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
        { statusOk = false; if (str) *str = "Frame-buffer status check: Draw buffer color attachment has type GL_NONE!"; break; }
        case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
        { statusOk = false; if (str) *str = "Frame-buffer status check: Read buffer color attachment has type GL_NONE!"; break; }
        case GL_FRAMEBUFFER_UNSUPPORTED:
        { statusOk = false; if (str) *str = "Frame-buffer status check: Invalid combination of attachment internal formats!"; break; }
        case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
        { statusOk = false; if (str) *str = "Frame-buffer status check: Not all attachments have the same multi-sample settings.!"; break; }
        case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
        { statusOk = false; if (str) *str = "Frame-buffer status check: Not all attachments are layered!"; break; }
        default:
        { statusOk = false; if (str) *str = "Frame-buffer status check: Unknown error!"; break; }
    }

    return statusOk;
}

void insertTriangle(std::vector<glm::vec4> &vertices, std::vector<GLint> &indices,
    const glm::vec4 &v0, const glm::vec4 &v1, const glm::vec4 &v2,
    float epsilon_not, std::size_t window)
{
    float epsilon{0.0000001f};

    if(glm::any(glm::isnan(v0)) || glm::any(glm::isnan(v1)) || glm::any(glm::isnan(v2)))
        return;

    static constexpr auto INVALID_IDX{ std::numeric_limits<std::size_t>::max() };
    std::size_t i0{ INVALID_IDX };
    std::size_t i1{ INVALID_IDX };
    std::size_t i2{ INVALID_IDX };
    glm::vec4 val{ };
    float diff0{ };
    float diff1{ };
    float diff2{ };
    // Go over (most) existing vertices and test each one for being reasonably similar to our current vert:
    const auto startIndex{
        window > vertices.size() ?
            0u : vertices.size() - window
    };
    for (std::size_t t = startIndex; t < vertices.size(); ++t)
    {
        val = vertices[t];
        if (i0 == INVALID_IDX)
        {
            diff0 = glm::length(v0 - val);
            if (diff0 < epsilon)
            { i0 = t + 1u; }
        }
        if (i1 == INVALID_IDX)
        {
            diff1 = glm::length(v1 - val);
            if (diff1 < epsilon)
            { i1 = t + 1u; }
        }
        if (i2 == INVALID_IDX)
        {
            diff2 = glm::length(v2 - val);
            if (diff2 < epsilon)
            { i2 = t + 1u; }
        }
    }
    // Resolve case where multiple indeces land on the same value (insert the later as a new one)
    if(i0 == i1)
        i1 = INVALID_IDX;
    if(i1 == i2)
        i2 = INVALID_IDX;
    if(i0 == i2)
        i2 = INVALID_IDX;
    // If any vertices have not merged: insert them.
    if (i0 == INVALID_IDX)
    { i0 = vertices.size() + 1u; vertices.push_back(v0); }
    if (i1 == INVALID_IDX)
    { i1 = vertices.size() + 1u; vertices.push_back(v1); }
    if (i2 == INVALID_IDX)
    { i2 = vertices.size() + 1u; vertices.push_back(v2); }

    // by this point indeces should have been figured out so we just store them:
    // (reverse winding order, because OpenGL flips it against OBJ for some reason)
    indices.push_back(static_cast<GLint>(i0));
    indices.push_back(static_cast<GLint>(i1));
    indices.push_back(static_cast<GLint>(i2));
}

void printShaderCompileError(std::ostream &out, GLuint shader)
{
    GLint logSize{ };
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logSize);
    auto logMsg{ new char[logSize] };
    glGetShaderInfoLog(shader, logSize, nullptr, logMsg);
    out << logMsg << std::endl;
    delete [] logMsg;
}

void printAnnotatedShaderSource(std::ostream &out, const std::string &name, GLenum type, const std::string &source)
{
    out << name << " (" << shaderTypeToStr(type) << ") : [\n";

    std::size_t lineNumber{ 1u };
    std::istringstream iss{ source };
    std::string line{ };
    while (std::getline(iss, line))
    { out << lineNumber++ << "  | " << line << "\n"; }

    out << "\n]" << std::endl;
}

void printProgramLinkError(std::ostream &out, GLuint program)
{
    GLint logSize{ };
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logSize);
    auto logMsg{ new char[logSize] };
    glGetProgramInfoLog(program, logSize, NULL, logMsg);
    out << logMsg << std::endl;
    delete [] logMsg;
}

#ifdef IO_USE_EGL

namespace impl
{

// Implementation is mostly taken over from https://github.com/KDAB/eglinfo .

/// @brief Record containing value-name mapping.
struct ValuePair
{
    /// EGL enumeration value.
    EGLint value{ };
    /// Human readable name.
    const char* displayName{ };
}; // struct ValuePair

/// Mapping for EGL bool types.
static constexpr ValuePair EGL_BOOL_MAP[]{
    { EGL_TRUE, "true" },
    { EGL_FALSE, "false" }
};

/// Mapping for EGL buffer types.
static constexpr ValuePair EGL_BUFFER_MAP[]{
    { EGL_RGB_BUFFER, "RGB" },
    { EGL_LUMINANCE_BUFFER, "Luminance" }
};

/// Mapping for EGL caveat types.
static constexpr ValuePair EGL_CAVEAT_MAP[]{
    { EGL_NONE, "none" },
    { EGL_SLOW_CONFIG, "slow" },
    { EGL_NON_CONFORMANT_CONFIG, "non-conformant" }
};

/// Mapping for EGL transparency types.
static constexpr ValuePair EGL_TRANSPARENCY_MAP[]{
    { EGL_NONE, "none" },
    { EGL_TRANSPARENT_RGB, "transparent RGB" }
};

/// Mapping for EGL surface types.
static constexpr ValuePair EGL_SURFACE_MAP[]{
    { EGL_PBUFFER_BIT, "pbuffer" },
    { EGL_PIXMAP_BIT, "pixmap" },
    { EGL_WINDOW_BIT, "window" },
    { EGL_VG_COLORSPACE_LINEAR_BIT, "VG (linear colorspace)" },
    { EGL_VG_ALPHA_FORMAT_PRE_BIT, "VG (alpha format pre)" },
    { EGL_MULTISAMPLE_RESOLVE_BOX_BIT, "multisample resolve box" },
    { EGL_SWAP_BEHAVIOR_PRESERVED_BIT, "swap behavior preserved" },
#ifdef EGL_STREAM_BIT_KHR
    { EGL_STREAM_BIT_KHR, "stream" },
#endif
};

/// Mapping for EGL renderable types.
static constexpr ValuePair EGL_RENDERABLE_MAP[]{
    { EGL_OPENGL_ES_BIT, "OpenGL ES" },
    { EGL_OPENVG_BIT, "OpenVG" },
    { EGL_OPENGL_ES2_BIT, "OpenGL ES2" },
    { EGL_OPENGL_BIT, "OpenGL" },
#ifdef EGL_OPENGL_ES3_BIT
    { EGL_OPENGL_ES3_BIT, "OpenGL ES3" }
#endif
};

/// @brief Information about a single type of EGL device attribute.
struct AttributeRecord
{
    /// Identifier of the EGL attribute.
    EGLint attribute{ };
    /// Human readable name.
    const char* displayName{ };
    /// Mapping from attribute values to human readable names.
    const ValuePair* enumMap{ };
    /// Number of elements in the enumMap.
    int enumMapSize{ };
    /// Is the attribute a flag?
    bool isFlag{ };
}; // struct AttributeRecord

// Helper macros for definition of EGL_ATTRIBUTES
#define A_NUM(x) { x, #x, 0, 0, false }
#define A_MAP(x, map) { x, #x, map, sizeof(map) / sizeof(map[0]), false }
#define A_FLAG(x, map) { x, #x, map, sizeof(map) / sizeof(map[0]), true }

/// List of all EGL device attributes.
static constexpr const AttributeRecord EGL_ATTRIBUTES[] {
    A_NUM(EGL_ALPHA_SIZE),
    A_NUM(EGL_ALPHA_MASK_SIZE),
    A_MAP(EGL_BIND_TO_TEXTURE_RGB, EGL_BOOL_MAP),
    A_MAP(EGL_BIND_TO_TEXTURE_RGBA, EGL_BOOL_MAP),
    A_NUM(EGL_BLUE_SIZE),
    A_NUM(EGL_BUFFER_SIZE),
    A_MAP(EGL_COLOR_BUFFER_TYPE, EGL_BUFFER_MAP),
    A_MAP(EGL_CONFIG_CAVEAT, EGL_CAVEAT_MAP),
    A_NUM(EGL_CONFIG_ID),
    A_FLAG(EGL_CONFORMANT, EGL_RENDERABLE_MAP),
    A_NUM(EGL_DEPTH_SIZE),
    A_NUM(EGL_GREEN_SIZE),
    A_NUM(EGL_LEVEL),
    A_NUM(EGL_LUMINANCE_SIZE),
    A_NUM(EGL_MAX_PBUFFER_WIDTH),
    A_NUM(EGL_MAX_PBUFFER_HEIGHT),
    A_NUM(EGL_MAX_PBUFFER_PIXELS),
    A_NUM(EGL_MAX_SWAP_INTERVAL),
    A_NUM(EGL_MIN_SWAP_INTERVAL),
    A_MAP(EGL_NATIVE_RENDERABLE, EGL_BOOL_MAP),
    A_NUM(EGL_NATIVE_VISUAL_ID),
    A_NUM(EGL_NATIVE_VISUAL_TYPE),
    A_NUM(EGL_RED_SIZE),
    A_FLAG(EGL_RENDERABLE_TYPE, EGL_RENDERABLE_MAP),
    A_NUM(EGL_SAMPLE_BUFFERS),
    A_NUM(EGL_SAMPLES),
    A_NUM(EGL_STENCIL_SIZE),
    A_FLAG(EGL_SURFACE_TYPE, EGL_SURFACE_MAP),
    A_MAP(EGL_TRANSPARENT_TYPE, EGL_TRANSPARENCY_MAP),
    A_NUM(EGL_TRANSPARENT_RED_VALUE),
    A_NUM(EGL_TRANSPARENT_GREEN_VALUE),
    A_NUM(EGL_TRANSPARENT_BLUE_VALUE)
};

#undef A_NUM
#undef A_MAP
#undef A_FLAG

/// Total number of attributes in the EGL_ATTRIBUTES array.
static constexpr std::size_t EGL_ATTRIBUTE_COUNT{ sizeof(EGL_ATTRIBUTES) / sizeof(EGL_ATTRIBUTES[0]) };

/// @brief A single EGL device property.
struct DeviceProperty
{
    /// Identifier of the EGL property.
    EGLint property{ };
    /// Human radable name.
    const char* displayName{ };
    /// Extension used.
    const char* extension{ };
    /// Type of value.
    enum Type
    {
        String,
        Attribute,
    } type{ };
}; // struct DeviceProperty

/// List of all EGL device properties.
static constexpr DeviceProperty EGL_DEVICE_PROPERTIES[]{
#ifdef EGL_DRM_DEVICE_FILE_EXT
    { EGL_DRM_DEVICE_FILE_EXT, "DRM device file", "EGL_EXT_device_drm", DeviceProperty::String },
#endif
#ifdef EGL_CUDA_DEVICE_NV
    { EGL_CUDA_DEVICE_NV, "CUDA device", "EGL_NV_device_cuda", DeviceProperty::Attribute }
#endif
};

/// Total number of properties in the EGL_DEVICE_PROPERTIES array.
static constexpr std::size_t EGL_PROPERTY_COUNT{ sizeof(EGL_DEVICE_PROPERTIES) / sizeof(EGL_DEVICE_PROPERTIES[0]) };

/// @brief Get display for given device.
EGLDisplay displayForDevice(EGLDeviceEXT device)
{
#ifdef EGL_EXT_platform_base
    const auto eglGetPlatformDisplayExt{ reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
        eglGetProcAddress("eglGetPlatformDisplayEXT")
    ) };
    EGLint attribs[]{ EGL_NONE };
    const auto display{ eglGetPlatformDisplayExt(EGL_PLATFORM_DEVICE_EXT, device, attribs) };
    return display;
#else
#warning "Compiling without EGL_EXT_platform_base extension support!"
    return EGL_NO_DISPLAY;
#endif
}

/// @brief Print information about EGL enumeration attribute.
void printEnum(std::ostream &out, int value, const AttributeRecord &attribute)
{
    for (std::size_t iii = 0u; iii < attribute.enumMapSize; ++iii)
    {
        const auto &enumValue{ attribute.enumMap[iii] };
        if (value == enumValue.value)
        { out << enumValue.displayName; return; }
    }
    out << "0x" << std::hex << value << std::dec;
}

/// @brief Print information about EGL flag attribute.
void printFlags(std::ostream &out, int value, const AttributeRecord &attribute)
{
    auto firstEntry{ true };
    int handledFlags{ 0 };

    for (std::size_t iii = 0u; iii < attribute.enumMapSize; ++iii)
    {
        const auto &enumValue{ attribute.enumMap[iii] };
        if (value & enumValue.value)
        {
            out << (firstEntry ? "" : ", ") << enumValue.displayName;
            firstEntry = false;
            handledFlags |= enumValue.value;
        }
    }

    if (handledFlags != value)
    { out << (firstEntry ? "" : ", ") << "unhandled flags 0x" << std::hex << (value - handledFlags) << std::dec; }
}

/// @brief Print information about EGL output layers
void printOutputLayers(std::ostream &out, EGLDisplay display, const char* indent = "")
{
#ifdef EGL_EXT_output_base
    const auto eglGetOutputLayersEXT{ reinterpret_cast<PFNEGLGETOUTPUTLAYERSEXTPROC>(
        eglGetProcAddress("eglGetOutputLayersEXT")
    ) };
    if (!eglGetOutputLayersEXT)
    { out << indent << "Failed to resolve eglGetOutputLayersEXT function." << std::endl; return; }

    EGLint num_layers{ 0 };
    if (!eglGetOutputLayersEXT(display, nullptr, nullptr, 0, &num_layers))
    { out << indent << "Failed to query output layers." << std::endl; return; }
    out << indent << "Found " << num_layers << " output layers." << std::endl;
#endif
}

/// @brief Print information about EGL output ports.
void printOutputPorts(std::ostream &out, EGLDisplay display, const char* indent = "")
{
#ifdef EGL_EXT_output_base
    const auto eglGetOutputPortsEXT{ reinterpret_cast<PFNEGLGETOUTPUTPORTSEXTPROC>(
        eglGetProcAddress("eglGetOutputPortsEXT")
    ) };
    if (!eglGetOutputPortsEXT)
    { out << indent << "Failed to resolve eglGetOutputPortsEXT function." << std::endl; return; }

    EGLint num_ports{ 0 };
    if (!eglGetOutputPortsEXT(display, nullptr, nullptr, 0, &num_ports))
    { out << indent << "Failed to query output ports." << std::endl; return; }
    out << indent << "Found " << num_ports << " output ports." << std::endl;
#endif
}

/// @brief Print information about a given EGL display.
void printDisplay(std::ostream &out, EGLDisplay display, const char* indent = "")
{
    EGLint majorVersion{ }; EGLint minorVersion{ };
    if (!eglInitialize(display, &majorVersion, &minorVersion))
    { out << "Could not initialize EGL!" << std::endl; return; }

    out << indent << "EGL version: " << majorVersion << "." << minorVersion << std::endl;
    const auto clientAPIs{ eglQueryString(display, EGL_CLIENT_APIS) };
    out << indent << "Client APIs for display: " << treeutil::sanitizeGlString(clientAPIs) << std::endl;
    const auto vendor{ eglQueryString(display, EGL_VENDOR) };
    out << indent << "Vendor: " << treeutil::sanitizeGlString(vendor) << std::endl;
    const auto displayExtensions{ eglQueryString(display, EGL_EXTENSIONS) };
    out << indent << "Display extensions: " << treeutil::sanitizeGlString(displayExtensions) << std::endl;

    if (displayExtensions && std::string(displayExtensions).find("EGL_EXT_output_base") != std::string::npos)
    { printOutputLayers(out, display, indent); printOutputPorts(out, display, indent); }

    EGLint numConfigs{ };
    if (!eglGetConfigs(display, nullptr, 0, &numConfigs) && numConfigs > 0)
    { out << "Could not retrieve the number of EGL configurations!" << std::endl; return; }

    out << indent << "Found " << numConfigs << " configurations." << std::endl;

    auto configs{ std::make_unique<EGLConfig[]>(numConfigs) };
    if (!eglGetConfigs(display, configs.get(), numConfigs, &numConfigs))
    { out << "Could not retrieve EGL configurations!" << std::endl; return; }

    for (std::size_t iii = 0u; iii < numConfigs; ++iii)
    {
        out << indent << "Configuration " << iii << ":" << std::endl;
        for (const auto &attribute : EGL_ATTRIBUTES)
        {
            EGLint value{ };
            const auto result{ eglGetConfigAttrib(display, configs[iii], attribute.attribute, &value) };

            out << indent << "  " << attribute.displayName << ": ";

            if (result)
            {
                if (attribute.enumMap)
                {
                    if (!attribute.isFlag)
                    { printEnum(out, value, attribute); }
                    else
                    { printFlags(out, value, attribute); }
                } else
                { out << value; }
            }
            else
            { out << "<failed>"; }

            out << std::endl;
        }
        out << std::endl;
    }
}

}

void printEglDeviceInfo(std::ostream &out)
{
    static constexpr std::size_t MAX_DEVICES{ 0u };
    const auto eglQueryDevicesEXT{ reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(
        eglGetProcAddress("eglQueryDevicesEXT")
    ) };

    EGLDeviceEXT devices[MAX_DEVICES];
    EGLint deviceCount{ };

    if (!eglQueryDevicesEXT(MAX_DEVICES, devices, &deviceCount))
    { out << "Failed to query devices." << std::endl; return; }
    if (deviceCount == 0)
    { out << "Found no devices." << std::endl; return; }

    out << "Found " << deviceCount << " device(s)." << std::endl;
    const auto eglQueryDeviceAttribEXT{ reinterpret_cast<PFNEGLQUERYDEVICEATTRIBEXTPROC>(
        eglGetProcAddress("eglQueryDeviceAttribEXT")
    ) };
    const auto eglQueryDeviceStringEXT{ reinterpret_cast<PFNEGLQUERYDEVICESTRINGEXTPROC>(
        eglGetProcAddress("eglQueryDeviceStringEXT")
    ) };

    for (std::size_t iii = 0u; iii < deviceCount; ++iii)
    {
        out << "Device " << iii << ":" << std::endl;

        const auto device{ devices[iii] };
        const auto devExtensionsQuery{ eglQueryDeviceStringEXT(device, EGL_EXTENSIONS) };
        const auto devExtensions{ std::string(devExtensionsQuery ? devExtensionsQuery : "")};

        if (devExtensionsQuery)
        {
            out << "  Device Extensions: ";
            if (devExtensions.size() > 0)
            { out << devExtensions << std::endl; }
            else
            { out << "none" << std::endl; }
        }
        else
        { out << "  Failed to retrieve device extensions." << std::endl; }

        for (const auto &property : impl::EGL_DEVICE_PROPERTIES)
        {
            if (!devExtensionsQuery || devExtensions.find(property.extension) == std::string::npos)
            {  continue; }
            switch (property.type)
            {
                case impl::DeviceProperty::String:
                {
                    const auto value{ eglQueryDeviceStringEXT(device, property.property) };
                    out << "  " << property.displayName << ": " << value << std::endl;
                    break;
                }
                case impl::DeviceProperty::Attribute:
                {
                    EGLAttrib attribute{ };
                    if (eglQueryDeviceAttribEXT(device, property.property, &attribute) == EGL_FALSE)
                    { break; }
                    out << "  " << property.displayName << ": " << attribute << std::endl;
                    break;
                }
            }
        }

        const auto display{ impl::displayForDevice(device) };
        if (display == EGL_NO_DISPLAY)
        { out << "  No attached display." << std::endl; }
        else
        { out << "  Device display:" << std::endl; impl::printDisplay(out, display, "    "); }

        out << std::endl;
    }
}

#endif // IO_USE_EGL

} // namespace treeutil
