/**
 * @author Tomas Polasek
 * @date 16.4.2020
 * @version 1.0
 * @brief Wrapper around OpenGL texture buffer.
 */

#include "TreeTextureBuffer.h"

#include <TreeIO/TreeIO.h>

namespace treerndr
{

TextureBuffer::TextureBuffer()
{ reset(); }

TextureBuffer::~TextureBuffer()
{ /* Automatic */ }

TextureBuffer::TextureBuffer(const treeio::ImageImporter &importer,
    bool highPrecision, bool snorm) :
    TextureBuffer()
{
    const auto width{ importer.width() };
    const auto height{ importer.height() };
    const auto samples{ 1u };

    std::vector<glm::vec3> pixels{ };
    pixels.resize(width * height);

    for (std::size_t yPos = 0u; yPos < height; ++yPos)
    {
        for (std::size_t xPos = 0u; xPos < width; ++xPos)
        {
            pixels[xPos + yPos * width] = {
                importer.pixelValue(xPos, yPos, 0),
                importer.pixelValue(xPos, yPos, 1),
                importer.pixelValue(xPos, yPos, 2)
            };
        }
    }

    reset(highPrecision, snorm, samples);

    fillDataHolder(*mState->data, width, height, 0u, pixels.begin(), pixels.end());
    mState->dataChanged = true;
}

TextureBuffer::Ptr TextureBuffer::createDeepCopy(const TextureBuffer &other)
{ return instantiate(TextureBuffer{ deepCopyState(*other. mState) }); }

void TextureBuffer::reset()
{ mState = createInternalState(); }

TextureBuffer::Ptr TextureBuffer::deepCopy() const
{ return createDeepCopy(*this); }

void TextureBuffer::reset(bool highPrecision, bool snorm, std::size_t samples)
{ mState = createInternalState(); mState->highPrecision = highPrecision; mState->sNorm = snorm; mState->parameters.samples = samples; }

bool TextureBuffer::upload()
{ return uploadTextureData(*mState); }

bool TextureBuffer::download()
{ return downloadTextureData(*mState); }

void TextureBuffer::setDataChanged()
{ mState->dataChanged = true; }

bool TextureBuffer::bind(int loc, std::size_t unit) const
{
    if (!bind())
    { return false; }

    // Set uniform and use texture.
    const auto unitId{ static_cast<int>(unit) };
    glActiveTexture(static_cast<GLenum>(GL_TEXTURE0 + unitId));
    glUniform1i(loc, unitId);

    return true;
}

bool TextureBuffer::bind() const
{
    const auto dimensionCount{ mState->texture->dimensions() };
    if (dimensionCount < 1u || dimensionCount > 3u || mState->texture->id == 0u)
    { return false; }

    // Const-cast is safe since we only change the data updated flag.
    const_cast<TextureBuffer*>(this)->upload();

    switch (dimensionCount)
    {
        case 1u:
        { glBindTexture(GL_TEXTURE_1D, mState->texture->id); break; }
        case 2u:
        { glBindTexture(mState->texture->samples > 1u ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, mState->texture->id); break; }
        case 3u:
        { glBindTexture(GL_TEXTURE_3D, mState->texture->id); break; }
        default:
        { Error << "Failed to bind texture: Unknown number of dimension " << dimensionCount << std::endl; }
    }

    // Update sampler parameters if necessary. Const-cast is safe, since we only update mSamplingChanged flag.
    const_cast<TextureBuffer*>(this)->mState->parameters.updateSamplingParameters(mState->texture->id);

    return true;
}

GLuint TextureBuffer::id() const
{ return mState->texture->id; }

std::size_t TextureBuffer::width() const
{ return mState->data->width; }

std::size_t TextureBuffer::height() const
{ return mState->data->height; }

std::size_t TextureBuffer::depth() const
{ return mState->data->depth; }

std::size_t TextureBuffer::samples() const
{ return mState->texture->samples; }

std::size_t TextureBuffer::dimensions() const
{ return mState->data->dimensions(); }

std::size_t TextureBuffer::elementCount() const
{ return mState->data->elementsAllocated; }

bool TextureBuffer::empty() const
{ return mState->data->elementsAllocated == 0u; }

TextureBuffer &TextureBuffer::minFilter(Filter filter, MipFilter mipFilter)
{ mState->parameters.minFilter = filter; mState->parameters.minMipFilter = mipFilter; mState->parameters.samplingChanged = true; return *this; }
TextureBuffer &TextureBuffer::magFilter(Filter filter)
{ mState->parameters.magFilter = filter; mState->parameters.samplingChanged = true; return *this; }
TextureBuffer &TextureBuffer::wrapWidth(Wrap wrap)
{ mState->parameters.sWrap = wrap; mState->parameters.samplingChanged = true; return *this; }
TextureBuffer &TextureBuffer::wrapHeight(Wrap wrap)
{ mState->parameters.tWrap = wrap; mState->parameters.samplingChanged = true; return *this; }
TextureBuffer &TextureBuffer::wrapDepth(Wrap wrap)
{ mState->parameters.rWrap = wrap; mState->parameters.samplingChanged = true; return *this; }
TextureBuffer &TextureBuffer::samples(std::size_t samples)
{ mState->parameters.samples = samples; mState->parameters.samplingChanged = true; mState->dataChanged = true; return *this; }

void TextureBuffer::describe(std::ostream &out, const std::string &indent) const
{
    out << "[ Texture: \n"
        << indent << "\tData Format = " << treeutil::glTextureFormatToStr(mState->data->format) << "\n"
        << indent << "\tData Type = " << treeutil::glTextureTypeToStr(mState->data->type) << "\n"
        << indent << "\tData Element Size = " << mState->data->elementSize << "\n"
        << indent << "\tData Element Count = " << mState->data->elementCount << "\n"
        << indent << "\tData Elements Allocated = " << mState->data->elementsAllocated << "\n"
        << indent << "\tData Dimensions = " << mState->data->width << "x" << mState->data->height << "x" << mState->data->depth << "\n"
        << indent << "\tInternal Format = " << treeutil::glTextureInternalFormatToStr(mState->texture->internalFormat) << "\n"
        << indent << "\tTexture Dimensions = " << mState->texture->width << "x" << mState->texture->height << "x" << mState->texture->depth << "\n"
        << indent << "\tTexture ID = " << mState->texture->id << "\n"
        << indent << "\tTexture Min Filter = " << treeutil::textureFilterToStr(mState->parameters.minFilter, mState->parameters.minMipFilter) << "\n"
        << indent << "\tTexture Mag Filter = " << treeutil::textureFilterToStr(mState->parameters.magFilter, MipFilter::None) << "\n"
        << indent << "\tTexture Width Wrap = " << treeutil::textureWrapToStr(mState->parameters.sWrap) << "\n"
        << indent << "\tTexture Height Wrap = " << treeutil::textureWrapToStr(mState->parameters.tWrap) << "\n"
        << indent << "\tTexture Depth Wrap = " << treeutil::textureWrapToStr(mState->parameters.rWrap) << "\n"
        << indent << " ]";
}

void TextureBuffer::TextureParameters::updateSamplingParameters(GLuint id, bool force)
{
    if ((!samplingChanged && !force) || id == 0u || samples > 1u)
    { return; }

    glTextureParameteri(id, GL_TEXTURE_MIN_FILTER,
        treeutil::textureFilterToGLEnum(minFilter, minMipFilter));
    glTextureParameteri(id, GL_TEXTURE_MAG_FILTER,
        treeutil::textureFilterToGLEnum(magFilter, MipFilter::None));
    glTextureParameteri(id, GL_TEXTURE_WRAP_S,
        treeutil::textureWrapToGLEnum(sWrap));
    glTextureParameteri(id, GL_TEXTURE_WRAP_T,
        treeutil::textureWrapToGLEnum(tWrap));
    glTextureParameteri(id, GL_TEXTURE_WRAP_R,
        treeutil::textureWrapToGLEnum(rWrap));

    samplingChanged = false;
}

TextureBuffer::BufferDataHolder TextureBuffer::BufferDataHolder::deepCopy()
{ auto result{ *this }; result.array = array->copy(); return result; }

void *TextureBuffer::BufferDataHolder::data()
{ return elementCount == elementsAllocated ? array->data() : nullptr; }

const void *TextureBuffer::BufferDataHolder::data() const
{ return elementCount == elementsAllocated ? array->data() : nullptr; }

std::size_t TextureBuffer::BufferDataHolder::dimensions() const
{ return (depth ? 3u : (height ? 2u : (width ? 1u : 0u))); }

bool TextureBuffer::BufferDataHolder::hasData() const
{ return array && elementsAllocated == elementCount; }

TextureBuffer::TextureHolder::~TextureHolder()
{
    if (id)
    { glDeleteTextures(1u, &id); id = 0u; }
}

bool TextureBuffer::TextureHolder::generateTexture(GLenum target)
{
    if (!id)
    { glCreateTextures(target, 1u, &id); return true; }
    else
    { return false; }
}

std::size_t TextureBuffer::TextureHolder::dimensions() const
{ return (depth ? 3u : (height ? 2u : (width ? 1u : 0u))); }

std::size_t TextureBuffer::TextureHolder::elementCount() const
{ return width * (height ? height : 1u) * (depth ? depth : 1u); }

TextureBuffer::TextureBuffer(const std::shared_ptr<InternalState> &state) :
    mState{ state }
{ }

std::shared_ptr<TextureBuffer::InternalState> TextureBuffer::createInternalState()
{
    const auto state{ std::make_shared<InternalState>() };
    state->data = createDataHolder(); state->texture = createTextureHolder(); state->dataChanged = true;
    return state;
}

std::shared_ptr<TextureBuffer::BufferDataHolder> TextureBuffer::createDataHolder()
{ const auto holderPtr{ std::make_shared<BufferDataHolder>() }; return holderPtr; }

void TextureBuffer::fillDataHolderNoInitialize(BufferDataHolder &holder,
    std::size_t width, std::size_t height, std::size_t depth,
    GLenum format, GLenum type, std::size_t elementSize,
    const std::shared_ptr<BufferDataHolder::DataArrayBase> &arrayPtr)
{
    const auto hasWidth{ width > 0u };
    const auto hasHeight{ hasWidth && height > 0u };
    const auto hasDepth{ hasWidth && hasHeight && depth > 0u }; TREE_UNUSED(hasDepth);
    const auto realWidth{ width };
    const auto realHeight{ hasWidth ? (height ? height : 1u) : 0u };
    const auto realDepth{ hasHeight ? (depth ? depth : 1u) : 0u };
    const auto elementCount{ realWidth * realHeight * realDepth };

    holder.format = format;
    holder.type = type;
    holder.width = width;
    holder.height = hasWidth ? height : 0u;
    holder.depth = hasHeight ? depth : 0u;
    holder.elementSize = elementSize;
    holder.elementCount = elementCount;
    holder.array = arrayPtr;
}

std::shared_ptr<TextureBuffer::TextureHolder> TextureBuffer::createTextureHolder()
{ const auto holderPtr{ std::make_shared<TextureHolder>() }; return holderPtr; }

bool TextureBuffer::uploadTextureData(InternalState &state)
{
    if (!state.dataChanged)
    { return false; }

    state.data->array->uploadTexture(*state.data, *state.texture, state.highPrecision, state.sNorm, state.parameters);

    state.dataChanged = false;

    return true;
}

void TextureBuffer::uploadTexture1D(TextureHolder &texture, std::size_t width,
    GLenum internalFormat, GLenum format, GLenum type, TextureParameters &parameters, const void *data)
{
    if (parameters.samples > 1u)
    { throw std::runtime_error("TextureBuffer::uploadTexture1D : Cannot create 1D multi-sampled texture!"); }

    const auto textureCreated{ texture.generateTexture(GL_TEXTURE_1D) };
    const auto textureCompatible{
        !textureCreated &&
        static_cast<std::size_t>(texture.width) == width &&
        texture.internalFormat == internalFormat
    };

    glBindTexture(GL_TEXTURE_1D, texture.id);
    parameters.updateSamplingParameters(texture.id, true);

    const auto glWidth{ static_cast<GLsizei>(width) };

    // Either substitute data or just create new texture.
    if (textureCompatible)
    { if (data) glTexSubImage1D(GL_TEXTURE_1D, 0u, 0, texture.width, format, type, data); }
    else
    { glTexImage1D(GL_TEXTURE_1D, 0u, internalFormat, glWidth, 0, format, type, data); }

    texture.internalFormat = internalFormat;
    texture.width = glWidth; texture.height = 0u; texture.depth = 0u;
}

void TextureBuffer::uploadTexture2D(TextureHolder &texture, std::size_t width, std::size_t height,
    GLenum internalFormat, GLenum format, GLenum type, TextureParameters &parameters, const void *data)
{
    const auto multiSampleTexture{ parameters.samples > 1u};
    if (data && multiSampleTexture)
    { throw std::runtime_error( "TextureBuffer::uploadTexture2D : Cannot initialize 2D multi-sampled texture with data!"); }
    const auto target{
        multiSampleTexture ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D
    };

    const auto textureCreated{ texture.generateTexture(target) };
    const auto textureCompatible{
        !textureCreated &&
        static_cast<std::size_t>(texture.width) == width &&
        static_cast<std::size_t>(texture.height) == height &&
        static_cast<std::size_t>(texture.samples) == parameters.samples &&
        texture.internalFormat == internalFormat
    };

    glBindTexture(target, texture.id);
    parameters.updateSamplingParameters(texture.id, true);

    const auto glWidth{ static_cast<GLsizei>(width) };
    const auto glHeight{ static_cast<GLsizei>(height) };

    // Either substitute data or just create new texture.
    if (textureCompatible)
    { if (data) glTexSubImage2D(target, 0u, 0, 0, texture.width, texture.height, format, type, data); }
    else
    {
        if (multiSampleTexture)
        { glTexImage2DMultisample(target, parameters.samples, internalFormat, glWidth, glHeight, false); }
        else
        { glTexImage2D(target, 0u, internalFormat, glWidth, glHeight, 0, format, type, data); }
    }

    texture.internalFormat = internalFormat;
    texture.width = glWidth; texture.height = glHeight; texture.depth = 0u; texture.samples = parameters.samples;
}

void TextureBuffer::uploadTexture3D(TextureHolder &texture, std::size_t width, std::size_t height, std::size_t depth,
    GLenum internalFormat, GLenum format, GLenum type, TextureParameters &parameters, const void *data)
{
    if (parameters.samples > 1u)
    { throw std::runtime_error("TextureBuffer::uploadTexture3D : Cannot create 3D multi-sampled texture!"); }

    const auto textureCreated{ texture.generateTexture(GL_TEXTURE_3D) };
    const auto textureCompatible{
        !textureCreated &&
        static_cast<std::size_t>(texture.width) == width &&
        static_cast<std::size_t>(texture.height) == height &&
        static_cast<std::size_t>(texture.depth) == depth &&
        texture.internalFormat == internalFormat
    };

    glBindTexture(GL_TEXTURE_3D, texture.id);
    parameters.updateSamplingParameters(texture.id, true);

    const auto glWidth{ static_cast<GLsizei>(width) };
    const auto glHeight{ static_cast<GLsizei>(height) };
    const auto glDepth{ static_cast<GLsizei>(depth) };

    // Either substitute data or just create new texture.
    if (textureCompatible)
    { if (data) glTexSubImage3D(GL_TEXTURE_3D, 0u, 0, 0, 0, texture.width, texture.height, texture.depth, format, type, data); }
    else
    { glTexImage3D(GL_TEXTURE_3D, 0u, internalFormat, glWidth, glHeight, glDepth, 0, format, type, data); }

    texture.internalFormat = internalFormat;
    texture.width = glWidth; texture.height = glHeight; texture.depth = glDepth;
}

bool TextureBuffer::downloadTextureData(InternalState &state)
{ state.data->array->downloadTexture(*state.data, *state.texture); state.dataChanged = false; return true; }

void TextureBuffer::downloadTexture1D(const TextureHolder &texture,
    GLenum format, GLenum type, void *data)
{
    glBindTexture(GL_TEXTURE_1D, texture.id);
    glGetTexImage(GL_TEXTURE_1D, 0, format, type, data);
}

void TextureBuffer::downloadTexture2D(const TextureHolder &texture,
    GLenum format, GLenum type, void *data)
{
    if (texture.samples > 1u)
    { throw std::runtime_error("TextureBuffer::downloadTexture2D : Cannot download 2D multi-sampled texture!"); }

    glBindTexture(GL_TEXTURE_2D, texture.id);
    glGetTexImage(GL_TEXTURE_2D, 0, format, type, data);
}

void TextureBuffer::downloadTexture3D(const TextureHolder &texture,
    GLenum format, GLenum type, void *data)
{
    glBindTexture(GL_TEXTURE_3D, texture.id);
    glGetTexImage(GL_TEXTURE_3D, 0, format, type, data);
}

std::shared_ptr<TextureBuffer::InternalState> TextureBuffer::deepCopyState(const InternalState &state)
{
    const auto result{ createInternalState() };

    result->dataChanged = state.dataChanged;
    result->highPrecision = state.highPrecision;
    result->sNorm = state.sNorm;
    result->parameters = state.parameters;

    // Copy CPU-side state.
    *result->data = state.data->deepCopy();
    // Copy GPU-side state.
    if (state.texture->id)
    { result->dataChanged = true; uploadTextureData(*result); copyGPUTexture(*state.texture, *result->texture); }

    return result;
}

bool TextureBuffer::copyGPUTexture(const TextureHolder &src, const TextureHolder &dst)
{
    if (src.width != dst.width || src.height != dst.height || src.depth != dst.depth ||
        src.id == 0u || dst.id == 0u)
    { return false; }

    if (src.depth != 0u)
    {
        glCopyImageSubData(
            src.id, GL_TEXTURE_3D, 0, 0, 0, 0,
            dst.id, GL_TEXTURE_3D, 0, 0, 0, 0,
            src.width, src.height, src.depth
        );
        return true;
    }
    else if (src.height != 0u)
    {
        glCopyImageSubData(
            src.id, GL_TEXTURE_2D, 0, 0, 0, 0,
            dst.id, GL_TEXTURE_2D, 0, 0, 0, 0,
            src.width, src.height, 1
        );
        return true;
    }
    else if (src.width != 0u)
    {
        glCopyImageSubData(
            src.id, GL_TEXTURE_1D, 0, 0, 0, 0,
            dst.id, GL_TEXTURE_1D, 0, 0, 0, 0,
            src.width, 1, 1
        );
        return true;
    }

    return false;
}

} // namespace treerndr
