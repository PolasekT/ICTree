/**
 * @author Tomas Polasek
 * @date 16.4.2020
 * @version 1.0
 * @brief Wrapper around OpenGL texture buffer.
 */

#ifndef TREE_TEXTURE_BUFFER_H
#define TREE_TEXTURE_BUFFER_H

#include "TreeUtils.h"
#include "TreeGLUtils.h"

// Forward declarations:
namespace treeio
{ class ImageImporter; }

namespace treerndr
{

/// @brief Wrapper around OpenGL texture buffer.
class TextureBuffer : public treeutil::PointerWrapper<TextureBuffer>
{
public:
    // Shortcuts:
    using Filter = treeutil::TextureFilter;
    using MipFilter = treeutil::TextureMipFilter;
    using Wrap = treeutil::TextureWrap;

    /// @brief Initialize without creating the texture.
    TextureBuffer();
    /// @brief Clean-up and destroy.
    ~TextureBuffer();

    /// @brief Create 1D texture buffer for given data array.
    template <typename ArrT>
    TextureBuffer(const ArrT &arr,
        std::size_t width,
        bool highPrecision = false, bool snorm = false, std::size_t samples = 1u);
    /// @brief Create 1D texture buffer for given iterator range.
    template <typename ItT>
    TextureBuffer(const ItT &begin, const ItT &end,
        std::size_t width,
        bool highPrecision = false, bool snorm = false, std::size_t samples = 1u);

    /// @brief Create 2D texture buffer for given data array.
    template <typename ArrT>
    TextureBuffer(const ArrT &arr,
        std::size_t width, std::size_t height,
        bool highPrecision = false, bool snorm = false, std::size_t samples = 1u);
    /// @brief Create 2D texture buffer for given iterator range.
    template <typename ItT>
    TextureBuffer(const ItT &begin, const ItT &end,
        std::size_t width, std::size_t height,
        bool highPrecision = false, bool snorm = false, std::size_t samples = 1u);

    /// @brief Create 3D texture buffer for given data array.
    template <typename ArrT>
    TextureBuffer(const ArrT &arr,
        std::size_t width, std::size_t height, std::size_t depth,
        bool highPrecision = false, bool snorm = false, std::size_t samples = 1u);
    /// @brief Create 3D texture buffer for given iterator range.
    template <typename ItT>
    TextureBuffer(const ItT &begin, const ItT &end,
        std::size_t width, std::size_t height, std::size_t depth,
        bool highPrecision = false, bool snorm = false, std::size_t samples = 1u);

    /// @brief Create default initialized 1D texture buffer of type ElementT with given properties.
    template <typename ElementT>
    TextureBuffer(std::size_t width, const ElementT &val = { },
        bool highPrecision = false, bool snorm = false, std::size_t samples = 1u);
    /// @brief Create default initialized 2D texture buffer of type ElementT with given properties.
    template <typename ElementT>
    TextureBuffer(std::size_t width, std::size_t height, const ElementT &val = { },
        bool highPrecision = false, bool snorm = false, std::size_t samples = 1u);
    /// @brief Create default initialized 3D texture buffer of type ElementT with given properties.
    template <typename ElementT>
    TextureBuffer(std::size_t width, std::size_t height, std::size_t depth, const ElementT &val = { },
        bool highPrecision = false, bool snorm = false, std::size_t samples = 1u);

    /// @brief Create texture from given initialized image importer.
    TextureBuffer(const treeio::ImageImporter &importer,
        bool highPrecision = false, bool snorm = false);

    /// @brief Create default initialized 2D depth/stencil buffer.
    template <typename DepthStencilT>
    static Ptr createDepthStencil(std::size_t width, std::size_t height);

    /// @brief Create precise copy of given texture buffer.
    static Ptr createDeepCopy(const TextureBuffer &other);

    /// @brief Convert input image into normalized RGB texture.
    template <typename T>
    static Ptr createNormalized(const std::vector<T> &data,
        std::size_t width, std::size_t height,
        bool highPrecision = false, bool snorm = false, std::size_t samples = 1u);

    /// @brief Access the cpu-side texture data as given type. If changed, update dirty flag using setDataChanged().
    template <typename T>
    T* as();
    /// @brief Access the cpu-side texture data as given type.
    template <typename T>
    const T* as() const;

    /// @brief Access the cpu-side texture data as vector. If changed, update dirty flag using setDataChanged().
    template <typename T>
    std::vector<T> &asVector();
    /// @brief Access the cpu-side texture data as vector.
    template <typename T>
    const std::vector<T> &asVector() const;

    /// @brief Delete the current texture. No effect if no texture is created.
    void reset();

    /// @brief Create a deep copy of this texture.
    Ptr deepCopy() const;

    /// @brief Delete the current texture. No effect if no texture is created.
    void reset(bool highPrecision, bool snorm, std::size_t samples);

    /// @brief Upload the current cpu-side data to the GPU buffer. Returns whether the operation completed.
    bool upload();

    /// @brief Download gpu data into the cpu buffer. Returns whether the operation completed.
    bool download();

    /// @brief Set all elements on the cpu-side to given value. Does not upload, only sets data changed flag.
    template <typename T>
    void clear(const T &val = { });

    /// @brief Set data changed flag to mark it needs re-upload().
    void setDataChanged();

    /// @brief Bind the texture. Automatically uploads the data if dirty. Returns success. This version also sets uniform.
    bool bind(int loc, std::size_t unit) const;

    /// @brief Bind the texture. Automatically uploads the data if dirty. Returns success. This version just binds.
    bool bind() const;

    /// @brief Get OpenGL texture handle/name.
    GLuint id() const;

    /// @brief Get width of the current cpu-side texture.
    std::size_t width() const;

    /// @brief Get width of the current cpu-side texture.
    std::size_t height() const;

    /// @brief Get width of the current cpu-side texture.
    std::size_t depth() const;

    /// @brief Get sampling of the texture. Values greater than 1 mean multi-sampling is enabled.
    std::size_t samples() const;

    /// @brief Get number of dimensions in the cpu-side texture has. 0 for empty or {1, 2, 3}.
    std::size_t dimensions() const;

    /// @brief Get number of elements in the cpu-side texture.
    std::size_t elementCount() const;

    /// @brief Is this cpu-side texture empty?
    bool empty() const;

    // Functional parameter setters:
    /// @brief Set minification filter option used by the texture.
    TextureBuffer &minFilter(Filter filter, MipFilter mipFilter);
    /// @brief Set magnification filter option used by the texture.
    TextureBuffer &magFilter(Filter filter);
    /// @brief Set texture wrapping in the width direction.
    TextureBuffer &wrapWidth(Wrap wrap);
    /// @brief Set texture wrapping in the height direction.
    TextureBuffer &wrapHeight(Wrap wrap);
    /// @brief Set texture wrapping in the depth direction.
    TextureBuffer &wrapDepth(Wrap wrap);
    /// @brief Set texture sample count.
    TextureBuffer &samples(std::size_t samples);

    /// @brief Print information about this texture buffer.
    void describe(std::ostream &out, const std::string &indent = "") const;
private:
    // Forward declaration for initialization.
    struct TextureHolder;

    /// @brief Holder for texture parameters.
    struct TextureParameters
    {
        // Sampling parameters:
        /// Did we change sampling parameters?
        bool samplingChanged{ true };
        /// Minification filter used by the texture.
        Filter minFilter{ Filter::Linear };
        /// Minification mip-map filter used by the texture.
        MipFilter minMipFilter{ MipFilter::None };
        /// Maginification filter used by the texture.
        Filter magFilter{ Filter::Linear };
        /// Texture wrapping in the width direction.
        Wrap sWrap{ Wrap::Repeat };
        /// Texture wrapping in the height direction.
        Wrap tWrap{ Wrap::Repeat };
        /// Texture wrapping in the depth direction.
        Wrap rWrap{ Wrap::Repeat };
        /// Sampling of the texture. Values greater than 1 mean multi-sampling is enabled.
        std::size_t samples{ 1u };

        /// @brief Check if we changed sampling parameters and update them for the current texture.
        void updateSamplingParameters(GLuint id, bool force = false);
    }; // struct TextureParameters

    /// @brief Helper structure used to hold the cpu-side data.
    struct BufferDataHolder
    {
        /// @brief Helper for automatic destruction of held data array.
        struct DataArrayBase
        {
            /// @brief Used for automatic data disposal.
            virtual ~DataArrayBase() = default;
            /// @brief Upload texture based on internal data type.
            virtual void uploadTexture(const BufferDataHolder &data, TextureHolder &texture,
                bool highPrecision, bool snorm, TextureParameters &parameters) const = 0;
            /// @brief Download texture into given data buffer resizing as necessary.
            virtual void downloadTexture(BufferDataHolder &data, const TextureHolder &texture) = 0;
            /// @brief Access the underlying data memory.
            virtual void *data() = 0;
            /// @brief Access the underlying data memory.
            virtual const void *data() const = 0;
            /// @brief Create a copy of this data array.
            virtual std::shared_ptr<DataArrayBase> copy() const = 0;
        }; // struct DataArrayBase
        /// @brief Holder of the actual data.
        template <typename T>
        struct DataArray : public DataArrayBase
        {
            /// @brief Used for automatic data disposal.
            virtual ~DataArray() = default;
            /// @brief Upload texture based on internal data type.
            virtual void uploadTexture(const BufferDataHolder &data, TextureHolder &texture,
                bool highPrecision, bool snorm, TextureParameters &parameters) const override final;
            /// @brief Download texture into given data buffer resizing as necessary.
            virtual void downloadTexture(BufferDataHolder &data, const TextureHolder &texture) override final;
            /// @brief Access the underlying data memory.
            virtual void *data() override final;
            /// @brief Access the underlying data memory.
            virtual const void *data() const override final;
            /// @brief Create a copy of this data array.
            virtual std::shared_ptr<DataArrayBase> copy() const override final;
            /// @brief Holder for the actual data.
            std::vector<T> d{ };
        }; // struct DataArray

        /// @brief Create a deep copy of this data holder.
        BufferDataHolder deepCopy();

        /// @brief Access the buffer data as given type.
        template <typename T>
        T *as();
        /// @brief Access the buffer data as given type.
        template <typename T>
        const T *as() const;

        /// @brief Access the underlying data vector.
        template <typename T>
        std::vector<T> &data();

        /// @brief Access the underlying data memory.
        void *data();
        /// @brief Access the underlying data memory.
        const void *data() const;

        /// @brief Get number of dimensions this buffer has. 0 for empty or {1, 2, 3}.
        std::size_t dimensions() const;

        /// @brief Is the internal data buffer initialized and filled?
        bool hasData() const;

        /// OpenGL format of the data.
        GLenum format{ GL_NONE };
        /// OpenGL type of the data.
        GLenum type{ GL_NONE };

        /// Width of the data in number of elements.
        std::size_t width{ 0u };
        /// Height of the data in number of elements.
        std::size_t height{ 0u };
        /// Depth of the data in number of elements.
        std::size_t depth{ 0u };
        /// Size of a single element.
        std::size_t elementSize{ 0u };
        /// Total number of elements.
        std::size_t elementCount{ 0u };
        /// Total number of elements which are actually allocated.
        std::size_t elementsAllocated{ 0u };

        /// Holder of the actual data.
        std::shared_ptr<DataArrayBase> array;
    }; // struct BufferDataHolder

    /// @brief Helper structure used to hold the texture buffer.
    struct TextureHolder
    {
        /// @brief Automatically destroy held texture buffer.
        ~TextureHolder();

        /// @brief Generate texture identifier if necessary. Returns whether new texture had to be created.
        bool generateTexture(GLenum target);

        /// @brief Get number of dimensions this texture has. 0 for empty or {1, 2, 3}.
        std::size_t dimensions() const;

        /// @brief Calculate total number of elements in the texture.
        std::size_t elementCount() const;

        /// Internal format of the texture buffer.
        GLenum internalFormat{ GL_NONE };

        /// Width of the texture in pixels.
        GLsizei width{ 0u };
        /// Height of the texture in pixels.
        GLsizei height{ 0u };
        /// Depth of the texture in pixels.
        GLsizei depth{ 0u };

        /// Sampling of the texture. Values greater than 1 mean multi-sampling is enabled.
        GLsizei samples{ 1u };

        /// Identifier of the texture buffer.
        GLuint id{ };
    }; // struct TextureHolder

    /// @brief Container for current internal state.
    struct InternalState
    {
        /// Do we need to perform
        bool dataChanged{ false };
        /// Use high precision internal format?
        bool highPrecision{ false };
        /// Use normalized signed internal format?
        bool sNorm{ false };

        /// Current texture parameters.
        TextureParameters parameters{ };

        /// Holder for the cpu-side data.
        std::shared_ptr<BufferDataHolder> data{ };
        /// Holder for the currently managed texture.
        std::shared_ptr<TextureHolder> texture{ };
    }; // struct InternalState

    /// @brief Construct texture buffer from already existing internal state.
    TextureBuffer(const std::shared_ptr<InternalState> &state);

    /// @brief Create internal state and fill all members.
    static std::shared_ptr<InternalState> createInternalState();

    /// @brief Create the cpu-side data holder without initializing its data.
    static std::shared_ptr<BufferDataHolder> createDataHolder();

    /// @brief Initialize data holder with provided values. Set any dimension to 0 to disable it.
    static void fillDataHolderNoInitialize(BufferDataHolder &holder,
        std::size_t width, std::size_t height, std::size_t depth,
        GLenum format, GLenum type, std::size_t elementSize,
        const std::shared_ptr<BufferDataHolder::DataArrayBase> &arrayPtr);

    /// @brief Initialize data holder with provided values. Set any dimensions to 0 to disable it.
    template <typename T>
    static std::shared_ptr<BufferDataHolder::DataArray<T>> fillDataHolderNoInitialize(
        BufferDataHolder &holder, std::size_t width, std::size_t height, std::size_t depth);

    /// @brief initialize data holder with provided values. Set any dimensions to 0 to disable it.
    template <typename T>
    static void fillDataHolder(BufferDataHolder &holder, std::size_t width, std::size_t height, std::size_t depth,
        const T *data);

    /// @brief initialize data holder with provided intial value. Set any dimensions to 0 to disable it.
    template <typename T>
    static void fillDataHolder(BufferDataHolder &holder, std::size_t width, std::size_t height, std::size_t depth,
        const T &initial);

    /// @brief initialize data holder with provided values. Set any dimensions to 0 to disable it.
    template <typename ItT>
    static void fillDataHolder(BufferDataHolder &holder, std::size_t width, std::size_t height, std::size_t depth,
        const ItT &begin, const ItT &end);

    /// @brief Replace all data within given data holder with initial value.
    template <typename T>
    static void fillDataHolder(BufferDataHolder &holder, const T &initial);

    /// @brief Create the texture holder without initializing its data.
    static std::shared_ptr<TextureHolder> createTextureHolder();

    /// @brief Upload provided texture data to given texture.
    static bool uploadTextureData(InternalState &state);

    /// @brief Upload 1D texture with given parameters.
    static void uploadTexture1D(TextureHolder &texture, std::size_t width,
        GLenum internalFormat, GLenum format, GLenum type,
        TextureParameters &parameters, const void *data);

    /// @brief Upload 2D texture with given parameters.
    static void uploadTexture2D(TextureHolder &texture, std::size_t width, std::size_t height,
        GLenum internalFormat, GLenum format, GLenum type,
        TextureParameters &parameters, const void *data);

    /// @brief Upload 3D texture with given parameters.
    static void uploadTexture3D(TextureHolder &texture, std::size_t width, std::size_t height, std::size_t depth,
        GLenum internalFormat, GLenum format, GLenum type,
        TextureParameters &parameters, const void *data);

    /// @brief Download provided texture data to given texture.
    static bool downloadTextureData(InternalState &state);

    /// @brief Download 1D texture with given parameters.
    static void downloadTexture1D(const TextureHolder &texture,
        GLenum format, GLenum type, void *data);

    /// @brief Download 2D texture with given parameters.
    static void downloadTexture2D(const TextureHolder &texture,
        GLenum format, GLenum type, void *data);

    /// @brief Download 3D texture with given parameters.
    static void downloadTexture3D(const TextureHolder &texture,
        GLenum format, GLenum type, void *data);

    /// @brief Create a deep copy of given internal state.
    static std::shared_ptr<InternalState> deepCopyState(const InternalState &state);

    /// @brief Perform GPU-side copy of source texture to destination texture. Returns success.
    static bool copyGPUTexture(const TextureHolder &src, const TextureHolder &dst);

    /// Internal state of this texture.
    std::shared_ptr<InternalState> mState{ };
protected:
}; // class TextureBuffer

} // namespace treerndr

/// @brief Print information about the texture buffer.
inline std::ostream &operator<<(std::ostream &out, const treerndr::TextureBuffer &textureBuffer);

// Template implementation begin.

namespace treerndr
{

template <typename ArrT>
TextureBuffer::TextureBuffer(const ArrT &arr,
    std::size_t width,
    bool highPrecision, bool snorm, std::size_t samples)
{
    reset(highPrecision, snorm, samples);
    fillDataHolder(*mState->data, width, 0u, 0u, arr.begin(), arr.end());
    mState->dataChanged = true;
}
template <typename ItT>
TextureBuffer::TextureBuffer(const ItT &begin, const ItT &end,
    std::size_t width,
    bool highPrecision, bool snorm, std::size_t samples)
{
    reset(highPrecision, snorm, samples);
    fillDataHolder(*mState->data, width, 0u, 0u, begin, end);
    mState->dataChanged = true;
}

template <typename ArrT>
TextureBuffer::TextureBuffer(const ArrT &arr,
    std::size_t width, std::size_t height,
    bool highPrecision, bool snorm, std::size_t samples)
{
    reset(highPrecision, snorm, samples);
    fillDataHolder(*mState->data, width, height, 0u, arr.begin(), arr.end());
    mState->dataChanged = true;
}
template <typename ItT>
TextureBuffer::TextureBuffer(const ItT &begin, const ItT &end,
    std::size_t width, std::size_t height,
    bool highPrecision, bool snorm, std::size_t samples)
{
    reset(highPrecision, snorm, samples);
    fillDataHolder(*mState->data, width, height, 0u, begin, end);
    mState->dataChanged = true;
}

template <typename ArrT>
TextureBuffer::TextureBuffer(const ArrT &arr,
    std::size_t width, std::size_t height, std::size_t depth,
    bool highPrecision, bool snorm, std::size_t samples)
{
    reset(highPrecision, snorm, samples);
    fillDataHolder(*mState->data, width, height, depth, arr.begin(), arr.end());
    mState->dataChanged = true;
}
template <typename ItT>
TextureBuffer::TextureBuffer(const ItT &begin, const ItT &end,
    std::size_t width, std::size_t height, std::size_t depth,
    bool highPrecision, bool snorm, std::size_t samples)
{
    reset(highPrecision, snorm, samples);
    fillDataHolder(*mState->data, width, height, depth, begin, end);
    mState->dataChanged = true;
}

template <typename ElementT>
TextureBuffer::TextureBuffer(std::size_t width, const ElementT &val,
    bool highPrecision, bool snorm, std::size_t samples)
{
    reset(highPrecision, snorm, samples);
    fillDataHolder(*mState->data, width, 0u, 0u, val);
    mState->dataChanged = true;
}
template <typename ElementT>
TextureBuffer::TextureBuffer(std::size_t width, std::size_t height, const ElementT &val,
    bool highPrecision, bool snorm, std::size_t samples)
{
    reset(highPrecision, snorm, samples);
    fillDataHolder(*mState->data, width, height, 0u, val);
    mState->dataChanged = true;
}

// Special case for depth/stencil texture -> Do not initialize them.
template <>
inline TextureBuffer::TextureBuffer(std::size_t width, std::size_t height, const treeutil::Depth16&,
    bool highPrecision, bool snorm, std::size_t samples)
{
    reset(highPrecision, snorm, samples);
    fillDataHolder<treeutil::Depth16>(*mState->data, width, height, 0u, nullptr);
    mState->dataChanged = true;
}
template <>
inline TextureBuffer::TextureBuffer(std::size_t width, std::size_t height, const treeutil::Depth32&,
                             bool highPrecision, bool snorm, std::size_t samples)
{
    reset(highPrecision, snorm, samples);
    fillDataHolder<treeutil::Depth32>(*mState->data, width, height, 0u, nullptr);
    mState->dataChanged = true;
}
template <>
inline TextureBuffer::TextureBuffer(std::size_t width, std::size_t height, const treeutil::Depth32f&,
    bool highPrecision, bool snorm, std::size_t samples)
{
    reset(highPrecision, snorm, samples);
    fillDataHolder<treeutil::Depth32f>(*mState->data, width, height, 0u, nullptr);
    mState->dataChanged = true;
}
template <>
inline TextureBuffer::TextureBuffer(std::size_t width, std::size_t height, const treeutil::Depth24Stencil8&,
    bool highPrecision, bool snorm, std::size_t samples)
{
    reset(highPrecision, snorm, samples);
    fillDataHolder<treeutil::Depth24Stencil8>(*mState->data, width, height, 0u, nullptr);
    mState->dataChanged = true;
}
template <>
inline TextureBuffer::TextureBuffer(std::size_t width, std::size_t height, const treeutil::Stencil8&,
    bool highPrecision, bool snorm, std::size_t samples)
{
    reset(highPrecision, snorm, samples);
    fillDataHolder<treeutil::Stencil8>(*mState->data, width, height, 0u, nullptr);
    mState->dataChanged = true;
}

template <typename ElementT>
TextureBuffer::TextureBuffer(std::size_t width, std::size_t height, std::size_t depth, const ElementT &val,
    bool highPrecision, bool snorm, std::size_t samples)
{ reset(highPrecision, snorm, samples); fillDataHolder(*mState->data, width, height, depth, val); mState->dataChanged = true; }
template <typename DepthStencilT>
TextureBuffer::Ptr TextureBuffer::createDepthStencil(std::size_t width, std::size_t height)
{ return instantiate(width, height, DepthStencilT{ }); }

template <typename T>
TextureBuffer::Ptr TextureBuffer::createNormalized(const std::vector<T> &data,
    std::size_t width, std::size_t height,
    bool highPrecision, bool snorm, std::size_t samples)
{
    const auto convertedData{ treeutil::convertImageNormalizedRGB(data) };
    return TextureBuffer::instantiate(convertedData, width, height, highPrecision, snorm, samples);
}

template <typename T>
T* TextureBuffer::as()
{ return mState->data->as<T>(); }
template <typename T>
const T* TextureBuffer::as() const
{ return mState->data->as<T>(); }

template <typename T>
std::vector<T> &TextureBuffer::asVector()
{ return mState->data->data<T>(); }
template <typename T>
const std::vector<T> &TextureBuffer::asVector() const
{ return mState->data->data<T>(); }

template <typename T>
void TextureBuffer::clear(const T &val)
{ fillDataHolder(*mState->data, val); mState->dataChanged = true; }

template <typename T>
void TextureBuffer::BufferDataHolder::DataArray<T>::uploadTexture(
    const BufferDataHolder &data, TextureHolder &texture,
    bool highPrecision, bool snorm, TextureParameters &parameters) const
{
    const auto dataPtr{ data.data() };
    const auto internalFormat{ highPrecision ?
        treeutil::typeToGLTextureInternalFormatHP<T>(snorm) :
        treeutil::typeToGLTextureInternalFormatLP<T>(snorm)
    };
    const auto dimensions{ data.dimensions() };

    switch (dimensions)
    {
        case 1u:
        { // We have a 1D texture.
            TextureBuffer::uploadTexture1D(texture, data.width,
                internalFormat, data.format, data.type, parameters, dataPtr);
            break;
        }
        case 2u:
        { // We have a 2D texture.
            TextureBuffer::uploadTexture2D(texture, data.width, data.height,
                internalFormat, data.format, data.type, parameters, dataPtr);
            break;
        }
        case 3u:
        { // We have a 3D texture.
            TextureBuffer::uploadTexture3D(texture, data.width, data.height, data.depth,
                internalFormat, data.format, data.type, parameters, dataPtr);
            break;
        }
        default:
        { Error << "Unable to upload texture with " << dimensions << " dimensions!" << std::endl; break; }
    }
}

template <typename T>
void TextureBuffer::BufferDataHolder::DataArray<T>::downloadTexture(
    BufferDataHolder &data, const TextureHolder &texture)
{
    const auto textureElementCount{ texture.elementCount() };
    const auto bufferCompatible{
        d.size() == textureElementCount
    };
    if (!bufferCompatible)
    {
        d.resize(textureElementCount);
        data.width = texture.width;
        data.height = texture.height;
        data.depth = texture.depth;
        data.elementCount = textureElementCount;
        data.elementsAllocated = textureElementCount;
    }

    const auto dimensions{ texture.dimensions() };

    switch (dimensions)
    {
        case 1u:
        { // We have a 1D texture.
            TextureBuffer::downloadTexture1D(texture, data.format, data.type, d.data());
            break;
        }
        case 2u:
        { // We have a 2D texture.
            TextureBuffer::downloadTexture2D(texture, data.format, data.type, d.data());
            break;
        }
        case 3u:
        { // We have a 3D texture.
            TextureBuffer::downloadTexture3D(texture, data.format, data.type, d.data());
            break;
        }
        default:
        { Error << "Unable to download texture with " << dimensions << " dimensions!" << std::endl; break; }
    }
}

template <typename T>
void *TextureBuffer::BufferDataHolder::DataArray<T>::data()
{ return d.data(); }

template <typename T>
const void *TextureBuffer::BufferDataHolder::DataArray<T>::data() const
{ return d.data(); }

template <typename T>
std::shared_ptr<TextureBuffer::BufferDataHolder::DataArrayBase>
    TextureBuffer::BufferDataHolder::DataArray<T>::copy() const
{ return std::dynamic_pointer_cast<DataArrayBase>(std::make_shared<DataArray<T>>(*this)); }

template <typename T>
T *TextureBuffer::BufferDataHolder::as()
{
    if (elementCount != elementsAllocated)
    { return nullptr; }
    const auto ptr{ std::dynamic_pointer_cast<DataArray<T>>(array) };
    if (!ptr)
    { throw std::runtime_error("Unable to cast texture content as<>: Invalid request, buffer of different type!"); }
    return ptr->d.data();
}
template <typename T>
const T *TextureBuffer::BufferDataHolder::as() const
{
    if (elementCount != elementsAllocated)
    { return nullptr; }
    const auto ptr{ std::dynamic_pointer_cast<DataArray<T>>(array) };
    if (!ptr)
    { throw std::runtime_error("Unable to cast texture content as<>: Invalid request, buffer of different type!"); }
    return ptr->d.data();
}

template <typename T>
std::vector<T> &TextureBuffer::BufferDataHolder::data()
{
    const auto ptr{ std::dynamic_pointer_cast<DataArray<T>>(array) };
    if (!ptr)
    { throw std::runtime_error("Unable to get texture data<>: Invalid request, buffer of different type!"); }
    return ptr->d;
}

template <typename T>
std::shared_ptr<TextureBuffer::BufferDataHolder::DataArray<T>> TextureBuffer::fillDataHolderNoInitialize(
    BufferDataHolder &holder, std::size_t width, std::size_t height, std::size_t depth)
{
    using ElementT = T;
    const auto format{ treeutil::typeToGLTextureFormat<ElementT>() };
    const auto type{ treeutil::typeToGLTextureType<ElementT>()};
    const auto elementSize{ sizeof(ElementT) };
    const auto arrayPtr{ std::make_shared<BufferDataHolder::DataArray<T>>() };

    fillDataHolderNoInitialize(holder, width, height, depth, format, type, elementSize,
        std::static_pointer_cast<BufferDataHolder::DataArrayBase>(arrayPtr)
    );

    return arrayPtr;
}

template <typename T>
void TextureBuffer::fillDataHolder(BufferDataHolder &holder, std::size_t width, std::size_t height, std::size_t depth,
    const T *data)
{
    const auto arrayPtr{ fillDataHolderNoInitialize<T>(holder, width, height, depth) };
    if (data)
    { arrayPtr->d = std::vector<T>(data, data + holder.elementCount); holder.elementsAllocated = holder.elementCount; }
    else
    { holder.elementsAllocated = 0u; }
}

template <typename T>
void TextureBuffer::fillDataHolder(BufferDataHolder &holder, std::size_t width, std::size_t height, std::size_t depth,
    const T &initial)
{
    const auto arrayPtr{ fillDataHolderNoInitialize<T>(holder, width, height, depth) };
    arrayPtr->d.resize(holder.elementCount, initial);
}

template <typename ItT>
void TextureBuffer::fillDataHolder(BufferDataHolder &holder, std::size_t width, std::size_t height, std::size_t depth,
    const ItT &begin, const ItT &end)
{
    using T = typename std::remove_const<typename std::remove_reference<decltype(*begin)>::type>::type;
    const auto arrayPtr{ fillDataHolderNoInitialize<T>(holder, width, height, depth) };

    arrayPtr->d = std::vector<T>(begin, end);

    if (arrayPtr->d.size() != holder.elementCount)
    {
        Error << "Failed to initialize whole data holder from iterators: "
              << arrayPtr->d.size() << " vs " << holder.elementCount << std::endl;
        arrayPtr->d.resize(holder.elementCount);
    }

    holder.elementsAllocated = holder.elementCount;
}

template <typename T>
void TextureBuffer::fillDataHolder(BufferDataHolder &holder, const T &initial)
{
    if (!holder.hasData())
    { return; }
    const auto &data{ holder.data<T>() };

    data = decltype(data)(data.size(), initial);
}

} // namespace treerndr

inline std::ostream &operator<<(std::ostream &out, const treerndr::TextureBuffer &textureBuffer)
{ textureBuffer.describe(out); return out; }

// Template implementation end.

#endif // TREE_TEXTURE_BUFFER_H
