/**
 * @author Tomas Polasek
 * @date 16.4.2020
 * @version 1.0
 * @brief Wrapper around OpenGL buffer.
 */

#ifndef TREE_BUFFER_H
#define TREE_BUFFER_H

#include "TreeUtils.h"
#include "TreeGLUtils.h"

namespace treerndr
{

/// @brief Buffer as a container for data shared between CPU and GPU.
class Buffer : public treeutil::PointerWrapper<Buffer>
{
public:
    // Shortcuts:
    using Target = treeutil::BufferTarget;
    using UsageFrequency = treeutil::BufferUsageFrequency;
    using UsageAccess = treeutil::BufferUsageAccess;

    /// @brief Initialize without creating the buffer.
    Buffer();
    /// @brief Clean-up and destroy.
    ~Buffer();

    /// @brief Wrap around already existing buffer. This buffer will not get destroyed on destruction if specified.
    template <typename T>
    Buffer(GLuint id, std::size_t elementCount, bool destroy = false,
           UsageFrequency frequency = UsageFrequency::Static,
           UsageAccess access = UsageAccess::Draw,
           Target target = Target::ArrayBuffer,
           const T& = { });

    /// @brief Wrap around already existing buffer. This buffer will not get destroyed on destruction if specified.
    template <typename T>
    static Ptr createWrapBuffer(GLuint id, std::size_t elementCount, bool destroy = false,
        UsageFrequency frequency = UsageFrequency::Static,
        UsageAccess access = UsageAccess::Draw,
        Target target = Target::ArrayBuffer);

    /// @brief Create buffer and initialize it with provided data. Provided target is only initial.
    template <typename ArrT>
    Buffer(const ArrT &arr, std::size_t elementCount,
        UsageFrequency frequency = UsageFrequency::Static,
        UsageAccess access = UsageAccess::Draw,
        Target target = Target::ArrayBuffer);
    /// @brief Create buffer and initialize it with provided data. Provided target is only initial.
    template <typename ItT>
    Buffer(const ItT &begin, const ItT &end,
        std::size_t elementCount,
        UsageFrequency frequency = UsageFrequency::Static,
        UsageAccess access = UsageAccess::Draw,
        Target target = Target::ArrayBuffer);

    /// @brief Create default initialized buffer of type ElementT. Provided target is only initial.
    template <typename ElementT>
    Buffer(std::size_t elementCount, const ElementT &val = { },
        UsageFrequency frequency = UsageFrequency::Static,
        UsageAccess access = UsageAccess::Draw,
        Target target = Target::ArrayBuffer);

    /// @brief Create uninitialized buffer of type ElementT. Provided target is only initial.
    template <typename ElementT>
    Buffer(std::size_t elementCount,
           UsageFrequency frequency = UsageFrequency::Static,
           UsageAccess access = UsageAccess::Draw,
           Target target = Target::ArrayBuffer,
           const ElementT& = { });

    /// @brief Create uninitialized buffer of type ElementT. Provided target is only initial.
    template <typename ElementT>
    static Ptr createEmpty(std::size_t elementCount,
        UsageFrequency frequency = UsageFrequency::Static,
        UsageAccess access = UsageAccess::Draw,
        Target target = Target::ArrayBuffer);

    /// @brief Access the cpu-side buffer data as given type. If changed, update dirty flag using setDataChanged().
    template <typename T>
    T* as();
    /// @brief Access the cpu-side buffer data as given type.
    template <typename T>
    const T* as() const;

    /// @brief Access the cpu-side buffer data as vector. If changed, update dirty flag using setDataChanged().
    template <typename T>
    std::vector<T> &asVector();
    /// @brief Access the cpu-side buffer data as vector.
    template <typename T>
    const std::vector<T> &asVector() const;

    /// @brief Does this buffer use given internal type?
    template <typename T>
    bool usesType() const;

    /// @brief Delete the current buffer. No effect if no buffer is created.
    void reset();

    /// @brief Delete the current buffer. No effect if no buffer is created.
    void reset(UsageFrequency frequency, UsageAccess access, Target target);

    /// @brief Upload the current cpu-side data to the GPU buffer. Returns whether the operation completed.
    bool upload();

    /// @brief Download gpu data into the cpu buffer. Returns whether the operation completed.
    bool download();

    /// @brief Set all elements on the cpu-side data to value. Does not upload, only sets data changed flag.
    template <typename T>
    void clear(const T &val = { });

    /// @brief Set data changed flag to mark it needs re-upload().
    void setDataChanged();

    /// @brief Bind the buffer to given target. Automatically uploads the data if dirty. Returns success.
    bool bind(Target target) const;

    /// @brief Bind the buffer to given layout as target. Automatically uploads the data if dirty. Returns success.
    bool bind(std::size_t layout, Target target = Target::ShaderStorage) const;

    /// @brief Get OpenGL buffer handle/name.
    GLuint id() const;

    /// @brief Get number of elements in the cpu-side buffer.
    std::size_t elementCount() const;

    /// @brief Is this cpu-side texture buffer?
    bool empty() const;

    /// @brief Print information about this buffer.
    void describe(std::ostream &out, const std::string &indent = "") const;
private:
    // Forward declaration for initialization.
    struct BufferHolder;

    /// @brief Helper structure used to hold the cpu-side data.
    struct BufferDataHolder
    {
        /// @brief Helper for automatic destruction of held data array.
        struct DataArrayBase
        {
            /// @brief Used for automatic data disposal.
            virtual ~DataArrayBase() = default;
            /// @brief Upload buffer based on internal data type.
            virtual void uploadBuffer(const BufferDataHolder &data, BufferHolder &buffer,
                Target target, UsageFrequency frequency, UsageAccess access) const = 0;
            /// @brief Download data into given buffer resizing as necessary.
            virtual void downloadBuffer(BufferDataHolder &data, const BufferHolder &buffer,
                Target target) = 0;
            /// @brief Access the underlying data memory.
            virtual void *data() = 0;
            /// @brief Access the underlying data memory.
            virtual const void *data() const = 0;
        }; // struct DataArrayBase
        /// @brief Holder of the actual data.
        template <typename T>
        struct DataArray : public DataArrayBase
        {
            /// @brief Used for automatic data disposal.
            virtual ~DataArray() = default;
            /// @brief Upload buffer based on internal data type.
            virtual void uploadBuffer(const BufferDataHolder &data, BufferHolder &buffer,
                Target target, UsageFrequency frequency, UsageAccess access) const override final;
            /// @brief Download data into given buffer resizing as necessary.
            virtual void downloadBuffer(BufferDataHolder &data, const BufferHolder &buffer,
                Target target) override final;
            /// @brief Access the underlying data memory.
            virtual void *data() override final;
            /// @brief Access the underlying data memory.
            virtual const void *data() const override final;
            /// @brief Holder for the actual data.
            std::vector<T> d{ };
        }; // struct DataArray

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
        void *data() const;

        /// @brief Is the internal data buffer initialized and filled?
        bool hasData() const;
        /// @brief Does this holder use given internal type?
        template <typename T>
        bool usesType() const;

        /// Size of a single element.
        std::size_t elementSize{ 0u };
        /// Total number of elements.
        std::size_t elementCount{ 0u };
        /// Total number of elements which are actually allocated.
        std::size_t elementsAllocated{ 0u };

        /// Holder of the actual data.
        std::shared_ptr<DataArrayBase> array;
    }; // struct BufferDataHolder

    /// @brief Helper structure used to hold the buffer.
    struct BufferHolder
    {
        /// @brief Automatically destroy held buffer.
        ~BufferHolder();

        /// @brief Generate texture identifier if necessary. Returns whether new texture had to be created.
        bool generateBuffer();

        /// Byte size of the allocated buffer.
        GLsizei byteSize{ 0u };

        /// Usage frequency specifier for the created buffer.
        UsageFrequency usageFrequency{ };
        /// Usage access specifier for the created buffer.
        UsageAccess usageAccess{ };

        /// Identifier of the buffer.
        GLuint id{ };
        /// Should the internal buffer be destroyed on destruction?
        bool destroy{ true };
    }; // struct BufferHolder

    /// @brief Container for current internal state
    struct InternalState
    {
        /// Do we need to perform
        bool dataChanged{ false };
        /// Usage frequency of the created buffers.
        UsageFrequency usageFrequency{ };
        /// Usage access of the created buffers.
        UsageAccess usageAccess{ };
        /// Default target of the created buffers.
        Target target{ };
        /// Holder for the cpu-side data.
        std::shared_ptr<BufferDataHolder> data{ };
        /// Holder for the currently managed buffer.
        std::shared_ptr<BufferHolder> buffer{ };
    }; // struct InternalState

    /// @brief Create the internal state and initialize.
    std::shared_ptr<InternalState> createInternalState() const;

    /// @brief Create the cpu-side data holder without initializing its data.
    std::shared_ptr<BufferDataHolder> createDataHolder() const;

    /// @brief Initialize data holder with provided values.
    void fillDataHolderNoInitialize(BufferDataHolder &holder, std::size_t elementCount,
        std::size_t elementSize, const std::shared_ptr<BufferDataHolder::DataArrayBase> &arrayPtr) const;

    /// @brief Initialize data holder with provided values.
    template <typename T>
    std::shared_ptr<BufferDataHolder::DataArray<T>> fillDataHolderNoInitialize(
        BufferDataHolder &holder, std::size_t elementCount) const;

    /// @brief initialize data holder with provided values.
    template <typename T>
    void fillDataHolder(BufferDataHolder &holder, std::size_t elementCount, const T *data) const;

    /// @brief initialize data holder with provided initial values.
    template <typename T>
    void fillDataHolder(BufferDataHolder &holder, std::size_t elementCount, const T &initial) const;

    /// @brief initialize data holder with provided values.
    template <typename ItT>
    void fillDataHolder(BufferDataHolder &holder, std::size_t elementCount, const ItT &begin, const ItT &end) const;

    /// @brief Replace all data within given data holder with initial value.
    template <typename T>
    void fillDataHolder(BufferDataHolder &holder, const T &initial) const;

    /// @brief Create the buffer holder without initializing its data.
    std::shared_ptr<BufferHolder> createBufferHolder() const;

    /// @brief Wrap around already existing buffer with given properties.
    template <typename T>
    void wrapBuffer(InternalState &state, GLuint id, std::size_t elementCount, bool destroy);

    /// @brief Upload provided texture data to given buffer.
    bool uploadBufferData(const BufferDataHolder &data, BufferHolder &buffer);

    /// @brief Upload buffer with given parameters.
    static void uploadBuffer(BufferHolder &buffer, std::size_t byteSize,
        Target target, UsageFrequency frequency, UsageAccess access, const void *data);

    /// @brief Download provided buffer data to given buffer.
    bool downloadBufferData(BufferDataHolder &data, const BufferHolder &buffer);

    /// @brief Download buffer with given parameters.
    static void downloadBuffer(const BufferHolder &buffer,
        Target target, void *data);

    /// Current internal state.
    std::shared_ptr<InternalState> mState{ };
protected:
}; // class Buffer

} // namespace treerndr

/// @brief Print information about the buffer.
inline std::ostream &operator<<(std::ostream &out, const treerndr::Buffer &buffer);

// Template implementation begin.

namespace treerndr
{

template <typename T>
Buffer::Buffer(GLuint id, std::size_t elementCount, bool destroy,
    UsageFrequency frequency, UsageAccess access, Target target, const T&)
{ reset(frequency, access, target); wrapBuffer<T>(*mState, id, elementCount, destroy); }
template <typename T>
Buffer::Ptr Buffer::createWrapBuffer(GLuint id, std::size_t elementCount, bool destroy,
    UsageFrequency frequency, UsageAccess access, Target target)
{ return instantiate(id, elementCount, destroy, frequency, access, target, T{ }); }
template <typename ArrT>
Buffer::Buffer(const ArrT &arr, std::size_t elementCount,
       UsageFrequency frequency, UsageAccess access, Target target)
{ reset(frequency, access, target); fillDataHolder(*mState->data, elementCount, arr.begin(), arr.end()); mState->dataChanged = true; upload(); }
template <typename ItT>
Buffer::Buffer(const ItT &begin, const ItT &end,
       std::size_t elementCount,
       UsageFrequency frequency, UsageAccess access, Target target)
{ reset(frequency, access, target); fillDataHolder(*mState->data, elementCount, begin, end); mState->dataChanged = true; upload(); }

template <typename ElementT>
Buffer::Buffer(std::size_t elementCount, const ElementT &val,
    UsageFrequency frequency, UsageAccess access, Target target)
{ reset(frequency, access, target); fillDataHolder(*mState->data, elementCount, val); mState->dataChanged = true; upload(); }

template <typename ElementT>
Buffer::Buffer(std::size_t elementCount,
    UsageFrequency frequency, UsageAccess access, Target target, const ElementT&)
{ reset(frequency, access, target); fillDataHolder(*mState->data, elementCount, nullptr); mState->dataChanged = true; upload(); }

template <typename ElementT>
Buffer::Ptr Buffer::createEmpty(std::size_t elementCount,
    UsageFrequency frequency, UsageAccess access, Target target)
{ return instantiate(elementCount, frequency, access, target, ElementT{ }); }

template <typename T>
T* Buffer::as()
{ return mState->data->as<T>(); }
template <typename T>
const T* Buffer::as() const
{ return mState->data->as<T>(); }

template <typename T>
std::vector<T> &Buffer::asVector()
{ return mState->data->data<T>(); }
template <typename T>
const std::vector<T> &Buffer::asVector() const
{ return mState->data->data<T>(); }

template <typename T>
bool Buffer::usesType() const
{ return mState->data->usesType<T>(); }

template <typename T>
void Buffer::clear(const T &val)
{ fillDataHolder(*mState->data, val); mState->dataChanged = true; }

template <typename T>
void Buffer::BufferDataHolder::DataArray<T>::uploadBuffer(
    const BufferDataHolder &data, BufferHolder &buffer,
    Target target, UsageFrequency frequency, UsageAccess access) const
{ Buffer::uploadBuffer(buffer, data.elementSize * data.elementCount, target, frequency, access, data.data()); }
template <typename T>
void Buffer::BufferDataHolder::DataArray<T>::downloadBuffer(
    BufferDataHolder &data, const BufferHolder &buffer, Target target)
{
    const auto bufferCompatible{
        d.size() * sizeof(d[0]) == static_cast<std::size_t>(buffer.byteSize)
    };
    if (!bufferCompatible)
    {
        assert(buffer.byteSize % sizeof(d[0]) == 0u);
        const auto requiredElements{ buffer.byteSize / sizeof(d[0]) };
        d.resize(requiredElements);
        data.elementCount = requiredElements;
        data.elementsAllocated = requiredElements;
    }
    Buffer::downloadBuffer(buffer, target, d.data());
}
template <typename T>
void *Buffer::BufferDataHolder::DataArray<T>::data()
{ return static_cast<void*>(d.data()); }
template <typename T>
const void *Buffer::BufferDataHolder::DataArray<T>::data() const
{ return static_cast<const void*>(d.data()); }

template <typename T>
T *Buffer::BufferDataHolder::as()
{
    if (elementCount != elementsAllocated)
    { return nullptr; }
    const auto ptr{ std::dynamic_pointer_cast<DataArray<T>>(array) };
    if (!ptr)
    { throw std::runtime_error("Unable to cast buffer content as<>: Invalid request, buffer of different type!"); }
    return ptr->d.data();
}
template <typename T>
const T *Buffer::BufferDataHolder::as() const
{
    if (elementCount != elementsAllocated)
    { return nullptr; }
    const auto ptr{ std::dynamic_pointer_cast<DataArray<T>>(array) };
    if (!ptr)
    { throw std::runtime_error("Unable to cast buffer content as<>: Invalid request, buffer of different type!"); }
    return ptr->d.data();
}

template <typename T>
std::vector<T> &Buffer::BufferDataHolder::data()
{
    const auto ptr{ std::dynamic_pointer_cast<DataArray<T>>(array) };
    if (!ptr)
    { throw std::runtime_error("Unable to get buffer data<>: Invalid request, buffer of different type!"); }
    return ptr->d;
}

template <typename T>
bool Buffer::BufferDataHolder::usesType() const
{
    const auto ptr{ std::dynamic_pointer_cast<DataArray<T>>(array) };
    return ptr != nullptr;
}

template <typename T>
std::shared_ptr<Buffer::BufferDataHolder::DataArray<T>> Buffer::fillDataHolderNoInitialize(
    BufferDataHolder &holder, std::size_t elementCount) const
{
    using ElementT = T;
    const auto elementSize{ sizeof(ElementT) };
    const auto arrayPtr{ std::make_shared<BufferDataHolder::DataArray<T>>() };

    fillDataHolderNoInitialize(
        holder, elementCount, elementSize,
        std::static_pointer_cast<BufferDataHolder::DataArrayBase>(arrayPtr)
    );

    return arrayPtr;
}

template <typename T>
void Buffer::fillDataHolder(BufferDataHolder &holder,
    std::size_t elementCount, const T *data) const
{
    const auto arrayPtr{ fillDataHolderNoInitialize<T>(holder, elementCount) };
    if (data)
    { arrayPtr->d = std::vector<T>(data, data + holder.elementCount); holder.elementsAllocated = holder.elementCount; }
    else
    { holder.elementsAllocated = 0u; }
}

// Base case where we use the value as initializer:
template <typename T>
void Buffer::fillDataHolder(BufferDataHolder &holder,
    std::size_t elementCount, const T &initial) const
{
    const auto arrayPtr{ fillDataHolderNoInitialize<T>(holder, elementCount) };
    arrayPtr->d.resize(holder.elementCount, initial);
    holder.elementsAllocated = holder.elementCount;
}

template <typename ItT>
void Buffer::fillDataHolder(BufferDataHolder &holder,
    std::size_t elementCount, const ItT &begin, const ItT &end) const
{
    using T = typename std::iterator_traits<ItT>::value_type;
    const auto arrayPtr{ fillDataHolderNoInitialize<T>(holder, elementCount) };

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
void Buffer::fillDataHolder(BufferDataHolder &holder,
    const T &initial) const
{
    if (!holder.hasData())
    { return; }
    const auto &data{ holder.data<T>() };

    data = decltype(data)(data.size(), initial);
}

template <typename T>
void Buffer::wrapBuffer(InternalState &state,
    GLuint id, std::size_t elementCount, bool destroy)
{
    fillDataHolder<T>(*state.data, elementCount, nullptr);
    state.buffer->byteSize = sizeof(T) * elementCount;
    state.buffer->id = id;
    state.buffer->destroy = destroy;
    state.dataChanged = false;
}

} // namespace treerndr

inline std::ostream &operator<<(std::ostream &out, const treerndr::Buffer &buffer)
{ buffer.describe(out); return out; }

// Template implementation end.

#endif // TREE_BUFFER_H
