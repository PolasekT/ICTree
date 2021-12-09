/**
 * @author Tomas Polasek
 * @date 16.4.2020
 * @version 1.0
 * @brief Wrapper around OpenGL buffer.
 */

#include "TreeBuffer.h"

namespace treerndr
{

Buffer::Buffer()
{ reset(); }
Buffer::~Buffer()
{ /* Automatic */ }

void Buffer::reset()
{ mState = createInternalState(); }

void Buffer::reset(Buffer::UsageFrequency frequency, Buffer::UsageAccess access, Buffer::Target target)
{
    mState = createInternalState();
    mState->usageFrequency = frequency; mState->usageAccess = access; mState->target = target;
}

bool Buffer::upload()
{ return uploadBufferData(*mState->data, *mState->buffer); }

bool Buffer::download()
{ return downloadBufferData(*mState->data, *mState->buffer); }

void Buffer::setDataChanged()
{ mState->dataChanged = true; }

bool Buffer::bind(Buffer::Target target) const
{
    if (mState->buffer->id == 0u)
    { return false; }

    // Const-cast is safe since we only change the data updated flag.
    const_cast<Buffer*>(this)->upload();

    glBindBuffer(treeutil::bufferTargetToGLEnum(target), mState->buffer->id);
    return true;
}

bool Buffer::bind(std::size_t layout, Target target) const
{
    if (mState->buffer->id == 0u)
    { return false; }

    // Const-cast is safe since we only change the data updated flag.
    const_cast<Buffer*>(this)->upload();

    const auto glTarget{ treeutil::bufferTargetToGLEnum(target) };
    glBindBuffer(glTarget, mState->buffer->id);
    glBindBufferBase(glTarget, static_cast<GLuint>(layout), mState->buffer->id);

    return true;
}

GLuint Buffer::id() const
{ return mState->buffer->id; }

std::size_t Buffer::elementCount() const
{ return mState->data->elementsAllocated; }

bool Buffer::empty() const
{ return mState->data->elementsAllocated == 0u; }

void Buffer::describe(std::ostream &out, const std::string &indent) const
{
    out << "[ Buffer: \n"
        << indent << "\tData Element Size = " << mState->data->elementSize << "\n"
        << indent << "\tData Element Count = " << mState->data->elementCount << "\n"
        << indent << "\tData Elements Allocated = " << mState->data->elementsAllocated << "\n"
        << indent << "\tBuffer Byte Size = " << mState->buffer->byteSize << "\n"
        << indent << "\tBuffer Usage = " << treeutil::bufferUsageToStr(mState->buffer->usageFrequency, mState->usageAccess) << "\n"
        << indent << "\tBuffer Default Target = " << treeutil::bufferTargetToStr(mState->target) << "\n"
        << indent << "\tBuffer ID = " << mState->buffer->id << "\n"
        << indent << " ]";
}

void *Buffer::BufferDataHolder::data()
{ return elementCount == elementsAllocated ? array->data() : nullptr; }
void *Buffer::BufferDataHolder::data() const
{ return elementCount == elementsAllocated ? array->data() : nullptr; }

bool Buffer::BufferDataHolder::hasData() const
{ return array && elementsAllocated == elementCount; }

Buffer::BufferHolder::~BufferHolder()
{
    if (id && destroy)
    { glDeleteBuffers(1u, &id); id = 0u; }
}

bool Buffer::BufferHolder::generateBuffer()
{
    if (!id)
    { glCreateBuffers(1u, &id); return true; }
    else
    { return false; }
}

std::shared_ptr<Buffer::InternalState> Buffer::createInternalState() const
{
    const auto state{ std::make_shared<InternalState>() };
    state->data = createDataHolder(); state->buffer = createBufferHolder(); state->dataChanged = true;
    return state;
}

std::shared_ptr<Buffer::BufferDataHolder> Buffer::createDataHolder() const
{ const auto holderPtr{ std::make_shared<BufferDataHolder>() }; return holderPtr; }

void Buffer::fillDataHolderNoInitialize(Buffer::BufferDataHolder &holder, std::size_t elementCount,
    std::size_t elementSize, const std::shared_ptr<BufferDataHolder::DataArrayBase> &arrayPtr) const
{
    holder.elementSize = elementSize;
    holder.elementCount = elementCount;
    holder.array = arrayPtr;
}

std::shared_ptr<Buffer::BufferHolder> Buffer::createBufferHolder() const
{ const auto holderPtr{ std::make_shared<BufferHolder>() }; return holderPtr; }

bool Buffer::uploadBufferData(const BufferDataHolder &data, BufferHolder &buffer)
{
    if (!mState->dataChanged)
    { return false; }

    data.array->uploadBuffer(data, buffer, mState->target, mState->usageFrequency, mState->usageAccess);

    mState->dataChanged = false;

    return true;
}

void Buffer::uploadBuffer(BufferHolder &buffer, std::size_t byteSize,
    Target target, UsageFrequency frequency,
    UsageAccess access, const void *data)
{
    const auto bufferCreated{ buffer.generateBuffer() };
    const auto bufferCompatible{
        !bufferCreated && static_cast<std::size_t>(buffer.byteSize) == byteSize &&
        buffer.usageFrequency == frequency && buffer.usageAccess == access
    };

    const auto glTarget{ treeutil::bufferTargetToGLEnum(target) };
    const auto glUsage{ treeutil::bufferUsageToGLEnum(frequency, access) };

    glBindBuffer(glTarget, buffer.id);

    // Either substitute data or just create new buffer.
    if (bufferCompatible)
    { if (data) glBufferSubData(glTarget, 0, byteSize, data); }
    else
    { glBufferData(glTarget, byteSize, data, glUsage); }

    buffer.byteSize = static_cast<GLsizei>(byteSize);
    buffer.usageFrequency = frequency;
    buffer.usageAccess = access;
}

bool Buffer::downloadBufferData(Buffer::BufferDataHolder &data, const BufferHolder &buffer)
{ data.array->downloadBuffer(data, buffer, mState->target); mState->dataChanged = false; return true; }

void Buffer::downloadBuffer(const BufferHolder &buffer, Target target, void *data)
{
    const auto glTarget{ treeutil::bufferTargetToGLEnum(target) };
    glBindBuffer(glTarget, buffer.id);
    glGetBufferSubData(glTarget, 0, buffer.byteSize, data);
}

} // namespace treerndr
