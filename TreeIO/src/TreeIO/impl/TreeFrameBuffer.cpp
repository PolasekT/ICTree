/**
 * @author Tomas Polasek
 * @date 16.4.2020
 * @version 1.0
 * @brief Wrapper around OpenGL frame buffer.
 */

#include "TreeFrameBuffer.h"

namespace treerndr
{

FrameBuffer::FrameBufferAttachment::FrameBufferAttachment(AttachmentType t, TextureBuffer *tex) :
    type{ t }, texture{ tex }
{ }
FrameBuffer::FrameBufferAttachment::FrameBufferAttachment(AttachmentType t, TextureBuffer &tex) :
    type{ t }, texture{ &tex }
{ }
FrameBuffer::FrameBufferAttachment::FrameBufferAttachment(AttachmentType t, const TextureBuffer::Ptr &tex) :
    type{ t }, texture{ tex.get() }
{ }

FrameBuffer::FrameBuffer()
{ reset(); }
FrameBuffer::~FrameBuffer()
{ /* Automatic */ }

FrameBuffer::FrameBuffer(std::initializer_list<FrameBufferAttachment> attachments)
{ reset(); for (const auto &att : attachments) addAttachment(att); }

FrameBuffer::Ptr FrameBuffer::createFrameBuffer(
    std::size_t width, std::size_t height,
    bool color, bool depth, bool stencil)
{
    const auto result{ FrameBuffer::instantiate() };

    if (color)
    { result->addColorAttachment(TextureBuffer::instantiate( width, height, glm::vec3{ } )); }

    if (depth && stencil)
    { result->addDepthStencilAttachment(TextureBuffer::instantiate( width, height, treeutil::Depth24Stencil8{ } )); }
    else
    {
        if (depth)
        { result->addDepthAttachment(TextureBuffer::instantiate( width, height, treeutil::Depth32f{ } )); }
        if (stencil)
        { result->addStencilAttachment(TextureBuffer::instantiate( width, height, treeutil::Stencil8{ } )); }
    }

    result->updateAttachments();

    std::string errorCheckString{ };
    if (!result->checkComplete(&errorCheckString))
    { Error << "Error while constructing FBO: \n\t" << errorCheckString << std::endl; }

    return result;
}

FrameBuffer::Ptr FrameBuffer::createMultiSampleFrameBuffer(
    std::size_t width, std::size_t height,
    bool color, bool depth, bool stencil,
    std::size_t samples)
{
    const auto result{ FrameBuffer::instantiate() };

    if (color)
    {
        result->addColorAttachment(
            TextureBuffer::instantiate( width, height,
                glm::vec3{ }, false, false, samples )
        );
    }

    if (depth && stencil)
    {
        result->addDepthStencilAttachment(
            TextureBuffer::instantiate( width, height,
                treeutil::Depth24Stencil8{ }, false, false, samples )
        );
    }
    else
    {
        if (depth)
        {
            result->addDepthAttachment(
                TextureBuffer::instantiate( width, height,
                    treeutil::Depth32f{ }, false, false, samples )
            );
        }
        if (stencil)
        {
            result->addStencilAttachment(
                TextureBuffer::instantiate( width, height,
                    treeutil::Stencil8{ }, false, false, samples )
            );
        }
    }

    result->updateAttachments();

    std::string errorCheckString{ };
    if (!result->checkComplete(&errorCheckString))
    { Error << "Error while constructing FBO: \n\t" << errorCheckString << std::endl; }

    return result;
}

void FrameBuffer::reset()
{ mState = createInternalState(); }

void FrameBuffer::clearColor(const treeutil::Color &color, int8_t attachmentIdx)
{ clearColor(glm::vec4{ color.r, color.g, color.b, color.a }, attachmentIdx); }

void FrameBuffer::clearColor(const glm::vec4 &color, int8_t attachmentIdx)
{
    updateAttachments();

    auto c{ color };
    const auto cPtr{ &c.r };

    if (mState->frameBuffer->id == 0u)
    { // Default frame-buffer.
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0u);
        glClearColor(c.x, c.y, c.z, c.w);
        glClear(GL_COLOR_BUFFER_BIT);
    }
    else
    { // Custom frame-buffer.
        if (attachmentIdx < 0)
        {
            for (int8_t attIdx = 0; attIdx < static_cast<int8_t>(mState->attachments->color.size()); ++attIdx)
            { glClearNamedFramebufferfv(mState->frameBuffer->id, GL_COLOR, attIdx, cPtr); }
        }
        else
        { glClearNamedFramebufferfv(mState->frameBuffer->id, GL_COLOR, attachmentIdx, cPtr); }
    }
}

void FrameBuffer::clearDepth(float depth)
{
    updateAttachments();
    if (mState->frameBuffer->id == 0u)
    { // Default frame-buffer.
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0u);
        glClearDepth(depth);
        glClear(GL_DEPTH_BUFFER_BIT);
    }
    else
    { // Custom frame-buffer.
        glClearNamedFramebufferfv(mState->frameBuffer->id, GL_DEPTH, 0, &depth);
    }
}

void FrameBuffer::clearStencil(int stencil)
{
    updateAttachments();
    if (mState->frameBuffer->id == 0u)
    { // Default frame-buffer.
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0u);
        glClearStencil(stencil);
        glClear(GL_STENCIL_BUFFER_BIT);
    }
    else
    { // Custom frame-buffer.
        glClearNamedFramebufferiv(mState->frameBuffer->id, GL_STENCIL, 0, &stencil);
    }
}

void FrameBuffer::clearStencil(float depth, int stencil)
{
    updateAttachments();
    if (mState->frameBuffer->id == 0u)
    { // Default frame-buffer.
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0u);
        glClearDepth(depth);
        glClearStencil(stencil);
        glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    }
    else
    { // Custom frame-buffer.
        glClearNamedFramebufferfi(mState->frameBuffer->id, GL_DEPTH_STENCIL, 0, depth, stencil);
    }
}

void FrameBuffer::clearAttachments()
{ clearAllAttachments(); }

std::size_t FrameBuffer::addAttachment(TextureBuffer &texture, AttachmentType type)
{ return addAttachment({ type, texture }); }
std::size_t FrameBuffer::addAttachment(const TextureBuffer::Ptr &texture, AttachmentType type)
{ return addAttachment({ type, texture }); }

std::size_t FrameBuffer::addColorAttachment(TextureBuffer &texture)
{ return addAttachment(texture, AttachmentType::Color); }
std::size_t FrameBuffer::addColorAttachment(const TextureBuffer::Ptr &texture)
{ return addAttachment(*texture, AttachmentType::Color); }
const std::vector<TextureBuffer::Ptr> &FrameBuffer::getColorAttachments()
{ return mState->attachments->color; }

std::size_t FrameBuffer::addDepthAttachment(TextureBuffer &texture)
{ return addAttachment(texture, AttachmentType::Depth); }
std::size_t FrameBuffer::addDepthAttachment(const TextureBuffer::Ptr &texture)
{ return addAttachment(*texture, AttachmentType::Depth); }
const TextureBuffer::Ptr &FrameBuffer::getDepthAttachment()
{ return mState->attachments->depth; }

std::size_t FrameBuffer::addStencilAttachment(TextureBuffer &texture)
{ return addAttachment(texture, AttachmentType::Stencil); }
std::size_t FrameBuffer::addStencilAttachment(const TextureBuffer::Ptr &texture)
{ return addAttachment(*texture, AttachmentType::Stencil); }
const TextureBuffer::Ptr &FrameBuffer::getStencilAttachment()
{ return mState->attachments->stencil; }

std::size_t FrameBuffer::addDepthStencilAttachment(TextureBuffer &texture)
{ return addAttachment(texture, AttachmentType::DepthStencil); }
std::size_t FrameBuffer::addDepthStencilAttachment(const TextureBuffer::Ptr &texture)
{ return addAttachment(*texture, AttachmentType::DepthStencil); }
const TextureBuffer::Ptr &FrameBuffer::getDepthStencilAttachment()
{ return mState->attachments->depthStencil; }

void FrameBuffer::setViewport(std::size_t x, std::size_t y, std::size_t width, std::size_t height)
{
    mState->frameBuffer->viewportX = x;
    mState->frameBuffer->viewportY = y;
    mState->frameBuffer->viewportWidth = width;
    mState->frameBuffer->viewportHeight = height;
}

void FrameBuffer::bind(Target target) const
{
    // Update sampler parameters if necessary. Const-cast is safe, since we only update dirty flag.
    const_cast<FrameBuffer*>(this)->updateAttachments();
    glBindFramebuffer(treeutil::frameBufferTargetToGLEnum(target), mState->frameBuffer->id);
    glViewport(
        0, 0,
        static_cast<GLsizei>(mState->frameBuffer->viewportWidth),
        static_cast<GLsizei>(mState->frameBuffer->viewportHeight)
    );
}

void FrameBuffer::unbind(Target target) const
{
    // Update sampler parameters if necessary. Const-cast is safe, since we only update dirty flag.
    const_cast<FrameBuffer*>(this)->updateAttachments();
    unbindAll(target);
}

void FrameBuffer::bindDefault(Target target)
{ glBindFramebuffer(treeutil::frameBufferTargetToGLEnum(target), 0); glViewport(0, 0, 0, 0); }

void FrameBuffer::unbindAll(Target target)
{ glBindFramebuffer(treeutil::frameBufferTargetToGLEnum(target), 0u); }

void FrameBuffer::blitFrameBuffers(const FrameBuffer &src, FrameBuffer &dst,
    std::size_t xOffset1, std::size_t yOffset1, std::size_t width1, std::size_t height1,
    std::size_t xOffset2, std::size_t yOffset2, std::size_t width2, std::size_t height2,
    const std::vector<AttachmentType> &attachmentTypes, TextureBuffer::Filter filter)
{
    src.bind(Target::ReadFrameBuffer);
    dst.bind(Target::DrawFrameBuffer);

    GLbitfield mask{ 0u };
    for (const auto &type : attachmentTypes)
    {
        switch (type)
        {
            default:
            case AttachmentType::Color:
            { mask |= GL_COLOR_BUFFER_BIT; break; }
            case AttachmentType::Depth:
            { mask |= GL_DEPTH_BUFFER_BIT; break; }
            case AttachmentType::Stencil:
            { mask |= GL_STENCIL_BUFFER_BIT; break; }
            case AttachmentType::DepthStencil:
            { mask |= GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT; break; }
        }
    }

    glBlitFramebuffer(
        static_cast<GLint>(xOffset1), static_cast<GLint>(yOffset1), static_cast<GLint>(width1), static_cast<GLint>(height1),
        static_cast<GLint>(xOffset2), static_cast<GLint>(yOffset2), static_cast<GLint>(width2), static_cast<GLint>(height2),
        mask, treeutil::textureFilterToGLEnum(filter)
    );

    dst.unbind(Target::DrawFrameBuffer);
    src.unbind(Target::ReadFrameBuffer);
}

GLuint FrameBuffer::id() const
{ return mState->frameBuffer->id; }

bool FrameBuffer::checkComplete(std::string *str) const
{ bind(Target::FrameBuffer); return treeutil::checkFrameBufferStatus(Target::FrameBuffer, str); }

bool FrameBuffer::isDefaultFrameBuffer() const
{ return id() == 0u; }

void FrameBuffer::describe(std::ostream &out, const std::string &indent) const
{
    std::string completeString{ };
    const auto isComplete{ checkComplete(&completeString) };

    out << "[ Frame Buffer: \n"
        << indent << "\tFrame Buffer ID = " << mState->frameBuffer->id << "\n"
        << indent << "\tIs Default Frame Buffer = " << (isDefaultFrameBuffer() ? "yes" : "no") << "\n"
        << indent << "\tIs Complete = " << (isComplete ? "yes" : completeString) << "\n"
        << indent << "\tColor Attachments = " << (mState->attachments->color.empty() ? "NONE\n" : "\n");
    for (std::size_t iii = 0u; iii < mState->attachments->color.size(); ++iii)
    {
        out << indent << "\t\t" << iii << " : ";
        mState->attachments->color[iii]->describe(out, indent + "\t\t");
        out << "\n";
    }
    out << indent << "\tDepth Attachment = " << (mState->attachments->depth ? "NONE\n" : "\n");
    if (mState->attachments->depth) mState->attachments->depth->describe(out, indent + "\t\t");
    out << "\n" << indent << "\tStencil Attachment = " << (mState->attachments->stencil ? "NONE\n" : "\n");
    if (mState->attachments->stencil) mState->attachments->stencil->describe(out, indent + "\t\t");
    out << "\n" << indent << "\tDepth-Stencil Attachment = " << (mState->attachments->depthStencil ? "NONE\n" : "\n");
    if (mState->attachments->depthStencil) mState->attachments->depthStencil->describe(out, indent + "\t\t");
    out << "\n" << indent << " ]";
}

FrameBuffer::FrameBufferHolder::~FrameBufferHolder()
{
    if (id)
    { glDeleteFramebuffers(1u, &id); id = 0u; }
}

bool FrameBuffer::FrameBufferHolder::generateFrameBuffer()
{
    if (!id)
    { *this = { }; glCreateFramebuffers(1u, &id); return true; }
    else
    { return false; }
}

std::shared_ptr<FrameBuffer::InternalState> FrameBuffer::createInternalState() const
{
    const auto state{ std::make_shared<InternalState>() };
    state->frameBuffer = createFrameBufferHolder(); state->attachments = createAttachmentsHolder();
    return state;
}

std::shared_ptr<FrameBuffer::FrameBufferHolder> FrameBuffer::createFrameBufferHolder() const
{ const auto holderPtr{ std::make_shared<FrameBufferHolder>() }; return holderPtr; }

std::shared_ptr<FrameBuffer::AttachmentsHolder> FrameBuffer::createAttachmentsHolder() const
{ const auto holderPtr{ std::make_shared<AttachmentsHolder>() }; return holderPtr; }

void FrameBuffer::clearAllAttachments()
{ *mState->attachments = { }; }

std::size_t FrameBuffer::addAttachment(const FrameBufferAttachment &attachment)
{
    // Make sure the texture is created before adding it to the frame-buffer.
    attachment.texture->upload();

    switch (attachment.type)
    {
        case treeutil::FrameBufferAttachmentType::Color:
        { // We got a new color attachment:
            mState->attachments->color.emplace_back(std::make_shared<TextureBuffer>(*attachment.texture));
            mState->attachments->attachmentsChanged = true;
            return mState->attachments->color.size() - 1u;
        }
        case treeutil::FrameBufferAttachmentType::Depth:
        { // We got a new depth attachment:
            // TODO - Remove the DepthStencil attachment?
            mState->attachments->depth = std::make_shared<TextureBuffer>(*attachment.texture);
            mState->attachments->attachmentsChanged = true;
            return 0u;
        }
        case treeutil::FrameBufferAttachmentType::Stencil:
        { // We got a new color attachment:
            // TODO - Remove the DepthStencil attachment?
            mState->attachments->stencil = std::make_shared<TextureBuffer>(*attachment.texture);
            mState->attachments->attachmentsChanged = true;
            return 0u;
        }
        case treeutil::FrameBufferAttachmentType::DepthStencil:
        { // We got a new color attachment:
            // TODO - Remove the Depth and Stencil attachments?
            mState->attachments->depthStencil= std::make_shared<TextureBuffer>(*attachment.texture);
            mState->attachments->attachmentsChanged = true;
            return 0u;
        }
        default:
        { // Unknown attachment type:
            Error << "Unable to addAttachment: Unknown frame-buffer attachment type!" << std::endl;
            return 0u;
        }
    }
}

void FrameBuffer::updateAttachments()
{
    if (mState->attachments->attachmentsChanged && mState->frameBuffer->id == 0u &&
        (!mState->attachments->color.empty() || mState->attachments->depth ||
         mState->attachments->stencil || mState->attachments->depthStencil))
    { mState->frameBuffer->generateFrameBuffer(); }

    if (!mState->attachments->attachmentsChanged || mState->frameBuffer->id == 0u)
    { return; }

    auto viewportWidth{ std::numeric_limits<std::size_t>::max() };
    auto viewportHeight{ std::numeric_limits<std::size_t>::max() };

    // Remove all of the old color attachments:
    for (GLint attIdx = 0; attIdx < static_cast<GLint>(mState->frameBuffer->colorAttachmentCount); ++attIdx)
    { glNamedFramebufferTexture(mState->frameBuffer->id, static_cast<GLenum>(GL_COLOR_ATTACHMENT0 + attIdx), 0u, 0); }

    // Attach new list of attachments:
    for (GLint attIdx = 0; attIdx < static_cast<GLint>(mState->attachments->color.size()); ++attIdx)
    { // Color attachments:
        const auto &texture{ *mState->attachments->color[attIdx] };
        glNamedFramebufferTexture(mState->frameBuffer->id,
            static_cast<GLenum>(GL_COLOR_ATTACHMENT0 + attIdx), texture.id(), 0);
        viewportWidth = std::min<std::size_t>(viewportWidth, texture.width());
        viewportHeight = std::min<std::size_t>(viewportHeight, texture.width());
    }
    // Depth attachment:
    if (mState->attachments->depth)
    {
        const auto &texture{ *mState->attachments->depth };
        glNamedFramebufferTexture(mState->frameBuffer->id,
            GL_DEPTH_ATTACHMENT, texture.id(), 0);
        viewportWidth = std::min<std::size_t>(viewportWidth, texture.width());
        viewportHeight = std::min<std::size_t>(viewportHeight, texture.width());
    }
    // Stencil attachment:
    if (mState->attachments->stencil)
    {
        const auto &texture{ *mState->attachments->stencil };
        glNamedFramebufferTexture(mState->frameBuffer->id,
            GL_STENCIL_ATTACHMENT, texture.id(), 0);
        viewportWidth = std::min<std::size_t>(viewportWidth, texture.width());
        viewportHeight = std::min<std::size_t>(viewportHeight, texture.width());
    }
    // Depth-stencil attachment:
    if (mState->attachments->depthStencil)
    {
        const auto &texture{ *mState->attachments->depthStencil };
        glNamedFramebufferTexture(mState->frameBuffer->id,
            GL_DEPTH_STENCIL_ATTACHMENT, texture.id(), 0);
        viewportWidth = std::min<std::size_t>(viewportWidth, texture.width());
        viewportHeight = std::min<std::size_t>(viewportHeight, texture.width());
    }

    mState->frameBuffer->colorAttachmentCount = mState->attachments->color.size();
    mState->frameBuffer->viewportWidth = viewportWidth == std::numeric_limits<std::size_t>::max() ? 0u : viewportWidth;
    mState->frameBuffer->viewportHeight = viewportHeight == std::numeric_limits<std::size_t>::max() ? 0u : viewportHeight;
    mState->attachments->attachmentsChanged = false;
}

} // namespace treerndr
