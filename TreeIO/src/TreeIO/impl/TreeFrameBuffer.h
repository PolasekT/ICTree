/**
 * @author Tomas Polasek
 * @date 16.4.2020
 * @version 1.0
 * @brief Wrapper around OpenGL frame buffer.
 */

#ifndef TREE_FRAME_BUFFER_H
#define TREE_FRAME_BUFFER_H

#include "TreeUtils.h"
#include "TreeGLUtils.h"

#include "TreeTextureBuffer.h"

namespace treerndr
{

/// @brief Frame buffer is a container for a set of internal or external texture buffers.
class FrameBuffer : public treeutil::PointerWrapper<FrameBuffer>
{
public:
    // Shortcuts:
    using AttachmentType = treeutil::FrameBufferAttachmentType;
    using Target = treeutil::FrameBufferTarget;

    /// @brief Specification of a single frame-buffer attachment.
    struct FrameBufferAttachment
    {
        /// @brief Specify attachment with given properties.
        FrameBufferAttachment(AttachmentType t, TextureBuffer *tex);
        /// @brief Specify attachment with given properties.
        FrameBufferAttachment(AttachmentType t, TextureBuffer &tex);
        /// @brief Specify attachment with given properties.
        FrameBufferAttachment(AttachmentType t, const TextureBuffer::Ptr &tex);

        /// Type of the attachment.
        AttachmentType type{ AttachmentType::Color };
        /// Texture to attach.
        TextureBuffer *texture{ nullptr };
    }; // struct FrameBufferAttachment

    /// @brief Create a handle to the default frame-buffer. New frame-buffer will be created when adding attachments!
    FrameBuffer();
    /// @brief Clean-up and destroy.
    ~FrameBuffer();

    /// @brief Create frame-buffer with provided list of attachments.
    FrameBuffer(std::initializer_list<FrameBufferAttachment> attachments);

    /// @brief Simple factory for creating color/depth frame-buffers with provided resolution.
    static Ptr createFrameBuffer(std::size_t width, std::size_t height,
        bool color = true, bool depth = true, bool stencil = false);

    /// @brief Simple factory for creating multi-sampled color/depth frame-buffers with provided resolution.
    static Ptr createMultiSampleFrameBuffer(std::size_t width, std::size_t height,
        bool color = true, bool depth = true, bool stencil = false, std::size_t samples = 2u);

    /// @brief Delete the current frame-buffer. No effect when no frame-buffer is created.
    void reset();

    /// @brief Clear color to given value. Specify attachment index to clear or negative value to clear all.
    void clearColor(const treeutil::Color &color, int8_t attachmentIdx = -1);
    /// @brief Clear color to given value. Specify attachment index to clear or negative value to clear all.
    void clearColor(const glm::vec4 &color, int8_t attachmentIdx = -1);

    /// @brief Clear depth to given value.
    void clearDepth(float depth);

    /// @brief Clear stencil to given value.
    void clearStencil(int stencil);

    /// @brief Clear depth and stencil to given values.
    void clearStencil(float depth, int stencil);

    /// @brief Clear the current list of attached textures.
    void clearAttachments();

    /// @brief Add attachment of given type. Returns index of the new attachment starting at zero.
    std::size_t addAttachment(TextureBuffer &texture, AttachmentType type);
    /// @brief Add attachment of given type. Returns index of the new attachment starting at zero.
    std::size_t addAttachment(const TextureBuffer::Ptr &texture, AttachmentType type);

    /// @brief Add new color attachment. Returns index of the new attachment starting at zero.
    std::size_t addColorAttachment(TextureBuffer &texture);
    /// @brief Add new color attachment. Returns index of the new attachment starting at zero.
    std::size_t addColorAttachment(const TextureBuffer::Ptr &texture);
    /// @brief Get list of currently attached color attachments.
    const std::vector<TextureBuffer::Ptr> &getColorAttachments();

    /// @brief Add new depth attachment. Returns index of the new attachment starting at zero.
    std::size_t addDepthAttachment(TextureBuffer &texture);
    /// @brief Add new depth attachment. Returns index of the new attachment starting at zero.
    std::size_t addDepthAttachment(const TextureBuffer::Ptr &texture);
    /// @brief Get currently attached depth attachment.
    const TextureBuffer::Ptr &getDepthAttachment();

    /// @brief Add new stencil attachment. Returns index of the new attachment starting at zero.
    std::size_t addStencilAttachment(TextureBuffer &texture);
    /// @brief Add new stencil attachment. Returns index of the new attachment starting at zero.
    std::size_t addStencilAttachment(const TextureBuffer::Ptr &texture);
    /// @brief Get currently attached stencil attachment.
    const TextureBuffer::Ptr &getStencilAttachment();

    /// @brief Add new depth-stencil attachment. Returns index of the new attachment starting at zero.
    std::size_t addDepthStencilAttachment(TextureBuffer &texture);
    /// @brief Add new depth-stencil attachment. Returns index of the new attachment starting at zero.
    std::size_t addDepthStencilAttachment(const TextureBuffer::Ptr &texture);
    /// @brief Get currently attached depth-stencil attachment.
    const TextureBuffer::Ptr &getDepthStencilAttachment();

    /// @brief Override current viewport settings with provided values. This is necessary for default frame-buffer.
    void setViewport(std::size_t x, std::size_t y, std::size_t width, std::size_t height);

    /// @brief Bind the frame-buffer to given target.
    void bind(Target target = Target::FrameBuffer) const;
    /// @brief Unbind frame-buffer, returning to default frame-buffer.
    void unbind(Target target = Target::FrameBuffer) const;

    /// @brief Bind the default frame-buffer to given target.
    static void bindDefault(Target target = Target::FrameBuffer);
    /// @brief Unbind frame-buffer, returning to default frame-buffer.
    static void unbindAll(Target target = Target::FrameBuffer);

    /// @brief Blit from source to destination framebuffer.
    static void blitFrameBuffers(const FrameBuffer &src, FrameBuffer &dst,
        std::size_t xOffset1, std::size_t yOffset1, std::size_t width1, std::size_t height1,
        std::size_t xOffset2, std::size_t yOffset2, std::size_t width2, std::size_t height2,
        const std::vector<AttachmentType> &attachmentTypes, TextureBuffer::Filter filter);

    /// @brief Get OpenGL frame-buffer handle/name.
    GLuint id() const;

    /// @brief Check whether the currently configured frame-buffer is complete and usable. Optionally returns string message.
    bool checkComplete(std::string *str = nullptr) const;

    /// @brief Does this frame-buffer represent the default frame-buffer?
    bool isDefaultFrameBuffer() const;

    /// @brief Print information about this frame-buffer buffer.
    void describe(std::ostream &out, const std::string &indent = "") const;
private:
    /// @brief Helper structure used to hold the frame buffer.
    struct FrameBufferHolder
    {
        /// @brief Automatically destroy held frame buffer.
        ~FrameBufferHolder();

        /// @brief Generate frame-buffer if necessary and return whether it was actually created.
        bool generateFrameBuffer();

        /// Current number of color attachments attached to the frame-buffer.
        std::size_t colorAttachmentCount{ 0u };
        /// Pixel at which the rendering starts.
        std::size_t viewportX{ 0u };
        /// Pixel at which the rendering starts.
        std::size_t viewportY{ 0u };
        /// Maximum width of the render area in pixels.
        std::size_t viewportWidth{ 0u };
        /// Maximum height of the render area in pixels.
        std::size_t viewportHeight{ 0u };
        /// Identifier of the frame buffer.
        GLuint id{ };
    }; // struct FrameBufferHolder

    /// @brief Helper for holding information about frame-buffer attachments.
    struct AttachmentsHolder
    {
        /// Did we change attachments?
        bool attachmentsChanged{ true };
        /// List of color attachments.
        std::vector<std::shared_ptr<TextureBuffer>> color{ };
        /// Depth attachment.
        std::shared_ptr<TextureBuffer> depth{ };
        /// Stencil attachment.
        std::shared_ptr<TextureBuffer> stencil{ };
        /// Depth-stencil attachment.
        std::shared_ptr<TextureBuffer> depthStencil{ };
    }; // struct AttachmentsHolder

    /// @brief Container for current internal state.
    struct InternalState
    {
        /// Holder for the currently managed frame-buffer.
        std::shared_ptr<FrameBufferHolder> frameBuffer{ };
        /// List of frame-buffer attachments.
        std::shared_ptr<AttachmentsHolder> attachments{ };
    }; // struct InternalState

    /// @brief Create the internal state and initialize.
    std::shared_ptr<InternalState> createInternalState() const;

    /// @brief Create the frame-buffer holder without initializing it.
    std::shared_ptr<FrameBufferHolder> createFrameBufferHolder() const;

    /// @brief Create the attachment holder without initializing it.
    std::shared_ptr<AttachmentsHolder> createAttachmentsHolder() const;

    /// @brief Clear the attachment holder.
    void clearAllAttachments();

    /// @brief Process given attachment and add it to the list of attachments. Returns index of the new attachment starting at 0.
    std::size_t addAttachment(const FrameBufferAttachment &attachment);

    /// @brief Update frame-buffer attachments if necessary.
    void updateAttachments();

    /// Current internal state.
    std::shared_ptr<InternalState> mState{ };
protected:
}; // class FrameBuffer

} // namespace treerndr

/// @brief Print information about the frame-buffer.
inline std::ostream &operator<<(std::ostream &out, const treerndr::FrameBuffer &frameBuffer);

// Template implementation begin.

namespace treerndr
{

} // namespace treerndr

inline std::ostream &operator<<(std::ostream &out, const treerndr::FrameBuffer &frameBuffer)
{ frameBuffer.describe(out); return out; }

// Template implementation end.

#endif // TREE_FRAME_BUFFER_H
