/**
 * @author Tomas Polasek, David Hrusa
 * @date 4.14.2020
 * @version 1.0
 * @brief Concrete renderer systems used for rendering the current scene.
 */

#include "TreeRenderSystem.h"

namespace treerndr
{

RenderSystem::RenderSystem(const std::string &name) :
    mName{ name }
{ }
RenderSystem::~RenderSystem()
{ /* Automatic */ }

void RenderSystem::describe(std::ostream &out, const std::string &indent) const
{
    out << "[ RenderSystem: \n"
        << indent << "\tName = " << mName
        << indent << " ]";
}

const std::string &RenderSystem::identifier() const
{ return mName; }

} // namespace treerndr
