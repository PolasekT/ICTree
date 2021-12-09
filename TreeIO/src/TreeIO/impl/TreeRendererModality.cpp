/**
 * @author Tomas Polasek
 * @date 7.14.2020
 * @version 1.0
 * @brief Support for displaying different modalities.
 */

#include "TreeRendererModality.h"

namespace treerndr
{

RendererModality::RendererModality(DisplayModality modality)
{ fromModality(modality); }

RendererModality::RendererModality(const std::string &name)
{ fromName(name); }

RendererModality::RendererModality(std::size_t idx)
{ fromIdx(idx); }

DisplayModality RendererModality::modality() const
{ return mModality; }

std::string RendererModality::name() const
{
    switch (mModality)
    {
        default:
        case DisplayModality::Shaded:
        { return "Shaded"; }
        case DisplayModality::Albedo:
        { return "Albedo"; }
        case DisplayModality::Light:
        { return "Light"; }
        case DisplayModality::Shadow:
        { return "Shadow"; }
        case DisplayModality::Normal:
        { return "Normal"; }
        case DisplayModality::Depth:
        { return "Depth"; }
    }
}

std::size_t RendererModality::idx() const
{
    switch (mModality)
    {
        default:
        case DisplayModality::Shaded:
        { return 0u; }
        case DisplayModality::Albedo:
        { return 1u; }
        case DisplayModality::Light:
        { return 2u; }
        case DisplayModality::Shadow:
        { return 3u; }
        case DisplayModality::Normal:
        { return 4u; }
        case DisplayModality::Depth:
        { return 5u; }
    }
}

void RendererModality::fromModality(DisplayModality modality)
{ mModality = modality; }

void RendererModality::fromName(const std::string &name)
{
    if (name == "Shaded")
    { fromModality(DisplayModality::Shaded); }
    else if (name == "Albedo")
    { fromModality(DisplayModality::Albedo); }
    else if (name == "Light")
    { fromModality(DisplayModality::Light); }
    else if (name == "Shadow")
    { fromModality(DisplayModality::Shadow); }
    else if (name == "Normal")
    { fromModality(DisplayModality::Normal); }
    else if (name == "Depth")
    { fromModality(DisplayModality::Depth); }
    else
    { fromModality(DisplayModality::Shaded); }
}

void RendererModality::fromIdx(std::size_t idx)
{
    switch (idx)
    {
        default:
        case 0u:
        { fromModality(DisplayModality::Shaded); break; }
        case 1u:
        { fromModality(DisplayModality::Albedo); break; }
        case 2u:
        { fromModality(DisplayModality::Light); break; }
        case 3u:
        { fromModality(DisplayModality::Shadow); break; }
        case 4u:
        { fromModality(DisplayModality::Normal); break; }
        case 5u:
        { fromModality(DisplayModality::Depth); break; }
    }
}

treeutil::Color RendererModality::getClearColor() const
{
    switch (mModality)
    {
        default:
        case DisplayModality::Shaded:
        { return treeutil::Color{ 1.0f, 1.0f, 1.0f, 0.0f }; }
        case DisplayModality::Albedo:
        { return treeutil::Color{ 0.36f, 0.80f, 1.0f, 0.0f }; }
        case DisplayModality::Light:
        { return treeutil::Color{ 0.0f, 0.0f, 0.0f, 0.0f }; }
        case DisplayModality::Shadow:
        { return treeutil::Color{ 0.0f, 0.0f, 0.0f, 0.0f }; }
        case DisplayModality::Normal:
        { return treeutil::Color{ 0.0f, 0.0f, 0.0f, 0.0f }; }
        case DisplayModality::Depth:
        { return treeutil::Color{ 0.0f, 0.0f, 0.0f, 0.0f }; }
    }
}

} // namespace treerndr
