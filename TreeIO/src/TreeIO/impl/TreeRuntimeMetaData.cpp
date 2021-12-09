/**
 * @author Tomas Polasek, David Hrusa
 * @date 11.8.2020
 * @version 1.0
 * @brief Implementation of tree runtime metadata used by our application (runtime metadata is always application specfic.)
 */

#include "TreeRuntimeMetaData.h"

#include "TreeScene.h"

namespace treeio
{

void RuntimeTreeProperties::applyScales(const treerndr::MeshInstancePtr &instance,
    bool setScales, bool setPositions, bool setVisibility, bool setColors)
{
    // Points and lines are the same, but we separate them since their visibility may be toggled separately.
    for (const auto &s: instancePointNames)
    {
        if (instance->name == s)
        {
            if (setScales)
            { instance->scale = scaleGraph; }
            if (setPositions)
            { instance->translate = { offsetGraph[0], offsetGraph[1], offsetGraph[2] }; }
            if (setVisibility)
            { instance->show = showPoints; }
        }
    }
    for (const auto &s: instanceSegmentNames)
    {
        if (instance->name == s)
        {
            if (setScales)
            { instance->scale = scaleGraph; }
            if (setPositions)
            { instance->translate = { offsetGraph[0], offsetGraph[1], offsetGraph[2] }; }
            if (setVisibility)
            { instance->show = showSegments; }
        }
    }
    for (const auto &s: instanceReferenceNames)
    {
        if (instance->name == s)
        {
            if (setScales)
            { instance->scale = scaleReference; }
            if (setPositions)
            { instance->translate = { offsetReference[0], offsetReference[1], offsetReference[2] }; }
            if (setVisibility)
            { instance->show = showReference; }
            if (setColors)
            { instance->overrideColor = { colorReference[0], colorReference[1], colorReference[2], colorReference[3] }; }
        }
    }
    for (const auto &s: instanceReconstructionNames)
    {
        if (instance->name == s)
        {
            auto &metaData{ instance->mesh->tree->currentTree().metaData() };
            auto &visParameters{ instance->mesh->reconstruction->parameters() };
            if (setScales)
            {
                instance->scale = scaleReconstruction;

                // TODO - Make this work for multiple tree instances?
                // Apply reverse scaling for the branch widths.
                const auto scalingCoefficient{ scaleReconstruction / appliedScaleReference };
                metaData.branchWidthMultiplier *= scalingCoefficient;
                visParameters.branchWidthMultiplier = metaData.branchWidthMultiplier;
                appliedScaleReference = scaleReconstruction;
            }
            if (setPositions)
            { instance->translate = { offsetReconstruction[0], offsetReconstruction[1], offsetReconstruction[2] }; }
            if (setVisibility)
            { instance->show = showReconstruction; }
            if (setColors)
            { instance->overrideColor = { colorReconstruction[0], colorReconstruction[1], colorReconstruction[2], colorReconstruction[3] }; }
        }
    }
}

void RuntimeTreeProperties::changedScaleBase()
{
    auto rescaleCoefficient{ scaleBase / appliedScaleBase };
    if (rescaleCoefficient == 0)
    { // Protect from being zeroed out (while using a GUI slider)
        scaleBase = 0.001f;
        rescaleCoefficient = scaleBase / appliedScaleBase;
    }
    scaleReference *= rescaleCoefficient;
    scaleGraph *= rescaleCoefficient;
    scaleReconstruction *= rescaleCoefficient;
    appliedScaleBase = scaleBase;
}

TreeRuntimeMetaData::Ptr RuntimeMetaData::duplicate() const
{ return treeutil::WrapperCtrT<RuntimeMetaData>(*this); }

void RuntimeMetaData::onLoad(TreeMetaData &metaData)
{
    runtimeTreeProperties.scaleBase = metaData.baseScale;
    runtimeTreeProperties.appliedScaleBase = runtimeTreeProperties.scaleBase;

    runtimeTreeProperties.scaleGraph = metaData.skeletonScale;

    runtimeTreeProperties.scaleReconstruction = metaData.reconstructionScale;

    runtimeTreeProperties.scaleReference = metaData.referenceScale;
    runtimeTreeProperties.appliedScaleReference = runtimeTreeProperties.scaleReference;
}

void RuntimeMetaData::onSave(TreeMetaData &metaData)
{
    metaData.baseScale = runtimeTreeProperties.scaleBase;
    metaData.skeletonScale = runtimeTreeProperties.scaleGraph;
    metaData.reconstructionScale = runtimeTreeProperties.scaleReconstruction;
    metaData.referenceScale = runtimeTreeProperties.scaleReference;
}

} // namespace treeio
