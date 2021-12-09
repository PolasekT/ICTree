/**
 * @author Tomas Polasek, David Hrusa
 * @date 1.15.2020
 * @version 1.0
 * @brief Scene representation for the main viewport.
 */

#ifndef TREE_SCENE_INSTANCE_NAMES_H
#define TREE_SCENE_INSTANCE_NAMES_H

namespace treescene
{

namespace instances
{

/// Name of the object containing tree skeleton points.
static constexpr const char *POINTS_NAME{ "points" };
/// Name of the object containing tree skeleton segments.
static constexpr const char *SEGMENTS_NAME{ "segments" };
/// Name of the object containing the reference model.
static constexpr const char *REFERENCE_NAME{ "reference" };
/// Name of the object containing tree skeleton segments.
static constexpr const char* RECONSTRUCTION_NAME{ "reconstruction" };
/// Name of the object containing the floor grid.
static constexpr const char *GRID_NAME{ "grid" };
/// Name of the object containing the floor ground plane.
static constexpr const char *PLANE_NAME{ "plane" };
/// Name of the object representing the light.
static constexpr const char *LIGHT_NAME{ "lightbulb" };

} // namespace instances

} // namespace treescene

#endif // TREE_SCENE_INSTANCE_NAMES_H
