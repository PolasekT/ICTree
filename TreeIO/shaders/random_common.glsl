/**
 * @author David Hrusa, Tomas Polasek
 * @date 5.22.2020
 * @version 1.0
 * @brief Basic library for pseudo-random number generation.
 */

/// @brief Create pseudo-random value from vec3.
float random(vec3 seed)
{ return fract(sin(dot(seed.xyz, vec3(12.9898f, 78.233f, 131.1337f))) * 43758.5453123); }
