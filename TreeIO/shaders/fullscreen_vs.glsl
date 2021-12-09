/**
 * @author Tomas Polasek
 * @date 10.14.2020
 * @version 1.0
 * @brief Vertex shader used for the fullscreen texture display.
 */

#version 400 core

/// Position on the input image.
out vec4 vPosition;

/// Input texture.
uniform sampler2D uInput;

void main()
{
    // Generate full-screen triangle:
    vec2 xy = vec2(
        -1.0f + float((gl_VertexID & 1) << 2),
        -1.0f + float((gl_VertexID & 2) << 1)
    );
    gl_Position = vec4(xy.x, xy.y, 0.0f, 1.0f);

    // Generate UV coordinates:
    vec2 uv = vec2(
        (xy.x + 1.0f) * 0.5f,
        (xy.y + 1.0f) * 0.5f
    );
    vPosition = vec4(uv, 0.0f, 0.0f);
}
