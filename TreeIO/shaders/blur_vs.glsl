/**
 * @author David Hrusa, Tomas Polasek
 * @date 5.20.2020
 * @version 1.0
 * @brief Vertex shader used for the blur shader.
 */

#version 400 core

/// Position on the input image.
out vec4 vPosition;

/// Input texture to be blurred.
uniform sampler2D uInput;
/// Whether to perform vertical (0) or horizontal (1) blurring pass.
uniform int uBlurHorizontal = 0;

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
