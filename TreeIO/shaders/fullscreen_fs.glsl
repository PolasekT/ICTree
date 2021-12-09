/**
 * @author Tomas Polasek
 * @date 10.14.2020
 * @version 1.0
 * @brief Fragment shader used for the fullscreen texture display.
 */

#version 400 core

/// Position on the input image.
in vec4 vPosition;
/// Final color of the fragment.
out vec4 fFragmentColor;

/// Input texture to be blurred.
uniform sampler2D uInput;

void main()
{
    vec4 color = texture(uInput, vPosition.st);

    fFragmentColor = color;
    gl_FragDepth = color.r;
}
