/**
 * @author David Hrusa, Tomas Polasek
 * @date 5.20.2020
 * @version 1.0
 * @brief Fragment shader used for the blur shader.
 */

#version 400 core

/// Position on the input image.
in vec4 vPosition;
/// Final color of the fragment.
out vec4 fFragmentColor;

/// Input texture to be blurred.
uniform sampler2D uInput;
/// Whether to perform vertical (0) or horizontal (1) blurring pass.
uniform int uBlurHorizontal = 0;

// Relative filter weights indexed by distance from "home" texel
//    This set for 9-texel sampling
#define WT9_0 1.0
#define WT9_1 0.8
#define WT9_2 0.6
#define WT9_3 0.4
#define WT9_4 0.2

#define WT9_NORMALIZE (WT9_0+2.0*(WT9_1+WT9_2+WT9_3+WT9_4))

/// @brief Perform horizontal blur pass.
vec4 blurH(sampler2D tex, vec2 coord, vec2 dimensions, float stride)
{
    float TexelIncrement = stride / dimensions.x;
    //float3 Coord = float3(TexCoord.xy+QuadTexelOffsets.xy, 1);

    vec2 c0 = vec2(coord.x + TexelIncrement * 1.0f, coord.y);
    vec2 c1 = vec2(coord.x + TexelIncrement * 2.0f, coord.y);
    vec2 c2 = vec2(coord.x + TexelIncrement * 3.0f, coord.y);
    vec2 c3 = vec2(coord.x + TexelIncrement * 4.0f, coord.y);
    vec2 c4 = vec2(coord.x                        , coord.y);
    vec2 c5 = vec2(coord.x - TexelIncrement * 1.0f, coord.y);
    vec2 c6 = vec2(coord.x - TexelIncrement * 2.0f, coord.y);
    vec2 c7 = vec2(coord.x - TexelIncrement * 3.0f, coord.y);
    vec2 c8 = vec2(coord.x - TexelIncrement * 4.0f, coord.y);

    vec4 OutCol;

    OutCol  = texture(tex, c0) * (WT9_1/WT9_NORMALIZE);
    OutCol += texture(tex, c1) * (WT9_2/WT9_NORMALIZE);
    OutCol += texture(tex, c2) * (WT9_3/WT9_NORMALIZE);
    OutCol += texture(tex, c3) * (WT9_4/WT9_NORMALIZE);
    OutCol += texture(tex, c4) * (WT9_0/WT9_NORMALIZE);
    OutCol += texture(tex, c5) * (WT9_1/WT9_NORMALIZE);
    OutCol += texture(tex, c6) * (WT9_2/WT9_NORMALIZE);
    OutCol += texture(tex, c7) * (WT9_3/WT9_NORMALIZE);
    OutCol += texture(tex, c8) * (WT9_4/WT9_NORMALIZE);

    return OutCol;
}

/// @brief Perform vertical blur pass.
vec4 blurV(sampler2D tex, vec2 coord, vec2 dimensions, float stride)
{
    float TexelIncrement = stride / dimensions.y;
    //float3 Coord = float3(TexCoord.xy+QuadTexelOffsets.xy, 1);

    vec2 c0 = vec2(coord.x, coord.y + TexelIncrement * 1.0f);
    vec2 c1 = vec2(coord.x, coord.y + TexelIncrement * 2.0f);
    vec2 c2 = vec2(coord.x, coord.y + TexelIncrement * 3.0f);
    vec2 c3 = vec2(coord.x, coord.y + TexelIncrement * 4.0f);
    vec2 c4 = vec2(coord.x, coord.y);
    vec2 c5 = vec2(coord.x, coord.y - TexelIncrement * 1.0f);
    vec2 c6 = vec2(coord.x, coord.y - TexelIncrement * 2.0f);
    vec2 c7 = vec2(coord.x, coord.y - TexelIncrement * 3.0f);
    vec2 c8 = vec2(coord.x, coord.y - TexelIncrement * 4.0f);

    vec4 OutCol;

    OutCol  = texture(tex, c0) * (WT9_1/WT9_NORMALIZE);
    OutCol += texture(tex, c1) * (WT9_2/WT9_NORMALIZE);
    OutCol += texture(tex, c2) * (WT9_3/WT9_NORMALIZE);
    OutCol += texture(tex, c3) * (WT9_4/WT9_NORMALIZE);
    OutCol += texture(tex, c4) * (WT9_0/WT9_NORMALIZE);
    OutCol += texture(tex, c5) * (WT9_1/WT9_NORMALIZE);
    OutCol += texture(tex, c6) * (WT9_2/WT9_NORMALIZE);
    OutCol += texture(tex, c7) * (WT9_3/WT9_NORMALIZE);
    OutCol += texture(tex, c8) * (WT9_4/WT9_NORMALIZE);

    return OutCol;
}

void main()
{
    vec4 color = vec4(vPosition.r, vPosition.g, 0.0f, 1.0f);
    vec2 inputDimension = textureSize(uInput, 0);

    if (uBlurHorizontal == 1)
    { color = blurH(uInput, vPosition.st, inputDimension, 0.01f); }
    else
    { color = blurV(uInput, vPosition.st, inputDimension, 0.01f); }

    fFragmentColor = color;
    gl_FragDepth = color.r;
}
