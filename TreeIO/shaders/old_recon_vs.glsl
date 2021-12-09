#version 400 core

#extension GL_ARB_separate_shader_objects : enable

#define VERT_POSITION	0
#define VERT_NORMAL     1
#define VERT_COLOR		2
#define VERT_TEXTURE    3

layout(location = VERT_POSITION) in vec4 Position; // Position
layout(location = VERT_NORMAL)   in vec4 Normal; // Direction + Global Texture Coordinates
layout(location = VERT_COLOR)    in vec4 Color; // V from PTF, VertColor.w = thickness
layout(location = VERT_TEXTURE)  in vec4 Texture; // Tangent

out vec4  VertPosition;
out vec3  VertNormal;
out vec4  VertColor;
out vec4  VertTexture;
out float VertLengthFromBeginning;

uniform mat4 matModel;
uniform mat4 matNtrans;
uniform float meshScl = 1.0;

void main()
{
	vec4 mycol = Color;
	//mycol.w /= meshScl;
  VertPosition = matModel * Position; // Position
  VertNormal   = (matModel * Normal).xyz; // Direction
	VertColor    = matModel * mycol;	// V from PTF, VertColor.w = thickness
	VertTexture  = matModel * Texture;  // Tangent
	VertLengthFromBeginning = Normal.w; // Global Texture Coordinates
}
