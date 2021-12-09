#version 400 core

#extension GL_ARB_separate_shader_objects : enable

layout(vertices = 2) out;

in vec4 VertPosition[];  // Position
in vec3 VertNormal[]; // Direction
in vec4 VertColor[]; // V from PTF, VertColor.w = thickness
in vec4 VertTexture[];  // Tangent
in float VertLengthFromBeginning[]; // Global Texture Coordinates

out vec4 ContPosition[];
out vec3 ContNormal[];
out vec4 ContColor[];
out vec4 ContTexture[];
out float ContLengthFromBeginning[];
out float ContNrVertices[];
out float ContEdgeLength[];

uniform vec3 camPos;
uniform int  isExplicit;
uniform mat4 matModel;
uniform int maxTessSegments = 20;
uniform int visu_lentess = 1;

#define ID gl_InvocationID

void main()
{
	vec3 vSWorld = VertPosition[0].xyz;
	vec3 vTWorld = VertPosition[1].xyz;

	float dS = length(camPos - vSWorld);
	float dT = length(camPos - vTWorld);

    float dist = max(1, min(dS, dT));
		float r = VertColor[0].w;
    float p = 0;

    if(dist <= 0.01)
			dist = 0.01;

    float lenval = length(VertPosition[0].xyz - VertPosition[1].xyz);
	//if(log((r/dist)+1) > 0.00035 || dist < 15)

	/*	float d = length(VertPosition[0].xyz - camPos);
		float c = 1;
		float t = c * lenval/d; //ORIG: c * lenval/d
		t = lenval

        //if(isExplicit == 1)
		    //p = max(1, min(10, t));
            //p = 1;
        //else
            p = max(1, t);*/
	p = min(lenval*visu_lentess,maxTessSegments);//+floor(lenval/1000.0);

	gl_TessLevelOuter[0] = 1;
	gl_TessLevelOuter[1] = p;//OG: p
	//gl_TessLevelOuter[1] = 1;

	ContPosition[ID] = VertPosition[ID];
	ContColor[ID]    = VertColor[ID]; // V from PTF, VertColor.w = thickness
	ContNormal[ID]   = VertNormal[ID]; // Direction
	ContTexture[ID]  = VertTexture[ID]; // Tangent
	ContLengthFromBeginning[ID] = VertLengthFromBeginning[ID];
	ContNrVertices[ID] = p;
	ContEdgeLength[ID] = lenval;
}
