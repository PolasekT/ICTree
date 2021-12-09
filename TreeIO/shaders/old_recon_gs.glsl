#version 400 core

#extension GL_ARB_separate_shader_objects : enable

#extension GL_EXT_geometry_shader4 : enable
// MODIFICATION FOR TRANSFORM FEEDBACK - TF

layout(lines, invocations = 1) in;
layout(triangle_strip, max_vertices = 42) out;

uniform mat4 matLightView;
uniform mat4 matViewProjection;

//uniform vec3 lightPos;
uniform vec3 camPos;
uniform int  isExplicit;
uniform float meshScl = 1.0; //scales the mesh after all the geometry generation has been finished.
uniform int visu_hortess = 11;

in vec4 VertPosition[];
in vec4 EvalPosition[];
in vec4 EvalColor[];
in vec3 EvalNormal[];
in vec3 Eval2ndDerivative[];
in vec4 EvalTexture[]; // Tangent
in vec4 EvalTessCoord[];
in vec2 EvalTextureCoord[];
in float EvalNrVertices[];
in float EvalEdgeLength[];

out float diffuseFactor;
out vec4 texCoordA;
out vec4 texCoordB; //shadow coord
out float vboColor;
out float thickness;
out vec4 normal;
out vec4 tangent;
out vec4 outpos;
out vec4 retrieve;
//out float key;

void main()
{
		float c = 1000;
    float PI2 = 2 * 3.141592654;

    for(int i=0; i<gl_VerticesIn-1; ++i)
    {
		//Reading Data
        vec4 posS = EvalPosition[i];
        vec4 posT = EvalPosition[i+1];

        //vec4 normS = EvalNormal[i];
        //vec4 normT = EvalNormal[i+1];

        vec3 vS = EvalColor[i].xyz; // binormal?
        vec3 vT = EvalColor[i+1].xyz; // binormal?

        vec3 tS = EvalTexture[i].xyz; // tangent
        vec3 tT = EvalTexture[i+1].xyz; // tangent

        float thickS = EvalColor[i].w; // thickness
        float thickT = EvalColor[i+1].w; // thickness

		//Computing
        vec3 v11 = normalize(vS); // binormal?
        vec3 v12 = normalize(cross(vS, tS))*1; // normal?
				//v12 = Eval2ndDerivative[i+1].xyz;

        vec3 v21 = normalize(vT); // binormal?
        vec3 v22 = normalize(cross(vT, tT))*1; // normal?
				//v22 = Eval2ndDerivative[i+1].xyz;

				//float t1 = (i)/(gl_VerticesIn);
				//float t2 = (i+1)/(gl_VerticesIn);
				//vec = Slerp(vec4(v11.xyz,1),vec4(v21.xyz,1),t).xyz * rS;
				//newPT = posS.xyz + Slerp(vec4(v12.xyz,1),vec4(v22.xyz,1),t).xyz * rT;

        float rS = max(0.0001, thickS);
        float rT = max(0.0001, thickT);

		int forMax = min(visu_hortess,11);

        for(int k=0; k<=forMax; k+=1)
        {
            float angleS = (PI2 / forMax) * k;
            float angleT = (PI2 / forMax) * k;

            vec3 newNormalS = normalize(v11 * sin(angleS) + v12 * cos(angleS));
						vec3 newNormalT = normalize(v21 * sin(angleT) + v22 * cos(angleT));

            vec3 newPS = posS.xyz + newNormalS * rS;
            vec3 newPT = posT.xyz + newNormalT * rT;

            //Source Vertex
            vec3 N = normalize(posS.xyz - newPS);
            thickness = rS;
                        outpos = (vec4(newPS, 1) * vec4(meshScl,meshScl,meshScl,1));
						gl_Position = matViewProjection * (outpos);
                        texCoordB = matLightView * (outpos);
			            normal = vec4(N, 1);
						retrieve = outpos;
						tangent = vec4(normalize(newPS-newPT), 1);
						//key = k;
            EmitVertex();

            //Target Vertex
            N = normalize(posT.xyz - newPT);
            thickness = rT;
                        outpos = (vec4(newPT, 1) * vec4(meshScl,meshScl,meshScl,1));
						gl_Position = matViewProjection * (outpos);
                        texCoordB = matLightView * (outpos);
			            normal = vec4(N, 1);
						retrieve = outpos;
						tangent = vec4(normalize(newPS-newPT), 1);
						//key = k;
            EmitVertex();
        }
    }

	EndPrimitive();
}
