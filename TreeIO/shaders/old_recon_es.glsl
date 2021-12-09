#version 400 core

layout(isolines, equal_spacing) in;

in vec4 ContPosition[];
in vec3 ContNormal[]; // Direction
in vec4 ContColor[]; // V from PTF, VertColor.w = thickness
in vec4 ContTexture[]; // Tangent
in float ContLengthFromBeginning[];
in float ContNrVertices[];
in float ContEdgeLength[];

// For PTF see TreeGraph.cpp - line 1437.

out vec4 EvalPosition;
out vec3 EvalNormal;
out vec3 Eval2ndDerivative;
out vec4 EvalColor;
out vec4 EvalTexture;
out vec2 EvalTextureCoord;
out float EvalNrVertices;
out float EvalEdgeLength;

void hermiteInterpolation(in vec3 p0, in vec3 p1, in vec3 p2, in vec3 p3, out vec3 position, out vec3 tangent, out vec3 norm, float t)
{
    float tension = 0.0f;
    float bias = 0.0f;

    vec3 m0, m1;
    float t2,t3;
    float a0, a1, a2, a3;

    t2 = t * t;
    t3 = t2 * t;

    m0  = (p1-p0)*(1+bias)*(1-tension)/2;
    m0 += (p2-p1)*(1-bias)*(1-tension)/2;
    m1  = (p2-p1)*(1+bias)*(1-tension)/2;
    m1 += (p3-p2)*(1-bias)*(1-tension)/2;

    a0 =  2*t3 - 3*t2 + 1;
    a1 =    t3 - 2*t2 + t;
    a2 =    t3 -   t2;
    a3 = -2*t3 + 3*t2;

    position = vec3(a0*p1 + a1*m0 + a2*m1 + a3*p2);

    vec3 d1 = ((6*t2 - 6*t) * p1) + ((3*t2 - 4*t +1) * m0) + ((3*t2 - 2*t) * m1) + ((-6*t2 + 6*t) * p2);
    vec3 d2 = ((12*t - 6) * p1) + ((6*t - 4) * m0) + ((6*t - 2) * m1) + ((-12*t + 6) * p2);
    tangent = normalize(d1);
    if(t<=0)
        tangent = -(p0-p1);
    if(t>=1)
        tangent = (p3-p2);
    norm = normalize(d2);
}

void cubicInterpolation(in vec3 v0, in vec3 v1, in vec3 v2, in vec3 v3, out vec3 position, out vec3 tangent, float t)
{
   float t2 = t * t;
   vec3 a0, a1, a2, a3;

   a0 = v3 - v2 - v0 + v1;
   a1 = v0 - v1 - a0;
   a2 = v2 - v0;
   a3 = v1;

   position = vec3(a0*t*t2 + a1*t2 + a2*t + a3);

   vec3 d1 = vec3(3*a0*t2 + 2*a1*t + a2);
   tangent = normalize(d1);
}

// edit by David Hrusa to reduce artifacts
void cubicInterpolation2(in vec3 v0, in vec3 v1, in vec3 v2, in vec3 v3, out vec3 position, out vec3 tangent, float t)
{
   float t2 = t * t;
   vec3 a0, a1, a2, a3;

   a0 = v3 - v2 - v0 + v1;
   a1 = v0 - v1 - a0;
   a2 = v2 - v0;
   a3 = v1;

   position = vec3(a0*t*t2 + a1*t2 + a2*t + a3);

   vec3 d1 = vec3(3*a0*t2 + 2*a1*t + a2);
   tangent = normalize(d1);
}


// edit by David Hrusa to reduce artifacts and fix connectivity
void cubicInterpolation3(in vec3 v0, in vec3 v1, in vec3 v2, in vec3 v3, out vec3 position, out vec3 tangent, float t)
{
   float t2 = t * t;
   vec3 a0, a1, a2, a3;

   a0 = v3 - v2 - v0 + v1;
   a1 = v0 - v1 - a0;
   a2 = v2 - v0;
   a3 = v1;

   position = vec3(a0*t*t2 + a1*t2 + a2*t + a3);

   vec3 d1 = vec3(3*a0*t2 + 2*a1*t + a2);
   tangent = normalize(d1);
}

// by David Hrusa
void linearInterpolation(in vec3 v0, in vec3 v1, in vec3 v2, in vec3 v3, out vec3 position, out vec3 tangent, float t)
{
   position = vec3(v1*t+v2*(1-t));
   tangent = normalize(v2-v1);
}

/// taken from: http://www.geeks3d.com/20140205/glsl-simple-morph-target-animation-opengl-glslhacker-demo/
/// Avoid linearly interpolating normals. Slerp them instead.
vec4 Slerp(vec4 p0, vec4 p1, float t)
{
  float dotp = dot(normalize(p0), normalize(p1));
  if ((dotp > 0.9999) || (dotp<-0.9999))
  {
    if (t<=0.5)
      return p0;
    return p1;
  }
  float theta = acos(dotp);
  vec4 P = ((p0*sin((1-t)*theta) + p1*sin(t*theta)) / sin(theta));
  P.w = 1;
  return P;
}

void main()
{
    vec3 t = vec3(gl_TessCoord.x, gl_TessCoord.y, gl_TessCoord.z);

    vec3 vS = ContColor[0].xyz; // V of PTF
    vec3 vT = ContColor[1].xyz;

    vec3 tS = ContTexture[0].xyz; // Tangent
    vec3 tT = ContTexture[1].xyz;

    float thickS = ContColor[0].w; // Thickness
    float thickT = ContColor[1].w;

	//vec3 p_minus_1 = ContPosition[0].xyz - ContNormal[0].xyz;
	//vec3 pi        = ContPosition[0].xyz;
	//vec3 p_plus_1  = ContPosition[1].xyz;
	//vec3 p_plus_2  = ContPosition[1].xyz + ContNormal[1].xyz;

	//scale the addition with the distance between the two points to avoid squeezed branches
	float len = 1;//length(ContPosition[0].xyz - ContPosition[1].xyz)*1;

	// Are we using some made up points shifted by a tangent instead of the proper previous points?
	// Maybe fix that for continuity
	vec3 p_minus_1 = ContPosition[0].xyz - len*ContTexture[0].xyz; // position - tangent
	vec3 pi        = ContPosition[0].xyz; // position
	vec3 p_plus_1  = ContPosition[1].xyz; // position
	vec3 p_plus_2  = ContPosition[1].xyz + len*ContTexture[1].xyz; // position + tangent


    vec3 pos, tan, norm;
	//linearInterpolation(p_minus_1, pi, p_plus_1, p_plus_2, pos, tan, t.x);
	//cubicInterpolation3(p_minus_1, pi, p_plus_1, p_plus_2, pos, tan, t.x);
	hermiteInterpolation(p_minus_1, pi, p_plus_1, p_plus_2, pos, tan, norm, t.x);

    float thickness = mix(thickS, thickT, t.x);
    //vec3 V = normalize(mix(vS, vT, t.x)); //slerp it
    vec3 V = normalize(Slerp(vec4(vS,0), vec4(vT,0), t.x)).xyz;
   // tan = normalize(mix(tS.xyz, tT.xyz, t.x));

	float newTexCoord = mix(ContLengthFromBeginning[0], ContLengthFromBeginning[1], t.x);
	EvalTextureCoord = vec2(newTexCoord, t.y) * 1;

    EvalPosition = vec4(pos, 1);

    float vboColor = (ContTexture[0].w + ContTexture[1].w) / 2;
    EvalTexture  = vec4(tan, vboColor);

    EvalColor.xyz = V;
    EvalColor.w = thickness;
    //EvalNormal = mix(ContNormal[0], ContNormal[1], t.x); // instead Slerp them
    EvalNormal = mix(vec4(ContNormal[0].xyz,0), vec4(ContNormal[1].xyz,0), t.x).xyz;
    Eval2ndDerivative = norm;

    EvalNrVertices = ContNrVertices[0];

    EvalEdgeLength = ContEdgeLength[0];
}
