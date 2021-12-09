#version 400 core

#define FRAG_COLOR 0

layout(location = FRAG_COLOR) out vec4 FragColor;

in float diffuseFactor;
in vec4 texCoordA;
in vec4 texCoordB; //shadow coord
in float vboColor;
in float thickness;
in vec4 normal;
in vec4 tangent;
in vec4 outpos;

uniform sampler2D shadowMap;
uniform sampler2D texBark;
uniform sampler2D texBark2;
uniform sampler2D texBump;
uniform int applyShadow;
uniform int renderMode;
uniform float interpoland;
uniform vec3 lightPos;
uniform vec3 camPos;
uniform mat4 matView;
uniform mat4 matModel;
uniform mat4 matRot;


float lookup(vec2 offSet, vec4 sCoord)
{
    float unitX = 1.0/(569);
    float unitY = 1.0/(596);

    vec4 coords = sCoord / sCoord.w;

    float x = (offSet.x * unitX);
    float y = (offSet.y * unitY);

    float expDepth = texture2D(shadowMap, coords.xy + vec2(x, y)).x;

    return expDepth;
}

float calcShadow(vec4 ShadowCoord)
{
    float shadow = 0.0;
    float shadowStrenght = 0.4;

    float c = 0.0;
    float r = 0.1;
    float s = 0.05;

    vec4 coord = ShadowCoord / ShadowCoord.w;

	for (float y = -r ; y <=r ; y+=s)
	{
		for (float x = -r ; x <=r ; x+=s)
		{
			float temp = lookup(vec2(x,y), ShadowCoord);

			if(temp < (coord.z - 0.008))
				shadow += shadowStrenght;
			else
			    shadow += 1;

			c+=1.0;
		}
	}
	shadow /= c ;

    return shadow;
}

void main()
{
   vec4 binormal = vec4(cross(normal.xyz,tangent.xyz),1);
   vec4 bumpCol = texture(texBump, texCoordA.xy);
   vec4 surfNormal = normalize( vec4(normal.xyz + tangent.xyz*bumpCol.r + binormal.xyz*bumpCol.g + normal.xyz*bumpCol.b, 0));
   mat3 tangent2world;
   tangent2world[0] = normalize( tangent.xyz );
   tangent2world[1] = normalize( binormal.xyz  );
   tangent2world[2] = normalize( normal.xyz  );
   mat3 world2tangent = transpose(tangent2world);

   vec3 tangentEye = (outpos.xyz-camPos) * world2tangent;

   //float fHeightMapScale = 1;
   //float fParallaxLimit = -length( tangentEye.xy ) / tangentEye.z;
   //fParallaxLimit *= fHeightMapScale;
   //
	//vec2 vOffsetDir = normalize( tangentEye.xy );
	//vec2 vMaxOffset = vOffsetDir * fParallaxLimit;
	//int nMaxSamples = 10;
	//int nMinSamples = 3;
	//int nNumSamples = (int)lerp( nMaxSamples, nMinSamples,  dot(normalize(outpos.xyz-camPos), normalize((surfNormal).xyz)) );
	//
	//float fStepSize = 1.0 / (float)nNumSamples;


   vec4 texColor1 = texture(texBark, texCoordA.xy);
   vec4 texColor2 = texture(texBark2, texCoordA.xy);

   vec4 texColor = vec4(mix(texColor1.xyz, texColor2.xyz, interpoland), 1);
   //vec4 texColor = texture(texBump, texCoordA.xy);
   //vec4 texColor = vec4(1,interpoland,0,1);

   vec3 camDir = outpos.xyz-camPos;

   float camAngle = dot(normalize(camDir), normalize((surfNormal).xyz));
   //texColor = mix(vec4(0,0,1,1),vec4(1,1,0,1),camAngle);
   //texColor = vec4(0.5,0.5,0.5,1);

   float shadow = 1.0;
   if(applyShadow == 1 && thickness > 0.005)
       shadow = calcShadow(texCoordB);

   /*shadow = camAngle;
   vec3 a = vec3(0.4, 0.4, 0.4);
   vec3 d = texColor.xyz * diffuseFactor;
   vec3 outcolor = (texColor.xyz * (a + d)) * shadow;// * 1.8;
*/

   	if(renderMode == 0)
   {
		//outcolor = vec3(0.8) + vec3(0.2, 0.2, 0.2) * d * shadow;
   }

   ////outcolor = vec3(vboColor, vboColor, vboColor);
   ////outcolor += 0.7;

  // FragColor = vec4(outcolor, 1);

   // render tree
   float frontc = 20;
   float frontsc = 70;
   float backc = 70;

   float depth = 1-max(0,outpos.z-0.5);
   backc -= frontc;

   vec3 basecolor = vec3(frontc/255,frontc/255,frontc/255);
   vec3 basescolor = vec3(frontsc/255.0,frontsc/255.0,frontsc/255.0);

   float lightAngle = dot(normalize(lightPos-outpos.xyz), normalize((normal).xyz));
   //shadow = min(shadow,1-lightAngle);

   FragColor = vec4(basescolor * shadow, 1);
   FragColor.xyz = FragColor.xyz+(depth)*vec3(backc/255.0,backc/255.0,backc/255.0);

}
