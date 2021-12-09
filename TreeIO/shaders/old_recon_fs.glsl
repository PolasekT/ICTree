#version 400 core

#extension GL_ARB_separate_shader_objects : enable

#define FRAG_COLOR 0

layout(location = FRAG_COLOR) out vec4 FragColor;

/*in vec4  VertPosition;
in vec3  VertNormal;
in vec4  VertColor;
in vec4  VertTexture;
in float VertLengthFromBeginning;*/

uniform vec4 visu_warmcol = vec4(0.98,0.86,0.37,1);
uniform vec4 visu_coolcol = vec4(0.6,0.94,1,1);
uniform float visu_warmangle = 3.93; // 5/8 PI
uniform float visu_coolangle = 0;
uniform float visu_opacity = 0.5;
uniform int visu_hortess = 11;
uniform vec3 camPos;
uniform mat4 matViewProjection;

// lighting uniforms
uniform bool visu_shaded = true;
uniform vec3 lightPos = vec3(10,10,10);
uniform sampler2D shadowMap;
uniform mat4 matLightView;

in float diffuseFactor;
in vec4 texCoordA;
in vec4 texCoordB; //shadow coord
in float vboColor;
in float thickness;
in vec4 normal;
in vec4 tangent;
in vec4 outpos;
//in float key;

float lookup(vec2 offSet, vec4 sCoord)
{
   float unitX = 1.0/(569);
   float unitY = 1.0/(596);

   //vec4 coords = sCoord / sCoord.w;
   vec2 coords = sCoord.xy;//(((sCoord.xy)*0.5) + vec2(0.5,0.5));

   float x = (offSet.x * unitX);
   float y = (offSet.y * unitY);

   float expDepth = texture2D(shadowMap, coords.xy + vec2(x, y)).x;

   return expDepth;
}

float calcShadow(vec4 ShadowCoordRaw, vec4 ShadowCoord)
{
   float shadow = 0.0;
   float shadowStrenght = 0.4;

   float c = 0.0;
   float r = 3.0;
   float s = 1.0;

   float bias = 0.008;

   vec4 coord = ShadowCoord;// / ShadowCoord.w;

   for (float y = -r ; y <=r ; y+=s)
   {
      for (float x = -r ; x <=r ; x+=s)
      {
         float temp = lookup(vec2(x,y), ShadowCoord);

         if(temp > (ShadowCoordRaw.z - bias))
         shadow += shadowStrenght;
         else
         shadow += 1;

         c+=1.0;
      }
   }
   shadow /= c ;

   return shadow;
}

/// calculates the color in case we want the presentation render and not the modeling render.
vec4 visu_render_main(float shadow,  vec4 unprojpos) {
   vec4 texCoordC =  ((texCoordB*vec4(0.5,0.5,1,1)) / texCoordB.w) + vec4(0.5,0.5,0,0);
   vec4 final_color = vec4(0,0,0,1);

   float depthfactor = 1.0;
   float depth = depthfactor*unprojpos.z;
   depth = 1-max(0,min(depth,1));

   float crange = 40;
   float frontc = 40 + crange*depth;
   float frontsc = 0 + crange*depth;
   vec3 basecolor = vec3(frontc/255,frontc/255,frontc/255);
   vec3 basescolor = vec3(frontsc/255.0,frontsc/255.0,frontsc/255.0);

   float lightAngle = max(0,dot(normalize(outpos.xyz-lightPos), normalize((normal).xyz)));
   if(lightAngle>0)
      lightAngle = 1;
   else {
      float mydot = dot(normalize(lightPos-outpos.xyz),normalize((normal).xyz));
      lightAngle = (max(0,1-mydot));
   }

   float shadowed = 1.0;
   shadowed = calcShadow(texCoordB, texCoordC);
   shadowed = min(lightAngle,shadowed);
   shadowed = max(0,shadowed);
   final_color.xyz = mix(basescolor,basecolor,lightAngle);

   return final_color;
}

void main()
{
   vec4 unprojpos = matViewProjection * outpos;
   //gl_FragDepth = (unprojpos.z)/(100);//gl_FragCoord.z;////unprojpos.z/unprojpos.w;
   if(visu_shaded)
      FragColor = visu_render_main(1.0, outpos);
   else {
      float forMax = float(min(visu_hortess, 11));
      //FragColor = vec4(texCoordA.rgb,1
      //FragColor = vec4(0.5,0.7,0.5,0.5);
      vec4 warm = normalize(vec4(cos(visu_warmangle), 0, sin(visu_warmangle), 0));
      vec4 cool = normalize(vec4(cos(visu_coolangle), 0, sin(visu_coolangle), 0));
      float warmangle = max(0, dot(normal, warm));
      float coolangle = max(0, dot(normal, cool));
      //FragColor = vec4(vec3(0.5,0.5,0.5)-(0.5*normal.xyz),0.5);
      FragColor = vec4(0.1, 0.1, 0.1, 0)+vec4(warmangle*visu_warmcol.rgb+coolangle*visu_coolcol.rgb, visu_opacity);
      /*if(key/3.0-floor(key/3.0) <= 0.3) {
     FragColor = vec4(0,1,0,1);
      }*/
   }
}
