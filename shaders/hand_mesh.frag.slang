/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "shaderio.h"

struct FragmentInput
{
  [[vk::location(1)]] float3 worldPos : TEXCOORD1;
  [[vk::location(2)]] float3 worldNrm : TEXCOORD2;
  [[vk::location(3)]] float3 viewDir : TEXCOORD3;
  [[vk::location(4)]] float2 texCoord : TEXCOORD4;
};

struct FragmentOutput
{
  float4 color : SV_Target0;
};

[shader("fragment")]
FragmentOutput main(FragmentInput input)
{
  FragmentOutput output;

  float3 normal = normalize(input.worldNrm);
  float3 viewDir = normalize(-input.viewDir);

  // Main light for cel-shading
  float3 lightDir = normalize(float3(0.5, 1.0, 0.3));
  
  // Cartoony skin tone - more saturated and vibrant
  float3 skinColor = float3(1.0, 0.82, 0.72);
  float3 shadowColor = float3(0.85, 0.55, 0.5);
  
  // Cel-shading: quantize lighting into discrete bands
  float NdotL = dot(normal, lightDir);
  float lightIntensity;
  
  // 3-band toon shading
  if (NdotL > 0.5)
    lightIntensity = 1.0;      // Fully lit
  else if (NdotL > 0.0)
    lightIntensity = 0.7;      // Mid-tone
  else
    lightIntensity = 0.45;     // Shadow
  
  // Blend between skin and shadow color based on band
  float3 baseColor = lerp(shadowColor, skinColor, lightIntensity);
  
  // Cartoon specular highlight - sharp, stylized
  float3 halfVec = normalize(lightDir + viewDir);
  float spec = dot(normal, halfVec);
  float3 specular = float3(0.0);
  if (spec > 0.92)
    specular = float3(1.0, 1.0, 1.0) * 0.5;  // Sharp white highlight
  
  // Rim light for cartoon pop - crisp edge
  float rim = 1.0 - max(dot(normal, viewDir), 0.0);
  float3 rimColor = float3(0.0);
  if (rim > 0.7)
    rimColor = float3(1.0, 0.9, 0.85) * 0.35;
  
  float3 finalColor = baseColor + specular + rimColor;
  
  // Slightly boost saturation for cartoony look
  float luminance = dot(finalColor, float3(0.299, 0.587, 0.114));
  finalColor = lerp(float3(luminance), finalColor, 1.2);
  finalColor = saturate(finalColor);
  
  // Fully opaque - no ghostly transparency
  float alpha = 1.0;

  output.color = float4(finalColor, alpha);

  return output;
}
