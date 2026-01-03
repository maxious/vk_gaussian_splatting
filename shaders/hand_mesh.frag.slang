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

  float3 lightDir = normalize(float3(0.5, 1.0, 0.3));
  
  float3 baseColor = float3(0.9, 0.75, 0.65);
  
  float ambient = 0.3;
  float diffuse = max(dot(normal, lightDir), 0.0) * 0.6;
  
  float3 halfVec = normalize(lightDir + viewDir);
  float specular = pow(max(dot(normal, halfVec), 0.0), 32.0) * 0.2;
  
  float rim = pow(1.0 - max(dot(normal, viewDir), 0.0), 3.0) * 0.15;

  float3 finalColor = baseColor * (ambient + diffuse) + float3(specular + rim);

  output.color = float4(finalColor, 0.85);

  return output;
}
