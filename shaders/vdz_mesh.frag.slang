/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "shaderio.h"

// clang-format off
[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]] ConstantBuffer<FrameInfo> frameInfo;
[[vk::binding(BINDING_VDZ_VIDEO_TEXTURE, 0)]] Texture2D<float4> vdzVideoTexture : register(t25);
[[vk::binding(BINDING_VDZ_VIDEO_TEXTURE, 0)]] SamplerState vdzSampler : register(s25);

struct FragmentInput
{
  [[vk::location(0)]] float2 uv : TEXCOORD0;
};

struct FragmentOutput
{
  [[vk::location(0)]] float4 color : SV_Target0;
};
// clang-format on

[shader("fragment")]
FragmentOutput main(FragmentInput input)
{
  FragmentOutput output;
  
  // Sample video texture with flipped UVs (match vertex shader convention)
  float2 sampleUv = float2(1.0 - input.uv.x, 1.0 - input.uv.y);
  float4 color = vdzVideoTexture.Sample(vdzSampler, sampleUv);
  
  // Use a turbo colormap for depth visualization if input is single channel
  // Or just pass through if it's actual video
  // Since we know for now vdzVideoTexture IS the depth texture (R32F), we visualize it
  
  float d = color.r;
  
  // Normalize visualization range (adjust as needed, e.g. 0 to 5 meters)
  float v = clamp(d / 5.0, 0.0, 1.0);
  
  // Simple heatmap (Blue -> Green -> Red)
  float3 heatmap;
  if (v < 0.5)
      heatmap = lerp(float3(0,0,1), float3(0,1,0), v * 2.0);
  else
      heatmap = lerp(float3(0,1,0), float3(1,0,0), (v - 0.5) * 2.0);

  output.color = float4(heatmap, 1.0);
  
  return output;
}
