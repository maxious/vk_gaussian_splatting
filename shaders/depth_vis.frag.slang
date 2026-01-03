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
[[vk::binding(BINDING_OPACITY_TEXTURE, 0)]] Texture2D<float> depthTexture : register(t16);
[[vk::binding(BINDING_OPACITY_SAMPLER, 0)]] SamplerState depthSampler : register(s16);
[[vk::push_constant]] ConstantBuffer<PushConstant> pcRaster;

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

  // Sample depth value from texture
  float depth = depthTexture.Sample(depthSampler, input.uv);

  // Visualize depth with a color gradient
  // Near = red, Far = blue, Mid = green
  float t = clamp(depth, 0.0f, 1.0f);

  float3 color;
  if(t < 0.5f)
  {
    // Red to Green
    float localT = t * 2.0f;
    color = float3(1.0f - localT, localT, 0.0f);
  }
  else
  {
    // Green to Blue
    float localT = (t - 0.5f) * 2.0f;
    color = float3(0.0f, 1.0f - localT, localT);
  }

  // Output color with alpha
  output.color = float4(color, 1.0f);

  return output;
}
