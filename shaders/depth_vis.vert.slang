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

struct VertexInput
{
  [[vk::location(0)]] float2 position : POSITION;
};

struct VertexOutput
{
  [[vk::location(0)]] float2 uv : TEXCOORD0;
  float4 position : SV_Position;
};
// clang-format on

[shader("vertex")]
VertexOutput main(VertexInput input)
{
  VertexOutput output;

  // Fullscreen quad: positions are in [-1, 1], UVs are in [0, 1]
  output.uv       = input.position * 0.5f + 0.5f;  // Convert [-1,1] to [0,1]
  output.position = float4(input.position, 0.0f, 1.0f);

  return output;
}
