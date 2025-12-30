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
[[vk::binding(BINDING_VDZ_DEPTH_TEXTURE, 0)]] Texture2D<float> vdzDepthTexture : register(t26);
[[vk::binding(BINDING_VDZ_DEPTH_TEXTURE, 0)]] SamplerState vdzSampler : register(s26);
[[vk::push_constant]] ConstantBuffer<PushConstant> pcRaster;

struct VertexInput
{
  [[vk::location(0)]] float3 position : POSITION;  // xy in [0,1], z unused
  [[vk::location(1)]] float2 uv : TEXCOORD0;       // [0,1]
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
  
  float2 uv = input.uv;
  output.uv = uv;

  // Sample depth from VDZ texture
  // Flip UV for sampling (match WebGL convention)
  float2 sampleUv = float2(1.0 - uv.x, 1.0 - uv.y);
  float depth = vdzDepthTexture.SampleLevel(vdzSampler, sampleUv, 0);
  
  // Apply gamma correction
  depth = max(depth, 0.0);
  depth = pow(depth, frameInfo.vdzZGamma);
  
  // Apply scaling and clipping
  float zDepth = clamp(depth * frameInfo.vdzZScale, 0.0, frameInfo.vdzZMaxClip);
  float z = zDepth + frameInfo.vdzZBias;
  
  // Build camera-aligned plane position
  // The plane is centered at camera, scaled by planeScale and aspect ratio
  float planeScale = frameInfo.vdzPlaneScale;
  float aspect = frameInfo.vdzAspect;
  
  float x = (0.5 - uv.x) * aspect * planeScale;
  float y = (0.5 - uv.y) * planeScale;
  
  // View-space position: camera looks down -Z
  float4 viewPos = float4(x, y, -z, 1.0);
  
  // Transform to clip space
  // For camera-attached mesh: directly multiply by projection
  output.position = mul(viewPos, frameInfo.projectionMatrix);
  
  return output;
}
