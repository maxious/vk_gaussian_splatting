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
// Combined image sampler - sampler is embedded with texture
// For HLS mode: RGB-packed depth (float4). For legacy mode: float depth
[[vk::binding(BINDING_VDZ_DEPTH_TEXTURE, 0)]] Sampler2D<float4> vdzDepthTexture;
[[vk::push_constant]] ConstantBuffer<PushConstant> pcRaster;

// Helper function to unpack RGB-packed 16-bit depth to float
// RGB packing: R = high byte, G = low byte, B = 128 (midpoint)
// Returns depth in meters using zMin and scale from frameInfo
float unpackPackedDepth(float4 color)
{
    // Unpack 16-bit value from RGB
    uint16_t uint16Val = (uint16_t(color.r * 255.0) << 8) | uint16_t(color.g * 255.0);
    
    // Convert to float depth: depth = uint16 * scale + z_min
    float scale = (frameInfo.vdzZMax > frameInfo.vdzZMin) ? (frameInfo.vdzZMax - frameInfo.vdzZMin) / 65535.0 : 1.0;
    return float(uint16Val) * scale + frameInfo.vdzZMin;
}

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
  // For HLS side-by-side mode: sample RGB and unpack
  // For legacy VDZ mode: texture is float, unpackPackedDepth handles both
  float4 depthColor = vdzDepthTexture.SampleLevel(uv, 0);
  float depth = unpackPackedDepth(depthColor);
  
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
  
  // X: left-to-right maps to positive-to-negative (looking into screen)
  // Y: top-to-bottom maps to positive-to-negative (Vulkan convention)
  float x = (0.5 - uv.x) * aspect * planeScale;
  float y = (0.5 - uv.y) * planeScale;
  
  // View-space position: camera looks down -Z
  float4 viewPos = float4(x, y, -z, 1.0);
  
  // Transform to clip space
  if (frameInfo.vdzWorldSpaceMode != 0)
  {
    // World-space mode: transform through model -> view -> projection
    // The mesh is defined in its own local space (0..1 UV), displaced to View-space-like coordinates
    // We treat 'viewPos' as a local coordinate that needs to be transformed by the captured model matrix
    // to get into World Space, then by current View/Proj to get to Clip Space.
    float4 worldPos = mul(viewPos, frameInfo.vdzModelMatrix);
    output.position = mul(mul(worldPos, frameInfo.viewMatrix), frameInfo.projectionMatrix);
  }
  else
  {
    // Camera-attached mode: directly multiply by projection (viewPos is already in View Space)
    output.position = mul(viewPos, frameInfo.projectionMatrix);
  }
  
  return output;
}
