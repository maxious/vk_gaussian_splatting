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
// Combined image samplers - sampler is embedded with texture
// For HLS mode: RGB video (left half) and RGB-packed depth (right half)
[[vk::binding(BINDING_VDZ_VIDEO_TEXTURE, 0)]] Sampler2D<float4> vdzVideoTexture;
// For HLS mode: RGB-packed depth (float4). For legacy mode: float depth
[[vk::binding(BINDING_VDZ_DEPTH_TEXTURE, 0)]] Sampler2D<float4> vdzDepthTexture;

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

struct FragmentInput
{
  [[vk::location(0)]] float2 uv : TEXCOORD0;
};

struct FragmentOutput
{
  [[vk::location(0)]] float4 color : SV_Target0;
};
// clang-format on

// Turbo-like colormap for depth visualization
float3 depthToColormap(float depth, float minD, float maxD)
{
  float v = clamp((depth - minD) / (maxD - minD), 0.0, 1.0);
  
  // Turbo-like heatmap (Blue -> Cyan -> Green -> Yellow -> Red)
  float3 c;
  if (v < 0.25)
      c = lerp(float3(0.18995, 0.07176, 0.23217), float3(0.19903, 0.52157, 0.71861), v * 4.0);
  else if (v < 0.5)
      c = lerp(float3(0.19903, 0.52157, 0.71861), float3(0.56468, 0.87997, 0.25378), (v - 0.25) * 4.0);
  else if (v < 0.75)
      c = lerp(float3(0.56468, 0.87997, 0.25378), float3(0.99000, 0.65154, 0.09877), (v - 0.5) * 4.0);
  else
      c = lerp(float3(0.99000, 0.65154, 0.09877), float3(0.72271, 0.07562, 0.15135), (v - 0.75) * 4.0);
  
  return c;
}

[shader("fragment")]
FragmentOutput main(FragmentInput input)
{
  FragmentOutput output;
  
  // Sample directly (match vertex shader convention)
  float2 sampleUv = input.uv;
  
  // Sample depth and unpack from RGB-packed format
  float4 depthColor = vdzDepthTexture.Sample(sampleUv);
  float depth = unpackPackedDepth(depthColor);
  
  // Detect depth discontinuities using screen-space derivatives
  // Large gradients indicate triangle spanning a depth edge
  if (frameInfo.vdzEdgeThreshold > 0.0)
  {
    float dDepthX = abs(ddx(depth));
    float dDepthY = abs(ddy(depth));
    float maxGradient = max(dDepthX, dDepthY);
    
    // Discard fragments at depth edges to prevent stretched triangles
    if (maxGradient > frameInfo.vdzEdgeThreshold)
    {
      discard;
    }
  }
  
  // Black for invalid/empty depth
  if (depth < 0.0001)
  {
    discard;
  }
  
  if (frameInfo.vdzUseVideoTexture != 0)
  {
    // Video+Depth mode: Use video RGB texture
    // For HLS side-by-side: sample from left half (UV.x stays the same)
    float4 videoColor = vdzVideoTexture.Sample(sampleUv);
    output.color = float4(videoColor.rgb, 1.0);
  }
  else
  {
    // Depth-only mode: Use colormap visualization
    // Use zMin/zMax from metadata for proper range mapping
    float minD = frameInfo.vdzZMin;
    float maxD = frameInfo.vdzZMax > minD ? frameInfo.vdzZMax : 1.0;
    float3 colormap = depthToColormap(depth, minD, maxD);
    output.color = float4(colormap, 1.0);
  }
  
  return output;
}
