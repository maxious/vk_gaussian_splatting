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

// Helper function to sample video color at a specific UV coordinate
// Returns the color and a validity flag
float4 sampleVideoWithValidity(float2 uv, out bool isValid)
{
  isValid = false;
  
  // Check if UV is in valid range
  if (any(uv < float2(0.0, 0.0)) || any(uv > float2(1.0, 1.0)))
  {
    return float4(0.0, 0.0, 0.0, 0.0);
  }
  
  float4 depthColor = vdzDepthTexture.Sample(uv);
  float depth = unpackPackedDepth(depthColor);
  
  // Valid if depth is above threshold
  isValid = depth > 0.0001;
  
  if (!isValid)
  {
    return float4(0.0, 0.0, 0.0, 0.0);
  }
  
  if (frameInfo.vdzUseVideoTexture != 0)
  {
    return vdzVideoTexture.Sample(uv);
  }
  
  return float4(depthToColormap(depth, frameInfo.vdzZMin, frameInfo.vdzZMax > frameInfo.vdzZMin ? frameInfo.vdzZMax : 1.0), 1.0);
}

// Helper function to compute average color from valid neighboring pixels
float3 computeNeighborAverage(float2 centerUv, int radius)
{
  float3 sum = float3(0.0, 0.0, 0.0);
  float count = 0.0;
  
  float2 texelSize = float2(1.0 / frameInfo.viewport.x, 1.0 / frameInfo.viewport.y);
  
  // Sample neighbors in a square pattern
  for (int dx = -radius; dx <= radius; dx++)
  {
    for (int dy = -radius; dy <= radius; dy++)
    {
      if (dx == 0 && dy == 0) continue;  // Skip center
      
      float2 neighborUv = centerUv + float2(dx, dy) * texelSize;
      
      bool isValid;
      float4 neighborColor = sampleVideoWithValidity(neighborUv, isValid);
      
      if (isValid)
      {
        sum += neighborColor.rgb;
        count += 1.0;
      }
    }
  }
  
  if (count > 0.0)
  {
    return sum / count;
  }
  
  // Fallback to solid fill color if no valid neighbors
  return frameInfo.vdzGapFillColor;
}

// Helper function to compute depth gradient magnitude for edge detection
float computeDepthGradient(float2 uv)
{
  float2 texelSize = float2(1.0 / frameInfo.viewport.x, 1.0 / frameInfo.viewport.y);
  
  // Sample depth at neighboring pixels
  float4 leftColor = vdzDepthTexture.Sample(uv - float2(texelSize.x, 0.0));
  float4 rightColor = vdzDepthTexture.Sample(uv + float2(texelSize.x, 0.0));
  float4 topColor = vdzDepthTexture.Sample(uv - float2(0.0, texelSize.y));
  float4 bottomColor = vdzDepthTexture.Sample(uv + float2(0.0, texelSize.y));
  
  float leftDepth = unpackPackedDepth(leftColor);
  float rightDepth = unpackPackedDepth(rightColor);
  float topDepth = unpackPackedDepth(topColor);
  float bottomDepth = unpackPackedDepth(bottomColor);
  
  // Compute gradient
  float dDepthX = abs(rightDepth - leftDepth);
  float dDepthY = abs(bottomDepth - topDepth);
  
  return max(dDepthX, dDepthY);
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
  float maxGradient = 0.0;
  if (frameInfo.vdzEdgeThreshold > 0.0)
  {
    maxGradient = computeDepthGradient(sampleUv);
  }
  
  // Determine if this pixel is invalid or at an edge
  bool isInvalid = depth < 0.0001;
  bool isEdge = maxGradient > frameInfo.vdzEdgeThreshold;
  
  // Handle gap filling vs discarding
  if (frameInfo.vdzGapFillEnabled != 0 && (isInvalid || isEdge))
  {
    // Calculate gap mask based on edge feathering
    float gapMask = 0.0;
    
    if (isInvalid)
    {
      // Invalid depth - full gap
      gapMask = 1.0;
    }
    else if (isEdge && frameInfo.vdzGapFeather > 0.0)
    {
      // Edge region - apply feathering based on gradient magnitude
      float edgeDistance = (maxGradient - frameInfo.vdzEdgeThreshold) / 
                          (frameInfo.vdzEdgeThreshold * frameInfo.vdzGapFeather);
      gapMask = clamp(edgeDistance, 0.0, 1.0);
    }
    
    // Get the base video color (will be used for non-gap regions)
    float4 videoColor = float4(0.0, 0.0, 0.0, 1.0);
    if (frameInfo.vdzUseVideoTexture != 0)
    {
      videoColor = vdzVideoTexture.Sample(sampleUv);
    }
    else
    {
      videoColor = float4(depthToColormap(depth, frameInfo.vdzZMin, 
                                           frameInfo.vdzZMax > frameInfo.vdzZMin ? frameInfo.vdzZMax : 1.0), 1.0);
    }
    
    // Compute fill color based on mode
    float3 fillColor;
    if (frameInfo.vdzGapFillNeighbor != 0 && gapMask > 0.0)
    {
      // Neighbor average mode - sample surrounding valid pixels
      fillColor = computeNeighborAverage(sampleUv, frameInfo.vdzGapFillRadius);
    }
    else
    {
      // Solid color mode
      fillColor = frameInfo.vdzGapFillColor;
    }
    
    // Blend between video and fill based on gap mask
    output.color = float4(lerp(videoColor.rgb, fillColor, gapMask), 1.0);
  }
  else
  {
    // Original behavior: discard invalid or edge pixels
    if (isInvalid || isEdge)
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
  }
  
  return output;
}
