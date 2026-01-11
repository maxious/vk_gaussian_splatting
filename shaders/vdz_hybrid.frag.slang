/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Hybrid Fragment Shader: Combines divided mesh with Parallax Occlusion Mapping
// - Applies POM ray-marching for fine parallax detail
// - Combines with coarse mesh displacement from vertex shader
// - Supports video texture or depth colormap visualization

#include "shaderio.h"

// clang-format off
[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]] ConstantBuffer<FrameInfo> frameInfo;
[[vk::binding(BINDING_VDZ_VIDEO_TEXTURE, 0)]] Sampler2D<float4> vdzVideoTexture;
[[vk::binding(BINDING_VDZ_DEPTH_TEXTURE, 0)]] Sampler2D<float4> vdzDepthTexture;
[[vk::push_constant]] ConstantBuffer<PushConstant> pcRaster;

// Helper function to unpack RGB-packed 16-bit depth to float
float unpackPackedDepth(float4 color)
{
    uint16_t uint16Val = (uint16_t(color.r * 255.0) << 8) | uint16_t(color.g * 255.0);
    float scale = (frameInfo.vdzZMax > frameInfo.vdzZMin) ? (frameInfo.vdzZMax - frameInfo.vdzZMin) / 65535.0 : 1.0;
    return float(uint16Val) * scale + frameInfo.vdzZMin;
}

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

// Parallax Occlusion Mapping with adaptive layer count (VR-optimized)
float2 parallaxOcclusionMapping(float2 texCoords, float3 viewDirTS, float heightScale, int maxLayers)
{
    // Adaptive layer count based on viewing angle (critical for VR!)
    // More layers at grazing angles, fewer when looking straight on
    const float minLayers = 4.0;
    float numLayers = lerp(float(maxLayers), minLayers, abs(viewDirTS.z));
    float layerDepth = 1.0 / numLayers;
    
    // Calculate UV offset per layer
    // P = viewDir.xy / viewDir.z * heightScale
    // This projects the view ray onto the texture plane
    float2 P = viewDirTS.xy / max(viewDirTS.z, 0.01) * heightScale;
    float2 deltaTexCoords = P / numLayers;
    
    // Initial values
    float2 currentTexCoords = texCoords;
    float currentDepthMapValue = vdzDepthTexture.SampleLevel(currentTexCoords, 0).r;
    float currentLayerDepth = 0.0;
    
    // Ray-march through the height field
    // We're looking for where the view ray crosses the surface defined by the depth map
    while(currentLayerDepth < currentDepthMapValue)
    {
        currentTexCoords -= deltaTexCoords;
        currentDepthMapValue = vdzDepthTexture.SampleLevel(currentTexCoords, 0).r;
        currentLayerDepth += layerDepth;
    }
    
    // Parallax Occlusion: Smooth interpolation between layers
    // This produces smoother results than steep parallax mapping
    float2 prevTexCoords = currentTexCoords + deltaTexCoords;
    
    float afterDepth = currentDepthMapValue - currentLayerDepth;
    float beforeDepth = vdzDepthTexture.SampleLevel(prevTexCoords, 0).r - currentLayerDepth + layerDepth;
    
    // Linear interpolation between the two layer boundaries
    float weight = afterDepth / (afterDepth - beforeDepth + 0.00001);
    float2 finalTexCoords = prevTexCoords * weight + currentTexCoords * (1.0 - weight);
    
    return finalTexCoords;
}

// Helper function to sample video color at a specific UV coordinate with validity check
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
    
    return float4(depthToColormap(depth, frameInfo.vdzZMin, 
                                  frameInfo.vdzZMax > frameInfo.vdzZMin ? frameInfo.vdzZMax : 1.0), 1.0);
}

// Compute average color from valid neighboring pixels for gap filling
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
            if (dx == 0 && dy == 0) continue;
            
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
    
    return frameInfo.vdzGapFillColor;
}

struct FragmentInput
{
  [[vk::location(0)]] float2 uv : TEXCOORD0;
  [[vk::location(1)]] float2 depthUV : TEXCOORD1;
  [[vk::location(2)]] float3 viewDirTS : TEXCOORD2;
  [[vk::location(3)]] float3 normalWS : TEXCOORD3;
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
    
    // Determine which mode to use
    int hybridMode = frameInfo.vdzHybridMode;
    
    // Calculate base POM UV offset if needed
    float2 pomUV = input.uv;
    float pomDepth = 0.0;
    
    // Apply POM if in POM-only or hybrid mode
    if ((hybridMode == 1 || hybridMode == 2) && frameInfo.vdzHybridPomStrength > 0.0)
    {
        float heightScale = frameInfo.vdzHybridPomDepthScale * frameInfo.vdzHybridPomStrength;
        int maxLayers = frameInfo.vdzHybridPomLayers;
        
        // Apply POM with adaptive layer count
        pomUV = parallaxOcclusionMapping(input.uv, normalize(input.viewDirTS), heightScale, maxLayers);
        
        // Sample depth at POM-offset position for visualization
        float4 pomDepthColor = vdzDepthTexture.SampleLevel(pomUV, 0);
        pomDepth = unpackPackedDepth(pomDepthColor);
        pomDepth = max(pomDepth, 0.0);
        pomDepth = pow(pomDepth, frameInfo.vdzZGamma);
    }
    
    // Use the POM UV for final sampling
    float2 sampleUV = pomUV;
    
    // Sample depth and check validity
    float4 depthColor = vdzDepthTexture.Sample(sampleUV);
    float depth = unpackPackedDepth(depthColor);
    depth = max(depth, 0.0);
    
    // Apply gamma correction for visualization
    float displayDepth = pow(depth, frameInfo.vdzZGamma);
    
    // Edge detection using screen-space derivatives
    float maxGradient = 0.0;
    if (frameInfo.vdzEdgeThreshold > 0.0)
    {
        float2 texelSize = float2(1.0 / frameInfo.viewport.x, 1.0 / frameInfo.viewport.y);
        
        float4 leftColor = vdzDepthTexture.Sample(sampleUV - float2(texelSize.x, 0.0));
        float4 rightColor = vdzDepthTexture.Sample(sampleUV + float2(texelSize.x, 0.0));
        float4 topColor = vdzDepthTexture.Sample(sampleUV - float2(0.0, texelSize.y));
        float4 bottomColor = vdzDepthTexture.Sample(sampleUV + float2(0.0, texelSize.y));
        
        float leftDepth = unpackPackedDepth(leftColor);
        float rightDepth = unpackPackedDepth(rightColor);
        float topDepth = unpackPackedDepth(topColor);
        float bottomDepth = unpackPackedDepth(bottomColor);
        
        float dDepthX = abs(rightDepth - leftDepth);
        float dDepthY = abs(bottomDepth - topDepth);
        
        maxGradient = max(dDepthX, dDepthY);
    }
    
    // Determine if this pixel is invalid or at an edge
    bool isInvalid = depth < 0.0001;
    bool isEdge = maxGradient > frameInfo.vdzEdgeThreshold;
    
    // Handle gap filling vs discarding
    if (frameInfo.vdzGapFillEnabled != 0 && (isInvalid || isEdge))
    {
        float gapMask = 0.0;
        
        if (isInvalid)
        {
            gapMask = 1.0;
        }
        else if (isEdge && frameInfo.vdzGapFeather > 0.0)
        {
            float edgeDistance = (maxGradient - frameInfo.vdzEdgeThreshold) / 
                                (frameInfo.vdzEdgeThreshold * frameInfo.vdzGapFeather);
            gapMask = clamp(edgeDistance, 0.0, 1.0);
        }
        
        float4 videoColor = float4(0.0, 0.0, 0.0, 1.0);
        if (frameInfo.vdzUseVideoTexture != 0)
        {
            videoColor = vdzVideoTexture.Sample(sampleUV);
        }
        else
        {
            videoColor = float4(depthToColormap(displayDepth, frameInfo.vdzZMin, 
                                                 frameInfo.vdzZMax > frameInfo.vdzZMin ? frameInfo.vdzZMax : 1.0), 1.0);
        }
        
        float3 fillColor;
        if (frameInfo.vdzGapFillNeighbor != 0 && gapMask > 0.0)
        {
            fillColor = computeNeighborAverage(sampleUV, frameInfo.vdzGapFillRadius);
        }
        else
        {
            fillColor = frameInfo.vdzGapFillColor;
        }
        
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
            float4 videoColor = vdzVideoTexture.Sample(sampleUV);
            output.color = float4(videoColor.rgb, 1.0);
        }
        else
        {
            float minD = frameInfo.vdzZMin;
            float maxD = frameInfo.vdzZMax > minD ? frameInfo.vdzZMax : 1.0;
            float3 colormap = depthToColormap(displayDepth, minD, maxD);
            output.color = float4(colormap, 1.0);
        }
    }
    
    return output;
}
