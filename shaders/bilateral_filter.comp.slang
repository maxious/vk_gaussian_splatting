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

// Bilateral filter compute shader for depth smoothing
// Edge-preserving smoothing that respects depth discontinuities

#include "shaderio.h"

// clang-format off
[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]] ConstantBuffer<FrameInfo> frameInfo;

// Input depth texture (RGB-packed format, same as VDZ)
[[vk::binding(BINDING_VDZ_DEPTH_TEXTURE, 0)]] Sampler2D<float4> depthTexture;
// Output filtered depth
[[vk::binding(0, 1)]] RWTexture2D<float4> outputTexture;
// clang-format on

// Helper to unpack RGB-packed 16-bit depth to float
float unpackPackedDepth(float4 color)
{
    uint16_t uint16Val = (uint16_t(color.r * 255.0) << 8) | uint16_t(color.g * 255.0);
    float scale = (frameInfo.vdzZMax > frameInfo.vdzZMin) ? (frameInfo.vdzZMax - frameInfo.vdzZMin) / 65535.0 : 1.0;
    return float(uint16Val) * scale + frameInfo.vdzZMin;
}

// Helper to pack float depth to RGB format
float4 packDepthForOutput(float depth)
{
    // Normalize to [0, 1] for display
    float minD = frameInfo.vdzZMin;
    float maxD = frameInfo.vdzZMax > minD ? frameInfo.vdzZMax : minD + 1.0;
    float normalized = clamp((depth - minD) / (maxD - minD), 0.0, 1.0);
    return float4(normalized, normalized, normalized, 1.0);
}

// Gaussian function for spatial weight
float gaussian(float x, float sigma)
{
    return exp(-(x * x) / (2.0 * sigma * sigma));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    // Get image dimensions
    uint width, height;
    outputTexture.GetDimensions(width, height);
    
    // Check bounds
    if (any(threadId.xy >= uint2(width, height)))
        return;
    
    // Get current pixel UV coordinates
    float2 uv = (float2(threadId.xy) + 0.5) / float2(width, height);
    
    // Sample current depth
    float4 currentDepthColor = depthTexture.Load(int3(threadId.xy, 0));
    float currentDepth = unpackPackedDepth(currentDepthColor);
    
    // Early exit if depth is invalid
    if (currentDepth < 0.0001)
    {
        outputTexture[threadId.xy] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    
    // Bilateral filter parameters
    float sigmaSpace = frameInfo.vdzBilateralSigmaSpace;
    float sigmaDepth = frameInfo.vdzBilateralSigmaDepth;
    int kernelRadius = int(ceil(sigmaSpace * 2.0));  // Adaptive kernel size
    
    // Ensure minimum kernel size for quality
    kernelRadius = max(kernelRadius, 2);
    kernelRadius = min(kernelRadius, 8);  // Clamp to reasonable size for performance
    
    float2 texelSize = float2(1.0 / float(width), 1.0 / float(height));
    
    float3 weightedSum = float3(0.0, 0.0, 0.0);
    float totalWeight = 0.0;
    
    // Bilateral filtering loop
    for (int dy = -kernelRadius; dy <= kernelRadius; dy++)
    {
        for (int dx = -kernelRadius; dx <= kernelRadius; dx++)
        {
            // Skip center pixel (will be weighted heavily anyway)
            if (dx == 0 && dy == 0)
                continue;
            
            // Sample neighbor
            int2 neighborCoord = int2(threadId.xy) + int2(dx, dy);
            
            // Check bounds
            if (neighborCoord.x < 0 || neighborCoord.x >= int(width) ||
                neighborCoord.y < 0 || neighborCoord.y >= int(height))
                continue;
            
            float2 neighborUv = uv + float2(dx, dy) * texelSize;
            float4 neighborDepthColor = depthTexture.Sample(neighborUv);
            float neighborDepth = unpackPackedDepth(neighborDepthColor);
            
            // Skip invalid neighbors
            if (neighborDepth < 0.0001)
                continue;
            
            // Spatial weight (Gaussian based on distance)
            float spatialDist = sqrt(float(dx * dx + dy * dy));
            float spatialWeight = gaussian(spatialDist, sigmaSpace);
            
            // Range weight (Gaussian based on depth difference)
            float depthDiff = abs(neighborDepth - currentDepth);
            float rangeWeight = gaussian(depthDiff, sigmaDepth);
            
            // Combined weight
            float weight = spatialWeight * rangeWeight;
            
            // Accumulate
            weightedSum += float3(neighborDepth, neighborDepth, neighborDepth) * weight;
            totalWeight += weight;
        }
    }
    
    // Add center pixel with maximum weight
    weightedSum += float3(currentDepth, currentDepth, currentDepth) * 1.0;
    totalWeight += 1.0;
    
    // Compute final filtered depth
    float filteredDepth = weightedSum.r / totalWeight;
    
    // Write output
    outputTexture[threadId.xy] = packDepthForOutput(filteredDepth);
}
