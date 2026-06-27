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
 *
 * Stochastic Gaussian Splatting - resolve kernel.
 *
 * For each pixel, reads the winning gaussian index from the index buffer
 * (written by the accumulate kernel), evaluates the SH color, and writes
 * the final pixel color with progressive EMA accumulation.
 */

#include "shaderio.h"
#include "stochasticgs.h.slang"
#include "threedgs_particles_storage.h.slang"

// clang-format off
[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]]
ConstantBuffer<FrameInfo>    frameInfo;

[[vk::binding(BINDING_STOCHASTIC_DEPTH_BUFFER, 1)]]
StructuredBuffer<uint>  depthBuffer;

[[vk::binding(BINDING_STOCHASTIC_INDEX_BUFFER, 1)]]
StructuredBuffer<uint>  indexBuffer;

[[vk::binding(BINDING_STOCHASTIC_OUTPUT_IMAGE, 1)]]
RWTexture2D<float4>         outputImage;

[[vk::binding(BINDING_STOCHASTIC_ACCUMULATION_IMAGE, 1)]]
RWTexture2D<float4>         accumulationImage;
// clang-format on

[numthreads(8, 8, 1)]
[shader("compute")]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
  const int2 pixel = int2(dispatchThreadID.xy);
  const uint width  = uint(frameInfo.stochasticWidth);
  const uint height = uint(frameInfo.stochasticHeight);
  if(pixel.x >= int(width) || pixel.y >= int(height))
    return;

  const uint linearIndex = uint(pixel.y) * width + uint(pixel.x);
  const uint depthVal    = depthBuffer[linearIndex];
  if(depthVal == 0xFFFFFFFFu)
  {
    // No fragment -> transparent black
    outputImage[pixel]        = float4(0.0f, 0.0f, 0.0f, 0.0f);
    accumulationImage[pixel]  = float4(0.0f, 0.0f, 0.0f, 0.0f);
    return;
  }

  const uint gaussianIdx = indexBuffer[linearIndex];
  if(gaussianIdx == 0xFFFFFFFFu || gaussianIdx >= uint(frameInfo.splatCount))
  {
    outputImage[pixel]        = float4(0.0f, 0.0f, 0.0f, 0.0f);
    accumulationImage[pixel]  = float4(0.0f, 0.0f, 0.0f, 0.0f);
    return;
  }

  // Evaluate SH color for the winning Gaussian
  const float3 splatCenter = fetchCenter(gaussianIdx);
  const float3 viewDir     = normalize(splatCenter - frameInfo.cameraPosition);
  float4 splatColor  = fetchColor(gaussianIdx);
  splatColor.rgb    += fetchViewDependentRadiance(gaussianIdx, viewDir);
  const float4 color = float4(max(splatColor.rgb, 0.0f), 1.0f);  // force alpha=1

  // Progressive EMA accumulation
  if(frameInfo.stochasticFrameCounter == 0)
    accumulationImage[pixel] = color;
  else
  {
    const float4 prev = accumulationImage[pixel];
    const float  a    = 1.0f / (float(frameInfo.stochasticFrameCounter) + 1.0f);
    accumulationImage[pixel] = lerp(prev, color, a);
  }
  outputImage[pixel] = color;
}
