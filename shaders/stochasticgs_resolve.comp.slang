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
 * Stochastic Gaussian Splatting - per-pixel resolve kernel.
 *
 * Reads the winning Gaussian index (packed with depth key) from the atomic
 * framebuffer SSBO written by the rasterization pass, evaluates the
 * view-dependent spherical harmonics color for that Gaussian at the camera
 * viewing direction, and writes the result to the output image while
 * progressively accumulating into the accumulation image via an EMA filter.
 *
 * Ported from the CUDA reference implementation (render.cu) from
 * https://github.com/fhahlbohm/stochastic-gaussian-splatting
 */

#include "shaderio.h"
#include "stochasticgs.h.slang"
#include "threedgs_particles_storage.h.slang"

// clang-format off
[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]]                ConstantBuffer<FrameInfo>    frameInfo;
[[vk::binding(BINDING_STOCHASTIC_FRAMEBUFFER_SSBO, 1)]]   StructuredBuffer<uint64_t>  framebufferSSBO;
[[vk::binding(BINDING_STOCHASTIC_OUTPUT_IMAGE, 1)]]       RWTexture2D<float4>         outputImage;
[[vk::binding(BINDING_STOCHASTIC_ACCUMULATION_IMAGE, 1)]] RWTexture2D<float4>         accumulationImage;
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

  // Read the winning fragment for this pixel. The clear kernel initializes
  // every entry to 0xFFFFFFFFFFFFFFFF (no hit); the rasterization pass
  // atomically stores (depth_key << 32) | gaussian_idx for the nearest splat.
  const uint  linearIndex = uint(pixel.y) * width + uint(pixel.x);
  const uint64_t fragment = framebufferSSBO[linearIndex];
  if(fragment == 0xFFFFFFFFFFFFFFFFul)
  {
    // No Gaussian contributed to this pixel - write transparent black.
    outputImage[pixel]        = float4(0.0f, 0.0f, 0.0f, 0.0f);
    accumulationImage[pixel]  = float4(0.0f, 0.0f, 0.0f, 0.0f);
    return;
  }

  // Extract the winning Gaussian index from the low 32 bits of the fragment.
  // The high 32 bits hold the depth key used for atomicMin-based sorting.
  const uint gaussianIdx = uint(fragment & 0xFFFFFFFFu);

  // Fetch the splat center and evaluate the view-dependent SH color.
  const float3 splatCenter = fetchCenter(gaussianIdx);
  const float3 viewDir     = normalize(splatCenter - frameInfo.cameraPosition);

  // fetchColor returns the DC (degree-0) term as float4(rgb, opacity).
  // fetchViewDependentRadiance adds the degree 1..3 view-dependent component.
  float4 splatColor  = fetchColor(gaussianIdx);
  splatColor.rgb    += fetchViewDependentRadiance(gaussianIdx, viewDir);

  const float4 color = float4(max(splatColor.rgb, 0.0f), splatColor.a);

  // Progressive EMA accumulation. On the first frame (counter == 0) the
  // accumulation image is seeded with the current sample; on subsequent
  // frames it converges toward the running mean.
  if(frameInfo.stochasticFrameCounter == 0)
  {
    accumulationImage[pixel] = color;
  }
  else
  {
    const float4 prev = accumulationImage[pixel];
    const float  a    = 1.0f / (float(frameInfo.stochasticFrameCounter) + 1.0f);
    accumulationImage[pixel] = lerp(prev, color, a);
  }

  outputImage[pixel] = color;
}