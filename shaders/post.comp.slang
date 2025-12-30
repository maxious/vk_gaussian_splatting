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
[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]]  ConstantBuffer<FrameInfo> frameInfo;
[[vk::binding(POST_BINDING_MAIN_IMAGE, 0)]] RWTexture2D<float4>       mainColorImage;
[[vk::binding(POST_BINDING_AUX1_IMAGE, 0)]] RWTexture2D<float4>       aux1ColorImage;
[[vk::binding(POST_BINDING_DEPTH_TEXTURE, 0)]] Texture2D<float>      depthTexture;
[[vk::binding(POST_BINDING_DEPTH_TEXTURE, 0)]] SamplerState           depthSampler;
// clang-format on

[numthreads(32, 32, 1)]
[shader("compute")]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
  const int2 pixel = int2(dispatchThreadID.xy);

  if(pixel.x >= int(frameInfo.viewport.x) || pixel.y >= int(frameInfo.viewport.y))
    return;

  float4 finalColor = mainColorImage[pixel];

  // Visualize VDZ depth texture if requested
  if (frameInfo.visualize == VISUALIZE_VDZ_DEPTH)
  {
    // Sample depth texture with normalized coordinates
    float2 uv = float2(pixel) / frameInfo.viewport;
    float depthValue = depthTexture.Sample(depthSampler, uv);
    
    // Clamp depth for visualization (normalize to displayable range)
    depthValue = clamp(depthValue, 0.0, 1.0);
    
    // Apply a simple grayscale visualization
    finalColor = float4(float3(depthValue), 1.0);
  }
  else
  {
    // Do temporal accumulation if enabled (frameSampleId > 0 means we have previous samples)
    if (frameInfo.frameSampleId > 0)
    {
      const float  a         = 1.0F / float(frameInfo.frameSampleId + 1);
      const float4 aux1Color = aux1ColorImage[pixel];
      finalColor = lerp(finalColor, aux1Color, a);
    }

    // Linear to sRGB conversion for ML-SHARP linearRGB Gaussians
    // SHARP outputs linearRGB colors that need gamma correction for proper display
  }

  // Linear to sRGB conversion for ML-SHARP linearRGB Gaussians
  // SHARP outputs linearRGB colors that need gamma correction for proper display
  if (frameInfo.linearToSrgb != 0)
  {
    // Standard sRGB transfer function (IEC 61966-2-1)
    float3 linear = finalColor.rgb;
    float3 srgb;
    srgb.r = linear.r <= 0.0031308 ? linear.r * 12.92 : 1.055 * pow(linear.r, 1.0 / 2.4) - 0.055;
    srgb.g = linear.g <= 0.0031308 ? linear.g * 12.92 : 1.055 * pow(linear.g, 1.0 / 2.4) - 0.055;
    srgb.b = linear.b <= 0.0031308 ? linear.b * 12.92 : 1.055 * pow(linear.b, 1.0 / 2.4) - 0.055;
    finalColor.rgb = srgb;
  }

  mainColorImage[pixel] = float4(finalColor.rgb, 1.0);
}
