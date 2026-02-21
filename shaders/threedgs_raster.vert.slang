/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
* The code has been adapted to Vulkan from the WebGL-based implementation 
* https://github.com/mkkellogg/GaussianSplats3D. Some mathematical formulations 
* and comments have been directly retained from this source. Original source code  
* licence hereafter.
* ----------------------------------
* The MIT License (MIT)
* 
* Copyright (c) 2023 Mark Kellogg
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#include "shaderio.h"
#include "threedgs.h.slang"
#include "threedgs_particles_storage.h.slang"

// clang-format off

// Vertex input
struct VertexInput
{
  [[vk::location(ATTRIBUTE_LOC_POSITION)]]    float3   inPosition : POSITION;
  [[vk::location(ATTRIBUTE_LOC_SPLAT_INDEX)]] uint32_t inSplatIndex : TEXCOORD0;
#ifdef MULTIVIEW_ENABLED
  uint viewIndex : SV_ViewID;
#endif
};

// Vertex output
struct VertexOutput
{
  [[vk::location(0)]] nointerpolation float4 outFragCol : COLOR0;
#if !USE_BARYCENTRIC
  [[vk::location(1)]] float2 outFragPos : TEXCOORD1;
#endif
  [[vk::location(2)]] float4 position : SV_Position;
};

[[vk::binding(BINDING_FRAME_INFO_UBO, 0)]] ConstantBuffer<FrameInfo> frameInfo;
[[vk::push_constant]] ConstantBuffer<PushConstant> pcRaster;

// clang-format on

[shader("vertex")]
VertexOutput main(VertexInput input)
{
  VertexOutput output;

  const uint splatIndex = input.inSplatIndex;

  // Select view/projection matrices based on multiview mode
#ifdef MULTIVIEW_ENABLED
  float4x4 viewMatrix = (frameInfo.multiviewEnabled != 0) ? frameInfo.viewMatrixArray[input.viewIndex] : frameInfo.viewMatrix;
  float4x4 projMatrix = (frameInfo.multiviewEnabled != 0) ? frameInfo.projectionMatrixArray[input.viewIndex] : frameInfo.projectionMatrix;
  float3 cameraPos = (frameInfo.multiviewEnabled != 0) ? frameInfo.cameraPositionArray[input.viewIndex] : frameInfo.cameraPosition;

  float4x4 modelViewMatrix;
  if (frameInfo.multiviewEnabled != 0) {
      modelViewMatrix = mul(pcRaster.modelMatrix, viewMatrix);
  } else {
      // Optimization: use precomputed modelViewMatrix (stored in modelMatrixRotScaleInverse)
      modelViewMatrix = pcRaster.modelMatrixRotScaleInverse;
  }
#else
  float4x4 viewMatrix = frameInfo.viewMatrix;
  float4x4 projMatrix = frameInfo.projectionMatrix;
  float3 cameraPos = frameInfo.cameraPosition;

  // Optimization: use precomputed modelViewMatrix (stored in modelMatrixRotScaleInverse)
  const float4x4 modelViewMatrix = pcRaster.modelMatrixRotScaleInverse;
#endif

  // Fetch data as early as possible
  const float3x3 cov3Dm      = fetchCovariance(splatIndex);
  float4         splatColor  = fetchColor(splatIndex);
  float3         splatCenter = fetchCenter(splatIndex);

  // Apply 4D Motion
  float3 motion = fetchMotion(splatIndex);
  float2 timeData = fetchTime(splatIndex);
  float t_scale = timeData.x; // JS Order: x = scale
  float t_center = timeData.y; // JS Order: y = center

  // Apply linear motion: P(t) = P0 + V * (t - t0)
  float dt = frameInfo.currentTime - t_center;
  
  if (frameInfo.temporalCulling != 0 && t_center > -5.0) {
      // FreeTimeGS Gaussian opacity decay: sigma(t) = exp(-0.5 * ((t - mu) / s)^2)
      if (t_scale > 1e-5) {
          float v = dt / t_scale;
          float temporalOpacity = exp(-0.5 * v * v);
          splatColor.a *= temporalOpacity;

          // Cull if opacity is negligible to improve performance
          if (splatColor.a < 0.004) { // approx 1/255
              VertexOutput output;
              output.position = float4(0.0, 0.0, 2.0, 1.0);
              return output;
          }
      } else {
          // Zero duration: only visible at exact time center
          if (abs(dt) > 1e-5) {
              VertexOutput output;
              output.position = float4(0.0, 0.0, 2.0, 1.0);
              return output;
          }
      }
  }
  
  if (frameInfo.applyMotion != 0) {
      splatCenter += motion * dt;
  }

  const float4 viewCenter = mul(float4(splatCenter, 1.0), modelViewMatrix);
  const float4 clipCenter = mul(viewCenter, projMatrix);

#if FRUSTUM_CULLING_MODE == FRUSTUM_CULLING_AT_RASTER
  const float clip = (1.0 + frameInfo.frustumDilation) * clipCenter.w;
  if(abs(clipCenter.x) > clip || abs(clipCenter.y) > clip
     || clipCenter.z < (0.0 - frameInfo.frustumDilation) * clipCenter.w || clipCenter.z > clipCenter.w)
  {
    // emit same vertex to get degenerate triangle
    output.position = float4(0.0, 0.0, 2.0, 1.0);
    return output;
  }
#endif

  const float2 fragPos = input.inPosition.xy;
#if !USE_BARYCENTRIC
  // emit as early as possible
  // Scale the position data we send to the fragment shader (using configurable sigma coverage)
  output.outFragPos = fragPos * frameInfo.sigmaCoverage;
#endif

  // alpha based culling
  if(splatColor.a < frameInfo.alphaCullThreshold)
  {
    // emit same vertex to get degenerate triangle
    output.position = float4(0.0, 0.0, 2.0, 1.0);
    return output;
  }

  // Computes the projected covariance
  const float3 cov2Dv = threedgsCovarianceProjection(cov3Dm, viewCenter, frameInfo.focal, modelViewMatrix);

  // computes the basis vectors of the extent of the projected covariance
  // We use configurable sigma coverage (default sqrt(8) ≈ 2.83) standard deviations
  float2 basisVector1, basisVector2;
  if(!threedgsProjectedExtentBasis(cov2Dv, frameInfo.sigmaCoverage, frameInfo.splatScale, splatColor.a, basisVector1, basisVector2))
  {
    // emit same vertex to get degenerate triangle
    output.position = float4(0.0, 0.0, 2.0, 1.0);
    return output;
  }

  // Compute pixel radius from basis vectors for min/max culling
  const float pixelRadius = max(length(basisVector1), length(basisVector2));

  // Min pixel radius culling: cull tiny noise splats
  if(pixelRadius < frameInfo.minPixelRadius)
  {
    output.position = float4(0.0, 0.0, 2.0, 1.0);
    return output;
  }

  // Max pixel radius clamping: limit oversized splats
  if(pixelRadius > frameInfo.maxPixelRadius)
  {
    const float clampScale = frameInfo.maxPixelRadius / pixelRadius;
    basisVector1 *= clampScale;
    basisVector2 *= clampScale;
  }

#if SHOW_SH_ONLY == 1
  splatColor.rgb = float3(0.5, 0.5, 0.5);
#endif

  const float3 worldViewDir =
      normalize(splatCenter - mul(float4(cameraPos, 1.0), pcRaster.modelMatrixInverse).xyz);
  splatColor.rgb += fetchViewDependentRadiance(splatIndex, worldViewDir);

#if MS_ANTIALIASING == 0
  output.outFragCol = splatColor;
#endif
#if MS_ANTIALIASING == 1
  // emit the color with alpha compensation
  output.outFragCol = splatColor;
#endif

  const float3 ndcCenter = clipCenter.xyz / clipCenter.w;

  const float2 ndcOffset = float2(fragPos.x * basisVector1 + fragPos.y * basisVector2) * frameInfo.basisViewport * 2.0
                           * frameInfo.inverseFocalAdjustment;

  const float4 quadPos = float4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0);

  // Emit the vertex position
  output.position = quadPos;

  return output;
}
