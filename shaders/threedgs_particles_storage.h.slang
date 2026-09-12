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

#ifndef _THREEDGS_PARTICLE_STORAGE_H_
#define _THREEDGS_PARTICLE_STORAGE_H_

// clang-format off
#if DATA_STORAGE == STORAGE_LCC_PACKED
  // LCC packed format: raw 32-byte packed data, GPU-side decompression
  #include "lcc_packed_storage.h.slang"
#elif DATA_STORAGE == STORAGE_TEXTURES
  // textures map describing the 3DGS model
  [[vk::binding(BINDING_CENTERS_TEXTURE, 0)]] Sampler2D centersTexture;
  [[vk::binding(BINDING_SCALES_TEXTURE, 0)]] Sampler2D scalesTexture;
  [[vk::binding(BINDING_ROTATIONS_TEXTURE, 0)]] Sampler2D rotationsTexture;
  [[vk::binding(BINDING_COLORS_TEXTURE, 0)]] Sampler2D colorsTexture;
  [[vk::binding(BINDING_COVARIANCES_TEXTURE, 0)]] Sampler2D covariancesTexture;
  [[vk::binding(BINDING_SH_TEXTURE, 0)]] Sampler2D sphericalHarmonicsTexture;
  [[vk::binding(BINDING_MOTION_TEXTURE, 0)]] Sampler2D motionTexture;
  [[vk::binding(BINDING_TIME_TEXTURE, 0)]] Sampler2D timeTexture;
  [[vk::binding(BINDING_BASECOLOR_TEXTURE, 0)]] Sampler2D basecolorTexture;
  [[vk::binding(BINDING_ROUGHNESS_TEXTURE, 0)]] Sampler2D roughnessTexture;
  [[vk::binding(BINDING_METALLIC_TEXTURE, 0)]] Sampler2D metallicTexture;
#else
  // buffers describing the 3DGS model (alternative to textures)
  [[vk::binding(BINDING_CENTERS_BUFFER, 0)]] RWStructuredBuffer<float> centersBuffer;
  [[vk::binding(BINDING_SCALES_BUFFER, 0)]] RWStructuredBuffer<float> scalesBuffer;
  [[vk::binding(BINDING_ROTATIONS_BUFFER, 0)]] RWStructuredBuffer<float> rotationsBuffer;
  [[vk::binding(BINDING_COLORS_BUFFER, 0)]] RWStructuredBuffer<float> colorsBuffer;
  [[vk::binding(BINDING_COVARIANCES_BUFFER, 0)]] RWStructuredBuffer<float> covariancesBuffer;
  [[vk::binding(BINDING_MOTION_BUFFER, 0)]] RWStructuredBuffer<float> motionBuffer;
  [[vk::binding(BINDING_TIME_BUFFER, 0)]] RWStructuredBuffer<float> timeBuffer;
  [[vk::binding(BINDING_VARIANCE_BUFFER, 0)]] RWStructuredBuffer<float> varianceBuffer;
  [[vk::binding(BINDING_BASECOLOR_BUFFER, 0)]] RWStructuredBuffer<float> basecolorBuffer;
  [[vk::binding(BINDING_ROUGHNESS_BUFFER, 0)]] RWStructuredBuffer<float> roughnessBuffer;
  [[vk::binding(BINDING_METALLIC_BUFFER, 0)]] RWStructuredBuffer<float> metallicBuffer;
  #if SH_FORMAT == FORMAT_FLOAT32
    [[vk::binding(BINDING_SH_BUFFER, 0)]] RWStructuredBuffer<float> sphericalHarmonicsBuffer;
  #else
  #if SH_FORMAT == FORMAT_FLOAT16 
    [[vk::binding(BINDING_SH_BUFFER, 0)]] RWStructuredBuffer<float16_t> sphericalHarmonicsBuffer;
  #else
  #if SH_FORMAT == FORMAT_UINT8
    [[vk::binding(BINDING_SH_BUFFER, 0)]] RWStructuredBuffer<uint8_t> sphericalHarmonicsBuffer;
  #else
    #error "Unsupported SH format"
  #endif
  #endif
  #endif
#endif
// clang-format on

////////////
// constants

static const float sqrt8    = sqrt(8.0);
// SH_C1 may also be defined by stochasticgs.h.slang (same value, different
// trailing digits). Guard against redefinition when both headers are included
// in the same translation unit (e.g. stochasticgs_resolve.comp.slang).
#ifndef SH_C1_DEFINED
#define SH_C1_DEFINED
static const float SH_C1    = 0.4886025119029199f;
#endif
static const float SH_C2[5] = { 1.0925484, -1.0925484, 0.3153916, -1.0925484, 0.5462742 };
static const float SH_C3[7] = { -0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f, 0.3731763325901154f,
                                -0.4570457994644658f, 1.445305721320277f, -0.5900435899266435f };

// data texture accessors
int2 getDataPos(in uint splatIndex, in uint stride, in uint offset, in int2 dimensions)
{
  const uint fullOffset = splatIndex * stride + offset;

  return int2(fullOffset % dimensions.x, fullOffset / dimensions.x);
}

// data texture accessors
int2 getDataPosF(in uint splatIndex, in float stride, in uint offset, in int2 dimensions)
{
  const uint fullOffset = uint(float(splatIndex) * stride) + offset;

  return int2(fullOffset % dimensions.x, fullOffset / dimensions.x);
}

#if DATA_STORAGE == STORAGE_LCC_PACKED
// fetch center value from LCC packed buffer (GPU decompression)
float3 fetchCenter(in uint splatIndex)
{
  return fetchCenterLcc(splatIndex);
}
#elif DATA_STORAGE == STORAGE_TEXTURES
// fetch center value from texture map
float3 fetchCenter(in uint splatIndex)
{
  int2 texSize;
  centersTexture.GetDimensions(texSize.x, texSize.y);
  return centersTexture.Load(int3(getDataPos(splatIndex, 1, 0, texSize), 0)).xyz;
}
#else
// fetch center value from data buffer
float3 fetchCenter(in uint splatIndex)
{
  return float3(centersBuffer[splatIndex * 3 + 0], centersBuffer[splatIndex * 3 + 1], centersBuffer[splatIndex * 3 + 2]);
}
#endif

#if DATA_STORAGE == STORAGE_LCC_PACKED
// fetch scale value from LCC packed buffer (GPU decompression)
// Note: frameInfo.lccScaleMin/Max must be set from LCC metadata
float3 fetchScale(in uint splatIndex)
{
  return fetchScaleLcc(splatIndex, frameInfo.lccScaleMin, frameInfo.lccScaleMax);
}
#elif DATA_STORAGE == STORAGE_TEXTURES
// fetch scale value from texture map
float3 fetchScale(in uint splatIndex)
{
  int2 texSize;
  scalesTexture.GetDimensions(texSize.x, texSize.y);
  return scalesTexture.Load(int3(getDataPos(splatIndex, 1, 0, texSize), 0)).xyz;
}
#else
// fetch scale value from data buffer
float3 fetchScale(in uint splatIndex)
{
  return float3(scalesBuffer[splatIndex * 3 + 0], scalesBuffer[splatIndex * 3 + 1], scalesBuffer[splatIndex * 3 + 2]);
}
#endif

#if DATA_STORAGE == STORAGE_LCC_PACKED
// fetch pre-computed variance (exp(2*scale)) from LCC packed buffer
float3 fetchVariance(in uint splatIndex)
{
  // LCC doesn't have a variance buffer; compute from scale on-the-fly
  return exp(2.0f * fetchScale(splatIndex));
}
#elif DATA_STORAGE == STORAGE_TEXTURES
// fetch pre-computed variance from texture (fallback: compute from scale)
float3 fetchVariance(in uint splatIndex)
{
  // Textures don't have a variance texture; compute from scale on-the-fly
  return exp(2.0f * fetchScale(splatIndex));
}
#else
// fetch pre-computed variance from data buffer
float3 fetchVariance(in uint splatIndex)
{
  return float3(varianceBuffer[splatIndex * 3 + 0], varianceBuffer[splatIndex * 3 + 1], varianceBuffer[splatIndex * 3 + 2]);
}
#endif

#if DATA_STORAGE == STORAGE_LCC_PACKED
// fetch rotation value from LCC packed buffer (GPU decompression)
float4 fetchRotation(in uint splatIndex)
{
  return fetchRotationLcc(splatIndex);
}
#elif DATA_STORAGE == STORAGE_TEXTURES
// fetch rotation value from texture map
float4 fetchRotation(in uint splatIndex)
{
  int2 texSize;
  rotationsTexture.GetDimensions(texSize.x, texSize.y);
  return rotationsTexture.Load(int3(getDataPos(splatIndex, 1, 0, texSize), 0));
}
#else
// fetch rotation value from data buffer
float4 fetchRotation(in uint splatIndex)
{
  return float4(rotationsBuffer[splatIndex * 4 + 0], rotationsBuffer[splatIndex * 4 + 1],
                rotationsBuffer[splatIndex * 4 + 2], rotationsBuffer[splatIndex * 4 + 3]);
}
#endif

#if DATA_STORAGE == STORAGE_LCC_PACKED
// LCC doesn't have motion/time data, return zeros
float3 fetchMotion(in uint splatIndex)
{
  return float3(0.0, 0.0, 0.0);
}
float2 fetchTime(in uint splatIndex)
{
  return float2(0.0, 0.0);
}
float fetchGate(in uint splatIndex)
{
  return 0.0;
}
#elif DATA_STORAGE == STORAGE_TEXTURES
// fetch motion value from texture map
float3 fetchMotion(in uint splatIndex)
{
  int2 texSize;
  motionTexture.GetDimensions(texSize.x, texSize.y);
  return motionTexture.Load(int3(getDataPos(splatIndex, 1, 0, texSize), 0)).xyz;
}

// fetch time value from texture map
float2 fetchTime(in uint splatIndex)
{
  int2 texSize;
  timeTexture.GetDimensions(texSize.x, texSize.y);
  return timeTexture.Load(int3(getDataPos(splatIndex, 1, 0, texSize), 0)).xy;
}

// fetch temporal gate from the motion texture's unused .w component
float fetchGate(in uint splatIndex)
{
  int2 texSize;
  motionTexture.GetDimensions(texSize.x, texSize.y);
  return motionTexture.Load(int3(getDataPos(splatIndex, 1, 0, texSize), 0)).w;
}
#else
// fetch motion value from data buffer
float3 fetchMotion(in uint splatIndex)
{
  return float3(motionBuffer[splatIndex * 4 + 0], motionBuffer[splatIndex * 4 + 1], motionBuffer[splatIndex * 4 + 2]);
}

// fetch time value from data buffer
float2 fetchTime(in uint splatIndex)
{
  return float2(timeBuffer[splatIndex * 2 + 0], timeBuffer[splatIndex * 2 + 1]);
}

// fetch temporal gate from the padded motion buffer .w component
float fetchGate(in uint splatIndex)
{
  return motionBuffer[splatIndex * 4 + 3];
}
#endif

// --- Material data (TRON PBR) ---
#if DATA_STORAGE == STORAGE_LCC_PACKED
float3 fetchBasecolor(in uint splatIndex) { return fetchColor(splatIndex).rgb; }
float fetchRoughness(in uint splatIndex) { return 0.5; }
float fetchMetallic(in uint splatIndex) { return 0.0; }
#elif DATA_STORAGE == STORAGE_TEXTURES
float3 fetchBasecolor(in uint splatIndex) {
    int2 texSize; basecolorTexture.GetDimensions(texSize.x, texSize.y);
    return basecolorTexture.Load(int3(getDataPos(splatIndex, 1, 0, texSize), 0)).xyz;
}
float fetchRoughness(in uint splatIndex) {
    int2 texSize; roughnessTexture.GetDimensions(texSize.x, texSize.y);
    return roughnessTexture.Load(int3(getDataPos(splatIndex, 1, 0, texSize), 0)).r;
}
float fetchMetallic(in uint splatIndex) {
    int2 texSize; metallicTexture.GetDimensions(texSize.x, texSize.y);
    return metallicTexture.Load(int3(getDataPos(splatIndex, 1, 0, texSize), 0)).r;
}
#else
float3 fetchBasecolor(in uint splatIndex) {
    return float3(basecolorBuffer[splatIndex * 3 + 0],
                  basecolorBuffer[splatIndex * 3 + 1],
                  basecolorBuffer[splatIndex * 3 + 2]);
}
float fetchRoughness(in uint splatIndex) { return roughnessBuffer[splatIndex]; }
float fetchMetallic(in uint splatIndex) { return metallicBuffer[splatIndex]; }
#endif

/*
// fetch opacity value from texture map
float fetchOpacity(in uint splatIndex)
{
#if DATA_STORAGE == STORAGE_TEXTURES
  // TODO this is wrong
  return texelFetch(opacityTexture, getDataPos(splatIndex, 1, 0, textureSize(opacityTexture, 0));
#else
  return opacityBuffer[splatIndex];
#endif
}*/

#if DATA_STORAGE == STORAGE_LCC_PACKED
// fetchColor for LCC packed format (GPU decompression)
float4 fetchColor(in uint splatIndex)
{
  return fetchColorLcc(splatIndex);
}
#elif DATA_STORAGE == STORAGE_TEXTURES
// fetchColor replaces fetchSH0 since non view dependent color is precomputed on CPU
float4 fetchColor(in uint splatIndex)
{
  int2 texSize;
  colorsTexture.GetDimensions(texSize.x, texSize.y);
  return colorsTexture.Load(int3(getDataPos(splatIndex, 1, 0, texSize), 0));
}
#else
// fetch center value from data buffer
float4 fetchColor(in uint splatIndex)
{
  return float4(colorsBuffer[splatIndex * 4 + 0], colorsBuffer[splatIndex * 4 + 1], colorsBuffer[splatIndex * 4 + 2],
                colorsBuffer[splatIndex * 4 + 3]);
}
#endif

// fetchOpacity - read only the opacity component (avoids fetching RGB when only alpha needed)
#if DATA_STORAGE == STORAGE_LCC_PACKED
float fetchOpacity(in uint splatIndex)
{
  return fetchColor(splatIndex).w;
}
#elif DATA_STORAGE == STORAGE_TEXTURES
float fetchOpacity(in uint splatIndex)
{
  int2 texSize;
  colorsTexture.GetDimensions(texSize.x, texSize.y);
  return colorsTexture.Load(int3(getDataPos(splatIndex, 1, 0, texSize), 0)).w;
}
#else
float fetchOpacity(in uint splatIndex)
{
  return colorsBuffer[splatIndex * 4 + 3];
}
#endif

#if DATA_STORAGE == STORAGE_LCC_PACKED
// LCC Portable format doesn't have full SH coefficients, return zeros
void fetchSh(in uint    splatIndex,
             out float3 shd1[3]
#if MAX_SH_DEGREE >= 2
             ,
             out float3 shd2[5]
#endif
#if MAX_SH_DEGREE >= 3
             ,
             out float3 shd3[7]
#endif
)
{
  // LCC Portable mode only has DC color, no higher-order SH
  shd1[0] = float3(0.0, 0.0, 0.0);
  shd1[1] = float3(0.0, 0.0, 0.0);
  shd1[2] = float3(0.0, 0.0, 0.0);
#if MAX_SH_DEGREE >= 2
  shd2[0] = float3(0.0, 0.0, 0.0);
  shd2[1] = float3(0.0, 0.0, 0.0);
  shd2[2] = float3(0.0, 0.0, 0.0);
  shd2[3] = float3(0.0, 0.0, 0.0);
  shd2[4] = float3(0.0, 0.0, 0.0);
#endif
#if MAX_SH_DEGREE >= 3
  shd3[0] = float3(0.0, 0.0, 0.0);
  shd3[1] = float3(0.0, 0.0, 0.0);
  shd3[2] = float3(0.0, 0.0, 0.0);
  shd3[3] = float3(0.0, 0.0, 0.0);
  shd3[4] = float3(0.0, 0.0, 0.0);
  shd3[5] = float3(0.0, 0.0, 0.0);
  shd3[6] = float3(0.0, 0.0, 0.0);
#endif
}
#elif DATA_STORAGE == STORAGE_TEXTURES
// fetch from data textures
void fetchSh(in uint    splatIndex,
             out float3 shd1[3]
#if MAX_SH_DEGREE >= 2
             ,
             out float3 shd2[5]
#endif
#if MAX_SH_DEGREE >= 3
             ,
             out float3 shd3[7]
#endif
)
{
  const float SphericalHarmonics8BitCompressionRange = 2.0;
  const float3 vec8BitSHShift = float3(SphericalHarmonics8BitCompressionRange / 2.0, SphericalHarmonics8BitCompressionRange / 2.0,
                                       SphericalHarmonics8BitCompressionRange / 2.0);

  int2 texSize;
  sphericalHarmonicsTexture.GetDimensions(texSize.x, texSize.y);

  const uint stride = 12;  // 12 for degree 3, 6 for degree 2
  // fetching degree 1
  const float4 sampledSH0123   = sphericalHarmonicsTexture.Load(int3(getDataPos(splatIndex, stride, 0, texSize), 0));
  const float4 sampledSH4567   = sphericalHarmonicsTexture.Load(int3(getDataPos(splatIndex, stride, 1, texSize), 0));
  const float4 sampledSH891011 = sphericalHarmonicsTexture.Load(int3(getDataPos(splatIndex, stride, 2, texSize), 0));

  const float3 sh1 = sampledSH0123.rgb;
  const float3 sh2 = float3(sampledSH0123.a, sampledSH4567.rg);
  const float3 sh3 = float3(sampledSH4567.ba, sampledSH891011.r);

#if SH_FORMAT != FORMAT_UINT8
  shd1[0] = sh1;
  shd1[1] = sh2;
  shd1[2] = sh3;
#else
  shd1[0] = sh1 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
  shd1[1] = sh2 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
  shd1[2] = sh3 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
#endif

  // fetching degree 2
#if MAX_SH_DEGREE >= 2
  const float4 sampledSH12131415 = sphericalHarmonicsTexture.Load(int3(getDataPos(splatIndex, stride, 3, texSize), 0));
  const float4 sampledSH16171819 = sphericalHarmonicsTexture.Load(int3(getDataPos(splatIndex, stride, 4, texSize), 0));
  const float4 sampledSH20212223 = sphericalHarmonicsTexture.Load(int3(getDataPos(splatIndex, stride, 5, texSize), 0));

  const float3 sh4 = sampledSH891011.gba;
  const float3 sh5 = sampledSH12131415.rgb;
  const float3 sh6 = float3(sampledSH12131415.a, sampledSH16171819.rg);
  const float3 sh7 = float3(sampledSH16171819.ba, sampledSH20212223.r);
  const float3 sh8 = sampledSH20212223.gba;

#if SH_FORMAT != FORMAT_UINT8
  shd2[0] = sh4;
  shd2[1] = sh5;
  shd2[2] = sh6;
  shd2[3] = sh7;
  shd2[4] = sh8;
#else
  shd2[0] = sh4 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
  shd2[1] = sh5 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
  shd2[2] = sh6 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
  shd2[3] = sh7 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
  shd2[4] = sh8 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
#endif
#endif

  // Fetching degree 3
#if MAX_SH_DEGREE >= 3
  const float4 sampledSH24252627 = sphericalHarmonicsTexture.Load(int3(getDataPos(splatIndex, stride, 6, texSize), 0));
  const float4 sampledSH28293031 = sphericalHarmonicsTexture.Load(int3(getDataPos(splatIndex, stride, 7, texSize), 0));
  const float4 sampledSH32333435 = sphericalHarmonicsTexture.Load(int3(getDataPos(splatIndex, stride, 8, texSize), 0));
  const float4 sampledSH36373839 = sphericalHarmonicsTexture.Load(int3(getDataPos(splatIndex, stride, 9, texSize), 0));
  const float4 sampledSH404142   = sphericalHarmonicsTexture.Load(int3(getDataPos(splatIndex, stride, 10, texSize), 0));
  const float4 sampledSH434445   = sphericalHarmonicsTexture.Load(int3(getDataPos(splatIndex, stride, 11, texSize), 0));

  const float3 sh9  = sampledSH24252627.rgb;
  const float3 sh10 = float3(sampledSH24252627.a, sampledSH28293031.rg);
  const float3 sh11 = float3(sampledSH28293031.ba, sampledSH32333435.r);
  const float3 sh12 = sampledSH32333435.gba;
  const float3 sh13 = sampledSH36373839.rgb;
  const float3 sh14 = float3(sampledSH36373839.a, sampledSH404142.rg);
  const float3 sh15 = float3(sampledSH404142.ba, sampledSH434445.r);

#if SH_FORMAT != FORMAT_UINT8
  shd3[0] = sh9;
  shd3[1] = sh10;
  shd3[2] = sh11;
  shd3[3] = sh12;
  shd3[4] = sh13;
  shd3[5] = sh14;
  shd3[6] = sh15;
#else
  shd3[0] = sh9 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
  shd3[1] = sh10 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
  shd3[2] = sh11 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
  shd3[3] = sh12 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
  shd3[4] = sh13 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
  shd3[5] = sh14 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
  shd3[6] = sh15 * SphericalHarmonics8BitCompressionRange - vec8BitSHShift;
#endif
#endif
}
#else
// fetch from data buffers
void fetchSh(in uint    splatIndex,
             out float3 shd1[3]
#if MAX_SH_DEGREE >= 2
             ,
             out float3 shd2[5]
#endif
#if MAX_SH_DEGREE >= 3
             ,
             out float3 shd3[7]
#endif
)
{
  const uint splatStride = 45;

  const float SphericalHarmonics8BitCompressionRange     = 2.0;
  const float SphericalHarmonics8BitCompressionHalfRange = SphericalHarmonics8BitCompressionRange / 2.0;
  const float3 vec8BitSHShift = float3(SphericalHarmonics8BitCompressionHalfRange, SphericalHarmonics8BitCompressionHalfRange,
                                       SphericalHarmonics8BitCompressionHalfRange);
  const float SphericalHarmonics8BitScale = SphericalHarmonics8BitCompressionRange / 255.0f;

  // fetching degree 1
  const float3 sh1 = float3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 0 + 0],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 0 + 1],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 0 + 2]);

  const float3 sh2 = float3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 1 + 0],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 1 + 1],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 1 + 2]);

  const float3 sh3 = float3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 2 + 0],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 2 + 1],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 2 + 2]);

#if SH_FORMAT != FORMAT_UINT8
  shd1[0] = sh1;
  shd1[1] = sh2;
  shd1[2] = sh3;
#else
  shd1[0] = sh1 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd1[1] = sh2 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd1[2] = sh3 * SphericalHarmonics8BitScale - vec8BitSHShift;
#endif

  // fetching degree 2
#if MAX_SH_DEGREE >= 2
  const float3 sh4 = float3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 3 + 0],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 3 + 1],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 3 + 2]);

  const float3 sh5 = float3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 4 + 0],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 4 + 1],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 4 + 2]);

  const float3 sh6 = float3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 5 + 0],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 5 + 1],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 5 + 2]);

  const float3 sh7 = float3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 6 + 0],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 6 + 1],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 6 + 2]);

  const float3 sh8 = float3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 7 + 0],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 7 + 1],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 7 + 2]);

#if SH_FORMAT != FORMAT_UINT8
  shd2[0] = sh4;
  shd2[1] = sh5;
  shd2[2] = sh6;
  shd2[3] = sh7;
  shd2[4] = sh8;
#else
  shd2[0] = sh4 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd2[1] = sh5 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd2[2] = sh6 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd2[3] = sh7 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd2[4] = sh8 * SphericalHarmonics8BitScale - vec8BitSHShift;
#endif
#endif

  // fetching degree 3
#if MAX_SH_DEGREE >= 3
  const float3 sh9 = float3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 8 + 0],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 8 + 1],
                            sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 8 + 2]);

  const float3 sh10 = float3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 9 + 0],
                             sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 9 + 1],
                             sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 9 + 2]);

  const float3 sh11 = float3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 10 + 0],
                             sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 10 + 1],
                             sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 10 + 2]);

  const float3 sh12 = float3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 11 + 0],
                             sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 11 + 1],
                             sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 11 + 2]);

  const float3 sh13 = float3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 12 + 0],
                             sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 12 + 1],
                             sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 12 + 2]);

  const float3 sh14 = float3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 13 + 0],
                             sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 13 + 1],
                             sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 13 + 2]);

  const float3 sh15 = float3(sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 14 + 0],
                             sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 14 + 1],
                             sphericalHarmonicsBuffer[splatStride * splatIndex + 3 * 14 + 2]);

#if SH_FORMAT != FORMAT_UINT8
  shd3[0] = sh9;
  shd3[1] = sh10;
  shd3[2] = sh11;
  shd3[3] = sh12;
  shd3[4] = sh13;
  shd3[5] = sh14;
  shd3[6] = sh15;
#else
  shd3[0] = sh9 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd3[1] = sh10 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd3[2] = sh11 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd3[3] = sh12 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd3[4] = sh13 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd3[5] = sh14 * SphericalHarmonics8BitScale - vec8BitSHShift;
  shd3[6] = sh15 * SphericalHarmonics8BitScale - vec8BitSHShift;
#endif
#endif
}
#endif

// Helper to compute 3D covariance matrix from scale and rotation
// Used by LCC packed storage which stores scale/rotation rather than pre-computed covariance
float3x3 computeCovariance3D(float3 scale, float4 quat)
{
  // Reconstruct rotation matrix from quaternion (q.x=w, q.y=x, q.z=y, q.w=z)
  float w = quat.x, x = quat.y, y = quat.z, z = quat.w;
  
  float3x3 R;
  R[0][0] = 1.0 - 2.0*(y*y + z*z);
  R[0][1] = 2.0*(x*y - w*z);
  R[0][2] = 2.0*(x*z + w*y);
  R[1][0] = 2.0*(x*y + w*z);
  R[1][1] = 1.0 - 2.0*(x*x + z*z);
  R[1][2] = 2.0*(y*z - w*x);
  R[2][0] = 2.0*(x*z - w*y);
  R[2][1] = 2.0*(y*z + w*x);
  R[2][2] = 1.0 - 2.0*(x*x + y*y);
  
  // Scale matrix (diagonal): S = exp(scale) for log-scale
  float3 s = exp(scale);
  float3x3 S = float3x3(s.x, 0.0, 0.0,
                         0.0, s.y, 0.0,
                         0.0, 0.0, s.z);
  
  // Covariance = R * S * S^T * R^T = R * S^2 * R^T
  // Since S is diagonal, S^2 is just squared diagonal
  float3x3 RS = mul(R, S);
  float3x3 cov = mul(RS, transpose(RS));
  
  return cov;
}

#if DATA_STORAGE == STORAGE_LCC_PACKED
float3x3 fetchCovariance(in uint splatIndex)
{
  // LCC packed storage: compute covariance from scale and rotation
  float3 scale = fetchScale(splatIndex);
  float4 rotation = fetchRotation(splatIndex);
  return computeCovariance3D(scale, rotation);
}
#elif DATA_STORAGE == STORAGE_TEXTURES
float3x3 fetchCovariance(in uint splatIndex)
{
  // Use RGBA texture map to store sets of 3 elements requires some offset shifting depending on splatIndex

  const uint  oddOffset        = uint(splatIndex) & uint(0x00000001);
  const uint  doubleOddOffset  = oddOffset * uint(2);
  const bool  isEven           = oddOffset == uint(0);
  const uint  nearestEvenIndex = uint(splatIndex) - oddOffset;
  const float fOddOffset       = float(oddOffset);

  int2 texSize;
  covariancesTexture.GetDimensions(texSize.x, texSize.y);

  const float4 sampledCovarianceA = covariancesTexture.Load(int3(getDataPosF(nearestEvenIndex, 1.5, oddOffset, texSize), 0));

  const float4 sampledCovarianceB =
      covariancesTexture.Load(int3(getDataPosF(nearestEvenIndex, 1.5, oddOffset + uint(1), texSize), 0));

  const float3 cov3D_M11_M12_M13 =
      float3(sampledCovarianceA.rgb) * (1.0 - fOddOffset) + float3(sampledCovarianceA.ba, sampledCovarianceB.r) * fOddOffset;

  const float3 cov3D_M22_M23_M33 =
      float3(sampledCovarianceA.a, sampledCovarianceB.rg) * (1.0 - fOddOffset) + float3(sampledCovarianceB.gba) * fOddOffset;

  return float3x3(cov3D_M11_M12_M13.x, cov3D_M11_M12_M13.y, cov3D_M11_M12_M13.z, cov3D_M11_M12_M13.y, cov3D_M22_M23_M33.x,
                  cov3D_M22_M23_M33.y, cov3D_M11_M12_M13.z, cov3D_M22_M23_M33.y, cov3D_M22_M23_M33.z);
}
#else
float3x3 fetchCovariance(in uint splatIndex)
{
  // Use RGBA texture map to store sets of 3 elements requires some offset shifting depending on splatIndex
  const float3 cov3D_M11_M12_M13 = float3(covariancesBuffer[splatIndex * 6 + 0], covariancesBuffer[splatIndex * 6 + 1],
                                          covariancesBuffer[splatIndex * 6 + 2]);
  const float3 cov3D_M22_M23_M33 = float3(covariancesBuffer[splatIndex * 6 + 3], covariancesBuffer[splatIndex * 6 + 4],
                                          covariancesBuffer[splatIndex * 6 + 5]);

  return float3x3(cov3D_M11_M12_M13.x, cov3D_M11_M12_M13.y, cov3D_M11_M12_M13.z, cov3D_M11_M12_M13.y, cov3D_M22_M23_M33.x,
                  cov3D_M22_M23_M33.y, cov3D_M11_M12_M13.z, cov3D_M22_M23_M33.y, cov3D_M22_M23_M33.z);
}
#endif

float3 fetchViewDependentRadiance(in uint splatIndex, in float3 worldViewDir)
{
  // contribution is null if MAX_SH_DEGREE < 1
  float3 rgb = float3(0.0, 0.0, 0.0);

#if MAX_SH_DEGREE >= 1
  // SH coefficients for degree 1 (1,2,3)
  float3 shd1[3];
#if MAX_SH_DEGREE >= 2
  // SH coefficients for degree 2 (4 5 6 7 8)
  float3 shd2[5];
#endif
#if MAX_SH_DEGREE >= 3
  // SH coefficients for degree 3 (9,10,11,12,13,14,15)
  float3 shd3[7];
#endif
  // fetch the data (only what is needed according to degree)
  fetchSh(splatIndex, shd1
#if MAX_SH_DEGREE >= 2
          ,
          shd2
#endif
#if MAX_SH_DEGREE >= 3
          ,
          shd3
#endif
  );

  const float x = worldViewDir.x;
  const float y = worldViewDir.y;
  const float z = worldViewDir.z;

  // Degree 1 contributions
  rgb += SH_C1 * (-shd1[0] * y + shd1[1] * z - shd1[2] * x);

#if MAX_SH_DEGREE >= 2
  const float xx = x * x;
  const float yy = y * y;
  const float zz = z * z;
  const float xy = x * y;
  const float yz = y * z;
  const float xz = x * z;

  // Degree 2 contributions
  rgb += (SH_C2[0] * xy) * shd2[0] + (SH_C2[1] * yz) * shd2[1] + (SH_C2[2] * (2.0 * zz - xx - yy)) * shd2[2]
         + (SH_C2[3] * xz) * shd2[3] + (SH_C2[4] * (xx - yy)) * shd2[4];
#endif
#if MAX_SH_DEGREE >= 3

  const float xyy = x * yy;
  const float yzz = y * zz;
  const float zxx = z * xx;
  const float xyz = x * y * z;

  // Degree 3 contributions
  rgb += SH_C3[0] * shd3[0] * (3.0 * x * x - y * y) * y + SH_C3[1] * shd3[1] * x * y * z
         + SH_C3[2] * shd3[2] * (4.0 * z * z - x * x - y * y) * y
         + SH_C3[3] * shd3[3] * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y)
         + SH_C3[4] * shd3[4] * x * (4.0 * z * z - x * x - y * y) + SH_C3[5] * shd3[5] * (x * x - y * y) * z
         + SH_C3[6] * shd3[6] * x * (x * x - 3.0 * y * y);
#endif
#endif

  return rgb;
}

#endif 