/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * AgX tone mapping implementation ported from Three.js (MIT licensed).
 * Benjamin Wrensch's approximation of Troy Sobotka's AgX curve.
 * See: https://iolite-engine.com/blog_posts/minimal_agx_implementation
 */

#ifndef _TONEMAP_
#define _TONEMAP_

/*-------------------------------------------------------------------------------------------------
# Function `tonemapAgX`
> Benjamin Wrensch's approximation to the AgX tone mapping curve by Troy Sobotka.

Converts linear HDR RGB to sRGB display space using the AgX tone mapping operator.
The sRGB gamma is baked into the output — do NOT call additional sRGB conversion.

From https://iolite-engine.com/blog_posts/minimal_agx_implementation
-------------------------------------------------------------------------------------------------*/
float3 tonemapAgX(float3 color)
{
  // AgX input transform matrix (row-major for mul(color, matrix) convention)
  const float3x3 agx_mat = float3x3(0.842479062253094F, 0.0423282422610123F, 0.0423756549057051F,  //
                                    0.0784335999999992F, 0.878468636469772F, 0.0784336F,           //
                                    0.0792237451477643F, 0.0791661274605434F, 0.879142973793104F);
  color = mul(color, agx_mat);

  // Log2 space encoding
  const float min_ev = -12.47393f;
  const float max_ev =  4.026069f;
  color = clamp(log2(color), min_ev, max_ev);
  color = (color - min_ev) / (max_ev - min_ev);

  // Apply 6th-order sigmoid function approximation via Horner's method
  float3 v = fma(float3(15.5F), color, float3(-40.14F));
  v        = fma(color, v, float3(31.96F));
  v        = fma(color, v, float3(-6.868F));
  v        = fma(color, v, float3(0.4298F));
  v        = fma(color, v, float3(0.1191F));
  v        = fma(color, v, float3(-0.0023F));

  // AgX output transform matrix (inverse of input, row-major for mul(vector, matrix) convention)
  const float3x3 agx_mat_inv = float3x3(1.19687900512017F, -0.0528968517574562F, -0.0529716355144438F,  //
                                        -0.0980208811401368F, 1.15190312990417F, -0.0980434501171241F,  //
                                        -0.0990297440797205F, -0.0989611768448433F, 1.15107367264116F);
  v = mul(v, agx_mat_inv);

  // Skip pow(v, 2.2) — output is already in sRGB display space.
  return v;
}

/*-------------------------------------------------------------------------------------------------
# Function `applyToneMapping`
> Wrapper that applies exposure and the AgX tone mapper for the full HDR → sRGB pipeline.

- Multiplies linear HDR color by exposure
- Applies tonemapAgX for HDR-to-sRGB conversion
- Clamps result to [0, 1] for display-safe output
-------------------------------------------------------------------------------------------------*/
float3 applyToneMapping(float3 color, float exposure)
{
  color *= exposure;
  color = tonemapAgX(color);
  return clamp(color, float3(0.0), float3(1.0));
}

#endif  // _TONEMAP_
