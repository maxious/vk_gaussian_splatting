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

#ifndef _SPLAT_SET_H_
#define _SPLAT_SET_H_

#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>

// 3rd party spz library, used here for coordinate system convertions
#include "splat-types.h"

namespace vk_gaussian_splatting {

// Storage for a 3D gaussian splatting (3DGS) model loaded from PLY file
struct SplatSet
{
  // standard poiont cloud attributes
  std::vector<float> positions = {};  // point positions (x,y,z)
  // specific data fields introduced by INRIA for 3DGS
  std::vector<float> f_dc     = {};  // 3 components per point (f_dc_0, f_dc_1, f_dc_2 in ply file)
  std::vector<float> f_rest   = {};  // 45 components per point (f_rest_0 to f_rest_44 in ply file), SH coeficients
  std::vector<float> opacity  = {};  // 1 value per point in ply file
  std::vector<float> scale    = {};  // 3 components per point in ply file
  std::vector<float> rotation = {};  // 4 components per point in ply file - a quaternion

  // returns the number of splats in the set
  inline size_t size() const { return positions.size() / 3; }

  // returns the maximumSH degree of splat in the set
  // returns -1 if splat set is invalid
  inline int32_t maxShDegree() const
  {
    const size_t splatCount = size();
    if(splatCount == 0)
      return -1;
    const size_t totalSphericalHarmonicsComponentCount    = (uint32_t)f_rest.size() / splatCount;
    const size_t sphericalHarmonicsCoefficientsPerChannel = totalSphericalHarmonicsComponentCount / 3;
    // find the maximum SH degree stored in the file
    uint32_t sphericalHarmonicsDegree = 0;
    if(sphericalHarmonicsCoefficientsPerChannel >= 3)
    {
      sphericalHarmonicsDegree = 1;
    }
    if(sphericalHarmonicsCoefficientsPerChannel >= 8)
    {
      sphericalHarmonicsDegree = 2;
    }
    if(sphericalHarmonicsCoefficientsPerChannel == 15)
    {
      sphericalHarmonicsDegree = 3;
    }
    return sphericalHarmonicsDegree;
  }

  // sRGB to linearRGB conversion (IEC 61966-2-1)
  // Use this to convert SHARP PLY files exported with compatibility mode back to linear space
  static inline float sRGBToLinear(float srgb)
  {
    if(srgb <= 0.04045f)
      return srgb / 12.92f;
    else
      return std::pow((srgb + 0.055f) / 1.055f, 2.4f);
  }

  // linearRGB to sRGB conversion (IEC 61966-2-1)
  static inline float linearToSRGB(float linear)
  {
    if(linear <= 0.0031308f)
      return linear * 12.92f;
    else
      return 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
  }

  // Convert color space of SH coefficients (f_dc and f_rest)
  // toLinear=true: sRGB -> linearRGB (undo SHARP compatibility export)
  // toLinear=false: linearRGB -> sRGB (for compatibility export)
  void convertColorSpace(bool toLinear)
  {
    auto convertFunc = toLinear ? sRGBToLinear : linearToSRGB;

    // Convert f_dc (base color, 3 components per splat)
    for(size_t i = 0; i < f_dc.size(); ++i)
    {
      // f_dc stores SH coefficients, convert the underlying color value
      // The color is computed as: 0.5 + SH_C0 * f_dc
      // So we need to: (1) extract color, (2) convert, (3) store back as SH coeff
      constexpr float SH_C0     = 0.28209479177387814f;
      float           color     = 0.5f + SH_C0 * f_dc[i];
      float           converted = convertFunc(std::clamp(color, 0.0f, 1.0f));
      f_dc[i]                   = (converted - 0.5f) / SH_C0;
    }

    // Note: f_rest (higher-order SH) represents view-dependent color variations
    // These should also be scaled, but the relationship is more complex.
    // For now, we only convert f_dc which has the dominant effect.
  }

  // Convert between two coordinate systems
  // This is performed in-place.
  void convertCoordinates(spz::CoordinateSystem from, spz::CoordinateSystem to)
  {
    spz::CoordinateConverter c = coordinateConverter(from, to);

    const auto numPoints = size();

    for(size_t i = 0; i < positions.size(); i += 3)
    {
      positions[i + 0] *= c.flipP[0];
      positions[i + 1] *= c.flipP[1];
      positions[i + 2] *= c.flipP[2];
    }
    for(size_t i = 0; i < rotation.size(); i += 4)
    {
      // Don't modify the scalar component (index 0)
      rotation[i + 1] *= c.flipQ[0];
      rotation[i + 2] *= c.flipQ[1];
      rotation[i + 3] *= c.flipQ[2];
    }
    // Rotate spherical harmonics by inverting coefficients that reference the y and z axes, for
    // each RGB channel. See spherical_harmonics_kernel_impl.h for spherical harmonics formulas.
    const size_t numCoeffs         = f_rest.size() / 3;
    const size_t numCoeffsPerPoint = numCoeffs / numPoints;
    size_t       idx               = 0;
    for(size_t i = 0; i < numPoints; ++i)
    {
      // Process R, G, and B coefficients for each point
      for(size_t j = 0; j < numCoeffsPerPoint; ++j)
      {
        const auto flip = c.flipSh[j];
        f_rest[idx + j] *= flip;                          // R
        f_rest[idx + numCoeffsPerPoint + j] *= flip;      // G
        f_rest[idx + numCoeffsPerPoint * 2 + j] *= flip;  // B
      }
      idx += 3 * numCoeffsPerPoint;
    }
  }
};

}  // namespace vk_gaussian_splatting

#endif
