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
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _MORTON_ORDER_HPP_
#define _MORTON_ORDER_HPP_

#include <cstdint>
#include <algorithm>
#include <limits>

namespace vk_viewer {

// Expand 10-bit integer to 30 bits by spacing out bits for 3D Morton code
inline uint32_t expandBits10(uint32_t v)
{
  v = (v | (v << 16)) & 0x030000FF;  // 0000 0011 0000 0000 0000 0000 1111 1111
  v = (v | (v << 8)) & 0x0300F00F;   // 0000 0011 0000 0000 1111 0000 0000 1111
  v = (v | (v << 4)) & 0x030C30C3;   // 0000 0011 0000 1100 0011 0000 1100 0011
  v = (v | (v << 2)) & 0x09249249;   // 0000 1001 0010 0100 1001 0010 0100 1001
  return v;
}

// Compute 30-bit Morton code for 3D point (10 bits per axis)
// x, y, z should be in [0, 1023]
inline uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z)
{
  return expandBits10(x) | (expandBits10(y) << 1) | (expandBits10(z) << 2);
}

// Compute Morton code from floating point position given bounding box
// Normalizes position to [0, 1023] range based on bounding box
inline uint32_t mortonFromPosition(float px, float py, float pz,
                                   float minX, float minY, float minZ,
                                   float invRangeX, float invRangeY, float invRangeZ)
{
  float nx = (px - minX) * invRangeX;
  float ny = (py - minY) * invRangeY;
  float nz = (pz - minZ) * invRangeZ;

  uint32_t ix = static_cast<uint32_t>(std::clamp(nx * 1023.0f, 0.0f, 1023.0f));
  uint32_t iy = static_cast<uint32_t>(std::clamp(ny * 1023.0f, 0.0f, 1023.0f));
  uint32_t iz = static_cast<uint32_t>(std::clamp(nz * 1023.0f, 0.0f, 1023.0f));

  return morton3D(ix, iy, iz);
}

}  // namespace vk_viewer

#endif
