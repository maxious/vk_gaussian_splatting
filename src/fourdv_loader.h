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

#ifndef _FOURDV_LOADER_H_
#define _FOURDV_LOADER_H_

#include <filesystem>
#include <vector>
#include <functional>
#include "splat_set.h"

namespace vk_viewer {

class FourDvLoader
{
public:
  static bool load(const std::filesystem::path& filename, SplatSet& output, std::function<void(float)> progressCallback = nullptr);

private:
  struct ChunkInfo
  {
    float min_x, max_x;
    float min_y, max_y;
    float min_z, max_z;
    float min_scale_x, max_scale_x;
    float min_scale_y, max_scale_y;
    float min_scale_z, max_scale_z;
    float min_r, max_r;
    float min_g, max_g;
    float min_b, max_b;
    float min_motion_x, max_motion_x;
    float min_motion_y, max_motion_y;
    float min_motion_z, max_motion_z;
    float min_time_scale, max_time_scale;
    float min_time, max_time;
  };

  struct PackedVertex
  {
    uint32_t packed_position;
    uint32_t packed_rotation;
    uint32_t packed_scale;
    uint32_t packed_color;
    uint32_t packed_motion;
    uint32_t packed_time;
  };

  static void unpackVertex(const PackedVertex& packed, const ChunkInfo& chunk, SplatSet& output, size_t index);
};

} // namespace vk_viewer

#endif
