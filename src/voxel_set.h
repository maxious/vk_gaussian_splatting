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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <map>

namespace vk_viewer {

struct VoxelAttribute {
    std::string name;
    uint32_t channels;
    std::vector<uint8_t> data; // Flattened N * channels
};

// Storage for Voxel model loaded from VXZ file
struct VoxelSet
{
  std::vector<int32_t> coordinates; // N * 3 (x, y, z)
  uint32_t resolution = 0;
  
  std::vector<VoxelAttribute> attributes;

  // returns the number of voxels in the set
  inline size_t size() const { return coordinates.size() / 3; }

  // Clear all data
  void clear()
  {
    coordinates.clear();
    resolution = 0;
    attributes.clear();
  }
};

}  // namespace vk_viewer
