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

#include "voxel_set.h"
#include <filesystem>
#include <string>
#include <vector>

namespace vk_gaussian_splatting {

class VxzLoader
{
public:
  VxzLoader() = default;
  ~VxzLoader() = default;

  // Load a .vxz file into a VoxelSet
  bool load(const std::filesystem::path& filepath, VoxelSet& voxelSet);

private:
  // Helper to decompress data
  bool decompress(const std::vector<uint8_t>& compressed, 
                  std::vector<uint8_t>& decompressed, 
                  const std::string& method, 
                  size_t originalSize);
};

}  // namespace vk_gaussian_splatting
