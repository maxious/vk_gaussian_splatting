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

#ifndef _SPLAT_LOADER_FAST_H_
#define _SPLAT_LOADER_FAST_H_

#include <filesystem>
#include <functional>
#include "splat_set.h"

namespace vk_viewer {

// Fast PLY loader for Gaussian Splatting models.
// Uses memory mapping, parallel parsing, and SIMD acceleration.
// Only supports binary_little_endian PLY files with standard 3DGS properties.
class SplatLoaderFast
{
public:
  // Returns true if the file can be loaded by the fast loader.
  // Performs a quick check of the header.
  static bool canLoad(const std::filesystem::path& filename);

  // Loads the scene into the output SplatSet.
  // progressCallback: called with progress in [0, 1].
  static bool load(const std::filesystem::path& filename, SplatSet& output, std::function<void(float)> progressCallback = nullptr);
};

}  // namespace vk_viewer

#endif
