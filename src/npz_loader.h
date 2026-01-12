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

#ifndef _NPZ_LOADER_H_
#define _NPZ_LOADER_H_

#include <filesystem>
#include <functional>
#include <string>
#include <vector>
#include <memory>
#include "splat_set.h"

namespace vk_viewer {

// NPZ loader for Gaussian Splatting models from SplatAD and similar formats.
// Supports compressed NumPy archives containing:
// - means3D: [N, 3] or [T, N, 3] positions
// - rgb_colors: [N, 3] or [T, N, 3] colors (SH DC coefficients)
// - logit_opacities: [N] opacity values
// - log_scales: [N, 3] log-scales
// - unnorm_rotations: [N, 4] or [T, N, 4] quaternions
class NpzLoader
{
public:
  // Returns true if the file appears to be a valid NPZ file with splat data.
  static bool canLoad(const std::filesystem::path& filename);

  // Loads the scene into the output SplatSet.
  // For temporal data (T > 1), loads the first frame by default.
  // Use loadTimestep() to load specific frames.
  // progressCallback: called with progress in [0, 1].
  static bool load(const std::filesystem::path& filename, SplatSet& output, std::function<void(float)> progressCallback = nullptr);

  // Load a specific timestep from a temporal NPZ file.
  // timestep: 0-indexed frame index (only valid if hasMultipleTimesteps() returns true)
  static bool loadTimestep(const std::filesystem::path& filename, SplatSet& output, size_t timestep, std::function<void(float)> progressCallback = nullptr);

  // Returns true if the NPZ file contains multiple timesteps (temporal data)
  static bool hasMultipleTimesteps(const std::filesystem::path& filename);

  // Returns the number of timesteps in the file, or 1 for static files
  static size_t getTimestepCount(const std::filesystem::path& filename);

  // Get the expected timestep duration in milliseconds (for playback)
  // Returns 0 if temporal info not available
  static uint32_t getTimestepDurationMs(const std::filesystem::path& filename);
};

}  // namespace vk_viewer

#endif
