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

#ifndef _SOG_LOADER_H_
#define _SOG_LOADER_H_

#include <filesystem>
#include <string>
#include <vector>
#include <functional>

#include "splat_set.h"

namespace vk_gaussian_splatting {

// SOG (Spatially Ordered Gaussians) format loader
// Supports both bundled .sog (ZIP archive) and unbundled (meta.json + webp files) formats
// Reference: https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/sog/
class SogLoader
{
public:
  // Load SOG file (bundled .sog or unbundled meta.json)
  // progressCallback: optional callback for progress updates (0.0 to 1.0)
  static bool load(const std::filesystem::path&        filename,
                   SplatSet&                           output,
                   std::function<void(float)>          progressCallback = nullptr);

private:
  // Internal structures for meta.json parsing
  struct MeansInfo
  {
    std::vector<float>       mins;
    std::vector<float>       maxs;
    std::vector<std::string> files;
  };

  struct CodebookInfo
  {
    std::vector<float>       codebook;
    std::vector<std::string> files;
  };

  struct ShNInfo
  {
    uint32_t                 count = 0;
    uint32_t                 bands = 0;
    std::vector<float>       codebook;
    std::vector<std::string> files;
  };

  struct SogMeta
  {
    uint32_t     version = 0;
    uint32_t     count   = 0;
    MeansInfo    means;
    CodebookInfo scales;
    CodebookInfo quats;
    CodebookInfo sh0;
    ShNInfo      shN;
  };

  // WebP image data
  struct WebPImage
  {
    std::vector<uint8_t> rgba;
    uint32_t             width  = 0;
    uint32_t             height = 0;
  };

  // File provider interface for reading files from ZIP or filesystem
  using FileReader = std::function<std::vector<uint8_t>(const std::string& filename)>;

  // Parse meta.json content
  static bool parseMeta(const std::vector<uint8_t>& jsonData, SogMeta& meta);

  // Decode WebP image to RGBA
  static bool decodeWebP(const std::vector<uint8_t>& webpData, WebPImage& output);

  // Decode positions from means_l and means_u
  static void decodePositions(const WebPImage& meansL, const WebPImage& meansU, const SogMeta& meta, SplatSet& output);

  // Decode quaternions
  static void decodeQuaternions(const WebPImage& quats, uint32_t count, SplatSet& output);

  // Decode scales
  static void decodeScales(const WebPImage& scales, const std::vector<float>& codebook, uint32_t count, SplatSet& output);

  // Decode base color (sh0) and opacity
  static void decodeSh0(const WebPImage& sh0, const std::vector<float>& codebook, uint32_t count, SplatSet& output);

  // Decode higher-order SH coefficients (optional)
  static void decodeShN(const WebPImage& centroids, const WebPImage& labels, const ShNInfo& shN, uint32_t count, SplatSet& output);

  // Load from bundled .sog (ZIP archive)
  static bool loadBundled(const std::filesystem::path& sogPath, SplatSet& output, std::function<void(float)> progressCallback);

  // Load from unbundled directory (meta.json + webp files)
  static bool loadUnbundled(const std::filesystem::path& metaPath, SplatSet& output, std::function<void(float)> progressCallback);

  // Common loading logic using file reader
  static bool loadWithReader(const SogMeta& meta, FileReader reader, SplatSet& output, std::function<void(float)> progressCallback);
};

}  // namespace vk_gaussian_splatting

#endif
