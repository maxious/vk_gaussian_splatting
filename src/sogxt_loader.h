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
 */

#ifndef _SOGXT_LOADER_H_
#define _SOGXT_LOADER_H_

#include <filesystem>
#include <string>
#include <vector>
#include <functional>

#include "splat_set.h"

namespace vk_viewer {

// SOG-XT (KISS-GS) container loader.
//
// SOG-XT is the container format described by the KISS-GS paper
// (arXiv:2608.26948, https://fraunhoferhhi.github.io/KISS-GS/).  It extends
// the Self-Organizing Gaussians (SOG) format with:
//   * positions stored as a coarse/high byte and detail/low byte plane
//     (means_bytes_1.webp / means_bytes_0.webp) after a signed log remap
//   * a view-dependent color codebook stored as a tiled 2D image
//     (f_rest_centroids.webp) indexed by a UV label plane (f_rest_labels.webp)
//
// A container is a directory containing a meta.json (v3) manifest plus a set
// of lossless WebP planes.
class SogXtLoader
{
public:
  // Load a SOG-XT container.  `filename` may point to the container directory,
  // to its meta.json, or to its scene.json manifest.
  static bool load(const std::filesystem::path&       filename,
                   SplatSet&                          output,
                   std::function<void(float)>         progressCallback = nullptr);

public:
  // Internal structures for meta.json (v3) parsing
  struct FieldInfo
  {
    std::vector<float>       mins;   // scalar or per-channel
    std::vector<float>       maxs;   // scalar or per-channel
    std::vector<std::string> files;
    std::string              normalize;  // "observed-minmax"
    std::string              encoding;   // e.g. "direct"
  };

  struct ShNInfo
  {
    uint32_t               coeffs       = 15;
    uint32_t               centroidSide = 0;
    uint32_t               tileRows     = 3;
    uint32_t               tileCols     = 5;
    std::vector<float>     centroidsMins;
    std::vector<float>     centroidsMaxs;
    std::vector<std::string> files;
  };

  struct SogXtMeta
  {
    uint32_t  version  = 0;
    uint32_t  count    = 0;
    uint32_t  gridSide = 0;
    FieldInfo mask;
    FieldInfo means;
    FieldInfo opacities;
    FieldInfo scales;
    FieldInfo quats;
    FieldInfo sh0;
    ShNInfo   shN;
  };

  // Parse meta.json (v3) content
  static bool parseMeta(const std::vector<uint8_t>& jsonData, SogXtMeta& meta);

  // Quick check whether `path` (a container directory, meta.json, or
  // scene.json manifest) is a SOG-XT container, without decoding planes.
  static bool isSogXtManifest(const std::filesystem::path& path);

private:
  // WebP image data
  struct WebPImage
  {
    std::vector<uint8_t> rgba;
    uint32_t             width  = 0;
    uint32_t             height = 0;
  };

  // Decode WebP image to RGBA
  static bool decodeWebP(const std::vector<uint8_t>& webpData, WebPImage& output);

  // Common loading logic using a file reader lambda
  using FileReader = std::function<std::vector<uint8_t>(const std::string& filename)>;
  static bool loadWithReader(const SogXtMeta& meta, FileReader reader, SplatSet& output,
                             std::function<void(float)> progressCallback);

  // Field decode helpers
  static void decodeMask(const WebPImage& mask, uint32_t gridSide, SplatSet& output,
                         std::vector<bool>& active);
  static void decodeMeans(const WebPImage& low, const WebPImage& high, const SogXtMeta& meta,
                          SplatSet& output);
  static void decodeOpacities(const WebPImage& img, const FieldInfo& info, uint32_t gridSide,
                              SplatSet& output);
  static void decodeScales(const WebPImage& img, const FieldInfo& info, uint32_t gridSide,
                           SplatSet& output);
  static void decodeQuats(const WebPImage& img, const FieldInfo& info, uint32_t gridSide,
                          SplatSet& output);
  static void decodeSh0(const WebPImage& img, const FieldInfo& info, uint32_t gridSide,
                        SplatSet& output);
  static void decodeShN(const WebPImage& centroidsImg, const WebPImage& labelsImg,
                        const ShNInfo& shN, uint32_t gridSide, SplatSet& output);

  // Filter decoded grid attributes by the active mask
  static void applyActiveMask(SplatSet& output, const std::vector<bool>& active);
};

}  // namespace vk_viewer

#endif