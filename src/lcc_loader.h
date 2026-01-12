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

#ifndef _LCC_LOADER_H_
#define _LCC_LOADER_H_

#include <filesystem>
#include <string>
#include <vector>
#include <functional>
#include <cstdint>

#include <glm/glm.hpp>

#include "splat_set.h"

// 3rd party spz library for coordinate system conversion
#include "splat-types.h"

namespace vk_viewer {

// LCC (Lixel CyberColor) format loader
// Reference: Developer documentation from XGrids
class LccLoader
{
public:
    // Load LCC file from directory containing meta.lcc
    // progressCallback: optional callback for progress updates (0.0 to 1.0)
    static bool load(const std::filesystem::path&        path,
                     SplatSet&                           output,
                     std::function<void(float)>          progressCallback = nullptr);

    // Load specific LOD level (0 = highest quality, higher = lower quality)
    // If targetLod is -1, loads all visible LODs based on view frustum
    static bool loadWithLod(const std::filesystem::path&        path,
                            SplatSet&                           output,
                            int                                 targetLod,
                            const glm::mat4*                    viewProj      = nullptr,
                            const glm::vec3*                    cameraPos     = nullptr,
                            std::function<void(float)>          progressCallback = nullptr);

    // Get number of LOD levels available
    static uint32_t getLodCount(const std::filesystem::path& path);

    // Check if path is an LCC directory (contains meta.lcc)
    static bool canLoad(const std::filesystem::path& path);

public:
    // Metadata parsed from meta.lcc
    struct LccMeta
    {
        std::string  version;     // Version string (e.g., "4.0")
        uint32_t     totalSplats = 0;
        uint32_t     totalLevel = 0;
        uint32_t     indexDataSize = 0;
        float        cellLengthX = 15.0f;  // Spatial cell dimensions in meters
        float        cellLengthY = 15.0f;
        std::string fileType;           // "Portable" or "Quality"
        std::string guid;

        // Bounding box
        glm::vec3 boundingMin{0.0f};
        glm::vec3 boundingMax{0.0f};

        // Attribute ranges for decompression
        struct AttrRange
        {
            float min = 0.0f;
            float max = 1.0f;
        };
        AttrRange position;
        AttrRange scale;
        AttrRange shcoef;
        AttrRange opacity;
        AttrRange normal;

        // LOD info
        std::vector<uint32_t> splatsPerLevel;
    };

    // Index entry from index.bin (one per spatial node)
    struct LccIndexEntry
    {
        uint32_t indexX = 0;      // Lower 16 bits: X index
        uint32_t indexY = 0;      // Upper 16 bits: Y index
        uint32_t pointsCount[7];  // Splat count per LOD level (max 7 levels)
        uint64_t lodOffset[7];    // Byte offset in data.bin per LOD level
        uint32_t lodSize[7];      // Byte size per LOD level
    };

    // Parse meta.lcc JSON content
    static bool parseMeta(const std::filesystem::path& metaPath, LccMeta& meta);

private:
    // Rotation decoding LUT (16 entries as per spec)
    static constexpr uint8_t ROTATION_LUT[16] = {3, 0, 1, 2, 0, 3, 1, 2, 0, 1, 3, 2, 0, 1, 2, 3};

    // Decode rotation from uint32 to quaternion (w, x, y, z)
    static void decodeRotation(uint32_t encoded, float* quatOut);

    // Decode scale from uint16 using min/max range
    static float decodeScale(uint16_t encoded, float min, float max);

    // Decode color from uint32 (RGBA 8-bit per channel)
    static void decodeColor(uint32_t encoded, float* colorOut, float& opacityOut);

    // Parse index.bin to get spatial index entries
    static bool parseIndex(const std::filesystem::path& indexPath,
                           const LccMeta&              meta,
                           std::vector<LccIndexEntry>& entries);

    // Parse data.bin and convert to SplatSet
    static bool parseData(const LccMeta&        meta,
                          const uint8_t*        data,
                          size_t                dataSize,
                          SplatSet&             output,
                          std::function<void(float)> progressCallback);

    // Convert LCC coordinate system (RDF) to viewer coordinate system (RUB)
    static void convertCoordinates(SplatSet& output);
};

}  // namespace vk_viewer

#endif
