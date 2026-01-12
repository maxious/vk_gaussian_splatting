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

#include "lcc_loader.h"

#include <fstream>
#include <cmath>
#include <algorithm>

#include <nvutils/logger.hpp>
#include <tinygltf/json.hpp>

using nlohmann::json;
using namespace vk_viewer;

namespace {

std::vector<uint8_t> readFile(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if(!file.is_open())
        return {};

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(size);
    if(!file.read(reinterpret_cast<char*>(buffer.data()), size))
        return {};

    return buffer;
}

}  // namespace

bool LccLoader::canLoad(const std::filesystem::path& path)
{
    // Check if path is a directory containing meta.lcc
    if(std::filesystem::is_directory(path))
    {
        std::filesystem::path metaPath = path / "meta.lcc";
        return std::filesystem::exists(metaPath);
    }
    
    // Check if path is meta.lcc file itself
    if(path.filename() == "meta.lcc" && std::filesystem::exists(path))
    {
        return true;
    }
    
    return false;
}

bool LccLoader::load(const std::filesystem::path&        path,
                     SplatSet&                           output,
                     std::function<void(float)>          progressCallback)
{
    if(progressCallback)
        progressCallback(0.0f);

    // Determine the base path (directory containing the LCC files)
    std::filesystem::path basePath;
    if(std::filesystem::is_directory(path))
    {
        basePath = path;
    }
    else
    {
        // path is meta.lcc, get parent directory
        basePath = path.parent_path();
    }

    // Parse meta.lcc
    LccMeta meta;
    std::filesystem::path metaPath = basePath / "meta.lcc";
    if(!parseMeta(metaPath, meta))
    {
        return false;
    }

    if(progressCallback)
        progressCallback(0.1f);

    // Read data.bin
    std::filesystem::path dataPath = basePath / "data.bin";
    std::vector<uint8_t> data = readFile(dataPath);
    if(data.empty())
    {
        LOGE("Failed to read data.bin: %s\n", dataPath.string().c_str());
        return false;
    }

    if(progressCallback)
        progressCallback(0.2f);

    // Parse data.bin and convert to SplatSet
    if(!parseData(meta, data.data(), data.size(), output, progressCallback))
    {
        return false;
    }

    // Convert coordinate system from RDF (LCC) to RUB (viewer)
    convertCoordinates(output);

    if(progressCallback)
        progressCallback(1.0f);

    LOGI("Loaded LCC file: %u splats\n", static_cast<uint32_t>(output.size()));
    return true;
}

bool LccLoader::parseMeta(const std::filesystem::path& metaPath, LccMeta& meta)
{
    std::vector<uint8_t> jsonData = readFile(metaPath);
    if(jsonData.empty())
    {
        LOGE("Failed to read meta.lcc: %s\n", metaPath.string().c_str());
        return false;
    }

    try
    {
        json j = json::parse(jsonData.begin(), jsonData.end());

        meta.version      = j.value("version", std::string("0.0"));
        meta.totalSplats  = j.value("totalSplats", 0u);
        meta.totalLevel   = j.value("totalLevel", 1u);
        meta.indexDataSize = j.value("indexDataSize", 0u);
        meta.fileType     = j.value("fileType", "Portable");
        meta.guid         = j.value("guid", "");

        // Parse bounding box
        if(j.contains("boundingBox"))
        {
            auto& bb = j["boundingBox"];
            auto  minArr = bb["min"].get<std::vector<float>>();
            auto  maxArr = bb["max"].get<std::vector<float>>();
            if(minArr.size() == 3 && maxArr.size() == 3)
            {
                meta.boundingMin = glm::vec3(minArr[0], minArr[1], minArr[2]);
                meta.boundingMax = glm::vec3(maxArr[0], maxArr[1], maxArr[2]);
            }
        }

        // Parse splats per level
        if(j.contains("splats"))
        {
            meta.splatsPerLevel = j["splats"].get<std::vector<uint32_t>>();
        }

        // Parse attribute ranges
        if(j.contains("attributes"))
        {
            for(const auto& attr : j["attributes"])
            {
                std::string name = attr.value("name", "");
                auto        minArr =
                    attr.contains("min") ? attr["min"].get<std::vector<float>>() : std::vector<float>{0.0f};
                auto maxArr =
                    attr.contains("max") ? attr["max"].get<std::vector<float>>() : std::vector<float>{1.0f};

                if(name == "position" && minArr.size() == 3 && maxArr.size() == 3)
                {
                    meta.position.min = minArr[0];
                    meta.position.max = maxArr[0];  // Use X range for normalization
                }
                else if(name == "scale" && minArr.size() == 3 && maxArr.size() == 3)
                {
                    meta.scale.min = minArr[0];
                    meta.scale.max = maxArr[0];
                }
                else if(name == "opacity" && !minArr.empty() && !maxArr.empty())
                {
                    meta.opacity.min = minArr[0];
                    meta.opacity.max = maxArr[0];
                }
                else if(name == "shcoef" && minArr.size() == 3 && maxArr.size() == 3)
                {
                    meta.shcoef.min = minArr[0];
                    meta.shcoef.max = maxArr[0];
                }
                else if(name == "normal" && minArr.size() == 3 && maxArr.size() == 3)
                {
                    meta.normal.min = minArr[0];
                    meta.normal.max = maxArr[0];
                }
            }
        }

        // If no attributes found, use defaults
        if(meta.scale.min == 0.0f && meta.scale.max == 1.0f)
        {
            meta.scale.min = 0.00001f;
            meta.scale.max = 5.0f;
        }
        if(meta.opacity.min == 0.0f && meta.opacity.max == 1.0f)
        {
            meta.opacity.min = 0.0f;
            meta.opacity.max = 1.0f;
        }

        return meta.totalSplats > 0;
    }
    catch(const std::exception& e)
    {
        LOGE("Failed to parse meta.lcc: %s\n", e.what());
        return false;
    }
}

void LccLoader::decodeRotation(uint32_t encoded, float* quatOut)
{
    // Extract 4 × 10-bit values from uint32
    // Layout: [9:0] = a, [19:10] = b, [29:20] = c, [31:30] = mode
    uint32_t a = encoded & 0x3FF;
    uint32_t b = (encoded >> 10) & 0x3FF;
    uint32_t c = (encoded >> 20) & 0x3FF;
    uint32_t mode = (encoded >> 30) & 0x3;

    // Normalize to [0, 1]
    float fa = static_cast<float>(a) / 1023.0f;
    float fb = static_cast<float>(b) / 1023.0f;
    float fc = static_cast<float>(c) / 1023.0f;

    // Transform to range [-0.707, 0.707] approximately
    fa = (fa - 0.5f) * 1.414213562f;
    fb = (fb - 0.5f) * 1.414213562f;
    fc = (fc - 0.5f) * 1.414213562f;

    // Use LUT to identify missing component and place values
    const uint8_t lut = ROTATION_LUT[mode];

    // Determine the 4th component (w) from normalization
    float sumSquares = fa * fa + fb * fb + fc * fc;
    float fd = std::sqrt(std::max(0.0f, 1.0f - sumSquares));

    // Place components based on mode (from LUT)
    // LUT format: {3,0,1,2, 0,3,1,2, 0,1,3,2, 0,1,2,3}
    // Each group of 4: mode 0, 1, 2, 3
    // Values: which component goes in that position
    // Components: 0=x, 1=y, 2=z, 3=w
    float components[4] = {fa, fb, fc, fd};

    // Mode 0: w is missing (LUT[0]=3, LUT[1]=0, LUT[2]=1, LUT[3]=2)
    // Mode 1: x is missing
    // Mode 2: y is missing
    // Mode 3: z is missing
    quatOut[0] = components[lut == 3 ? 3 : lut];        // w
    quatOut[1] = components[lut == 0 ? 0 : lut == 3 ? 0 : lut + 1];  // x
    quatOut[2] = components[lut == 1 ? 1 : lut == 3 ? 1 : lut + 1];  // y
    quatOut[3] = components[lut == 2 ? 2 : lut == 3 ? 2 : lut + 1];  // z

    // Actually, let's be more explicit based on the spec:
    // LUT: {3,0,1,2, 0,3,1,2, 0,1,3,2, 0,1,2,3}
    // For mode 0: missing=3 (w), so {x,y,z,w} from {0,1,2,3} mapped by LUT[0..3]
    // LUT[0]=3 means position 0 gets component 3 (w)
    // LUT[1]=0 means position 1 gets component 0 (x)
    // LUT[2]=1 means position 2 gets component 1 (y)
    // LUT[3]=2 means position 3 gets component 2 (z)
    // So for mode 0: quat = {w, x, y, z}

    float q[4] = {fa, fb, fc, fd};
    quatOut[0] = q[ROTATION_LUT[mode * 4 + 0]];  // Position 0
    quatOut[1] = q[ROTATION_LUT[mode * 4 + 1]];  // Position 1
    quatOut[2] = q[ROTATION_LUT[mode * 4 + 2]];  // Position 2
    quatOut[3] = q[ROTATION_LUT[mode * 4 + 3]];  // Position 3

    // Normalize quaternion
    float len = std::sqrt(quatOut[0] * quatOut[0] + quatOut[1] * quatOut[1] + quatOut[2] * quatOut[2]
                          + quatOut[3] * quatOut[3]);
    if(len > 0.0f)
    {
        quatOut[0] /= len;
        quatOut[1] /= len;
        quatOut[2] /= len;
        quatOut[3] /= len;
    }
}

float LccLoader::decodeScale(uint16_t encoded, float min, float max)
{
    // Convert uint16 to float in [0, 1], then interpolate to [min, max]
    float t = static_cast<float>(encoded) / 65535.0f;
    return min + t * (max - min);
}

void LccLoader::decodeColor(uint32_t encoded, float* colorOut, float& opacityOut)
{
    // RGBA: 8 bits per channel
    uint8_t r = (encoded) & 0xFF;
    uint8_t g = (encoded >> 8) & 0xFF;
    uint8_t b = (encoded >> 16) & 0xFF;
    uint8_t a = (encoded >> 24) & 0xFF;

    colorOut[0] = static_cast<float>(r) / 255.0f;
    colorOut[1] = static_cast<float>(g) / 255.0f;
    colorOut[2] = static_cast<float>(b) / 255.0f;

    // Convert opacity (alpha) to sigmoid space for SplatSet
    // LCC stores alpha in [0, 1], convert to logit for sigmoid
    float alpha = static_cast<float>(a) / 255.0f;
    alpha       = std::clamp(alpha, 0.001f, 0.999f);
    opacityOut  = std::log(alpha / (1.0f - alpha));  // Inverse sigmoid
}

bool LccLoader::parseIndex(const std::filesystem::path& indexPath,
                           const LccMeta&              meta,
                           std::vector<LccIndexEntry>& entries)
{
    std::vector<uint8_t> data = readFile(indexPath);
    if(data.empty())
    {
        LOGE("Failed to read index.bin: %s\n", indexPath.string().c_str());
        return false;
    }

    const uint8_t* ptr   = data.data();
    size_t         bytes = data.size();
    size_t         stride = meta.indexDataSize > 0 ? meta.indexDataSize : 68;  // Default stride for v4.0+

    entries.clear();
    for(size_t offset = 0; offset + sizeof(LccIndexEntry) <= bytes; offset += stride)
    {
        const uint8_t* entryPtr = ptr + offset;
        LccIndexEntry  entry;

        // Parse index (uint32: lower 16=X, upper 16=Y)
        uint32_t indexVal = *reinterpret_cast<const uint32_t*>(entryPtr);
        entry.indexX = indexVal & 0xFFFF;
        entry.indexY = (indexVal >> 16) & 0xFFFF;

        // Parse per-LOD data (up to 7 levels)
        const uint32_t* lodPtr = reinterpret_cast<const uint32_t*>(entryPtr + 4);
        for(int i = 0; i < 7; i++)
        {
            entry.pointsCount[i] = lodPtr[i];
        }

        const uint64_t* offsetPtr = reinterpret_cast<const uint64_t*>(entryPtr + 4 + 28);
        for(int i = 0; i < 7; i++)
        {
            entry.lodOffset[i] = offsetPtr[i];
        }

        const uint32_t* sizePtr = reinterpret_cast<const uint32_t*>(entryPtr + 4 + 28 + 56);
        for(int i = 0; i < 7; i++)
        {
            entry.lodSize[i] = sizePtr[i];
        }

        entries.push_back(entry);
    }

    return !entries.empty();
}

bool LccLoader::parseData(const LccMeta&        meta,
                          const uint8_t*        data,
                          size_t                dataSize,
                          SplatSet&             output,
                          std::function<void(float)> progressCallback)
{
    const uint32_t splatCount = meta.totalSplats;
    if(splatCount == 0)
    {
        LOGE("LCC file has 0 splats\n");
        return false;
    }

    // 32 bytes per splat (Portable mode)
    constexpr size_t SPLAT_SIZE = 32;
    size_t           expectedSize = static_cast<size_t>(splatCount) * SPLAT_SIZE;

    if(dataSize < expectedSize)
    {
        LOGE("data.bin too small: expected %zu bytes, got %zu\n", expectedSize, dataSize);
        return false;
    }

    // Reserve space in output
    output.positions.resize(splatCount * 3);
    output.f_dc.resize(splatCount * 3);
    output.f_rest.clear();  // LCC Portable mode doesn't have full SH
    output.opacity.resize(splatCount);
    output.scale.resize(splatCount * 3);
    output.rotation.resize(splatCount * 4);

    // Process splats
    const uint8_t* ptr = data;
    for(uint32_t i = 0; i < splatCount; i++)
    {
        // Position: 3 × float32 (12 bytes) at offset 0
        const float* pos = reinterpret_cast<const float*>(ptr);
        output.positions[i * 3 + 0] = pos[0];
        output.positions[i * 3 + 1] = pos[1];
        output.positions[i * 3 + 2] = pos[2];

        // Color + Opacity: uint32 (4 bytes) at offset 12
        uint32_t colorVal = *reinterpret_cast<const uint32_t*>(ptr + 12);
        float    color[3];
        float    opacity;
        decodeColor(colorVal, color, opacity);
        output.f_dc[i * 3 + 0] = (color[0] - 0.5f) / 0.28209479177387814f;  // Convert RGB to SH coeff
        output.f_dc[i * 3 + 1] = (color[1] - 0.5f) / 0.28209479177387814f;
        output.f_dc[i * 3 + 2] = (color[2] - 0.5f) / 0.28209479177387814f;
        output.opacity[i]      = opacity;

        // Scale: 3 × uint16 (6 bytes) at offset 16
        const uint16_t* scale16 = reinterpret_cast<const uint16_t*>(ptr + 16);
        output.scale[i * 3 + 0] = decodeScale(scale16[0], meta.scale.min, meta.scale.max);
        output.scale[i * 3 + 1] = decodeScale(scale16[1], meta.scale.min, meta.scale.max);
        output.scale[i * 3 + 2] = decodeScale(scale16[2], meta.scale.min, meta.scale.max);

        // Rotation: uint32 (4 bytes) at offset 22
        uint32_t rotVal = *reinterpret_cast<const uint32_t*>(ptr + 22);
        float    quat[4];
        decodeRotation(rotVal, quat);
        // Convert from {w,x,y,z} to SplatSet {w,x,y,z} format
        output.rotation[i * 4 + 0] = quat[0];  // w
        output.rotation[i * 4 + 1] = quat[1];  // x
        output.rotation[i * 4 + 2] = quat[2];  // y
        output.rotation[i * 4 + 3] = quat[3];  // z

        // Normal: 3 × uint16 (6 bytes) at offset 26 (unused for rendering)
        // Skip normals as they're not used by the renderer

        ptr += SPLAT_SIZE;

        // Progress callback
        if(progressCallback && (i % 100000 == 0))
        {
            progressCallback(0.2f + 0.7f * static_cast<float>(i) / splatCount);
        }
    }

    return true;
}

void LccLoader::convertCoordinates(SplatSet& output)
{
    // LCC uses RDF (Right Down Front), viewer uses RUB (Right Up Back)
    // Use spz coordinate converter
    spz::CoordinateConverter c = spz::coordinateConverter(spz::CoordinateSystem::RDF, spz::CoordinateSystem::RUB);

    const size_t numPoints = output.size();

    // Convert positions
    for(size_t i = 0; i < output.positions.size(); i += 3)
    {
        output.positions[i + 0] *= c.flipP[0];
        output.positions[i + 1] *= c.flipP[1];
        output.positions[i + 2] *= c.flipP[2];
    }

    // Convert rotations (only x, y, z components, not w)
    for(size_t i = 0; i < output.rotation.size(); i += 4)
    {
        output.rotation[i + 1] *= c.flipQ[0];
        output.rotation[i + 2] *= c.flipQ[1];
        output.rotation[i + 3] *= c.flipQ[2];
    }
}
