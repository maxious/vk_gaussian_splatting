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

#include "vxz_loader.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

#include <tinygltf/json.hpp>
#include <nvutils/logger.hpp>
#include <zlib.h>

// Include o-voxel API
// Use angle brackets and rely on include directories set in CMake
#include <io/api.h>

using nlohmann::json;

namespace vk_viewer {

// Helper to decode 3D Morton code (Z-order curve)
// Decodes a 32-bit morton code into x, y, z coordinates
// Assumes standard interleaving: bit 0->x, bit 1->y, bit 2->z, bit 3->x ...
static void mortonDecode(uint32_t code, uint32_t depth, int32_t& x, int32_t& y, int32_t& z) {
    x = 0; y = 0; z = 0;
    // depth includes root level, so number of bits per component is depth - 1
    // Example: depth 2 means root + 1 level of subdivision -> 1 bit per component
    uint32_t levels = depth - 1;
    
    for (uint32_t l = 0; l < levels; ++l) {
        uint32_t shift = l * 3;
        uint32_t octant = (code >> shift) & 0x7;
        
        if (octant & 1) x |= (1 << l);
        if (octant & 2) y |= (1 << l);
        if (octant & 4) z |= (1 << l);
    }
}

bool VxzLoader::load(const std::filesystem::path& filepath, VoxelSet& voxelSet)
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        LOGE("Failed to open file: %s\n", filepath.string().c_str());
        return false;
    }

    // Read header (8 bytes)
    char header[8];
    file.read(header, 8);
    if (file.gcount() != 8) {
        LOGE("Failed to read header from: %s\n", filepath.string().c_str());
        return false;
    }

    if (header[0] != 'V' || header[1] != 'X' || header[2] != 'Z') {
        LOGE("Invalid VXZ file signature\n");
        return false;
    }

    uint8_t version = header[3];
    if (version != 0) {
        LOGE("Unsupported VXZ version: %d\n", version);
        return false;
    }

    // Big-endian binary start offset
    uint32_t binStart = 0;
    uint8_t* binStartPtr = reinterpret_cast<uint8_t*>(header + 4);
    binStart = (binStartPtr[0] << 24) | (binStartPtr[1] << 16) | (binStartPtr[2] << 8) | binStartPtr[3];

    // Read JSON header
    size_t jsonSize = binStart - 8;
    std::vector<char> jsonBuffer(jsonSize);
    file.read(jsonBuffer.data(), jsonSize);
    if ((size_t)file.gcount() != jsonSize) {
        LOGE("Failed to read JSON header\n");
        return false;
    }

    json headerJson;
    try {
        headerJson = json::parse(jsonBuffer.begin(), jsonBuffer.end());
    } catch (const json::parse_error& e) {
        LOGE("JSON parse error: %s\n", e.what());
        return false;
    }

    // Parse metadata
    int numVoxel = headerJson["num_voxel"].get<int>();
    int chunkSize = headerJson["chunk_size"].get<int>();
    std::string filter = headerJson["filter"].get<std::string>();
    std::string compression = headerJson["compression"].get<std::string>();
    std::string attrInterleave = headerJson["attr_interleave"].get<std::string>();

    voxelSet.resolution = 0; 

    // Setup attributes
    std::vector<std::pair<std::string, int>> attrSpecs;
    if (headerJson.contains("attr")) {
        for (const auto& attr : headerJson["attr"]) {
            std::string name = attr[0].get<std::string>();
            int channels = attr[1].get<int>();
            attrSpecs.push_back({name, channels});
            voxelSet.attributes.push_back({name, (uint32_t)channels, {}});
            voxelSet.attributes.back().data.reserve(numVoxel * channels);
        }
    }
    voxelSet.coordinates.reserve(numVoxel * 3);

    // Prepare for binary reading
    if (!headerJson.contains("chunks")) {
        return true; // Empty file?
    }

    for (const auto& chunk : headerJson["chunks"]) {
        std::vector<int> idx = chunk["idx"].get<std::vector<int>>();
        std::vector<size_t> ptr = chunk["ptr"].get<std::vector<size_t>>(); // [offset, length]
        std::vector<size_t> svoInfo = chunk["svo"].get<std::vector<size_t>>(); // [offset, length] relative to chunk start

        size_t chunkOffset = ptr[0];
        size_t chunkLength = ptr[1];

        std::vector<uint8_t> chunkData(chunkLength);
        file.seekg(binStart + chunkOffset, std::ios::beg);
        file.read(reinterpret_cast<char*>(chunkData.data()), chunkLength);
        if ((size_t)file.gcount() != chunkLength) {
            LOGE("Failed to read chunk data\n");
            return false;
        }

        // Decompress SVO
        size_t svoOffset = svoInfo[0];
        size_t svoLen = svoInfo[1];
        if (svoOffset + svoLen > chunkLength) {
            LOGE("SVO data out of bounds\n");
            return false;
        }
        
        std::vector<uint8_t> compressedSvo(chunkData.begin() + svoOffset, chunkData.begin() + svoOffset + svoLen);
        std::vector<uint8_t> decompressedSvo;
        
        // Assume depth corresponds to chunk size
        // 256 -> depth 9 (2^8, plus root)
        uint32_t depth = (uint32_t)std::log2(chunkSize) + 1;

        if (!decompress(compressedSvo, decompressedSvo, compression, 0)) {
             LOGE("Failed to decompress SVO\n");
             return false;
        }
        
        // Decode SVO to Morton codes
        std::vector<uint32_t> morton = decode_sparse_voxel_octree_cpu(decompressedSvo, depth);
        
        // Decode Morton to local coordinates
        size_t numChunkVoxels = morton.size();
        std::vector<int32_t> localCoords(numChunkVoxels * 3);
        
        for(size_t i=0; i<numChunkVoxels; ++i) {
            int32_t x, y, z;
            mortonDecode(morton[i], depth, x, y, z);
            localCoords[i*3 + 0] = x;
            localCoords[i*3 + 1] = y;
            localCoords[i*3 + 2] = z;
        }

        // Add to global coordinate list
        int chunkX = idx[0] * chunkSize;
        int chunkY = idx[1] * chunkSize;
        int chunkZ = idx[2] * chunkSize;
        
        for(size_t i=0; i<numChunkVoxels; ++i) {
            voxelSet.coordinates.push_back(localCoords[i*3 + 0] + chunkX);
            voxelSet.coordinates.push_back(localCoords[i*3 + 1] + chunkY);
            voxelSet.coordinates.push_back(localCoords[i*3 + 2] + chunkZ);
        }
        
        // Process Attributes
        for(size_t a=0; a<attrSpecs.size(); ++a) {
            std::string attrName = attrSpecs[a].first;
            int C = attrSpecs[a].second;
            
            std::vector<size_t> attrInfo;
            if (attrInterleave == "none") {
                std::string key = "attr_" + std::to_string(a);
                if (chunk.contains(key))
                    attrInfo = chunk[key].get<std::vector<size_t>>();
                else continue;
            } else if (attrInterleave == "as_is") {
                if (chunk.contains(attrName))
                    attrInfo = chunk[attrName].get<std::vector<size_t>>();
                else continue;
            } else { // "all"
                 if (chunk.contains("attr")) {
                     attrInfo = chunk["attr"].get<std::vector<size_t>>();
                     // Note: If "all", this logic assumes only 1 attribute or 
                     // that we need to handle interleaving. 
                     // For now, assuming standard TRELLIS VXZ which usually has 1 or split attributes.
                 } else {
                     continue;
                 }
            }
            
            size_t attrOffset = attrInfo[0];
            size_t attrLen = attrInfo[1];
            
            if (attrOffset + attrLen > chunkLength) {
                LOGE("Attribute data out of bounds\n");
                return false;
            }

            std::vector<uint8_t> compressedAttr(chunkData.begin() + attrOffset, chunkData.begin() + attrOffset + attrLen);
            std::vector<uint8_t> decompressedAttr; 
            
            if (!decompress(compressedAttr, decompressedAttr, compression, numChunkVoxels * C)) {
                 LOGE("Failed to decompress attribute %s\n", attrName.c_str());
                 return false;
            }
            
            // Apply inverse filter
            std::vector<uint8_t> finalAttr;
            if (filter == "parent") {
                finalAttr = decode_sparse_voxel_octree_attr_parent_cpu(decompressedSvo, depth, decompressedAttr, C);
            } else if (filter == "neighbor") {
                finalAttr = decode_sparse_voxel_octree_attr_neighbor_cpu(localCoords, chunkSize, decompressedAttr, C);
            } else {
                finalAttr = decompressedAttr;
            }
            
            // Append to global attribute storage
            // Note: voxelSet.attributes order matches attrSpecs order
            auto& globalAttr = voxelSet.attributes[a];
            globalAttr.data.insert(globalAttr.data.end(), finalAttr.begin(), finalAttr.end());
        }
    }

    return true;
}

bool VxzLoader::decompress(const std::vector<uint8_t>& compressed, 
                           std::vector<uint8_t>& decompressed, 
                           const std::string& method, 
                           size_t originalSize)
{
    if (method == "none") {
        decompressed = compressed;
        return true;
    } 
    else if (method == "deflate") {
        // Use zlib
        if (originalSize > 0) {
            decompressed.resize(originalSize);
        } else {
            decompressed.resize(compressed.size() * 10); // Initial guess if unknown
        }
        
        z_stream zs;
        zs.zalloc = Z_NULL;
        zs.zfree = Z_NULL;
        zs.opaque = Z_NULL;
        zs.avail_in = (uInt)compressed.size();
        zs.next_in = (Bytef*)compressed.data();
        zs.avail_out = (uInt)decompressed.size();
        zs.next_out = (Bytef*)decompressed.data();
        
        // raw deflate: windowBits = -15
        if (inflateInit2(&zs, -15) != Z_OK) return false;
        
        int ret = inflate(&zs, Z_FINISH);
        
        if (ret != Z_STREAM_END && ret != Z_OK && ret != Z_BUF_ERROR) {
             inflateEnd(&zs);
             return false;
        }
        
        if (originalSize == 0) {
             decompressed.resize(zs.total_out);
        }
        
        inflateEnd(&zs);
        return true;
    }
    else {
        LOGE("Unsupported compression method: %s\n", method.c_str());
        return false;
    }
}

}  // namespace vk_viewer
