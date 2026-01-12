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

#ifndef _LCC_TILE_MANAGER_H_
#define _LCC_TILE_MANAGER_H_

#include <filesystem>
#include <vector>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <functional>

#include <glm/glm.hpp>

#include "splat_set.h"
#include "lcc_loader.h"

namespace vk_viewer {

// A loaded tile containing a subset of splats
struct LccTile
{
    uint32_t    cellX = 0;
    uint32_t    cellY = 0;
    uint32_t    lodLevel = 0;
    uint32_t    splatCount = 0;
    size_t      dataOffset = 0;
    size_t      dataSize = 0;
    glm::vec3   center{0.0f};
    float       halfExtent = 0.0f;

    // Per-LOD data (shared with index)
    uint32_t    pointsCount[7] = {0};
    uint64_t    lodOffset[7] = {0};
    uint32_t    lodSize[7] = {0};

    SplatSet    splatSet;
    bool        isLoaded = false;
    bool        isLoading = false;
    bool        isDirty = false;

    std::unique_ptr<std::mutex> loadMutex;
};

// Manages tiled loading of LCC scenes with frustum culling and LOD streaming
class LccTileManager
{
public:
    struct Config
    {
        bool        enabled = false;           // Enable tiled streaming
        uint32_t    maxTilesLoaded = 16;       // Max tiles in memory
        uint32_t    minLod = 1;                // Minimum LOD to use
        uint32_t    maxLod = 3;                // Maximum LOD to use
        float       nearDist = 10.0f;          // Near clip for high LOD
        float       midDist = 50.0f;           // Distance for medium LOD
        float       farDist = 150.0f;          // Far clip for low LOD
        float       tileSize = 30.0f;          // Spatial cell size in meters
        bool        asyncLoading = true;       // Load tiles in background
    };

public:
    LccTileManager();
    ~LccTileManager();

    // Initialize with an LCC scene
    bool initialize(const std::filesystem::path& scenePath, const LccLoader::LccMeta& meta);

    // Shutdown and cleanup
    void shutdown();

    // Update based on camera position and view
    void update(const glm::mat4& viewProj, const glm::vec3& cameraPos, float dt);

    // Get the current visible tiles as a merged SplatSet
    void getVisibleSplats(SplatSet& output);

    // Get statistics
    uint32_t getLoadedTileCount() const { return m_loadedTileCount.load(); }
    uint32_t getTotalTileCount() const { return static_cast<uint32_t>(m_allTiles.size()); }
    float    getStreamingProgress() const { return m_streamingProgress.load(); }
    
    // Check if visible tiles have changed since last getVisibleSplats call
    bool hasVisibleTilesChanged() const { return m_visibleTilesDirty.load(); }

    // Configuration
    Config& config() { return m_config; }
    const Config& config() const { return m_config; }

    // Check if tiling is active
    bool isActive() const { return m_config.enabled && m_isInitialized.load(); }

    // Force reload all tiles
    void invalidateAll();

    // Set LOD counts from metadata
    void setLodCount(uint32_t count) { m_lodCount = count; }

private:
    // Parse index file and create tile definitions
    bool parseTiles(const std::filesystem::path& indexPath, const LccLoader::LccMeta& meta);

    // Determine which LOD level to use for a tile based on distance
    uint32_t getLodForDistance(float distance) const;

    // Update the set of visible tiles based on frustum and distance
    void updateVisibleTiles(const glm::mat4& viewProj, const glm::vec3& cameraPos);

    // Load a specific tile
    bool loadTile(LccTile& tile, uint32_t targetLod);

    // Unload a tile to free memory
    void unloadTile(LccTile& tile);

    // Streaming worker thread
    void streamingWorker();

    // Check if a tile is in the view frustum
    bool isInFrustum(const LccTile& tile, const glm::mat4& viewProj) const;

    // Extract 8 corners of a tile's bounding box
    std::array<glm::vec3, 8> getTileCorners(const LccTile& tile) const;

private:
    Config                              m_config;
    std::filesystem::path               m_scenePath;
    std::vector<LccTile>                m_allTiles;
    std::vector<LccTile*>               m_visibleTiles;
    std::vector<LccTile*>               m_loadQueue;
    std::vector<LccTile*>               m_unloadQueue;

    std::atomic<bool>                   m_isInitialized{false};
    std::atomic<bool>                   m_shutdownRequested{false};
    std::atomic<uint32_t>               m_loadedTileCount{0};
    std::atomic<float>                  m_streamingProgress{0.0f};
    std::atomic<uint32_t>               m_lodCount{1};
    std::atomic<bool>                   m_visibleTilesDirty{true};  // Track if visible tiles changed

    std::mutex                          m_tileMutex;
    std::mutex                          m_queueMutex;
    std::condition_variable             m_loadCV;
    std::thread                         m_streamingThread;

    // Memory budget
    std::atomic<size_t>                 m_memoryUsed{0};
    size_t                              m_memoryBudget{0};

    // For frustum culling
    glm::mat4                           m_lastViewProj;
    glm::vec3                           m_lastCameraPos;
};

}  // namespace vk_viewer

#endif
