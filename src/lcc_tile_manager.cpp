/*
 * LCC Tile Manager - Tiled streaming with frustum culling and LOD
 */

#include "lcc_tile_manager.h"

#include <fstream>
#include <algorithm>
#include <cmath>

#include <nvutils/logger.hpp>

namespace vk_viewer {

LccTileManager::LccTileManager()
{
}

LccTileManager::~LccTileManager()
{
    shutdown();
}

bool LccTileManager::initialize(const std::filesystem::path& scenePath, const LccLoader::LccMeta& meta)
{
    m_scenePath = scenePath;
    m_config.tileSize = meta.cellLengthX > 0 ? meta.cellLengthX : 30.0f;

    // Memory budget: ~256MB for tiles
    m_memoryBudget = 256 * 1024 * 1024;

    // Parse tiles from index
    std::filesystem::path indexPath = scenePath / "index.bin";
    if(!parseTiles(indexPath, meta))
    {
        LOGE("Failed to parse LCC index for tiling\n");
        return false;
    }

    // Start streaming thread if async loading enabled
    if(m_config.asyncLoading)
    {
        m_shutdownRequested = false;
        m_streamingThread = std::thread(&LccTileManager::streamingWorker, this);
    }

    m_isInitialized = true;
    LOGI("LccTileManager initialized: %u tiles, %.1f meter cells\n",
         static_cast<uint32_t>(m_allTiles.size()), m_config.tileSize);
    return true;
}

void LccTileManager::shutdown()
{
    m_shutdownRequested = true;
    m_loadCV.notify_all();

    if(m_streamingThread.joinable())
    {
        m_streamingThread.join();
    }

    // Unload all tiles
    for(auto& tile : m_allTiles)
    {
        unloadTile(tile);
    }

    m_allTiles.clear();
    m_visibleTiles.clear();
    m_loadQueue.clear();
    m_isInitialized = false;
}

bool LccTileManager::parseTiles(const std::filesystem::path& indexPath, const LccLoader::LccMeta& meta)
{
    std::ifstream file(indexPath, std::ios::binary | std::ios::ate);
    if(!file.is_open())
    {
        LOGE("Failed to open index.bin: %s\n", indexPath.string().c_str());
        return false;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(size);
    if(!file.read(reinterpret_cast<char*>(data.data()), size))
    {
        return false;
    }

    // Determine entry size from metadata
    size_t entrySize = meta.indexDataSize > 0 ? meta.indexDataSize : 68;
    size_t numEntries = data.size() / entrySize;

    m_allTiles.reserve(numEntries);

    const uint8_t* ptr = data.data();
    for(size_t i = 0; i < numEntries; i++)
    {
        const uint8_t* entry = ptr + i * entrySize;

        uint32_t indexVal = *reinterpret_cast<const uint32_t*>(entry);
        uint32_t cellX = indexVal & 0xFFFF;
        uint32_t cellY = (indexVal >> 16) & 0xFFFF;

        LccTile tile;
        tile.loadMutex = std::make_unique<std::mutex>();
        tile.cellX = cellX;
        tile.cellY = cellY;
        tile.center = glm::vec3(
            static_cast<float>(cellX) * m_config.tileSize,
            static_cast<float>(cellY) * m_config.tileSize,
            (meta.boundingMin.z + meta.boundingMax.z) * 0.5f
        );
        tile.halfExtent = m_config.tileSize * 0.5f;

        // Parse per-LOD data
        const uint8_t* lodPtr = entry + 4;
        for(uint32_t lod = 0; lod < meta.totalLevel && lod < 7; lod++)
        {
            tile.pointsCount[lod] = *reinterpret_cast<const uint32_t*>(lodPtr);
            tile.lodOffset[lod] = *reinterpret_cast<const uint64_t*>(lodPtr + 4);
            tile.lodSize[lod] = *reinterpret_cast<const uint32_t*>(lodPtr + 12);
            lodPtr += 16;
        }

        m_allTiles.emplace_back(std::move(tile));
    }

    return !m_allTiles.empty();
}

uint32_t LccTileManager::getLodForDistance(float distance) const
{
    uint32_t maxLod = m_lodCount.load() > 0 ? m_lodCount.load() - 1 : 0;
    
    if(distance < m_config.nearDist)
        return std::min(maxLod, 0u);  // Highest quality near camera
    else if(distance < m_config.midDist)
        return std::min(maxLod, 1u);
    else if(distance < m_config.farDist)
        return std::min(maxLod, 2u);
    else
        return std::min(maxLod, 3u);  // Lowest quality far away
}

std::array<glm::vec3, 8> LccTileManager::getTileCorners(const LccTile& tile) const
{
    std::array<glm::vec3, 8> corners;
    float h = tile.halfExtent;

    corners[0] = tile.center + glm::vec3(-h, -h, -h);
    corners[1] = tile.center + glm::vec3( h, -h, -h);
    corners[2] = tile.center + glm::vec3( h,  h, -h);
    corners[3] = tile.center + glm::vec3(-h,  h, -h);
    corners[4] = tile.center + glm::vec3(-h, -h,  h);
    corners[5] = tile.center + glm::vec3( h, -h,  h);
    corners[6] = tile.center + glm::vec3( h,  h,  h);
    corners[7] = tile.center + glm::vec3(-h,  h,  h);

    return corners;
}

bool LccTileManager::isInFrustum(const LccTile& tile, const glm::mat4& viewProj) const
{
    // Simple sphere-frustum test using tile center and extent
    glm::vec4 center4(tile.center, 1.0f);
    glm::vec4 clipCenter = viewProj * center4;

    // Check if behind near plane
    if(clipCenter.w < 0.0f && clipCenter.w < -tile.halfExtent)
        return false;

    // Simple distance-based culling as fallback
    float dist = glm::length(tile.center - m_lastCameraPos);
    if(dist > m_config.farDist * 2.0f)
        return false;

    return true;
}

void LccTileManager::updateVisibleTiles(const glm::mat4& viewProj, const glm::vec3& cameraPos)
{
    m_lastViewProj = viewProj;
    m_lastCameraPos = cameraPos;

    size_t prevVisibleCount = m_visibleTiles.size();
    m_visibleTiles.clear();

    // DEBUG: Only load the first tile at LOD 3 to minimize data
    bool first = true;
    for(auto& tile : m_allTiles)
    {
        if(first) // && isInFrustum(tile, viewProj))
        {
            float dist = glm::length(tile.center - cameraPos);
            uint32_t targetLod = getLodForDistance(dist); // Will return 3 (from previous hack)

            // Check if tile needs to be (re)loaded
            bool needsLoad = !tile.isLoaded && !tile.isLoading;
            bool needsReload = tile.isLoaded && tile.lodLevel != targetLod;

            if(needsLoad || needsReload)
            {
                tile.lodLevel = targetLod;
                tile.dataOffset = tile.lodOffset[targetLod];
                tile.dataSize = tile.lodSize[targetLod];
                tile.splatCount = tile.pointsCount[targetLod];

                if(needsReload)
                {
                    unloadTile(tile);
                }

                // Queue for loading
                std::lock_guard<std::mutex> lock(m_queueMutex);
                m_loadQueue.push_back(&tile);
                tile.isLoading = true;
            }

            if(tile.isLoaded)
            {
                m_visibleTiles.push_back(&tile);
            }
            
            // Just one tile for now!
            first = false;
        }
        else
        {
            // Unload everything else
            if(tile.isLoaded && !tile.isLoading)
            {
                unloadTile(tile);
            }
        }
    }

    // Mark dirty if visible tile count changed
    if(m_visibleTiles.size() != prevVisibleCount)
    {
        m_visibleTilesDirty = true;
    }

    m_loadCV.notify_one();
}

bool LccTileManager::loadTile(LccTile& tile, uint32_t targetLod)
{
    if(tile.isLoaded || tile.dataSize == 0)
        return true;

    std::filesystem::path dataPath = m_scenePath / "data.bin";
    std::ifstream file(dataPath, std::ios::binary);
    if(!file.is_open())
    {
        LOGE("Failed to open data.bin for tile loading\n");
        return false;
    }

    file.seekg(static_cast<std::streamoff>(tile.dataOffset));
    std::vector<uint8_t> data(tile.dataSize);
    if(!file.read(reinterpret_cast<char*>(data.data()), tile.dataSize))
    {
        LOGE("Failed to read tile data from data.bin\n");
        return false;
    }

    // Parse the tile data
    constexpr size_t SPLAT_SIZE = 32;
    uint32_t splatCount = tile.dataSize / SPLAT_SIZE;

    tile.splatSet.positions.resize(splatCount * 3);
    tile.splatSet.f_dc.resize(splatCount * 3);
    tile.splatSet.opacity.resize(splatCount);
    tile.splatSet.scale.resize(splatCount * 3);
    tile.splatSet.rotation.resize(splatCount * 4);

    const uint8_t* ptr = data.data();
    for(uint32_t i = 0; i < splatCount; i++)
    {
        const float* pos = reinterpret_cast<const float*>(ptr);
        // Sanitize Position
        float px = pos[0], py = pos[1], pz = pos[2];
        if(!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz)) { px = py = pz = 0.0f; }
        tile.splatSet.positions[i * 3 + 0] = px;
        tile.splatSet.positions[i * 3 + 1] = py;
        tile.splatSet.positions[i * 3 + 2] = pz;

        uint32_t colorVal = *reinterpret_cast<const uint32_t*>(ptr + 12);
        float    color[3];
        float    opacity;
        LccLoader::decodeColor(colorVal, color, opacity);
        tile.splatSet.f_dc[i * 3 + 0] = (color[0] - 0.5f) / 0.28209479177387814f;
        tile.splatSet.f_dc[i * 3 + 1] = (color[1] - 0.5f) / 0.28209479177387814f;
        tile.splatSet.f_dc[i * 3 + 2] = (color[2] - 0.5f) / 0.28209479177387814f;
        // Sanitize Opacity
        if(!std::isfinite(opacity)) opacity = -10.0f; // effectively invisible
        tile.splatSet.opacity[i] = opacity;

        const uint16_t* scale16 = reinterpret_cast<const uint16_t*>(ptr + 16);
        // Use fixed scale range for now
        float sx = LccLoader::decodeScale(scale16[0], 0.00001f, 5.0f);
        float sy = LccLoader::decodeScale(scale16[1], 0.00001f, 5.0f);
        float sz = LccLoader::decodeScale(scale16[2], 0.00001f, 5.0f);
        // Sanitize Scale
        if(!std::isfinite(sx)) sx = 0.01f;
        if(!std::isfinite(sy)) sy = 0.01f;
        if(!std::isfinite(sz)) sz = 0.01f;
        tile.splatSet.scale[i * 3 + 0] = sx;
        tile.splatSet.scale[i * 3 + 1] = sy;
        tile.splatSet.scale[i * 3 + 2] = sz;

        uint32_t rotVal = *reinterpret_cast<const uint32_t*>(ptr + 22);
        float    quat[4];
        LccLoader::decodeRotation(rotVal, quat);
        // Sanitize Rotation
        if(!std::isfinite(quat[0]) || !std::isfinite(quat[1]) || !std::isfinite(quat[2]) || !std::isfinite(quat[3]))
        {
            quat[0] = 1.0f; quat[1] = 0.0f; quat[2] = 0.0f; quat[3] = 0.0f;
        }
        tile.splatSet.rotation[i * 4 + 0] = quat[0];
        tile.splatSet.rotation[i * 4 + 1] = quat[1];
        tile.splatSet.rotation[i * 4 + 2] = quat[2];
        tile.splatSet.rotation[i * 4 + 3] = quat[3];

        ptr += SPLAT_SIZE;
    }

    // Convert coordinates RDF -> RUB
    spz::CoordinateConverter c = spz::coordinateConverter(spz::CoordinateSystem::RDF, spz::CoordinateSystem::RUB);
    for(size_t j = 0; j < tile.splatSet.positions.size(); j += 3)
    {
        tile.splatSet.positions[j + 0] *= c.flipP[0];
        tile.splatSet.positions[j + 1] *= c.flipP[1];
        tile.splatSet.positions[j + 2] *= c.flipP[2];
    }

    for(size_t j = 0; j < tile.splatSet.rotation.size(); j += 4)
    {
        tile.splatSet.rotation[j + 1] *= c.flipQ[0];
        tile.splatSet.rotation[j + 2] *= c.flipQ[1];
        tile.splatSet.rotation[j + 3] *= c.flipQ[2];
    }

    tile.isLoaded = true;
    tile.isLoading = false;
    tile.lodLevel = targetLod;
    m_memoryUsed += tile.dataSize;
    m_loadedTileCount++;

    return true;
}

void LccTileManager::unloadTile(LccTile& tile)
{
    if(!tile.isLoaded)
        return;

    tile.splatSet.clear();
    tile.isLoaded = false;
    tile.isLoading = false;
    m_memoryUsed -= tile.dataSize;
    m_loadedTileCount--;
}

void LccTileManager::streamingWorker()
{
    while(!m_shutdownRequested)
    {
        LccTile* tile = nullptr;

        {
            std::unique_lock<std::mutex> lock(m_queueMutex);
            m_loadCV.wait(lock, [this] {
                return !m_loadQueue.empty() || m_shutdownRequested;
            });

            if(m_shutdownRequested)
                break;

            // Find next tile to load (respecting memory budget)
            while(!m_loadQueue.empty())
            {
                LccTile* candidate = m_loadQueue.back();
                m_loadQueue.pop_back();

                if(candidate->isLoading && !candidate->isLoaded)
                {
                    // Check memory budget
                    size_t required = candidate->dataSize;
                    if(m_memoryUsed + required < m_memoryBudget)
                    {
                        tile = candidate;
                        break;
                    }
                    else
                    {
                        // Memory full, try to unload distant tiles
                        m_unloadQueue.clear();
                        for(auto& t : m_allTiles)
                        {
                            if(t.isLoaded && t.cellX != tile->cellX && t.cellY != tile->cellY)
                            {
                                float tileDist = glm::length(t.center - m_lastCameraPos);
                                float candidateDist = glm::length(tile->center - m_lastCameraPos);
                                if(tileDist > candidateDist)
                                {
                                    m_unloadQueue.push_back(&t);
                                }
                            }
                        }

                        // Unload the most distant tile
                        for(auto* tileToUnload : m_unloadQueue)
                        {
                            if(tileToUnload->isLoaded)
                            {
                                unloadTile(*tileToUnload);
                                break;
                            }
                        }
                    }
                }
            }
        }

        if(tile)
        {
            if(loadTile(*tile, tile->lodLevel))
            {
                m_visibleTilesDirty = true;  // Mark dirty when a tile finishes loading
            }
        }
    }
}

void LccTileManager::update(const glm::mat4& viewProj, const glm::vec3& cameraPos, float dt)
{
    if(!m_isInitialized || !m_config.enabled)
        return;

    // Update visible tiles
    updateVisibleTiles(viewProj, cameraPos);

    // Update streaming progress
    if(!m_loadQueue.empty())
    {
        m_streamingProgress = 1.0f - static_cast<float>(m_loadQueue.size()) / m_allTiles.size();
    }
    else
    {
        m_streamingProgress = 1.0f;
    }
}

void LccTileManager::getVisibleSplats(SplatSet& output)
{
    // Skip rebuild if nothing has changed
    if(!m_visibleTilesDirty.load())
    {
        return;  // output already contains valid data from previous call
    }
    
    output.clear();

    size_t totalSplats = 0;
    for(auto* tile : m_visibleTiles)
    {
        if(tile->isLoaded)
        {
            totalSplats += tile->splatSet.size();
        }
    }

    if(totalSplats == 0)
    {
        m_visibleTilesDirty = false;
        return;
    }

    // Reserve space
    output.positions.resize(totalSplats * 3);
    output.f_dc.resize(totalSplats * 3);
    output.opacity.resize(totalSplats);
    output.scale.resize(totalSplats * 3);
    output.rotation.resize(totalSplats * 4);

    // Merge tiles
    size_t splatIndex = 0;
    for(auto* tile : m_visibleTiles)
    {
        if(!tile->isLoaded)
            continue;

        const auto& src = tile->splatSet;
        for(size_t i = 0; i < src.size(); i++)
        {
            output.positions[splatIndex * 3 + 0] = src.positions[i * 3 + 0];
            output.positions[splatIndex * 3 + 1] = src.positions[i * 3 + 1];
            output.positions[splatIndex * 3 + 2] = src.positions[i * 3 + 2];

            output.f_dc[splatIndex * 3 + 0] = src.f_dc[i * 3 + 0];
            output.f_dc[splatIndex * 3 + 1] = src.f_dc[i * 3 + 1];
            output.f_dc[splatIndex * 3 + 2] = src.f_dc[i * 3 + 2];

            output.opacity[splatIndex] = src.opacity[i];

            output.scale[splatIndex * 3 + 0] = src.scale[i * 3 + 0];
            output.scale[splatIndex * 3 + 1] = src.scale[i * 3 + 1];
            output.scale[splatIndex * 3 + 2] = src.scale[i * 3 + 2];

            output.rotation[splatIndex * 4 + 0] = src.rotation[i * 4 + 0];
            output.rotation[splatIndex * 4 + 1] = src.rotation[i * 4 + 1];
            output.rotation[splatIndex * 4 + 2] = src.rotation[i * 4 + 2];
            output.rotation[splatIndex * 4 + 3] = src.rotation[i * 4 + 3];

            splatIndex++;
        }
    }
    
    m_visibleTilesDirty = false;
}

void LccTileManager::invalidateAll()
{
    std::lock_guard<std::mutex> lock(m_queueMutex);
    for(auto& tile : m_allTiles)
    {
        if(tile.isLoaded)
        {
            unloadTile(tile);
        }
        tile.isDirty = true;
    }
    m_loadQueue.clear();
}

}  // namespace vk_viewer
