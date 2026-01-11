/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under Apache License, Version 2.0 (the "License");
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

#include <filesystem>
#include <vector>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <string>
#include <algorithm>
#include "splat_set.h"

namespace vk_viewer {

/**
 * @brief Metadata for a frame in a PLY sequence
 */
struct PlyFrameInfo {
    std::filesystem::path filepath;
    uint32_t timestampMs;
    size_t frameIndex;
    bool loaded;
    
    PlyFrameInfo() : timestampMs(0), frameIndex(0), loaded(false) {}
};

/**
 * @brief Loads a sequence of PLY files as an animation
 * 
 * This loader scans a directory for numbered PLY files following the pattern:
 * frame_000000.ply, frame_000001.ply, etc.
 * 
 * It supports optional audio files (audio.wav, audio.mp3) for synchronized playback.
 * Frames can be accessed by index or timestamp for smooth animation.
 */
class PlySequenceLoader {
public:
    PlySequenceLoader() = default;
    ~PlySequenceLoader();

    /**
     * @brief Open a directory containing PLY sequence and build frame index
     * @param dirPath Path to directory containing frame_*.ply files
     * @param frameRate Expected frame rate (default: 30fps)
     * @return true if successfully opened and indexed
     */
    bool open(const std::filesystem::path& dirPath, float frameRate = 30.0f);

    /**
     * @brief Close sequence and release resources
     */
    void close();

    /**
     * @brief Get frame count
     */
    size_t getFrameCount() const { return m_frames.size(); }

    /**
     * @brief Get frame by index
     * @param index Frame index (0-based)
     * @param outFrame Output SplatSet
     * @return true if successful
     */
    bool getFrame(size_t index, SplatSet& outFrame);

    /**
     * @brief Get frame by timestamp (finds closest frame)
     * @param timestampMs Desired timestamp in milliseconds
     * @param outFrame Output SplatSet
     * @return true if successful
     */
    bool getFrameByTimestamp(uint32_t timestampMs, SplatSet& outFrame);

    /**
     * @brief Get frame index for a given timestamp
     * @param timestampMs Desired timestamp in milliseconds
     * @return Frame index of closest matching frame
     */
    size_t getFrameIndexForTimestamp(uint32_t timestampMs) const;

    /**
     * @brief Get timestamp for a given frame index
     * @param index Frame index
     * @return Timestamp in milliseconds
     */
    uint32_t getTimestamp(size_t index) const;

    /**
     * @brief Get frame duration in milliseconds
     */
    uint32_t getFrameDurationMs() const { return m_frameDurationMs; }

    /**
     * @brief Get total duration in milliseconds
     */
    uint32_t getDurationMs() const;

    /**
     * @brief Get frame rate
     */
    float getFrameRate() const { return m_frameRate; }

    /**
     * @brief Check if a sequence is open
     */
    bool isOpen() const { return m_isOpen; }

    /**
     * @brief Check if audio file is available
     */
    bool hasAudio() const { return m_hasAudio; }

    /**
     * @brief Get audio file path
     */
    const std::filesystem::path& getAudioPath() const { return m_audioPath; }

    /**
     * @brief Get directory path
     */
    const std::filesystem::path& getDirectoryPath() const { return m_dirPath; }

    /**
     * @brief Clear frame cache
     */
    void clearCache();

    /**
     * @brief Set cache size (number of frames to keep in memory)
     */
    void setCacheSize(size_t size) { m_maxCacheSize = size; }

    /**
     * @brief Get current cache size
     */
    size_t getCacheSize() const { return m_maxCacheSize; }

    /**
     * @brief Get current cache usage (number of cached frames)
     */
    size_t getCacheUsage() const { return m_frameCache.size(); }

    /**
     * @brief Set target memory usage for adaptive cache sizing
     * @param targetMemoryMB Target memory usage in megabytes
     */
    void setTargetMemoryMB(size_t targetMemoryMB) { m_targetMemoryMB = targetMemoryMB; }

private:
    bool scanDirectory(const std::filesystem::path& dirPath);
    bool loadFrame(const PlyFrameInfo& frameInfo, SplatSet& outFrame);
    void updateAdaptiveCacheSize(const SplatSet& frame);
    void evictOldFrames(size_t currentFrameIndex);
    
    std::filesystem::path                    m_dirPath;
    std::vector<PlyFrameInfo>                m_frames;
    std::filesystem::path                    m_audioPath;
    std::mutex                             m_mutex;
    
    // Sliding window cache for forward-only playback
    std::unordered_map<size_t, SplatSet>     m_frameCache;
    size_t                                 m_maxCacheSize = 10;
    size_t                                 m_targetMemoryMB = 512;
    size_t                                 m_estimatedFrameSize = 0;
    
    float                                  m_frameRate = 30.0f;
    uint32_t                               m_frameDurationMs = 33;
    bool                                   m_isOpen = false;
    bool                                   m_hasAudio = false;
};

}  // namespace vk_viewer