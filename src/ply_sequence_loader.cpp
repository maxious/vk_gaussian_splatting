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

#include "ply_sequence_loader.h"
#include "splat_loader_fast.h"
#include <algorithm>
#include <sstream>

#include <nvutils/logger.hpp>

namespace vk_gaussian_splatting {

PlySequenceLoader::~PlySequenceLoader() {
    close();
}

bool PlySequenceLoader::open(const std::filesystem::path& dirPath, float frameRate) {
    close();
    
    if (!std::filesystem::exists(dirPath) || !std::filesystem::is_directory(dirPath)) {
        LOGE("PLY sequence: Invalid directory path: %s", dirPath.string().c_str());
        return false;
    }
    
    m_dirPath = dirPath;
    m_frameRate = frameRate;
    m_frameDurationMs = static_cast<uint32_t>(1000.0f / frameRate);
    
    // Scan directory for PLY files
    if (!scanDirectory(dirPath)) {
        LOGE("PLY sequence: Failed to scan directory: %s", dirPath.string().c_str());
        return false;
    }
    
    // Check for audio file
    std::vector<std::string> audioExtensions = {".wav", ".mp3", ".aac", ".ogg", ".flac"};
    for (const auto& ext : audioExtensions) {
        std::filesystem::path audioPath = dirPath / ("audio" + ext);
        if (std::filesystem::exists(audioPath)) {
            m_audioPath = audioPath;
            m_hasAudio = true;
            LOGI("PLY sequence: Found audio file: %s", audioPath.string().c_str());
            break;
        }
    }
    
    if (m_frames.empty()) {
        LOGE("PLY sequence: No valid PLY files found in directory: %s", dirPath.string().c_str());
        return false;
    }
    
    m_isOpen = true;
    LOGI("PLY sequence: Loaded %zu frames (%.2f fps, duration: %.2fs)", 
           m_frames.size(), m_frameRate, getDurationMs() / 1000.0f);
    
    return true;
}

void PlySequenceLoader::close() {
    std::lock_guard<std::mutex> lock(m_mutex);
    clearCache();
    m_frames.clear();
    m_audioPath.clear();
    m_dirPath.clear();
    m_isOpen = false;
    m_hasAudio = false;
}

bool PlySequenceLoader::scanDirectory(const std::filesystem::path& dirPath) {
    m_frames.clear();
    
    // Simple pattern matching for frame_XXXXX.ply format
    for (const auto& entry : std::filesystem::directory_iterator(dirPath)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".ply") {
            continue;
        }
        
        std::string filename = entry.path().filename().string();
        
        // Look for pattern: frame_XXXXXXXX.ply
        if (filename.length() >= 6 && filename.substr(0, 6) == "frame_" && 
            filename.substr(filename.length() - 4) == ".ply") {
            
            PlyFrameInfo frameInfo;
            frameInfo.filepath = entry.path();
            frameInfo.timestampMs = 0; // Will be set later
            frameInfo.loaded = false;
            
            m_frames.push_back(frameInfo);
        }
    }
    
    // Sort by filename (which should be frame_000000.ply, frame_000001.ply, etc.)
    std::sort(m_frames.begin(), m_frames.end(), 
              [](const PlyFrameInfo& a, const PlyFrameInfo& b) {
                  return a.filepath.filename().string() < b.filepath.filename().string();
              });
    
    // Re-index and set timestamps
    for (size_t i = 0; i < m_frames.size(); ++i) {
        m_frames[i].frameIndex = i;
        m_frames[i].timestampMs = static_cast<uint32_t>(i * m_frameDurationMs);
    }
    
    return !m_frames.empty();
}

bool PlySequenceLoader::getFrame(size_t index, SplatSet& outFrame) {
    if (!m_isOpen || index >= m_frames.size()) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // Check cache first
    auto it = m_frameCache.find(index);
    if (it != m_frameCache.end()) {
        outFrame = it->second;
        return true;
    }
    
    // Load from disk
    const PlyFrameInfo& frameInfo = m_frames[index];
    if (!loadFrame(frameInfo, outFrame)) {
        LOGE("PLY sequence: Failed to load frame %zu: %s", 
               index, frameInfo.filepath.string().c_str());
        return false;
    }
    
    // Add to cache (with LRU eviction if needed)
    if (m_frameCache.size() >= m_maxCacheSize) {
        // Simple LRU: remove oldest entry
        size_t oldestIndex = index;
        if (!m_frameCache.empty()) {
            oldestIndex = m_frameCache.begin()->first;
        }
        m_frameCache.erase(oldestIndex);
    }
    
    m_frameCache[index] = outFrame;
    return true;
}

bool PlySequenceLoader::getFrameByTimestamp(uint32_t timestampMs, SplatSet& outFrame) {
    size_t index = getFrameIndexForTimestamp(timestampMs);
    return getFrame(index, outFrame);
}

size_t PlySequenceLoader::getFrameIndexForTimestamp(uint32_t timestampMs) const {
    if (m_frames.empty()) {
        return 0;
    }
    
    // Binary search for closest frame
    size_t left = 0;
    size_t right = m_frames.size() - 1;
    
    while (left < right) {
        size_t mid = left + (right - left) / 2;
        if (m_frames[mid].timestampMs < timestampMs) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    // Check if we need to adjust to previous frame
    if (left > 0 && m_frames[left].timestampMs > timestampMs) {
        left--;
    }
    
    return std::min(left, m_frames.size() - 1);
}

uint32_t PlySequenceLoader::getTimestamp(size_t index) const {
    if (index >= m_frames.size()) {
        return 0;
    }
    return m_frames[index].timestampMs;
}

uint32_t PlySequenceLoader::getDurationMs() const {
    if (m_frames.empty()) {
        return 0;
    }
    return static_cast<uint32_t>(m_frames.size() * m_frameDurationMs);
}

bool PlySequenceLoader::loadFrame(const PlyFrameInfo& frameInfo, SplatSet& outFrame) {
    return SplatLoaderFast::load(frameInfo.filepath, outFrame);
}

void PlySequenceLoader::clearCache() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_frameCache.clear();
}

} // namespace vk_gaussian_splatting