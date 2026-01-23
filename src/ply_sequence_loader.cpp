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

namespace vk_viewer {

PlySequenceLoader::~PlySequenceLoader() {
    stopPrefetch();
    close();
}

bool PlySequenceLoader::open(const std::filesystem::path& dirPath, float frameRate) {
    close();
    
    if (!std::filesystem::exists(dirPath) || !std::filesystem::is_directory(dirPath)) {
        LOGE("PLY sequence: Invalid directory path: %s", dirPath.string().c_str());
        return false;
    }
    
    LOGI("PLY sequence: Processing directory: %s", dirPath.string().c_str());
    
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
    stopPrefetch();
    
    std::lock_guard<std::mutex> lock(m_mutex);
    m_frameCache.clear();
    m_lruOrder.clear();
    m_lruMap.clear();
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
            
            LOGD("PLY sequence: Found frame file: %s", filename.c_str());
            
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
    m_lruOrder.clear();
    m_lruMap.clear();
}

void PlySequenceLoader::touchCacheEntry(size_t index) {
    // Move entry to front of LRU list (most recently used)
    auto it = m_lruMap.find(index);
    if (it != m_lruMap.end()) {
        m_lruOrder.erase(it->second);
        m_lruOrder.push_front(index);
        it->second = m_lruOrder.begin();
    }
}

void PlySequenceLoader::evictLRU() {
    // Evict least recently used entries until under max cache size
    while (m_frameCache.size() >= m_maxCacheSize && !m_lruOrder.empty()) {
        size_t evictIdx = m_lruOrder.back();
        m_lruOrder.pop_back();
        m_lruMap.erase(evictIdx);
        m_frameCache.erase(evictIdx);
        LOGD("PLY sequence: Evicted frame %zu from cache", evictIdx);
    }
}

void PlySequenceLoader::getSurroundingKeyframes(uint32_t timestampMs, size_t& lowerIdx, size_t& upperIdx, float& t) const {
    if (m_frames.empty()) {
        lowerIdx = upperIdx = 0;
        t = 0.0f;
        return;
    }
    
    // Find surrounding keyframes
    lowerIdx = 0;
    for (size_t i = 0; i < m_frames.size(); ++i) {
        if (m_frames[i].timestampMs <= timestampMs) {
            lowerIdx = i;
        } else {
            break;
        }
    }
    
    upperIdx = std::min(lowerIdx + 1, m_frames.size() - 1);
    
    // Compute blend factor t in [0, 1]
    if (lowerIdx == upperIdx) {
        t = 0.0f;
    } else {
        uint32_t t0 = m_frames[lowerIdx].timestampMs;
        uint32_t t1 = m_frames[upperIdx].timestampMs;
        t = static_cast<float>(timestampMs - t0) / static_cast<float>(t1 - t0);
        t = std::max(0.0f, std::min(1.0f, t));
    }
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
        touchCacheEntry(index);
        return true;
    }
    
    // Load from disk
    const PlyFrameInfo& frameInfo = m_frames[index];
    if (!loadFrame(frameInfo, outFrame)) {
        LOGE("PLY sequence: Failed to load frame %zu: %s", 
               index, frameInfo.filepath.string().c_str());
        return false;
    }
    
    // Estimate frame size and adjust cache if needed
    updateAdaptiveCacheSize(outFrame);
    
    // Evict LRU entries if cache is full
    evictLRU();
    
    // Add new frame to cache and LRU tracking
    m_frameCache[index] = outFrame;
    m_lruOrder.push_front(index);
    m_lruMap[index] = m_lruOrder.begin();
    
    // Trigger async prefetch of upcoming frames
    prefetchFramesAsync(index);
    
    return true;
}

bool PlySequenceLoader::getInterpolatedFrame(uint32_t timestampMs, SplatSet& outFrame) {
    if (!m_isOpen || m_frames.empty()) {
        return false;
    }
    
    // Find surrounding keyframes
    size_t lowerIdx, upperIdx;
    float t;
    getSurroundingKeyframes(timestampMs, lowerIdx, upperIdx, t);
    
    // If interpolation disabled or same frame, return nearest keyframe
    if (!m_interpolationEnabled || lowerIdx == upperIdx || t < 0.001f) {
        return getFrame(lowerIdx, outFrame);
    }
    if (t > 0.999f) {
        return getFrame(upperIdx, outFrame);
    }
    
    // Load both keyframes
    SplatSet frame0, frame1;
    if (!getFrame(lowerIdx, frame0) || !getFrame(upperIdx, frame1)) {
        return false;
    }
    
    // Interpolate between frames
    if (!interpolateSplatSet(outFrame, frame0, frame1, t)) {
        LOGW("PLY sequence: Interpolation failed (splat count mismatch: %zu vs %zu)", 
             frame0.size(), frame1.size());
        return getFrame(lowerIdx, outFrame);  // Fallback to nearest
    }
    
    return true;
}

bool PlySequenceLoader::getInterpolatedFrameNormalized(float normalizedTime, SplatSet& outFrame) {
    if (!m_isOpen || m_frames.empty()) {
        return false;
    }
    
    normalizedTime = std::max(0.0f, std::min(1.0f, normalizedTime));
    uint32_t timestampMs = static_cast<uint32_t>(normalizedTime * getDurationMs());
    return getInterpolatedFrame(timestampMs, outFrame);
}

void PlySequenceLoader::updateAdaptiveCacheSize(const SplatSet& frame) {
    // Estimate frame memory: each splat is approximately 256 bytes
    // (positions + f_dc + f_rest + opacity + scale + rotation)
    constexpr size_t BYTES_PER_SPLAT = 256;

    size_t splatCount = frame.positions.size() / 3;
    m_estimatedFrameSize = splatCount * BYTES_PER_SPLAT;

    // Adaptive cache size based on target memory
    if (m_targetMemoryMB > 0 && m_estimatedFrameSize > 0) {
        size_t targetBytes = m_targetMemoryMB * 1024 * 1024;
        size_t newCacheSize = targetBytes / m_estimatedFrameSize;
        newCacheSize = std::max(static_cast<size_t>(2), newCacheSize);
        newCacheSize = std::min(static_cast<size_t>(50), newCacheSize);

        if (newCacheSize > m_maxCacheSize * 2 || newCacheSize < m_maxCacheSize / 2) {
            m_maxCacheSize = newCacheSize;
            LOGD("PLY sequence: Adaptive cache size set to %zu frames (~%.1f MB per frame)",
                 m_maxCacheSize, m_estimatedFrameSize / (1024.0 * 1024.0));
        }
    }
}

void PlySequenceLoader::prefetchFramesAsync(size_t currentFrame) {
    if (m_prefetchCount == 0) {
        return;
    }
    
    // Queue upcoming frames for prefetch
    std::vector<size_t> framesToPrefetch;
    for (size_t i = 1; i <= m_prefetchCount; ++i) {
        size_t nextFrame = currentFrame + i;
        if (nextFrame < m_frames.size()) {
            // Only prefetch if not already cached
            if (m_frameCache.find(nextFrame) == m_frameCache.end()) {
                framesToPrefetch.push_back(nextFrame);
            }
        }
    }
    
    if (framesToPrefetch.empty()) {
        return;
    }
    
    // Start prefetch thread if not running
    {
        std::lock_guard<std::mutex> lock(m_prefetchMutex);
        m_prefetchQueue.insert(m_prefetchQueue.end(), framesToPrefetch.begin(), framesToPrefetch.end());
        
        if (!m_prefetching.load()) {
            m_stopPrefetch.store(false);
            m_prefetching.store(true);
            
            // Stop existing thread if any
            if (m_prefetchThread.joinable()) {
                m_prefetchThread.join();
            }
            
            m_prefetchThread = std::thread(&PlySequenceLoader::prefetchWorker, this);
        }
    }
    m_prefetchCV.notify_one();
}

void PlySequenceLoader::prefetchWorker() {
    LOGD("PLY sequence: Prefetch worker started");
    
    while (!m_stopPrefetch.load()) {
        std::vector<size_t> framesToLoad;
        
        {
            std::unique_lock<std::mutex> lock(m_prefetchMutex);
            m_prefetchCV.wait_for(lock, std::chrono::milliseconds(100), [this] {
                return !m_prefetchQueue.empty() || m_stopPrefetch.load();
            });
            
            if (m_stopPrefetch.load()) {
                break;
            }
            
            // Take up to 2 frames at a time
            size_t count = std::min(m_prefetchQueue.size(), static_cast<size_t>(2));
            for (size_t i = 0; i < count; ++i) {
                framesToLoad.push_back(m_prefetchQueue.front());
                m_prefetchQueue.erase(m_prefetchQueue.begin());
            }
        }
        
        // Load frames outside the lock
        for (size_t frameIdx : framesToLoad) {
            if (m_stopPrefetch.load()) {
                break;
            }
            
            // Check if already cached (might have been loaded by main thread)
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (m_frameCache.find(frameIdx) != m_frameCache.end()) {
                    continue;
                }
            }
            
            // Load frame
            if (frameIdx < m_frames.size()) {
                SplatSet frame;
                if (loadFrame(m_frames[frameIdx], frame)) {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    
                    // Double-check it wasn't loaded while we were loading
                    if (m_frameCache.find(frameIdx) == m_frameCache.end()) {
                        evictLRU();
                        m_frameCache[frameIdx] = std::move(frame);
                        m_lruOrder.push_front(frameIdx);
                        m_lruMap[frameIdx] = m_lruOrder.begin();
                        LOGD("PLY sequence: Prefetched frame %zu", frameIdx);
                    }
                }
            }
        }
        
        // Check if queue is empty, if so we can stop
        {
            std::lock_guard<std::mutex> lock(m_prefetchMutex);
            if (m_prefetchQueue.empty()) {
                break;
            }
        }
    }
    
    m_prefetching.store(false);
    LOGD("PLY sequence: Prefetch worker stopped");
}

void PlySequenceLoader::stopPrefetch() {
    m_stopPrefetch.store(true);
    m_prefetchCV.notify_all();
    
    if (m_prefetchThread.joinable()) {
        m_prefetchThread.join();
    }
    
    std::lock_guard<std::mutex> lock(m_prefetchMutex);
    m_prefetchQueue.clear();
}

} // namespace vk_viewer