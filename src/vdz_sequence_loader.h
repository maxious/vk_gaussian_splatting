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

#pragma once

#include <filesystem>
#include <vector>
#include <cstdint>
#include <fstream>
#include <mutex>
#include "depth_parser.h"

namespace vk_gaussian_splatting {

struct VDZFrameIndex {
    uint32_t timestampMs;
    size_t   fileOffset;
    size_t   frameSize;
};

/**
 * @brief Loads a VDZ sequence file containing multiple depth frames
 * 
 * VDZ sequence files contain concatenated frames, each with a 32-byte header
 * followed by depth data (compressed or uncompressed).
 * 
 * This loader builds an index of all frames on open, allowing random access
 * by timestamp or frame index for synchronized video+depth playback.
 */
class VDZSequenceLoader {
public:
    VDZSequenceLoader() = default;
    ~VDZSequenceLoader();

    /**
     * @brief Open a VDZ sequence file and build frame index
     * @param filepath Path to the .vdz file
     * @return true if successfully opened and indexed
     */
    bool open(const std::filesystem::path& filepath);

    /**
     * @brief Close the file and release resources
     */
    void close();

    /**
     * @brief Get frame count
     */
    size_t getFrameCount() const { return m_frameIndex.size(); }

    /**
     * @brief Get frame by index
     * @param index Frame index (0-based)
     * @param outFrame Output depth frame
     * @return true if successful
     */
    bool getFrame(size_t index, DepthFrame& outFrame);

    /**
     * @brief Get frame by timestamp (finds closest frame)
     * @param timestampMs Desired timestamp in milliseconds
     * @param outFrame Output depth frame
     * @return true if successful
     */
    bool getFrameByTimestamp(uint32_t timestampMs, DepthFrame& outFrame);

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
     * @brief Get depth frame dimensions
     */
    uint32_t getWidth() const { return m_width; }
    uint32_t getHeight() const { return m_height; }

    /**
     * @brief Check if a file is open
     */
    bool isOpen() const { return m_file.is_open(); }

    /**
     * @brief Get total duration in milliseconds
     */
    uint32_t getDurationMs() const;

private:
    bool buildIndex();
    bool readFrameAt(size_t offset, size_t size, DepthFrame& outFrame);

    std::ifstream             m_file;
    std::filesystem::path     m_filepath;
    std::vector<VDZFrameIndex> m_frameIndex;
    std::mutex                m_mutex;
    
    uint32_t m_width = 0;
    uint32_t m_height = 0;
};

} // namespace vk_gaussian_splatting
