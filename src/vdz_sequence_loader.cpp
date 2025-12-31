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

#include "vdz_sequence_loader.h"
#include <nvutils/logger.hpp>
#include <algorithm>
#include <cstring>
#include <zlib.h>

namespace vk_gaussian_splatting {

static constexpr size_t VDZ_HEADER_SIZE = 32;

VDZSequenceLoader::~VDZSequenceLoader()
{
    close();
}

bool VDZSequenceLoader::open(const std::filesystem::path& filepath)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_file.is_open()) {
        m_file.close();
    }

    if (!std::filesystem::exists(filepath)) {
        LOGE("VDZ sequence file not found: %s\n", filepath.string().c_str());
        return false;
    }

    m_filepath = filepath;
    m_file.open(filepath, std::ios::binary);
    
    if (!m_file) {
        LOGE("Failed to open VDZ sequence file: %s\n", filepath.string().c_str());
        return false;
    }

    if (!buildIndex()) {
        LOGE("Failed to build frame index for VDZ sequence: %s\n", filepath.string().c_str());
        m_file.close();
        return false;
    }

    LOGI("VDZ sequence loaded: %s (%zu frames, %ux%u, duration=%u ms)\n",
         filepath.string().c_str(), m_frameIndex.size(), m_width, m_height, getDurationMs());

    return true;
}

void VDZSequenceLoader::close()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (m_file.is_open()) {
        m_file.close();
    }
    m_frameIndex.clear();
    m_width = 0;
    m_height = 0;
}

bool VDZSequenceLoader::buildIndex()
{
    m_frameIndex.clear();

    m_file.seekg(0, std::ios::end);
    size_t fileSize = m_file.tellg();
    m_file.seekg(0, std::ios::beg);

    if (fileSize < VDZ_HEADER_SIZE) {
        LOGE("VDZ file too small\n");
        return false;
    }

    size_t offset = 0;
    DepthHeader header;

    while (offset + VDZ_HEADER_SIZE <= fileSize) {
        m_file.seekg(offset);
        m_file.read(reinterpret_cast<char*>(&header), VDZ_HEADER_SIZE);

        if (!m_file) {
            break;
        }

        bool isCompressed = false;
        if (std::memcmp(header.magic, "VDZ2", 4) == 0) {
            isCompressed = true;
        } else if (std::memcmp(header.magic, "VDZ1", 4) != 0) {
            LOGW("Invalid magic at offset %zu, stopping index build\n", offset);
            break;
        }

        if (header.version != 1 || header.dataType != 1) {
            LOGW("Unsupported version/dataType at offset %zu\n", offset);
            break;
        }

        size_t dataSize;
        size_t expectedDecompressedSize = header.width * header.height * sizeof(uint16_t);
        
        if (isCompressed) {
            // For compressed data, we need to actually decompress to find the end
            // Use zlib streaming to find where the compressed data ends
            size_t compressedSize = 0;
            size_t dataStart = offset + VDZ_HEADER_SIZE;
            size_t maxRead = fileSize - dataStart;
            
            // Read compressed data in chunks and try to decompress
            std::vector<uint8_t> compressedBuffer;
            compressedBuffer.reserve(std::min(maxRead, expectedDecompressedSize * 2));
            
            m_file.seekg(dataStart);
            
            // Read in chunks until we successfully decompress
            const size_t chunkSize = 8192;
            std::vector<uint8_t> decompressed(expectedDecompressedSize);
            bool found = false;
            
            while (compressedBuffer.size() < maxRead && !found) {
                size_t toRead = std::min(chunkSize, maxRead - compressedBuffer.size());
                size_t oldSize = compressedBuffer.size();
                compressedBuffer.resize(oldSize + toRead);
                m_file.read(reinterpret_cast<char*>(compressedBuffer.data() + oldSize), toRead);
                size_t bytesRead = m_file.gcount();
                compressedBuffer.resize(oldSize + bytesRead);
                
                if (bytesRead == 0) break;
                
                // Try zlib format (with header - Python's zlib.compress uses this)
                z_stream stream{};
                stream.next_in = compressedBuffer.data();
                stream.avail_in = static_cast<uInt>(compressedBuffer.size());
                stream.next_out = decompressed.data();
                stream.avail_out = static_cast<uInt>(decompressed.size());
                
                if (inflateInit(&stream) == Z_OK) {
                    int ret = inflate(&stream, Z_FINISH);
                    if (ret == Z_STREAM_END) {
                        // Success! Calculate how many compressed bytes were consumed
                        compressedSize = compressedBuffer.size() - stream.avail_in;
                        found = true;
                    }
                    inflateEnd(&stream);
                }
                
                // If zlib format failed, try raw deflate
                if (!found) {
                    stream = z_stream{};
                    stream.next_in = compressedBuffer.data();
                    stream.avail_in = static_cast<uInt>(compressedBuffer.size());
                    stream.next_out = decompressed.data();
                    stream.avail_out = static_cast<uInt>(decompressed.size());
                    
                    if (inflateInit2(&stream, -MAX_WBITS) == Z_OK) {
                        int ret = inflate(&stream, Z_FINISH);
                        if (ret == Z_STREAM_END) {
                            compressedSize = compressedBuffer.size() - stream.avail_in;
                            found = true;
                        }
                        inflateEnd(&stream);
                    }
                }
            }
            
            if (!found) {
                LOGW("Could not decompress frame at offset %zu, skipping rest of file\n", offset);
                break;
            }
            
            dataSize = compressedSize;
        } else {
            dataSize = expectedDecompressedSize;
        }

        VDZFrameIndex idx;
        idx.timestampMs = header.timestampMs;
        idx.fileOffset = offset;
        idx.frameSize = VDZ_HEADER_SIZE + dataSize;

        m_frameIndex.push_back(idx);

        if (m_width == 0) {
            m_width = header.width;
            m_height = header.height;
        }

        offset += idx.frameSize;
    }

    return !m_frameIndex.empty();
}

bool VDZSequenceLoader::getFrame(size_t index, DepthFrame& outFrame)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    if (index >= m_frameIndex.size()) {
        return false;
    }

    const auto& idx = m_frameIndex[index];
    return readFrameAt(idx.fileOffset, idx.frameSize, outFrame);
}

bool VDZSequenceLoader::getFrameByTimestamp(uint32_t timestampMs, DepthFrame& outFrame)
{
    size_t index = getFrameIndexForTimestamp(timestampMs);
    return getFrame(index, outFrame);
}

size_t VDZSequenceLoader::getFrameIndexForTimestamp(uint32_t timestampMs) const
{
    if (m_frameIndex.empty()) {
        return 0;
    }

    auto it = std::lower_bound(m_frameIndex.begin(), m_frameIndex.end(), timestampMs,
        [](const VDZFrameIndex& a, uint32_t ts) { return a.timestampMs < ts; });

    if (it == m_frameIndex.end()) {
        return m_frameIndex.size() - 1;
    }
    
    if (it == m_frameIndex.begin()) {
        return 0;
    }

    auto prev = std::prev(it);
    if ((timestampMs - prev->timestampMs) <= (it->timestampMs - timestampMs)) {
        return std::distance(m_frameIndex.begin(), prev);
    }
    return std::distance(m_frameIndex.begin(), it);
}

uint32_t VDZSequenceLoader::getTimestamp(size_t index) const
{
    if (index >= m_frameIndex.size()) {
        return 0;
    }
    return m_frameIndex[index].timestampMs;
}

uint32_t VDZSequenceLoader::getDurationMs() const
{
    if (m_frameIndex.empty()) {
        return 0;
    }
    return m_frameIndex.back().timestampMs;
}

bool VDZSequenceLoader::readFrameAt(size_t offset, size_t size, DepthFrame& outFrame)
{
    if (!m_file.is_open()) {
        return false;
    }

    std::vector<uint8_t> buffer(size);
    m_file.seekg(offset);
    m_file.read(reinterpret_cast<char*>(buffer.data()), size);

    if (m_file.gcount() != static_cast<std::streamsize>(size)) {
        LOGE("Failed to read frame at offset %zu\n", offset);
        return false;
    }

    return parseDepthFrame(buffer, outFrame);
}

} // namespace vk_gaussian_splatting
