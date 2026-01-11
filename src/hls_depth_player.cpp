/*
 * HLS Depth Player - Standards-compliant HLS stream player with RGB + Depth unpacking
 *
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hls_depth_player.h"
#include <nvutils/logger.hpp>
#include <fstream>
#include <tinygltf/json.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
}

namespace vk_viewer {

HlsDepthPlayer::HlsDepthPlayer() : m_decoder(std::make_unique<VideoDecoder>())
{
}

HlsDepthPlayer::~HlsDepthPlayer()
{
    stop();
}

bool HlsDepthPlayer::open(const std::filesystem::path& playlistPath)
{
    // First, try to load metadata from side-by-side JSON
    std::filesystem::path metadataPath = playlistPath;
    if (metadataPath.extension() == ".m3u8") {
        metadataPath = metadataPath.parent_path() / "metadata.json";
    }

    if (std::filesystem::exists(metadataPath)) {
        if (!loadMetadata(metadataPath)) {
            LOGW("Failed to load metadata from %s, using defaults\n", metadataPath.string().c_str());
        }
    }

    // Open the video (HLS playlist or regular video)
    if (!m_decoder->open(playlistPath)) {
        LOGE("Failed to open video: %s\n", playlistPath.string().c_str());
        return false;
    }

    int width, height;
    m_decoder->getDimensions(width, height);

    // Calculate dimensions based on metadata
    if (m_metadata.rgbWidth == 0) {
        // Default: assume side-by-side (half width each)
        m_metadata.rgbWidth = width / 2;
        m_metadata.depthWidth = width / 2;
    }
    m_metadata.height = height;

    // Calculate scale if not provided
    if (m_metadata.scale <= 0.0f) {
        m_metadata.scale = (m_metadata.zMax - m_metadata.zMin) / 65535.0f;
        if (m_metadata.scale <= 0.0f) {
            m_metadata.scale = 1.0f;
        }
    }

    m_metadata.duration = m_decoder->getDuration();
    m_metadata.fps = static_cast<float>(m_decoder->getFrameRate());

    m_initialized = true;

    LOGI("HLS Depth Player opened: %dx%d (RGB: %dx%d, Depth: %dx%d), z_min: %.2f, z_max: %.2f\n",
         width, height, m_metadata.rgbWidth, m_metadata.height,
         m_metadata.depthWidth, m_metadata.height, m_metadata.zMin, m_metadata.zMax);

    return true;
}

bool HlsDepthPlayer::loadMetadata(const std::filesystem::path& metadataPath)
{
    try {
        std::ifstream file(metadataPath);
        if (!file.is_open()) {
            LOGE("Failed to open metadata file: %s\n", metadataPath.string().c_str());
            return false;
        }

        auto json = nlohmann::json::parse(file);

        m_metadata.rgbWidth = json.value("rgb_width", 0);
        m_metadata.depthWidth = json.value("depth_width", 0);
        m_metadata.height = json.value("height", 0);
        m_metadata.fps = json.value("fps", 30.0f);
        m_metadata.duration = json.value("duration_s", 0.0);
        m_metadata.zMin = json.value("z_min", 0.0f);
        m_metadata.zMax = json.value("z_max", 10.0f);
        m_metadata.scale = json.value("scale", 0.0f);

        // Recalculate scale if needed
        if (m_metadata.scale <= 0.0f && m_metadata.zMax > m_metadata.zMin) {
            m_metadata.scale = (m_metadata.zMax - m_metadata.zMin) / 65535.0f;
        }

        LOGI("Loaded HLS metadata: z_min=%.2f, z_max=%.2f, scale=%.6f\n",
             m_metadata.zMin, m_metadata.zMax, m_metadata.scale);

        return true;
    } catch (const std::exception& e) {
        LOGE("Failed to parse metadata: %s\n", e.what());
        return false;
    }
}

void HlsDepthPlayer::start()
{
    if (m_decoder) {
        m_decoder->startDecoding();
    }
}

void HlsDepthPlayer::stop()
{
    if (m_decoder) {
        m_decoder->stopDecoding();
    }

    std::lock_guard<std::mutex> lock(m_queueMutex);
    m_frameQueue.clear();
}

bool HlsDepthPlayer::getNextFrame(HlsDecodedFrame& frame)
{
    if (!m_initialized || !m_decoder) {
        return false;
    }

    DecodedFrame decodedFrame;
    if (!m_decoder->getNextFrame(decodedFrame)) {
        return false;
    }

    frame = separateFrame(decodedFrame);
    return true;
}

bool HlsDepthPlayer::seekToTime(double timestamp)
{
    if (m_decoder) {
        return m_decoder->seekToTime(timestamp);
    }
    return false;
}

double HlsDepthPlayer::getCurrentTime() const
{
    // Return the timestamp of the most recent frame in queue
    std::lock_guard<std::mutex> lock(m_queueMutex);
    if (!m_frameQueue.empty()) {
        return m_frameQueue.front().timestamp;
    }
    return 0.0;
}

void HlsDepthPlayer::pause()
{
    if (m_decoder) {
        m_decoder->pause();
    }
}

void HlsDepthPlayer::resume()
{
    if (m_decoder) {
        m_decoder->resume();
    }
}

std::vector<float> HlsDepthPlayer::unpackDepth(const uint8_t* rgbaData, int width, int height)
{
    std::vector<float> depthData(width * height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 4;  // RGBA = 4 bytes per pixel

            // Unpack RGB to 16-bit value
            // R = high byte, G = low byte, B = 128 (unused)
            uint16_t uint16Val = (static_cast<uint16_t>(rgbaData[idx]) << 8) |
                                 static_cast<uint16_t>(rgbaData[idx + 1]);

            // Convert to float depth
            float depth = static_cast<float>(uint16Val) * m_metadata.scale + m_metadata.zMin;
            depthData[y * width + x] = depth;
        }
    }

    return depthData;
}

HlsDecodedFrame HlsDepthPlayer::separateFrame(const DecodedFrame& frame)
{
    HlsDecodedFrame result;
    result.width = m_metadata.rgbWidth;
    result.height = m_metadata.height;
    result.timestamp = frame.timestamp;

    const int totalWidth = frame.width;
    const int totalHeight = frame.height;

    if (totalWidth < m_metadata.rgbWidth + m_metadata.depthWidth) {
        LOGE("Frame width %d too small for expected layout (RGB: %d, Depth: %d)\n",
             totalWidth, m_metadata.rgbWidth, m_metadata.depthWidth);
        return result;
    }

    // Extract RGB portion (left half)
    const int rgbDataSize = m_metadata.rgbWidth * m_metadata.height * 4;  // RGBA
    result.rgbData.resize(rgbDataSize);

    for (int y = 0; y < m_metadata.height; y++) {
        const uint8_t* srcRow = frame.data.data() + (y * totalWidth * 4);
        uint8_t* dstRow = result.rgbData.data() + (y * m_metadata.rgbWidth * 4);
        memcpy(dstRow, srcRow, m_metadata.rgbWidth * 4);
    }

    // Extract and unpack depth portion (right half)
    const uint8_t* depthRgbaStart = frame.data.data() + (m_metadata.rgbWidth * 4);
    result.depthData = unpackDepth(depthRgbaStart, m_metadata.depthWidth, m_metadata.height);

    return result;
}

} // namespace vk_viewer
