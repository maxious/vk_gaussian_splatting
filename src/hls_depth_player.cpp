/*
 * HLS Depth Player - Standards-compliant HLS stream player with Depth+Normals unpacking
 *
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hls_depth_player.h"
#include <nvutils/logger.hpp>
#include <fstream>
#include <tinygltf/json.hpp>

#ifdef WITH_VIDEO_DECODER
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
}
#endif

namespace vk_viewer {

#ifdef WITH_VIDEO_DECODER

HlsDepthPlayer::HlsDepthPlayer() : m_decoder(std::make_unique<VideoDecoder>())
{
}

HlsDepthPlayer::~HlsDepthPlayer()
{
    stop();
}

bool HlsDepthPlayer::open(const std::filesystem::path& playlistPath)
{
    // Load metadata from JSON
    if (!loadMetadata(playlistPath)) {
        LOGW("Failed to load metadata from %s\n", playlistPath.string().c_str());
        return false;
    }

    // Open depth video
    if (!m_decoder->open(m_metadata.depthVideoPath)) {
        LOGE("Failed to open depth video: %s\n", m_metadata.depthVideoPath.c_str());
        return false;
    }

    int width, height;
    m_decoder->getDimensions(width, height);

    // Set dimensions from metadata
    m_metadata.width = width;
    m_metadata.height = height;
    m_metadata.duration = static_cast<double>(m_metadata.frameCount) / m_metadata.fps;

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

    LOGI("HLS Depth Player: %dx%d @ %.1f FPS, z_min: %.2f, z_max: %.2f\n",
         width, height, m_metadata.fps, m_metadata.zMin, m_metadata.zMax);

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

        m_metadata.depthVideoPath = json.value("depth_video_path", "");
        m_metadata.frameCount = json.value("frame_count", 0);
        m_metadata.fps = json.value("fps", 30.0f);
        m_metadata.duration = json.value("duration_s", 0.0);
        m_metadata.zMin = json.value("z_min", 0.0f);
        m_metadata.zMax = json.value("z_max", 10.0f);
        m_metadata.scale = json.value("scale", 0.0f);

        // Parse MoGe-specific fields
        m_metadata.hasNormals = json.value("has_normals", false);
        m_metadata.sideBySide = json.value("side_by_side", false);

        // Get dimensions from JSON or calculate from frame count
        m_metadata.sourceWidth = json.value("source_width", 0);
        m_metadata.sourceHeight = json.value("source_height", 0);

        // For side-by-side videos, video width is doubled
        if (m_metadata.sideBySide) {
            m_metadata.width = m_metadata.sourceWidth * 2;
            m_metadata.height = m_metadata.sourceHeight;
            m_metadata.rgbWidth = m_metadata.sourceWidth * 2;  // Full width for RGB display
        } else {
            m_metadata.width = m_metadata.sourceWidth;
            m_metadata.height = m_metadata.sourceHeight;
            m_metadata.rgbWidth = m_metadata.sourceWidth;
        }

        // Recalculate scale if needed
        if (m_metadata.scale <= 0.0f && m_metadata.zMax > m_metadata.zMin) {
            m_metadata.scale = (m_metadata.zMax - m_metadata.zMin) / 65535.0f;
        }

        LOGI("Loaded HLS metadata: width=%d, height=%d, z_min=%.2f, z_max=%.2f, scale=%.6f\n",
             m_metadata.width, m_metadata.height, m_metadata.zMin, m_metadata.zMax, m_metadata.scale);

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

    // Extract depth and normals from side-by-side frame
    frame.width = decodedFrame.width;
    frame.height = decodedFrame.height;
    frame.timestamp = decodedFrame.timestamp;

    // For side-by-side videos, left half is depth, right half is normals
    if (m_metadata.sideBySide) {
        int halfWidth = decodedFrame.width / 2;

        // Extract depth data (left half, grayscale)
        size_t depthPixels = halfWidth * decodedFrame.height;
        frame.depthData.resize(depthPixels);
        frame.normalsData.resize(depthPixels * 3); // RGB normals

        // Extract normals data (right half, RGB)
        size_t normalsPixels = depthPixels * 3; // RGB channels

        for (int y = 0; y < decodedFrame.height; y++) {
            for (int x = 0; x < halfWidth; x++) {
                // Depth from left half (grayscale)
                int leftIdx = (y * decodedFrame.width + x) * 4; // RGBA
                uint8_t depthByte = decodedFrame.data[leftIdx]; // Use R channel
                float normalizedDepth = depthByte / 255.0f;
                float depthMeters = m_metadata.zMin + normalizedDepth * (m_metadata.zMax - m_metadata.zMin);
                frame.depthData[y * halfWidth + x] = depthMeters;

                // Normals from right half (RGB)
                int rightIdx = (y * decodedFrame.width + x + halfWidth) * 4; // RGBA, right half
                frame.normalsData[(y * halfWidth + x) * 3 + 0] = decodedFrame.data[rightIdx + 0]; // R
                frame.normalsData[(y * halfWidth + x) * 3 + 1] = decodedFrame.data[rightIdx + 1]; // G
                frame.normalsData[(y * halfWidth + x) * 3 + 2] = decodedFrame.data[rightIdx + 2]; // B
            }
        }

        // Reconstruct RGB data for video display (use normals visualization)
        frame.rgbData.resize(normalsPixels);
        for (size_t i = 0; i < normalsPixels; i++) {
            frame.rgbData[i] = frame.normalsData[i];
        }
    } else {
        // For regular depth videos, create grayscale RGB data from depth
        size_t pixels = decodedFrame.width * decodedFrame.height;
        frame.depthData.resize(pixels);
        frame.rgbData.resize(pixels * 3);

        for (size_t i = 0; i < pixels; i++) {
            int srcIdx = i * 4; // RGBA
            uint8_t depthByte = decodedFrame.data[srcIdx]; // Use R channel
            float normalizedDepth = depthByte / 255.0f;
            float depthMeters = m_metadata.zMin + normalizedDepth * (m_metadata.zMax - m_metadata.zMin);
            frame.depthData[i] = depthMeters;

            // Create grayscale RGB
            frame.rgbData[i * 3 + 0] = depthByte; // R
            frame.rgbData[i * 3 + 1] = depthByte; // G
            frame.rgbData[i * 3 + 2] = depthByte; // B
        }
    }

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
    // Return decoder current time if available
    if (m_decoder) {
        return m_decoder->getCurrentTime();
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

#else

HlsDepthPlayer::HlsDepthPlayer() {}

HlsDepthPlayer::~HlsDepthPlayer() {}

bool HlsDepthPlayer::open(const std::filesystem::path&) {
    LOGW("HlsDepthPlayer: Not available (FFmpeg not enabled)\\n");
    return false;
}

bool HlsDepthPlayer::loadMetadata(const std::filesystem::path&) {
    return false;
}

void HlsDepthPlayer::start() {}

void HlsDepthPlayer::stop() {}

bool HlsDepthPlayer::getNextFrame(HlsDecodedFrame&) {
    return false;
}

bool HlsDepthPlayer::seekToTime(double) {
    return false;
}

double HlsDepthPlayer::getCurrentTime() const {
    return 0.0;
}

void HlsDepthPlayer::pause() {}

void HlsDepthPlayer::resume() {}

std::vector<float> HlsDepthPlayer::unpackDepthFromGrayscale(const uint8_t*, int, int) {
    return {};
}

HlsDecodedFrame HlsDepthPlayer::separateFrame(const DecodedFrame&) {
    return {};
}

#endif

} // namespace vk_viewer
