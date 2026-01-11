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
    // Load metadata from JSON
    if (!loadMetadata(playlistPath)) {
        LOGW("Failed to load metadata from %s\n", playlistPath.string().c_str());
        return false;
    }

    // Open depth video
    if (!m_decoder->open(m_metadata.depthVideoPath)) {
        LOGE("Failed to open depth video: %s\n", m_metadata.depthVideoPath.string().c_str());
        return false;
    }

    int width, height;
    m_decoder->getDimensions(width, height);

    // Set dimensions from metadata
    m_metadata.width = width;
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

        m_metadata.width = width;
        m_metadata.height = height;

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

std::lock_guard<std::mutex> lock(m_queueMutex);
m_frameQueue.clear();

bool HlsDepthPlayer::getNextFrame(HlsDecodedFrame& frame)
{
    if (!m_initialized || !m_decoder) {
        return false;
    }

    DecodedFrame decodedFrame;
    if (!m_decoder->getNextFrame(decodedFrame)) {
        return false;
    }

    frame = decodedFrame;
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
    // Return timestamp of most recent frame in queue
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

bool HlsDepthPlayer::isPaused() const
{
    return m_decoder && m_decoder->isPaused();
}

} // namespace vk_viewer
