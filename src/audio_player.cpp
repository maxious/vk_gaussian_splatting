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

#include "audio_player.h"
#include <thread>
#include <chrono>

// FFmpeg includes - these need to be available at build time
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/error.h>
}

#include <nvutils/logger.hpp>

namespace vk_gaussian_splatting {

AudioPlayer::~AudioPlayer() {
    close();
}

bool AudioPlayer::initFFmpeg() {
    // FFmpeg 4.0+ doesn't require av_register_all()
    return true;
}

bool AudioPlayer::load(const std::string& filepath) {
    close();
    
    m_filePath = filepath;
    m_isLoaded = true;
    m_durationMs = 0;
    m_positionMs = 0;
    m_volume = 1.0f;
    m_isPlaying = false;
    
    LOGI("AudioPlayer: Audio tracking enabled for: %s", filepath.c_str());
    return true;
}

void AudioPlayer::close() {
    m_isLoaded = false;
    m_isPlaying = false;
    m_filePath.clear();
}

void AudioPlayer::play() {
    if (m_isLoaded) {
        m_isPlaying = true;
    }
}

void AudioPlayer::pause() {
    m_isPlaying = false;
}

void AudioPlayer::stop() {
    m_isPlaying = false;
    m_positionMs = 0;
}

void AudioPlayer::setVolume(float volume) {
    m_volume = std::max(0.0f, std::min(1.0f, volume));
}

bool AudioPlayer::seek(uint64_t positionMs) {
    if (!m_isLoaded) {
        return false;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    m_positionMs = std::min(positionMs, m_durationMs);
    return true;
}

void AudioPlayer::update() {
    // This would be called regularly to update playback position
    // For now, it's a no-op since we're not implementing actual playback
    // In a full implementation, this would update m_positionMs based on time
}

} // namespace vk_gaussian_splatting
