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

#include <memory>
#include <string>
#include <mutex>
#include <atomic>

// FFmpeg includes
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libswresample/swresample.h>
}

namespace vk_viewer {

/**
 * @brief Audio player for synchronized playback with PLY sequences
 * 
 * Uses FFmpeg libraries to decode and play various audio formats
 * (WAV, MP3, AAC, OGG, FLAC). Provides seeking, volume control,
 * and timing synchronization for animation playback.
 */
class AudioPlayer {
public:
    AudioPlayer() = default;
    ~AudioPlayer();

    /**
     * @brief Load an audio file
     * @param filepath Path to audio file
     * @return true if successfully loaded
     */
    bool load(const std::string& filepath);

    /**
     * @brief Close audio and release resources
     */
    void close();

    /**
     * @brief Check if audio is loaded
     */
    bool isLoaded() const { return m_isLoaded; }

    /**
     * @brief Play audio
     */
    void play();

    /**
     * @brief Pause audio
     */
    void pause();

    /**
     * @brief Stop audio and reset to beginning
     */
    void stop();

    /**
     * @brief Set volume (0.0 to 1.0)
     */
    void setVolume(float volume);

    /**
     * @brief Get current volume
     */
    float getVolume() const { return m_volume; }

    /**
     * @brief Get audio duration in milliseconds
     */
    uint64_t getDurationMs() const { return m_durationMs; }

    /**
     * @brief Get current playback position in milliseconds
     */
    uint64_t getPositionMs() const { return m_positionMs; }

    /**
     * @brief Seek to position in milliseconds
     */
    bool seek(uint64_t positionMs);

    /**
     * @brief Update playback state (call during playback loop)
     */
    void update();

private:
    bool initFFmpeg();
    void cleanup();

    AVFormatContext*                         m_formatContext = nullptr;
    AVCodecContext*                          m_codecContext = nullptr;
    AVCodecParameters*                       m_codecParameters = nullptr;
    SwrContext*                              m_resampler = nullptr;
    void*                                    m_swsContext = nullptr;
    
    uint8_t* m_audioBuffer = nullptr;
    int m_audioBufferSize = 0;
    
    std::string                     m_filePath;
    uint64_t                       m_durationMs = 0;
    uint64_t                       m_positionMs = 0;
    float                           m_volume = 1.0f;
    bool                            m_isLoaded = false;
    bool                            m_isPlaying = false;
    std::mutex                       m_mutex;
};

} // namespace vk_viewer