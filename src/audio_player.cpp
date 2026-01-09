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
#include <libswresample/swresample.h>
}

#include <nvutils/logger.hpp>

namespace vk_gaussian_splatting {

AudioPlayer::~AudioPlayer() {
    close();
}

bool AudioPlayer::initFFmpeg() {
    // Register all formats and codecs
    av_register_all();
    
    // Open audio file
    AVFormatContext* formatContext = nullptr;
    const char* formatFilePath = m_filePath.c_str();
    
    int result = avformat_open_input(&formatContext, formatFilePath, nullptr, nullptr);
    if (result != 0) {
        char errorBuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(result, errorBuf, AV_ERROR_MAX_STRING_SIZE);
        LOGE("AudioPlayer: Failed to open audio file %s: %s", formatFilePath, errorBuf);
        return false;
    }
    
    // Find audio stream
    AVCodecParameters* codecParams = avcodec_parameters_alloc();
    codecParams->sample_rate = 44100;
    codecParams->channels = 2;
    
    AVStream* audioStream = nullptr;
    result = avformat_new_stream(formatContext, nullptr, codecParams);
    if (result < 0) {
        LOGE("AudioPlayer: Failed to create audio stream: %d", result);
        avcodec_parameters_free(&codecParams);
        avformat_close_input(&formatContext);
        return false;
    }
    
    // Find decoder
    const AVCodec* decoder = avcodec_find_decoder(AV_CODEC_ID_MP3);
    if (!decoder) {
        LOGE("AudioPlayer: Failed to find MP3 decoder");
        avformat_close_input(&formatContext);
        return false;
    }
    
    // Open decoder
    AVCodecContext* codecContext = avcodec_alloc_context3(decoder);
    if (!codecContext) {
        LOGE("AudioPlayer: Failed to allocate codec context");
        avformat_free_streams(formatContext);
        avformat_close_input(&formatContext);
        avcodec_parameters_free(&codecParams);
        return false;
    }
    
    result = avcodec_open2(codecContext, decoder, nullptr, codecParams);
    if (result < 0) {
        char errorBuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(result, errorBuf, AV_ERROR_MAX_STRING_SIZE);
        LOGE("AudioPlayer: Failed to open codec: %d", result);
        avformat_free_streams(formatContext);
        avformat_close_input(&formatContext);
        avcodec_free_context(&codecContext);
        avcodec_parameters_free(&codecParams);
        return false;
    }
    
    // Store format context
    m_formatContext = formatContext;
    m_codecContext = codecContext;
    
    // Get audio duration
    AVStream* stream = formatContext->streams[0];
    if (stream && stream->duration) {
        m_durationMs = static_cast<uint64_t>(stream->duration * 1000.0 * 
                                          static_cast<double>(stream->time_base.num) / 
                                          stream->time_base.den));
    }
    
    // Setup resampler for converting to our format
    m_resampler = swr_alloc();
    if (!m_resampler) {
        LOGE("AudioPlayer: Failed to allocate resampler");
        return false;
    }
    
    av_opt_set_int(m_codecContext, "refcounted_packets", 1);
    
    // Setup resampler to convert input sample rate to output
    int64_t outSampleRate = 44100;
    result = swr_init(m_resampler, 
                     stream->codecpar->sample_rate, // Input sample rate
                     AV_CH_LAYOUT_STEREO,       // Input channel layout  
                     AV_SAMPLE_FMT_S16,           // Input format
                     &outSampleRate,             // Output sample rate
                     AV_CH_LAYOUT_STEREO,       // Output channel layout
                     AV_SAMPLE_FMT_S16);           // Output format
    
    if (result < 0) {
        LOGE("AudioPlayer: Failed to initialize resampler: %d", result);
        swr_free(&m_resampler);
        return false;
    }
    
    // Allocate audio buffer
    m_audioBufferSize = 8192; // 2 channels * 1024 samples * 4 bytes
    m_audioBuffer = static_cast<uint8_t*>(av_malloc(m_audioBufferSize));
    if (!m_audioBuffer) {
        LOGE("AudioPlayer: Failed to allocate audio buffer");
        return false;
    }
    
    avcodec_parameters_free(&codecParams);
    return true;
}

void AudioPlayer::cleanup() {
    if (m_resampler) {
        swr_free(&m_resampler);
        m_resampler = nullptr;
    }
    
    if (m_audioBuffer) {
        av_free(m_audioBuffer);
        m_audioBuffer = nullptr;
        m_audioBufferSize = 0;
    }
    
    if (m_codecContext) {
        avcodec_free_context(&m_codecContext);
        m_codecContext = nullptr;
    }
    
    if (m_formatContext) {
        avformat_close_input(&m_formatContext);
        m_formatContext = nullptr;
    }
}

bool AudioPlayer::load(const std::string& filepath) {
    close();
    
    m_filePath = filepath;
    
    if (!initFFmpeg()) {
        return false;
    }
    
    m_isLoaded = true;
    return true;
}

void AudioPlayer::close() {
    m_isLoaded = false;
    m_isPlaying = false;
    cleanup();
}

void AudioPlayer::play() {
    if (!m_isLoaded || !m_formatContext) {
        LOGE("AudioPlayer: Cannot play - no audio loaded");
        return;
    }
    
    m_isPlaying = true;
}

void AudioPlayer::pause() {
    m_isPlaying = false;
}

void AudioPlayer::stop() {
    m_isPlaying = false;
    if (m_positionMs != 0) {
        seek(0);
    }
}

void AudioPlayer::setVolume(float volume) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_volume = std::max(0.0f, std::min(1.0f, volume));
}

float AudioPlayer::getVolume() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_volume;
}

uint64_t AudioPlayer::getDurationMs() const {
    return m_durationMs;
}

uint64_t AudioPlayer::getPositionMs() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_positionMs;
}

bool AudioPlayer::seek(uint64_t positionMs) {
    if (!m_isLoaded || !m_formatContext) {
        return false;
    }
    
    // Seeking would require more complex implementation with FFmpeg
    // For now, just update position for timing synchronization
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