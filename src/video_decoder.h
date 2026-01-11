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
#include <memory>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>

struct AVFormatContext;
struct AVCodecContext;
struct AVFrame;
struct AVPacket;
struct SwsContext;

namespace vk_viewer {

/**
 * @brief Decoded video frame data
 */
struct DecodedFrame {
    std::vector<uint8_t> data;  // RGBA pixel data
    int width;
    int height;
    double timestamp;  // In seconds
    int64_t pts;       // Presentation timestamp
};

/**
 * @brief Video decoder using FFmpeg for real-time video decoding to textures
 *
 * Based on the ffplay.c pattern with send_packet/receive_frame decoding loop.
 * Decodes video frames to RGBA format suitable for texture upload.
 */
class VideoDecoder {
public:
    VideoDecoder();
    ~VideoDecoder();

    /**
     * @brief Open and initialize video file
     * @param filepath Path to video file
     * @return true if successful
     */
    bool open(const std::filesystem::path& filepath);

    /**
     * @brief Close video file and cleanup resources
     */
    void close();

    /**
     * @brief Start decoding thread
     */
    void startDecoding();

    /**
     * @brief Stop decoding thread
     */
    void stopDecoding();

    /**
     * @brief Get next decoded frame (thread-safe)
     * @param frame Output frame data
     * @return true if frame available, false if end of stream or error
     */
    bool getNextFrame(DecodedFrame& frame);

    /**
     * @brief Seek to specific timestamp
     * @param timestamp Timestamp in seconds
     * @return true if successful
     */
    bool seekToTime(double timestamp);

    /**
     * @brief Get video duration in seconds
     */
    double getDuration() const;

    /**
     * @brief Get video frame rate
     */
    double getFrameRate() const;

    /**
     * @brief Get video dimensions
     */
    void getDimensions(int& width, int& height) const;

    /**
     * @brief Check if decoder is running
     */
    bool isRunning() const { return m_running.load(); }

    /**
     * @brief Get current decoding position in seconds
     */
    double getCurrentTime() const;

    /**
     * @brief Pause decoding
     */
    void pause();

    /**
     * @brief Resume decoding
     */
    void resume();

    /**
     * @brief Check if decoding is paused
     */
    bool isPaused() const { return m_paused.load(); }

private:
    /**
     * @brief Decoding thread function
     */
    void decodingThread();

    /**
     * @brief Decode frames until end of stream
     */
    void decodeFrames();

    /**
     * @brief Initialize FFmpeg contexts
     */
    bool initializeFFmpeg();

    /**
     * @brief Cleanup FFmpeg contexts
     */
    void cleanupFFmpeg();

    /**
     * @brief Process a decoded frame and add to queue
     */
    void processFrame();

    // FFmpeg contexts
    AVFormatContext* m_formatContext;
    AVCodecContext* m_codecContext;
    SwsContext* m_swsContext;
    AVFrame* m_avFrame;
    AVFrame* m_rgbaFrame;
    AVPacket* m_packet;

    // Video stream information
    int m_videoStreamIndex;
    int m_width;
    int m_height;
    double m_frameRate;
    double m_duration;

    // Decoding thread
    std::thread m_decodeThread;
    std::atomic<bool> m_running;
    std::atomic<bool> m_stopRequested;
    std::atomic<bool> m_paused;
    std::condition_variable m_pauseCondition;
    std::mutex m_pauseMutex;

    // Frame queue for thread safety
    std::vector<DecodedFrame> m_frameQueue;
    std::mutex m_queueMutex;
    std::condition_variable m_queueCondition;
    size_t m_maxQueueSize;

    // Seek state
    std::atomic<bool> m_seekRequested;
    double m_seekTimestamp;

    // Error handling
    std::string m_errorMessage;
};

} // namespace vk_viewer