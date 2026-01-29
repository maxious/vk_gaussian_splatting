/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING, software
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
#include <string>
#include <vector>
#include <cstdint>
#include <mutex>
#include <atomic>
#include <memory>
#include <thread>
#include <chrono>
#include <volk.h>

#include "video_decoder.h"

#ifdef WITH_VIDEO_DECODER
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}
#endif

namespace vk_viewer {

/**
 * @brief Metadata structure for depth video playback
 *
 * Loaded from metadata.json, contains paths and configuration
 * for synchronized video + depth video playback.
 */
struct DepthVideoMetadata {
    std::string videoPath;       // Path to source video
    std::string depthVideoPath;  // Path to depth video (H.265/HEVC lossless)
    int32_t frameCount = 0;
    double fps = 0.0;
    int32_t sourceWidth = 0;
    int32_t sourceHeight = 0;
    std::string modelId;
    std::string deviceSpec;
    int32_t processRes = 0;
    double processingTimeS = 0.0;
    double avgFps = 0.0;
    std::string format;          // e.g., "hevc_lossless"
    float zMin = 0.0f;
    float zMax = 0.0f;

    // Side-by-side depth + normals support
    bool hasNormals = false;      // Whether normals are available
    bool sideBySide = false;      // Whether video is side-by-side (depth | normals)
    int32_t normalsWidth = 0;     // Width of normals portion (same as depthWidth if side-by-side)

    // Computed values
    int32_t depthWidth = 0;
    int32_t depthHeight = 0;
};

/**
 * @brief Single depth frame with metric depth values
 */
struct DepthVideoFrame {
    uint32_t timestampMs;
    uint32_t width;
    uint32_t height;
    std::vector<float> data;  // Depth values in meters
    float zMin;
    float zMax;
};

/**
 * @brief Decoded video frame from FFmpeg
 */
struct DecodedVideoFrame {
    std::vector<uint8_t> data;  // RGBA or grayscale pixel data
    int width = 0;
    int height = 0;
    double timestamp = 0.0;     // In seconds
};

/**
 * @brief Load metadata.json file
 *
 * @param metadataPath Path to metadata.json file or directory containing it
 * @param outMetadata Output metadata structure
 * @return true if successfully loaded
 */
bool loadDepthVideoMetadata(const std::filesystem::path& metadataPath, DepthVideoMetadata& outMetadata);

/**
 * @brief Synchronized RGB+Depth frame for buffered playback
 * 
 * Contains both RGB video data and metric depth data for a single frame,
 * enabling instant random-access seeking once buffered.
 */
struct PlaybackFrame
{
    double timestampSec = 0.0;
    uint32_t frameIndex = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    
    std::vector<uint8_t> rgbRGBA;      // RGBA8 video data (CPU fallback)
    std::vector<float> depthMeters;    // Per-pixel depth in meters
    
    // HW Decoding Support
    VkImage rgbImage = VK_NULL_HANDLE;
    VkFormat rgbFormat = VK_FORMAT_UNDEFINED;
    VkSemaphore rgbSemaphore = VK_NULL_HANDLE;
    std::shared_ptr<void> hwFrameRef; // Keep alive the AVFrame/VkImage
    
    float zMin = 0.0f;
    float zMax = 1.0f;
};

/**
 * @brief Thread-safe buffer for synchronized video+depth frames
 * 
 * Used by VideoDepthPlaybackManager to enable instant random-access
 * seeking within buffered regions. Frames are appended by a background
 * thread and can be read by the render thread.
 */
class SynchronizedFrameBuffer
{
public:
    void init(double totalDurationSec, uint32_t frameCount, double fps);
    void clear();
    
    void pushFrame(PlaybackFrame&& frame);
    
    size_t getBufferedFrameCount() const { return m_bufferedCount.load(); }
    double getBufferedDurationSec() const;
    double getTotalDurationSec() const { return m_totalDurationSec; }
    uint32_t getTotalFrameCount() const { return m_totalFrameCount; }
    double getFps() const { return m_fps; }
    
    float getBufferedRatio() const;
    bool isFullyBuffered() const;
    
    bool getFrameAtTime(double tSec, PlaybackFrame& out) const;
    bool getFrameByIndex(uint32_t index, PlaybackFrame& out) const;
    
private:
    std::vector<PlaybackFrame> m_frames;
    std::atomic<size_t> m_bufferedCount{0};
    double m_totalDurationSec = 0.0;
    uint32_t m_totalFrameCount = 0;
    double m_fps = 30.0;
    mutable std::mutex m_mutex;
};

/**
 * @brief FFmpeg-based depth video loader for offline preprocessed sequences
 *
 * Loads depth video (H.265 lossless or similar) and decodes frames on demand.
 * Depth values are stored as normalized 8-bit grayscale in the video,
 * then converted back to metric depth using zMin/zMax from metadata.
 *
 * This class is designed to work alongside the existing VideoDecoder class
 * for synchronized video + depth playback.
 */
class DepthVideoLoader {
public:
    DepthVideoLoader();
    ~DepthVideoLoader();

    /**
     * @brief Open depth video file and initialize FFmpeg
     * @param depthVideoPath Path to depth video file
     * @param metadata Associated metadata
     * @return true if successful
     */
    bool open(const std::filesystem::path& depthVideoPath, const DepthVideoMetadata& metadata);

    /**
     * @brief Close and cleanup resources
     */
    void close();

    /**
     * @brief Check if loader is open
     */
    bool isOpen() const { return m_isOpen.load(); }

    /**
     * @brief Get frame count
     */
    int64_t getFrameCount() const { return m_frameCount; }

    /**
     * @brief Get frame rate
     */
    double getFrameRate() const { return m_frameRate; }

    /**
     * @brief Get video dimensions
     */
    void getDimensions(int& width, int& height) const;

    /**
     * @brief Get duration in seconds
     */
    double getDuration() const { return m_duration; }

    /**
     * @brief Get current playback position in seconds
     */
    double getCurrentTime() const { return m_currentTime.load(); }

    /**
     * @brief Seek to specific timestamp
     * @param timestamp Timestamp in seconds
     * @return true if successful
     */
    bool seekToTime(double timestamp);

    /**
     * @brief Get next depth frame (thread-safe)
     * @param frame Output depth frame with metric values
     * @return true if frame available, false if end of stream or error
     */
    bool getNextFrame(DepthVideoFrame& frame);

    /**
     * @brief Get depth frame by index
     * @param index Frame index (0-based)
     * @param frame Output depth frame
     * @return true if successful
     */
    bool getFrame(int64_t index, DepthVideoFrame& frame);

    /**
     * @brief Get depth frame by timestamp (closest match)
     * @param timestampMs Timestamp in milliseconds
     * @param frame Output depth frame
     * @return true if successful
     */
    bool getFrameByTimestamp(uint32_t timestampMs, DepthVideoFrame& frame);

private:
    /**
     * @brief Initialize FFmpeg contexts
     */
    bool initializeFFmpeg();

    /**
     * @brief Cleanup FFmpeg contexts
     */
    void cleanupFFmpeg();

    /**
     * @brief Decode next frame from video
     * @return true if frame decoded successfully
     */
    bool decodeNextFrame();

    /**
     * @brief Convert decoded grayscale frame to metric depth
     * @param grayscaleData 8-bit grayscale pixel data
     * @param width Frame width
     * @param height Frame height
     * @param outputFrame Output depth frame with metric values
     */
    void convertToMetricDepth(const uint8_t* grayscaleData, int width, int height, DepthVideoFrame& outputFrame);

#ifdef WITH_VIDEO_DECODER
    // FFmpeg contexts
    AVFormatContext* m_formatContext = nullptr;
    AVCodecContext* m_codecContext = nullptr;
    SwsContext* m_swsContext = nullptr;
    AVFrame* m_avFrame = nullptr;
    AVFrame* m_grayscaleFrame = nullptr;
    AVPacket* m_packet = nullptr;
#endif

    // Video stream information
    int m_videoStreamIndex = -1;
    int m_width = 0;
    int m_height = 0;
    double m_frameRate = 0.0;
    double m_duration = 0.0;
    int64_t m_frameCount = 0;

    // Playback state
    std::atomic<bool> m_isOpen{false};
    std::atomic<bool> m_eof{false};
    std::atomic<double> m_currentTime{0.0};

    // Frame buffer for current frame
    std::vector<uint8_t> m_currentFrameData;
    std::mutex m_frameMutex;
    int64_t m_currentFrameIndex = -1;
    double m_currentFramePts = 0.0;

    // Metadata
    DepthVideoMetadata m_metadata;

    // Error handling
    std::string m_errorMessage;
};

/**
 * @brief Combined video + depth playback manager with buffered frames
 *
 * Manages both video and depth video playback with synchronization.
 * Uses a buffered approach: frames are decoded into memory first,
 * then playback and seeking operate on the local buffer.
 */
class VideoDepthPlaybackManager {
public:
    VideoDepthPlaybackManager() = default;
    ~VideoDepthPlaybackManager();

    /**
     * @brief Open video and depth video from metadata.json
     * @param metadataPath Path to metadata.json or directory containing it
     * @param instance Vulkan instance (optional, for HW decoding)
     * @param physicalDevice Physical device (optional, for HW decoding)
     * @param device Vulkan device (optional, for HW decoding)
     * @param queueFamilyIndex Queue family index (optional, for HW decoding)
     * @param queueIndex Queue index (optional, for HW decoding)
     * @return true if successfully opened both video and depth
     */
    bool openFromMetadata(const std::filesystem::path& metadataPath,
                          VkInstance instance = VK_NULL_HANDLE,
                          VkPhysicalDevice physicalDevice = VK_NULL_HANDLE,
                          VkDevice device = VK_NULL_HANDLE,
                          uint32_t queueFamilyIndex = 0,
                          uint32_t queueIndex = 0);

    /**
     * @brief Close and cleanup
     */
    void close();

    /**
     * @brief Check if playback is active
     */
    bool isPlaying() const { return m_isPlaying.load(); }

    /**
     * @brief Get metadata
     */
    const DepthVideoMetadata& getMetadata() const { return m_metadata; }

    /**
     * @brief Start playback
     */
    void play();

    /**
     * @brief Pause playback
     */
    void pause();

    /**
     * @brief Toggle play/pause
     */
    void togglePlayPause();

    /**
     * @brief Seek to timestamp (clamped to buffered region)
     * @param timestamp Timestamp in seconds
     */
    void seek(double timestamp);

    /**
     * @brief Get current playback time (based on clock, not decoder)
     */
    double getCurrentTime() const;

    /**
     * @brief Check if paused
     */
    bool isPaused() const { return m_paused.load(); }

    /**
     * @brief Check if playback reached end of buffered content
     */
    bool isAtEnd() const;
    
    /**
     * @brief Get total duration from metadata
     */
    double getDuration() const;
    
    /**
     * @brief Get buffered duration (how much is ready for playback)
     */
    double getBufferedDuration() const;
    
    /**
     * @brief Get buffer fill ratio (0.0 to 1.0)
     */
    float getBufferedRatio() const;
    
    /**
     * @brief Check if fully buffered
     */
    bool isFullyBuffered() const;
    
    /**
     * @brief Check if buffering is in progress
     */
    bool isBuffering() const { return m_bufferingActive.load(); }
    
    /**
     * @brief Get frame at specified time from buffer
     * @param tSec Time in seconds
     * @param out Output frame
     * @return true if frame available
     */
    bool getFrameAtTime(double tSec, PlaybackFrame& out) const;

private:
    void prebufferThread();
    void updatePlaybackTime();
    
    std::unique_ptr<VideoDecoder> m_videoDecoder;
    std::unique_ptr<DepthVideoLoader> m_depthLoader;
    DepthVideoMetadata m_metadata;
    
    SynchronizedFrameBuffer m_buffer;
    std::thread m_prebufferThread;
    std::atomic<bool> m_bufferingActive{false};
    std::atomic<bool> m_stopBuffering{false};

    std::atomic<bool> m_isPlaying{false};
    std::atomic<bool> m_paused{true};
    
    std::chrono::steady_clock::time_point m_playbackStartTime;
    double m_seekOffsetSec = 0.0;
    mutable std::mutex m_timeMutex;
};

} // namespace vk_viewer
