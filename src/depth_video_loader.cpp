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

#include "depth_video_loader.h"
#include <nvutils/logger.hpp>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstring>

#ifdef WITH_VIDEO_DECODER
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}
#endif

namespace vk_viewer {

bool loadDepthVideoMetadata(const std::filesystem::path& metadataPath, DepthVideoMetadata& outMetadata)
{
    std::filesystem::path jsonPath = metadataPath;

    if (std::filesystem::is_directory(jsonPath))
    {
        jsonPath = jsonPath / "metadata.json";
    }

    if (!std::filesystem::exists(jsonPath))
    {
        LOGE("Metadata file not found: %s\n", jsonPath.string().c_str());
        return false;
    }

    std::ifstream file(jsonPath);
    if (!file.is_open())
    {
        LOGE("Failed to open metadata file: %s\n", jsonPath.string().c_str());
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string jsonContent = buffer.str();

    auto findValue = [&](const std::string& json, const std::string& key) -> std::string {
        std::string searchKey = "\"" + key + "\"";
        size_t pos = json.find(searchKey);
        if (pos == std::string::npos)
            return "";

        pos = json.find(":", pos);
        if (pos == std::string::npos)
            return "";

        pos = json.find_first_not_of(" \t\n", pos + 1);
        if (pos == std::string::npos)
            return "";

        char firstChar = json[pos];

        if (firstChar == '"')
        {
            size_t start = pos + 1;
            size_t end = json.find("\"", start);
            if (end == std::string::npos)
                return "";
            return json.substr(start, end - start);
        }
        else if (firstChar == '[')
        {
            size_t start = pos;
            size_t end = json.find("]", start);
            if (end == std::string::npos)
                return "";
            return json.substr(start, end - start + 1);
        }
        else
        {
            size_t start = pos;
            size_t end = json.find_first_of(",}\n", start);
            if (end == std::string::npos)
                end = json.length();
            return json.substr(start, end - start);
        }
    };

    auto getString = [&](const std::string& key) -> std::string {
        std::string value = findValue(jsonContent, key);
        size_t start = value.find_first_not_of(" \t\n");
        size_t end = value.find_last_not_of(" \t\n");
        if (start == std::string::npos)
            return "";
        return value.substr(start, end - start + 1);
    };

    auto getFloat = [&](const std::string& key) -> float {
        std::string value = getString(key);
        if (value.empty())
            return 0.0f;
        try
        {
            return std::stof(value);
        }
        catch (...)
        {
            return 0.0f;
        }
    };

    auto getInt = [&](const std::string& key) -> int32_t {
        std::string value = getString(key);
        if (value.empty())
            return 0;
        try
        {
            return static_cast<int32_t>(std::stol(value));
        }
        catch (...)
        {
            return 0;
        }
    };

    auto getDouble = [&](const std::string& key) -> double {
        std::string value = getString(key);
        if (value.empty())
            return 0.0;
        try
        {
            return std::stod(value);
        }
        catch (...)
        {
            return 0.0;
        }
    };

    auto getIntArray = [&](const std::string& key) -> std::vector<int32_t> {
        std::vector<int32_t> result;
        std::string value = findValue(jsonContent, key);
        
        size_t start = value.find("[");
        size_t end = value.find("]");
        
        if (start != std::string::npos && end != std::string::npos) {
            std::string content = value.substr(start + 1, end - start - 1);
            std::stringstream ss(content);
            std::string item;
            while (std::getline(ss, item, ',')) {
                try {
                    item.erase(0, item.find_first_not_of(" \t\n"));
                    item.erase(item.find_last_not_of(" \t\n") + 1);
                    result.push_back(std::stoi(item));
                } catch (...) {
                    LOGW("Failed to parse int from array item: %s\n", item.c_str());
                }
            }
        }
        return result;
    };

    outMetadata.videoPath = getString("video_path");
    outMetadata.depthVideoPath = getString("depth_video_path");
    outMetadata.frameCount = getInt("frame_count");
    outMetadata.fps = getDouble("fps");
    
    outMetadata.sourceWidth = getInt("source_width");
    outMetadata.sourceHeight = getInt("source_height");
    
    if (outMetadata.sourceWidth == 0 || outMetadata.sourceHeight == 0) {
        std::vector<int32_t> res = getIntArray("source_resolution");
        if (res.size() >= 2) {
            outMetadata.sourceWidth = res[0];
            outMetadata.sourceHeight = res[1];
        }
    }

    outMetadata.modelId = getString("model_id");

    outMetadata.deviceSpec = getString("device_spec");
    outMetadata.processRes = getInt("process_res");
    outMetadata.processingTimeS = getDouble("processing_time_s");
    outMetadata.avgFps = getDouble("avg_fps");
    outMetadata.format = getString("format");
    outMetadata.zMin = getFloat("z_min");
    outMetadata.zMax = getFloat("z_max");

    // Parse MoGe-specific fields
    outMetadata.hasNormals = getString("has_normals") == "true";
    outMetadata.sideBySide = getString("side_by_side") == "true";

    // For side-by-side videos, the video width is doubled (depth | normals)
    if (outMetadata.sideBySide) {
        outMetadata.depthWidth = outMetadata.sourceWidth / 2;  // Left half is depth
        outMetadata.normalsWidth = outMetadata.sourceWidth / 2; // Right half is normals
        outMetadata.depthHeight = outMetadata.sourceHeight;
    } else {
        outMetadata.depthWidth = outMetadata.sourceWidth;
        outMetadata.depthHeight = outMetadata.sourceHeight;
        outMetadata.normalsWidth = 0;
    }

    std::filesystem::path basePath = jsonPath.parent_path();
    LOGI("Resolving paths relative to: %s\n", basePath.string().c_str());

    if (!outMetadata.videoPath.empty() && !std::filesystem::path(outMetadata.videoPath).is_absolute())
    {
        outMetadata.videoPath = (basePath / outMetadata.videoPath).string();
        LOGI("Resolved video path: %s\n", outMetadata.videoPath.c_str());
    }

    if (!outMetadata.depthVideoPath.empty() && !std::filesystem::path(outMetadata.depthVideoPath).is_absolute())
    {
        outMetadata.depthVideoPath = (basePath / outMetadata.depthVideoPath).string();
        LOGI("Resolved depth video path: %s\n", outMetadata.depthVideoPath.c_str());
    }

    LOGI("Loaded depth video metadata:\n");
    LOGI("  Video: %s\n", outMetadata.videoPath.c_str());
    LOGI("  Depth Video: %s\n", outMetadata.depthVideoPath.c_str());
    LOGI("  Frames: %d, FPS: %.2f\n", outMetadata.frameCount, outMetadata.fps);
    LOGI("  Resolution: %dx%d\n", outMetadata.sourceWidth, outMetadata.sourceHeight);
    LOGI("  Z Range: [%.2f, %.2f] m\n", outMetadata.zMin, outMetadata.zMax);
    LOGI("  Side-by-side: %s, Has normals: %s\n",
         outMetadata.sideBySide ? "yes" : "no",
         outMetadata.hasNormals ? "yes" : "no");
    if (outMetadata.sideBySide) {
        LOGI("  Depth portion: %dx%d, Normals portion: %dx%d\n",
             outMetadata.depthWidth, outMetadata.depthHeight,
             outMetadata.normalsWidth, outMetadata.depthHeight);
    }

    return true;
}

DepthVideoLoader::DepthVideoLoader()
{
}

DepthVideoLoader::~DepthVideoLoader()
{
    close();
}

bool DepthVideoLoader::open(const std::filesystem::path& depthVideoPath, const DepthVideoMetadata& metadata)
{
#ifdef WITH_VIDEO_DECODER
    m_metadata = metadata;
    m_currentFrameIndex = -1;

    if (!initializeFFmpeg())
    {
        return false;
    }

    m_isOpen.store(true);
    LOGI("Depth video loader opened: %s\n", depthVideoPath.string().c_str());
    return true;
#else
    LOGE("Video decoder not available - rebuild with ENABLE_VIDEO_DECODER=ON\n");
    return false;
#endif
}

void DepthVideoLoader::close()
{
#ifdef WITH_VIDEO_DECODER
    if (m_isOpen.load())
    {
        cleanupFFmpeg();
        m_isOpen.store(false);
        LOGI("Depth video loader closed\n");
    }
#endif
}

void DepthVideoLoader::getDimensions(int& width, int& height) const
{
    width = m_width;
    height = m_height;
}

bool DepthVideoLoader::seekToTime(double timestamp)
{
#ifdef WITH_VIDEO_DECODER
    if (!m_isOpen.load() || !m_formatContext)
        return false;

    double targetFrameTime = timestamp * 1000.0;
    int64_t targetFrame = static_cast<int64_t>(targetFrameTime * m_frameRate / 1000.0);
    targetFrame = std::max<int64_t>(0, std::min(targetFrame, m_frameCount - 1));

    int64_t seekTarget = static_cast<int64_t>(timestamp * AV_TIME_BASE);
    int flags = AVSEEK_FLAG_BACKWARD;

    int ret = av_seek_frame(m_formatContext, m_videoStreamIndex, seekTarget, flags);
    if (ret < 0)
    {
        LOGW("Failed to seek to timestamp %.3f\n", timestamp);
        return false;
    }

    avcodec_flush_buffers(m_codecContext);

    m_currentFrameIndex = targetFrame - 1;
    m_eof.store(false);

    return decodeNextFrame();
#else
    return false;
#endif
}

bool DepthVideoLoader::getNextFrame(DepthVideoFrame& frame)
{
#ifdef WITH_VIDEO_DECODER
    if (!m_isOpen.load())
        return false;

    std::lock_guard<std::mutex> lock(m_frameMutex);

    if (m_currentFrameData.empty())
    {
        if (!decodeNextFrame())
            return false;
    }

    convertToMetricDepth(m_currentFrameData.data(), m_width, m_height, frame);
    m_currentFrameData.clear();

    return true;
#else
    return false;
#endif
}

bool DepthVideoLoader::getFrame(int64_t index, DepthVideoFrame& frame)
{
#ifdef WITH_VIDEO_DECODER
    if (!m_isOpen.load() || index < 0 || index >= m_frameCount)
        return false;

    double targetTime = static_cast<double>(index) / m_frameRate;

    if (!seekToTime(targetTime))
    {
        return false;
    }

    return getNextFrame(frame);
#else
    return false;
#endif
}

bool DepthVideoLoader::getFrameByTimestamp(uint32_t timestampMs, DepthVideoFrame& frame)
{
    if (!m_isOpen.load())
        return false;

    double timeSeconds = static_cast<double>(timestampMs) / 1000.0;
    int64_t frameIndex = static_cast<int64_t>(timeSeconds * m_frameRate);
    frameIndex = std::max<int64_t>(0, std::min(frameIndex, m_frameCount - 1));

    return getFrame(frameIndex, frame);
}

#ifdef WITH_VIDEO_DECODER

bool DepthVideoLoader::initializeFFmpeg()
{
    int ret = avformat_open_input(&m_formatContext, m_metadata.depthVideoPath.c_str(), nullptr, nullptr);
    if (ret < 0)
    {
        char errorBuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errorBuf, sizeof(errorBuf));
        LOGE("Failed to open depth video: %s (error: %s)\n", m_metadata.depthVideoPath.c_str(), errorBuf);
        return false;
    }

    ret = avformat_find_stream_info(m_formatContext, nullptr);
    if (ret < 0)
    {
        LOGE("Failed to find stream info in depth video\n");
        cleanupFFmpeg();
        return false;
    }

    m_videoStreamIndex = -1;
    for (unsigned int i = 0; i < m_formatContext->nb_streams; i++)
    {
        if (m_formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
        {
            m_videoStreamIndex = i;
            break;
        }
    }

    if (m_videoStreamIndex < 0)
    {
        LOGE("No video stream found in depth video\n");
        cleanupFFmpeg();
        return false;
    }

    AVStream* stream = m_formatContext->streams[m_videoStreamIndex];
    const AVCodec* codec = avcodec_find_decoder(stream->codecpar->codec_id);

    if (!codec)
    {
        LOGE("Codec not found for depth video\n");
        cleanupFFmpeg();
        return false;
    }

    m_codecContext = avcodec_alloc_context3(codec);
    if (!m_codecContext)
    {
        LOGE("Failed to allocate codec context\n");
        cleanupFFmpeg();
        return false;
    }

    ret = avcodec_parameters_to_context(m_codecContext, stream->codecpar);
    if (ret < 0)
    {
        LOGE("Failed to copy codec parameters\n");
        cleanupFFmpeg();
        return false;
    }

    ret = avcodec_open2(m_codecContext, codec, nullptr);
    if (ret < 0)
    {
        LOGE("Failed to open codec\n");
        cleanupFFmpeg();
        return false;
    }

    m_avFrame = av_frame_alloc();
    m_grayscaleFrame = av_frame_alloc();
    m_packet = av_packet_alloc();

    if (!m_avFrame || !m_grayscaleFrame || !m_packet)
    {
        LOGE("Failed to allocate FFmpeg frames\n");
        cleanupFFmpeg();
        return false;
    }

    m_width = stream->codecpar->width;
    m_height = stream->codecpar->height;

    if (stream->avg_frame_rate.num > 0)
    {
        m_frameRate = static_cast<double>(stream->avg_frame_rate.num) / stream->avg_frame_rate.den;
    }
    else if (stream->r_frame_rate.num > 0)
    {
        m_frameRate = static_cast<double>(stream->r_frame_rate.num) / stream->r_frame_rate.den;
    }
    else
    {
        m_frameRate = 30.0;
    }

    if (stream->duration != AV_NOPTS_VALUE)
    {
        m_duration = static_cast<double>(stream->duration) * av_q2d(stream->time_base);
    }
    else if (m_formatContext->duration != AV_NOPTS_VALUE)
    {
        m_duration = static_cast<double>(m_formatContext->duration) / AV_TIME_BASE;
    }
    else
    {
        m_duration = static_cast<double>(m_metadata.frameCount) / m_frameRate;
    }

    m_frameCount = static_cast<int64_t>(m_duration * m_frameRate);

    m_grayscaleFrame->width = m_width;
    m_grayscaleFrame->height = m_height;
    m_grayscaleFrame->format = AV_PIX_FMT_GRAY8;
    m_grayscaleFrame->linesize[0] = m_width;

    size_t bufferSize = m_width * m_height;
    m_grayscaleFrame->buf[0] = av_buffer_alloc(bufferSize);
    m_grayscaleFrame->data[0] = m_grayscaleFrame->buf[0]->data;

    m_swsContext = sws_getContext(m_width, m_height, static_cast<AVPixelFormat>(stream->codecpar->format),
                                   m_width, m_height, AV_PIX_FMT_GRAY8,
                                   SWS_BILINEAR, nullptr, nullptr, nullptr);

    if (!m_swsContext)
    {
        LOGE("Failed to create scaling context\n");
        cleanupFFmpeg();
        return false;
    }

    LOGI("Depth video initialized: %dx%d @ %.2f fps, %zu frames\n",
         m_width, m_height, m_frameRate, static_cast<size_t>(m_frameCount));

    return true;
}

void DepthVideoLoader::cleanupFFmpeg()
{
    if (m_swsContext)
    {
        sws_freeContext(m_swsContext);
        m_swsContext = nullptr;
    }

    if (m_packet)
    {
        av_free(m_packet);
        m_packet = nullptr;
    }

    if (m_grayscaleFrame)
    {
        if (m_grayscaleFrame->buf[0])
        {
            av_buffer_unref(&m_grayscaleFrame->buf[0]);
        }
        av_frame_free(&m_grayscaleFrame);
    }

    if (m_avFrame)
    {
        av_frame_free(&m_avFrame);
    }

    if (m_codecContext)
    {
        avcodec_free_context(&m_codecContext);
    }

    if (m_formatContext)
    {
        avformat_close_input(&m_formatContext);
    }
}

bool DepthVideoLoader::decodeNextFrame()
{
    if (!m_isOpen.load() || m_eof.load())
        return false;

    // Note: caller must already hold m_frameMutex

    int ret = 0;
    bool frameDecoded = false;

    while (av_read_frame(m_formatContext, m_packet) >= 0)
    {
        if (m_packet->stream_index == m_videoStreamIndex)
        {
            ret = avcodec_send_packet(m_codecContext, m_packet);
            if (ret < 0)
            {
                if (ret != AVERROR(EAGAIN))
                    break;
            }

            while (ret >= 0)
            {
                ret = avcodec_receive_frame(m_codecContext, m_avFrame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                {
                    break;
                }
                else if (ret < 0)
                {
                    LOGE("Error decoding frame\n");
                    break;
                }

                uint8_t* dstData[1] = { m_grayscaleFrame->data[0] };
                int dstLinesize[1] = { m_grayscaleFrame->linesize[0] };

                sws_scale(m_swsContext, m_avFrame->data, m_avFrame->linesize, 0, m_height, dstData, dstLinesize);

                size_t frameSize = m_width * m_height;
                m_currentFrameData.resize(frameSize);
                std::memcpy(m_currentFrameData.data(), m_grayscaleFrame->data[0], frameSize);

                m_currentFrameIndex++;
                m_currentFramePts = m_avFrame->pts * av_q2d(m_formatContext->streams[m_videoStreamIndex]->time_base);
                m_currentTime.store(m_currentFramePts);

                frameDecoded = true;
                break;
            }
        }

        av_packet_unref(m_packet);

        if (frameDecoded)
            break;
    }

    // Flush decoder to get any buffered frames (important for B-frame codecs like HEVC)
    if (!frameDecoded)
    {
        // Send flush packet
        avcodec_send_packet(m_codecContext, nullptr);
        
        while (true)
        {
            ret = avcodec_receive_frame(m_codecContext, m_avFrame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            {
                break;
            }
            else if (ret < 0)
            {
                break;
            }

            uint8_t* dstData[1] = { m_grayscaleFrame->data[0] };
            int dstLinesize[1] = { m_grayscaleFrame->linesize[0] };

            sws_scale(m_swsContext, m_avFrame->data, m_avFrame->linesize, 0, m_height, dstData, dstLinesize);

            size_t frameSize = m_width * m_height;
            m_currentFrameData.resize(frameSize);
            std::memcpy(m_currentFrameData.data(), m_grayscaleFrame->data[0], frameSize);

            m_currentFrameIndex++;
            m_currentFramePts = m_avFrame->pts * av_q2d(m_formatContext->streams[m_videoStreamIndex]->time_base);
            m_currentTime.store(m_currentFramePts);

            frameDecoded = true;
            break;
        }
        
        m_eof.store(true);
    }

    return frameDecoded;
}

void DepthVideoLoader::convertToMetricDepth(const uint8_t* grayscaleData, int width, int height, DepthVideoFrame& outputFrame)
{
    outputFrame.timestampMs = static_cast<uint32_t>(m_currentFramePts * 1000.0);
    outputFrame.zMin = m_metadata.zMin;
    outputFrame.zMax = m_metadata.zMax;

    // Handle side-by-side videos (depth | normals)
    if (m_metadata.sideBySide) {
        // For side-by-side videos, extract only the left half (depth portion)
        int depthWidth = width / 2;
        outputFrame.width = static_cast<uint32_t>(depthWidth);
        outputFrame.height = static_cast<uint32_t>(height);

        size_t numDepthPixels = static_cast<size_t>(depthWidth) * height;
        outputFrame.data.resize(numDepthPixels);

        float zRange = m_metadata.zMax - m_metadata.zMin;

        // Extract left half of the frame (depth data)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < depthWidth; x++) {
                size_t inputIdx = y * width + x;  // Left half
                size_t outputIdx = y * depthWidth + x;
                float normalized = grayscaleData[inputIdx] / 255.0f;
                outputFrame.data[outputIdx] = m_metadata.zMin + normalized * zRange;
            }
        }
    } else {
        // Regular depth-only video
        outputFrame.width = static_cast<uint32_t>(width);
        outputFrame.height = static_cast<uint32_t>(height);

        size_t numPixels = static_cast<size_t>(width) * height;
        outputFrame.data.resize(numPixels);

        float zRange = m_metadata.zMax - m_metadata.zMin;

        for (size_t i = 0; i < numPixels; i++) {
            float normalized = grayscaleData[i] / 255.0f;
            outputFrame.data[i] = m_metadata.zMin + normalized * zRange;
        }
    }
}

#endif // WITH_VIDEO_DECODER

// SynchronizedFrameBuffer implementation

void SynchronizedFrameBuffer::init(double totalDurationSec, uint32_t frameCount, double fps)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_totalDurationSec = totalDurationSec;
    m_totalFrameCount = frameCount;
    m_fps = fps;
    m_frames.clear();
    m_frames.resize(frameCount);
    m_bufferedCount.store(0);
}

void SynchronizedFrameBuffer::clear()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_frames.clear();
    m_bufferedCount.store(0);
    m_totalDurationSec = 0.0;
    m_totalFrameCount = 0;
}

void SynchronizedFrameBuffer::pushFrame(PlaybackFrame&& frame)
{
    size_t index = frame.frameIndex;
    if (index >= m_frames.size())
    {
        LOGW("Frame index %u out of bounds (max %zu)\n", frame.frameIndex, m_frames.size());
        return;
    }
    
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_frames[index] = std::move(frame);
    }
    
    size_t expected = index;
    while (m_bufferedCount.compare_exchange_weak(expected, index + 1))
    {
        if (expected >= index + 1) break;
        expected = index;
    }
    if (expected < index + 1)
    {
        m_bufferedCount.store(index + 1);
    }
}

double SynchronizedFrameBuffer::getBufferedDurationSec() const
{
    size_t count = m_bufferedCount.load();
    if (count == 0 || m_fps <= 0.0) return 0.0;
    return static_cast<double>(count) / m_fps;
}

float SynchronizedFrameBuffer::getBufferedRatio() const
{
    if (m_totalDurationSec <= 0.0) return 0.0f;
    return static_cast<float>(getBufferedDurationSec() / m_totalDurationSec);
}

bool SynchronizedFrameBuffer::isFullyBuffered() const
{
    return m_bufferedCount.load() >= m_totalFrameCount;
}

bool SynchronizedFrameBuffer::getFrameAtTime(double tSec, PlaybackFrame& out) const
{
    if (m_fps <= 0.0) return false;
    
    int64_t frameIndex = static_cast<int64_t>(tSec * m_fps);
    if (frameIndex < 0) frameIndex = 0;
    
    size_t buffered = m_bufferedCount.load();
    if (buffered == 0) return false;
    
    if (static_cast<size_t>(frameIndex) >= buffered)
    {
        frameIndex = static_cast<int64_t>(buffered) - 1;
    }
    
    return getFrameByIndex(static_cast<uint32_t>(frameIndex), out);
}

bool SynchronizedFrameBuffer::getFrameByIndex(uint32_t index, PlaybackFrame& out) const
{
    size_t buffered = m_bufferedCount.load();
    if (index >= buffered) return false;
    
    std::lock_guard<std::mutex> lock(m_mutex);
    if (index >= m_frames.size()) return false;
    
    out = m_frames[index];
    return true;
}

// VideoDepthPlaybackManager implementation

VideoDepthPlaybackManager::~VideoDepthPlaybackManager()
{
    close();
}

    bool VideoDepthPlaybackManager::openFromMetadata(const std::filesystem::path& metadataPath,
                                                     VkInstance instance,
                                                     VkPhysicalDevice physicalDevice,
                                                     VkDevice device,
                                                     uint32_t queueFamilyIndex,
                                                     uint32_t queueIndex)
{
    if (!loadDepthVideoMetadata(metadataPath, m_metadata))
    {
        LOGE("Failed to load metadata from: %s\n", metadataPath.string().c_str());
        return false;
    }

#ifdef WITH_VIDEO_DECODER
    m_videoDecoder = std::make_unique<VideoDecoder>();
    
    // Initialize Vulkan for HW decoding if context provided
    if (device != VK_NULL_HANDLE) {
        m_videoDecoder->initializeVulkan(instance, physicalDevice, device, queueFamilyIndex, queueIndex);
    }
    
    if (!m_videoDecoder->open(m_metadata.videoPath))
    {
        LOGE("Failed to open video: %s\n", m_metadata.videoPath.c_str());
        m_videoDecoder.reset();
        return false;
    }
    m_videoDecoder->startDecoding();

    m_depthLoader = std::make_unique<DepthVideoLoader>();
    if (!m_depthLoader->open(m_metadata.depthVideoPath, m_metadata))
    {
        LOGE("Failed to open depth video: %s\n", m_metadata.depthVideoPath.c_str());
        m_videoDecoder->stopDecoding();
        m_videoDecoder.reset();
        m_depthLoader.reset();
        return false;
    }

    double totalDuration = m_metadata.frameCount > 0 && m_metadata.fps > 0.0 
        ? static_cast<double>(m_metadata.frameCount) / m_metadata.fps 
        : 0.0;
    m_buffer.init(totalDuration, m_metadata.frameCount, m_metadata.fps);

    m_isPlaying.store(false);
    m_paused.store(true);
    m_seekOffsetSec = 0.0;
    m_playbackStartTime = std::chrono::steady_clock::now();
    
    m_stopBuffering.store(false);
    m_bufferingActive.store(true);
    m_prebufferThread = std::thread(&VideoDepthPlaybackManager::prebufferThread, this);

    LOGI("VideoDepthPlaybackManager initialized: %s + %s (buffering %d frames)\n",
         m_metadata.videoPath.c_str(), m_metadata.depthVideoPath.c_str(), m_metadata.frameCount);
    return true;
#else
    LOGE("Video decoder not available - rebuild with ENABLE_VIDEO_DECODER=ON\n");
    return false;
#endif
}

void VideoDepthPlaybackManager::prebufferThread()
{
#ifdef WITH_VIDEO_DECODER
    LOGI("Prebuffer thread started\n");
    
    uint32_t frameIndex = 0;
    while (!m_stopBuffering.load() && frameIndex < static_cast<uint32_t>(m_metadata.frameCount))
    {
        DecodedFrame videoFrame;
        if (!m_videoDecoder->getNextFrame(videoFrame))
        {
            LOGD("Video decoder finished at frame %u\n", frameIndex);
            break;
        }

        DepthVideoFrame depthFrame;
        if (!m_depthLoader->getNextFrame(depthFrame))
        {
            LOGD("Depth loader finished at frame %u\n", frameIndex);
            break;
        }

        PlaybackFrame pf;
        pf.frameIndex = frameIndex;
        pf.timestampSec = videoFrame.timestamp;
        pf.width = static_cast<uint32_t>(videoFrame.width);
        pf.height = static_cast<uint32_t>(videoFrame.height);
        
        if (videoFrame.image != VK_NULL_HANDLE) {
            // Hardware decoded frame
            pf.rgbImage = videoFrame.image;
            pf.rgbFormat = videoFrame.format;
            pf.rgbLayout = videoFrame.layout;
            pf.rgbSemaphore = videoFrame.semaphore;
            pf.hwFrameRef = videoFrame.hwFrameRef; // Keep reference alive
        } else {
            // Software decoded frame
            pf.rgbRGBA = std::move(videoFrame.data);
        }

        pf.depthMeters = std::move(depthFrame.data);
        pf.zMin = m_metadata.zMin;
        pf.zMax = m_metadata.zMax;

        m_buffer.pushFrame(std::move(pf));
        frameIndex++;

        if (frameIndex % 30 == 0)
        {
            LOGD("Buffered %u / %d frames (%.1f%%)\n", 
                 frameIndex, m_metadata.frameCount, 
                 100.0f * m_buffer.getBufferedRatio());
        }
    }

    m_bufferingActive.store(false);
    LOGI("Prebuffer thread finished: %zu frames buffered\n", m_buffer.getBufferedFrameCount());
#endif
}

void VideoDepthPlaybackManager::close()
{
    m_stopBuffering.store(true);
    
    if (m_prebufferThread.joinable())
    {
        m_prebufferThread.join();
    }
    
    m_buffer.clear();

    if (m_depthLoader)
    {
        m_depthLoader->close();
        m_depthLoader.reset();
    }

#ifdef WITH_VIDEO_DECODER
    if (m_videoDecoder)
    {
        m_videoDecoder->stopDecoding();
        m_videoDecoder->close();
        m_videoDecoder.reset();
    }
#endif

    m_isPlaying.store(false);
    m_paused.store(true);
    m_bufferingActive.store(false);
}

void VideoDepthPlaybackManager::play()
{
    std::lock_guard<std::mutex> lock(m_timeMutex);
    
    if (m_paused.load())
    {
        m_playbackStartTime = std::chrono::steady_clock::now() - 
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(m_seekOffsetSec));
    }
    
    m_isPlaying.store(true);
    m_paused.store(false);
}

void VideoDepthPlaybackManager::pause()
{
    std::lock_guard<std::mutex> lock(m_timeMutex);
    
    if (!m_paused.load())
    {
        auto now = std::chrono::steady_clock::now();
        m_seekOffsetSec = std::chrono::duration<double>(now - m_playbackStartTime).count();
    }
    
    m_paused.store(true);
}

void VideoDepthPlaybackManager::togglePlayPause()
{
    if (m_paused.load())
    {
        play();
    }
    else
    {
        pause();
    }
}

void VideoDepthPlaybackManager::seek(double timestamp)
{
    std::lock_guard<std::mutex> lock(m_timeMutex);
    
    double maxTime = m_buffer.getBufferedDurationSec();
    if (timestamp < 0.0) timestamp = 0.0;
    if (timestamp > maxTime) timestamp = maxTime;
    
    m_seekOffsetSec = timestamp;
    m_playbackStartTime = std::chrono::steady_clock::now() - 
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(m_seekOffsetSec));
}

double VideoDepthPlaybackManager::getCurrentTime() const
{
    std::lock_guard<std::mutex> lock(m_timeMutex);
    
    if (m_paused.load())
    {
        return m_seekOffsetSec;
    }
    
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - m_playbackStartTime).count();
    
    double maxTime = m_buffer.getBufferedDurationSec();
    if (elapsed > maxTime) elapsed = maxTime;
    
    return elapsed;
}

bool VideoDepthPlaybackManager::isAtEnd() const
{
    double current = getCurrentTime();
    double buffered = m_buffer.getBufferedDurationSec();
    
    if (!m_bufferingActive.load() && buffered > 0.0)
    {
        return current >= buffered - (1.0 / m_metadata.fps);
    }
    return false;
}

double VideoDepthPlaybackManager::getDuration() const
{
    return m_buffer.getTotalDurationSec();
}

double VideoDepthPlaybackManager::getBufferedDuration() const
{
    return m_buffer.getBufferedDurationSec();
}

float VideoDepthPlaybackManager::getBufferedRatio() const
{
    return m_buffer.getBufferedRatio();
}

bool VideoDepthPlaybackManager::isFullyBuffered() const
{
    return m_buffer.isFullyBuffered();
}

bool VideoDepthPlaybackManager::getFrameAtTime(double tSec, PlaybackFrame& out) const
{
    return m_buffer.getFrameAtTime(tSec, out);
}

} // namespace vk_viewer
