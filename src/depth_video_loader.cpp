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

namespace vk_gaussian_splatting {

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

    outMetadata.depthWidth = outMetadata.sourceWidth;
    outMetadata.depthHeight = outMetadata.sourceHeight;

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
        {
            return false;
        }
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

    std::lock_guard<std::mutex> lock(m_frameMutex);

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

                const uint8_t* srcData[1] = { m_avFrame->data[0] };
                int srcLinesize[1] = { m_avFrame->linesize[0] };
                uint8_t* dstData[1] = { m_grayscaleFrame->data[0] };
                int dstLinesize[1] = { m_grayscaleFrame->linesize[0] };

                sws_scale(m_swsContext, srcData, srcLinesize, 0, m_height, dstData, dstLinesize);

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

    if (!frameDecoded && ret == AVERROR_EOF)
    {
        m_eof.store(true);
    }

    return frameDecoded;
}

void DepthVideoLoader::convertToMetricDepth(const uint8_t* grayscaleData, int width, int height, DepthVideoFrame& outputFrame)
{
    outputFrame.timestampMs = static_cast<uint32_t>(m_currentFramePts * 1000.0);
    outputFrame.width = static_cast<uint32_t>(width);
    outputFrame.height = static_cast<uint32_t>(height);
    outputFrame.zMin = m_metadata.zMin;
    outputFrame.zMax = m_metadata.zMax;

    size_t numPixels = static_cast<size_t>(width) * height;
    outputFrame.data.resize(numPixels);

    float zRange = m_metadata.zMax - m_metadata.zMin;

    for (size_t i = 0; i < numPixels; i++)
    {
        float normalized = grayscaleData[i] / 255.0f;
        outputFrame.data[i] = m_metadata.zMin + normalized * zRange;
    }
}

#endif // WITH_VIDEO_DECODER

// VideoDepthPlaybackManager implementation

VideoDepthPlaybackManager::~VideoDepthPlaybackManager()
{
    close();
}

bool VideoDepthPlaybackManager::openFromMetadata(const std::filesystem::path& metadataPath)
{
    if (!loadDepthVideoMetadata(metadataPath, m_metadata))
    {
        LOGE("Failed to load metadata from: %s\n", metadataPath.string().c_str());
        return false;
    }

#ifdef WITH_VIDEO_DECODER
    m_videoDecoder = std::make_unique<VideoDecoder>();
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

    m_isPlaying.store(false);
    m_paused.store(true);
    m_currentTime.store(0.0);

    LOGI("VideoDepthPlaybackManager initialized: %s + %s\n",
         m_metadata.videoPath.c_str(), m_metadata.depthVideoPath.c_str());
    return true;
#else
    LOGE("Video decoder not available - rebuild with ENABLE_VIDEO_DECODER=ON\n");
    return false;
#endif
}

void VideoDepthPlaybackManager::close()
{
    if (m_depthLoader)
    {
        m_depthLoader->close();
        m_depthLoader.reset();
    }

    if (m_videoDecoder)
    {
        m_videoDecoder->stopDecoding();
        m_videoDecoder->close();
        m_videoDecoder.reset();
    }

    m_isPlaying.store(false);
    m_paused.store(false);
}

void VideoDepthPlaybackManager::play()
{
    if (m_videoDecoder && !m_paused.load())
    {
        m_videoDecoder->resume();
    }
    m_isPlaying.store(true);
    m_paused.store(false);
}

void VideoDepthPlaybackManager::pause()
{
    if (m_videoDecoder)
    {
        m_videoDecoder->pause();
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
    if (m_depthLoader)
    {
        m_depthLoader->seekToTime(timestamp);
    }
    if (m_videoDecoder)
    {
        m_videoDecoder->seekToTime(timestamp);
    }
    m_currentTime.store(timestamp);
}

} // namespace vk_gaussian_splatting
