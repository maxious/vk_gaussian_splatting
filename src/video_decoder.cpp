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

#include "video_decoder.h"
#include <nvutils/logger.hpp>
#include <cstring>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
}

namespace vk_viewer {

static enum AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts)
{
    const enum AVPixelFormat *p;
    for (p = pix_fmts; *p != -1; p++) {
        if (*p == AV_PIX_FMT_VULKAN)
            return *p;
    }
    LOGW("Failed to get HW surface format, falling back to software decoding.\n");
    return AV_PIX_FMT_NONE;
}

static void ffmpeg_log_callback(void* ptr, int level, const char* fmt, va_list vl)
{
  char line[1024];
  vsnprintf(line, sizeof(line), fmt, vl);

  size_t len = strlen(line);
  if(len > 0 && line[len - 1] == '\n')
    line[len - 1] = '\0';

  if(level <= AV_LOG_ERROR)
    LOGE("[FFmpeg] %s\n", line);
  else if(level <= AV_LOG_WARNING)
    LOGW("[FFmpeg] %s\n", line);
  else if(level <= AV_LOG_INFO)
    LOGI("[FFmpeg] %s\n", line);
  else
    LOGD("[FFmpeg] %s\n", line);
}

VideoDecoder::VideoDecoder()
    : m_formatContext(nullptr)
    , m_codecContext(nullptr)
    , m_swsContext(nullptr)
    , m_avFrame(nullptr)
    , m_rgbaFrame(nullptr)
    , m_packet(nullptr)
    , m_videoStreamIndex(-1)
    , m_width(0)
    , m_height(0)
    , m_frameRate(0.0)
    , m_duration(0.0)
    , m_running(false)
    , m_stopRequested(false)
    , m_paused(false)
    , m_maxQueueSize(10)
    , m_seekRequested(false)
    , m_seekTimestamp(0.0)
    , m_hwDeviceContext(nullptr)
{
}

void VideoDecoder::initializeVulkan(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device, uint32_t queueFamilyIndex, uint32_t queueIndex)
{
    m_vkInstance = instance;
    m_vkPhysicalDevice = physicalDevice;
    m_vkDevice = device;
    m_vkQueueFamilyIndex = queueFamilyIndex;
    m_vkQueueIndex = queueIndex;
}

bool VideoDecoder::initHWDevice()
{
    int ret = av_hwdevice_ctx_create(&m_hwDeviceContext, AV_HWDEVICE_TYPE_VULKAN, nullptr, nullptr, 0);
    if (ret < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errbuf, sizeof(errbuf));
        LOGE("Failed to create FFmpeg Vulkan HW device: %s\n", errbuf);
        return false;
    }

    LOGI("FFmpeg Vulkan HW context created (Internal Device).\n");
    return true;
}

VideoDecoder::~VideoDecoder()
{
    stopDecoding();
    close();
}

bool VideoDecoder::open(const std::filesystem::path& filepath)
{
    if (m_formatContext) {
        LOGE("Video decoder already open\n");
        return false;
    }

    LOGI("Initializing FFmpeg...\n");
    if (!initializeFFmpeg()) {
        LOGE("Failed to initialize FFmpeg\n");
        return false;
    }

    int ret = avformat_open_input(&m_formatContext, filepath.string().c_str(), nullptr, nullptr);
    if (ret < 0) {
        LOGE("Failed to open input file: %s\n", filepath.string().c_str());
        cleanupFFmpeg();
        return false;
    }

    ret = avformat_find_stream_info(m_formatContext, nullptr);
    if (ret < 0) {
        LOGE("Failed to find stream info\n");
        close();
        return false;
    }

    m_videoStreamIndex = av_find_best_stream(m_formatContext, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (m_videoStreamIndex < 0) {
        LOGE("No video stream found\n");
        close();
        return false;
    }

    AVCodecParameters* codecpar = m_formatContext->streams[m_videoStreamIndex]->codecpar;
    const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
    if (!codec) {
        LOGE("Failed to find codec\n");
        close();
        return false;
    }

    m_codecContext = avcodec_alloc_context3(codec);
    if (!m_codecContext) {
        LOGE("Failed to allocate codec context\n");
        close();
        return false;
    }

    ret = avcodec_parameters_to_context(m_codecContext, codecpar);
    if (ret < 0) {
        LOGE("Failed to copy codec parameters\n");
        close();
        return false;
    }

    ret = avcodec_open2(m_codecContext, codec, nullptr);
    if (ret < 0) {
        LOGE("Failed to open codec\n");
        close();
        return false;
    }

    // FFmpeg Vulkan HW decoding is currently disabled. The integration with the app's
    // Vulkan device context is incomplete - FFmpeg creates its own internal device
    // which conflicts with the app's device, causing synchronization issues and
    // render chain freezes. Using software decoding for now (sws_scale to RGBA).
    // TODO: Properly share Vulkan device context with FFmpeg (requires passing
    // AVVulkanDeviceContext with vkGetInstanceProcAddr, enabled extensions, etc.)
    bool hwInitSuccess = false;
#if 0  // Disabled: FFmpeg Vulkan HW decoding causes render chain freezes
    if (m_vkDevice != VK_NULL_HANDLE) {
        if (initHWDevice()) {
            m_codecContext->hw_device_ctx = av_buffer_ref(m_hwDeviceContext);
            m_codecContext->get_format = get_hw_format;
            hwInitSuccess = true;
            LOGI("FFmpeg HW decoding enabled.\n");
        }
    }
#endif

    m_width = m_codecContext->width;
    m_height = m_codecContext->height;

    AVRational frame_rate = av_guess_frame_rate(m_formatContext, m_formatContext->streams[m_videoStreamIndex], nullptr);
    m_frameRate = av_q2d(frame_rate);

    if (m_formatContext->duration != AV_NOPTS_VALUE) {
        m_duration = m_formatContext->duration * av_q2d(AV_TIME_BASE_Q);
    }

    if (!hwInitSuccess) {
        m_swsContext = sws_getContext(
            m_width, m_height, m_codecContext->pix_fmt,
            m_width, m_height, AV_PIX_FMT_RGBA,
            SWS_BILINEAR, nullptr, nullptr, nullptr
        );

        if (!m_swsContext) {
            LOGE("Failed to create scaling context\n");
            close();
            return false;
        }
        
        int num_bytes = av_image_get_buffer_size(AV_PIX_FMT_RGBA, m_width, m_height, 1);
        uint8_t* buffer = (uint8_t*)av_malloc(num_bytes * sizeof(uint8_t));
        if (!buffer) {
            LOGE("Failed to allocate RGBA buffer\n");
            close();
            return false;
        }

        m_rgbaFrame = av_frame_alloc();
        if (!m_rgbaFrame) {
             LOGE("Failed to allocate RGBA frame\n");
             av_free(buffer);
             close();
             return false;
        }

        av_image_fill_arrays(m_rgbaFrame->data, m_rgbaFrame->linesize, buffer,
                             AV_PIX_FMT_RGBA, m_width, m_height, 1);
    }

    m_avFrame = av_frame_alloc();
    m_packet = av_packet_alloc();

    if (!m_avFrame || !m_packet) {
        LOGE("Failed to allocate frames/packet\n");
        close();
        return false;
    }

    LOGI("Video decoder opened: %dx%d @ %.2f fps, duration: %.2f s%s\n",
         m_width, m_height, m_frameRate, m_duration, hwInitSuccess ? " (HW Accelerated)" : "");

    return true;
}

void VideoDecoder::close()
{
    stopDecoding();
    cleanupFFmpeg();
}

void VideoDecoder::startDecoding()
{
    if (m_running || !m_formatContext) {
        return;
    }

    if (m_decodeThread.joinable()) {
        m_decodeThread.join();
    }

    m_running = true;
    m_stopRequested = false;
    m_seekRequested = false;
    m_paused = false;

    m_decodeThread = std::thread(&VideoDecoder::decodingThread, this);
}

void VideoDecoder::stopDecoding()
{
    m_stopRequested = true;
    m_running = false;
    m_paused = false;
    m_pauseCondition.notify_one();

    if (m_decodeThread.joinable()) {
        m_decodeThread.join();
    }
}

bool VideoDecoder::getNextFrame(DecodedFrame& frame)
{
    std::unique_lock<std::mutex> lock(m_queueMutex);
    m_queueCondition.wait(lock, [this]() {
        return !m_frameQueue.empty() || !m_running;
    });

    if (m_frameQueue.empty()) {
        return false; 
    }

    frame = std::move(m_frameQueue.front());
    m_frameQueue.erase(m_frameQueue.begin());

    return true;
}

bool VideoDecoder::seekToTime(double timestamp)
{
    if (!m_formatContext) {
        return false;
    }

    bool wasRunning = m_running.load();
    if (!wasRunning) {
        AVRational time_base = m_formatContext->streams[m_videoStreamIndex]->time_base;
        int64_t seek_pts = static_cast<int64_t>(timestamp / av_q2d(time_base));
        
        int ret = av_seek_frame(m_formatContext, m_videoStreamIndex, seek_pts, AVSEEK_FLAG_BACKWARD);
        if (ret >= 0) {
            avcodec_flush_buffers(m_codecContext);
            LOGI("Seeked to timestamp: %.2f s (restarting decoder)\n", timestamp);
        } else {
            LOGE("Seek failed\n");
            return false;
        }
        
        {
            std::lock_guard<std::mutex> lock(m_queueMutex);
            m_frameQueue.clear();
        }
        
        startDecoding();
        return true;
    }

    m_seekRequested = true;
    m_seekTimestamp = timestamp;

    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        m_frameQueue.clear();
    }
    m_queueCondition.notify_all();
    
    if (m_paused) {
        m_paused = false;
        m_pauseCondition.notify_one();
    }

    return true;
}

double VideoDecoder::getDuration() const
{
    return m_duration;
}

double VideoDecoder::getFrameRate() const
{
    return m_frameRate;
}

void VideoDecoder::getDimensions(int& width, int& height) const
{
    width = m_width;
    height = m_height;
}

double VideoDecoder::getCurrentTime() const
{
    return 0.0;
}

void VideoDecoder::pause()
{
    m_paused = true;
}

void VideoDecoder::resume()
{
    m_paused = false;
    m_pauseCondition.notify_one();
}

void VideoDecoder::decodingThread()
{
    LOGI("Video decoding thread started\n");

    while (m_running && !m_stopRequested) {
        if (m_paused) {
            std::unique_lock<std::mutex> lock(m_pauseMutex);
            m_pauseCondition.wait(lock, [this]() {
                return !m_paused || m_stopRequested;
            });
            if (m_stopRequested) break;
        }

        if (m_seekRequested) {
            double seek_ts = m_seekTimestamp;
            m_seekRequested = false;

            AVRational time_base = m_formatContext->streams[m_videoStreamIndex]->time_base;
            int64_t seek_pts = static_cast<int64_t>(seek_ts / av_q2d(time_base));

            int ret = av_seek_frame(m_formatContext, m_videoStreamIndex, seek_pts, AVSEEK_FLAG_BACKWARD);
            if (ret >= 0) {
                avcodec_flush_buffers(m_codecContext);
                LOGI("Seeked to timestamp: %.2f s\n", seek_ts);
            } else {
                LOGE("Seek failed\n");
            }
        }

        decodeFrames();
    }

    LOGI("Video decoding thread stopped\n");
}

void VideoDecoder::decodeFrames()
{
    while (m_running && !m_stopRequested) {
        {
            std::lock_guard<std::mutex> lock(m_queueMutex);
            if (m_frameQueue.size() >= m_maxQueueSize) {
                return; 
            }
        }

        int ret = av_read_frame(m_formatContext, m_packet);
        if (ret < 0) {
            if (ret == AVERROR_EOF) {
                avcodec_send_packet(m_codecContext, nullptr);
                while (true) {
                    ret = avcodec_receive_frame(m_codecContext, m_avFrame);
                    if (ret == AVERROR_EOF) {
                        break;
                    } else if (ret >= 0) {
                        processFrame();
                    }
                }
                m_running = false; 
            }
            break;
        }

        if (m_packet->stream_index == m_videoStreamIndex) {
            ret = avcodec_send_packet(m_codecContext, m_packet);
            if (ret < 0) {
                LOGE("Error sending packet to decoder\n");
                continue;
            }

            while (true) {
                ret = avcodec_receive_frame(m_codecContext, m_avFrame);
                if (ret == AVERROR(EAGAIN)) {
                    break;
                } else if (ret == AVERROR_EOF) {
                    m_running = false;
                    break;
                } else if (ret < 0) {
                    LOGE("Error receiving frame from decoder\n");
                    break;
                } else {
                    processFrame();
                }
            }
        }

        av_packet_unref(m_packet);
    }
}

void VideoDecoder::processFrame()
{
    DecodedFrame frame;
    frame.width = m_width;
    frame.height = m_height;
    frame.pts = m_avFrame->pts;

    AVRational time_base = m_formatContext->streams[m_videoStreamIndex]->time_base;
    frame.timestamp = m_avFrame->pts * av_q2d(time_base);

    if (m_avFrame->format == AV_PIX_FMT_VULKAN) {
        AVVkFrame* vkFrame = (AVVkFrame*)m_avFrame->data[0];
        frame.image = vkFrame->img[0];
        frame.layout = vkFrame->layout[0];
        frame.format = VK_FORMAT_UNDEFINED; 
        frame.semaphore = vkFrame->sem[0];
        
        AVFrame* clonedFrame = av_frame_clone(m_avFrame);
        if (clonedFrame) {
            frame.hwFrameRef = std::shared_ptr<void>(clonedFrame, [](void* p) {
                AVFrame* f = static_cast<AVFrame*>(p);
                av_frame_free(&f);
            });
        }
    } else {
        if (m_swsContext && m_rgbaFrame->data[0]) {
            sws_scale(m_swsContext, m_avFrame->data, m_avFrame->linesize,
                      0, m_height, m_rgbaFrame->data, m_rgbaFrame->linesize);

            int num_bytes = av_image_get_buffer_size(AV_PIX_FMT_RGBA, m_width, m_height, 1);
            frame.data.resize(num_bytes);
            memcpy(frame.data.data(), m_rgbaFrame->data[0], num_bytes);
        }
    }

    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        m_frameQueue.push_back(std::move(frame));
    }
    m_queueCondition.notify_one();
}

bool VideoDecoder::initializeFFmpeg()
{
  av_log_set_callback(ffmpeg_log_callback);
  av_log_set_level(AV_LOG_DEBUG);
  return true;
}

void VideoDecoder::cleanupFFmpeg()
{
    if (m_packet) {
        av_packet_free(&m_packet);
        m_packet = nullptr;
    }

    if (m_rgbaFrame) {
        if (m_rgbaFrame->data[0]) {
            av_free(m_rgbaFrame->data[0]);
        }
        av_frame_free(&m_rgbaFrame);
        m_rgbaFrame = nullptr;
    }

    if (m_avFrame) {
        av_frame_free(&m_avFrame);
        m_avFrame = nullptr;
    }

    if (m_swsContext) {
        sws_freeContext(m_swsContext);
        m_swsContext = nullptr;
    }

    if (m_codecContext) {
        avcodec_free_context(&m_codecContext);
        m_codecContext = nullptr;
    }

    if (m_hwDeviceContext) {
        av_buffer_unref(&m_hwDeviceContext);
    }

    m_videoStreamIndex = -1;
    m_width = 0;
    m_height = 0;
    m_frameRate = 0.0;
    m_duration = 0.0;
}

} // namespace vk_viewer
