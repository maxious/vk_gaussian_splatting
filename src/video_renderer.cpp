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

#include "video_renderer.h"
#include <nvutils/logger.hpp>
#include <sstream>
#include <iomanip>
#include <cstdlib>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#endif

namespace vk_gaussian_splatting {

VideoRenderer::~VideoRenderer()
{
  cancelRender();
}

bool VideoRenderer::isFFmpegAvailable()
{
#ifdef _WIN32
  int result = std::system("where ffmpeg >nul 2>&1");
#else
  int result = std::system("which ffmpeg >/dev/null 2>&1");
#endif
  return result == 0;
}

std::string VideoRenderer::getFFmpegPath()
{
#ifdef _WIN32
  FILE* pipe = _popen("where ffmpeg 2>nul", "r");
#else
  FILE* pipe = popen("which ffmpeg 2>/dev/null", "r");
#endif

  if(!pipe)
    return "";

  char        buffer[256];
  std::string result;
  if(fgets(buffer, sizeof(buffer), pipe) != nullptr)
  {
    result = buffer;
    while(!result.empty() && (result.back() == '\n' || result.back() == '\r'))
      result.pop_back();
  }

#ifdef _WIN32
  _pclose(pipe);
#else
  pclose(pipe);
#endif

  return result;
}

int VideoRenderer::getFFmpegMajorVersion()
{
#ifdef _WIN32
  FILE* pipe = _popen("ffmpeg -version 2>nul", "r");
#else
  FILE* pipe = popen("ffmpeg -version 2>/dev/null", "r");
#endif

  if(!pipe)
    return 0;

  char buffer[512];
  int  majorVersion = 0;

  if(fgets(buffer, sizeof(buffer), pipe) != nullptr)
  {
    std::string line = buffer;
    size_t      pos  = line.find("ffmpeg version ");
    if(pos != std::string::npos)
    {
      pos += 15;  // Skip "ffmpeg version "
      std::string versionStr;
      while(pos < line.size() && std::isdigit(line[pos]))
      {
        versionStr += line[pos];
        pos++;
      }
      if(!versionStr.empty())
      {
        majorVersion = std::stoi(versionStr);
      }
    }
  }

#ifdef _WIN32
  _pclose(pipe);
#else
  pclose(pipe);
#endif

  return majorVersion;
}

bool VideoRenderer::supportsHDR10Encoding()
{
  return getFFmpegMajorVersion() >= 6;
}

bool VideoRenderer::supportsNVENC()
{
  static int cached = -1;
  if(cached >= 0)
    return cached == 1;

#ifdef _WIN32
  FILE* pipe = _popen("ffmpeg -encoders 2>nul", "r");
#else
  FILE* pipe = popen("ffmpeg -encoders 2>/dev/null", "r");
#endif

  if(!pipe)
  {
    cached = 0;
    return false;
  }

  char buffer[512];
  bool found = false;

  while(fgets(buffer, sizeof(buffer), pipe) != nullptr)
  {
    std::string line = buffer;
    if(line.find("hevc_nvenc") != std::string::npos)
    {
      found = true;
      break;
    }
  }

#ifdef _WIN32
  _pclose(pipe);
#else
  pclose(pipe);
#endif

  cached = found ? 1 : 0;
  return found;
}

void VideoRenderer::startRender(const VideoRenderSettings&                        settings,
                                const Camera&                                      startCamera,
                                const std::vector<Camera>&                         keyframes,
                                std::function<void(const Camera&, int frameIndex)> renderFrameCallback,
                                std::function<void(const std::filesystem::path&)>  saveFrameCallback)
{
  if(m_state == VideoRenderState::STATE_RENDERING || m_state == VideoRenderState::STATE_ENCODING)
  {
    return;
  }

  m_settings         = settings;
  m_renderCallback   = std::move(renderFrameCallback);
  m_saveCallback     = std::move(saveFrameCallback);
  m_currentFrame     = 0;
  m_framesSaved      = 0;
  m_waitCheckCounter = 0;
  m_errorMessage.clear();

  if(!std::filesystem::exists(m_settings.outputDir))
  {
    std::filesystem::create_directories(m_settings.outputDir);
  }

  generateTrajectory(settings, startCamera, keyframes);

  if(m_trajectory.empty())
  {
    m_state        = VideoRenderState::STATE_ERROR;
    m_errorMessage = "Failed to generate camera trajectory";
    return;
  }

  m_state         = VideoRenderState::STATE_RENDERING;
  m_statusMessage = "Rendering frame 0/" + std::to_string(m_trajectory.size());
}

void VideoRenderer::cancelRender()
{
  if(m_state == VideoRenderState::STATE_RENDERING || m_state == VideoRenderState::STATE_ENCODING)
  {
    m_state         = VideoRenderState::STATE_CANCELLED;
    m_statusMessage = "Render cancelled";
  }
}

VideoRenderProgress VideoRenderer::getProgress() const
{
  VideoRenderProgress progress;
  progress.state         = m_state.load();
  progress.currentFrame  = m_currentFrame;
  progress.totalFrames   = static_cast<int>(m_trajectory.size());
  progress.statusMessage = m_statusMessage;
  progress.errorMessage  = m_errorMessage;

  if(progress.totalFrames > 0)
  {
    progress.progressPct = static_cast<float>(m_currentFrame) / static_cast<float>(progress.totalFrames) * 100.0f;
  }

  return progress;
}

void VideoRenderer::renderNextFrame()
{
  if(m_state != VideoRenderState::STATE_RENDERING)
    return;

  if(m_currentFrame >= static_cast<int>(m_trajectory.size()))
  {
    m_state         = VideoRenderState::STATE_WAITING_FRAMES;
    m_statusMessage = "Waiting for frames to save...";
    return;
  }

  const Camera& cam = m_trajectory[m_currentFrame];

  if(m_renderCallback)
  {
    m_renderCallback(cam, m_currentFrame);
  }

  if(m_saveCallback)
  {
    auto framePath = m_settings.outputDir / getFrameFilename(m_currentFrame, m_settings.outputFormat);
    m_saveCallback(framePath);
  }

  m_currentFrame++;
  m_statusMessage = "Rendering frame " + std::to_string(m_currentFrame) + "/" + std::to_string(m_trajectory.size());
}

void VideoRenderer::notifyFrameSaved(int frameIndex)
{
  m_framesSaved++;
}

void VideoRenderer::checkFramesComplete()
{
  if(m_state != VideoRenderState::STATE_WAITING_FRAMES)
    return;

  int totalFrames = static_cast<int>(m_trajectory.size());

  // Frames are saved synchronously, so proceed to encoding immediately
  if(m_settings.encodeVideo && isFFmpegAvailable())
  {
    m_state         = VideoRenderState::STATE_ENCODING;
    m_statusMessage = "Encoding video with FFmpeg...";

    if(encodeVideo(m_settings))
    {
      m_state         = VideoRenderState::STATE_COMPLETED;
      m_statusMessage = "Video render completed!";

      if(m_settings.deleteFramesAfterEncode)
      {
        for(int i = 0; i < totalFrames; ++i)
        {
          auto framePath = m_settings.outputDir / getFrameFilename(i, m_settings.outputFormat);
          std::filesystem::remove(framePath);
        }
      }
    }
    else
    {
      m_state = VideoRenderState::STATE_ERROR;
    }
  }
  else
  {
    m_state         = VideoRenderState::STATE_COMPLETED;
    m_statusMessage = "Frame sequence saved. ";
    if(!isFFmpegAvailable())
    {
      m_statusMessage += "FFmpeg not found - run manually:\n" + buildFFmpegCommand(m_settings, totalFrames);
    }
  }
}

bool VideoRenderer::encodeVideo(const VideoRenderSettings& settings)
{
  bool isHDR = (settings.outputFormat == VideoOutputFormat::FORMAT_HDR);
  
  if(isHDR)
  {
    int ffmpegVersion = getFFmpegMajorVersion();
    if(ffmpegVersion >= 6)
    {
      LOGI("Encoding HDR10 video with FFmpeg %d (BT.2020/PQ)\n", ffmpegVersion);
    }
    else
    {
      LOGW("FFmpeg version %d detected. HDR10 encoding requires FFmpeg 6+. Output will be SDR.\n", ffmpegVersion);
    }
  }

  std::string cmd = buildFFmpegCommand(settings, static_cast<int>(m_trajectory.size()));

  LOGI("Running FFmpeg command:\n%s\n", cmd.c_str());

  int result = std::system(cmd.c_str());

  if(result != 0)
  {
    m_errorMessage = "FFmpeg encoding failed with code " + std::to_string(result);
    LOGE("FFmpeg error (code %d). Command: %s\n", result, cmd.c_str());
    return false;
  }

  LOGI("FFmpeg encoding completed successfully\n");
  return true;
}

std::string VideoRenderer::buildFFmpegCommand(const VideoRenderSettings& settings, int totalFrames)
{
  std::stringstream ss;

  std::string extension    = settings.outputFormat == VideoOutputFormat::FORMAT_HDR ? ".hdr" : ".png";
  std::string inputPattern = (settings.outputDir / ("frame_%04d" + extension)).string();

  std::string outputPath = (settings.outputDir / (settings.outputName + ".mp4")).string();

  bool isHDR         = (settings.outputFormat == VideoOutputFormat::FORMAT_HDR);
  bool canEncodeHDR  = isHDR && supportsHDR10Encoding();

  ss << "ffmpeg -y -loglevel warning ";
  ss << "-framerate " << settings.frameRate << " ";
  ss << "-i \"" << inputPattern << "\" ";

  if(canEncodeHDR)
  {
    // HDR10 encoding: use zscale to convert linear BT.709 to BT.2020 with PQ transfer
    // tin=linear, pin=bt709 = input is linear BT.709 (from .hdr files)
    // t=smpte2084, p=bt2020, m=bt2020nc = output is HDR10 (BT.2020 + PQ)
    // Also pad to even dimensions (required by H.265)
    ss << "-vf \"zscale=tin=linear:pin=bt709:t=smpte2084:p=bt2020:m=bt2020nc,pad=ceil(iw/2)*2:ceil(ih/2)*2\" ";

    // Use NVENC HEVC for HDR if an NVENC codec is selected, otherwise fall back to libx265
    // Note: HDR requires HEVC, so NVENC H.264 selections also use HEVC for HDR
    bool useNvenc = (settings.codec == VideoCodec::CODEC_NVENC_HEVC_HQ || settings.codec == VideoCodec::CODEC_NVENC_H264_HQ
                     || settings.codec == VideoCodec::CODEC_NVENC_HEVC_LOSSLESS
                     || settings.codec == VideoCodec::CODEC_NVENC_H264_LOSSLESS);
    bool useLossless = (settings.codec == VideoCodec::CODEC_NVENC_HEVC_LOSSLESS
                        || settings.codec == VideoCodec::CODEC_NVENC_H264_LOSSLESS);

    if(useNvenc)
    {
      if(useLossless)
      {
        ss << "-c:v hevc_nvenc -preset p7 -tune lossless -rc constqp -qp 0 ";
      }
      else
      {
        ss << "-c:v hevc_nvenc -preset p7 -tune hq -rc vbr -cq 18 -b:v 0 -spatial-aq 1 -aq-strength 8 ";
      }
    }
    else
    {
      ss << "-c:v libx265 -preset slow -x265-params \"lossless=1:hdr-opt=1:repeat-headers=1:max-cll=1000,400\" ";
    }
    ss << "-pix_fmt yuv420p10le ";
    ss << "-color_primaries bt2020 ";
    ss << "-color_trc smpte2084 ";
    ss << "-colorspace bt2020nc ";
  }
  else
  {
    // Pad to even dimensions (required by H.264/H.265)
    ss << "-vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" ";
    switch(settings.codec)
    {
      case VideoCodec::CODEC_NVENC_HEVC_HQ:
        ss << "-c:v hevc_nvenc -preset p7 -tune hq -rc vbr -cq 18 -b:v 0 -spatial-aq 1 -aq-strength 8 ";
        break;
      case VideoCodec::CODEC_NVENC_H264_HQ:
        ss << "-c:v h264_nvenc -preset p7 -tune hq -rc vbr -cq 18 -b:v 0 -spatial-aq 1 -aq-strength 8 ";
        break;
      case VideoCodec::CODEC_NVENC_HEVC_LOSSLESS:
        ss << "-c:v hevc_nvenc -preset p7 -tune lossless -rc constqp -qp 0 ";
        break;
      case VideoCodec::CODEC_NVENC_H264_LOSSLESS:
        ss << "-c:v h264_nvenc -preset p7 -tune lossless -rc constqp -qp 0 ";
        break;
      case VideoCodec::CODEC_H264:
        ss << "-c:v libx264 -crf 0 -preset veryslow ";
        break;
      case VideoCodec::CODEC_H265:
        ss << "-c:v libx265 -preset slow -x265-params lossless=1 ";
        break;
      case VideoCodec::CODEC_PRORES:
        ss << "-c:v prores_ks -profile:v 4 ";  // ProRes 4444 for highest quality
        break;
      default:
        ss << "-c:v hevc_nvenc -preset p7 -tune hq -rc vbr -cq 18 -b:v 0 -spatial-aq 1 -aq-strength 8 ";
        break;
    }
    ss << "-pix_fmt yuv420p ";
  }

  ss << "\"" << outputPath << "\"";

  return ss.str();
}

void VideoRenderer::generateTrajectory(const VideoRenderSettings& settings,
                                        const Camera&              startCamera,
                                        const std::vector<Camera>& keyframes)
{
  int totalFrames = settings.getTotalFrames();

  TrajectoryParams params = settings.trajectory;
  params.numFrames        = totalFrames;

  m_trajectory = CameraTrajectory::generate(startCamera, params);
}

std::string VideoRenderer::getFrameFilename(int frameIndex, VideoOutputFormat format) const
{
  std::stringstream ss;
  ss << "frame_" << std::setfill('0') << std::setw(4) << frameIndex;

  switch(format)
  {
    case VideoOutputFormat::FORMAT_HDR:
      ss << ".hdr";
      break;
    case VideoOutputFormat::FORMAT_PNG:
    default:
      ss << ".png";
      break;
  }

  return ss.str();
}

}  // namespace vk_gaussian_splatting
