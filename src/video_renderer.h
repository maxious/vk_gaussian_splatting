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

#ifndef _VIDEO_RENDERER_H_
#define _VIDEO_RENDERER_H_

#include <string>
#include <vector>
#include <filesystem>
#include <functional>
#include <atomic>

#include "camera_set.h"
#include "camera_trajectory.h"

namespace vk_gaussian_splatting {

enum class VideoOutputFormat
{
  FORMAT_PNG,  // Lossless PNG sequence
  FORMAT_HDR,  // Radiance HDR (high dynamic range)
};

enum class VideoCodec
{
  CODEC_NVENC_HEVC_HQ,        // NVIDIA NVENC HEVC high quality (GPU, default)
  CODEC_NVENC_H264_HQ,        // NVIDIA NVENC H.264 high quality (GPU)
  CODEC_NVENC_HEVC_LOSSLESS,  // NVIDIA NVENC HEVC lossless (GPU)
  CODEC_NVENC_H264_LOSSLESS,  // NVIDIA NVENC H.264 lossless (GPU)
  CODEC_H264,                 // H.264 lossless (CPU)
  CODEC_H265,                 // H.265/HEVC lossless (CPU)
  CODEC_PRORES,               // Apple ProRes 4444
};

struct VideoRenderSettings
{
  TrajectoryParams trajectory;

  int   frameRate      = 30;
  float durationSec    = 10.0f;
  int   width          = 1920;
  int   height         = 1080;

  bool  enableSBS      = false;
  float stereoIPD      = 0.063f;
  float stereoConvergence = 1.0f;
  bool  stereoOffAxis  = true;

  VideoOutputFormat outputFormat = VideoOutputFormat::FORMAT_PNG;
  VideoCodec        codec        = VideoCodec::CODEC_NVENC_HEVC_HQ;

  std::filesystem::path outputDir;
  std::string           outputName = "video";

  bool deleteFramesAfterEncode = false;
  bool encodeVideo             = true;

  int getTotalFrames() const { return static_cast<int>(frameRate * durationSec); }
};

enum class VideoRenderState
{
  STATE_IDLE,
  STATE_RENDERING,
  STATE_WAITING_FRAMES,  // Waiting for async screenshots to complete
  STATE_ENCODING,
  STATE_COMPLETED,
  STATE_ERROR,
  STATE_CANCELLED,
};

struct VideoRenderProgress
{
  VideoRenderState state          = VideoRenderState::STATE_IDLE;
  int              currentFrame   = 0;
  int              totalFrames    = 0;
  float            progressPct    = 0.0f;
  std::string      statusMessage;
  std::string      errorMessage;
};

class VideoRenderer
{
public:
  VideoRenderer() = default;
  ~VideoRenderer();

  static bool isFFmpegAvailable();

  static std::string getFFmpegPath();

  static int getFFmpegMajorVersion();

  static bool supportsHDR10Encoding();

  static bool supportsNVENC();

  void startRender(const VideoRenderSettings&                        settings,
                   const Camera&                                      startCamera,
                   const std::vector<Camera>&                         keyframes,
                   std::function<void(const Camera&, int frameIndex)> renderFrameCallback,
                   std::function<void(const std::filesystem::path&)>  saveFrameCallback);

  void cancelRender();

  bool isRendering() const { return m_state == VideoRenderState::STATE_RENDERING || m_state == VideoRenderState::STATE_WAITING_FRAMES || m_state == VideoRenderState::STATE_ENCODING; }

  VideoRenderProgress getProgress() const;

  void renderNextFrame();

  void checkFramesComplete();

  void notifyFrameSaved(int frameIndex);

  bool encodeVideo(const VideoRenderSettings& settings);

  const std::vector<Camera>& getTrajectory() const { return m_trajectory; }

  static std::string buildFFmpegCommand(const VideoRenderSettings& settings, int totalFrames);

private:
  void generateTrajectory(const VideoRenderSettings& settings,
                          const Camera&              startCamera,
                          const std::vector<Camera>& keyframes);

  std::string getFrameFilename(int frameIndex, VideoOutputFormat format) const;

  std::atomic<VideoRenderState> m_state{VideoRenderState::STATE_IDLE};
  VideoRenderSettings           m_settings;
  std::vector<Camera>           m_trajectory;
  int                           m_currentFrame = 0;
  std::string                   m_statusMessage;
  std::string                   m_errorMessage;

  std::function<void(const Camera&, int)>           m_renderCallback;
  std::function<void(const std::filesystem::path&)> m_saveCallback;

  std::atomic<int> m_framesSaved{0};
  int              m_waitCheckCounter{0};
};

}  // namespace vk_gaussian_splatting

#endif
