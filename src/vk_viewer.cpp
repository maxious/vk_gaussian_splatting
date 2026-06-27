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

// Vulkan Memory Allocator
#define VMA_IMPLEMENTATION
#define VMA_LEAK_LOG_FORMAT(format, ...)                                                                               \
  {                                                                                                                    \
    printf((format), __VA_ARGS__);                                                                                     \
    printf("\n");                                                                                                      \
  }

#include "vk_viewer.h"
#include "hdr_support.h"
#include "utilities.h"
#include "animation_controller.h"
#include "animation_ui.h"
#include "depth_video_loader.h"

#include <nvutils/logger.hpp>
#include <nvvk/barriers.hpp>

#define GLM_ENABLE_SWIZZLE
#include <glm/gtc/packing.hpp>  // Required for half-float operations

#include <nvvk/check_error.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/sbt_generator.hpp>
#include <nvvk/formats.hpp>

namespace vk_viewer {

VkViewer::VkViewer(nvutils::ProfilerManager* profilerManager, nvutils::ParameterRegistry* parameterRegistry)
    : m_profilerManager(profilerManager)
    , m_parameterRegistry(parameterRegistry)
    , cameraManip(std::make_shared<nvutils::CameraManipulator>()) {

    };

VkViewer::~VkViewer(){
    // all threads must be stopped,
    // work done in onDetach(),
    // could be done here, same result
};

void VkViewer::onAttach(nvapp::Application* app)
{
  if (m_attached)
    return;
  m_attached = true;

  // shortcuts
  m_app    = app;
  m_device = m_app->getDevice();

  // profiling
  m_profilerTimeline = m_profilerManager->createTimeline({.name = "Primary Timeline"});
  m_profilerGpuTimer.init(m_profilerTimeline, m_app->getDevice(), m_app->getPhysicalDevice(), m_app->getQueue(0).familyIndex, false);

  // starts the asynchronous services
  m_splatLoader.initialize();
  m_cpuSorter.initialize(m_profilerTimeline);

  // Memory allocator
  m_alloc.init(VmaAllocatorCreateInfo{
      .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
      .physicalDevice   = app->getPhysicalDevice(),
      .device           = app->getDevice(),
      .instance         = app->getInstance(),
      .vulkanApiVersion = VK_API_VERSION_1_4,
  });

  // DEBUG: uncomment and set id to find object leak
  // m_alloc.setLeakID(70);

  // set up buffer uploading utility
  m_uploader.init(&m_alloc, true);

  // Acquiring the sampler which will be used for displaying the GBuffer and accessing textures
  m_samplerPool.init(app->getDevice());
  NVVK_CHECK(m_samplerPool.acquireSampler(m_sampler));
  NVVK_DBG_NAME(m_sampler);

  // GBuffer
  m_depthFormat = nvvk::findDepthFormat(app->getPhysicalDevice());

  // Color attachments:
  // - COLOR_MAIN: main output
  // - COLOR_AUX1: temporal sampling with 3DGUT
  // - COLOR_MOTION: RG motion vectors (for Space Warp & DLSS)
  // - DLSS-RR buffers (when enabled): diffuse albedo, specular albedo, normal+roughness, 
  //   linear depth, specular hit distance, DLSS output
  std::vector<VkFormat> colorFormats = {m_colorFormat, m_colorFormat};
  colorFormats.push_back(VK_FORMAT_R16G16_SFLOAT); // COLOR_MOTION (index 2)

#ifdef WITH_DLSS_RR
  colorFormats.push_back(VK_FORMAT_R16G16B16A16_SFLOAT);  // COLOR_DLSS_DIFFUSE_ALBEDO
  colorFormats.push_back(VK_FORMAT_R16G16B16A16_SFLOAT);  // COLOR_DLSS_SPECULAR_ALBEDO
  colorFormats.push_back(VK_FORMAT_R16G16B16A16_SFLOAT);  // COLOR_DLSS_NORMAL_ROUGH
  // COLOR_DLSS_MOTION is index 2, already added
  colorFormats.push_back(VK_FORMAT_R32_SFLOAT);           // COLOR_DLSS_LINEAR_DEPTH
  colorFormats.push_back(VK_FORMAT_R16_SFLOAT);           // COLOR_DLSS_SPEC_HIT_DIST
  colorFormats.push_back(m_colorFormat);                  // COLOR_DLSS_OUTPUT
#endif


  m_gBuffers.init({
      .allocator      = &m_alloc,
      .colorFormats   = colorFormats,
      .depthFormat    = m_depthFormat,
      .imageSampler   = m_sampler,
      .descriptorPool = m_app->getTextureDescriptorPool(),
  });

  // Setting up the Slang compiler
  {
    // Where to find shaders source code
    m_slangCompiler.addSearchPaths(getShaderDirs());
    // SPIRV 1.6 and VULKAN 1.4
    m_slangCompiler.defaultTarget();
    m_slangCompiler.defaultOptions();
    m_slangCompiler.addOption({slang::CompilerOptionName::MatrixLayoutRow, {slang::CompilerOptionValueKind::Int, 1}});
    m_slangCompiler.addOption({slang::CompilerOptionName::DebugInformation,
                               {slang::CompilerOptionValueKind::Int, SLANG_DEBUG_INFO_LEVEL_MAXIMAL}});
    m_slangCompiler.addOption({slang::CompilerOptionName::Optimization,
                               {slang::CompilerOptionValueKind::Int, SLANG_OPTIMIZATION_LEVEL_DEFAULT}});
  }

  // Get device information
  m_physicalDeviceInfo.init(m_app->getPhysicalDevice(), VK_API_VERSION_1_4);

  // Get ray tracing properties
  m_rtProperties.pNext = &m_accelStructProps;
  VkPhysicalDeviceProperties2 prop2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &m_rtProperties};
  vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);

  // init the Vulkan splatSet and the mesh set for mesh compositing
  m_splatSetVk.init(m_app, &m_alloc, &m_uploader, &m_sampler, &m_physicalDeviceInfo, &m_accelStructProps);
  m_meshSetVk.init(m_app, &m_alloc, &m_uploader, &m_accelStructProps);
  m_cameraSet.init(cameraManip.get());

  // Initialize VDZ depth mesh for 2.5D depth visualization
  // Use higher resolution (512x288) for better VR/World Space quality
  m_vdzMesh.initialize(m_device, &m_alloc, 512, 288);

  // Initialize dummy texture for bindings
  {
      VkImageCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      info.imageType = VK_IMAGE_TYPE_2D;
      info.format = VK_FORMAT_R8G8B8A8_UNORM;
      info.extent = {1, 1, 1};
      info.mipLevels = 1;
      info.arrayLayers = 2; // Array size 2 to satisfy Texture2DArray binding
      info.samples = VK_SAMPLE_COUNT_1_BIT;
      info.tiling = VK_IMAGE_TILING_OPTIMAL;
      info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      
      m_alloc.createImage(m_dummyTextureArray.image, info);

      VkImageViewCreateInfo viewInfo = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      viewInfo.image = m_dummyTextureArray.image.image;
      viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
      viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
      viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      viewInfo.subresourceRange.baseMipLevel = 0;
      viewInfo.subresourceRange.levelCount = 1;
      viewInfo.subresourceRange.baseArrayLayer = 0;
      viewInfo.subresourceRange.layerCount = 2;
      
      vkCreateImageView(m_device, &viewInfo, nullptr, &m_dummyTextureArray.view);

      // Transition to shader read only
      VkCommandBuffer cmd = m_app->createTempCmdBuffer();
      VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
      barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      barrier.srcAccessMask = 0;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      barrier.image = m_dummyTextureArray.image.image;
      barrier.subresourceRange = viewInfo.subresourceRange;
      
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 
          0, 0, nullptr, 0, nullptr, 1, &barrier);
      m_app->submitAndWaitTempCmdBuffer(cmd);
  }

  // Log HDR support status
  {
    // Use global accessor instead of m_app->getSwapchain().getColorSpace()
    // because the original nvvk::Swapchain header does not support getColorSpace()
    const VkColorSpaceKHR colorSpace = HDRSupport::getGlobalColorSpace();
    const VkFormat imageFormat = HDRSupport::getGlobalFormat();
    HDRSupport::logHDRStatus(colorSpace, imageFormat);
  }


#ifdef WITH_OPENXR
  // Initialize OpenXR if enabled
  if(m_useXrHmd)
  {
    initializeOpenXR();
  }
#endif
};

void VkViewer::onDetach()
{
#ifdef WITH_OPENXR
  shutdownOpenXR();
#endif

  // Wait for GPU to finish before cleanup
  vkDeviceWaitIdle(m_device);

  // stops the threads
  m_splatLoader.shutdown();
  m_cpuSorter.shutdown();
  // release scene and rendering related resources
  deinitAll();
  // release application wide related resources
  m_splatSetVk.deinit();
  m_meshSetVk.deinit();
  m_vdzMesh.cleanup();

#ifdef WITH_TCP_DEPTH
  if(m_tcpServerManager)
  {
    m_tcpServerManager->disconnectAll();
    m_tcpServerManager.reset();
  }
  m_tcpDepthEnabled = false;
#endif
  
  if(m_depthManager)
  {
    m_depthManager->cleanup();
    m_depthManager.reset();
  }
   m_depthClient.reset();

  // Stop the managed backend process if we started it
  if(m_backendManager && m_localBackendStarted)
  {
    m_backendManager->stop();
  }
   m_backendManager.reset();
}

void VkViewer::enableDepthRendering(const std::string& host, int port, const std::string& videoPath)
{
  LOGI("enableDepthRendering called: host=%s, port=%d, video=%s\n", host.c_str(), port, videoPath.c_str());

  // Configure depth client with host and port
  if(m_depthClient)
  {
    m_depthClient->setBackendAddress(host, port);
  }

  // Ensure depth manager is initialized
  if(!m_depthManager)
  {
    m_depthManager = std::make_unique<DepthTextureManager>();
    m_depthManager->initialize(m_device, m_app->getPhysicalDevice(), m_app->getQueue(0).queue, &m_alloc);
  }

  // Set default VDZ parameters
  prmFrame.vdzZScale = 10.0f;
  prmFrame.vdzZBias = 2.0f;
  prmFrame.vdzZGamma = 5.0f;
  prmFrame.vdzZMaxClip = 0.2f;
  prmFrame.vdzPlaneScale = 1.4f;
  prmFrame.vdzEdgeThreshold = 1.0f;
  prmFrame.vdzAspect = 1.777f;  // 16:9 default
  prmFrame.vdzUseVideoTexture = 1;

  // Ensure shaders and pipelines are initialized
  if(!m_shaders.valid || m_graphicsPipelineVdzMesh == VK_NULL_HANDLE)
  {
    if(!m_shaders.valid)
    {
      m_lightSet.init(m_app, &m_alloc, &m_uploader);
      initShaders();
      initRendererBuffers();
    }
    else
    {
      deinitPipelines();
    }
    initPipelines();
    initRtDescriptorSet();
    initRtPipeline();
    initDescriptorSetPostProcessing();
    initPipelinePostProcessing();
  }

  m_videoDepthPlaybackMode = false;
#ifdef WITH_VIDEO_DECODER
  m_hlsPlaybackMode = false;
#endif
  m_enableDepthRendering = true;
  m_playbackStartTime = std::chrono::steady_clock::now();
  m_playbackTimeOffset = 0.0;
  m_playbackPaused = false;
  m_lastVdzFrameIndex = SIZE_MAX;

  LOGI("Depth rendering enabled for streaming session\n");
}

#ifdef WITH_TCP_DEPTH
void VkViewer::enableTcpDepth(const std::string& serverList, const std::string& videoPath)
{
  LOGI("enableTcpDepth called: servers=%s, video=%s\n", serverList.c_str(), videoPath.c_str());

  if(!m_depthManager)
  {
    m_depthManager = std::make_unique<DepthTextureManager>();
    m_depthManager->initialize(m_device, m_app->getPhysicalDevice(), m_app->getQueue(0).queue, &m_alloc);
  }

  if(!m_tcpServerManager)
  {
    m_tcpServerManager = std::make_unique<TcpServerManager>();
    m_tcpServerManager->setDepthBuffer(&m_depthBuffer);
    m_tcpServerManager->setDepthFrameCallback([this](const DepthFrame&) {});
    m_tcpServerManager->parseServerList(serverList);
    m_tcpServerManager->connectAll();
  }

#ifdef WITH_VIDEO_DECODER
  if(!videoPath.empty())
  {
    if(m_videoDecoder)
    {
      vkDeviceWaitIdle(m_device);
      m_videoDecoder->stopDecoding();
      m_videoDecoder->close();
      m_videoDecoder.reset();
    }

    m_videoDecoder = std::make_unique<VideoDecoder>();
    m_videoDecoder->initializeVulkan(m_app->getInstance(), m_app->getPhysicalDevice(), m_app->getDevice(), m_app->getQueue(0).familyIndex, 0);
    if(m_videoDecoder->open(videoPath))
    {
      m_videoDecoder->startDecoding();
      LOGI("TCP depth video decoder opened: %s\n", videoPath.c_str());
    }
    else
    {
      LOGE("Failed to open TCP depth video: %s\n", videoPath.c_str());
      m_videoDecoder.reset();
    }
  }
#endif

  prmFrame.vdzZScale = 10.0f;
  prmFrame.vdzZBias = 2.0f;
  prmFrame.vdzZGamma = 5.0f;
  prmFrame.vdzZMaxClip = 0.2f;
  prmFrame.vdzPlaneScale = 1.4f;
  prmFrame.vdzEdgeThreshold = 1.0f;

    m_tcpDepthEnabled = true;
    m_tcpDepthBuffering = true;
    m_enableDepthRendering = true;
    m_videoDepthPlaybackMode = false;
#ifdef WITH_VIDEO_DECODER
    m_hlsPlaybackMode = false;
#endif
    m_playbackStartTime = std::chrono::steady_clock::now();
    m_playbackTimeOffset = 0.0;
    m_playbackPaused = false;
    m_lastVdzFrameIndex = SIZE_MAX;

    if(m_tcpDepthKeyframeOnly && m_tcpDepthMinBufferedFrames > 2)
    {
      m_tcpDepthMinBufferedFrames = 2;
      LOGI("TCP depth: I-frame mode — lowered buffering threshold to %d\n", m_tcpDepthMinBufferedFrames);
    }

  LOGI("TCP depth enabled with servers: %s\n", serverList.c_str());
}

void VkViewer::ensureTcpServerManager()
{
  if(!m_tcpServerManager)
  {
    m_tcpServerManager = std::make_unique<TcpServerManager>();
    m_tcpServerManager->setDepthBuffer(&m_depthBuffer);
    LOGI("TcpServerManager created (lazy init from UI)\n");
  }
}

void VkViewer::startLocalDepthServer(const std::string& modelPath, int port, int workers, const std::string& backend)
{
  ensureTcpServerManager();

  if(!m_localDepthServer)
  {
    m_localDepthServer = std::make_unique<LocalDepthServerManager>();
  }

  if(!m_localDepthServer->start(modelPath, port, workers, backend))
  {
    LOGE("Failed to start local depth_server\n");
    return;
  }

  m_tcpServerManager->addServer("127.0.0.1", port);
  m_tcpServerManager->connectAll();

  if(!m_depthManager)
  {
    m_depthManager = std::make_unique<DepthTextureManager>();
    m_depthManager->initialize(m_device, m_app->getPhysicalDevice(), m_app->getQueue(0).queue, &m_alloc);
  }

  m_tcpDepthEnabled = true;
  m_enableDepthRendering = true;

  LOGI("Local depth_server started and added to TCP server list (port %d)\n", port);
}

void VkViewer::requestSingleImageDepth(const std::string& imagePath, const std::string& serverList)
{
  LOGI("Single image depth: %s\n", imagePath.c_str());

  m_tcpDepthSingleImageRequested = false;
  m_tcpDepthSingleImageDone = false;

#ifdef WITH_VIDEO_DECODER
  auto decoder = std::make_unique<VideoDecoder>();
  decoder->initializeVulkan(m_app->getInstance(), m_app->getPhysicalDevice(), m_app->getDevice(), m_app->getQueue(0).familyIndex, 0);
  if(!decoder->open(imagePath))
  {
    LOGE("Failed to open image: %s\n", imagePath.c_str());
    return;
  }

  decoder->startDecoding();

  DecodedFrame frame;
  if(!decoder->getNextFrame(frame) || frame.data.empty() || frame.width == 0 || frame.height == 0)
  {
    LOGE("Failed to decode image frame: %s\n", imagePath.c_str());
    decoder->stopDecoding();
    decoder->close();
    return;
  }

  const uint32_t w = static_cast<uint32_t>(frame.width);
  const uint32_t h = static_cast<uint32_t>(frame.height);
  std::vector<uint8_t> rgb(w * h * 3);
  for(uint32_t i = 0; i < w * h; ++i)
  {
    rgb[i * 3 + 0] = frame.data[i * 4 + 0];
    rgb[i * 3 + 1] = frame.data[i * 4 + 1];
    rgb[i * 3 + 2] = frame.data[i * 4 + 2];
  }

  if(!m_depthManager)
  {
    m_depthManager = std::make_unique<DepthTextureManager>();
    m_depthManager->initialize(m_device, m_app->getPhysicalDevice(), m_app->getQueue(0).queue, &m_alloc);
  }

  if(!m_tcpServerManager)
  {
    m_tcpServerManager = std::make_unique<TcpServerManager>();
  }
  m_tcpServerManager->setDepthBuffer(&m_depthBuffer);
  m_tcpServerManager->setDepthFrameCallback([this](const DepthFrame&) {
    // Depth frames are polled on the render thread in updateDepthRendering().
  });
  m_tcpServerManager->parseServerList(serverList);
  m_tcpServerManager->connectAll();

  if(m_tcpServerManager->sendFrame(0, 0, rgb.data(), w, h) >= 0)
  {
    m_tcpDepthSingleImageRequested = true;
    LOGI("Single image depth frame sent: %dx%d\n", w, h);
  }
  else
  {
    LOGE("Failed to send single image depth frame: %s\n", imagePath.c_str());
  }

  decoder->stopDecoding();
  decoder->close();
#else
  (void)serverList;
  LOGE("Video decoder not available - rebuild with ENABLE_VIDEO_DECODER=ON\n");
#endif
}

void VkViewer::stopTcpDepth()
{
  m_tcpDepthEnabled = false;
  m_tcpDepthSingleImageRequested = false;
  m_tcpDepthSingleImageDone = false;
  m_tcpDepthBuffering = false;
  m_tcpVideoFrameIndex = 0;

#ifdef WITH_VIDEO_DECODER
  if(m_videoDecoder)
  {
    m_videoDecoder->stopDecoding();
    m_videoDecoder->close();
    m_videoDecoder.reset();
  }
#endif

  if(m_tcpServerManager)
  {
    m_tcpServerManager->disconnectAll();
  }

  LOGI("TCP depth stopped\n");
}

void VkViewer::rewindTcpDepth()
{
  m_tcpDepthBuffering = true;
  m_tcpVideoFrameIndex = 0;
#ifdef WITH_VIDEO_DECODER
  if(m_videoDecoder)
  {
    m_videoDecoder->seekToTime(0.0);
  }
#endif
  m_playbackStartTime = std::chrono::steady_clock::now();
  m_playbackTimeOffset = 0.0;
  LOGI("TCP depth rewound to start\n");
}
#endif

void VkViewer::onResize(VkCommandBuffer cmd, const VkExtent2D& viewportSize)
{
  m_viewSize = {viewportSize.width, viewportSize.height};
  NVVK_CHECK(m_gBuffers.update(cmd, viewportSize));
  updateKBuffers(viewportSize.width, viewportSize.height);  // GRTX: resize K-buffers
  updateRtDescriptorSet();
  updateDescriptorSetPostProcessing();
  resetFrameCounter();
  
  // Invalidate sort cache on resize (projection changes)
  m_lastSortValid = false;
}

void VkViewer::onPreRender()
{
  m_profilerTimeline->frameAdvance();

#ifdef WITH_OPENXR
  // Clear the resize flag from previous frame
  m_xrResizedThisFrame = false;

  // Check if XR requires GBuffer resize (must happen before command buffer recording)
  if(m_xrInitialized && m_xr && m_xr->isValid())
  {
    VkExtent2D xrExtent = m_xr->getFullExtent();
    if(xrExtent.width > 0 && xrExtent.height > 0
       && (m_viewSize.x != xrExtent.width || m_viewSize.y != xrExtent.height))
    {
      // Wait for GPU to finish all work before resizing
      vkDeviceWaitIdle(m_device);

      // Create a temporary command buffer for the resize operation
      VkCommandBuffer cmd = m_app->createTempCmdBuffer();
      m_viewSize = glm::vec2(xrExtent.width, xrExtent.height);
      NVVK_CHECK(m_gBuffers.update(cmd, xrExtent));
      updateKBuffers(xrExtent.width, xrExtent.height);  // GRTX: resize K-buffers
      updateRtDescriptorSet();
      updateDescriptorSetPostProcessing();
      resetFrameCounter();
      m_app->submitAndWaitTempCmdBuffer(cmd);

      // Skip rendering this frame to let descriptor sets stabilize
      m_xrResizedThisFrame = true;
    }
  }
#endif

#ifdef WITH_TCP_DEPTH
  // TCP depth video frame dispatch: decode frames and send to depth servers
  if(m_tcpDepthEnabled && m_videoDecoder && m_tcpServerManager && !m_playbackPaused)
  {
    m_tcpServerManager->update();

    double videoFps = m_videoDecoder->getFrameRate();
    if(videoFps <= 0.0) videoFps = 30.0;

    // Recalculate frame-skip only every 30 dispatched frames to avoid
    // useless setFrameSkip() calls every render frame.
    static int lastSkipEstimate = 0;
    static uint32_t lastSkipRecalcAt = 0;
    if(m_tcpVideoFrameIndex - lastSkipRecalcAt >= 30 || lastSkipEstimate == 0)
    {
      double serverFpsEstimate = 2.0 * static_cast<double>(m_tcpServerManager->serverCount());
      const auto& servers = m_tcpServerManager->getServers();
      for(const auto& srv : servers)
      {
        if(srv.avg_latency_ms > 0.0)
        {
          double measuredFps = 1000.0 / srv.avg_latency_ms;
          if(measuredFps > serverFpsEstimate)
            serverFpsEstimate = measuredFps;
        }
      }
      int newEstimate = static_cast<int>(serverFpsEstimate);
      if(newEstimate != lastSkipEstimate)
      {
        m_tcpServerManager->setFrameSkip(static_cast<int>(videoFps), newEstimate);
        lastSkipEstimate = newEstimate;
      }
      lastSkipRecalcAt = m_tcpVideoFrameIndex;
    }

    // Backpressure: don't dispatch if too many frames are in-flight.
    // Allow 2× server_count buffered frames to keep the pipeline fed.
    const size_t maxInFlight = m_tcpServerManager->serverCount() * 2;
    if(m_tcpServerManager->getInFlightCount() >= maxInFlight)
      return;

    double frameIntervalSec = 1.0 / videoFps;
    auto now = std::chrono::steady_clock::now();
    double elapsedSinceLastDispatch =
        std::chrono::duration<double>(now - m_tcpVideoLastDispatchTime).count();

    if(elapsedSinceLastDispatch >= frameIntervalSec)
    {
      DecodedFrame decoded;
      bool gotValidFrame = false;
      while(m_videoDecoder->tryGetNextFrame(decoded))
      {
        if(decoded.data.empty() || decoded.width <= 0 || decoded.height <= 0)
          continue;

        if(m_tcpDepthKeyframeOnly && !decoded.is_keyframe)
          continue;

        gotValidFrame = true;
        m_tcpVideoRgba = decoded.data;
        m_tcpVideoRgbaWidth = static_cast<uint32_t>(decoded.width);
        m_tcpVideoRgbaHeight = static_cast<uint32_t>(decoded.height);
        break;
      }

      if(gotValidFrame)
      {
        uint32_t tsMs = static_cast<uint32_t>(decoded.timestamp * 1000.0);

        // Skip dispatch if this timestamp is already in the depth cache
        bool alreadyCached = m_depthBuffer.hasCachedFrame(tsMs);
        if(!alreadyCached)
        {
          const uint32_t w = static_cast<uint32_t>(decoded.width);
          const uint32_t h = static_cast<uint32_t>(decoded.height);
          std::vector<uint8_t> rgb(w * h * 3);
          for(uint32_t i = 0; i < w * h; ++i)
          {
            rgb[i * 3 + 0] = decoded.data[i * 4 + 0];
            rgb[i * 3 + 1] = decoded.data[i * 4 + 1];
            rgb[i * 3 + 2] = decoded.data[i * 4 + 2];
          }

          int sent = m_tcpServerManager->sendFrame(m_tcpVideoFrameIndex, tsMs, rgb.data(), w, h);
          if(sent >= 0)
          {
            ++m_tcpVideoFrameIndex;
          }
        }
        m_tcpVideoLastDispatchTime = now;
      }
      else if(!m_videoDecoder->isRunning() || m_videoDecoder->isAtEnd())
      {
        m_videoDecoder->seekToTime(0.0);
        m_tcpVideoFrameIndex = 0;
        m_tcpVideoLastDispatchTime = now;
        LOGI("TCP depth video looped back to start\n");
      }
    }
  }
#endif
}

bool VkViewer::initAll()
{
  vkDeviceWaitIdle(m_device);

  if(!m_radianceFields.empty() && m_radianceFields[0].isLcc)
  {
    initializeLccStreaming(m_radianceFields[0].filename);
  }

  // resize the CPU sorter indices buffer
  m_splatIndices.resize(m_splatIndices.size());
  // TODO: use BBox of point cloud to set far plane, eye and center
  m_cameraSet.setCamera(Camera());
  // record default cam for reset in UI
  m_cameraSet.setHomePreset(m_cameraSet.getCamera());
  // reset general parameters
  resetRenderSettings();

  m_lightSet.init(m_app, &m_alloc, &m_uploader);
  // init a new setup
  if(!initShaders())
  {
    return false;
  }
  initRendererBuffers();
  m_splatSetVk.initDataStorage(m_splatSet, prmData.dataStorage, prmData.shFormat);
  
  // Compute chunk bounds for hierarchical frustum culling
  computeChunkBounds();
  initChunkCullingBuffers();
  
  initPipelines();

  // RTX specifics
  m_splatSetVk.rtxInitSplatModel(m_splatSet, prmRtxData.useTlasInstances, prmRtxData.useAABBs, prmRtxData.useSpheres,
                                 prmRtxData.compressBlas, prmRtx.kernelDegree, prmRtx.kernelMinResponse, prmRtx.kernelAdaptiveClamping);

  m_splatSetVk.rtxInitAccelerationStructures(m_splatSet);

  initRtDescriptorSet();
  initRtPipeline();

  // Post processing
  initDescriptorSetPostProcessing();
  initPipelinePostProcessing();

  return true;
}

void VkViewer::deinitAll()
{
  vkDeviceWaitIdle(m_device);

#ifdef WITH_DLSS_RR
  shutdownDlssRR();
#endif

  m_canCollectReadback = false;
  
  // Invalidate sort cache when scene is unloaded
  m_lastSortValid = false;
  
  deinitScene();
  m_splatSetVk.resetTransform();
  m_splatSetVk.deinitDataStorage();
  m_splatSetVk.rtxDeinitPtlas();
  m_splatSetVk.rtxDeinitSplatModel();
  m_splatSetVk.rtxDeinitAccelerationStructures();
  m_meshSetVk.deinitDataStorage();
  m_meshSetVk.rtxDeinitAccelerationStructures();
  m_lightSet.deinit();
  m_cameraSet.deinit();
  deinitShaders();
  deinitPipelines();
  deinitRendererBuffers();
  deinitChunkCullingBuffers();
  resetRenderSettings();
  // record default cam for reset in UI
  m_cameraSet.setCamera(Camera());
  // record default cam for reset in UI
  m_cameraSet.setHomePreset(m_cameraSet.getCamera());
}

void VkViewer::enableDepthVideoPlayback(const std::string& metadataPath)
{
  LOGI("enableDepthVideoPlayback called with: %s\n", metadataPath.c_str());
#ifdef WITH_VIDEO_DECODER
  // Clean up any existing playback manager first
  if(m_videoDepthManager)
  {
    vkDeviceWaitIdle(m_device);
    m_videoDepthManager->close();
    m_videoDepthManager.reset();
  }

  // Initialize or reinitialize depth manager
  if(!m_depthManager)
  {
    m_depthManager = std::make_unique<DepthTextureManager>();
    m_depthManager->initialize(m_device, m_app->getPhysicalDevice(), m_app->getQueue(0).queue, &m_alloc);
  }

  m_videoDepthManager = std::make_unique<VideoDepthPlaybackManager>();

  uint32_t videoQueueFamily = m_app->getQueue(0).familyIndex;
  uint32_t count            = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(m_app->getPhysicalDevice(), &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  vkGetPhysicalDeviceQueueFamilyProperties(m_app->getPhysicalDevice(), &count, families.data());
  for(uint32_t i = 0; i < count; i++)
  {
    if(families[i].queueFlags & VK_QUEUE_VIDEO_DECODE_BIT_KHR)
    {
      videoQueueFamily = i;
      break;
    }
  }

  if(!m_videoDepthManager->openFromMetadata(metadataPath, m_app->getInstance(), m_app->getPhysicalDevice(), m_app->getDevice(), videoQueueFamily, 0))
  {
    LOGE("Failed to open depth video from: %s\n", metadataPath.c_str());
    m_videoDepthManager.reset();
    return;
  }

  const auto& metadata = m_videoDepthManager->getMetadata();
  
  m_videoTexture.width = metadata.sourceWidth;
  m_videoTexture.height = metadata.sourceHeight;

  prmFrame.vdzZMin = metadata.zMin;
  prmFrame.vdzZMax = metadata.zMax;
  prmFrame.vdzAspect = metadata.sourceWidth > 0 && metadata.sourceHeight > 0 
      ? static_cast<float>(metadata.sourceWidth) / static_cast<float>(metadata.sourceHeight) 
      : 1.777f;
  prmFrame.vdzZScale = 10.0f;
  prmFrame.vdzZBias = 2.0f;
  prmFrame.vdzZGamma = 5.0f;
  prmFrame.vdzZMaxClip = 0.2f;
  prmFrame.vdzPlaneScale = 1.4f;
  prmFrame.vdzEdgeThreshold = 1.0f;

  LOGI("Depth video initialized: %dx%d, fps=%.2f, frames=%d, z=[%.2f, %.2f]\n",
       metadata.sourceWidth, metadata.sourceHeight, metadata.fps, 
       metadata.frameCount, metadata.zMin, metadata.zMax);

  // Ensure shaders and pipelines are initialized
  // Also reinitialize if VDZ pipeline is missing (can happen if splats were unloaded)
  if(!m_shaders.valid || m_graphicsPipelineVdzMesh == VK_NULL_HANDLE)
  {
    if(!m_shaders.valid)
    {
      m_lightSet.init(m_app, &m_alloc, &m_uploader);
      initShaders();
      initRendererBuffers();
    }
    else
    {
      // Shaders valid but pipelines were destroyed - reinitialize
      deinitPipelines();
    }
    initPipelines();
    initRtDescriptorSet();
    initRtPipeline();
    initDescriptorSetPostProcessing();
    initPipelinePostProcessing();
  }

  m_videoDepthPlaybackMode = true;
  m_enableDepthRendering = true;
  m_playbackStartTime = std::chrono::steady_clock::now();
  m_playbackTimeOffset = 0.0;
  m_playbackPaused = false;
  m_lastVdzFrameIndex = SIZE_MAX;

  m_videoDepthManager->play();

  prmFrame.vdzUseVideoTexture = 1;

  LOGI("Depth video playback enabled from metadata: %s\n", metadataPath.c_str());
  LOGI("  Video: %s\n", metadata.videoPath.c_str());
  LOGI("  Depth: %s\n", metadata.depthVideoPath.c_str());
  LOGI("  Frames: %d, FPS: %.2f\n", metadata.frameCount, metadata.fps);
#else
  LOGE("Video decoder not available - rebuild with ENABLE_VIDEO_DECODER=ON\n");
#endif
}

void VkViewer::enableHlsPlayback(const std::string& hlsPlaylistPath)
{
  LOGI("enableHlsPlayback called with: %s\n", hlsPlaylistPath.c_str());
#ifdef WITH_VIDEO_DECODER
  // Clean up any existing HLS player or video depth manager first
  if(m_hlsPlayer)
  {
    vkDeviceWaitIdle(m_device);
    m_hlsPlayer->stop();
    m_hlsPlayer.reset();
  }

  if(m_videoDepthManager)
  {
    vkDeviceWaitIdle(m_device);
    m_videoDepthManager->close();
    m_videoDepthManager.reset();
  }

  // Initialize or reinitialize depth manager
  if(!m_depthManager)
  {
    m_depthManager = std::make_unique<DepthTextureManager>();
    m_depthManager->initialize(m_device, m_app->getPhysicalDevice(), m_app->getQueue(0).queue, &m_alloc);
  }

  m_hlsPlayer = std::make_unique<HlsDepthPlayer>();
  if(!m_hlsPlayer->open(hlsPlaylistPath))
  {
    LOGE("Failed to open HLS stream from: %s\n", hlsPlaylistPath.c_str());
    m_hlsPlayer.reset();
    return;
  }

  // Get metadata from HLS player
  m_hlsMetadata = m_hlsPlayer->getMetadata();

  // Update video texture dimensions
  m_videoTexture.width = m_hlsMetadata.rgbWidth;
  m_videoTexture.height = m_hlsMetadata.height;

  // Set depth rendering parameters
  prmFrame.vdzZMin = m_hlsMetadata.zMin;
  prmFrame.vdzZMax = m_hlsMetadata.zMax;
  prmFrame.vdzAspect = m_hlsMetadata.rgbWidth > 0 && m_hlsMetadata.height > 0
      ? static_cast<float>(m_hlsMetadata.rgbWidth) / static_cast<float>(m_hlsMetadata.height)
      : 1.777f;
  prmFrame.vdzZScale = 10.0f;
  prmFrame.vdzZBias = 2.0f;
  prmFrame.vdzZGamma = 5.0f;
  prmFrame.vdzZMaxClip = 0.2f;
  prmFrame.vdzPlaneScale = 1.4f;
  prmFrame.vdzEdgeThreshold = 1.0f;

  LOGI("HLS playback initialized: %dx%d (RGB: %dx%d), fps=%.2f, z=[%.2f, %.2f]\n",
       m_hlsMetadata.rgbWidth * 2, m_hlsMetadata.height,
       m_hlsMetadata.rgbWidth, m_hlsMetadata.height,
       m_hlsMetadata.fps, m_hlsMetadata.zMin, m_hlsMetadata.zMax);

  // Ensure shaders and pipelines are initialized
  if(!m_shaders.valid || m_graphicsPipelineVdzMesh == VK_NULL_HANDLE)
  {
    if(!m_shaders.valid)
    {
      m_lightSet.init(m_app, &m_alloc, &m_uploader);
      initShaders();
      initRendererBuffers();
    }
    else
    {
      // Shaders valid but pipelines were destroyed - reinitialize
      deinitPipelines();
    }
    initPipelines();
    initRtDescriptorSet();
    initRtPipeline();
    initDescriptorSetPostProcessing();
    initPipelinePostProcessing();
  }

  m_hlsPlaybackMode = true;
  m_enableDepthRendering = true;
  m_playbackStartTime = std::chrono::steady_clock::now();
  m_playbackTimeOffset = 0.0;
  m_playbackPaused = false;
  m_lastVdzFrameIndex = SIZE_MAX;

  m_hlsPlayer->start();

  prmFrame.vdzUseVideoTexture = 1;

  LOGI("HLS playback enabled from: %s\n", hlsPlaylistPath.c_str());
#else
  LOGE("Video decoder not available - rebuild with ENABLE_VIDEO_DECODER=ON\n");
#endif
}

bool VkViewer::isDepthVideoPlaying() const
{
#ifdef WITH_VIDEO_DECODER
    return m_videoDepthManager && m_videoDepthManager->isPlaying();
#else
    return false;
#endif
}

void VkViewer::deinitScene()

{
  m_splatSet.clear();
  m_splatSetPending.clear();
  m_radianceFields.clear();
  m_pendingLoadFilename = "";
  shutdownLccStreaming();
}

bool VkViewer::initializeLccStreaming(const std::filesystem::path& scenePath)
{
  if(!LccLoader::canLoad(scenePath))
  {
    return false;
  }

  // scenePath may be either the directory or the meta.lcc file itself
  std::filesystem::path sceneDir = scenePath;
  if(scenePath.filename() == "meta.lcc")
  {
    sceneDir = scenePath.parent_path();
  }

  LccLoader::LccMeta meta;
  std::filesystem::path metaPath = sceneDir / "meta.lcc";
  if(!LccLoader::parseMeta(metaPath, meta))
  {
    LOGW("Failed to parse LCC metadata for streaming: %s\n", scenePath.string().c_str());
    return false;
  }

  if(meta.totalLevel <= 1)
  {
    LOGI("Single-LOD LCC scene, skipping streaming: %s\n", scenePath.string().c_str());
    return false;
  }

  m_lccTileManager = std::make_unique<LccTileManager>();

  m_lccTileManager->config().enabled = true;
  m_lccTileManager->config().asyncLoading = true;
  m_lccTileManager->config().maxTilesLoaded = 16;
  m_lccTileManager->config().tileSize = meta.cellLengthX > 0 ? meta.cellLengthX : 30.0f;

  if(!m_lccTileManager->initialize(sceneDir, meta))
  {
    LOGE("Failed to initialize LCC tile manager: %s\n", sceneDir.string().c_str());
    m_lccTileManager.reset();
    return false;
  }

  m_lccTileManager->setLodCount(meta.totalLevel);
  
  // Store metadata for GPU-side decompression (scale min/max ranges)
  m_lccMeta = meta;
  
  // Enable packed mode for GPU-side decompression (much faster than CPU decode)
  m_lccTileManager->setPackedMode(true);

  LOGI("LCC streaming initialized: %s (tiles=%u, lodLevels=%u, packedMode=%d)\n",
       sceneDir.string().c_str(),
       m_lccTileManager->getTotalTileCount(),
       meta.totalLevel,
       m_lccTileManager->isPackedMode() ? 1 : 0);

  return true;
}

void VkViewer::shutdownLccStreaming()
{
  if(m_lccTileManager)
  {
    m_lccTileManager->shutdown();
    m_lccTileManager.reset();
  }
  m_streamingSplatSet.clear();
}

void VkViewer::updateLccStreaming(const glm::mat4& viewProj, const glm::vec3& cameraPos, float dt)
{
  if(!m_lccTileManager || !m_lccTileManager->isActive())
  {
    return;
  }

  m_lccTileManager->update(viewProj, cameraPos, dt);
  
  if(m_lccTileManager->isPackedMode())
  {
    // GPU-side decompression: get raw packed data
    uint32_t splatCount = 0;
    const uint8_t* packedData = m_lccTileManager->getVisiblePackedData(splatCount);
    
    if(packedData && splatCount > 0)
    {
      const bool wasEmpty = (m_lccPackedSplatCount == 0);
      const bool tilesChanged = m_lccTileManager->hasVisibleTilesChanged();
      
      if(wasEmpty || tilesChanged)
      {
        // Store packed data for upload in processUpdateRequests
        m_lccPackedData.assign(packedData, packedData + splatCount * 32);
        m_lccPackedSplatCount = splatCount;
        
        // Create a minimal SplatSet just to hold the count (for existing code paths)
        m_streamingSplatSet.clear();
        m_streamingSplatSet.positions.resize(splatCount * 3, 0.0f);  // Placeholder
        
        // Trigger data update to use the new packed data
        m_requestUpdateSplatData = true;
        
        if(wasEmpty)
        {
          LOGI("LCC packed data now available: %u splats\n", splatCount);
        }
      }
    }
  }
  else
  {
    // CPU-side decompression (legacy path)
    m_lccTileManager->getVisibleSplats(m_streamingSplatSet);
    m_lccPackedData.clear();
    m_lccPackedSplatCount = 0;
  }
}
void VkViewer::benchmarkAdvance()
{
  LOGI("BENCHMARK_ADV %d {\n", m_benchmarkId);
  LOGI(" Memory Scene; Host used \t%u; Device Used \t%u; Device Allocated \t%u; (bytes)\n",
        m_splatSetVk.memoryStats.srcAll, m_splatSetVk.memoryStats.odevAll, m_splatSetVk.memoryStats.devAll);
  LOGI(" Memory Rasterization; Host used \t%zu; Device Used \t%zu; Device Allocated \t%zu; (bytes)\n",
       m_renderMemoryStats.rasterHostTotal, m_renderMemoryStats.rasterDeviceUsedTotal,
       m_renderMemoryStats.rasterDeviceAllocTotal);
  LOGI(" Memory Raytracing; Host used \t%zu; Device Used \t%zu; Device Allocated \t%zu; (bytes)\n",
       m_renderMemoryStats.rtxHostTotal, m_renderMemoryStats.rtxDeviceUsedTotal,
       m_renderMemoryStats.rtxDeviceAllocTotal);
  LOGI("}\n");

  m_benchmarkId++;
}

#ifdef WITH_OPENXR
bool VkViewer::queryOpenXrVulkanExtensions(std::vector<std::string>& outInstanceExtensions,
                                                     std::vector<std::string>& outDeviceExtensions)
{
  if(!m_xr)
  {
    m_xr = std::make_unique<GsOpenXr>();
  }
  return m_xr->queryRequiredVulkanExtensions(outInstanceExtensions, outDeviceExtensions);
}

void VkViewer::initializeOpenXR()
{
  if(!m_xr)
  {
    m_xr = std::make_unique<GsOpenXr>();
  }

#ifndef _WIN32
  // Suppress OpenXR loader error messages on Linux when no runtime is installed
  // These errors are expected when running without VR hardware
  setenv("XR_SUPPRESS_LOADER_MESSAGES", "1", 1);
#endif

  // Use SRGB format for XR swapchain - GPU will automatically convert linear->sRGB on write
  // This is the correct way to handle color space for VR displays
  VkFormat xrColorFormat = VK_FORMAT_R8G8B8A8_SRGB;

  if(!m_xr->initialize(m_app->getInstance(), m_app->getPhysicalDevice(), m_app->getDevice(),
                       m_app->getQueue(0).familyIndex, 0, xrColorFormat, m_depthFormat))
  {
#ifndef _WIN32
    // On Linux, OpenXR is typically not available without VR hardware
    // Log at info level instead of error since this is expected
    LOGI("OpenXR not available (no runtime installed). Running in desktop mode.\n");
#else
    LOGE("Failed to initialize OpenXR\n");
#endif
    m_xr.reset();
    m_xrInitialized = false;
    m_useXrHmd      = false;
    return;
  }

  m_xrInitialized = true;
  m_xrResizedThisFrame = true;  // Skip rendering on this frame to allow resources to sync

  // Get the XR resolution and resize our buffers to match
  VkExtent2D xrExtent = m_xr->getFullExtent();
  LOGI("OpenXR initialized. Full extent: %dx%d\n", xrExtent.width, xrExtent.height);

  // Force SBS mode when XR is enabled
  m_renderSBS = true;

  // Initialize hand meshes now that XR is ready
  onXrInitialized();
}

void VkViewer::shutdownOpenXR()
{
  deinitXrMultiviewResources();
  
  // Cleanup environment depth views
  for (auto& pair : m_envDepthImageViews)
  {
    vkDestroyImageView(m_device, pair.second, nullptr);
  }
  m_envDepthImageViews.clear();
  
  if(m_xr)
  {
    m_xr->shutdown();
    m_xr.reset();
  }
  m_xrInitialized = false;
  m_xrFirstFrame = true;
}

void VkViewer::updateXrLocomotion(float deltaTime)
{
  if(!m_xrInitialized || !m_xr)
    return;

  m_xr->pollControllerInput();
  const auto& locomotion = m_xr->getLocomotionInput();

  // Check if in depth-only mode (depth content but no splats or meshes)
#ifdef WITH_VIDEO_DECODER
  bool hasDepthContent = m_enableDepthRendering || m_videoDepthPlaybackMode || m_hlsPlaybackMode;
#else
  bool hasDepthContent = m_enableDepthRendering || m_videoDepthPlaybackMode;
#endif
  bool hasSplats = m_splatSet.positions.size() > 0 || m_splatLoader.getStatus() == SplatLoaderAsync::State::STATE_READY;
  bool hasMeshes = !m_meshSetVk.instances.empty();
  bool isDepthOnlyMode = hasDepthContent && !hasSplats && !hasMeshes;

  // Handle depth video mode controls
#ifdef WITH_VIDEO_DECODER
  bool isHls = m_hlsPlaybackMode;
#else
  bool isHls = false;
#endif
  // Use local isHls instead of m_hlsPlaybackMode for subsequent checks if possible,
  // or guard usages.
  if(isDepthOnlyMode && hasDepthContent)
  {
    // Left stick: plane tilt with spring-back on release
    if(std::abs(locomotion.move.x) > 0.1f || std::abs(locomotion.move.y) > 0.1f)
    {
      // Map left stick to tilt angles (limited range)
      const float maxTilt = glm::radians(20.0f);  // ±20 degrees
      m_vrPlaneTiltTarget.x = locomotion.move.y * maxTilt;  // Pitch (up/down)
      m_vrPlaneTiltTarget.y = -locomotion.move.x * maxTilt; // Yaw (left/right)
    }
    else
    {
      // Spring back to neutral when stick released
      m_vrPlaneTiltTarget = glm::vec2(0.0f, 0.0f);
    }

    // Apply spring-back interpolation for plane tilt
    float springStiffness = 10.0f;
    glm::vec2 tiltDelta = m_vrPlaneTiltTarget - m_vrPlaneTilt;
    m_vrPlaneTilt += tiltDelta * springStiffness * deltaTime;

    // Right stick: parallax parameters
    // X-axis: parallax strength/scale
    // Y-axis: viewpoint distance from plane
    if(std::abs(locomotion.turn.x) > 0.1f)
    {
      m_vrParallaxScale = std::clamp(m_vrParallaxScale + locomotion.turn.x * deltaTime * 2.0f, 0.0f, 2.0f);
      prmFrame.vdzParallaxStrength = m_vrParallaxScale;
    }
    if(std::abs(locomotion.turn.y) > 0.1f)
    {
      m_vrViewpointDistance = std::clamp(m_vrViewpointDistance + locomotion.turn.y * deltaTime * 0.5f, -0.5f, 0.5f);
      // Apply to parallax focus as a rough proxy for viewpoint distance
      prmFrame.vdzParallaxFocus = std::clamp(0.5f - m_vrViewpointDistance * 2.0f, 0.0f, 1.0f);
    }

    // Stick button: play/pause toggle
    if(locomotion.sprintPressed && !m_playbackPaused)
    {
      m_playbackPaused = true;
    }
    else if(!locomotion.sprintPressed && m_playbackPaused && m_playbackTimeOffset == 0.0)
    {
      // Only resume if we were paused by the stick button (detected by timeOffset == 0)
      m_playbackPaused = false;
      m_playbackStartTime = std::chrono::steady_clock::now();
    }

    // Apply plane tilt to the VDZ mesh orientation
    // This simulates "leaning" into the scene
    // Since rotation is stored as Euler angles (vec3), we modify the rotation directly
    if(std::abs(m_vrPlaneTilt.x) > 0.001f || std::abs(m_vrPlaneTilt.y) > 0.001f)
    {
      m_splatSetVk.rotation.x = m_vrPlaneTilt.x;  // Pitch
      m_splatSetVk.rotation.y = m_vrPlaneTilt.y;  // Yaw
      computeTransform(m_splatSetVk.scale, m_splatSetVk.rotation, m_splatSetVk.translation,
                       m_splatSetVk.transform, m_splatSetVk.transformInverse);
    }

    return;  // Skip normal locomotion in depth-only mode
  }

  // Normal scene locomotion (not depth-only mode)
  // Determine if we should use hand input or controller input
  bool useHands = m_xr->handsSupported() &&
                  (m_xr->getHandInput(GsOpenXr::Hand::Left).tracked ||
                   m_xr->getHandInput(GsOpenXr::Hand::Right).tracked);

  if (!useHands) {
    // Use controller-based locomotion
    if (!m_xr->hasControllers())
      return;

    // Get HMD forward direction from the view matrix (use left eye as reference)
    GsOpenXr::EyeData eyeData = m_xr->getEyeData(0);
    glm::mat4 viewInverse = glm::inverse(eyeData.view);
    glm::vec3 forward = -glm::vec3(viewInverse[2]);  // -Z is forward in view space
    glm::vec3 right = glm::vec3(viewInverse[0]);     // +X is right

    // Flatten forward/right to XZ plane for locomotion (ignore vertical component)
    forward.y = 0.0f;
    if(glm::length(forward) > 0.001f)
      forward = glm::normalize(forward);
    else
      forward = glm::vec3(0.0f, 0.0f, -1.0f);

    right.y = 0.0f;
    if(glm::length(right) > 0.001f)
      right = glm::normalize(right);
    else
      right = glm::vec3(1.0f, 0.0f, 0.0f);

    // Calculate movement speed
    float speed = m_xrMoveSpeed;
    if(locomotion.sprintPressed)
    {
      speed *= m_xrSprintMultiplier;
    }

    // Calculate movement - move scene opposite to desired player movement
    glm::vec3 movement(0.0f);
    movement -= forward * locomotion.move.y * speed * deltaTime;  // Forward/back (inverted for scene)
    movement -= right * locomotion.move.x * speed * deltaTime;    // Strafe (inverted for scene)

    // Apply movement to splat set translation
    if(glm::length(movement) > 0.0001f)
    {
      m_splatSetVk.translation += movement;
      computeTransform(m_splatSetVk.scale, m_splatSetVk.rotation, m_splatSetVk.translation,
                       m_splatSetVk.transform, m_splatSetVk.transformInverse);
    }

    // Handle right stick - rotation (X) and vertical movement (Y)
    bool transformChanged = false;

    // Right stick X: rotate scene around Y axis
    float turnAmount = locomotion.turn.x * glm::radians(m_xrSmoothTurnSpeed) * deltaTime;
    if(std::abs(turnAmount) > 0.0001f)
    {
      m_splatSetVk.rotation.y += turnAmount;
      transformChanged = true;
    }

    // Right stick Y: move scene up/down
    float verticalMove = locomotion.turn.y * speed * deltaTime;
    if(std::abs(verticalMove) > 0.0001f)
    {
      m_splatSetVk.translation.y -= verticalMove;
      transformChanged = true;
    }

    if(transformChanged)
    {
      computeTransform(m_splatSetVk.scale, m_splatSetVk.rotation, m_splatSetVk.translation,
                       m_splatSetVk.transform, m_splatSetVk.transformInverse);
    }
  }

  // Handle hand-based pinch-drag locomotion
  if (m_xr->handsSupported()) {
    const auto& left  = m_xr->getHandInput(GsOpenXr::Hand::Left);
    const auto& right = m_xr->getHandInput(GsOpenXr::Hand::Right);

    static bool lastPinchLeft = false;
    static bool lastPinchRight = false;

    auto updateHandDrag = [&](const GsOpenXr::HandInput& hand, GsOpenXr::Hand handEnum, bool& lastPinch) {
      bool pinchingNow = hand.tracked && hand.indexPinching;

      if (pinchingNow && !lastPinch && hand.tracked) {
        m_handDrag.active = true;
        m_handDrag.hand = handEnum;
        m_handDrag.grabStartHandPos = hand.indexTipPos;
        m_handDrag.grabStartWorldOffset = m_splatSetVk.translation;
      }

      if (m_handDrag.active && m_handDrag.hand == handEnum && pinchingNow) {
        glm::vec3 delta = hand.indexTipPos - m_handDrag.grabStartHandPos;
        delta.y = 0.0f;
        m_splatSetVk.translation = m_handDrag.grabStartWorldOffset - delta;

        computeTransform(m_splatSetVk.scale, m_splatSetVk.rotation, m_splatSetVk.translation,
                         m_splatSetVk.transform, m_splatSetVk.transformInverse);
      }

      if (!pinchingNow && lastPinch && m_handDrag.active && m_handDrag.hand == handEnum) {
        m_handDrag.active = false;
      }

      lastPinch = pinchingNow;
    };

    updateHandDrag(left, GsOpenXr::Hand::Left, lastPinchLeft);
    updateHandDrag(right, GsOpenXr::Hand::Right, lastPinchRight);

    // Wrist button interaction (left wrist, triggered by right index finger)
    static bool lastWristButtonTouched = false;
    bool wristButtonTouched = false;
    if (left.tracked && right.tracked) {
      glm::vec3 buttonCenter = left.wristPos + left.wristRot * glm::vec3(0.0f, 0.0f, 0.05f);
      float dist = glm::length(right.indexTipPos - buttonCenter);
      wristButtonTouched = dist < 0.03f;
    }

    if (wristButtonTouched && !lastWristButtonTouched) {
      onWristButtonPressed();
    }
    lastWristButtonTouched = wristButtonTouched;


  }
}

void VkViewer::copyToXrSwapchain(VkCommandBuffer cmd)
{
  if(!m_xrInitialized || !m_xr || m_xrColorImage == VK_NULL_HANDLE)
    return;

  // Get the source image (our rendered GBuffer)
  VkImage srcColorImage = m_gBuffers.getColorImage(COLOR_MAIN);
  VkImage srcMotionImage = m_gBuffers.getColorImage(COLOR_MOTION);
  VkImage srcDepthImage = m_gBuffers.getDepthImage();

  VkExtent2D extent = m_xr->getFullExtent();

  // Transition XR images to transfer dst layout
  {
    std::vector<VkImageMemoryBarrier> barriers;
    
    // Color
    VkImageMemoryBarrier colorBarrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    colorBarrier.srcAccessMask = 0;
    colorBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    colorBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    colorBarrier.image = m_xrColorImage;
    colorBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barriers.push_back(colorBarrier);

    // Depth
    VkImageMemoryBarrier depthBarrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    depthBarrier.srcAccessMask = 0;
    depthBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    depthBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    depthBarrier.image = m_xrDepthImage;
    depthBarrier.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    barriers.push_back(depthBarrier);

    // Motion
    if (m_xrMotionImage != VK_NULL_HANDLE)
    {
        VkImageMemoryBarrier motionBarrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        motionBarrier.srcAccessMask = 0;
        motionBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        motionBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        motionBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        motionBarrier.image = m_xrMotionImage;
        motionBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barriers.push_back(motionBarrier);
    }

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, static_cast<uint32_t>(barriers.size()), barriers.data());
  }

  // Transition source images to transfer src layout
  // Note: Ray tracing uses VK_IMAGE_LAYOUT_GENERAL, rasterization uses COLOR_ATTACHMENT_OPTIMAL
  // We use GENERAL as the source layout since RTX path uses it
  {
    std::vector<VkImageMemoryBarrier> barriers;

    // Color
    VkImageMemoryBarrier colorBarrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    colorBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    colorBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    colorBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    colorBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    colorBarrier.image = srcColorImage;
    colorBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barriers.push_back(colorBarrier);

    // Depth
    VkImageMemoryBarrier depthBarrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    depthBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    depthBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    depthBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    depthBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    depthBarrier.image = srcDepthImage;
    depthBarrier.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    barriers.push_back(depthBarrier);

    // Motion (only if target exists)
    if (m_xrMotionImage != VK_NULL_HANDLE)
    {
        VkImageMemoryBarrier motionBarrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        motionBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        motionBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        motionBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        motionBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        motionBarrier.image = srcMotionImage;
        motionBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barriers.push_back(motionBarrier);
    }

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, static_cast<uint32_t>(barriers.size()), barriers.data());
  }

  // Blit color image (using blit instead of copy to handle UNORM->SRGB format conversion)
  // The XR swapchain uses VK_FORMAT_R8G8B8A8_SRGB, so the GPU will automatically
  // apply linear->sRGB gamma correction during the blit operation
  {
    VkImageBlit region = {};
    region.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    region.srcSubresource.mipLevel       = 0;
    region.srcSubresource.baseArrayLayer = 0;
    region.srcSubresource.layerCount     = 1;
    region.srcOffsets[0]                 = {0, 0, 0};
    region.srcOffsets[1]                 = {static_cast<int32_t>(extent.width), static_cast<int32_t>(extent.height), 1};
    region.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    region.dstSubresource.mipLevel       = 0;
    region.dstSubresource.baseArrayLayer = 0;
    region.dstSubresource.layerCount     = 1;
    region.dstOffsets[0]                 = {0, 0, 0};
    region.dstOffsets[1]                 = {static_cast<int32_t>(extent.width), static_cast<int32_t>(extent.height), 1};

    vkCmdBlitImage(cmd, srcColorImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_xrColorImage,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_NEAREST);
  }

  // Copy depth image
  {
    VkImageCopy region            = {};
    region.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
    region.srcSubresource.mipLevel       = 0;
    region.srcSubresource.baseArrayLayer = 0;
    region.srcSubresource.layerCount     = 1;
    region.srcOffset                     = {0, 0, 0};
    region.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
    region.dstSubresource.mipLevel       = 0;
    region.dstSubresource.baseArrayLayer = 0;
    region.dstSubresource.layerCount     = 1;
    region.dstOffset                     = {0, 0, 0};
    region.extent                        = {extent.width, extent.height, 1};

    vkCmdCopyImage(cmd, srcDepthImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_xrDepthImage,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  }

  // Copy motion image
  if (m_xrMotionImage != VK_NULL_HANDLE)
  {
    VkImageCopy region            = {};
    region.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    region.srcSubresource.mipLevel       = 0;
    region.srcSubresource.baseArrayLayer = 0;
    region.srcSubresource.layerCount     = 1;
    region.srcOffset                     = {0, 0, 0};
    region.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    region.dstSubresource.mipLevel       = 0;
    region.dstSubresource.baseArrayLayer = 0;
    region.dstSubresource.layerCount     = 1;
    region.dstOffset                     = {0, 0, 0};
    region.extent                        = {extent.width, extent.height, 1};

    vkCmdCopyImage(cmd, srcMotionImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_xrMotionImage,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  }

  // Transition XR images to attachment optimal for the compositor
  {
    std::vector<VkImageMemoryBarrier> barriers;

    // Color
    VkImageMemoryBarrier colorBarrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    colorBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    colorBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    colorBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    colorBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorBarrier.image = m_xrColorImage;
    colorBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barriers.push_back(colorBarrier);

    // Depth
    VkImageMemoryBarrier depthBarrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    depthBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    depthBarrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    depthBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    depthBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthBarrier.image = m_xrDepthImage;
    depthBarrier.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    barriers.push_back(depthBarrier);

    // Motion
    if (m_xrMotionImage != VK_NULL_HANDLE)
    {
        VkImageMemoryBarrier motionBarrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        motionBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        motionBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT; // Motion is color attachment in XR? Spec says "XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT" is allowed.
                                                                           // Actually usage is SAMPLED usually for compositor?
                                                                           // Spec: "The motion vector data is stored in the motionVectorSubImage’s RGB channels"
                                                                           // The compositor likely reads it as sampled image.
                                                                           // VK_ACCESS_SHADER_READ_BIT might be safer if we knew what compositor does.
                                                                           // But standard for swapchain submission is often COLOR_ATTACHMENT_OPTIMAL or PRESENT_SRC_KHR.
                                                                           // For XR, it's usually COLOR_ATTACHMENT_OPTIMAL.
        motionBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        motionBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        motionBarrier.image = m_xrMotionImage;
        motionBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barriers.push_back(motionBarrier);
    }

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, 0,
                         0, nullptr, 0, nullptr, static_cast<uint32_t>(barriers.size()), barriers.data());
  }

  // Transition source images back to GENERAL layout for next frame's ray tracing
  {
    std::vector<VkImageMemoryBarrier> barriers;

    // Color
    VkImageMemoryBarrier colorBarrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    colorBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    colorBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    colorBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    colorBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    colorBarrier.image = srcColorImage;
    colorBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barriers.push_back(colorBarrier);

    // Depth
    VkImageMemoryBarrier depthBarrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    depthBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    depthBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    depthBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    depthBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    depthBarrier.image = srcDepthImage;
    depthBarrier.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    barriers.push_back(depthBarrier);

    // Motion (only if target exists)
    if (m_xrMotionImage != VK_NULL_HANDLE)
    {
        VkImageMemoryBarrier motionBarrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        motionBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        motionBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        motionBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        motionBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        motionBarrier.image = srcMotionImage;
        motionBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barriers.push_back(motionBarrier);
    }

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0,
                         0, nullptr, 0, nullptr, static_cast<uint32_t>(barriers.size()), barriers.data());
  }
}

#endif  // WITH_OPENXR

void VkViewer::enablePlySequencePlayback(const std::filesystem::path& dirPath)
{
    // Close any existing animation
    if (m_animationController) {
        m_animationController->closeSequence();
        m_animationController.reset();
    }

    // Initialize animation UI if not already done
    if (!m_animationUI) {
        m_animationUI = std::make_unique<AnimationUI>();
    }

    // Create animation controller and load sequence
    m_animationController = std::make_shared<AnimationController>();
    if (!m_animationController->loadSequence(dirPath)) {
        LOGE("Failed to load PLY sequence from: %s\n", dirPath.string().c_str());
        m_animationController.reset();
        return;
    }

    // Initialize animation UI
    m_animationUI->initialize(m_animationController);

    m_isAnimationPlaying = true;

    LOGI("PLY sequence loaded: %zu frames from %s\n",
         m_animationController->getTotalFrames(), dirPath.string().c_str());
}

void VkViewer::updateAnimation(float deltaTime)
{
    if (!m_animationController || !m_isAnimationPlaying) {
        return;
    }

    // Update animation state
    m_animationController->update(deltaTime);

    // Get current frame and update splat set for rendering
    SplatSet currentFrame;
    if (m_animationController->getCurrentFrameData(currentFrame)) {
        // Replace the current splat set with the animation frame
        m_splatSet = std::move(currentFrame);

        // Trigger VRAM update on next frame
        m_requestUpdateSplatData = true;
    }
}

}  // namespace vk_viewer

// Include the split implementation files
#include "vk_viewer_frame.cpp"
#include "vk_viewer_render.cpp"
#include "vk_viewer_frame_ubo.cpp"
#include "vk_viewer_sorting.cpp"
#include "vk_viewer_shaders.cpp"
#include "vk_viewer_pipelines.cpp"
#include "vk_viewer_rtx.cpp"
#include "vk_viewer_postprocess.cpp"
#include "vk_viewer_multiview.cpp"
#include "vk_viewer_chunk_culling.cpp"
#include "vk_viewer_stochasticgs.cpp"

#include "vk_viewer_dlss_rr.cpp"
#include "vk_viewer_video.cpp"
