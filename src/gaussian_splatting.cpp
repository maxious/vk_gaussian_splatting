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

#include "gaussian_splatting.h"
#include "hdr_support.h"
#include "utilities.h"

#include <nvutils/logger.hpp>
#include <nvvk/barriers.hpp>

#define GLM_ENABLE_SWIZZLE
#include <glm/gtc/packing.hpp>  // Required for half-float operations

#include <nvvk/check_error.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/sbt_generator.hpp>
#include <nvvk/formats.hpp>

namespace vk_gaussian_splatting {

GaussianSplatting::GaussianSplatting(nvutils::ProfilerManager* profilerManager, nvutils::ParameterRegistry* parameterRegistry)
    : m_profilerManager(profilerManager)
    , m_parameterRegistry(parameterRegistry)
    , cameraManip(std::make_shared<nvutils::CameraManipulator>()) {

    };

GaussianSplatting::~GaussianSplatting(){
    // all threads must be stopped,
    // work done in onDetach(),
    // could be done here, same result
};

void GaussianSplatting::onAttach(nvapp::Application* app)
{
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

void GaussianSplatting::onDetach()
{
#ifdef WITH_OPENXR
  shutdownOpenXR();
#endif

  // stops the threads
  m_splatLoader.shutdown();
  m_cpuSorter.shutdown();
  // release scene and rendering related resources
  deinitAll();
  // release application wide related resources
  m_splatSetVk.deinit();
  m_meshSetVk.deinit();
  m_vdzMesh.cleanup();
  
  if(m_depthManager)
  {
    m_depthManager->cleanup();
    m_depthManager.reset();
  }
  m_depthClient.reset();
#ifdef WITH_VIDEO_DECODER
  m_videoDecoder.reset();
  // Clean up video texture
  if(m_videoTexture.view != VK_NULL_HANDLE)
  {
    vkDestroyImageView(m_device, m_videoTexture.view, nullptr);
    m_videoTexture.view = VK_NULL_HANDLE;
  }
  if(m_videoTexture.image.image != VK_NULL_HANDLE)
  {
    m_alloc.destroyImage(m_videoTexture.image);
    m_videoTexture.image = {};
  }
#endif

  if(m_dummyTextureArray.view != VK_NULL_HANDLE)
  {
    vkDestroyImageView(m_device, m_dummyTextureArray.view, nullptr);
    m_dummyTextureArray.view = VK_NULL_HANDLE;
  }
  if(m_dummyTextureArray.image.image != VK_NULL_HANDLE)
  {
    m_alloc.destroyImage(m_dummyTextureArray.image);
    m_dummyTextureArray.image = {};
  }

  m_profilerGpuTimer.deinit();
  m_profilerManager->destroyTimeline(m_profilerTimeline);
  m_profilerTimeline = nullptr;
  m_gBuffers.deinit();
  m_samplerPool.releaseSampler(m_sampler);
  m_samplerPool.deinit();
  m_uploader.deinit();
  m_alloc.deinit();
}

void GaussianSplatting::onResize(VkCommandBuffer cmd, const VkExtent2D& viewportSize)
{
  m_viewSize = {viewportSize.width, viewportSize.height};
  NVVK_CHECK(m_gBuffers.update(cmd, viewportSize));
  updateRtDescriptorSet();
  updateDescriptorSetPostProcessing();
  resetFrameCounter();
}

void GaussianSplatting::onPreRender()
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
      updateRtDescriptorSet();
      updateDescriptorSetPostProcessing();
      resetFrameCounter();
      m_app->submitAndWaitTempCmdBuffer(cmd);

      // Skip rendering this frame to let descriptor sets stabilize
      m_xrResizedThisFrame = true;
    }

  }
#endif
}

void GaussianSplatting::deinitAll()
{
  vkDeviceWaitIdle(m_device);

#ifdef WITH_DLSS_RR
  shutdownDlssRR();
#endif

  m_canCollectReadback = false;
  deinitScene();
  m_splatSetVk.resetTransform();
  m_splatSetVk.deinitDataStorage();
  m_splatSetVk.rtxDeinitSplatModel();
  m_splatSetVk.rtxDeinitAccelerationStructures();
  m_meshSetVk.deinitDataStorage();
  m_meshSetVk.rtxDeinitAccelerationStructures();
  m_lightSet.deinit();
  m_cameraSet.deinit();
  deinitShaders();
  deinitPipelines();
  deinitRendererBuffers();
  resetRenderSettings();
  // record default cam for reset in UI
  m_cameraSet.setCamera(Camera());
  // record default cam for reset in UI
  m_cameraSet.setHomePreset(m_cameraSet.getCamera());
}

bool GaussianSplatting::initAll()
{
  vkDeviceWaitIdle(m_device);

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
  initPipelines();

  // RTX specifics
  m_splatSetVk.rtxInitSplatModel(m_splatSet, prmRtxData.useTlasInstances, prmRtxData.useAABBs, prmRtxData.compressBlas,
                                 prmRtx.kernelDegree, prmRtx.kernelMinResponse, prmRtx.kernelAdaptiveClamping);

  m_splatSetVk.rtxInitAccelerationStructures(m_splatSet);

  initRtDescriptorSet();
  initRtPipeline();

  // Post processing
  initDescriptorSetPostProcessing();
  initPipelinePostProcessing();



  return true;
}

void GaussianSplatting::enableDepthRendering(const std::string& host, int port, const std::string& videoPath)
{
  if(!m_depthClient)
  {
    m_depthClient = std::make_unique<DepthStreamClient>();
    m_depthClient->setBackendAddress(host, port);
    m_depthClient->setDepthCallback([this](const DepthFrame& frame) {
      // Frame is already added to buffer by client
      // We will pick it up in updateDepthRendering on the main thread
    });
  }
  else
  {
    m_depthClient->setBackendAddress(host, port);
  }

  if(!m_depthManager)
  {
    m_depthManager = std::make_unique<DepthTextureManager>();
    // Initialize with device objects
    m_depthManager->initialize(m_device, m_app->getPhysicalDevice(), m_app->getQueue(0).queue, &m_alloc);
  }

#ifdef WITH_VIDEO_DECODER
  // Initialize video decoder for local VDZ+video playback
  m_videoDecoder = std::make_unique<VideoDecoder>();
  if(!m_videoDecoder->open(videoPath))
  {
    LOGE("Failed to open video file for decoding: %s\n", videoPath.c_str());
    m_videoDecoder.reset();
  }
  else
  {
    m_videoDecoder->startDecoding();
    LOGI("Video decoder initialized for file: %s\n", videoPath.c_str());
  }
#else
  // Upload video
  DepthStreamClient::SessionInfo session;
  if(!m_depthClient->uploadVideo(videoPath, session))
  {
    LOGE("Failed to upload video\n");
    return;
  }

  // Connect WebSocket
  if(!m_depthClient->connectWebSocket(session.sessionId))
  {
    LOGE("Failed to connect depth stream\n");
    return;
  }
#endif

  m_enableDepthRendering = true;
  LOGI("Depth rendering enabled for session: %s\n", videoPath.c_str());
}

void GaussianSplatting::enableVideoDepthPlayback(const std::string& videoPath, const std::string& vdzPath)
{
  if(!m_depthManager)
  {
    m_depthManager = std::make_unique<DepthTextureManager>();
    m_depthManager->initialize(m_device, m_app->getPhysicalDevice(), m_app->getQueue(0).queue, &m_alloc);
  }

  m_vdzSequence = std::make_unique<VDZSequenceLoader>();
  if(!m_vdzSequence->open(vdzPath))
  {
    LOGE("Failed to open VDZ sequence: %s\n", vdzPath.c_str());
    m_vdzSequence.reset();
    return;
  }

#ifdef WITH_VIDEO_DECODER
  m_videoDecoder = std::make_unique<VideoDecoder>();
  if(!m_videoDecoder->open(videoPath))
  {
    LOGE("Failed to open video file: %s\n", videoPath.c_str());
    m_videoDecoder.reset();
    m_vdzSequence.reset();
    return;
  }
  m_videoDecoder->startDecoding();
#else
  LOGE("Video decoder not available - build with ENABLE_VIDEO_DECODER=ON\n");
  m_vdzSequence.reset();
  return;
#endif

  m_videoDepthPlaybackMode = true;
  m_enableDepthRendering = true;
  m_playbackStartTime = std::chrono::steady_clock::now();
  m_playbackTimeOffset = 0.0;
  m_playbackPaused = false;
  m_lastVdzFrameIndex = SIZE_MAX;

  // Ensure shaders and pipelines are initialized for VDZ rendering
  // This is needed when no radiance field is loaded
  if(!m_shaders.valid)
  {
    initShaders();
    initRendererBuffers();
    initPipelines();
  }

  // Load and upload first depth frame immediately
  DepthFrame firstFrame;
  if(m_vdzSequence->getFrame(0, firstFrame))
  {
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    m_depthManager->uploadDepthFrame(firstFrame, cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    m_lastVdzFrameIndex = 0;
    
    // Set VDZ rendering parameters based on depth data
    prmFrame.vdzZMaxClip = firstFrame.zMax > 0.0f ? firstFrame.zMax : 2.0f;
    prmFrame.vdzAspect = static_cast<float>(firstFrame.width) / static_cast<float>(firstFrame.height);
    // Use good defaults for depth visualization
    prmFrame.vdzZScale = 10.0f;
    prmFrame.vdzZBias = 2.0f;
    prmFrame.vdzZGamma = 5.0f;
    prmFrame.vdzZMaxClip = 0.2f;
    prmFrame.vdzPlaneScale = 1.4f;
    prmFrame.vdzEdgeThreshold = 1.0f;
    
    LOGI("First depth frame uploaded: %ux%u, zMax=%.2f, aspect=%.3f\n", 
         firstFrame.width, firstFrame.height, firstFrame.zMax, prmFrame.vdzAspect);
  }

  LOGI("Video+Depth playback enabled: %s + %s (%zu depth frames)\n",
       videoPath.c_str(), vdzPath.c_str(), m_vdzSequence->getFrameCount());
}

void GaussianSplatting::updateDepthRendering(VkCommandBuffer cmd)
{
  if(!m_enableDepthRendering)
  {
    return;
  }

#ifdef WITH_VIDEO_DECODER
  // Handle video decoder case
  if(m_videoDecoder)
  {
    // Skip frame updates when paused
    if(m_playbackPaused)
    {
      return;
    }
    
    // Get next available frame from decoder
    DecodedFrame videoFrame;
    if(m_videoDecoder->getNextFrame(videoFrame))
    {
      if (videoFrame.width > 0 && videoFrame.height > 0)
      {
          bool updateDescriptor = false;

          // Check if texture needs (re)creation
          if(m_videoTexture.width != videoFrame.width || m_videoTexture.height != videoFrame.height)
          {
              // Destroy old - wait for GPU to finish using resources
              vkDeviceWaitIdle(m_device);
              if(m_videoTexture.view != VK_NULL_HANDLE) { vkDestroyImageView(m_device, m_videoTexture.view, nullptr); m_videoTexture.view = VK_NULL_HANDLE; }
              if(m_videoTexture.image.image != VK_NULL_HANDLE) { m_alloc.destroyImage(m_videoTexture.image); m_videoTexture.image = {}; }

              // Create new
              VkImageCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
              info.imageType = VK_IMAGE_TYPE_2D;
              info.format = VK_FORMAT_R8G8B8A8_UNORM;
              info.extent = {static_cast<uint32_t>(videoFrame.width), static_cast<uint32_t>(videoFrame.height), 1};
              info.mipLevels = 1;
              info.arrayLayers = 1;
              info.samples = VK_SAMPLE_COUNT_1_BIT;
              info.tiling = VK_IMAGE_TILING_OPTIMAL;
              info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
              info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
              info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
              
              m_alloc.createImage(m_videoTexture.image, info);

              VkImageViewCreateInfo viewInfo = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
              viewInfo.image = m_videoTexture.image.image;
              viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
              viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
              viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
              viewInfo.subresourceRange.baseMipLevel = 0;
              viewInfo.subresourceRange.levelCount = 1;
              viewInfo.subresourceRange.baseArrayLayer = 0;
              viewInfo.subresourceRange.layerCount = 1;
              
              vkCreateImageView(m_device, &viewInfo, nullptr, &m_videoTexture.view);
              
              m_videoTexture.width = videoFrame.width;
              m_videoTexture.height = videoFrame.height;
              
              updateDescriptor = true;

              // Transition to TRANSFER_DST
              VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
              barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
              barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
              barrier.srcAccessMask = 0;
              barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
              barrier.image = m_videoTexture.image.image;
              barrier.subresourceRange = viewInfo.subresourceRange;
              
              vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 
                  0, 0, nullptr, 0, nullptr, 1, &barrier);
          }
          else
          {
             // Transition to TRANSFER_DST for update
              VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
              barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
              barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
              barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
              barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
              barrier.image = m_videoTexture.image.image;
              barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
              
              vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 
                  0, 0, nullptr, 0, nullptr, 1, &barrier);
          }

          // Upload using StagingUploader
          size_t bufferSize = videoFrame.data.size();
          m_uploader.appendImage(m_videoTexture.image, bufferSize, videoFrame.data.data(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
          m_uploader.cmdUploadAppended(cmd);
               
          // Transition to SHADER_READ_ONLY
          {
              VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
              barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
              barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
              barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
              barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
              barrier.image = m_videoTexture.image.image;
              barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
              
              vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 
                  0, 0, nullptr, 0, nullptr, 1, &barrier);
          }
          
          if(updateDescriptor && m_descriptorSet != VK_NULL_HANDLE)
          {
              // Update descriptor set
              VkDescriptorImageInfo imageInfo{};
              imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
              imageInfo.imageView = m_videoTexture.view;
              imageInfo.sampler = m_sampler;

              VkWriteDescriptorSet write{};
              write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
              write.dstSet = m_descriptorSet;
              write.dstBinding = BINDING_VDZ_VIDEO_TEXTURE;
              write.dstArrayElement = 0;
              write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
              write.descriptorCount = 1;
              write.pImageInfo = &imageInfo;

              vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
          }
      }
      
      LOGD("Got video frame: %dx%d @ %.3f s\n", videoFrame.width, videoFrame.height, videoFrame.timestamp);
      
      if(m_videoDepthPlaybackMode && m_vdzSequence && m_vdzSequence->isOpen())
      {
        uint32_t videoTimestampMs = static_cast<uint32_t>(videoFrame.timestamp * 1000.0);
        size_t depthFrameIdx = m_vdzSequence->getFrameIndexForTimestamp(videoTimestampMs);
        
        if(depthFrameIdx != m_lastVdzFrameIndex)
        {
          DepthFrame depthFrame;
          if(m_vdzSequence->getFrame(depthFrameIdx, depthFrame))
          {
            if(m_depthManager)
            {
              m_depthManager->uploadDepthFrame(depthFrame, cmd);
              m_lastVdzFrameIndex = depthFrameIdx;
              m_depthFrameCounter++;
              
              if(m_depthFrameCounter % 30 == 0)
              {
                LOGI("Synced depth frame #%zu: %ux%u @ %u ms (video: %u ms)\n",
                     depthFrameIdx, depthFrame.width, depthFrame.height,
                     depthFrame.timestampMs, videoTimestampMs);
  if(m_xr && m_xr->isSpaceWarpSupported())
  {
    m_shaderMacros.push_back({"WITH_SPACE_WARP", "1"});
  }
}

            }
          }
        }
      }
    }
  }
  else
#endif
  {
    // Handle streaming case
    if(!m_depthClient)
    {
      return;
    }

    // Request depth for current time
    uint64_t timestampMs = static_cast<uint64_t>(prmFrame.currentTime * 1000.0f);

    m_depthClient->update(prmFrame.currentTime * 1000.0f, m_depthClient->getFps());

    static int frameCounter = 0;  // For selective debug logging
    DepthFrame frame;
    if(m_depthClient->getFrame(timestampMs, frame)) {
        if(m_depthManager) {
            m_depthManager->uploadDepthFrame(frame, cmd);
            frameCounter++;

            // Log every 30th frame for debugging
            if(frameCounter % 30 == 0) {
                LOGI("Depth frame #%d uploaded: %dx%d @ %u ms (req: %llu ms)\n",
                     frameCounter, frame.width, frame.height,
                     frame.timestampMs, timestampMs);
            }
        }
    }

    m_depthStats = m_depthClient->getStats();
  }
}

void GaussianSplatting::deinitScene()
{
  m_splatSet.clear();
  m_splatSetPending.clear();
  m_radianceFields.clear();
  m_pendingLoadFilename = "";
}

void GaussianSplatting::benchmarkAdvance()
{
  LOGI("BENCHMARK_ADV %d {\n", m_benchmarkId);
  LOGI(" Memory Scene; Host used \t%zu; Device Used \t%zu; Device Allocated \t%zu; (bytes)\n",
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
bool GaussianSplatting::queryOpenXrVulkanExtensions(std::vector<std::string>& outInstanceExtensions,
                                                     std::vector<std::string>& outDeviceExtensions)
{
  if(!m_xr)
  {
    m_xr = std::make_unique<GsOpenXr>();
  }
  return m_xr->queryRequiredVulkanExtensions(outInstanceExtensions, outDeviceExtensions);
}

void GaussianSplatting::initializeOpenXR()
{
  if(!m_xr)
  {
    m_xr = std::make_unique<GsOpenXr>();
  }

  // Use SRGB format for XR swapchain - GPU will automatically convert linear->sRGB on write
  // This is the correct way to handle color space for VR displays
  VkFormat xrColorFormat = VK_FORMAT_R8G8B8A8_SRGB;

  if(!m_xr->initialize(m_app->getInstance(), m_app->getPhysicalDevice(), m_app->getDevice(),
                       m_app->getQueue(0).familyIndex, 0, xrColorFormat, m_depthFormat))
  {
    LOGE("Failed to initialize OpenXR\n");
    m_xr.reset();
    m_xrInitialized = false;
    m_useXrHmd      = false;
    return;
  }

  m_xrInitialized = true;

  // Get the XR resolution and resize our buffers to match
  VkExtent2D xrExtent = m_xr->getFullExtent();
  LOGI("OpenXR initialized. Full extent: %dx%d\n", xrExtent.width, xrExtent.height);

  // Force SBS mode when XR is enabled
  m_renderSBS = true;
}

void GaussianSplatting::shutdownOpenXR()
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

void GaussianSplatting::updateXrLocomotion(float deltaTime)
{
  if(!m_xrInitialized || !m_xr || !m_xr->hasControllers())
    return;

  m_xr->pollControllerInput();
  const auto& locomotion = m_xr->getLocomotionInput();

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

void GaussianSplatting::copyToXrSwapchain(VkCommandBuffer cmd)
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

}  // namespace vk_gaussian_splatting

// Include the split implementation files
#include "gaussian_splatting_render.cpp"
#include "gaussian_splatting_frame_ubo.cpp"
#include "gaussian_splatting_sorting.cpp"
#include "gaussian_splatting_shaders.cpp"
#include "gaussian_splatting_pipelines.cpp"
#include "gaussian_splatting_rtx.cpp"
#include "gaussian_splatting_postprocess.cpp"
#include "gaussian_splatting_multiview.cpp"

#include "gaussian_splatting_dlss_rr.cpp"
