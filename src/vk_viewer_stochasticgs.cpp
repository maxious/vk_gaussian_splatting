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

// This file is included from vk_viewer.cpp - do not compile separately

#include <algorithm>
#include <cmath>
#include <vector>

namespace vk_viewer {

//--------------------------------------------------------------------------------------------------
// Initialize the stochastic GS compute pipelines (clear, accumulate, resolve).
// Pattern follows vk_viewer_rtx.cpp (own descriptor set, pipeline layout).
//
void VkViewer::initStochasticPipelines()
{
  if(!isStochasticGsSupported())
  {
    LOGW("Stochastic GS not supported on this device - requires NVIDIA GPU with shaderBufferInt64Atomics\n");
    return;
  }

  const uint32_t width  = static_cast<uint32_t>(m_viewSize.x);
  const uint32_t height = static_cast<uint32_t>(m_viewSize.y);
  if(width == 0 || height == 0)
  {
    LOGW("Stochastic GS init skipped - invalid viewport size (%u x %u)\n", width, height);
    return;
  }
  m_stochastic.descriptorBindings.clear();
  
  // Ensure the shared set 0 layout exists. Create it if needed since we depend on
  // set 0 being present for FrameInfo UBO and splat data bindings.
  if(m_descriptorSetLayout == VK_NULL_HANDLE)
  {
    LOGW("Stochastic GS: shared set 0 layout not ready, will retry next frame\n");
    return;
  }
  
  // Ensure renderer buffers are initialized (m_frameInfoBuffer is a reliable sentinel)
  if(m_frameInfoBuffer.buffer == VK_NULL_HANDLE)
  {
    LOGW("Stochastic GS: renderer buffers not ready, will retry next frame\n");
    return;
  }
  
  // Two separate 32-bit buffers instead of one 64-bit buffer.
  // 64-bit InterlockedMin on RWStructuredBuffer<uint64_t> is broken in Slang/SPIR-V.
  m_stochastic.descriptorBindings.addBinding(BINDING_STOCHASTIC_DEPTH_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                             VK_SHADER_STAGE_COMPUTE_BIT);
  m_stochastic.descriptorBindings.addBinding(BINDING_STOCHASTIC_INDEX_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                             VK_SHADER_STAGE_COMPUTE_BIT);
  m_stochastic.descriptorBindings.addBinding(BINDING_STOCHASTIC_OCCLUSION_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                             VK_SHADER_STAGE_COMPUTE_BIT);
  m_stochastic.descriptorBindings.addBinding(BINDING_STOCHASTIC_OUTPUT_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                                             VK_SHADER_STAGE_COMPUTE_BIT);
  m_stochastic.descriptorBindings.addBinding(BINDING_STOCHASTIC_ACCUMULATION_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                                             VK_SHADER_STAGE_COMPUTE_BIT);
  NVVK_CHECK(m_stochastic.descriptorBindings.createDescriptorSetLayout(m_device, 0, &m_stochastic.descriptorSetLayout));
  NVVK_DBG_NAME(m_stochastic.descriptorSetLayout);

  // Pipeline layout: set 0 (m_descriptorSetLayout: FrameInfo + splat data) + set 1 (stochastic)
  const VkPushConstantRange pcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(shaderio::PushConstant)};

  std::vector<VkDescriptorSetLayout> setLayouts = {m_descriptorSetLayout, m_stochastic.descriptorSetLayout};

  VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineLayoutInfo.setLayoutCount         = static_cast<uint32_t>(setLayouts.size());
  pipelineLayoutInfo.pSetLayouts            = setLayouts.data();
  pipelineLayoutInfo.pushConstantRangeCount = 1;
  pipelineLayoutInfo.pPushConstantRanges    = &pcRange;
  NVVK_CHECK(vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_stochastic.pipelineLayout));
  NVVK_DBG_NAME(m_stochastic.pipelineLayout);

  std::vector<VkDescriptorPoolSize> poolSizes;
  m_stochastic.descriptorBindings.appendPoolSizes(poolSizes);
  VkDescriptorPoolCreateInfo poolInfo{
      .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets       = 1,
      .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
      .pPoolSizes    = poolSizes.data(),
  };
  NVVK_CHECK(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_stochastic.descriptorPool));
  NVVK_DBG_NAME(m_stochastic.descriptorPool);

  VkDescriptorSetAllocateInfo allocInfo{
      .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool     = m_stochastic.descriptorPool,
      .descriptorSetCount = 1,
      .pSetLayouts        = &m_stochastic.descriptorSetLayout,
  };
  NVVK_CHECK(vkAllocateDescriptorSets(m_device, &allocInfo, &m_stochastic.descriptorSet));
  NVVK_DBG_NAME(m_stochastic.descriptorSet);

  // Depth buffer: 4 bytes (uint32) per pixel for 32-bit atomicMin
  const VkDeviceSize pixelStride = 4ULL;
  const VkDeviceSize bufferSize  = pixelStride * width * height;
  NVVK_CHECK(m_alloc.createBuffer(m_stochastic.depthBuffer, bufferSize,
                                  VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
                                  VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE));
  NVVK_DBG_NAME(m_stochastic.depthBuffer.buffer);

  NVVK_CHECK(m_alloc.createBuffer(m_stochastic.indexBuffer, bufferSize,
                                  VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
                                  VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE));
  NVVK_DBG_NAME(m_stochastic.indexBuffer.buffer);

  // Previous-frame block max-depth map (one uint per 8x8 pixel block)
  const uint32_t blockCols = (width + 7u) / 8u;
  const uint32_t blockRows = (height + 7u) / 8u;
  m_stochastic.occlusionBlockCols = blockCols;
  m_stochastic.occlusionBlockRows = blockRows;
  const VkDeviceSize occlusionSize = sizeof(uint32_t) * blockCols * blockRows;
  NVVK_CHECK(m_alloc.createBuffer(m_stochastic.occlusionBuffer, occlusionSize,
                                  VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
                                  VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE));
  NVVK_DBG_NAME(m_stochastic.occlusionBuffer.buffer);
  m_stochastic.occlusionMapValid = false;

  // Output image (matching display color format for blit compatibility)
  VkImageCreateInfo outputInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  outputInfo.imageType     = VK_IMAGE_TYPE_2D;
  // Must match the R16G16B16A16_SFLOAT storage image view created below (and the
  // float4 writes in the resolve shader); m_colorFormat is R8G8B8A8_UNORM, which
  // is not a view-compatible aliasing.
  outputInfo.format        = VK_FORMAT_R16G16B16A16_SFLOAT;
  outputInfo.extent        = {width, height, 1};
  outputInfo.mipLevels     = 1;
  outputInfo.arrayLayers   = 1;
  outputInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
  outputInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
  outputInfo.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  outputInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
  outputInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  NVVK_CHECK(m_alloc.createImage(m_stochastic.outputImage, outputInfo));
  NVVK_DBG_NAME(m_stochastic.outputImage.image);

  VkImageViewCreateInfo outputViewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  outputViewInfo.image            = m_stochastic.outputImage.image;
  outputViewInfo.viewType         = VK_IMAGE_VIEW_TYPE_2D;
  outputViewInfo.format           = VK_FORMAT_R16G16B16A16_SFLOAT;
  outputViewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  NVVK_CHECK(vkCreateImageView(m_device, &outputViewInfo, nullptr, &m_stochastic.outputImageView));
  NVVK_DBG_NAME(m_stochastic.outputImageView);
  m_stochastic.outputSize = {width, height};

  VkImageCreateInfo accumInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  accumInfo.imageType     = VK_IMAGE_TYPE_2D;
  accumInfo.format        = VK_FORMAT_R16G16B16A16_SFLOAT;  // must match accumulationImageView
  accumInfo.extent        = {width, height, 1};
  accumInfo.mipLevels     = 1;
  accumInfo.arrayLayers   = 1;
  accumInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
  accumInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
  accumInfo.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  accumInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
  accumInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  NVVK_CHECK(m_alloc.createImage(m_stochastic.accumulationImage, accumInfo));
  NVVK_DBG_NAME(m_stochastic.accumulationImage.image);

  VkImageViewCreateInfo accumViewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  accumViewInfo.image            = m_stochastic.accumulationImage.image;
  accumViewInfo.viewType         = VK_IMAGE_VIEW_TYPE_2D;
  accumViewInfo.format           = VK_FORMAT_R16G16B16A16_SFLOAT;
  accumViewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  NVVK_CHECK(vkCreateImageView(m_device, &accumViewInfo, nullptr, &m_stochastic.accumulationImageView));
  NVVK_DBG_NAME(m_stochastic.accumulationImageView);

  {
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    nvvk::cmdImageMemoryBarrier(cmd, {m_stochastic.outputImage.image, VK_IMAGE_LAYOUT_UNDEFINED,
                                      VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}});
    nvvk::cmdImageMemoryBarrier(cmd, {m_stochastic.accumulationImage.image, VK_IMAGE_LAYOUT_UNDEFINED,
                                      VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}});

    // Zero the occlusion map so an accidental early read sees the invalid sentinel.
    vkCmdFillBuffer(cmd, m_stochastic.occlusionBuffer.buffer, 0, occlusionSize, 0u);
    VkBufferMemoryBarrier occBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    occBarrier.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
    occBarrier.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    occBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    occBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    occBarrier.buffer              = m_stochastic.occlusionBuffer.buffer;
    occBarrier.offset              = 0;
    occBarrier.size                = occlusionSize;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1,
                         &occBarrier, 0, nullptr);

    m_app->submitAndWaitTempCmdBuffer(cmd);
  }

  auto makeComputePipeline = [this](VkShaderModule module) {
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    info.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = module;
    info.stage.pName  = "main";
    info.layout       = m_stochastic.pipelineLayout;
    return info;
  };

  VkComputePipelineCreateInfo clearInfo       = makeComputePipeline(m_shaders.stochasticClearShader);
  VkComputePipelineCreateInfo accumulateInfo  = makeComputePipeline(m_shaders.stochasticAccumulateShader);
  VkComputePipelineCreateInfo resolveInfo     = makeComputePipeline(m_shaders.stochasticResolveShader);
  VkComputePipelineCreateInfo depthReduceInfo = makeComputePipeline(m_shaders.stochasticDepthReduceShader);

  NVVK_CHECK(vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &clearInfo, nullptr, &m_stochastic.clearPipeline));
  NVVK_DBG_NAME(m_stochastic.clearPipeline);

  NVVK_CHECK(
      vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &accumulateInfo, nullptr, &m_stochastic.accumulatePipeline));
  NVVK_DBG_NAME(m_stochastic.accumulatePipeline);

  NVVK_CHECK(
      vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &resolveInfo, nullptr, &m_stochastic.resolvePipeline));
  NVVK_DBG_NAME(m_stochastic.resolvePipeline);

  NVVK_CHECK(vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &depthReduceInfo, nullptr,
                                      &m_stochastic.depthReducePipeline));
  NVVK_DBG_NAME(m_stochastic.depthReducePipeline);

  nvvk::WriteSetContainer writeContainer;
  writeContainer.append(
      m_stochastic.descriptorBindings.getWriteSet(BINDING_STOCHASTIC_DEPTH_BUFFER, m_stochastic.descriptorSet),
      m_stochastic.depthBuffer);
  writeContainer.append(
      m_stochastic.descriptorBindings.getWriteSet(BINDING_STOCHASTIC_INDEX_BUFFER, m_stochastic.descriptorSet),
      m_stochastic.indexBuffer);
  writeContainer.append(
      m_stochastic.descriptorBindings.getWriteSet(BINDING_STOCHASTIC_OCCLUSION_BUFFER, m_stochastic.descriptorSet),
      m_stochastic.occlusionBuffer);
  writeContainer.append(
      m_stochastic.descriptorBindings.getWriteSet(BINDING_STOCHASTIC_OUTPUT_IMAGE, m_stochastic.descriptorSet),
      m_stochastic.outputImageView, VK_IMAGE_LAYOUT_GENERAL);
  writeContainer.append(
      m_stochastic.descriptorBindings.getWriteSet(BINDING_STOCHASTIC_ACCUMULATION_IMAGE, m_stochastic.descriptorSet),
      m_stochastic.accumulationImageView, VK_IMAGE_LAYOUT_GENERAL);
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);

  m_stochastic.initialized = true;
  LOGI("Stochastic GS pipelines initialized (%u x %u)\n", width, height);
}

//--------------------------------------------------------------------------------------------------
// Destroy all stochastic GS resources.
//
void VkViewer::deinitStochasticPipelines()
{
  if(!m_stochastic.initialized && m_stochastic.pipelineLayout == VK_NULL_HANDLE
     && m_stochastic.descriptorPool == VK_NULL_HANDLE && m_stochastic.depthBuffer.buffer == VK_NULL_HANDLE
     && m_stochastic.indexBuffer.buffer == VK_NULL_HANDLE && m_stochastic.occlusionBuffer.buffer == VK_NULL_HANDLE)
  {
    return;
  }

  vkDeviceWaitIdle(m_device);

  if(m_stochastic.clearPipeline != VK_NULL_HANDLE)
  {
    vkDestroyPipeline(m_device, m_stochastic.clearPipeline, nullptr);
    m_stochastic.clearPipeline = VK_NULL_HANDLE;
  }
  if(m_stochastic.accumulatePipeline != VK_NULL_HANDLE)
  {
    vkDestroyPipeline(m_device, m_stochastic.accumulatePipeline, nullptr);
    m_stochastic.accumulatePipeline = VK_NULL_HANDLE;
  }
  if(m_stochastic.resolvePipeline != VK_NULL_HANDLE)
  {
    vkDestroyPipeline(m_device, m_stochastic.resolvePipeline, nullptr);
    m_stochastic.resolvePipeline = VK_NULL_HANDLE;
  }
  if(m_stochastic.depthReducePipeline != VK_NULL_HANDLE)
  {
    vkDestroyPipeline(m_device, m_stochastic.depthReducePipeline, nullptr);
    m_stochastic.depthReducePipeline = VK_NULL_HANDLE;
  }

  if(m_stochastic.pipelineLayout != VK_NULL_HANDLE)
  {
    vkDestroyPipelineLayout(m_device, m_stochastic.pipelineLayout, nullptr);
    m_stochastic.pipelineLayout = VK_NULL_HANDLE;
  }

  m_stochastic.descriptorSet = VK_NULL_HANDLE;
  if(m_stochastic.descriptorPool != VK_NULL_HANDLE)
  {
    vkDestroyDescriptorPool(m_device, m_stochastic.descriptorPool, nullptr);
    m_stochastic.descriptorPool = VK_NULL_HANDLE;
  }
  if(m_stochastic.descriptorSetLayout != VK_NULL_HANDLE)
  {
    vkDestroyDescriptorSetLayout(m_device, m_stochastic.descriptorSetLayout, nullptr);
    m_stochastic.descriptorSetLayout = VK_NULL_HANDLE;
  }
  m_stochastic.descriptorBindings.clear();

  if(m_stochastic.depthBuffer.buffer != VK_NULL_HANDLE)
  {
    m_alloc.destroyBuffer(m_stochastic.depthBuffer);
    m_stochastic.depthBuffer = {};
  }
  if(m_stochastic.indexBuffer.buffer != VK_NULL_HANDLE)
  {
    m_alloc.destroyBuffer(m_stochastic.indexBuffer);
    m_stochastic.indexBuffer = {};
  }
  if(m_stochastic.occlusionBuffer.buffer != VK_NULL_HANDLE)
  {
    m_alloc.destroyBuffer(m_stochastic.occlusionBuffer);
    m_stochastic.occlusionBuffer = {};
  }
  m_stochastic.occlusionBlockCols = 0;
  m_stochastic.occlusionBlockRows = 0;
  m_stochastic.occlusionMapValid  = false;
  m_stochastic.havePrevModelView  = false;

  if(m_stochastic.outputImageView != VK_NULL_HANDLE)
  {
    vkDestroyImageView(m_device, m_stochastic.outputImageView, nullptr);
    m_stochastic.outputImageView = VK_NULL_HANDLE;
  }
  if(m_stochastic.outputImage.image != VK_NULL_HANDLE)
  {
    m_alloc.destroyImage(m_stochastic.outputImage);
    m_stochastic.outputImage = {};
  }
  m_stochastic.outputSize = {0, 0};

  if(m_stochastic.accumulationImageView != VK_NULL_HANDLE)
  {
    vkDestroyImageView(m_device, m_stochastic.accumulationImageView, nullptr);
    m_stochastic.accumulationImageView = VK_NULL_HANDLE;
  }
  if(m_stochastic.accumulationImage.image != VK_NULL_HANDLE)
  {
    m_alloc.destroyImage(m_stochastic.accumulationImage);
    m_stochastic.accumulationImage = {};
  }

  m_stochastic.initialized = false;
}

//--------------------------------------------------------------------------------------------------
// Copy stochastic parameters from prmStochastic into prmFrame (FrameInfo UBO).
//
void VkViewer::updateStochasticFrameInfo(VkCommandBuffer /*cmd*/)
{
  prmFrame.stochasticSamplesPerPixel     = static_cast<int32_t>(prmStochastic.stochasticSamplesPerPixel);
  prmFrame.stochasticMaxSamples          = static_cast<int32_t>(prmStochastic.stochasticMaxSamples);
  prmFrame.stochasticSupersamplingFactor = static_cast<int32_t>(prmStochastic.stochasticSupersamplingFactor);
  prmFrame.stochasticUseGps              = static_cast<int32_t>(prmStochastic.stochasticUseGps);
  prmFrame.stochasticWidth               = static_cast<int32_t>(m_viewSize.x);
  prmFrame.stochasticHeight              = static_cast<int32_t>(m_viewSize.y);

  prmFrame.stochasticContributionThreshold = m_stochastic.contributionThreshold;
  prmFrame.stochasticOcclusionActive       = m_stochastic.occlusionActive ? 1.0f : 0.0f;
  prmFrame.stochasticOcclusionBlockCols    = static_cast<int32_t>(m_stochastic.occlusionBlockCols);
  prmFrame.stochasticOcclusionBlockRows    = static_cast<int32_t>(m_stochastic.occlusionBlockRows);
  prmFrame.stochasticPrevModelViewMatrix   = m_stochastic.prevModelViewMatrix;
}

//--------------------------------------------------------------------------------------------------
// Budget-driven motion cull (ported from supersplat PR #1048).
//
// While the camera or the splat transform moves, the contribution threshold is
// ramped toward a GPU budget (capped at 1.0 alpha mass). When the view settles the
// threshold decays to zero so the progressively accumulated image is full quality.
// The occlusion cull uses the previous stochastic frame's model-view and its block
// max-depth map, so it is only active once that map exists.
//
void VkViewer::updateStochasticMotionCull()
{
  // object -> view for the current frame. Shaders use the row-vector convention
  // (mul(v, M) == M^T v), so chaining model then view yields view * model in glm.
  const glm::mat4 currentModelView = prmFrame.viewMatrix * m_splatSetVk.transform;

  // Movement detection: max absolute element delta of the model-view matrix.
  bool moving = true;
  if(m_stochastic.havePrevModelView)
  {
    float delta = 0.0f;
    for(int c = 0; c < 4; ++c)
      for(int r = 0; r < 4; ++r)
        delta = std::max(delta, std::fabs(currentModelView[c][r] - m_stochastic.prevModelViewMatrix[c][r]));
    moving = delta > 1.0e-5f;
  }

  m_stochastic.motionCullEnabled = prmStochastic.stochasticMotionCull;
  m_stochastic.moving            = moving;

  if(!m_stochastic.motionCullEnabled)
  {
    m_stochastic.contributionThreshold = 0.0f;
    m_stochastic.occlusionActive       = false;
    m_stochastic.occlusionMapValid     = false;
  }
  else
  {
    // Occlusion is only valid when a previous stochastic frame wrote the map.
    m_stochastic.occlusionActive = m_stochastic.havePrevModelView && m_stochastic.occlusionMapValid;

    // Measure the stochastic pass GPU time (microseconds -> ms). The profiler result
    // is delayed by a few frames but that is fine for a budget controller.
    if(m_profilerTimeline)
    {
      nvutils::ProfilerTimeline::TimerInfo timerInfo;
      std::string                          apiName;
      if(m_profilerTimeline->getFrameTimerInfo("Stochastic GS", timerInfo, apiName) && timerInfo.numAveraged > 0)
        m_stochastic.measuredGpuMs = static_cast<float>(timerInfo.gpu.last * 1.0e-3);
    }

    const float target  = std::max(prmStochastic.stochasticBudgetMs, 0.5f);
    const float ceiling = std::min(prmStochastic.stochasticContributionCeiling, 1.0f);
    const float step    = 0.05f;

    if(moving)
    {
      if(m_stochastic.measuredGpuMs > target)
        m_stochastic.contributionThreshold = std::min(ceiling, m_stochastic.contributionThreshold + step);
      else if(m_stochastic.measuredGpuMs < target * 0.9f)
        m_stochastic.contributionThreshold = std::max(0.0f, m_stochastic.contributionThreshold - step);
    }
    else
    {
      // Settled: converge to full quality.
      m_stochastic.contributionThreshold = std::max(0.0f, m_stochastic.contributionThreshold * 0.5f - 0.01f);
    }
  }

  // Remember the pose used to generate this frame's occlusion map.
  m_stochastic.prevModelViewMatrix = currentModelView;
  m_stochastic.havePrevModelView   = true;

  if(m_stochastic.motionCullEnabled && (m_frameIndex % 60u) == 0u)
  {
    LOGI("Stochastic motion cull: %s gpu=%.2fms threshold=%.3f occlusion=%d\n",
         m_stochastic.moving ? "moving" : "settled", m_stochastic.measuredGpuMs,
         m_stochastic.contributionThreshold, m_stochastic.occlusionActive ? 1 : 0);
  }
}

//--------------------------------------------------------------------------------------------------
// Render one stochastic GS frame: clear (if reset), accumulate, resolve.
//
void VkViewer::renderStochasticFrame(VkCommandBuffer cmd, const FrameRenderContext& ctx)
{
  if(!m_stochastic.initialized)
  {
    initStochasticPipelines();
    if(!m_stochastic.initialized)
      return;
  }

  // Re-initialize if viewport size changed (e.g., splash→full scene)
  if(m_stochastic.outputSize.width != uint32_t(m_viewSize.x)
     || m_stochastic.outputSize.height != uint32_t(m_viewSize.y))
  {
    LOGI("Stochastic GS: viewport resized from %ux%u to %ux%u, reinitializing\n",
         m_stochastic.outputSize.width, m_stochastic.outputSize.height,
         uint32_t(m_viewSize.x), uint32_t(m_viewSize.y));
    deinitStochasticPipelines();
    initStochasticPipelines();
    if(!m_stochastic.initialized)
      return;
  }

  const uint32_t fbWidth  = m_stochastic.outputSize.width;
  const uint32_t fbHeight = m_stochastic.outputSize.height;
  if(fbWidth == 0 || fbHeight == 0)
    return;

  if(ctx.splatCount == 0)
    return;

  NVVK_DBG_SCOPE(cmd);
  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Stochastic GS");

  // Update FrameInfo with camera parameters (view matrix, focal, etc.)
  // This is normally called by renderSingleView() but we bypass that for stochastic mode.
  updateAndUploadFrameInfoUBO(cmd, ctx.splatCount);

  // Budget-driven motion cull must run after the camera UBO is current (view matrix)
  // and before the FrameInfo fields it produces are uploaded.
  updateStochasticMotionCull();

  updateStochasticFrameInfo(cmd);
  prmFrame.splatCount = static_cast<int32_t>(ctx.splatCount);

  vkCmdUpdateBuffer(cmd, m_frameInfoBuffer.buffer, 0, sizeof(shaderio::FrameInfo), &prmFrame);

  VkMemoryBarrier uboBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  uboBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  uboBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &uboBarrier, 0,
                       nullptr, 0, nullptr);

  // Make the previous frame's occlusion map writes visible to this frame's
  // accumulate pass (barriers apply to earlier submissions in submission order).
  if(m_stochastic.occlusionActive && m_stochastic.occlusionBuffer.buffer != VK_NULL_HANDLE)
  {
    VkBufferMemoryBarrier occBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    occBarrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    occBarrier.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
    occBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    occBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    occBarrier.buffer              = m_stochastic.occlusionBuffer.buffer;
    occBarrier.offset              = 0;
    occBarrier.size                = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                         nullptr, 1, &occBarrier, 0, nullptr);
  }

  m_pcRaster.modelMatrix                = m_splatSetVk.transform;
  m_pcRaster.modelMatrixInverse         = m_splatSetVk.transformInverse;
  m_pcRaster.modelMatrixRotScaleInverse = glm::inverse(glm::mat3(m_splatSetVk.transform));

  // Set 0 (m_descriptorSet) has 2 dynamic UBOs: FrameInfo (binding 0) + Indirect (binding 7)
  const uint32_t frameInfoOffset  = 0;  // uploaded at offset 0 above
  const uint32_t indirectOffset   = static_cast<uint32_t>(m_frameIndex * m_indirectStride);
  const uint32_t dynamicOffsets[] = {frameInfoOffset, indirectOffset};
  const VkDescriptorSet descSets[] = {m_descriptorSet, m_stochastic.descriptorSet};

  if(prmFrame.stochasticReset == 1)
  {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_stochastic.clearPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_stochastic.pipelineLayout, 0, 2, descSets, 2,
                            dynamicOffsets);
    vkCmdPushConstants(cmd, m_stochastic.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(shaderio::PushConstant),
                       &m_pcRaster);

    // Clear shader uses 1D dispatch: numthreads(256,1,1), linear pixel index
    const uint32_t totalPixels = fbWidth * fbHeight;
    const uint32_t dispatchX   = (totalPixels + 255) / 256;
    vkCmdDispatch(cmd, dispatchX, 1, 1);

    prmFrame.stochasticReset = 0;
  }

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_stochastic.accumulatePipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_stochastic.pipelineLayout, 0, 2, descSets, 2,
                          dynamicOffsets);
  vkCmdPushConstants(cmd, m_stochastic.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(shaderio::PushConstant),
                     &m_pcRaster);

  const uint32_t splatWorkgroupSize = 256;
  const uint32_t splatDispatchX      = (ctx.splatCount + splatWorkgroupSize - 1) / splatWorkgroupSize;
  vkCmdDispatch(cmd, splatDispatchX, 1, 1);

  VkMemoryBarrier resolveBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  resolveBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  resolveBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                       &resolveBarrier, 0, nullptr, 0, nullptr);

  // Fold this frame's depth buffer into the block max-depth map used by the next frame.
  if(m_stochastic.motionCullEnabled && m_stochastic.depthReducePipeline != VK_NULL_HANDLE
     && m_stochastic.occlusionBlockCols > 0 && m_stochastic.occlusionBlockRows > 0)
  {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_stochastic.depthReducePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_stochastic.pipelineLayout, 0, 2, descSets, 2,
                            dynamicOffsets);

    const uint32_t numBlocks = m_stochastic.occlusionBlockCols * m_stochastic.occlusionBlockRows;
    vkCmdDispatch(cmd, (numBlocks + 63u) / 64u, 1, 1);

    VkBufferMemoryBarrier occWriteBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    occWriteBarrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    occWriteBarrier.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
    occWriteBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    occWriteBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    occWriteBarrier.buffer              = m_stochastic.occlusionBuffer.buffer;
    occWriteBarrier.offset              = 0;
    occWriteBarrier.size                = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                         nullptr, 1, &occWriteBarrier, 0, nullptr);

    m_stochastic.occlusionMapValid = true;
  }

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_stochastic.resolvePipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_stochastic.pipelineLayout, 0, 2, descSets, 2,
                          dynamicOffsets);
  vkCmdPushConstants(cmd, m_stochastic.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(shaderio::PushConstant),
                     &m_pcRaster);

  const uint32_t resolveWg     = 16;
  const uint32_t resolveDispX = (fbWidth + resolveWg - 1) / resolveWg;
  const uint32_t resolveDispY = (fbHeight + resolveWg - 1) / resolveWg;
  vkCmdDispatch(cmd, resolveDispX, resolveDispY, 1);

  // Blit stochastic output to main color buffer for display (via post-process)
  {
    nvvk::cmdImageMemoryBarrier(cmd, {m_stochastic.accumulationImage.image, VK_IMAGE_LAYOUT_GENERAL,
                                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                      {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}});
    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(COLOR_MAIN), VK_IMAGE_LAYOUT_GENERAL,
                                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                      {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}});

    VkImageBlit blitRegion{};
    blitRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blitRegion.srcOffsets[1]  = {static_cast<int32_t>(fbWidth), static_cast<int32_t>(fbHeight), 1};
    blitRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blitRegion.dstOffsets[1]  = {static_cast<int32_t>(m_viewSize.x), static_cast<int32_t>(m_viewSize.y), 1};

    vkCmdBlitImage(cmd, m_stochastic.accumulationImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   m_gBuffers.getColorImage(COLOR_MAIN), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blitRegion,
                   VK_FILTER_LINEAR);

    nvvk::cmdImageMemoryBarrier(cmd, {m_stochastic.accumulationImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                      VK_IMAGE_LAYOUT_GENERAL,
                                      {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}});
    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(COLOR_MAIN), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                      VK_IMAGE_LAYOUT_GENERAL,
                                      {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}});
  }

  prmFrame.stochasticFrameCounter++;

  if(prmFrame.stochasticFrameCounter >= prmFrame.stochasticMaxSamples)
  {
    prmFrame.stochasticReset        = 1;
    prmFrame.stochasticFrameCounter = 0;
  }
}

}  // namespace vk_viewer
