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

  // Output image (matching display color format for blit compatibility)
  VkImageCreateInfo outputInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  outputInfo.imageType     = VK_IMAGE_TYPE_2D;
  outputInfo.format        = m_colorFormat;
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
  accumInfo.format        = m_colorFormat;
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

  VkComputePipelineCreateInfo clearInfo      = makeComputePipeline(m_shaders.stochasticClearShader);
  VkComputePipelineCreateInfo accumulateInfo = makeComputePipeline(m_shaders.stochasticAccumulateShader);
  VkComputePipelineCreateInfo resolveInfo    = makeComputePipeline(m_shaders.stochasticResolveShader);

  NVVK_CHECK(vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &clearInfo, nullptr, &m_stochastic.clearPipeline));
  NVVK_DBG_NAME(m_stochastic.clearPipeline);

  NVVK_CHECK(
      vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &accumulateInfo, nullptr, &m_stochastic.accumulatePipeline));
  NVVK_DBG_NAME(m_stochastic.accumulatePipeline);

  NVVK_CHECK(
      vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &resolveInfo, nullptr, &m_stochastic.resolvePipeline));
  NVVK_DBG_NAME(m_stochastic.resolvePipeline);

  nvvk::WriteSetContainer writeContainer;
  writeContainer.append(
      m_stochastic.descriptorBindings.getWriteSet(BINDING_STOCHASTIC_DEPTH_BUFFER, m_stochastic.descriptorSet),
      m_stochastic.depthBuffer);
  writeContainer.append(
      m_stochastic.descriptorBindings.getWriteSet(BINDING_STOCHASTIC_INDEX_BUFFER, m_stochastic.descriptorSet),
      m_stochastic.indexBuffer);
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
     && m_stochastic.indexBuffer.buffer == VK_NULL_HANDLE)
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

  updateStochasticFrameInfo(cmd);
  prmFrame.splatCount = static_cast<int32_t>(ctx.splatCount);

  vkCmdUpdateBuffer(cmd, m_frameInfoBuffer.buffer, 0, sizeof(shaderio::FrameInfo), &prmFrame);

  VkMemoryBarrier uboBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  uboBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  uboBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &uboBarrier, 0,
                       nullptr, 0, nullptr);

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

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_stochastic.resolvePipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_stochastic.pipelineLayout, 0, 2, descSets, 2,
                          dynamicOffsets);
  vkCmdPushConstants(cmd, m_stochastic.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(shaderio::PushConstant),
                     &m_pcRaster);

  const uint32_t resolveWg     = 8;
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
