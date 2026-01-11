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

// This file is included from gaussian_splatting.cpp - do not compile separately
// Contains: VK_KHR_multiview support for OpenXR stereo rendering

#ifdef WITH_OPENXR

namespace vk_viewer {

void VkViewer::initXrMultiviewResources(VkCommandBuffer cmd, VkExtent2D perEyeExtent)
{
  if(m_xrMultiviewInitialized && m_xrMultiviewExtent.width == perEyeExtent.width 
     && m_xrMultiviewExtent.height == perEyeExtent.height)
  {
    return;  // Already initialized at correct size
  }

  deinitXrMultiviewResources();

  m_xrMultiviewExtent = perEyeExtent;

  // Create 2-layer color image for multiview stereo
  {
    VkImageCreateInfo colorInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    colorInfo.imageType     = VK_IMAGE_TYPE_2D;
    colorInfo.format        = m_colorFormat;
    colorInfo.extent        = {perEyeExtent.width, perEyeExtent.height, 1};
    colorInfo.mipLevels     = 1;
    colorInfo.arrayLayers   = 2;  // 2 layers for stereo
    colorInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    colorInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    colorInfo.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT 
                            | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    colorInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    colorInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    NVVK_CHECK(m_alloc.createImage(m_xrMultiviewColor, colorInfo));
    NVVK_DBG_NAME(m_xrMultiviewColor.image);

    // Create array view for both layers
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image                           = m_xrMultiviewColor.image;
    viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    viewInfo.format                          = m_colorFormat;
    viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel   = 0;
    viewInfo.subresourceRange.levelCount     = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = 2;

    NVVK_CHECK(vkCreateImageView(m_device, &viewInfo, nullptr, &m_xrMultiviewColorView));
    NVVK_DBG_NAME(m_xrMultiviewColorView);
  }

  // Create 2-layer depth image for multiview stereo
  {
    VkImageCreateInfo depthInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    depthInfo.imageType     = VK_IMAGE_TYPE_2D;
    depthInfo.format        = m_depthFormat;
    depthInfo.extent        = {perEyeExtent.width, perEyeExtent.height, 1};
    depthInfo.mipLevels     = 1;
    depthInfo.arrayLayers   = 2;  // 2 layers for stereo
    depthInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    depthInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    depthInfo.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    depthInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    depthInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    NVVK_CHECK(m_alloc.createImage(m_xrMultiviewDepth, depthInfo));
    NVVK_DBG_NAME(m_xrMultiviewDepth.image);

    // Create array view for both layers
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image                           = m_xrMultiviewDepth.image;
    viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    viewInfo.format                          = m_depthFormat;
    viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewInfo.subresourceRange.baseMipLevel   = 0;
    viewInfo.subresourceRange.levelCount     = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = 2;

    NVVK_CHECK(vkCreateImageView(m_device, &viewInfo, nullptr, &m_xrMultiviewDepthView));
    NVVK_DBG_NAME(m_xrMultiviewDepthView);
  }

  // Transition images to general layout
  nvvk::cmdImageMemoryBarrier(cmd, {m_xrMultiviewColor.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                    {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 2}});
  nvvk::cmdImageMemoryBarrier(cmd, {m_xrMultiviewDepth.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                    {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 2}});

  m_xrMultiviewInitialized = true;
  LOGI("Multiview resources initialized: %ux%u per eye (2 layers)\n", perEyeExtent.width, perEyeExtent.height);
}

void VkViewer::deinitXrMultiviewResources()
{
  if(!m_xrMultiviewInitialized)
    return;

  vkDeviceWaitIdle(m_device);

  if(m_xrMultiviewColorView != VK_NULL_HANDLE)
  {
    vkDestroyImageView(m_device, m_xrMultiviewColorView, nullptr);
    m_xrMultiviewColorView = VK_NULL_HANDLE;
  }
  if(m_xrMultiviewDepthView != VK_NULL_HANDLE)
  {
    vkDestroyImageView(m_device, m_xrMultiviewDepthView, nullptr);
    m_xrMultiviewDepthView = VK_NULL_HANDLE;
  }
  if(m_xrMultiviewColor.image != VK_NULL_HANDLE)
  {
    m_alloc.destroyImage(m_xrMultiviewColor);
    m_xrMultiviewColor = {};
  }
  if(m_xrMultiviewDepth.image != VK_NULL_HANDLE)
  {
    m_alloc.destroyImage(m_xrMultiviewDepth);
    m_xrMultiviewDepth = {};
  }

  m_xrMultiviewInitialized = false;
  m_xrMultiviewExtent = {};
}

void VkViewer::renderMultiviewRaster(VkCommandBuffer cmd, uint32_t splatCount)
{
  if(!m_xrMultiviewInitialized || !m_shaders.valid)
    return;
  if(m_descriptorSet == VK_NULL_HANDLE || m_pipelineLayout == VK_NULL_HANDLE)
    return;

  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Multiview Raster");

  // Transition images to attachment layouts
  nvvk::cmdImageMemoryBarrier(cmd, {m_xrMultiviewColor.image, VK_IMAGE_LAYOUT_GENERAL,
                                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 2}});
  nvvk::cmdImageMemoryBarrier(cmd, {m_xrMultiviewDepth.image, VK_IMAGE_LAYOUT_GENERAL,
                                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                    {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 2}});

  // Set up multiview rendering info
  VkRenderingAttachmentInfo colorAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  colorAttachment.imageView   = m_xrMultiviewColorView;
  colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  colorAttachment.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
  colorAttachment.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
  colorAttachment.clearValue  = {m_clearColor};

  VkRenderingAttachmentInfo depthAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  depthAttachment.imageView   = m_xrMultiviewDepthView;
  depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
  depthAttachment.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depthAttachment.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
  depthAttachment.clearValue  = {.depthStencil = {1.0f, 0}};

  VkRenderingInfo renderingInfo{VK_STRUCTURE_TYPE_RENDERING_INFO};
  renderingInfo.renderArea           = {{0, 0}, m_xrMultiviewExtent};
  renderingInfo.layerCount           = 1;  // Must be 1 when using multiview
  renderingInfo.viewMask             = 0x3;  // Binary 11 = render to both views
  renderingInfo.colorAttachmentCount = 1;
  renderingInfo.pColorAttachments    = &colorAttachment;
  renderingInfo.pDepthAttachment     = &depthAttachment;

  vkCmdBeginRendering(cmd, &renderingInfo);

  // Set viewport and scissor for per-eye rendering
  VkViewport viewport{0.0f, 0.0f, float(m_xrMultiviewExtent.width), float(m_xrMultiviewExtent.height), 0.0f, 1.0f};
  VkRect2D scissor{{0, 0}, m_xrMultiviewExtent};
  vkCmdSetViewportWithCount(cmd, 1, &viewport);
  vkCmdSetScissorWithCount(cmd, 1, &scissor);

  // Draw splats with multiview pipeline
  if(splatCount > 0)
  {
    VkPipeline multiviewPipeline = VK_NULL_HANDLE;
    
    if(prmSelectedPipeline == PIPELINE_MESH || prmSelectedPipeline == PIPELINE_HYBRID)
    {
      multiviewPipeline = m_graphicsPipelineGsMeshMultiview;
    }
    else if(prmSelectedPipeline == PIPELINE_MESH_3DGUT || prmSelectedPipeline == PIPELINE_HYBRID_3DGUT)
    {
      multiviewPipeline = m_graphicsPipeline3dgutMeshMultiview;
    }
    else if(prmSelectedPipeline == PIPELINE_VERT)
    {
      multiviewPipeline = m_graphicsPipelineGsVertMultiview;
    }

    if(multiviewPipeline != VK_NULL_HANDLE)
    {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, multiviewPipeline);
      // Dynamic offsets: [FrameInfo, Indirect]
      uint32_t indirectOffset = static_cast<uint32_t>(m_frameIndex * m_indirectStride);
      uint32_t frameInfoOffset = m_lastFrameInfoOffset;
      uint32_t dynamicOffsets[2] = {frameInfoOffset, indirectOffset};
      
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorSet, 2, dynamicOffsets);
      
      m_pcRaster.modelMatrix                = m_splatSetVk.transform;

      m_pcRaster.modelMatrixInverse         = m_splatSetVk.transformInverse;
      m_pcRaster.modelMatrixRotScaleInverse = glm::inverse(glm::mat3(m_splatSetVk.transform));
      vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_ALL_GRAPHICS, 0, sizeof(shaderio::PushConstant), &m_pcRaster);

      // Disable depth write/test for splats (sorted back-to-front)
      vkCmdSetDepthWriteEnable(cmd, VK_FALSE);
      vkCmdSetDepthTestEnable(cmd, VK_FALSE);

      if(prmSelectedPipeline == PIPELINE_VERT)
      {
        // Vertex shader path
        VkDeviceSize offsets[2] = {0, 0};
        VkBuffer     buffers[2] = {m_quadVertices.buffer, m_splatIndicesDevice.buffer};
        vkCmdBindVertexBuffers(cmd, 0, 2, buffers, offsets);
        vkCmdBindIndexBuffer(cmd, m_quadIndices.buffer, 0, VK_INDEX_TYPE_UINT32);
        VkDeviceSize indirectOffset = m_frameIndex * m_indirectStride;
        vkCmdDrawIndexedIndirect(cmd, m_indirect.buffer, indirectOffset, 1, sizeof(VkDrawIndexedIndirectCommand));
      }
      else
      {
        // Mesh shader path
        VkDeviceSize indirectOffset = m_frameIndex * m_indirectStride;
        vkCmdDrawMeshTasksIndirectEXT(cmd, m_indirect.buffer,
                                      indirectOffset + offsetof(shaderio::IndirectParams, groupCountX), 1, sizeof(VkDrawMeshTasksIndirectCommandEXT));
      }

    }
  }

  // Allow subclasses to render additional content (e.g., hand meshes)
  onRenderMultiviewExtra(cmd);

  vkCmdEndRendering(cmd);

  // Transition images back to general
  nvvk::cmdImageMemoryBarrier(cmd, {m_xrMultiviewColor.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 2}});
  nvvk::cmdImageMemoryBarrier(cmd, {m_xrMultiviewDepth.image, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                    VK_IMAGE_LAYOUT_GENERAL,
                                    {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 2}});
}

}  // namespace vk_viewer

#endif  // WITH_OPENXR
