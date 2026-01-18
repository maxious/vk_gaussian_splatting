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
// Contains: Chunk-based hierarchical frustum culling implementation

#include <limits>
#include <algorithm>

namespace vk_viewer {

void VkViewer::computeChunkBounds()
{
  const uint32_t splatCount = static_cast<uint32_t>(m_splatSet.size());
  if (splatCount == 0)
  {
    m_numChunks = 0;
    m_chunkBoundsCpu.clear();
    return;
  }

  m_numChunks = (splatCount + CHUNK_SIZE - 1) / CHUNK_SIZE;
  m_chunkBoundsCpu.resize(m_numChunks);

  LOGI("Computing chunk bounds: %u splats -> %u chunks\n", splatCount, m_numChunks);

  const float* positions = m_splatSet.positions.data();

  for (uint32_t chunkIdx = 0; chunkIdx < m_numChunks; ++chunkIdx)
  {
    const uint32_t startSplat = chunkIdx * CHUNK_SIZE;
    const uint32_t endSplat = std::min(startSplat + CHUNK_SIZE, splatCount);

    glm::vec3 minPos(std::numeric_limits<float>::max());
    glm::vec3 maxPos(std::numeric_limits<float>::lowest());

    for (uint32_t i = startSplat; i < endSplat; ++i)
    {
      glm::vec3 pos(positions[i * 3 + 0], positions[i * 3 + 1], positions[i * 3 + 2]);
      minPos = glm::min(minPos, pos);
      maxPos = glm::max(maxPos, pos);
    }

    // Add small epsilon padding to account for splat size
    const float padding = prmFrame.frustumDilation * 0.5f;
    minPos -= glm::vec3(padding);
    maxPos += glm::vec3(padding);

    m_chunkBoundsCpu[chunkIdx].minPos = {minPos.x, minPos.y, minPos.z};
    m_chunkBoundsCpu[chunkIdx]._pad0 = 0.0f;
    m_chunkBoundsCpu[chunkIdx].maxPos = {maxPos.x, maxPos.y, maxPos.z};
    m_chunkBoundsCpu[chunkIdx]._pad1 = 0.0f;
  }

  LOGI("Chunk bounds computed successfully\n");
}

void VkViewer::initChunkCullingBuffers()
{
  if (m_numChunks == 0)
    return;

  const VkDeviceSize chunkBoundsSize = m_numChunks * sizeof(shaderio::ChunkBounds);
  const VkDeviceSize visibleChunksSize = m_numChunks * sizeof(uint32_t);
  const VkDeviceSize countSize = sizeof(uint32_t);

  // Create chunk bounds buffer and upload data
  m_alloc.createBuffer(m_chunkBoundsBuffer, chunkBoundsSize,
                       VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
                       VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
  NVVK_DBG_NAME(m_chunkBoundsBuffer.buffer);

  // Upload chunk bounds to GPU using staging uploader
  NVVK_CHECK(m_uploader.appendBuffer(m_chunkBoundsBuffer, 0, 
             std::span<const shaderio::ChunkBounds>(m_chunkBoundsCpu.data(), m_chunkBoundsCpu.size())));
  VkCommandBuffer cmd = m_app->createTempCmdBuffer();
  m_uploader.cmdUploadAppended(cmd);
  m_app->submitAndWaitTempCmdBuffer(cmd);

  // Create visible chunks output buffer
  m_alloc.createBuffer(m_visibleChunksBuffer, visibleChunksSize,
                       VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
                       VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
  NVVK_DBG_NAME(m_visibleChunksBuffer.buffer);

  // Create visible chunk count buffer (atomic counter)
  m_alloc.createBuffer(m_visibleChunkCountBuffer, countSize,
                       VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
                       VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
  NVVK_DBG_NAME(m_visibleChunkCountBuffer.buffer);

  LOGI("Chunk culling buffers initialized: %u chunks, %.2f KB\n",
       m_numChunks, (chunkBoundsSize + visibleChunksSize + countSize) / 1024.0f);
}

void VkViewer::deinitChunkCullingBuffers()
{
  m_alloc.destroyBuffer(m_chunkBoundsBuffer);
  m_alloc.destroyBuffer(m_visibleChunksBuffer);
  m_alloc.destroyBuffer(m_visibleChunkCountBuffer);
  m_numChunks = 0;
  m_chunkBoundsCpu.clear();
}

void VkViewer::initChunkCullingPipeline()
{
  if (!m_shaders.chunkCullShader)
    return;

  // Create descriptor set layout for chunk culling
  nvvk::DescriptorBindings bindings;
  bindings.addBinding(BINDING_FRAME_INFO_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  bindings.addBinding(BINDING_CHUNK_BOUNDS_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  bindings.addBinding(BINDING_VISIBLE_CHUNKS_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  bindings.addBinding(BINDING_VISIBLE_CHUNK_COUNT_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);

  NVVK_CHECK(bindings.createDescriptorSetLayout(m_device, 0, &m_chunkCullDescriptorSetLayout));
  NVVK_DBG_NAME(m_chunkCullDescriptorSetLayout);

  // Create descriptor pool
  std::vector<VkDescriptorPoolSize> poolSize;
  bindings.appendPoolSizes(poolSize);
  VkDescriptorPoolCreateInfo poolInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = uint32_t(poolSize.size()),
      .pPoolSizes = poolSize.data(),
  };
  NVVK_CHECK(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_chunkCullDescriptorPool));
  NVVK_DBG_NAME(m_chunkCullDescriptorPool);

  // Allocate descriptor set
  VkDescriptorSetAllocateInfo allocInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = m_chunkCullDescriptorPool,
      .descriptorSetCount = 1,
      .pSetLayouts = &m_chunkCullDescriptorSetLayout,
  };
  NVVK_CHECK(vkAllocateDescriptorSets(m_device, &allocInfo, &m_chunkCullDescriptorSet));
  NVVK_DBG_NAME(m_chunkCullDescriptorSet);

  // Write descriptors
  nvvk::WriteSetContainer writeContainer;
  
  VkDescriptorBufferInfo frameInfoDesc{m_frameInfoBuffer.buffer, 0, sizeof(shaderio::FrameInfo)};
  VkWriteDescriptorSet frameInfoWrite = bindings.getWriteSet(BINDING_FRAME_INFO_UBO, m_chunkCullDescriptorSet);
  frameInfoWrite.pBufferInfo = &frameInfoDesc;
  vkUpdateDescriptorSets(m_device, 1, &frameInfoWrite, 0, nullptr);

  if (m_chunkBoundsBuffer.buffer != VK_NULL_HANDLE)
  {
    writeContainer.append(bindings.getWriteSet(BINDING_CHUNK_BOUNDS_BUFFER, m_chunkCullDescriptorSet), m_chunkBoundsBuffer);
    writeContainer.append(bindings.getWriteSet(BINDING_VISIBLE_CHUNKS_BUFFER, m_chunkCullDescriptorSet), m_visibleChunksBuffer);
    writeContainer.append(bindings.getWriteSet(BINDING_VISIBLE_CHUNK_COUNT_BUFFER, m_chunkCullDescriptorSet), m_visibleChunkCountBuffer);
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);
  }

  // Create pipeline layout with push constants
  const VkPushConstantRange pcRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(shaderio::PushConstant)};
  VkPipelineLayoutCreateInfo layoutInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &m_chunkCullDescriptorSetLayout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pcRange,
  };
  NVVK_CHECK(vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_chunkCullPipelineLayout));
  NVVK_DBG_NAME(m_chunkCullPipelineLayout);

  // Create compute pipeline
  VkPipelineShaderStageCreateInfo stageInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = m_shaders.chunkCullShader,
      .pName = "main",
  };

  VkComputePipelineCreateInfo pipelineInfo{
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = stageInfo,
      .layout = m_chunkCullPipelineLayout,
  };

  NVVK_CHECK(vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_computePipelineChunkCull));
  NVVK_DBG_NAME(m_computePipelineChunkCull);

  LOGI("Chunk culling pipeline initialized\n");
}

void VkViewer::deinitChunkCullingPipeline()
{
  if (m_computePipelineChunkCull != VK_NULL_HANDLE)
  {
    vkDestroyPipeline(m_device, m_computePipelineChunkCull, nullptr);
    m_computePipelineChunkCull = VK_NULL_HANDLE;
  }
  if (m_chunkCullPipelineLayout != VK_NULL_HANDLE)
  {
    vkDestroyPipelineLayout(m_device, m_chunkCullPipelineLayout, nullptr);
    m_chunkCullPipelineLayout = VK_NULL_HANDLE;
  }
  if (m_chunkCullDescriptorPool != VK_NULL_HANDLE)
  {
    vkDestroyDescriptorPool(m_device, m_chunkCullDescriptorPool, nullptr);
    m_chunkCullDescriptorPool = VK_NULL_HANDLE;
  }
  m_chunkCullDescriptorSet = VK_NULL_HANDLE;
  if (m_chunkCullDescriptorSetLayout != VK_NULL_HANDLE)
  {
    vkDestroyDescriptorSetLayout(m_device, m_chunkCullDescriptorSetLayout, nullptr);
    m_chunkCullDescriptorSetLayout = VK_NULL_HANDLE;
  }
}

void VkViewer::extractFrustumPlanes(const glm::mat4& viewProj)
{
  // Extract frustum planes from the viewProj matrix
  // Each plane is represented as (A, B, C, D) where Ax + By + Cz + D = 0
  // A point is inside the frustum if dot(plane.xyz, point) + plane.w >= 0 for all planes
  
  const glm::mat4& m = viewProj;

  // Left plane: row3 + row0
  prmFrame.frustumPlanes[0] = {m[0][3] + m[0][0], m[1][3] + m[1][0], m[2][3] + m[2][0], m[3][3] + m[3][0]};
  
  // Right plane: row3 - row0
  prmFrame.frustumPlanes[1] = {m[0][3] - m[0][0], m[1][3] - m[1][0], m[2][3] - m[2][0], m[3][3] - m[3][0]};
  
  // Bottom plane: row3 + row1
  prmFrame.frustumPlanes[2] = {m[0][3] + m[0][1], m[1][3] + m[1][1], m[2][3] + m[2][1], m[3][3] + m[3][1]};
  
  // Top plane: row3 - row1
  prmFrame.frustumPlanes[3] = {m[0][3] - m[0][1], m[1][3] - m[1][1], m[2][3] - m[2][1], m[3][3] - m[3][1]};
  
  // Near plane: row3 + row2
  prmFrame.frustumPlanes[4] = {m[0][3] + m[0][2], m[1][3] + m[1][2], m[2][3] + m[2][2], m[3][3] + m[3][2]};
  
  // Far plane: row3 - row2
  prmFrame.frustumPlanes[5] = {m[0][3] - m[0][2], m[1][3] - m[1][2], m[2][3] - m[2][2], m[3][3] - m[3][2]};

  // Normalize planes
  for (int i = 0; i < 6; ++i)
  {
    float len = glm::length(glm::vec3(prmFrame.frustumPlanes[i].x, 
                                      prmFrame.frustumPlanes[i].y, 
                                      prmFrame.frustumPlanes[i].z));
    if (len > 0.0f)
    {
      prmFrame.frustumPlanes[i].x /= len;
      prmFrame.frustumPlanes[i].y /= len;
      prmFrame.frustumPlanes[i].z /= len;
      prmFrame.frustumPlanes[i].w /= len;
    }
  }
}

void VkViewer::dispatchChunkCulling(VkCommandBuffer cmd)
{
  if (!prmRaster.chunkCullingEnabled || m_numChunks == 0 || m_computePipelineChunkCull == VK_NULL_HANDLE)
    return;

  NVVK_DBG_SCOPE(cmd);
  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Chunk Cull");

  // Reset visible chunk count to 0
  const uint32_t zero = 0;
  vkCmdUpdateBuffer(cmd, m_visibleChunkCountBuffer.buffer, 0, sizeof(uint32_t), &zero);

  // Barrier to ensure reset is complete before compute shader reads/writes
  VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0, 1, &barrier, 0, nullptr, 0, nullptr);

  // Bind pipeline and descriptors
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipelineChunkCull);
  
  uint32_t frameInfoOffset = m_lastFrameInfoOffset;
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_chunkCullPipelineLayout, 0, 1,
                          &m_chunkCullDescriptorSet, 1, &frameInfoOffset);

  // Push model transform
  vkCmdPushConstants(cmd, m_chunkCullPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                     0, sizeof(shaderio::PushConstant), &m_pcRaster);

  // Dispatch chunk culling
  const uint32_t workgroups = (m_numChunks + CHUNK_CULL_WORKGROUP_SIZE - 1) / CHUNK_CULL_WORKGROUP_SIZE;
  vkCmdDispatch(cmd, workgroups, 1, 1);

  // Barrier to ensure chunk culling is complete before distance compute reads results
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0, 1, &barrier, 0, nullptr, 0, nullptr);
}

}  // namespace vk_viewer
