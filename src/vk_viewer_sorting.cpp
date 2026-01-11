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
// Contains: CPU/GPU sorting, draw commands, indirect buffer handling

namespace vk_viewer {

void VkViewer::tryConsumeAndUploadCpuSortingResult(VkCommandBuffer cmd, const uint32_t splatCount)
{
  NVVK_DBG_SCOPE(cmd);

  // upload CPU sorted indices to the GPU if needed
  bool newIndexAvailable = false;

  if(!prmRender.opacityGaussianDisabled)
  {
    // 1. Splatting/blending is on, we check for a newly sorted index table
    auto status = m_cpuSorter.getStatus();
    if(status != SplatSorterAsync::E_SORTING)
    {
      // sorter is sleeping, we can work on shared data
      // we take into account the result of the sort
      if(status == SplatSorterAsync::E_SORTED)
      {
        m_cpuSorter.consume(m_splatIndices);
        newIndexAvailable = true;
      }

      // let's wakeup the sorting thread to run a new sort if needed
      // will start work only if camera direction or position has changed
      m_cpuSorter.sortAsync(glm::normalize(m_center - m_eye), m_eye, m_splatSet.positions, m_splatSetVk.transform,
                            prmRaster.cpuLazySort);
    }
  }
  else
  {
    // splatting off, we disable the sorting
    // indices would not be needed for non splatted points
    // however, using the same mechanism allows to use exactly the same shader
    // so if splatting/blending is off we provide an ordered table of indices
    // if not already filled by any other previous frames (sorted or not)
    bool refill = (m_splatIndices.size() != splatCount);
    if(refill)
    {
      m_splatIndices.resize(splatCount);
      for(uint32_t i = 0; i < splatCount; ++i)
      {
        m_splatIndices[i] = i;
      }
      newIndexAvailable = true;
    }
  }

  // 2. upload to GPU is needed
  {
    auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Copy indices to GPU");

    if(newIndexAvailable)
    {
      // Prepare buffer on host using sorted indices
      memcpy(m_splatIndicesHost.mapping, m_splatIndices.data(), m_splatIndices.size() * sizeof(uint32_t));
      // copy buffer to device
      VkBufferCopy bc{.srcOffset = 0, .dstOffset = 0, .size = splatCount * sizeof(uint32_t)};
      vkCmdCopyBuffer(cmd, m_splatIndicesHost.buffer, m_splatIndicesDevice.buffer, 1, &bc);
      // sync with end of copy to device
      VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      barrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT;

      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
                           0, 1, &barrier, 0, NULL, 0, NULL);
    }
  }
}

void VkViewer::processSortingOnGPU(VkCommandBuffer cmd, const uint32_t splatCount)
{
  NVVK_DBG_SCOPE(cmd);

  // Validate descriptor set is valid
  if (m_descriptorSet == VK_NULL_HANDLE || m_pipelineLayout == VK_NULL_HANDLE)
    return;

  // when GPU sorting, we sort at each frame, all buffer in device memory, no copy from RAM

  // 1. reset the draw indirect parameters and counters, will be updated by compute shader
  {
    const shaderio::IndirectParams drawIndexedIndirectParams;
    VkDeviceSize indirectOffset = m_frameIndex * m_indirectStride;
    vkCmdUpdateBuffer(cmd, m_indirect.buffer, indirectOffset, sizeof(shaderio::IndirectParams), (void*)&drawIndexedIndirectParams);

    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                         0, 1, &barrier, 0, NULL, 0, NULL);
  }

  VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask   = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;

  // 2. invoke the distance compute shader
  {
    auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "GPU Dist");

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipelineGsDistCull);
    // Bind descriptor set with dynamic offsets
    // Order: [FrameInfo, Indirect]
    uint32_t indirectOffset = static_cast<uint32_t>(m_frameIndex * m_indirectStride);
    uint32_t frameInfoOffset = m_lastFrameInfoOffset;
    uint32_t dynamicOffsets[2] = {frameInfoOffset, indirectOffset};
    
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &m_descriptorSet, 2, dynamicOffsets);

    // Model transform
    m_pcRaster.modelMatrix        = m_splatSetVk.transform;
    m_pcRaster.modelMatrixInverse = m_splatSetVk.transformInverse;

    vkCmdPushConstants(cmd, m_pipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(shaderio::PushConstant), &m_pcRaster);

    vkCmdDispatch(cmd, (splatCount + prmRaster.distShaderWorkgroupSize - 1) / prmRaster.distShaderWorkgroupSize, 1, 1);

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                         0, 1, &barrier, 0, NULL, 0, NULL);
  }

  // 3. invoke the radix sort from vrdx lib
  {
    auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "GPU Sort");
    VkDeviceSize indirectOffset = m_frameIndex * m_indirectStride;

    vrdxCmdSortKeyValueIndirect(cmd, m_gpuSorter, splatCount, m_indirect.buffer,
                                indirectOffset + offsetof(shaderio::IndirectParams, instanceCount), m_splatDistancesDevice.buffer, 0,
                                m_splatIndicesDevice.buffer, 0, m_vrdxStorageDevice.buffer, 0, 0, 0);

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                         0, 1, &barrier, 0, NULL, 0, NULL);
  }
}

void VkViewer::drawSplatPrimitives(VkCommandBuffer cmd, const uint32_t splatCount)
{
  NVVK_DBG_SCOPE(cmd);

  // Validate descriptor set is valid
  if (m_descriptorSet == VK_NULL_HANDLE || m_pipelineLayout == VK_NULL_HANDLE) {
    LOGW("[Render] drawSplatPrimitives: descriptor set or pipeline layout is null, skipping\n");
    return;
  }

  // Do we need to activate depth test and Write ?
  bool needDepth = ((prmRaster.sortingMethod != SORTING_GPU_SYNC_RADIX) && prmRender.opacityGaussianDisabled)
                   || !m_meshSetVk.instances.empty();

  // Model transform
  m_pcRaster.modelMatrix        = m_splatSetVk.transform;
  m_pcRaster.modelMatrixInverse = m_splatSetVk.transformInverse;
  // cast to mat3 extracts only the rot/scale part of the transform
  glm::mat3 rotScale                    = glm::mat3(m_splatSetVk.transform);
  m_pcRaster.modelMatrixRotScaleInverse = glm::inverse(rotScale);

  vkCmdPushConstants(cmd, m_pipelineLayout,
                     VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                     0, sizeof(shaderio::PushConstant), &m_pcRaster);

  if(prmSelectedPipeline == PIPELINE_VERT)
  {  // Pipeline using vertex shader
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipelineGsVert);
    // Bind descriptor set with dynamic offsets
    // Order: [FrameInfo, Indirect]
    uint32_t indirectOffset = static_cast<uint32_t>(m_frameIndex * m_indirectStride);
    uint32_t frameInfoOffset = m_lastFrameInfoOffset;
    uint32_t dynamicOffsets[2] = {frameInfoOffset, indirectOffset};
    
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorSet, 2, dynamicOffsets);

    // overrides the pipeline setup for depth test/write
    vkCmdSetDepthWriteEnable(cmd, (VkBool32)needDepth);
    vkCmdSetDepthTestEnable(cmd, (VkBool32)needDepth);

    // display the quad as many times as we have visible splats
    const VkDeviceSize offsets{0};
    vkCmdBindIndexBuffer(cmd, m_quadIndices.buffer, 0, VK_INDEX_TYPE_UINT16);
    vkCmdBindVertexBuffers(cmd, 0, 1, &m_quadVertices.buffer, &offsets);
    if(prmRaster.sortingMethod != SORTING_GPU_SYNC_RADIX)
    {
      vkCmdBindVertexBuffers(cmd, 1, 1, &m_splatIndicesDevice.buffer, &offsets);
      vkCmdDrawIndexed(cmd, 6, (uint32_t)splatCount, 0, 0, 0);
    }
    else
    {
      vkCmdBindVertexBuffers(cmd, 1, 1, &m_splatIndicesDevice.buffer, &offsets);
      VkDeviceSize indirectOffset = m_frameIndex * m_indirectStride;
      vkCmdDrawIndexedIndirect(cmd, m_indirect.buffer, indirectOffset, 1, sizeof(VkDrawIndexedIndirectCommand));
    }
  }
  else
  {  // in mesh pipeline mode or in hybrid mode
    // Pipeline using mesh shader

    if(prmSelectedPipeline == PIPELINE_MESH || prmSelectedPipeline == PIPELINE_HYBRID)
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipelineGsMesh);
    if(prmSelectedPipeline == PIPELINE_MESH_3DGUT || prmSelectedPipeline == PIPELINE_HYBRID_3DGUT)
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline3dgutMesh);

    // Bind descriptor set with dynamic offsets
    // Order: [FrameInfo, Indirect]
    uint32_t indirectOffset = static_cast<uint32_t>(m_frameIndex * m_indirectStride);
    uint32_t frameInfoOffset = m_lastFrameInfoOffset;
    uint32_t dynamicOffsets[2] = {frameInfoOffset, indirectOffset};
    
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorSet, 2, dynamicOffsets);

    // overrides the pipeline setup for depth test/write
    vkCmdSetDepthWriteEnable(cmd, (VkBool32)needDepth);
    vkCmdSetDepthTestEnable(cmd, (VkBool32)needDepth);

    if(prmRaster.sortingMethod != SORTING_GPU_SYNC_RADIX)
    {
      // run the workgroups
      vkCmdDrawMeshTasksEXT(cmd, (prmFrame.splatCount + prmRaster.meshShaderWorkgroupSize - 1) / prmRaster.meshShaderWorkgroupSize,
                            1, 1);
    }
    else
    {
      // run the workgroups
      VkDeviceSize indirectOffset = m_frameIndex * m_indirectStride;
      vkCmdDrawMeshTasksIndirectEXT(cmd, m_indirect.buffer, indirectOffset + offsetof(shaderio::IndirectParams, groupCountX), 1,
                                    sizeof(VkDrawMeshTasksIndirectCommandEXT));
    }
  }
}

void VkViewer::drawMeshPrimitives(VkCommandBuffer cmd)
{
  static int logCounter = 0;
  if(logCounter++ < 5)
  {
    LOGI("drawMeshPrimitives: instances=%zu\n", m_meshSetVk.instances.size());
    for(size_t i = 0; i < m_meshSetVk.instances.size(); ++i)
    {
      const auto& model = m_meshSetVk.meshes[m_meshSetVk.instances[i].objIndex];
      LOGI("  Instance %zu: nbIndices=%u, nbVertices=%u\n", i, model.nbIndices, model.nbVertices);
    }
  }

  NVVK_DBG_SCOPE(cmd);

  VkDeviceSize offset{0};

  // Drawing all triangles
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipelineMesh);
  // Bind descriptor set with dynamic offsets
  // Order: [FrameInfo, Indirect]
  uint32_t indirectOffset = static_cast<uint32_t>(m_frameIndex * m_indirectStride);
  uint32_t frameInfoOffset = m_lastFrameInfoOffset;
  uint32_t dynamicOffsets[2] = {frameInfoOffset, indirectOffset};
  
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorSet, 2, dynamicOffsets);
  // overrides the pipeline setup for depth test/write
  vkCmdSetDepthWriteEnable(cmd, (VkBool32) true);
  vkCmdSetDepthTestEnable(cmd, (VkBool32) true);

  for(const Instance& inst : m_meshSetVk.instances)
  {
    auto& model                   = m_meshSetVk.meshes[inst.objIndex];
    m_pcRaster.objIndex           = inst.objIndex;  // Telling which object is drawn
    m_pcRaster.modelMatrix        = inst.transform;
    m_pcRaster.modelMatrixInverse = inst.transformInverse;

    vkCmdPushConstants(cmd, m_pipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(shaderio::PushConstant), &m_pcRaster);
    vkCmdBindVertexBuffers(cmd, 0, 1, &model.vertexBuffer.buffer, &offset);
    vkCmdBindIndexBuffer(cmd, model.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, model.nbIndices, 1, 0, 0, 0);
  }
}

void VkViewer::drawVdzMesh(VkCommandBuffer cmd)
{
  NVVK_DBG_SCOPE(cmd);

  if(m_vdzMesh.getIndexCount() == 0 || m_graphicsPipelineVdzMesh == VK_NULL_HANDLE)
  {
      return;
  }

  VkDeviceSize offset{0};

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipelineVdzMesh);
  // Bind descriptor set with dynamic offsets
  // Order: [FrameInfo, Indirect]
  uint32_t indirectOffset = static_cast<uint32_t>(m_frameIndex * m_indirectStride);
  uint32_t frameInfoOffset = m_lastFrameInfoOffset;
  uint32_t dynamicOffsets[2] = {frameInfoOffset, indirectOffset};
  
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorSet, 2, dynamicOffsets);

  if(prmFrame.vdzWorldSpaceMode)
  {
      if(!m_vdzWorldSpaceInitialized)
      {
          m_vdzModelMatrix = glm::inverse(cameraManip->getViewMatrix());
          m_vdzWorldSpaceInitialized = true;
      }
      // Set in FrameInfo UBO (shader reads from frameInfo.vdzModelMatrix)
      prmFrame.vdzModelMatrix = m_vdzModelMatrix;
  }
  else
  {
      prmFrame.vdzModelMatrix = glm::mat4(1.0f);
      m_vdzWorldSpaceInitialized = false;
  }

  // Update the vdzModelMatrix in the UBO (it was uploaded earlier, so we need to patch it)
  VkDeviceSize vdzModelMatrixOffset = offsetof(shaderio::FrameInfo, vdzModelMatrix);
  vkCmdUpdateBuffer(cmd, m_frameInfoBuffer.buffer, vdzModelMatrixOffset, sizeof(glm::mat4), &prmFrame.vdzModelMatrix);
  
  // Barrier to ensure the update is visible before vertex shader reads
  VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                       0, 1, &barrier, 0, nullptr, 0, nullptr);

  vkCmdPushConstants(cmd, m_pipelineLayout,
                     VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                     0, sizeof(shaderio::PushConstant), &m_pcRaster);

  VkBuffer vertexBuffer = m_vdzMesh.getVertexBuffer();
  vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer, &offset);
  vkCmdBindIndexBuffer(cmd, m_vdzMesh.getIndexBuffer(), 0, VK_INDEX_TYPE_UINT32);
  vkCmdDrawIndexed(cmd, m_vdzMesh.getIndexCount(), 1, 0, 0, 0);
}

void VkViewer::collectReadBackValuesIfNeeded(void)
{
  if(m_indirectReadbackHost.buffer != VK_NULL_HANDLE && prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX && m_canCollectReadback)
  {
    std::memcpy((void*)&m_indirectReadback, (void*)m_indirectReadbackHost.mapping, sizeof(shaderio::IndirectParams));
  }
}

void VkViewer::readBackIndirectParametersIfNeeded(VkCommandBuffer cmd)
{
  NVVK_DBG_SCOPE(cmd);

  if(m_indirectReadbackHost.buffer != VK_NULL_HANDLE && prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX)
  {
    auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Indirect readback");

    // ensures m_indirect buffer modified by GPU sort is available for transfer
    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask   = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask   = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT, 0, 1, &barrier,
                         0, NULL, 0, NULL);

    // copy from device to host buffer
    // Read from the current frame's indirect buffer offset
    VkDeviceSize indirectOffset = m_frameIndex * m_indirectStride;
    VkBufferCopy bc{.srcOffset = indirectOffset, .dstOffset = 0, .size = sizeof(shaderio::IndirectParams)};
    vkCmdCopyBuffer(cmd, m_indirect.buffer, m_indirectReadbackHost.buffer, 1, &bc);

    m_canCollectReadback = true;
  }
}

void VkViewer::updateRenderingMemoryStatistics(VkCommandBuffer cmd, const uint32_t splatCount)
{
  // update rendering memory statistics
  if(prmRaster.sortingMethod != SORTING_GPU_SYNC_RADIX)
  {
    m_renderMemoryStats.hostAllocIndices   = splatCount * sizeof(uint32_t);
    m_renderMemoryStats.hostAllocDistances = splatCount * sizeof(uint32_t);
    m_renderMemoryStats.allocIndices       = splatCount * sizeof(uint32_t);
    m_renderMemoryStats.usedIndices        = splatCount * sizeof(uint32_t);
    m_renderMemoryStats.allocDistances     = 0;
    m_renderMemoryStats.usedDistances      = 0;
    m_renderMemoryStats.usedIndirect       = 0;
  }
  else
  {
    m_renderMemoryStats.hostAllocDistances = 0;
    m_renderMemoryStats.hostAllocIndices   = 0;
    m_renderMemoryStats.allocDistances     = splatCount * sizeof(uint32_t);
    m_renderMemoryStats.usedDistances      = m_indirectReadback.instanceCount * sizeof(uint32_t);
    m_renderMemoryStats.allocIndices       = splatCount * sizeof(uint32_t);
    m_renderMemoryStats.usedIndices        = m_indirectReadback.instanceCount * sizeof(uint32_t);
    if(prmSelectedPipeline == PIPELINE_VERT)
    {
      m_renderMemoryStats.usedIndirect = 5 * sizeof(uint32_t);
    }
    else
    {
      m_renderMemoryStats.usedIndirect = sizeof(shaderio::IndirectParams);
    }
  }
  m_renderMemoryStats.usedUboFrameInfo = sizeof(shaderio::FrameInfo);
  //
  m_renderMemoryStats.rasterHostTotal =
      m_renderMemoryStats.hostAllocIndices + m_renderMemoryStats.hostAllocDistances + m_renderMemoryStats.usedUboFrameInfo;

  uint64_t vrdxSize = prmRaster.sortingMethod != SORTING_GPU_SYNC_RADIX ? 0 : m_renderMemoryStats.allocVdrxInternal;

  m_renderMemoryStats.rasterDeviceUsedTotal = m_renderMemoryStats.usedIndices + m_renderMemoryStats.usedDistances + vrdxSize
                                              + m_renderMemoryStats.usedIndirect + m_renderMemoryStats.usedUboFrameInfo;

  m_renderMemoryStats.rasterDeviceAllocTotal = m_renderMemoryStats.allocIndices + m_renderMemoryStats.allocDistances + vrdxSize
                                               + m_renderMemoryStats.usedIndirect + m_renderMemoryStats.usedUboFrameInfo;

  // RTX Acceleration Structures
  m_renderMemoryStats.rtxUsedTlas = m_splatSetVk.tlasSizeBytes;
  m_renderMemoryStats.rtxUsedBlas = m_splatSetVk.blasSizeBytes;

  m_renderMemoryStats.rtxHostTotal        = 0;
  m_renderMemoryStats.rtxDeviceUsedTotal  = m_renderMemoryStats.rtxUsedTlas + m_renderMemoryStats.rtxUsedBlas;
  m_renderMemoryStats.rtxDeviceAllocTotal = m_renderMemoryStats.rtxUsedTlas + m_renderMemoryStats.rtxUsedBlas;

  // Total
  m_renderMemoryStats.hostTotal = m_renderMemoryStats.rasterHostTotal + m_renderMemoryStats.rtxHostTotal;
  m_renderMemoryStats.deviceUsedTotal = m_renderMemoryStats.rasterDeviceUsedTotal + m_renderMemoryStats.rtxDeviceUsedTotal;
  m_renderMemoryStats.deviceAllocTotal = m_renderMemoryStats.rasterDeviceAllocTotal + m_renderMemoryStats.rtxDeviceAllocTotal;
}

}  // namespace vk_viewer
