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
// Contains: Main render loop (onRender) and update request processing

namespace vk_viewer {

void VkViewer::onRender(VkCommandBuffer cmd)
{
  NVVK_DBG_SCOPE(cmd);

  // Create frame context and populate it through orchestration helpers
  FrameRenderContext ctx{};
  ctx.cmd = cmd;

  // Phase 1: Begin frame (XR lifecycle, frame index setup)
  if(!beginFrame(ctx))
    return;

  if(isLccStreamingActive())
  {
    cameraManip->getLookat(m_eye, m_center, m_up);
    glm::mat4 viewMatrix = cameraManip->getViewMatrix();
    glm::mat4 projMatrix = cameraManip->getPerspectiveMatrix();
    glm::mat4 viewProj = projMatrix * viewMatrix;
    updateLccStreaming(viewProj, m_eye, 0.0f);

    // Wait for async sorter to finish before modifying geometry to prevent race condition
    while(m_cpuSorter.getStatus() == SplatSorterAsync::E_SORTING)
    {
      std::this_thread::yield();
    }
  }

  // Phase 2: Determine what content we have and output mode
  const size_t prevSplatCount = isLccStreamingActive() && m_lccTileManager->isPackedMode() 
                                ? m_lccPackedSplatCount : m_splatSet.size();
  buildContentState(ctx);

  if(isLccStreamingActive())
  {
    // Check if splat count changed (use appropriate count based on mode)
    size_t currentSplatCount = m_lccTileManager->isPackedMode() ? m_lccPackedSplatCount : m_splatSet.size();
    if(currentSplatCount != prevSplatCount && currentSplatCount > 0)
    {
      // Update GPU buffers only when splat count actually changed
      m_requestUpdateSplatData = true;
    }
  }

  // Process update requests (shader rebuilds, buffer updates)
  processUpdateRequests();

  // Early exit for RTX-only pipeline (separate code path for now)
  if(ctx.useRtxPipeline && ctx.shadersValid && ctx.hasSplats)
  {
#ifdef WITH_OPENXR
    if(ctx.xrFrameActive && m_xr)
    {
      // RTX with XR - use existing multiview raytrace or dual-pass
      VkExtent2D perEye = m_xr->getPerEyeExtent();
      const uint32_t halfWidth = perEye.width;
      const uint32_t height = perEye.height;

      GsOpenXr::EyeData leftEyeData = m_xr->getEyeData(0);
      GsOpenXr::EyeData rightEyeData = m_xr->getEyeData(1);

      if(prmRtx.temporalSampling && !updateFrameCounter())
        return;

      collectReadBackValuesIfNeeded();

      if(m_xr->supportsMultiview())
      {
        raytraceMultiview(cmd, false,
                          leftEyeData.view, leftEyeData.proj,
                          rightEyeData.view, rightEyeData.proj,
                          leftEyeData.eyePos, rightEyeData.eyePos,
                          glm::ivec2(halfWidth * 2, height));
      }
      else
      {
        updateAndUploadFrameInfoUBO(cmd, ctx.splatCount, leftEyeData.view, leftEyeData.proj, leftEyeData.eyePos, glm::vec2(halfWidth, height));
        raytrace(cmd, false, glm::ivec2(0, 0), glm::ivec2(halfWidth, height));

        updateAndUploadFrameInfoUBO(cmd, ctx.splatCount, rightEyeData.view, rightEyeData.proj, rightEyeData.eyePos, glm::vec2(halfWidth, height));
        raytrace(cmd, false, glm::ivec2(halfWidth, 0), glm::ivec2(halfWidth, height));
      }

      readBackIndirectParametersIfNeeded(cmd);
      updateRenderingMemoryStatistics(cmd, ctx.splatCount);

      copyToXrSwapchain(cmd);
      m_xr->releaseSwapchainImages();
      m_xr->endFrame();
      return;
    }
#endif

    // Desktop RTX path
    if(prmRtx.temporalSampling && !updateFrameCounter())
      return;

    cameraManip->getLookat(m_eye, m_center, m_up);
    ctx.viewMatrix = cameraManip->getViewMatrix();
    ctx.projMatrix = cameraManip->getPerspectiveMatrix();
    ctx.eyePosition = m_eye;
    ctx.centerPosition = m_center;
    ctx.upVector = m_up;

    if(m_renderSBS)
    {
      renderRtxStereoFrame(ctx);
    }
    else
    {
      renderRtxFrame(ctx);
    }

    // Post-processing and depth for RTX path
    if((prmRtx.temporalSampling && prmFrame.frameSampleId > 0) || prmFrame.linearToSrgb != 0)
    {
      postProcess(cmd);
    }

    updateDepthRendering(cmd);

    readBackIndirectParametersIfNeeded(cmd);
    updateRenderingMemoryStatistics(cmd, ctx.splatCount);
    return;
  }

  // Raster and Hybrid pipelines
  if(prmRtx.temporalSampling && !updateFrameCounter())
    return;

  // Phase 3: Build views (mono, stereo SBS, XR dual-pass, or XR multiview)
  buildViews(ctx);

  // Phase 4: Prepare scene data (sorting, UBO upload)
  prepareSceneForFrame(ctx);

  // Phase 5: Clear main render targets
  clearMainTargets(ctx);

  // Phase 6: Render based on output mode
#ifdef WITH_OPENXR
  if(ctx.useXrMultiview && m_xrMultiviewInitialized && ctx.shadersValid && ctx.hasSplats)
  {
    // Multiview path - upload UBO with multiview matrices
    vkCmdUpdateBuffer(cmd, m_frameInfoBuffer.buffer, 0, sizeof(shaderio::FrameInfo), &prmFrame);
    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT
                             | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         0, 1, &barrier, 0, NULL, 0, NULL);

    // Render with multiview (single dispatch for both eyes)
    renderMultiviewRaster(cmd, ctx.splatCount);

    // Copy multiview result to XR swapchain
    VkExtent2D perEye = m_xr->getPerEyeExtent();
    
    nvvk::cmdImageMemoryBarrier(cmd, {m_xrColorImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL});

    VkImageCopy copyRegions[2] = {};
    for(uint32_t eye = 0; eye < 2; ++eye)
    {
      copyRegions[eye].srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, eye, 1};
      copyRegions[eye].srcOffset = {0, 0, 0};
      copyRegions[eye].dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      copyRegions[eye].dstOffset = {static_cast<int32_t>(eye * perEye.width), 0, 0};
      copyRegions[eye].extent = {perEye.width, perEye.height, 1};
    }
    vkCmdCopyImage(cmd, m_xrMultiviewColor.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   m_xrColorImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 2, copyRegions);

    nvvk::cmdImageMemoryBarrier(cmd, {m_xrColorImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

    // Copy multiview result to desktop GBuffer for mirror display
    {
      nvvk::cmdImageMemoryBarrier(cmd, {m_xrMultiviewColor.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 2}});
      nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(COLOR_MAIN), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL});

      VkImageBlit blitRegions[2] = {};
      for(uint32_t eye = 0; eye < 2; ++eye)
      {
        blitRegions[eye].srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, eye, 1};
        blitRegions[eye].srcOffsets[0] = {0, 0, 0};
        blitRegions[eye].srcOffsets[1] = {static_cast<int32_t>(perEye.width), static_cast<int32_t>(perEye.height), 1};
        blitRegions[eye].dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        int32_t dstX = static_cast<int32_t>(eye * m_viewSize.x / 2);
        blitRegions[eye].dstOffsets[0] = {dstX, 0, 0};
        blitRegions[eye].dstOffsets[1] = {dstX + static_cast<int32_t>(m_viewSize.x / 2), static_cast<int32_t>(m_viewSize.y), 1};
      }
      vkCmdBlitImage(cmd, m_xrMultiviewColor.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     m_gBuffers.getColorImage(COLOR_MAIN), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 2, blitRegions, VK_FILTER_LINEAR);

      nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(COLOR_MAIN), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});
    }

    readBackIndirectParametersIfNeeded(cmd);
    updateRenderingMemoryStatistics(cmd, ctx.splatCount);

    m_xr->releaseSwapchainImages();
    {
      auto timer = m_profilerTimeline->frameSection("XR EndFrame");
      m_xr->endFrame();
    }
    return;
  }
#endif

  // Per-view rendering path (mono, SBS stereo, XR dual-pass)
  renderPerViewPath(ctx);

  // Phase 7: Finalize frame (post-process, readback, XR endFrame)
  finalizeFrame(ctx);
}

void VkViewer::processUpdateRequests(void)
{
  // Automatic and Sanity settings depending in pipeline
  if(prmSelectedPipeline != PIPELINE_RTX && prmSelectedPipeline != PIPELINE_HYBRID_3DGUT && prmSelectedPipeline != PIPELINE_MESH_3DGUT)
  {
    prmRtx.temporalSampling = false;
  }
  else
  {
    if(prmRtx.temporalSamplingMode == TEMPORAL_SAMPLING_AUTO && m_cameraSet.getCamera().dofEnabled)
    {
      prmRtx.temporalSampling = true;
    }
    else
    {
      prmRtx.temporalSampling = (prmRtx.temporalSamplingMode == TEMPORAL_SAMPLING_ENABLED);
    }
  }

  // process delayed requests
  if((prmSelectedPipeline == PIPELINE_RTX || prmSelectedPipeline == PIPELINE_HYBRID || prmSelectedPipeline == PIPELINE_HYBRID_3DGUT)
     && m_requestDelayedUpdateSplatAs)
  {
    m_requestUpdateSplatAs        = true;
    m_requestDelayedUpdateSplatAs = false;
  }

  bool needUpdate = m_requestUpdateSplatData || m_requestUpdateSplatAs || m_requestUpdateMeshData
                    || m_requestUpdateShaders || m_requestUpdateLightsBuffer || m_requestDeleteSelectedMesh;

  // Allow mesh-only or depth-only updates even without splats loaded
  bool hasMeshUpdate = m_requestUpdateMeshData || m_requestDeleteSelectedMesh;
  bool hasDepthContent = m_enableDepthRendering || m_videoDepthPlaybackMode;
  if(!needUpdate)
    return;
  if(!m_splatSet.size() && !hasMeshUpdate && !hasDepthContent)
    return;

  resetFrameCounter();

  vkDeviceWaitIdle(m_device);

  // Determine if we need full shader/pipeline rebuild or just data update
  bool needsShaderRebuild = m_requestUpdateShaders || m_requestUpdateMeshData || m_requestDeleteSelectedMesh;
  bool needsDataUpdate = m_requestUpdateSplatData || m_requestUpdateSplatAs;
  
  // updates that requires update of descriptor sets
  if(needsDataUpdate || needsShaderRebuild)
  {
    // Only rebuild shaders/pipelines when actually needed (not for streaming data updates)
    if(needsShaderRebuild || !m_shaders.valid)
    {
      deinitPipelines();
      deinitShaders();
    }

    if(m_requestUpdateSplatData)
    {
      m_splatSetVk.deinitDataStorage();
      
      // Track previous storage mode to detect changes requiring shader rebuild
      const uint32_t prevDataStorage = prmData.dataStorage;
      
      // Check if using LCC packed mode for GPU-side decompression
      if(isLccStreamingActive() && m_lccTileManager->isPackedMode() && m_lccPackedSplatCount > 0)
      {
        // GPU-side decompression: upload raw packed data directly
        m_splatSetVk.initLccPackedStorage(m_lccPackedData.data(), m_lccPackedSplatCount);
        prmData.dataStorage = STORAGE_LCC_PACKED;
        
        LOGD("LCC packed upload: %u splats, %zu bytes\n", m_lccPackedSplatCount, m_lccPackedData.size());
      }
      else
      {
        // Standard path: CPU-decoded data or non-LCC files
        m_splatSetVk.initDataStorage(m_splatSet, prmData.dataStorage, prmData.shFormat);
      }
      
      // If storage mode changed, force shader rebuild
      if(prmData.dataStorage != prevDataStorage)
      {
        LOGI("Data storage mode changed from %u to %u, forcing shader rebuild\n", prevDataStorage, prmData.dataStorage);
        needsShaderRebuild = true;
        deinitPipelines();
        deinitShaders();
      }

      // Re-initialize renderer buffers if capacity is insufficient (streaming optimization)
      // initRendererBuffers() will skip if buffers are already large enough
      const uint32_t splatCount = (prmData.dataStorage == STORAGE_LCC_PACKED) 
                                  ? m_lccPackedSplatCount : (uint32_t)m_splatSet.size();
      if(splatCount > m_rendererBufferCapacity)
      {
        deinitRendererBuffers();
        initRendererBuffers();
      }
      
      // Compute chunk bounds for hierarchical frustum culling
      deinitChunkCullingBuffers();
      computeChunkBounds();
      initChunkCullingBuffers();
    }
    if((m_requestUpdateSplatData || m_requestUpdateSplatAs) && prmSelectedPipeline == PIPELINE_RTX)
    {
      // RTX specific - only update when using RTX pipeline
      m_splatSetVk.rtxDeinitAccelerationStructures();
      m_splatSetVk.rtxDeinitSplatModel();
      m_splatSetVk.rtxInitSplatModel(m_splatSet, prmRtxData.useTlasInstances, prmRtxData.useAABBs, prmRtxData.compressBlas,
                                     prmRtx.kernelDegree, prmRtx.kernelMinResponse, prmRtx.kernelAdaptiveClamping);
      m_splatSetVk.rtxInitAccelerationStructures(m_splatSet);
    }

    if(m_requestUpdateMeshData || m_requestDeleteSelectedMesh)
    {
      if(m_requestDeleteSelectedMesh)
      {
        m_meshSetVk.deleteInstance(uint32_t(m_selectedItemIndex));
        m_selectedItemIndex = -1;
      }

      m_meshSetVk.rtxDeinitAccelerationStructures();
      m_meshSetVk.updateObjDescriptionBuffer();
      m_meshSetVk.rtxInitAccelerationStructures();
    }

    // Rebuild shaders/pipelines if needed, or just update descriptor sets for data-only updates
    if(needsShaderRebuild || !m_shaders.valid)
    {
      if(initShaders())
      {
        // Initialize renderer buffers if not already done (mesh-only mode)
        if(m_frameInfoBuffer.buffer == VK_NULL_HANDLE)
        {
          m_lightSet.init(m_app, &m_alloc, &m_uploader);
          initRendererBuffers();
        }
        initPipelines();
        initRtDescriptorSet();
        initRtPipeline();
        initDescriptorSetPostProcessing();
        initPipelinePostProcessing();
      }
    }
    else if(needsDataUpdate)
    {
      // For data-only updates during streaming, rebuild pipelines and descriptors
      // This is needed because the data storage buffers have changed
      initPipelines();
      if(prmSelectedPipeline == PIPELINE_RTX)
      {
        initRtDescriptorSet();
        initRtPipeline();
      }
      initDescriptorSetPostProcessing();
      initPipelinePostProcessing();
    }
  }

  // light buffer is never reallocated
  // updates does not require description set changes
  if(m_requestUpdateLightsBuffer)
  {
    m_lightSet.updateBuffer();
    m_requestUpdateLightsBuffer = false;
  }

  // reset request
  m_requestUpdateSplatData = m_requestUpdateSplatAs = m_requestUpdateMeshData = m_requestUpdateShaders =
      m_requestUpdateLightsBuffer = m_requestDeleteSelectedMesh = false;
}

}  // namespace vk_viewer
