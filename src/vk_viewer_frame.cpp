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
// Contains: Frame lifecycle helper methods extracted from onRender

namespace vk_viewer {

// Helper function to create an off-axis (asymmetric) stereo projection matrix
// This produces proper stereo with zero parallax at the convergence distance
// eyeOffset: positive for right eye, negative for left eye
static glm::mat4 makeOffAxisStereoProjection(float fovYRad, float aspect, float nearZ, float farZ, float eyeOffset, float convergenceDist)
{
  const float top    = nearZ * tanf(fovYRad * 0.5f);
  const float bottom = -top;
  const float width  = top * aspect;

  const float frustumShift = (eyeOffset * nearZ) / convergenceDist;
  const float left  = -width + frustumShift;
  const float right = width + frustumShift;

  glm::mat4 proj(0.0f);
  proj[0][0] = (2.0f * nearZ) / (right - left);
  proj[1][1] = -(2.0f * nearZ) / (top - bottom);  // Y flip for Vulkan
  proj[2][0] = (right + left) / (right - left);
  proj[2][1] = (top + bottom) / (top - bottom);
  proj[2][2] = farZ / (nearZ - farZ);
  proj[2][3] = -1.0f;
  proj[3][2] = (nearZ * farZ) / (nearZ - farZ);

  return proj;
}

bool VkViewer::beginFrame(FrameRenderContext& ctx)
{
  ctx.frameIndex = m_app->getFrameCycleIndex();
  m_frameIndex = ctx.frameIndex;
  
  m_currentFrameInfoOffset = m_frameIndex * 4 * (uint32_t)m_frameInfoStride;
  
  ctx.viewSize = glm::vec2(m_viewSize.x, m_viewSize.y);
  ctx.clipPlanes = cameraManip->getClipPlanes();

#ifdef WITH_OPENXR
  if(m_xrInitialized && m_xr && m_xr->isValid())
  {
    if(m_xrResizedThisFrame)
    {
      processUpdateRequests();
      return false;
    }

    GsOpenXr::BeginFrameResult frameResult;
    {
      auto timer = m_profilerTimeline->frameSection("XR BeginFrame");
      frameResult = m_xr->beginFrame();
    }
    
    if(frameResult == GsOpenXr::BeginFrameResult::SkipFully)
    {
      processUpdateRequests();
      return false;
    }
    
    ctx.xrFrameActive = true;
    ctx.xrShouldRender = (frameResult == GsOpenXr::BeginFrameResult::RenderFully);
    
    if(!ctx.xrShouldRender)
    {
      processUpdateRequests();
      m_xr->endFrame();
      return false;
    }

    if(!m_xr->acquireSwapchainImages(m_xrColorImage, m_xrDepthImage, m_xrMotionImage))
    {
      m_xr->endFrame();
      return false;
    }

    // Acquire Environment Depth if enabled
    VkImage envDepthImage = VK_NULL_HANDLE;
    XrEnvironmentDepthImageMETA envDepthInfo = {XR_TYPE_ENVIRONMENT_DEPTH_IMAGE_META};
    
    if (m_xr->isEnvironmentDepthEnabled() && m_xr->acquireEnvironmentDepthImage(envDepthImage, envDepthInfo))
    {
        VkImageView envDepthView = VK_NULL_HANDLE;
        auto it = m_envDepthImageViews.find(envDepthInfo.swapchainIndex);
        if (it != m_envDepthImageViews.end())
        {
            envDepthView = it->second;
        }
        else
        {
            VkImageViewCreateInfo viewInfo = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            viewInfo.image = envDepthImage;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
            viewInfo.format = m_depthFormat;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            viewInfo.subresourceRange.baseMipLevel = 0;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.baseArrayLayer = 0;
            viewInfo.subresourceRange.layerCount = 2;
            
            vkCreateImageView(m_device, &viewInfo, nullptr, &envDepthView);
            m_envDepthImageViews[envDepthInfo.swapchainIndex] = envDepthView;
        }

        if (m_descriptorSet != VK_NULL_HANDLE && envDepthView != VK_NULL_HANDLE)
        {
            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; 
            imageInfo.imageView = envDepthView;
            imageInfo.sampler = m_sampler; 

            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = m_descriptorSet;
            write.dstBinding = BINDING_ENV_DEPTH_TEXTURE;
            write.dstArrayElement = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.descriptorCount = 1;
            write.pImageInfo = &imageInfo;

            vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
        }

        prmFrame.envDepthAvailable = 1;
        prmFrame.envDepthNear = envDepthInfo.nearZ;
        prmFrame.envDepthFar = envDepthInfo.farZ;
        
        for (int i = 0; i < 2; ++i)
        {
            prmFrame.envDepthViewMatrixArray[i] = GsOpenXr::createViewMatrix(envDepthInfo.views[i].pose);
            prmFrame.envDepthProjectionMatrixArray[i] = GsOpenXr::createProjectionMatrix(envDepthInfo.views[i].fov, envDepthInfo.nearZ, envDepthInfo.farZ);
            ctx.envDepthViewMatrices[i] = prmFrame.envDepthViewMatrixArray[i];
            ctx.envDepthProjMatrices[i] = prmFrame.envDepthProjectionMatrixArray[i];
        }
    }
    else
    {
        prmFrame.envDepthAvailable = 0;
    }

    {
      auto timer = m_profilerTimeline->frameSection("XR LocateViews");
      if(!m_xr->locateViews(ctx.clipPlanes.x, ctx.clipPlanes.y))
      {
        processUpdateRequests();
        m_xr->releaseSwapchainImages(); 
        m_xr->endFrame();
        return false;
      }
    }

    {
      auto timer = m_profilerTimeline->frameSection("XR PollInput");
      m_xr->pollControllerInput();
      m_xr->pollHandInput();
    }

    auto now = std::chrono::steady_clock::now();
    if(!m_xrFirstFrame)
    {
      float deltaTime = std::chrono::duration<float>(now - m_xrLastFrameTime).count();
      deltaTime = std::min(deltaTime, 0.1f);
      updateXrLocomotion(deltaTime);
    }
    m_xrLastFrameTime = now;
    m_xrFirstFrame = false;
  }
#endif

  return true;
}

void VkViewer::buildContentState(FrameRenderContext& ctx)
{
  ctx.splatCount = 0;
  if(m_splatLoader.getStatus() == SplatLoaderAsync::State::STATE_READY)
  {
    if(isLccStreamingActive())
    {
      if(m_lccTileManager->isPackedMode() && m_lccPackedSplatCount > 0)
      {
        // GPU-side decompression: use packed splat count directly
        // m_splatSet is not used for rendering, but we need to set the count
        // for the rest of the pipeline (sorting, etc.)
        ctx.splatCount = m_lccPackedSplatCount;
      }
      else if(!m_streamingSplatSet.positions.empty())
      {
        // CPU-side decompression: move data from streaming set
        m_splatSet.clear();
        m_splatSet.positions = std::move(m_streamingSplatSet.positions);
        m_splatSet.f_dc = std::move(m_streamingSplatSet.f_dc);
        m_splatSet.f_rest = std::move(m_streamingSplatSet.f_rest);
        m_splatSet.opacity = std::move(m_streamingSplatSet.opacity);
        m_splatSet.scale = std::move(m_streamingSplatSet.scale);
        m_splatSet.rotation = std::move(m_streamingSplatSet.rotation);
        m_splatSet.has_time_data = m_streamingSplatSet.has_time_data;
        m_splatSet.minTime = m_streamingSplatSet.minTime;
        m_splatSet.maxTime = m_streamingSplatSet.maxTime;
        ctx.splatCount = (uint32_t)m_splatSet.size();
      }
    }
    else
    {
      ctx.splatCount = (uint32_t)m_splatSet.size();
    }
  }
  
  ctx.hasSplats = (ctx.splatCount > 0);
  ctx.hasDepthContent = m_enableDepthRendering;
  ctx.hasMeshes = !m_meshSetVk.instances.empty();
  ctx.shadersValid = m_shaders.valid;
  
  // Determine content mode
  if(ctx.hasSplats && ctx.hasDepthContent)
  {
    ctx.contentMode = RenderContentMode::SplatsAndDepth;
  }
  else if(ctx.hasDepthContent)
  {
    ctx.contentMode = RenderContentMode::DepthOnly;
  }
  else
  {
    ctx.contentMode = RenderContentMode::SplatsOnly;
  }
  
  // Determine output mode
#ifdef WITH_OPENXR
  if(ctx.xrFrameActive && m_xr)
  {
    VkExtent2D perEye = m_xr->getPerEyeExtent();
    if(m_xr->supportsMultiview() && m_graphicsPipelineGsMeshMultiview != VK_NULL_HANDLE)
    {
      ctx.outputMode = OutputMode::XrMultiview;
      ctx.useXrMultiview = true;
      initXrMultiviewResources(ctx.cmd, perEye);
    }
    else
    {
      ctx.outputMode = OutputMode::XrDualPass;
    }
  }
  else
#endif
  if(m_renderSBS)
  {
    ctx.outputMode = OutputMode::DesktopStereoSBS;
  }
  else
  {
    ctx.outputMode = OutputMode::DesktopMono;
  }
  
  // Determine if RTX pipeline
  ctx.useRtxPipeline = (prmSelectedPipeline == PIPELINE_RTX);
  ctx.useStochasticPipeline = (prmSelectedPipeline == PIPELINE_STOCHASTIC_GS);
  ctx.raytraceMeshDepth = ctx.shadersValid && ctx.hasMeshes && prmSelectedPipeline == PIPELINE_HYBRID_3DGUT;
  
  // Color buffer selection
  ctx.colorBufferId = COLOR_MAIN;
  if(prmRtx.temporalSampling && prmFrame.frameSampleId > 0)
  {
    ctx.colorBufferId = COLOR_AUX1;
  }
}

void VkViewer::buildViews(FrameRenderContext& ctx)
{
  ctx.views.clear();
  
  cameraManip->getLookat(m_eye, m_center, m_up);
  ctx.eyePosition = glm::vec3(m_eye);
  ctx.centerPosition = glm::vec3(m_center);
  ctx.upVector = glm::vec3(m_up);
  ctx.viewMatrix = cameraManip->getViewMatrix();
  ctx.projMatrix = cameraManip->getPerspectiveMatrix();

  if(isLccStreamingActive())
  {
    glm::mat4 viewProj = ctx.projMatrix * ctx.viewMatrix;
    updateLccStreaming(viewProj, m_eye, 0.0f);
  }

#ifdef WITH_OPENXR
  if(ctx.xrFrameActive && m_xr)
  {
    VkExtent2D perEye = m_xr->getPerEyeExtent();
    
    if(ctx.useXrMultiview)
    {
      GsOpenXr::EyeData leftEye = m_xr->getEyeData(0);
      GsOpenXr::EyeData rightEye = m_xr->getEyeData(1);

      float realIPD = glm::distance(leftEye.eyePos, rightEye.eyePos);
      if (realIPD > 0.001f) m_stereoSeparation = realIPD;

      ctx.eyePosition = (leftEye.eyePos + rightEye.eyePos) * 0.5f;
      m_eye = ctx.eyePosition;
      ctx.viewMatrix = leftEye.view;
      ctx.projMatrix = leftEye.proj;
      
      // Upload multiview matrices to UBO
      prmFrame.viewMatrixArray[0] = leftEye.view;
      prmFrame.viewMatrixArray[1] = rightEye.view;
      prmFrame.viewInverseArray[0] = glm::inverse(leftEye.view);
      prmFrame.viewInverseArray[1] = glm::inverse(rightEye.view);
      prmFrame.projectionMatrixArray[0] = leftEye.proj;
      prmFrame.projectionMatrixArray[1] = rightEye.proj;
      prmFrame.projInverseArray[0] = glm::inverse(leftEye.proj);
      prmFrame.projInverseArray[1] = glm::inverse(rightEye.proj);
      prmFrame.cameraPositionArray[0] = leftEye.eyePos;
      prmFrame.cameraPositionArray[1] = rightEye.eyePos;
      for(int eyeIdx = 0; eyeIdx < 2; ++eyeIdx)
      {
        glm::quat viewQuat = glm::quat_cast(prmFrame.viewMatrixArray[eyeIdx]);
        prmFrame.viewQuatArray[eyeIdx] = glm::vec4(viewQuat.x, viewQuat.y, viewQuat.z, viewQuat.w);
        prmFrame.viewTransArray[eyeIdx] = prmFrame.viewMatrixArray[eyeIdx][3];
      }
      prmFrame.multiviewEnabled = 1;
      
      // No per-view entries needed for multiview - rendering uses gl_ViewIndex
      return;
    }
    else
    {
      // Dual-pass XR
      const float halfWidth = float(perEye.width);
      const uint32_t halfWidthInt = perEye.width;
      const uint32_t heightInt = perEye.height;

      for(uint32_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx)
      {
        GsOpenXr::EyeData eyeData = m_xr->getEyeData(eyeIdx);

        RenderView xrView;
        xrView.eye = eyeData.eyePos;
        xrView.view = eyeData.view;
        xrView.proj = eyeData.proj;
        xrView.stereoShift = glm::vec2(0.0f, 0.0f);
        xrView.viewIndex = eyeIdx;

        float xOffset = float(eyeIdx * perEye.width);
        xrView.viewport = {xOffset, 0.0f, halfWidth, float(perEye.height), 0.0f, 1.0f};
        xrView.scissor = {{static_cast<int32_t>(eyeIdx * perEye.width), 0}, {halfWidthInt, heightInt}};

        ctx.views.push_back(xrView);
      }

      GsOpenXr::EyeData leftEye = m_xr->getEyeData(0);
      GsOpenXr::EyeData rightEye = m_xr->getEyeData(1);
      float realIPD = glm::distance(leftEye.eyePos, rightEye.eyePos);
      if (realIPD > 0.001f) m_stereoSeparation = realIPD;

      ctx.eyePosition = (leftEye.eyePos + rightEye.eyePos) * 0.5f;
      m_eye = ctx.eyePosition;
      ctx.viewMatrix = leftEye.view;
      ctx.projMatrix = leftEye.proj;
      return;
    }
  }
#endif

  if(m_renderSBS)
  {
    const float fovRad = cameraManip->getRadFov();
    const float halfAspect = (float(m_viewSize.x) * 0.5f) / float(m_viewSize.y);
    const float halfSeparation = m_stereoSeparation * 0.5f;
    const glm::vec3 rightDir = glm::vec3(ctx.viewMatrix[0][0], ctx.viewMatrix[1][0], ctx.viewMatrix[2][0]);
    const float halfWidth = float(m_viewSize.x) * 0.5f;
    const uint32_t halfWidthInt = static_cast<uint32_t>(halfWidth);
    const uint32_t heightInt = static_cast<uint32_t>(m_viewSize.y);

    const bool is3DGUT = (prmSelectedPipeline == PIPELINE_MESH_3DGUT) || (prmSelectedPipeline == PIPELINE_HYBRID_3DGUT);
    const bool useOffAxisMatrix = m_stereoOffAxisProj && !is3DGUT;
    
    glm::mat4 symmetricProj = glm::perspective(fovRad, halfAspect, ctx.clipPlanes.x, ctx.clipPlanes.y);
    symmetricProj[1][1] *= -1;
    
    const float height = float(m_viewSize.y);
    const float focalLengthPixels = (height * 0.5f) / tanf(fovRad * 0.5f);

    // Left eye
    RenderView left;
    left.eye = ctx.eyePosition - (rightDir * halfSeparation);
    left.view = glm::lookAt(left.eye, ctx.centerPosition, ctx.upVector);
    left.viewIndex = 0;
    if(useOffAxisMatrix)
    {
      left.proj = makeOffAxisStereoProjection(fovRad, halfAspect, ctx.clipPlanes.x, ctx.clipPlanes.y, -halfSeparation, m_stereoConvergence);
      left.stereoShift = glm::vec2(0.0f, 0.0f);
    }
    else
    {
      left.proj = symmetricProj;
      if(is3DGUT && m_stereoOffAxisProj)
        left.stereoShift = glm::vec2((-halfSeparation * focalLengthPixels) / m_stereoConvergence, 0.0f);
      else
        left.stereoShift = glm::vec2(0.0f, 0.0f);
    }
    left.viewport = {0.0f, 0.0f, halfWidth, float(m_viewSize.y), 0.0f, 1.0f};
    left.scissor = {{0, 0}, {halfWidthInt, heightInt}};
    ctx.views.push_back(left);

    // Right eye
    RenderView right;
    right.eye = ctx.eyePosition + (rightDir * halfSeparation);
    right.view = glm::lookAt(right.eye, ctx.centerPosition, ctx.upVector);
    right.viewIndex = 1;
    if(useOffAxisMatrix)
    {
      right.proj = makeOffAxisStereoProjection(fovRad, halfAspect, ctx.clipPlanes.x, ctx.clipPlanes.y, halfSeparation, m_stereoConvergence);
      right.stereoShift = glm::vec2(0.0f, 0.0f);
    }
    else
    {
      right.proj = symmetricProj;
      if(is3DGUT && m_stereoOffAxisProj)
        right.stereoShift = glm::vec2((halfSeparation * focalLengthPixels) / m_stereoConvergence, 0.0f);
      else
        right.stereoShift = glm::vec2(0.0f, 0.0f);
    }
    right.viewport = {halfWidth, 0.0f, halfWidth, float(m_viewSize.y), 0.0f, 1.0f};
    right.scissor = {{static_cast<int32_t>(halfWidth), 0}, {halfWidthInt, heightInt}};
    ctx.views.push_back(right);
  }
  else
  {
    // Mono view
    RenderView mono;
    mono.view = ctx.viewMatrix;
    mono.proj = ctx.projMatrix;
    mono.eye = ctx.eyePosition;
    mono.stereoShift = glm::vec2(0.0f, 0.0f);
    mono.viewIndex = 0;
    mono.viewport = {0.0f, 0.0f, float(m_viewSize.x), float(m_viewSize.y), 0.0f, 1.0f};
    mono.scissor = {{0, 0}, {static_cast<uint32_t>(m_viewSize.x), static_cast<uint32_t>(m_viewSize.y)}};
    ctx.views.push_back(mono);
  }
}

void VkViewer::prepareSceneForFrame(FrameRenderContext& ctx)
{
  if(!ctx.shadersValid || !ctx.hasSplats)
    return;

  collectReadBackValuesIfNeeded();

#ifdef WITH_OPENXR
  if (ctx.useXrMultiview && m_xr) {
    VkExtent2D perEye = m_xr->getPerEyeExtent();
    updateAndUploadFrameInfoUBO(ctx.cmd, ctx.splatCount, ctx.viewMatrix, ctx.projMatrix, 
                                ctx.eyePosition, glm::vec2(perEye.width, perEye.height));
  } else {
    updateAndUploadFrameInfoUBO(ctx.cmd, ctx.splatCount, ctx.viewMatrix, ctx.projMatrix, 
                                ctx.eyePosition, ctx.viewSize);
  }
#else
  updateAndUploadFrameInfoUBO(ctx.cmd, ctx.splatCount, ctx.viewMatrix, ctx.projMatrix, 
                              ctx.eyePosition, ctx.viewSize);
#endif

  if(prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX)
  {
    m_profilerTimeline->asyncRemoveTimer("CPU Dist");
    m_profilerTimeline->asyncRemoveTimer("CPU Sort");

    // Temporal stability: check if camera is stationary to skip sorting
    bool shouldSort = true;
    m_sortSkippedThisFrame = false;

    if(prmRaster.gpuSortSkipWhenStable && !prmRaster.gpuSortForceEveryFrame && m_lastSortValid)
    {
      // Calculate camera direction from eye to center
      glm::dvec3 currentDirection = glm::normalize(m_center - m_eye);
      
      // Check position delta
      float positionDelta = glm::length(m_eye - m_lastSortCameraPosition);
      
      // Check angle delta (dot product gives cos of angle)
      float dotProduct = glm::dot(currentDirection, m_lastSortCameraDirection);
      dotProduct = glm::clamp(dotProduct, -1.0f, 1.0f);
      float angleDelta = std::acos(dotProduct);

#ifdef WITH_OPENXR
      // Use tighter thresholds for XR (head tracking can be jittery)
      bool isXrActive = m_xrInitialized && m_xr && m_xr->isValid();
      float posEpsilon = isXrActive ? prmRaster.gpuSortPositionEpsilon * 0.1f : prmRaster.gpuSortPositionEpsilon;
      float angleEpsilon = isXrActive ? prmRaster.gpuSortAngleEpsilon * 0.1f : prmRaster.gpuSortAngleEpsilon;
#else
      float posEpsilon = prmRaster.gpuSortPositionEpsilon;
      float angleEpsilon = prmRaster.gpuSortAngleEpsilon;
#endif

      if(positionDelta < posEpsilon && angleDelta < angleEpsilon)
      {
        shouldSort = false;
        m_sortSkippedThisFrame = true;
        LOGD("GPU sort skipped: pos delta=%.6f, angle delta=%.6f rad\n", positionDelta, angleDelta);
      }
    }

    // Process sorting - distance computation always runs, but radix sort can be skipped
    // when camera is stationary (temporal stability optimization)
    bool skipRadixSort = !shouldSort;
    processSortingOnGPU(ctx.cmd, ctx.splatCount, ctx.viewMatrix, skipRadixSort);
    
    // Update last camera state after sorting (only if we actually sorted)
    if (shouldSort)
    {
      m_lastSortCameraPosition = m_eye;
      m_lastSortCameraDirection = glm::normalize(m_center - m_eye);
      m_lastSortValid = true;
    }
  }
  else
  {
    tryConsumeAndUploadCpuSortingResult(ctx.cmd, ctx.splatCount);
  }
}

void VkViewer::clearMainTargets(FrameRenderContext& ctx)
{
  nvvk::cmdImageMemoryBarrier(ctx.cmd, {m_gBuffers.getDepthImage(),
                                        VK_IMAGE_LAYOUT_UNDEFINED,
                                        VK_IMAGE_LAYOUT_GENERAL,
                                        {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}});

  VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
  colorAttachment.imageView = m_gBuffers.getColorImageView(ctx.colorBufferId);
  colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  colorAttachment.clearValue = {m_clearColor};

  VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
  depthAttachment.imageView = m_gBuffers.getDepthImageView();
  depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depthAttachment.clearValue = {.depthStencil = DEFAULT_VkClearDepthStencilValue};

  VkRenderingInfo clearRenderingInfo = DEFAULT_VkRenderingInfo;
  clearRenderingInfo.renderArea = {{0, 0}, {static_cast<uint32_t>(m_viewSize.x), static_cast<uint32_t>(m_viewSize.y)}};
  clearRenderingInfo.colorAttachmentCount = 1;
  clearRenderingInfo.pColorAttachments = &colorAttachment;
  clearRenderingInfo.pDepthAttachment = &depthAttachment;

  nvvk::cmdImageMemoryBarrier(ctx.cmd, {m_gBuffers.getColorImage(ctx.colorBufferId), VK_IMAGE_LAYOUT_GENERAL,
                                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
  nvvk::cmdImageMemoryBarrier(ctx.cmd, {m_gBuffers.getDepthImage(),
                                        VK_IMAGE_LAYOUT_GENERAL,
                                        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                        {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}});

  vkCmdBeginRendering(ctx.cmd, &clearRenderingInfo);
  vkCmdEndRendering(ctx.cmd);

  nvvk::cmdImageMemoryBarrier(ctx.cmd, {m_gBuffers.getColorImage(ctx.colorBufferId),
                                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});
  nvvk::cmdImageMemoryBarrier(ctx.cmd, {m_gBuffers.getDepthImage(),
                                        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                        VK_IMAGE_LAYOUT_GENERAL,
                                        {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}});
}

void VkViewer::renderSingleView(FrameRenderContext& ctx, const RenderView& view)
{
  prmFrame.multiviewEnabled = 0;
  
  // Ensure shader uses correct environment depth matrix for current eye
  if (prmFrame.envDepthAvailable && view.viewIndex < 2)
  {
    prmFrame.envDepthViewMatrixArray[0] = ctx.envDepthViewMatrices[view.viewIndex];
    prmFrame.envDepthProjectionMatrixArray[0] = ctx.envDepthProjMatrices[view.viewIndex];
  }

  updateAndUploadFrameInfoUBO(ctx.cmd, ctx.splatCount, view.view, view.proj, view.eye,
                              {view.viewport.width, view.viewport.height},
                              {view.viewport.x, view.viewport.y},
                              view.stereoShift);

  if(ctx.raytraceMeshDepth)
  {
    raytrace(ctx.cmd, true, glm::ivec2(view.viewport.x, view.viewport.y),
             glm::ivec2(view.viewport.width, view.viewport.height));
  }

  // Drawing the primitives in the G-Buffer
  {
    auto timerSection = m_profilerGpuTimer.cmdFrameSection(ctx.cmd, "Rasterization");

    VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
    colorAttachment.imageView = m_gBuffers.getColorImageView(ctx.colorBufferId);
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;

    VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
    depthAttachment.imageView = m_gBuffers.getDepthImageView();
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;

    VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
    renderingInfo.renderArea = {{0, 0}, {static_cast<uint32_t>(m_viewSize.x), static_cast<uint32_t>(m_viewSize.y)}};
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = &depthAttachment;

    nvvk::cmdImageMemoryBarrier(ctx.cmd, {m_gBuffers.getColorImage(ctx.colorBufferId), VK_IMAGE_LAYOUT_GENERAL,
                                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
    nvvk::cmdImageMemoryBarrier(ctx.cmd, {m_gBuffers.getDepthImage(),
                                          VK_IMAGE_LAYOUT_GENERAL,
                                          VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                          {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}});

    if(ctx.hasDepthContent)
    {
      updateDepthRendering(ctx.cmd);
    }

    vkCmdBeginRendering(ctx.cmd, &renderingInfo);

    vkCmdSetViewportWithCount(ctx.cmd, 1, &view.viewport);
    vkCmdSetScissorWithCount(ctx.cmd, 1, &view.scissor);

    if(ctx.shadersValid && ctx.hasMeshes && !ctx.raytraceMeshDepth)
    {
      drawMeshPrimitives(ctx.cmd);
    }

    if(ctx.shadersValid && ctx.hasSplats)
    {
      drawSplatPrimitives(ctx.cmd, ctx.splatCount, &view.view);
    }

    if(ctx.hasDepthContent)
    {
      drawVdzMesh(ctx.cmd);
    }

    vkCmdEndRendering(ctx.cmd);

    nvvk::cmdImageMemoryBarrier(ctx.cmd, {m_gBuffers.getColorImage(ctx.colorBufferId),
                                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});
    nvvk::cmdImageMemoryBarrier(ctx.cmd, {m_gBuffers.getDepthImage(),
                                          VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                          VK_IMAGE_LAYOUT_GENERAL,
                                          {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}});
  }

  // Raytrace secondary rays if needed (hybrid modes)
  if(ctx.shadersValid && ctx.hasSplats && m_splatSetVk.rtxValid && ctx.hasMeshes
     && (prmSelectedPipeline == PIPELINE_HYBRID || prmSelectedPipeline == PIPELINE_HYBRID_3DGUT))
  {
    raytrace(ctx.cmd, false, glm::ivec2(view.viewport.x, view.viewport.y),
             glm::ivec2(view.viewport.width, view.viewport.height));
  }
}

void VkViewer::renderPerViewPath(FrameRenderContext& ctx)
{
  for(const auto& view : ctx.views)
  {
    renderSingleView(ctx, view);
  }
}

void VkViewer::finalizeFrame(FrameRenderContext& ctx)
{
  // Post-processing
  if((prmRtx.temporalSampling && prmFrame.frameSampleId > 0) || prmFrame.linearToSrgb != 0)
  {
    postProcess(ctx.cmd);
  }

  readBackIndirectParametersIfNeeded(ctx.cmd);
  updateRenderingMemoryStatistics(ctx.cmd, ctx.splatCount);

#ifdef WITH_OPENXR
  if(ctx.xrFrameActive && m_xr)
  {
    copyToXrSwapchain(ctx.cmd);
    m_xr->releaseSwapchainImages();
    {
      auto timer = m_profilerTimeline->frameSection("XR EndFrame");
      m_xr->endFrame();
    }
  }
#endif
}

void VkViewer::renderRtxFrame(FrameRenderContext& ctx)
{
  if(!m_splatSetVk.rtxValid)
  {
    prmSelectedPipeline = PIPELINE_MESH;
    return;
  }

  if(prmRtx.temporalSampling && !updateFrameCounter())
    return;

  collectReadBackValuesIfNeeded();

  prmFrame.multiviewEnabled = 0;
  updateAndUploadFrameInfoUBO(ctx.cmd, ctx.splatCount, ctx.viewMatrix, ctx.projMatrix, 
                              ctx.eyePosition, ctx.viewSize);
  raytrace(ctx.cmd);

#ifdef WITH_DLSS_RR
  if(m_dlssRREnabled && m_dlssRRInitialized && m_dlssRR && m_dlssRR->isValid())
  {
    nvvk::cmdImageMemoryBarrier(ctx.cmd, {m_gBuffers.getColorImage(COLOR_MAIN),
                                          VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL});
    nvvk::cmdImageMemoryBarrier(ctx.cmd, {m_gBuffers.getColorImage(COLOR_DLSS_OUTPUT),
                                          VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL});

    glm::uvec2 renderSize = glm::uvec2(m_viewSize.x, m_viewSize.y);
    NVSDK_NGX_Result result = m_dlssRR->denoise(ctx.cmd, renderSize, prmFrame.dlssJitter,
                                                 ctx.viewMatrix, ctx.projMatrix, m_dlssRRNeedsReset);
    if(NVSDK_NGX_SUCCEED(result))
    {
      VkImageCopy copyRegion = {};
      copyRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      copyRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      copyRegion.extent = {static_cast<uint32_t>(m_viewSize.x), static_cast<uint32_t>(m_viewSize.y), 1};

      nvvk::cmdImageMemoryBarrier(ctx.cmd, {m_gBuffers.getColorImage(COLOR_DLSS_OUTPUT),
                                            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL});
      nvvk::cmdImageMemoryBarrier(ctx.cmd, {m_gBuffers.getColorImage(COLOR_MAIN),
                                            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL});

      vkCmdCopyImage(ctx.cmd, m_gBuffers.getColorImage(COLOR_DLSS_OUTPUT), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     m_gBuffers.getColorImage(COLOR_MAIN), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

      nvvk::cmdImageMemoryBarrier(ctx.cmd, {m_gBuffers.getColorImage(COLOR_MAIN),
                                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});
    }

    m_dlssRRNeedsReset = false;
    m_dlssRRFrameIndex++;
  }
#endif
}

void VkViewer::renderRtxStereoFrame(FrameRenderContext& ctx)
{
  if(!m_splatSetVk.rtxValid)
  {
    prmSelectedPipeline = PIPELINE_MESH;
    return;
  }

  collectReadBackValuesIfNeeded();

  const float fovRad = cameraManip->getRadFov();
  const float halfAspect = (float(m_viewSize.x) * 0.5f) / float(m_viewSize.y);
  const float halfSeparation = m_stereoSeparation * 0.5f;
  const glm::vec3 rightDir = glm::vec3(ctx.viewMatrix[0][0], ctx.viewMatrix[1][0], ctx.viewMatrix[2][0]);
  const uint32_t halfWidth = static_cast<uint32_t>(m_viewSize.x) / 2;
  const uint32_t height = static_cast<uint32_t>(m_viewSize.y);

  prmFrame.multiviewEnabled = 0;

  // Left eye
  glm::vec3 leftEye = ctx.eyePosition - (rightDir * halfSeparation);
  glm::mat4 leftView = glm::lookAt(leftEye, ctx.centerPosition, ctx.upVector);
  glm::mat4 leftProj;
  if(m_stereoOffAxisProj)
  {
    leftProj = makeOffAxisStereoProjection(fovRad, halfAspect, ctx.clipPlanes.x, ctx.clipPlanes.y, -halfSeparation, m_stereoConvergence);
  }
  else
  {
    leftProj = glm::perspective(fovRad, halfAspect, ctx.clipPlanes.x, ctx.clipPlanes.y);
    leftProj[1][1] *= -1;
  }
  updateAndUploadFrameInfoUBO(ctx.cmd, ctx.splatCount, leftView, leftProj, leftEye, glm::vec2(halfWidth, height));
  raytrace(ctx.cmd, false, glm::ivec2(0, 0), glm::ivec2(halfWidth, height));

  // Right eye
  glm::vec3 rightEye = ctx.eyePosition + (rightDir * halfSeparation);
  glm::mat4 rightView = glm::lookAt(rightEye, ctx.centerPosition, ctx.upVector);
  glm::mat4 rightProj;
  if(m_stereoOffAxisProj)
  {
    rightProj = makeOffAxisStereoProjection(fovRad, halfAspect, ctx.clipPlanes.x, ctx.clipPlanes.y, halfSeparation, m_stereoConvergence);
  }
  else
  {
    rightProj = glm::perspective(fovRad, halfAspect, ctx.clipPlanes.x, ctx.clipPlanes.y);
    rightProj[1][1] *= -1;
  }
  updateAndUploadFrameInfoUBO(ctx.cmd, ctx.splatCount, rightView, rightProj, rightEye, glm::vec2(halfWidth, height));
  raytrace(ctx.cmd, false, glm::ivec2(halfWidth, 0), glm::ivec2(halfWidth, height));
}

}  // namespace vk_viewer
