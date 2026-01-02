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
// Contains: Main render loop (onRender) and update request processing

namespace vk_gaussian_splatting {

// Helper function to create an off-axis (asymmetric) stereo projection matrix
// This produces proper stereo with zero parallax at the convergence distance
// eyeOffset: positive for right eye, negative for left eye
static glm::mat4 makeOffAxisStereoProjection(float fovYRad, float aspect, float nearZ, float farZ, float eyeOffset, float convergenceDist)
{
  const float top    = nearZ * tanf(fovYRad * 0.5f);
  const float bottom = -top;
  const float width  = top * aspect;

  // Shift the frustum based on eye offset and convergence distance
  // At convergence distance, both eyes see the same point at screen center
  const float frustumShift = (eyeOffset * nearZ) / convergenceDist;

  const float left  = -width + frustumShift;
  const float right = width + frustumShift;

  // Build asymmetric frustum projection matrix (Vulkan-style with Y flip)
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

void GaussianSplatting::onRender(VkCommandBuffer cmd)
{
  NVVK_DBG_SCOPE(cmd);

  // Sync frame index with application frame cycle
  m_frameIndex = m_app->getFrameCycleIndex();

#ifdef WITH_OPENXR
  // Handle OpenXR frame lifecycle

  bool xrFrameActive = false;
  bool xrShouldRender = false;
  if(m_xrInitialized && m_xr && m_xr->isValid())
  {
    // Skip XR rendering on the frame where we just resized GBuffers
    // to allow descriptor sets to be properly updated
    if(m_xrResizedThisFrame)
    {
      processUpdateRequests();
      return;
    }

    GsOpenXr::BeginFrameResult frameResult = m_xr->beginFrame();
    
    if(frameResult == GsOpenXr::BeginFrameResult::SkipFully)
    {
      // Session not ready, skip everything
      processUpdateRequests();
      return;
    }
    
    xrFrameActive = true;
    xrShouldRender = (frameResult == GsOpenXr::BeginFrameResult::RenderFully);
    
    if(!xrShouldRender)
    {
      // Must still call endFrame even when not rendering
      processUpdateRequests();
      m_xr->endFrame();
      return;
    }

    // GBuffer resize is now handled in onPreRender() before command recording

    // Acquire XR swapchain images
    if(!m_xr->acquireSwapchainImages(m_xrColorImage, m_xrDepthImage))
    {
      m_xr->endFrame();
      return;
    }

    // Locate views with current clip planes
    glm::vec2 clipPlanes = cameraManip->getClipPlanes();
    m_xr->locateViews(clipPlanes.x, clipPlanes.y);

    // Update locomotion from controller input (must be after beginFrame for valid time)
    auto now = std::chrono::steady_clock::now();
    if(!m_xrFirstFrame)
    {
      float deltaTime = std::chrono::duration<float>(now - m_xrLastFrameTime).count();
      deltaTime = std::min(deltaTime, 0.1f);  // Clamp to avoid large jumps
      updateXrLocomotion(deltaTime);
    }
    m_xrLastFrameTime = now;
    m_xrFirstFrame = false;
  }
#endif

  // update buffers, rebuild shaders and pipelines if needed
  processUpdateRequests();

  // 0 if not ready so the rendering does not
  // touch the splat set while loading
  // getStatus is thread safe.
  uint32_t splatCount = 0;
  if(m_splatLoader.getStatus() == SplatLoaderAsync::State::STATE_READY)
  {
    splatCount = (uint32_t)m_splatSet.size();
  }

  //////////////////
  // Full raytrace pipeline

  if(m_shaders.valid && splatCount && prmSelectedPipeline == PIPELINE_RTX)
  {
    if(!m_splatSetVk.rtxValid)
    {
      // let's switch back to raster, RTX is KO
      prmSelectedPipeline = PIPELINE_MESH;
      return;
    }

    if(prmRtx.temporalSampling && !updateFrameCounter())
      return;

    collectReadBackValuesIfNeeded();

    cameraManip->getLookat(m_eye, m_center, m_up);
    glm::mat4 viewMatrix = cameraManip->getViewMatrix();
    glm::mat4 projMatrix = cameraManip->getPerspectiveMatrix();

#ifdef WITH_OPENXR
     // RTX path with OpenXR
     if(xrFrameActive && m_xr)
     {
       VkExtent2D perEye = m_xr->getPerEyeExtent();
       const uint32_t halfWidth = perEye.width;
       const uint32_t height = perEye.height;

       // Get eye data for both views
       GsOpenXr::EyeData leftEyeData = m_xr->getEyeData(0);
       GsOpenXr::EyeData rightEyeData = m_xr->getEyeData(1);

       // Check if multiview is supported for mobile VR optimization
       if(m_xr->supportsMultiview())
       {
         // VK_KHR_multiview path: Upload both eye matrices, shader selects per-eye using gl_ViewIndex
         raytraceMultiview(cmd, false,
                           leftEyeData.view, leftEyeData.proj,
                           rightEyeData.view, rightEyeData.proj,
                           leftEyeData.eyePos, rightEyeData.eyePos,
                           glm::ivec2(halfWidth * 2, height));
       }
       else
       {
         // Dual-eye rendering: Two raytrace calls (legacy fallback)
         // Left eye
         updateAndUploadFrameInfoUBO(cmd, splatCount, leftEyeData.view, leftEyeData.proj, leftEyeData.eyePos, glm::vec2(halfWidth, height));
         raytrace(cmd, false, glm::ivec2(0, 0), glm::ivec2(halfWidth, height));

         // Right eye
         updateAndUploadFrameInfoUBO(cmd, splatCount, rightEyeData.view, rightEyeData.proj, rightEyeData.eyePos, glm::vec2(halfWidth, height));
         raytrace(cmd, false, glm::ivec2(halfWidth, 0), glm::ivec2(halfWidth, height));
       }

       // Note: XR swapchain uses VK_FORMAT_R8G8B8A8_SRGB, so the GPU automatically
       // applies linear->sRGB conversion during the blit to XR swapchain.
       // No manual post-process needed for XR path.

       readBackIndirectParametersIfNeeded(cmd);
       updateRenderingMemoryStatistics(cmd, splatCount);

       // Copy to XR swapchain and finish frame
       copyToXrSwapchain(cmd);
       m_xr->releaseSwapchainImages();
       m_xr->endFrame();
       return;
     }
#endif

    if(m_renderSBS)
    {
      prmFrame.multiviewEnabled = 0;
      const float     fovRad         = cameraManip->getRadFov();

      const float     halfAspect     = (float(m_viewSize.x) * 0.5f) / float(m_viewSize.y);
      const glm::vec2 clipPlanes     = cameraManip->getClipPlanes();
      const float     halfSeparation = m_stereoSeparation * 0.5f;

      const glm::vec3 rightDir = glm::vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]);

      const uint32_t halfWidth = static_cast<uint32_t>(m_viewSize.x) / 2;
      const uint32_t height    = static_cast<uint32_t>(m_viewSize.y);

      // Left eye
      glm::vec3 leftEye  = m_eye - (rightDir * halfSeparation);
      glm::mat4 leftView = glm::lookAt(leftEye, m_center, m_up);
      glm::mat4 leftProj;
      if(m_stereoOffAxisProj)
      {
        leftProj = makeOffAxisStereoProjection(fovRad, halfAspect, clipPlanes.x, clipPlanes.y, -halfSeparation, m_stereoConvergence);
      }
      else
      {
        leftProj = glm::perspective(fovRad, halfAspect, clipPlanes.x, clipPlanes.y);
        leftProj[1][1] *= -1;
      }
      updateAndUploadFrameInfoUBO(cmd, splatCount, leftView, leftProj, leftEye, glm::vec2(halfWidth, height));
      raytrace(cmd, false, glm::ivec2(0, 0), glm::ivec2(halfWidth, height));

      // Right eye
      glm::vec3 rightEye  = m_eye + (rightDir * halfSeparation);
      glm::mat4 rightView = glm::lookAt(rightEye, m_center, m_up);
      glm::mat4 rightProj;
      if(m_stereoOffAxisProj)
      {
        rightProj = makeOffAxisStereoProjection(fovRad, halfAspect, clipPlanes.x, clipPlanes.y, halfSeparation, m_stereoConvergence);
      }
      else
      {
        rightProj = glm::perspective(fovRad, halfAspect, clipPlanes.x, clipPlanes.y);
        rightProj[1][1] *= -1;
      }
      updateAndUploadFrameInfoUBO(cmd, splatCount, rightView, rightProj, rightEye, glm::vec2(halfWidth, height));
      raytrace(cmd, false, glm::ivec2(halfWidth, 0), glm::ivec2(halfWidth, height));
    }
    else
    {
      prmFrame.multiviewEnabled = 0;
      updateAndUploadFrameInfoUBO(cmd, splatCount, viewMatrix, projMatrix, m_eye, glm::vec2(m_viewSize.x, m_viewSize.y));

      raytrace(cmd);

#ifdef WITH_DLSS_RR
      // Apply DLSS-RR denoising if enabled
      if(m_dlssRREnabled && m_dlssRRInitialized && m_dlssRR && m_dlssRR->isValid())
      {
        // Transition G-buffer images to general layout for DLSS-RR
        nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(COLOR_MAIN),
                                          VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL});
        nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(COLOR_DLSS_OUTPUT),
                                          VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL});

        // Execute DLSS-RR denoising
        glm::uvec2 renderSize = glm::uvec2(m_viewSize.x, m_viewSize.y);
        NVSDK_NGX_Result result = m_dlssRR->denoise(cmd, renderSize, prmFrame.dlssJitter,
                                                     viewMatrix, projMatrix, m_dlssRRNeedsReset);
        if(NVSDK_NGX_SUCCEED(result))
        {
          // Copy denoised result back to main color buffer
          VkImageCopy copyRegion = {};
          copyRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
          copyRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
          copyRegion.extent = {static_cast<uint32_t>(m_viewSize.x), static_cast<uint32_t>(m_viewSize.y), 1};

          nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(COLOR_DLSS_OUTPUT),
                                            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL});
          nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(COLOR_MAIN),
                                            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL});

          vkCmdCopyImage(cmd, m_gBuffers.getColorImage(COLOR_DLSS_OUTPUT), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         m_gBuffers.getColorImage(COLOR_MAIN), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

          nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(COLOR_MAIN),
                                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});
        }

        m_dlssRRNeedsReset = false;
        m_dlssRRFrameIndex++;
      }
#endif
    }

  // Perform post processings if needed
  // Run post-process for temporal accumulation or linear-to-sRGB conversion
  if((prmRtx.temporalSampling && prmFrame.frameSampleId > 0) || prmFrame.linearToSrgb != 0)
  {
    postProcess(cmd);
  }

  updateDepthRendering(cmd);

  readBackIndirectParametersIfNeeded(cmd);

  updateRenderingMemoryStatistics(cmd, splatCount);


    // Attention: early return
    return;
  }

  ///////////////////
  // From this point we are using full raster or hybrid.

  if(prmRtx.temporalSampling && !updateFrameCounter())
    return;

  // Define stereo parameters
  struct StereoView
  {
    glm::mat4  view;
    glm::mat4  proj;
    glm::vec3  eye;
    glm::vec2  stereoShift;  // Principal point shift for off-axis stereo (in pixels)
    VkViewport viewport;
    VkRect2D   scissor;
  };

  std::vector<StereoView> views;
  views.reserve(m_renderSBS ? 2 : 1);

  cameraManip->getLookat(m_eye, m_center, m_up);
  glm::mat4 viewMatrix = cameraManip->getViewMatrix();
  glm::mat4 projMatrix = cameraManip->getPerspectiveMatrix();

#ifdef WITH_OPENXR
  // Use OpenXR view/projection when XR is active
  bool useXrMultiview = false;
  if(xrFrameActive && m_xr)
  {
    VkExtent2D perEye = m_xr->getPerEyeExtent();
    
    // Check if we can use multiview optimization
    if(m_xr->supportsMultiview() && m_graphicsPipelineGsMeshMultiview != VK_NULL_HANDLE)
    {
      useXrMultiview = true;
      
      // Initialize multiview resources if needed
      initXrMultiviewResources(cmd, perEye);
      
      // Get both eye data
      GsOpenXr::EyeData leftEye = m_xr->getEyeData(0);
      GsOpenXr::EyeData rightEye = m_xr->getEyeData(1);

      // Update IPD from OpenXR device
      float realIPD = glm::distance(leftEye.eyePos, rightEye.eyePos);
      if (realIPD > 0.001f)
      {
        m_stereoSeparation = realIPD;
      }

      m_eye = (leftEye.eyePos + rightEye.eyePos) * 0.5f;
      viewMatrix = leftEye.view;
      projMatrix = leftEye.proj;
      
      // Upload both eye matrices to UBO for multiview shader
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
      // Per-eye sensor pose for 3DGUT projection
      for(int eyeIdx = 0; eyeIdx < 2; ++eyeIdx)
      {
        glm::quat viewQuat         = glm::quat_cast(prmFrame.viewMatrixArray[eyeIdx]);
        prmFrame.viewQuatArray[eyeIdx]  = glm::vec4(viewQuat.x, viewQuat.y, viewQuat.z, viewQuat.w);
        prmFrame.viewTransArray[eyeIdx] = prmFrame.viewMatrixArray[eyeIdx][3];
      }
      prmFrame.multiviewEnabled = 1;
    }
    else
    {
      // Fallback to dual-pass rendering
      const float halfWidth = float(perEye.width);
      const uint32_t halfWidthInt = perEye.width;
      const uint32_t heightInt = perEye.height;

      for(uint32_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx)
      {
        GsOpenXr::EyeData eyeData = m_xr->getEyeData(eyeIdx);

        StereoView xrView;
        xrView.eye = eyeData.eyePos;
        xrView.view = eyeData.view;
        xrView.proj = eyeData.proj;
        xrView.stereoShift = glm::vec2(0.0f, 0.0f);

        float xOffset = float(eyeIdx * perEye.width);
        xrView.viewport = {xOffset, 0.0f, halfWidth, float(perEye.height), 0.0f, 1.0f};
        xrView.scissor = {{static_cast<int32_t>(eyeIdx * perEye.width), 0}, {halfWidthInt, heightInt}};

        views.push_back(xrView);
      }

      GsOpenXr::EyeData leftEye = m_xr->getEyeData(0);
      GsOpenXr::EyeData rightEye = m_xr->getEyeData(1);

      float realIPD = glm::distance(leftEye.eyePos, rightEye.eyePos);
      if (realIPD > 0.001f)
      {
        m_stereoSeparation = realIPD;
      }

      m_eye = (leftEye.eyePos + rightEye.eyePos) * 0.5f;
      viewMatrix = leftEye.view;
      projMatrix = leftEye.proj;
    }
  }
  else
#endif
  if(m_renderSBS)
  {
    const float     fovRad         = cameraManip->getRadFov();
    const float     halfAspect     = (float(m_viewSize.x) * 0.5f) / float(m_viewSize.y);
    const glm::vec2 clipPlanes     = cameraManip->getClipPlanes();
    const float     halfSeparation = m_stereoSeparation * 0.5f;

    const glm::vec3 rightDir = glm::vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]);

    const float    halfWidth    = float(m_viewSize.x) * 0.5f;
    const uint32_t halfWidthInt = static_cast<uint32_t>(halfWidth);
    const uint32_t heightInt    = static_cast<uint32_t>(m_viewSize.y);

    // Check if we should use off-axis projection
    // For 3DGUT, we use a different approach: symmetric projection matrix but shifted principal point
    const bool is3DGUT = (prmSelectedPipeline == PIPELINE_MESH_3DGUT) || (prmSelectedPipeline == PIPELINE_HYBRID_3DGUT);
    const bool useOffAxisMatrix = m_stereoOffAxisProj && !is3DGUT;

    // Pre-compute symmetric projection for 3DGUT modes or when off-axis is disabled
    glm::mat4 symmetricProj = glm::perspective(fovRad, halfAspect, clipPlanes.x, clipPlanes.y);
    symmetricProj[1][1] *= -1;

    // For 3DGUT off-axis, we need to compute the principal point shift in pixels
    // The formula is: shift = (eyeOffset * focalLength) / convergenceDistance
    // For vertical FOV and per-eye viewport height, focal length in pixels is:
    // f_pixels = (height / 2) / tan(fovY / 2)
    const float height            = float(m_viewSize.y);
    const float focalLengthPixels = (height * 0.5f) / tanf(fovRad * 0.5f);

    // Left eye
    StereoView left;
    left.eye  = m_eye - (rightDir * halfSeparation);
    left.view = glm::lookAt(left.eye, m_center, m_up);
    if(useOffAxisMatrix)
    {
      left.proj = makeOffAxisStereoProjection(fovRad, halfAspect, clipPlanes.x, clipPlanes.y, -halfSeparation, m_stereoConvergence);
      left.stereoShift = glm::vec2(0.0f, 0.0f);  // Off-axis is in the projection matrix
    }
    else
    {
      left.proj = symmetricProj;
      // For 3DGUT: compute principal point shift for off-axis stereo
      // Shift = (eyeOffset * focalLength) / convergenceDistance
      // Left eye has negative offset, so shift is negative (shifts left)
      if(is3DGUT && m_stereoOffAxisProj)
      {
        left.stereoShift = glm::vec2((-halfSeparation * focalLengthPixels) / m_stereoConvergence, 0.0f);
      }
      else
      {
        left.stereoShift = glm::vec2(0.0f, 0.0f);
      }
    }
    left.viewport = {0.0f, 0.0f, halfWidth, float(m_viewSize.y), 0.0f, 1.0f};
    left.scissor  = {{0, 0}, {halfWidthInt, heightInt}};
    views.push_back(left);

    // Right eye
    StereoView right;
    right.eye  = m_eye + (rightDir * halfSeparation);
    right.view = glm::lookAt(right.eye, m_center, m_up);
    if(useOffAxisMatrix)
    {
      right.proj = makeOffAxisStereoProjection(fovRad, halfAspect, clipPlanes.x, clipPlanes.y, halfSeparation, m_stereoConvergence);
      right.stereoShift = glm::vec2(0.0f, 0.0f);  // Off-axis is in the projection matrix
    }
    else
    {
      right.proj = symmetricProj;
      // For 3DGUT: compute principal point shift for off-axis stereo
      // Right eye has positive offset, so shift is positive (shifts right)
      if(is3DGUT && m_stereoOffAxisProj)
      {
        right.stereoShift = glm::vec2((halfSeparation * focalLengthPixels) / m_stereoConvergence, 0.0f);
      }
      else
      {
        right.stereoShift = glm::vec2(0.0f, 0.0f);
      }
    }
    right.viewport = {halfWidth, 0.0f, halfWidth, float(m_viewSize.y), 0.0f, 1.0f};
    right.scissor  = {{static_cast<int32_t>(halfWidth), 0}, {halfWidthInt, heightInt}};
    views.push_back(right);
  }
  else
  {
    StereoView mono;
    mono.view        = viewMatrix;
    mono.proj        = projMatrix;
    mono.eye         = m_eye;
    mono.stereoShift = glm::vec2(0.0f, 0.0f);  // No stereo shift for mono
    mono.viewport    = {0.0f, 0.0f, float(m_viewSize.x), float(m_viewSize.y), 0.0f, 1.0f};
    mono.scissor     = {{0, 0}, {static_cast<uint32_t>(m_viewSize.x), static_cast<uint32_t>(m_viewSize.y)}};
    views.push_back(mono);
  }

  // Handle device-host data update and splat sorting if a scene exist
  if(m_shaders.valid && splatCount)
  {
    collectReadBackValuesIfNeeded();

#ifdef WITH_OPENXR
    if (useXrMultiview && m_xr) {
        VkExtent2D perEye = m_xr->getPerEyeExtent();
        // Use XR viewport size for sorting/culling precision
        updateAndUploadFrameInfoUBO(cmd, splatCount, viewMatrix, projMatrix, m_eye, glm::vec2(perEye.width, perEye.height));
    } else {
        // Sort based on Center Eye (Mono or stereo center)
        updateAndUploadFrameInfoUBO(cmd, splatCount, viewMatrix, projMatrix, m_eye, glm::vec2(m_viewSize.x, m_viewSize.y));
    }
#else
    // Sort based on Center Eye (Mono or stereo center)
    updateAndUploadFrameInfoUBO(cmd, splatCount, viewMatrix, projMatrix, m_eye, glm::vec2(m_viewSize.x, m_viewSize.y));
#endif

    if(prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX)

    {
      m_profilerTimeline->asyncRemoveTimer("CPU Dist");
      m_profilerTimeline->asyncRemoveTimer("CPU Sort");
      processSortingOnGPU(cmd, splatCount);
    }
    else
    {
      tryConsumeAndUploadCpuSortingResult(cmd, splatCount);
    }
  }

  // In which color buffer are we going to render ?
  uint32_t colorBufferId = COLOR_MAIN;
  if(prmRtx.temporalSampling && prmFrame.frameSampleId > 0)
    colorBufferId = COLOR_AUX1;

  // raytrace the mesh depth using primary rays if needed
  bool raytraceMeshDepth = m_shaders.valid && !m_meshSetVk.instances.empty() && prmSelectedPipeline == PIPELINE_HYBRID_3DGUT;

  nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getDepthImage(),
                                    VK_IMAGE_LAYOUT_UNDEFINED,
                                    VK_IMAGE_LAYOUT_GENERAL,
                                    {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}});

  // Clear the full framebuffer once before the view loop (for SBS stereo or mono)
  // This ensures both eye regions are properly cleared
  {
    VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
    colorAttachment.imageView                 = m_gBuffers.getColorImageView(colorBufferId);
    colorAttachment.loadOp                    = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.clearValue                = {m_clearColor};

    VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
    depthAttachment.imageView                 = m_gBuffers.getDepthImageView();
    depthAttachment.loadOp                    = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.clearValue                = {.depthStencil = DEFAULT_VkClearDepthStencilValue};

    VkRenderingInfo clearRenderingInfo      = DEFAULT_VkRenderingInfo;
    clearRenderingInfo.renderArea           = {{0, 0}, {static_cast<uint32_t>(m_viewSize.x), static_cast<uint32_t>(m_viewSize.y)}};
    clearRenderingInfo.colorAttachmentCount = 1;
    clearRenderingInfo.pColorAttachments    = &colorAttachment;
    clearRenderingInfo.pDepthAttachment     = &depthAttachment;

    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(colorBufferId), VK_IMAGE_LAYOUT_GENERAL,
                                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getDepthImage(),
                                      VK_IMAGE_LAYOUT_GENERAL,
                                      VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                      {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}});

    vkCmdBeginRendering(cmd, &clearRenderingInfo);
    vkCmdEndRendering(cmd);

    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(colorBufferId),
                                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});
    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getDepthImage(),
                                      VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                      VK_IMAGE_LAYOUT_GENERAL,
                                      {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}});
  }

#ifdef WITH_OPENXR
  // Multiview rendering path (single draw call for both eyes)
  if(useXrMultiview && m_xrMultiviewInitialized && m_shaders.valid && splatCount)
  {
    // Upload UBO with multiview matrices
    vkCmdUpdateBuffer(cmd, m_frameInfoBuffer.buffer, 0, sizeof(shaderio::FrameInfo), &prmFrame);
    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT
                             | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         0, 1, &barrier, 0, NULL, 0, NULL);

    // Render with multiview (single dispatch for both eyes)
    renderMultiviewRaster(cmd, splatCount);

    // Copy multiview result to XR swapchain
    // Each layer goes to the corresponding half of the SBS swapchain
    VkExtent2D perEye = m_xr->getPerEyeExtent();
    
    // Transition XR swapchain to transfer dst
    nvvk::cmdImageMemoryBarrier(cmd, {m_xrColorImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL});

    // Copy both layers to the SBS swapchain
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

    readBackIndirectParametersIfNeeded(cmd);
    updateRenderingMemoryStatistics(cmd, splatCount);

    m_xr->releaseSwapchainImages();
    m_xr->endFrame();
    return;
  }
#endif

  // RENDER LOOP FOR VIEWS (dual-pass fallback)
  for(size_t viewIndex = 0; viewIndex < views.size(); ++viewIndex)
  {
    const auto& view = views[viewIndex];

    prmFrame.multiviewEnabled = 0;
    updateAndUploadFrameInfoUBO(cmd, splatCount, view.view, view.proj, view.eye,

                                {view.viewport.width, view.viewport.height},
                                {view.viewport.x, view.viewport.y},
                                view.stereoShift);

    if(raytraceMeshDepth)
    {
      raytrace(cmd, true, glm::ivec2(view.viewport.x, view.viewport.y),
               glm::ivec2(view.viewport.width, view.viewport.height));
    }

    // Drawing the primitives in the G-Buffer
    {
      auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Rasterization");

      VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
      colorAttachment.imageView                 = m_gBuffers.getColorImageView(colorBufferId);
      colorAttachment.loadOp                    = VK_ATTACHMENT_LOAD_OP_LOAD;

      VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
      depthAttachment.imageView                 = m_gBuffers.getDepthImageView();
      depthAttachment.loadOp                    = VK_ATTACHMENT_LOAD_OP_LOAD;

      VkRenderingInfo renderingInfo      = DEFAULT_VkRenderingInfo;
      renderingInfo.renderArea           = {{0, 0}, {static_cast<uint32_t>(m_viewSize.x), static_cast<uint32_t>(m_viewSize.y)}};
      renderingInfo.colorAttachmentCount = 1;
      renderingInfo.pColorAttachments    = &colorAttachment;
      renderingInfo.pDepthAttachment     = &depthAttachment;

      nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(colorBufferId), VK_IMAGE_LAYOUT_GENERAL,
                                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

      nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getDepthImage(),
                                        VK_IMAGE_LAYOUT_GENERAL,
                                        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                        {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}});

      vkCmdBeginRendering(cmd, &renderingInfo);

      vkCmdSetViewportWithCount(cmd, 1, &view.viewport);
      vkCmdSetScissorWithCount(cmd, 1, &view.scissor);

      if(m_shaders.valid && !m_meshSetVk.instances.empty() && !raytraceMeshDepth)
      {
        drawMeshPrimitives(cmd);
      }

      if(m_shaders.valid && splatCount)
      {
        drawSplatPrimitives(cmd, splatCount);
      }

      // Update depth/video textures before rendering VDZ mesh
      if(prmFrame.visualize == VISUALIZE_VDZ_MESH && m_enableDepthRendering)
      {
        updateDepthRendering(cmd);
      }

      // Render VDZ depth mesh if in VDZ mesh visualization mode
      if(prmFrame.visualize == VISUALIZE_VDZ_MESH)
      {
        drawVdzMesh(cmd);
      }

      vkCmdEndRendering(cmd);

      nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(colorBufferId),
                                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});
      nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getDepthImage(),
                                        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                        VK_IMAGE_LAYOUT_GENERAL,
                                        {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}});
    }

    // raytrace the secondary rays if needed
    if(m_shaders.valid && splatCount && m_splatSetVk.rtxValid && !m_meshSetVk.instances.empty()
       && (prmSelectedPipeline == PIPELINE_HYBRID || prmSelectedPipeline == PIPELINE_HYBRID_3DGUT))
    {
      raytrace(cmd, false, glm::ivec2(view.viewport.x, view.viewport.y),
               glm::ivec2(view.viewport.width, view.viewport.height));
    }
  }  // End View Loop

  // Perform post processings if needed
  // Run post-process for temporal accumulation or linear-to-sRGB conversion
  if((prmRtx.temporalSampling && prmFrame.frameSampleId > 0) || prmFrame.linearToSrgb != 0)
  {
    postProcess(cmd);
  }

  // Update depth streaming if enabled
  // This will request new frames from the backend and upload received ones
  updateDepthRendering(cmd);

  readBackIndirectParametersIfNeeded(cmd);

  updateRenderingMemoryStatistics(cmd, splatCount);


#ifdef WITH_OPENXR
  // Finalize XR frame
  if(xrFrameActive && m_xr)
  {
    // Copy our rendered GBuffer to the XR swapchain
    copyToXrSwapchain(cmd);

    // Release swapchain images and end the frame
    m_xr->releaseSwapchainImages();
    m_xr->endFrame();
  }
#endif
}

void GaussianSplatting::processUpdateRequests(void)
{

  // Automatic and Sanity settings depending in pipeline
  if(prmSelectedPipeline != PIPELINE_RTX && prmSelectedPipeline != PIPELINE_HYBRID_3DGUT && prmSelectedPipeline != PIPELINE_MESH_3DGUT)
  {
    prmRtx.temporalSampling = false;
    // prmRtx.dofEnabled       = false;
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

  if(!m_splatSet.size() || !needUpdate)
    return;

  resetFrameCounter();

  vkDeviceWaitIdle(m_device);

  // updates that requires update of descriptor sets
  if(m_requestUpdateSplatData || m_requestUpdateSplatAs || m_requestUpdateMeshData || m_requestUpdateShaders || m_requestDeleteSelectedMesh)
  {

    deinitPipelines();
    deinitShaders();

    if(m_requestUpdateSplatData)
    {
      m_splatSetVk.deinitDataStorage();
      m_splatSetVk.initDataStorage(m_splatSet, prmData.dataStorage, prmData.shFormat);
    }
    if(m_requestUpdateSplatData || m_requestUpdateSplatAs)
    {
      // RTX specific
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

    if(initShaders())
    {
      initPipelines();
      initRtDescriptorSet();
      initRtPipeline();
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

}  // namespace vk_gaussian_splatting
