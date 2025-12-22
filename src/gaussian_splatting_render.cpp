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

  // update buffers, rebuild shaders and pipelines if needed
  processUpdateRequests();

  // 0 if not ready so the rendering does not
  // touch the splat set while loading
  // getStatus is thread safe.
  uint32_t splatCount = 0;
  if(m_plyLoader.getStatus() == PlyLoaderAsync::State::E_READY)
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

    if(m_renderSBS)
    {
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
      updateAndUploadFrameInfoUBO(cmd, splatCount, viewMatrix, projMatrix, m_eye, glm::vec2(m_viewSize.x, m_viewSize.y));
      raytrace(cmd);
    }

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
    VkViewport viewport;
    VkRect2D   scissor;
  };

  std::vector<StereoView> views;
  views.reserve(m_renderSBS ? 2 : 1);

  cameraManip->getLookat(m_eye, m_center, m_up);
  glm::mat4 viewMatrix = cameraManip->getViewMatrix();
  glm::mat4 projMatrix = cameraManip->getPerspectiveMatrix();

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

    // For 3DGUT pipelines, use symmetric projection since they compute their own projection
    // and only use the projection matrix for depth (Z) calculation
    const bool useOffAxis = m_stereoOffAxisProj && (prmSelectedPipeline != PIPELINE_MESH_3DGUT) && (prmSelectedPipeline != PIPELINE_HYBRID_3DGUT);

    // Pre-compute symmetric projection for fallback
    glm::mat4 symmetricProj = glm::perspective(fovRad, halfAspect, clipPlanes.x, clipPlanes.y);
    symmetricProj[1][1] *= -1;

    // Left eye
    StereoView left;
    left.eye  = m_eye - (rightDir * halfSeparation);
    left.view = glm::lookAt(left.eye, m_center, m_up);
    if(useOffAxis)
    {
      left.proj = makeOffAxisStereoProjection(fovRad, halfAspect, clipPlanes.x, clipPlanes.y, -halfSeparation, m_stereoConvergence);
    }
    else
    {
      left.proj = symmetricProj;
    }
    left.viewport = {0.0f, 0.0f, halfWidth, float(m_viewSize.y), 0.0f, 1.0f};
    left.scissor  = {{0, 0}, {halfWidthInt, heightInt}};
    views.push_back(left);

    // Right eye
    StereoView right;
    right.eye  = m_eye + (rightDir * halfSeparation);
    right.view = glm::lookAt(right.eye, m_center, m_up);
    if(useOffAxis)
    {
      right.proj = makeOffAxisStereoProjection(fovRad, halfAspect, clipPlanes.x, clipPlanes.y, halfSeparation, m_stereoConvergence);
    }
    else
    {
      right.proj = symmetricProj;
    }
    right.viewport = {halfWidth, 0.0f, halfWidth, float(m_viewSize.y), 0.0f, 1.0f};
    right.scissor  = {{static_cast<int32_t>(halfWidth), 0}, {halfWidthInt, heightInt}};
    views.push_back(right);
  }
  else
  {
    StereoView mono;
    mono.view     = viewMatrix;
    mono.proj     = projMatrix;
    mono.eye      = m_eye;
    mono.viewport = {0.0f, 0.0f, float(m_viewSize.x), float(m_viewSize.y), 0.0f, 1.0f};
    mono.scissor  = {{0, 0}, {static_cast<uint32_t>(m_viewSize.x), static_cast<uint32_t>(m_viewSize.y)}};
    views.push_back(mono);
  }

  // Handle device-host data update and splat sorting if a scene exist
  if(m_shaders.valid && splatCount)
  {
    collectReadBackValuesIfNeeded();

    // Sort based on Center Eye (Mono or stereo center)
    updateAndUploadFrameInfoUBO(cmd, splatCount, viewMatrix, projMatrix, m_eye, glm::vec2(m_viewSize.x, m_viewSize.y));

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

  // RENDER LOOP FOR VIEWS
  for(size_t viewIndex = 0; viewIndex < views.size(); ++viewIndex)
  {
    const auto& view = views[viewIndex];

    updateAndUploadFrameInfoUBO(cmd, splatCount, view.view, view.proj, view.eye, {view.viewport.width, view.viewport.height});

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
  if(prmRtx.temporalSampling && prmFrame.frameSampleId > 0)
  {
    postProcess(cmd);
  }

  readBackIndirectParametersIfNeeded(cmd);

  updateRenderingMemoryStatistics(cmd, splatCount);
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


void GaussianSplatting::updateAndUploadFrameInfoUBO(VkCommandBuffer cmd, const uint32_t splatCount)
{
  NVVK_DBG_SCOPE(cmd);

  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "UBO update");

  Camera camera = m_cameraSet.getCamera();

  cameraManip->getLookat(m_eye, m_center, m_up);

  // Update frame parameters uniform buffer
  // some attributes of prmFrame were directly set by the user interface
  prmFrame.splatCount = splatCount;
  prmFrame.lightCount = int32_t(m_lightSet.size());

  prmFrame.cameraPosition = m_eye;
  prmFrame.viewMatrix     = cameraManip->getViewMatrix();
  prmFrame.viewInverse    = glm::inverse(prmFrame.viewMatrix);

  prmFrame.fovRad  = cameraManip->getRadFov();
  prmFrame.nearFar = cameraManip->getClipPlanes();
  // Projection matrix only viable in pinhole mode,
  // but is used as a fallback for 3DGS when Fisheye is on
  prmFrame.projectionMatrix = cameraManip->getPerspectiveMatrix();
  prmFrame.projInverse      = glm::inverse(prmFrame.projectionMatrix);

  float       devicePixelRatio     = 1.0;
  const bool  isOrthographicCamera = false;
  const float focalMultiplier      = isOrthographicCamera ? (1.0f / devicePixelRatio) : 1.0f;
  const float focalAdjustment      = focalMultiplier;
  prmFrame.orthoZoom               = 1.0f;
  prmFrame.orthographicMode        = 0;  // disabled (uses perspective) TODO: activate support for orthographic
  prmFrame.viewport                = glm::vec2(m_viewSize.x * devicePixelRatio, m_viewSize.y * devicePixelRatio);
  prmFrame.basisViewport           = glm::vec2(1.0f / m_viewSize.x, 1.0f / m_viewSize.y);
  prmFrame.inverseFocalAdjustment  = 1.0f / focalAdjustment;

  if(camera.model == CAMERA_FISHEYE && prmSelectedPipeline != PIPELINE_VERT && prmSelectedPipeline != PIPELINE_MESH
     && prmSelectedPipeline != PIPELINE_HYBRID)
  {
    // FISHEYE focal
    prmFrame.focal = glm::vec2(1.0, -1.0) * prmFrame.viewport / prmFrame.fovRad;
  }
  else
  {
    // PIHNOLE focal
    const float focalLengthX = prmFrame.projectionMatrix[0][0] * 0.5f * devicePixelRatio * m_viewSize.x;
    const float focalLengthY = prmFrame.projectionMatrix[1][1] * 0.5f * devicePixelRatio * m_viewSize.y;
    prmFrame.focal           = glm::vec2(focalLengthX, focalLengthY);
  }

  // Camera pose, used by unscented transform
  {
    prmFrame.viewTrans = prmFrame.viewMatrix[3];
    glm::quat viewQuat = glm::quat_cast(prmFrame.viewMatrix);
    // glm quaternion storage is scalar last, so we forward as is
    prmFrame.viewQuat = glm::vec4(viewQuat.x, viewQuat.y, viewQuat.z, viewQuat.w);
  }

  prmFrame.focusDist = camera.focusDist;
  prmFrame.aperture  = camera.aperture;

  // the buffer is small so we use vkCmdUpdateBuffer for the transfer
  vkCmdUpdateBuffer(cmd, m_frameInfoBuffer.buffer, 0, sizeof(shaderio::FrameInfo), &prmFrame);

  // sync with end of copy to device
  VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;

  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT
                           | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
                       0, 1, &barrier, 0, NULL, 0, NULL);
}

void GaussianSplatting::updateAndUploadFrameInfoUBO(VkCommandBuffer  cmd,
                                                    const uint32_t   splatCount,
                                                    const glm::mat4& view,
                                                    const glm::mat4& proj,
                                                    const glm::vec3& eye,
                                                    const glm::vec2& viewport)
{
  if(m_frameInfoBuffer.buffer == VK_NULL_HANDLE)
    return;

  NVVK_DBG_SCOPE(cmd);

  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "UBO update");

  Camera camera = m_cameraSet.getCamera();

  prmFrame.splatCount = splatCount;
  prmFrame.lightCount = int32_t(m_lightSet.size());

  prmFrame.cameraPosition = eye;
  prmFrame.viewMatrix     = view;
  prmFrame.viewInverse    = glm::inverse(prmFrame.viewMatrix);

  prmFrame.fovRad           = cameraManip->getRadFov();
  prmFrame.nearFar          = cameraManip->getClipPlanes();
  prmFrame.projectionMatrix = proj;
  prmFrame.projInverse      = glm::inverse(prmFrame.projectionMatrix);

  float       devicePixelRatio     = 1.0;
  const bool  isOrthographicCamera = false;
  const float focalMultiplier      = isOrthographicCamera ? (1.0f / devicePixelRatio) : 1.0f;
  const float focalAdjustment      = focalMultiplier;
  prmFrame.orthoZoom               = 1.0f;
  prmFrame.orthographicMode        = 0;
  prmFrame.viewport                = viewport;
  prmFrame.basisViewport           = glm::vec2(1.0f / viewport.x, 1.0f / viewport.y);
  prmFrame.inverseFocalAdjustment  = 1.0f / focalAdjustment;

  if(camera.model == CAMERA_FISHEYE && prmSelectedPipeline != PIPELINE_VERT && prmSelectedPipeline != PIPELINE_MESH
     && prmSelectedPipeline != PIPELINE_HYBRID)
  {
    prmFrame.focal = glm::vec2(1.0, -1.0) * prmFrame.viewport / prmFrame.fovRad;
  }
  else
  {
    const float focalLengthX = prmFrame.projectionMatrix[0][0] * 0.5f * devicePixelRatio * viewport.x;
    const float focalLengthY = prmFrame.projectionMatrix[1][1] * 0.5f * devicePixelRatio * viewport.y;
    prmFrame.focal           = glm::vec2(focalLengthX, focalLengthY);
  }

  {
    prmFrame.viewTrans = prmFrame.viewMatrix[3];
    glm::quat viewQuat = glm::quat_cast(prmFrame.viewMatrix);
    prmFrame.viewQuat  = glm::vec4(viewQuat.x, viewQuat.y, viewQuat.z, viewQuat.w);
  }

  prmFrame.focusDist = camera.focusDist;
  prmFrame.aperture  = camera.aperture;

  vkCmdUpdateBuffer(cmd, m_frameInfoBuffer.buffer, 0, sizeof(shaderio::FrameInfo), &prmFrame);

  VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;

  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT
                           | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
                       0, 1, &barrier, 0, NULL, 0, NULL);
}

void GaussianSplatting::tryConsumeAndUploadCpuSortingResult(VkCommandBuffer cmd, const uint32_t splatCount)
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

void GaussianSplatting::processSortingOnGPU(VkCommandBuffer cmd, const uint32_t splatCount)
{
  NVVK_DBG_SCOPE(cmd);

  // when GPU sorting, we sort at each frame, all buffer in device memory, no copy from RAM

  // 1. reset the draw indirect parameters and counters, will be updated by compute shader
  {
    const shaderio::IndirectParams drawIndexedIndirectParams;
    vkCmdUpdateBuffer(cmd, m_indirect.buffer, 0, sizeof(shaderio::IndirectParams), (void*)&drawIndexedIndirectParams);

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
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);

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

    vrdxCmdSortKeyValueIndirect(cmd, m_gpuSorter, splatCount, m_indirect.buffer,
                                offsetof(shaderio::IndirectParams, instanceCount), m_splatDistancesDevice.buffer, 0,
                                m_splatIndicesDevice.buffer, 0, m_vrdxStorageDevice.buffer, 0, 0, 0);

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                         0, 1, &barrier, 0, NULL, 0, NULL);
  }
}

void GaussianSplatting::drawSplatPrimitives(VkCommandBuffer cmd, const uint32_t splatCount)
{
  NVVK_DBG_SCOPE(cmd);

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
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);

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
      vkCmdDrawIndexedIndirect(cmd, m_indirect.buffer, 0, 1, sizeof(VkDrawIndexedIndirectCommand));
    }
  }
  else
  {  // in mesh pipeline mode or in hybrid mode
    // Pipeline using mesh shader

    if(prmSelectedPipeline == PIPELINE_MESH || prmSelectedPipeline == PIPELINE_HYBRID)
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipelineGsMesh);
    if(prmSelectedPipeline == PIPELINE_MESH_3DGUT || prmSelectedPipeline == PIPELINE_HYBRID_3DGUT)
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline3dgutMesh);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);

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
      vkCmdDrawMeshTasksIndirectEXT(cmd, m_indirect.buffer, offsetof(shaderio::IndirectParams, groupCountX), 1,
                                    sizeof(VkDrawMeshTasksIndirectCommandEXT));
    }
  }
}

void GaussianSplatting::drawMeshPrimitives(VkCommandBuffer cmd)
{

  NVVK_DBG_SCOPE(cmd);

  VkDeviceSize offset{0};

  // Drawing all triangles
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipelineMesh);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);
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

void GaussianSplatting::collectReadBackValuesIfNeeded(void)
{
  if(m_indirectReadbackHost.buffer != VK_NULL_HANDLE && prmRaster.sortingMethod == SORTING_GPU_SYNC_RADIX && m_canCollectReadback)
  {
    std::memcpy((void*)&m_indirectReadback, (void*)m_indirectReadbackHost.mapping, sizeof(shaderio::IndirectParams));
  }
}

void GaussianSplatting::readBackIndirectParametersIfNeeded(VkCommandBuffer cmd)
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
    VkBufferCopy bc{.srcOffset = 0, .dstOffset = 0, .size = sizeof(shaderio::IndirectParams)};
    vkCmdCopyBuffer(cmd, m_indirect.buffer, m_indirectReadbackHost.buffer, 1, &bc);

    m_canCollectReadback = true;
  }
}

void GaussianSplatting::updateRenderingMemoryStatistics(VkCommandBuffer cmd, const uint32_t splatCount)
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

}  // namespace vk_gaussian_splatting
