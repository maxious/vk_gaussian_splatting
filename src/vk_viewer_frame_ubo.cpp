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
// Contains: Frame info UBO update functions

namespace vk_viewer {

void VkViewer::updateAndUploadFrameInfoUBO(VkCommandBuffer cmd, const uint32_t splatCount)
{
  NVVK_DBG_SCOPE(cmd);

  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "UBO update");

  Camera camera = m_cameraSet.getCamera();

  cameraManip->getLookat(m_eye, m_center, m_up);

  // Update frame parameters uniform buffer
  // some attributes of prmFrame were directly set by the user interface
  prmFrame.splatCount = splatCount;
  prmFrame.lightCount = int32_t(m_lightSet.size());

  prmFrame.cameraPosition = glm::vec3(m_eye);
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
  // Guard against division by zero if viewSize is uninitialized
  prmFrame.basisViewport = glm::vec2(m_viewSize.x > 0 ? 1.0f / m_viewSize.x : 1.0f, m_viewSize.y > 0 ? 1.0f / m_viewSize.y : 1.0f);
  prmFrame.viewportOffset         = glm::vec2(0.0f, 0.0f);  // No offset for mono rendering
  prmFrame.stereoShift            = glm::vec2(0.0f, 0.0f);  // No stereo shift for mono rendering
  prmFrame.inverseFocalAdjustment = 1.0f / focalAdjustment;

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

  // prmFrame.multiviewEnabled is now managed by the caller (onRender)
  // Do not reset it here, as it might have been set for OpenXR

  // LCC packed storage: set scale min/max for GPU-side decompression
  if(prmData.dataStorage == STORAGE_LCC_PACKED && isLccStreamingActive())
  {
    // Use per-component scale ranges for accurate GPU decompression
    prmFrame.lccScaleMin = m_lccMeta.scaleVec3.min;
    prmFrame.lccScaleMax = m_lccMeta.scaleVec3.max;
  }

  // Chunk-based hierarchical frustum culling
  prmFrame.chunkCullingEnabled = prmRaster.chunkCullingEnabled ? 1 : 0;
  prmFrame.numChunks           = m_numChunks;

  // Extract frustum planes from viewProj matrix for GPU culling
  if(prmRaster.chunkCullingEnabled && m_numChunks > 0)
  {
    glm::mat4 viewProj = prmFrame.projectionMatrix * prmFrame.viewMatrix;
    extractFrustumPlanes(viewProj);
  }

  for(int eyeIdx = 0; eyeIdx < 2; ++eyeIdx)
  {
    const glm::mat4& mainProjInv = (prmFrame.multiviewEnabled != 0) ? prmFrame.projInverseArray[eyeIdx] : prmFrame.projInverse;
    const glm::mat4& mainViewInv = (prmFrame.multiviewEnabled != 0) ? prmFrame.viewInverseArray[eyeIdx] : prmFrame.viewInverse;
    prmFrame.envDepthMainClipToEnvClipArray[eyeIdx] =
        mainProjInv * mainViewInv * prmFrame.envDepthViewMatrixArray[eyeIdx] * prmFrame.envDepthProjectionMatrixArray[eyeIdx];
  }

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

void VkViewer::updateAndUploadFrameInfoUBO(VkCommandBuffer  cmd,
                                           const uint32_t   splatCount,
                                           const glm::mat4& view,
                                           const glm::mat4& proj,
                                           const glm::vec3& eye,
                                           const glm::vec2& viewport,
                                           const glm::vec2& viewportOffset,
                                           const glm::vec2& stereoShift)
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
  prmFrame.viewportOffset          = viewportOffset;
  prmFrame.stereoShift             = stereoShift;
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

  // prmFrame.multiviewEnabled is now managed by the caller (onRender)
  // Do not reset it here, as it might have been set for OpenXR

  // Chunk-based hierarchical frustum culling
  prmFrame.chunkCullingEnabled = prmRaster.chunkCullingEnabled ? 1 : 0;
  prmFrame.numChunks           = m_numChunks;

  // Extract frustum planes from viewProj matrix for GPU culling
  if(prmRaster.chunkCullingEnabled && m_numChunks > 0)
  {
    glm::mat4 viewProj = proj * view;
    extractFrustumPlanes(viewProj);
  }

  for(int eyeIdx = 0; eyeIdx < 2; ++eyeIdx)
  {
    const glm::mat4& mainProjInv = (prmFrame.multiviewEnabled != 0) ? prmFrame.projInverseArray[eyeIdx] : prmFrame.projInverse;
    const glm::mat4& mainViewInv = (prmFrame.multiviewEnabled != 0) ? prmFrame.viewInverseArray[eyeIdx] : prmFrame.viewInverse;
    prmFrame.envDepthMainClipToEnvClipArray[eyeIdx] =
        mainProjInv * mainViewInv * prmFrame.envDepthViewMatrixArray[eyeIdx] * prmFrame.envDepthProjectionMatrixArray[eyeIdx];
  }

#ifdef WITH_DLSS_RR

  // Store previous frame matrices for motion vector calculation
  static glm::mat4 s_prevViewMatrix = view;
  static glm::mat4 s_prevProjMatrix = proj;
  prmFrame.prevViewMatrix           = s_prevViewMatrix;
  prmFrame.prevProjectionMatrix     = s_prevProjMatrix;
  s_prevViewMatrix                  = view;
  s_prevProjMatrix                  = proj;

  // Compute DLSS jitter for temporal anti-aliasing
  if(m_dlssRREnabled && m_dlssRRInitialized)
  {
    // Halton sequence for temporal jitter
    auto halton = [](int index, int base) -> float {
      float result = 0.0f;
      float f      = 1.0f / float(base);
      int   i      = index;
      while(i > 0)
      {
        result += f * float(i % base);
        i = i / base;
        f = f / float(base);
      }
      return result;
    };

    const int   jitterIndex = m_dlssRRFrameIndex % 8;
    const float jitterX     = halton(jitterIndex + 1, 2) - 0.5f;
    const float jitterY     = halton(jitterIndex + 1, 3) - 0.5f;
    prmFrame.dlssJitter     = glm::vec2(jitterX, jitterY);
  }
  else
  {
    prmFrame.dlssJitter = glm::vec2(0.0f, 0.0f);
  }
#endif

  // Use dynamic buffering for FrameInfo
  uint32_t dynamicOffset = m_currentFrameInfoOffset;

  vkCmdUpdateBuffer(cmd, m_frameInfoBuffer.buffer, dynamicOffset, sizeof(shaderio::FrameInfo), &prmFrame);

  // Store the offset we just used, so subsequent draw calls bind the correct data
  m_lastFrameInfoOffset = dynamicOffset;

  // Advance the offset for the next update
  m_currentFrameInfoOffset += (uint32_t)m_frameInfoStride;

  VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};

  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;

  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT
                           | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
                       0, 1, &barrier, 0, NULL, 0, NULL);
}

}  // namespace vk_viewer
