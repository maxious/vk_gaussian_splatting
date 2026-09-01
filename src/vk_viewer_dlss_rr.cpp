/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// This file is included from gaussian_splatting.cpp - do not compile separately

#ifdef WITH_DLSS_RR

namespace vk_viewer {

void VkViewer::initializeDlssRR()
{
  if(m_dlssRRInitialized)
    return;

  // Check if DLSS-RR is available on this device
  NVSDK_NGX_Result result = NgxContext::isDlssRRAvailable(m_app->getInstance(), m_app->getPhysicalDevice());
  if(NVSDK_NGX_FAILED(result))
  {
    LOGW("DLSS-RR is not available on this device\n");
    m_dlssRREnabled = false;
    return;
  }

  // Create NGX context
  m_ngxContext = std::make_unique<NgxContext>();

  NgxContext::InitInfo initInfo = {};
  initInfo.instance             = m_app->getInstance();
  initInfo.physicalDevice       = m_app->getPhysicalDevice();
  initInfo.device               = m_device;
  initInfo.queue                = m_app->getQueue(0).queue;
  initInfo.queueFamilyIdx       = m_app->getQueue(0).familyIndex;
  initInfo.applicationPath      = nvutils::getExecutablePath().parent_path();

  result = m_ngxContext->init(initInfo);
  if(NVSDK_NGX_FAILED(result))
  {
    LOGE("Failed to initialize NGX context: %s\n", getNGXResultString(result).c_str());
    m_ngxContext.reset();
    m_dlssRREnabled = false;
    return;
  }

  // Create DLSS-RR instance
  m_dlssRR = std::make_unique<GsDlssRR>();

  VkExtent2D renderSize = {static_cast<uint32_t>(m_viewSize.x), static_cast<uint32_t>(m_viewSize.y)};
  
  NgxContext::DlssRRInitInfo dlssInitInfo = {};
  dlssInitInfo.inputSize  = renderSize;
  dlssInitInfo.outputSize = renderSize;  // Same size for now (no upscaling)
  dlssInitInfo.quality    = m_dlssRRQuality;
  dlssInitInfo.preset     = NVSDK_NGX_RayReconstruction_Hint_Render_Preset_Default;

  result = m_ngxContext->initDlssRR(dlssInitInfo, *m_dlssRR);
  if(NVSDK_NGX_FAILED(result))
  {
    LOGE("Failed to initialize DLSS-RR: %s\n", getNGXResultString(result).c_str());
    m_dlssRR.reset();
    m_ngxContext.reset();
    m_dlssRREnabled = false;
    return;
  }

  m_dlssRRInitialized = true;
  m_dlssRRNeedsReset  = true;
  m_dlssRRFrameIndex  = 0;

  LOGI("DLSS-RR initialized successfully\n");
}

void VkViewer::shutdownDlssRR()
{
  if(m_dlssRR)
  {
    m_dlssRR->deinit();
    m_dlssRR.reset();
  }

  if(m_ngxContext && !m_dlssInitialized)
  {
    m_ngxContext->deinit();
    m_ngxContext.reset();
  }

  m_dlssRRInitialized = false;
  m_dlssRRFrameIndex  = 0;
}

void VkViewer::initializeDlss()
{
  if(m_dlssInitialized)
    return;

  if(!m_ngxContext)
  {
    m_ngxContext = std::make_unique<NgxContext>();
    NgxContext::InitInfo initInfo = {};
    initInfo.instance        = m_app->getInstance();
    initInfo.physicalDevice  = m_app->getPhysicalDevice();
    initInfo.device          = m_device;
    initInfo.queue           = m_app->getQueue(0).queue;
    initInfo.queueFamilyIdx  = m_app->getQueue(0).familyIndex;
    initInfo.applicationPath = nvutils::getExecutablePath().parent_path();

    NVSDK_NGX_Result result = m_ngxContext->init(initInfo);
    if(NVSDK_NGX_FAILED(result))
    {
      LOGE("Failed to initialize NGX context for DLSS: %s\n", getNGXResultString(result).c_str());
      m_ngxContext.reset();
      return;
    }
  }

  m_dlss = std::make_unique<GsDlss>();
  NgxContext::DlssInitInfo initInfo = {};
  initInfo.inputSize  = {static_cast<uint32_t>(m_viewSize.x), static_cast<uint32_t>(m_viewSize.y)};
  initInfo.outputSize = initInfo.inputSize;
  initInfo.quality    = NVSDK_NGX_PerfQuality_Value_DLAA;

  const NVSDK_NGX_Result result = m_ngxContext->initDlss(initInfo, *m_dlss);
  if(NVSDK_NGX_FAILED(result))
  {
    LOGE("Failed to initialize DLSS Super Resolution: %s\n", getNGXResultString(result).c_str());
    m_dlss.reset();
    if(!m_dlssRRInitialized)
      m_ngxContext.reset();
    return;
  }

  m_dlssInitialized  = true;
  m_dlssRRNeedsReset  = true;
  m_dlssRRFrameIndex  = 0;
  LOGI("DLSS Super Resolution initialized in DLAA mode\n");
}

void VkViewer::shutdownDlss()
{
  if(m_dlss)
  {
    m_dlss->deinit();
    m_dlss.reset();
  }

  m_dlssInitialized = false;
  if(m_ngxContext && !m_dlssRRInitialized)
  {
    m_ngxContext->deinit();
    m_ngxContext.reset();
  }
}

void VkViewer::updateDlssDescriptorSet()
{
  if(!m_dlssInitialized || !m_dlss || !m_dlss->isValid())
    return;

  m_dlss->setResource(GsDlss::RESOURCE_COLOR_IN, m_gBuffers.getColorImage(COLOR_MAIN),
                      m_gBuffers.getColorImageView(COLOR_MAIN), m_colorFormat);
  m_dlss->setResource(GsDlss::RESOURCE_COLOR_OUT, m_gBuffers.getColorImage(COLOR_DLSS_OUTPUT),
                      m_gBuffers.getColorImageView(COLOR_DLSS_OUTPUT), m_colorFormat);
  m_dlss->setResource(GsDlss::RESOURCE_MOTION_VECTORS, m_gBuffers.getColorImage(COLOR_MOTION),
                      m_gBuffers.getColorImageView(COLOR_MOTION), VK_FORMAT_R16G16_SFLOAT);
  m_dlss->setResource(GsDlss::RESOURCE_DEPTH, m_gBuffers.getColorImage(COLOR_DLSS_LINEAR_DEPTH),
                      m_gBuffers.getColorImageView(COLOR_DLSS_LINEAR_DEPTH), VK_FORMAT_R32_SFLOAT);
}

void VkViewer::updateDlssRRDescriptorSet()
{
  if(!m_dlssRRInitialized || !m_dlssRR || !m_dlssRR->isValid())
    return;

  // Bind DLSS-RR input resources
  // Input color (noisy HDR from ray tracing)
  m_dlssRR->setResource(GsDlssRR::RESOURCE_COLOR_IN, 
                        m_gBuffers.getColorImage(COLOR_MAIN),
                        m_gBuffers.getColorImageView(COLOR_MAIN), 
                        m_colorFormat);

  // Output color (denoised)
  m_dlssRR->setResource(GsDlssRR::RESOURCE_COLOR_OUT, 
                        m_gBuffers.getColorImage(COLOR_DLSS_OUTPUT),
                        m_gBuffers.getColorImageView(COLOR_DLSS_OUTPUT), 
                        m_colorFormat);

  // Diffuse albedo
  m_dlssRR->setResource(GsDlssRR::RESOURCE_DIFFUSE_ALBEDO, 
                        m_gBuffers.getColorImage(COLOR_DLSS_DIFFUSE_ALBEDO),
                        m_gBuffers.getColorImageView(COLOR_DLSS_DIFFUSE_ALBEDO), 
                        VK_FORMAT_R16G16B16A16_SFLOAT);

  // Specular albedo
  m_dlssRR->setResource(GsDlssRR::RESOURCE_SPECULAR_ALBEDO, 
                        m_gBuffers.getColorImage(COLOR_DLSS_SPECULAR_ALBEDO),
                        m_gBuffers.getColorImageView(COLOR_DLSS_SPECULAR_ALBEDO), 
                        VK_FORMAT_R16G16B16A16_SFLOAT);

  // Normal + roughness
  m_dlssRR->setResource(GsDlssRR::RESOURCE_NORMAL_ROUGHNESS, 
                        m_gBuffers.getColorImage(COLOR_DLSS_NORMAL_ROUGH),
                        m_gBuffers.getColorImageView(COLOR_DLSS_NORMAL_ROUGH), 
                        VK_FORMAT_R16G16B16A16_SFLOAT);

  // Motion vectors
  m_dlssRR->setResource(GsDlssRR::RESOURCE_MOTION_VECTORS, 
                        m_gBuffers.getColorImage(COLOR_DLSS_MOTION),
                        m_gBuffers.getColorImageView(COLOR_DLSS_MOTION), 
                        VK_FORMAT_R16G16_SFLOAT);

  // Linear depth
  m_dlssRR->setResource(GsDlssRR::RESOURCE_LINEAR_DEPTH, 
                        m_gBuffers.getColorImage(COLOR_DLSS_LINEAR_DEPTH),
                        m_gBuffers.getColorImageView(COLOR_DLSS_LINEAR_DEPTH), 
                        VK_FORMAT_R32_SFLOAT);

  // Specular hit distance (optional)
  m_dlssRR->setResource(GsDlssRR::RESOURCE_SPECULAR_HIT_DIST, 
                        m_gBuffers.getColorImage(COLOR_DLSS_SPEC_HIT_DIST),
                        m_gBuffers.getColorImageView(COLOR_DLSS_SPEC_HIT_DIST), 
                        VK_FORMAT_R16_SFLOAT);
}

}  // namespace vk_viewer

#endif  // WITH_DLSS_RR
