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

#include "gs_dlss_rr.hpp"

#ifdef WITH_DLSS_RR

#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_helpers_dlssd_vk.h>
#include <nvsdk_ngx_helpers_dlssd.h>

#include <nvvk/commands.hpp>
#include <nvvk/debug_util.hpp>
#include <nvutils/logger.hpp>

#include <glm/gtc/type_ptr.hpp>

namespace vk_viewer {

// Application ID for NGX (arbitrary unique ID)
static constexpr uint64_t g_ApplicationID = 0x47535F444C535352;  // "GS_DLSSR"

// Macro for checking NGX results
#define NGX_CHECK(expr)                                                              \
  [&]() {                                                                            \
    NVSDK_NGX_Result _result = (expr);                                               \
    if(NVSDK_NGX_FAILED(_result))                                                    \
    {                                                                                \
      LOGE("NGX Error: %s at %s:%d\n", getNGXResultString(_result).c_str(),          \
           __FILE__, __LINE__);                                                      \
    }                                                                                \
    return _result;                                                                  \
  }()

#define NGX_RETURN_ON_FAIL(expr)                                                     \
  do {                                                                               \
    NVSDK_NGX_Result _r = NGX_CHECK(expr);                                           \
    if(NVSDK_NGX_FAILED(_r)) return _r;                                              \
  } while(0)

std::string getNGXResultString(NVSDK_NGX_Result result)
{
  switch(result)
  {
    case NVSDK_NGX_Result_Success: return "Success";
    case NVSDK_NGX_Result_Fail: return "Fail";
    case NVSDK_NGX_Result_FAIL_FeatureNotSupported: return "FeatureNotSupported";
    case NVSDK_NGX_Result_FAIL_PlatformError: return "PlatformError";
    case NVSDK_NGX_Result_FAIL_FeatureAlreadyExists: return "FeatureAlreadyExists";
    case NVSDK_NGX_Result_FAIL_FeatureNotFound: return "FeatureNotFound";
    case NVSDK_NGX_Result_FAIL_InvalidParameter: return "InvalidParameter";
    case NVSDK_NGX_Result_FAIL_ScratchBufferTooSmall: return "ScratchBufferTooSmall";
    case NVSDK_NGX_Result_FAIL_NotInitialized: return "NotInitialized";
    case NVSDK_NGX_Result_FAIL_UnsupportedInputFormat: return "UnsupportedInputFormat";
    case NVSDK_NGX_Result_FAIL_RWFlagMissing: return "RWFlagMissing";
    case NVSDK_NGX_Result_FAIL_MissingInput: return "MissingInput";
    case NVSDK_NGX_Result_FAIL_UnableToInitializeFeature: return "UnableToInitializeFeature";
    case NVSDK_NGX_Result_FAIL_OutOfDate: return "OutOfDate";
    case NVSDK_NGX_Result_FAIL_OutOfGPUMemory: return "OutOfGPUMemory";
    case NVSDK_NGX_Result_FAIL_UnsupportedFormat: return "UnsupportedFormat";
    case NVSDK_NGX_Result_FAIL_UnableToWriteToAppDataPath: return "UnableToWriteToAppDataPath";
    case NVSDK_NGX_Result_FAIL_UnsupportedParameter: return "UnsupportedParameter";
    case NVSDK_NGX_Result_FAIL_Denied: return "Denied";
    case NVSDK_NGX_Result_FAIL_NotImplemented: return "NotImplemented";
    default: return "Unknown(" + std::to_string(result) + ")";
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// NgxContext Implementation
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

NgxContext::~NgxContext()
{
  deinit();
}

NVSDK_NGX_Result NgxContext::init(const InitInfo& initInfo)
{
  if(!initInfo.instance || !initInfo.physicalDevice || !initInfo.device || !initInfo.queue)
  {
    return NVSDK_NGX_Result_FAIL_InvalidParameter;
  }

  if(m_device || m_ngxParams)
  {
    LOGE("NgxContext::init() already called\n");
    return NVSDK_NGX_Result_FAIL_FeatureAlreadyExists;
  }

  m_applicationPath.assign(initInfo.applicationPath.native().begin(), initInfo.applicationPath.native().end());

  NVSDK_NGX_FeatureCommonInfo info = {};
  info.LoggingInfo.MinimumLoggingLevel = initInfo.loggingLevel;

  // Initialize NGX API
  NGX_RETURN_ON_FAIL(NVSDK_NGX_VULKAN_Init(g_ApplicationID, m_applicationPath.c_str(), 
                                           initInfo.instance, initInfo.physicalDevice,
                                           initInfo.device, vkGetInstanceProcAddr, 
                                           vkGetDeviceProcAddr, &info));

  m_device         = initInfo.device;
  m_queue          = initInfo.queue;
  m_queueFamilyIdx = initInfo.queueFamilyIdx;

  NVSDK_NGX_Result result = NGX_CHECK(NVSDK_NGX_VULKAN_GetCapabilityParameters(&m_ngxParams));

  if(NVSDK_NGX_FAILED(result))
  {
    deinit();
    return result;
  }

  LOGI("NGX initialized successfully\n");
  return NVSDK_NGX_Result_Success;
}

void NgxContext::deinit()
{
  if(m_ngxParams)
  {
    NVSDK_NGX_VULKAN_DestroyParameters(m_ngxParams);
    m_ngxParams = nullptr;
  }

  if(m_device)
  {
    NVSDK_NGX_VULKAN_Shutdown1(m_device);
    m_device = VK_NULL_HANDLE;
  }

  m_queue          = VK_NULL_HANDLE;
  m_queueFamilyIdx = 0;
}

NVSDK_NGX_Result NgxContext::querySupportedInputSizes(const QuerySizeInfo& queryInfo, SupportedSizes& renderSizes)
{
  if(!m_ngxParams)
    return NVSDK_NGX_Result_FAIL_NotInitialized;

  NGX_RETURN_ON_FAIL(NGX_DLSSD_GET_OPTIMAL_SETTINGS(m_ngxParams, queryInfo.outputSize.width, queryInfo.outputSize.height,
                                                    queryInfo.quality, &renderSizes.optimalSize.width,
                                                    &renderSizes.optimalSize.height, &renderSizes.maxSize.width,
                                                    &renderSizes.maxSize.height, &renderSizes.minSize.width,
                                                    &renderSizes.minSize.height, nullptr));

  return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result NgxContext::initDlssRR(const DlssRRInitInfo& initInfo, GsDlssRR& dlssrr)
{
  if(!m_device || !m_ngxParams)
    return NVSDK_NGX_Result_FAIL_NotInitialized;

  return dlssrr.init(m_device, m_queue, m_queueFamilyIdx, m_ngxParams, initInfo);
}

NVSDK_NGX_Result NgxContext::isDlssRRAvailable(VkInstance instance, VkPhysicalDevice physicalDevice)
{
  NVSDK_NGX_FeatureDiscoveryInfo info   = {};
  info.SDKVersion                       = NVSDK_NGX_Version_API;
  info.FeatureID                        = NVSDK_NGX_Feature_RayReconstruction;
  info.Identifier.IdentifierType        = NVSDK_NGX_Application_Identifier_Type_Application_Id;
  info.Identifier.v.ApplicationId       = g_ApplicationID;
  info.ApplicationDataPath              = L"";
  info.FeatureInfo                      = nullptr;

  NVSDK_NGX_FeatureRequirement requirements = {};

  NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_GetFeatureRequirements(instance, physicalDevice, &info, &requirements);

  if(NVSDK_NGX_FAILED(result))
  {
    LOGW("DLSS-RR feature requirements query failed: %s\n", getNGXResultString(result).c_str());
    return result;
  }

  if(requirements.FeatureSupported != NVSDK_NGX_FeatureSupportResult_Supported)
  {
    LOGW("DLSS-RR not supported on this device\n");
    return NVSDK_NGX_Result_FAIL_FeatureNotSupported;
  }

  return NVSDK_NGX_Result_Success;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GsDlssRR Implementation
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

GsDlssRR::~GsDlssRR()
{
  deinit();
}

void GsDlssRR::deinit()
{
  if(m_dlssdHandle && m_device)
  {
    NVSDK_NGX_VULKAN_ReleaseFeature(m_dlssdHandle);
    m_dlssdHandle = nullptr;
  }

  m_device    = VK_NULL_HANDLE;
  m_ngxParams = nullptr;
  m_resources.fill({});
}

NVSDK_NGX_Result GsDlssRR::init(VkDevice device, VkQueue queue, uint32_t queueFamilyIdx,
                                 NVSDK_NGX_Parameter* ngxParams, const NgxContext::DlssRRInitInfo& info)
{
  if(m_dlssdHandle)
  {
    LOGE("GsDlssRR::init() already called\n");
    return NVSDK_NGX_Result_FAIL_FeatureAlreadyExists;
  }

  m_device     = device;
  m_ngxParams  = ngxParams;
  m_outputSize = info.outputSize;
  m_inputSize  = info.inputSize;

  m_resources.fill({.Resource = {.ImageViewInfo = {}}});

  NVSDK_NGX_DLSSD_Create_Params dlssdParams = {};

  dlssdParams.InDenoiseMode   = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;
  dlssdParams.InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode_Packed;  // Roughness in normal.w
  dlssdParams.InUseHWDepth    = NVSDK_NGX_DLSS_Depth_Type_Linear;      // Linear depth

  dlssdParams.InWidth        = m_inputSize.width;
  dlssdParams.InHeight       = m_inputSize.height;
  dlssdParams.InTargetWidth  = m_outputSize.width;
  dlssdParams.InTargetHeight = m_outputSize.height;

  // Required flags: HDR input, motion vectors at render resolution
  dlssdParams.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
  dlssdParams.InPerfQualityValue   = info.quality;

  const uint32_t creationNodeMask   = 0x1;
  const uint32_t visibilityNodeMask = 0x1;

  // Set render preset for all quality modes
  ngxParams->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Quality, info.preset);
  ngxParams->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraQuality, info.preset);
  ngxParams->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Balanced, info.preset);
  ngxParams->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Performance, info.preset);
  ngxParams->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraPerformance, info.preset);

  // Create command buffer for initialization
  VkCommandPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  poolInfo.flags                   = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  poolInfo.queueFamilyIndex        = queueFamilyIdx;

  VkCommandPool cmdPool;
  vkCreateCommandPool(device, &poolInfo, nullptr, &cmdPool);

  VkCommandBufferAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  allocInfo.commandPool                 = cmdPool;
  allocInfo.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount          = 1;

  VkCommandBuffer cmd;
  vkAllocateCommandBuffers(device, &allocInfo, &cmd);

  VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  beginInfo.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &beginInfo);

  NVSDK_NGX_Result result = NGX_CHECK(NGX_VULKAN_CREATE_DLSSD_EXT1(device, cmd, creationNodeMask, visibilityNodeMask,
                                                                   &m_dlssdHandle, ngxParams, &dlssdParams));

  vkEndCommandBuffer(cmd);

  VkSubmitInfo submitInfo       = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers    = &cmd;
  vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue);

  vkDestroyCommandPool(device, cmdPool, nullptr);

  if(NVSDK_NGX_FAILED(result))
  {
    m_dlssdHandle = nullptr;
    return result;
  }

  LOGI("DLSS-RR initialized: input %dx%d -> output %dx%d\n", 
       m_inputSize.width, m_inputSize.height, m_outputSize.width, m_outputSize.height);

  return NVSDK_NGX_Result_Success;
}

void GsDlssRR::setResource(Resource resourceId, VkImage image, VkImageView imageView, VkFormat format)
{
  if(!m_dlssdHandle)
    return;

  VkImageSubresourceRange range = {};
  range.aspectMask              = VK_IMAGE_ASPECT_COLOR_BIT;
  range.baseArrayLayer          = 0;
  range.baseMipLevel            = 0;
  range.layerCount              = 1;
  range.levelCount              = 1;

  VkExtent2D size = (resourceId == RESOURCE_COLOR_OUT) ? m_outputSize : m_inputSize;
  bool       readWrite = (resourceId == RESOURCE_COLOR_OUT);

  m_resources[resourceId] = NVSDK_NGX_Create_ImageView_Resource_VK(imageView, image, range, format, 
                                                                    size.width, size.height, readWrite);
}

void GsDlssRR::resetResource(Resource resourceId)
{
  m_resources[resourceId] = {};
}

NVSDK_NGX_Result GsDlssRR::denoise(VkCommandBuffer  cmd,
                                    glm::uvec2       renderSize,
                                    glm::vec2        jitter,
                                    const glm::mat4& modelView,
                                    const glm::mat4& projection,
                                    bool             reset)
{
  if(!m_dlssdHandle)
    return NVSDK_NGX_Result_FAIL_NotInitialized;

  nvvk::DebugUtil::ScopedCmdLabel cmdLabel(cmd, "DLSS-RR Denoising");

  auto getResource = [this](Resource res) -> NVSDK_NGX_Resource_VK* {
    return m_resources[res].Resource.ImageViewInfo.ImageView ? &m_resources[res] : nullptr;
  };

  NVSDK_NGX_VK_DLSSD_Eval_Params evalParams = {};

  evalParams.pInColor               = getResource(RESOURCE_COLOR_IN);
  evalParams.pInOutput              = getResource(RESOURCE_COLOR_OUT);
  evalParams.pInDiffuseAlbedo       = getResource(RESOURCE_DIFFUSE_ALBEDO);
  evalParams.pInSpecularAlbedo      = getResource(RESOURCE_SPECULAR_ALBEDO);
  evalParams.pInSpecularHitDistance = getResource(RESOURCE_SPECULAR_HIT_DIST);
  evalParams.pInNormals             = getResource(RESOURCE_NORMAL_ROUGHNESS);
  evalParams.pInDepth               = getResource(RESOURCE_LINEAR_DEPTH);
  evalParams.pInMotionVectors       = getResource(RESOURCE_MOTION_VECTORS);
  evalParams.pInRoughness           = getResource(RESOURCE_NORMAL_ROUGHNESS);

  // Jitter is negated for DLSS-RR
  evalParams.InJitterOffsetX = -jitter.x;
  evalParams.InJitterOffsetY = -jitter.y;
  evalParams.InMVScaleX      = 1.0f;
  evalParams.InMVScaleY      = 1.0f;

  evalParams.InRenderSubrectDimensions.Width  = renderSize.x;
  evalParams.InRenderSubrectDimensions.Height = renderSize.y;

  // GLM matrices are column-major, DLSS expects row-major with left-multiply
  // Double transpose cancels out, so we can pass GLM matrices directly
  evalParams.pInWorldToViewMatrix = const_cast<float*>(glm::value_ptr(modelView));
  evalParams.pInViewToClipMatrix  = const_cast<float*>(glm::value_ptr(projection));

  evalParams.InReset = reset;

  NGX_RETURN_ON_FAIL(NGX_VULKAN_EVALUATE_DLSSD_EXT(cmd, m_dlssdHandle, m_ngxParams, &evalParams));

  return NVSDK_NGX_Result_Success;
}

}  // namespace vk_viewer

#endif  // WITH_DLSS_RR
