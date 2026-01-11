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

#pragma once

#ifdef WITH_DLSS_RR

#include <vulkan/vulkan_core.h>
#include <nvsdk_ngx_vk.h>
#include <nvsdk_ngx_defs_dlssd.h>
#include <glm/glm.hpp>
#include <array>
#include <filesystem>
#include <string>

namespace vk_viewer {

// Helper to convert NGX error codes to string
std::string getNGXResultString(NVSDK_NGX_Result result);

// Forward declaration
class GsDlssRR;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// NgxContext - Manages NGX API initialization and DLSS-RR instance creation
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class NgxContext
{
public:
  NgxContext() = default;
  ~NgxContext();

  struct InitInfo
  {
    VkInstance       instance       = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice         device         = VK_NULL_HANDLE;
    VkQueue          queue          = VK_NULL_HANDLE;
    uint32_t         queueFamilyIdx = 0;
#ifdef NDEBUG
    NVSDK_NGX_Logging_Level loggingLevel = NVSDK_NGX_LOGGING_LEVEL_OFF;
#else
    NVSDK_NGX_Logging_Level loggingLevel = NVSDK_NGX_LOGGING_LEVEL_VERBOSE;
#endif
    std::filesystem::path applicationPath;
  };

  NVSDK_NGX_Result init(const InitInfo& initInfo);
  void             deinit();

  bool isValid() const { return m_device != VK_NULL_HANDLE && m_ngxParams != nullptr; }

  struct SupportedSizes
  {
    VkExtent2D minSize     = {};
    VkExtent2D maxSize     = {};
    VkExtent2D optimalSize = {};
  };

  struct QuerySizeInfo
  {
    VkExtent2D                  outputSize;
    NVSDK_NGX_PerfQuality_Value quality = NVSDK_NGX_PerfQuality_Value_MaxQuality;
  };

  NVSDK_NGX_Result querySupportedInputSizes(const QuerySizeInfo& queryInfo, SupportedSizes& renderSizes);

  struct DlssRRInitInfo
  {
    VkExtent2D                                     inputSize  = {};
    VkExtent2D                                     outputSize = {};
    NVSDK_NGX_PerfQuality_Value                    quality    = NVSDK_NGX_PerfQuality_Value_MaxQuality;
    NVSDK_NGX_RayReconstruction_Hint_Render_Preset preset = NVSDK_NGX_RayReconstruction_Hint_Render_Preset_Default;
  };

  NVSDK_NGX_Result initDlssRR(const DlssRRInitInfo& initInfo, GsDlssRR& dlssrr);

  // Check if DLSS-RR is available
  static NVSDK_NGX_Result isDlssRRAvailable(VkInstance instance, VkPhysicalDevice physicalDevice);

private:
  VkDevice             m_device         = VK_NULL_HANDLE;
  VkQueue              m_queue          = VK_NULL_HANDLE;
  uint32_t             m_queueFamilyIdx = 0;
  NVSDK_NGX_Parameter* m_ngxParams      = nullptr;
  std::wstring         m_applicationPath;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GsDlssRR - DLSS Ray Reconstruction denoiser wrapper
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GsDlssRR
{
public:
  GsDlssRR() = default;
  ~GsDlssRR();

  void deinit();
  bool isValid() const { return m_dlssdHandle != nullptr; }

  // Resource identifiers for DLSS-RR inputs/outputs
  enum Resource
  {
    RESOURCE_COLOR_IN = 0,        // Noisy input color (HDR)
    RESOURCE_COLOR_OUT,           // Denoised output color
    RESOURCE_DIFFUSE_ALBEDO,      // Diffuse albedo
    RESOURCE_SPECULAR_ALBEDO,     // Specular albedo
    RESOURCE_NORMAL_ROUGHNESS,    // Normal (xyz) + roughness (w)
    RESOURCE_MOTION_VECTORS,      // Screen-space motion vectors
    RESOURCE_LINEAR_DEPTH,        // Linear depth
    RESOURCE_SPECULAR_HIT_DIST,   // Optional: specular hit distance
    RESOURCE_COUNT
  };

  // Bind a Vulkan image to a DLSS-RR resource slot
  void setResource(Resource resourceId, VkImage image, VkImageView imageView, VkFormat format);
  void resetResource(Resource resourceId);

  // Execute denoising
  // renderSize: actual rendered subrectangle size
  // jitter: temporal jitter offset [-0.5, 0.5]
  // modelView/projection: camera matrices
  // reset: discard temporal history
  NVSDK_NGX_Result denoise(VkCommandBuffer  cmd,
                           glm::uvec2       renderSize,
                           glm::vec2        jitter,
                           const glm::mat4& modelView,
                           const glm::mat4& projection,
                           bool             reset = false);

  VkExtent2D getInputSize() const { return m_inputSize; }
  VkExtent2D getOutputSize() const { return m_outputSize; }

private:
  friend class NgxContext;
  NVSDK_NGX_Result init(VkDevice device, VkQueue queue, uint32_t queueFamilyIdx, 
                        NVSDK_NGX_Parameter* ngxParams, const NgxContext::DlssRRInitInfo& info);

  VkDevice                                           m_device      = VK_NULL_HANDLE;
  NVSDK_NGX_Parameter*                               m_ngxParams   = nullptr;
  NVSDK_NGX_Handle*                                  m_dlssdHandle = nullptr;
  VkExtent2D                                         m_inputSize   = {};
  VkExtent2D                                         m_outputSize  = {};
  std::array<NVSDK_NGX_Resource_VK, RESOURCE_COUNT>  m_resources   = {};
};

}  // namespace vk_viewer

#endif  // WITH_DLSS_RR
