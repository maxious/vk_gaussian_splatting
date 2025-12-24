/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vulkan/vulkan_core.h>
#include <nvutils/logger.hpp>

namespace vk_gaussian_splatting {

// HDR Display Support Utilities
class HDRSupport
{
public:
  // Check if the current swapchain color space is HDR
  static bool isHDRColorSpace(VkColorSpaceKHR colorSpace)
  {
    switch(colorSpace)
    {
      case VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT:
      case VK_COLOR_SPACE_DCI_P3_LINEAR_EXT:
      case VK_COLOR_SPACE_BT2020_LINEAR_EXT:
      case VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT:
        return true;
      default:
        return false;
    }
  }

  // Get a human-readable name for the color space
  static const char* getColorSpaceName(VkColorSpaceKHR colorSpace)
  {
    switch(colorSpace)
    {
      case VK_COLOR_SPACE_SRGB_NONLINEAR_KHR:
        return "sRGB (SDR)";
      case VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT:
        return "Extended sRGB Linear (HDR)";
      case VK_COLOR_SPACE_DCI_P3_LINEAR_EXT:
        return "DCI-P3 Linear (HDR)";
      case VK_COLOR_SPACE_BT2020_LINEAR_EXT:
        return "BT.2020 Linear (HDR)";
      case VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT:
        return "Extended sRGB Non-Linear (HDR)";
      default:
        return "Unknown";
    }
  }

  // Log HDR detection info
  static void logHDRStatus(VkColorSpaceKHR colorSpace, VkFormat format)
  {
    const bool hdrEnabled = isHDRColorSpace(colorSpace);
    const char* colorSpaceName = getColorSpaceName(colorSpace);
    
    LOGI("Display Configuration: %s\n", hdrEnabled ? "HDR ENABLED" : "SDR");
    LOGI("  Color Space: %s\n", colorSpaceName);
    LOGI("  Format: %d\n", format);
  }

  // Global accessors for swapchain state (since we cannot modify nvvk::Swapchain header)
  static void setGlobalColorSpace(VkColorSpaceKHR colorSpace);
  static VkColorSpaceKHR getGlobalColorSpace();

  static void setGlobalFormat(VkFormat format);
  static VkFormat getGlobalFormat();
};

}  // namespace vk_gaussian_splatting
