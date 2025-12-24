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

#include "hdr_support.h"

namespace vk_gaussian_splatting {

// Static storage for the global color space
static VkColorSpaceKHR g_globalColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
static VkFormat        g_globalFormat     = VK_FORMAT_UNDEFINED;

void HDRSupport::setGlobalColorSpace(VkColorSpaceKHR colorSpace)
{
  g_globalColorSpace = colorSpace;
}

VkColorSpaceKHR HDRSupport::getGlobalColorSpace()
{
  return g_globalColorSpace;
}

void HDRSupport::setGlobalFormat(VkFormat format)
{
  g_globalFormat = format;
}

VkFormat HDRSupport::getGlobalFormat()
{
  return g_globalFormat;
}

}  // namespace vk_gaussian_splatting
