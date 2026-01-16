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

#include "doctest.h"

#ifdef WITH_VULKAN_VIDEO
#include "../src/vulkan_video_decoder.h"
#include <volk.h>
#include <cstring>

TEST_CASE("Vulkan Video extension list")
{
    auto extensions = vk_viewer::getVulkanVideoExtensions();
    
    CHECK(extensions.size() >= 3);
    
    bool hasVideoQueue = false;
    bool hasDecodeQueue = false;
    bool hasH265 = false;
    
    for (const char* ext : extensions) {
        if (strcmp(ext, VK_KHR_VIDEO_QUEUE_EXTENSION_NAME) == 0) hasVideoQueue = true;
        if (strcmp(ext, VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME) == 0) hasDecodeQueue = true;
        if (strcmp(ext, VK_KHR_VIDEO_DECODE_H265_EXTENSION_NAME) == 0) hasH265 = true;
    }
    
    CHECK(hasVideoQueue);
    CHECK(hasDecodeQueue);
    CHECK(hasH265);
}

TEST_CASE("VulkanVideoCapabilities default state")
{
    vk_viewer::VulkanVideoCapabilities caps;
    
    CHECK_FALSE(caps.supported);
    CHECK_FALSE(caps.h264DecodeSupported);
    CHECK_FALSE(caps.h265DecodeSupported);
    CHECK(caps.videoQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED);
    CHECK(caps.maxDpbSlots == 0);
}

TEST_CASE("VulkanDecodedFrame default state")
{
    vk_viewer::VulkanDecodedFrame frame;
    
    CHECK(frame.image == VK_NULL_HANDLE);
    CHECK(frame.imageView == VK_NULL_HANDLE);
    CHECK(frame.format == VK_FORMAT_UNDEFINED);
    CHECK(frame.width == 0);
    CHECK(frame.height == 0);
    CHECK(frame.timestampSec == 0.0);
}

TEST_CASE("VulkanVideoDecoder default state")
{
    vk_viewer::VulkanVideoDecoder decoder;
    
    CHECK_FALSE(decoder.isInitialized());
    CHECK_FALSE(decoder.isOpen());
    CHECK(decoder.getDuration() == 0.0);
    CHECK(decoder.getFrameRate() == 0.0);
    CHECK(decoder.getFrameCount() == 0);
    CHECK(decoder.getVideoQueueFamilyIndex() == VK_QUEUE_FAMILY_IGNORED);
}

TEST_CASE("VulkanVideoDecoder getDimensions returns zero when not initialized")
{
    vk_viewer::VulkanVideoDecoder decoder;
    
    uint32_t width = 9999, height = 9999;
    decoder.getDimensions(width, height);
    CHECK(width == 0);
    CHECK(height == 0);
}

TEST_CASE("DpbSlot default state")
{
    vk_viewer::DpbSlot slot;
    
    CHECK(slot.image == VK_NULL_HANDLE);
    CHECK(slot.imageView == VK_NULL_HANDLE);
    CHECK(slot.memory == VK_NULL_HANDLE);
    CHECK(slot.slotIndex == -1);
    CHECK(slot.pts == 0);
    CHECK_FALSE(slot.inUse);
}

#else // WITH_VULKAN_VIDEO

TEST_CASE("Vulkan Video disabled - placeholder test")
{
    // When Vulkan Video is disabled, just verify the test framework works
    CHECK(true);
}

#endif // WITH_VULKAN_VIDEO
