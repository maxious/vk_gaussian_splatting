/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <nvutils/logger.hpp>

#include <nvvk/commands.hpp>
#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/swapchain.hpp>

// Include our local HDR support helper
#include "hdr_support.h"

using namespace vk_viewer;

VkResult nvvk::Swapchain::init(const InitInfo& info)
{
  m_physicalDevice  = info.physicalDevice;
  m_device          = info.device;
  m_queue           = info.queue;
  m_surface         = info.surface;
  m_imageUsage      = info.imageUsage;
  m_cmdPool         = info.cmdPool;
  m_preferredFormat = info.preferredFormat;
  if(info.preferredVsyncOffMode != VK_PRESENT_MODE_MAX_ENUM_KHR)
    m_preferredVsyncOffMode = info.preferredVsyncOffMode;
  if(info.preferredVsyncOnMode != VK_PRESENT_MODE_MAX_ENUM_KHR)
    m_preferredVsyncOnMode = info.preferredVsyncOnMode;
  m_preferredSurfaceFormat  = info.preferredFormat.surfaceFormat;
  m_preferredImageCount     = std::max(2u, info.preferredImageCount);
  m_preferredFramesInFlight = std::max(1u, info.preferredFramesInFlight);

  VkBool32 supportsPresent = VK_FALSE;
  NVVK_FAIL_RETURN(vkGetPhysicalDeviceSurfaceSupportKHR(info.physicalDevice, info.queue.familyIndex, info.surface, &supportsPresent));

  if(supportsPresent != VK_TRUE)
  {
    LOGW("Selected queue family %d cannot present on surface %px. Swapchain creation failed.\n", info.queue.familyIndex, info.surface);
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  return VK_SUCCESS;
}

void nvvk::Swapchain::deinit()
{
  if(m_device)
  {
    deinitResources();
  }
  *this = {};
}

VkResult nvvk::Swapchain::initResources(VkExtent2D& outWindowSize, bool vSync)
{
  // Query the physical device's capabilities for the given surface.
  const VkPhysicalDeviceSurfaceInfo2KHR surfaceInfo2{.sType   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR,
                                                     .surface = m_surface};
  VkSurfaceCapabilities2KHR             capabilities2{.sType = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR};
  NVVK_FAIL_RETURN(vkGetPhysicalDeviceSurfaceCapabilities2KHR(m_physicalDevice, &surfaceInfo2, &capabilities2));

  m_availableFormats = getAvailableFormats(m_physicalDevice, m_surface);
  if(m_availableFormats.empty())
  {
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  uint32_t presentModeCount;
  NVVK_FAIL_RETURN(vkGetPhysicalDeviceSurfacePresentModesKHR(m_physicalDevice, m_surface, &presentModeCount, nullptr));
  std::vector<VkPresentModeKHR> presentModes(presentModeCount);
  NVVK_FAIL_RETURN(
      vkGetPhysicalDeviceSurfacePresentModesKHR(m_physicalDevice, m_surface, &presentModeCount, presentModes.data()));

  // Choose the best available surface format and present mode and store it
  m_surfaceFormat = selectSwapSurfaceFormat(m_availableFormats);

  // Store HDR color space info via side-channel
  HDRSupport::setGlobalColorSpace(m_surfaceFormat.surfaceFormat.colorSpace);
  HDRSupport::setGlobalFormat(m_surfaceFormat.surfaceFormat.format);

  const VkPresentModeKHR presentMode = selectSwapPresentMode(presentModes, vSync);

  // Set the window size according to the surface's current extent.
  const VkSurfaceCapabilitiesKHR& caps = capabilities2.surfaceCapabilities;
  if(caps.currentExtent.width == UINT32_MAX)
  {
    outWindowSize.width  = std::max(caps.minImageExtent.width, outWindowSize.width);
    outWindowSize.height = std::max(caps.minImageExtent.height, outWindowSize.height);
  }
  else
  {
    outWindowSize = caps.currentExtent;
  }

  // Pick a swapchain image count: honour the user's preferred value but
  // clamp to the surface's [minImageCount, maxImageCount] bounds.
  const uint32_t minImageCount       = caps.minImageCount;
  const uint32_t preferredImageCount = std::max(m_preferredImageCount, minImageCount);

  // Handle the maxImageCount case where 0 means "no upper limit"
  const uint32_t maxImageCount = (caps.maxImageCount == 0) ?
                                     preferredImageCount  // No upper limit, use preferred
                                     :
                                     caps.maxImageCount;

  // Clamp preferredImageCount to valid range [minImageCount, maxImageCount]
  m_imageCount = std::max(minImageCount, std::min(preferredImageCount, maxImageCount));

  // Create the swapchain itself
  const VkSwapchainCreateInfoKHR swapchainCreateInfo{
      .sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .surface          = m_surface,
      .minImageCount    = m_imageCount,
      .imageFormat      = m_surfaceFormat.surfaceFormat.format,
      .imageColorSpace  = m_surfaceFormat.surfaceFormat.colorSpace,
      .imageExtent      = outWindowSize,
      .imageArrayLayers = 1,
      .imageUsage       = m_imageUsage,
      .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .preTransform     = caps.currentTransform,
      .compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      .presentMode      = presentMode,
      .clipped          = VK_TRUE,
  };
  NVVK_FAIL_RETURN(vkCreateSwapchainKHR(m_device, &swapchainCreateInfo, nullptr, &m_swapChain));
  NVVK_DBG_NAME(m_swapChain);

  // Retrieve the swapchain images.
  uint32_t imageCount;
  NVVK_FAIL_RETURN(vkGetSwapchainImagesKHR(m_device, m_swapChain, &imageCount, nullptr));
  // On llvmpipe for instance, we can get more images than the minimum requested.
  // We still need to get a handle for each image in the swapchain
  // (because vkAcquireNextImageKHR can return an index to each image),
  // so adjust m_imageCount.
  assert((m_imageCount <= imageCount) && "Wrong swapchain setup");
  m_imageCount = imageCount;
  std::vector<VkImage> swapImages(m_imageCount);
  NVVK_FAIL_RETURN(vkGetSwapchainImagesKHR(m_device, m_swapChain, &m_imageCount, swapImages.data()));

  // Frames-in-flight: clamp preferred to imageCount.
  m_framesInFlight = std::min(m_preferredFramesInFlight, m_imageCount);

  // Per-image storage: VkImage, VkImageView, and the presentSemaphore
  // (binary semaphore that present waits on; must follow the image).
  m_images.resize(m_imageCount);
  VkImageViewCreateInfo imageViewCreateInfo{
      .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format   = m_surfaceFormat.surfaceFormat.format,
      .components = {.r = VK_COMPONENT_SWIZZLE_IDENTITY, .g = VK_COMPONENT_SWIZZLE_IDENTITY, .b = VK_COMPONENT_SWIZZLE_IDENTITY, .a = VK_COMPONENT_SWIZZLE_IDENTITY},
      .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1},
  };
  const VkSemaphoreCreateInfo semaphoreCreateInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  for(uint32_t i = 0; i < m_imageCount; i++)
  {
    m_images[i].image = swapImages[i];
    NVVK_DBG_NAME(m_images[i].image);
    imageViewCreateInfo.image = m_images[i].image;
    NVVK_FAIL_RETURN(vkCreateImageView(m_device, &imageViewCreateInfo, nullptr, &m_images[i].imageView));
    NVVK_DBG_NAME(m_images[i].imageView);
    NVVK_FAIL_RETURN(vkCreateSemaphore(m_device, &semaphoreCreateInfo, nullptr, &m_images[i].presentSemaphore));
    NVVK_DBG_NAME(m_images[i].presentSemaphore);
    m_images[i].currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  }

  // Per-in-flight-slot storage: acquireSemaphore (consumed by acquire).
  m_frameResources.resize(m_framesInFlight);
  for(size_t i = 0; i < m_framesInFlight; ++i)
  {
    NVVK_FAIL_RETURN(vkCreateSemaphore(m_device, &semaphoreCreateInfo, nullptr, &m_frameResources[i].acquireSemaphore));
    NVVK_DBG_NAME(m_frameResources[i].acquireSemaphore);
  }

  return VK_SUCCESS;
}

VkResult nvvk::Swapchain::reinitResources(VkExtent2D& outWindowSize, bool vSync)
{
  // Wait for all frames to finish rendering before recreating the swapchain
  vkQueueWaitIdle(m_queue.queue);

  m_frameResourceIndex = 0;
  m_needRebuild        = false;
  deinitResources();
  return initResources(outWindowSize, vSync);
}

void nvvk::Swapchain::deinitResources()
{
  vkDestroySwapchainKHR(m_device, m_swapChain, nullptr);
  for(auto& frameRes : m_frameResources)
  {
    vkDestroySemaphore(m_device, frameRes.acquireSemaphore, nullptr);
  }
  m_frameResources.clear();
  for(auto& image : m_images)
  {
    vkDestroyImageView(m_device, image.imageView, nullptr);
    vkDestroySemaphore(m_device, image.presentSemaphore, nullptr);
  }
  m_images.clear();
}

VkResult nvvk::Swapchain::acquireNextImage(VkDevice device)
{
  assert((m_needRebuild == false) && "Swapbuffer need to call reinitResources()");

  // Get the frame resources for the current frame
  // We use m_frameResourceIndex here because we want to ensure we don't overwrite resources
  // that are still in use by previous frames
  auto& frame = m_frameResources[m_frameResourceIndex];

  // Acquire the next image from the swapchain
  // This will signal frame.acquireSemaphore when the image is ready
  // and store the index of the acquired image in m_frameImageIndex
  VkResult result = vkAcquireNextImageKHR(device, m_swapChain, std::numeric_limits<uint64_t>::max(),
                                          frame.acquireSemaphore, VK_NULL_HANDLE, &m_frameImageIndex);

  switch(result)
  {
    case VK_SUCCESS:
    case VK_SUBOPTIMAL_KHR:  // Still valid for presentation
      return result;

    case VK_ERROR_OUT_OF_DATE_KHR:  // The swapchain is no longer compatible with the surface and needs to be recreated
      m_needRebuild = true;
      return result;

    default:
      LOGW("Failed to acquire swapchain image: %d\n", result);
      return result;
  }
}

void nvvk::Swapchain::presentFrame(VkQueue queue)
{
  // Present must wait on the present semaphore that follows the image
  // (not the in-flight slot), because vkAcquireNextImageKHR can return images
  // out of order.
  const VkSemaphore presentSem = m_images[m_frameImageIndex].presentSemaphore;

  // Setup the presentation info, linking the swapchain and the image index
  const VkPresentInfoKHR presentInfo{
      .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .waitSemaphoreCount = 1,                   // Wait for rendering to finish
      .pWaitSemaphores    = &presentSem,         // Per-image semaphore
      .swapchainCount     = 1,                   // Swapchain to present the image
      .pSwapchains        = &m_swapChain,        // Pointer to the swapchain
      .pImageIndices      = &m_frameImageIndex,  // Index of the image to present
  };

  // Present the image and handle potential resizing issues
  const VkResult result = vkQueuePresentKHR(queue, &presentInfo);
  // If the swapchain is out of date (e.g., window resized), it needs to be rebuilt
  if(result == VK_ERROR_OUT_OF_DATE_KHR)
  {
    m_needRebuild = true;
  }
  else
  {
    assert((result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) && "Couldn't present swapchain image");
  }

  // Advance to the next CPU in-flight slot (NOT the next image -- images are
  // chosen by the presentation engine).
  m_frameResourceIndex = (m_frameResourceIndex + 1) % m_framesInFlight;
}

VkSurfaceFormat2KHR nvvk::Swapchain::selectSwapSurfaceFormat(const std::vector<VkSurfaceFormat2KHR>& availableFormats) const
{
  // If a preferred format is specified, use it.
  // We don't check for availability assuming the preferred format had been selected from the list of available formats.
  if(m_preferredFormat.surfaceFormat.format != VK_FORMAT_UNDEFINED)
  {
    return m_preferredFormat;
  }

  // If there's only one available format and it's undefined, return a default format.
  if(availableFormats.size() == 1 && availableFormats[0].surfaceFormat.format == VK_FORMAT_UNDEFINED)
  {
    VkSurfaceFormat2KHR result{.sType         = VK_STRUCTURE_TYPE_SURFACE_FORMAT_2_KHR,
                               .surfaceFormat = {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}};
    return result;
  }

  // Preferred HDR formats (try these first)
  const std::vector<VkSurfaceFormat2KHR> hdrFormats = {
      VkSurfaceFormat2KHR{.sType         = VK_STRUCTURE_TYPE_SURFACE_FORMAT_2_KHR,
                          .surfaceFormat = {VK_FORMAT_R16G16B16A16_SFLOAT, VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT}},
      VkSurfaceFormat2KHR{.sType         = VK_STRUCTURE_TYPE_SURFACE_FORMAT_2_KHR,
                          .surfaceFormat = {VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT}},
      VkSurfaceFormat2KHR{.sType         = VK_STRUCTURE_TYPE_SURFACE_FORMAT_2_KHR,
                          .surfaceFormat = {VK_FORMAT_R16G16B16A16_SFLOAT, VK_COLOR_SPACE_BT2020_LINEAR_EXT}},
      VkSurfaceFormat2KHR{.sType         = VK_STRUCTURE_TYPE_SURFACE_FORMAT_2_KHR,
                          .surfaceFormat = {VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_COLOR_SPACE_BT2020_LINEAR_EXT}}};

  // Try HDR formats first
  for(const auto& hdrFormat : hdrFormats)
  {
    for(const auto& availableFormat : availableFormats)
    {
      if(availableFormat.surfaceFormat.format == hdrFormat.surfaceFormat.format
         && availableFormat.surfaceFormat.colorSpace == hdrFormat.surfaceFormat.colorSpace)
      {
        LOGI("HDR display detected - using format: %d, colorspace: %d\n", hdrFormat.surfaceFormat.format, hdrFormat.surfaceFormat.colorSpace);
        return availableFormat;
      }
    }
  }

  // Fallback to SDR formats if no HDR format is available
  const std::vector<VkSurfaceFormat2KHR> sdrFormats = {
      VkSurfaceFormat2KHR{.sType         = VK_STRUCTURE_TYPE_SURFACE_FORMAT_2_KHR,
                          .surfaceFormat = {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}},
      VkSurfaceFormat2KHR{.sType         = VK_STRUCTURE_TYPE_SURFACE_FORMAT_2_KHR,
                          .surfaceFormat = {VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}}};

  for(const auto& sdrFormat : sdrFormats)
  {
    for(const auto& availableFormat : availableFormats)
    {
      if(availableFormat.surfaceFormat.format == sdrFormat.surfaceFormat.format
         && availableFormat.surfaceFormat.colorSpace == sdrFormat.surfaceFormat.colorSpace)
      {
        LOGI("SDR display - using format: %d, colorspace: %d\n", sdrFormat.surfaceFormat.format, sdrFormat.surfaceFormat.colorSpace);
        return availableFormat;
      }
    }
  }

  // If none of the preferred formats are available, return the first available format.
  LOGW("No preferred surface format found, using first available format: %d, colorspace: %d\n", 
       availableFormats[0].surfaceFormat.format, availableFormats[0].surfaceFormat.colorSpace);
  return availableFormats[0];
}

VkPresentModeKHR nvvk::Swapchain::selectSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes, bool vSync) const
{
  bool mailboxSupported = false, immediateSupported = false;

  for(VkPresentModeKHR mode : availablePresentModes)
  {
    if(vSync && (mode == m_preferredVsyncOnMode))
      return mode;

    if(!vSync && (mode == m_preferredVsyncOffMode))
      return mode;

    if(mode == VK_PRESENT_MODE_MAILBOX_KHR)
      mailboxSupported = true;
    if(mode == VK_PRESENT_MODE_IMMEDIATE_KHR)
      immediateSupported = true;
  }

  if(!vSync && immediateSupported)
  {
    return VK_PRESENT_MODE_IMMEDIATE_KHR;  // Best mode for low latency
  }

  if(mailboxSupported)
  {
    return VK_PRESENT_MODE_MAILBOX_KHR;
  }

  return VK_PRESENT_MODE_FIFO_KHR;
}
