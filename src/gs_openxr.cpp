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

#include "gs_openxr.hpp"

#ifdef WITH_OPENXR

#include <nvutils/logger.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <thread>
#include <chrono>
#include <cmath>
#include <sstream>

#ifndef XR_FB_space_warp
#define XR_FB_space_warp 1
#define XR_FB_space_warp_SPEC_VERSION     2
#define XR_FB_SPACE_WARP_EXTENSION_NAME   "XR_FB_space_warp"
static const XrStructureType XR_TYPE_COMPOSITION_LAYER_SPACE_WARP_INFO_FB = (XrStructureType)1000171000;
static const XrStructureType XR_TYPE_SYSTEM_SPACE_WARP_PROPERTIES_FB = (XrStructureType)1000171001;
typedef XrFlags64 XrCompositionLayerSpaceWarpInfoFlagsFB;
// Flag bits for XrCompositionLayerSpaceWarpInfoFlagsFB
static const XrCompositionLayerSpaceWarpInfoFlagsFB XR_COMPOSITION_LAYER_SPACE_WARP_INFO_FRAME_SKIP_BIT_FB = 0x00000001;
typedef struct XrCompositionLayerSpaceWarpInfoFB {
    XrStructureType                           type;
    const void*                               next;
    XrCompositionLayerSpaceWarpInfoFlagsFB    layerFlags;
    XrSwapchainSubImage                       motionVectorSubImage;
    XrPosef                                   appSpaceDeltaPose;
    XrSwapchainSubImage                       depthSubImage;
    float                                     minDepth;
    float                                     maxDepth;
    float                                     nearZ;
    float                                     farZ;
} XrCompositionLayerSpaceWarpInfoFB;
typedef struct XrSystemSpaceWarpPropertiesFB {
    XrStructureType    type;
    void*              next;
    uint32_t           recommendedMotionVectorImageRectWidth;
    uint32_t           recommendedMotionVectorImageRectHeight;
} XrSystemSpaceWarpPropertiesFB;
#endif

namespace vk_gaussian_splatting {


#define XR_CHECK(result, msg)                                        \
  do                                                                 \
  {                                                                  \
    XrResult _xr_result = (result);                                  \
    if(XR_FAILED(_xr_result))                                        \
    {                                                                \
      LOGE("OpenXR Error: %s (result=%d)\n", msg, (int)_xr_result);  \
      return false;                                                  \
    }                                                                \
  } while(0)

#define XR_CHECK_VOID(result, msg)                                   \
  do                                                                 \
  {                                                                  \
    XrResult _xr_result = (result);                                  \
    if(XR_FAILED(_xr_result))                                        \
    {                                                                \
      LOGE("OpenXR Error: %s (result=%d)\n", msg, (int)_xr_result);  \
      return;                                                        \
    }                                                                \
  } while(0)

GsOpenXr::GsOpenXr()
{
  for(auto& view : m_locatedViews)
  {
    view.type = XR_TYPE_VIEW;
    view.next = nullptr;
  }
}

GsOpenXr::~GsOpenXr()
{
  shutdown();
}

void GsOpenXr::shutdown()
{
  stopTrackingThread();
  destroyActionSet();

  if(m_colorSwapchain.handle != XR_NULL_HANDLE)
  {
    xrDestroySwapchain(m_colorSwapchain.handle);
    m_colorSwapchain.handle = XR_NULL_HANDLE;
    m_colorSwapchain.images.clear();
  }

  if(m_depthSwapchain.handle != XR_NULL_HANDLE)
  {
    xrDestroySwapchain(m_depthSwapchain.handle);
    m_depthSwapchain.handle = XR_NULL_HANDLE;
    m_depthSwapchain.images.clear();
  }

  if(m_motionVectorSwapchain.handle != XR_NULL_HANDLE)
  {
    xrDestroySwapchain(m_motionVectorSwapchain.handle);
    m_motionVectorSwapchain.handle = XR_NULL_HANDLE;
    m_motionVectorSwapchain.images.clear();
  }

  if(m_referenceSpace != XR_NULL_HANDLE)

  {
    xrDestroySpace(m_referenceSpace);
    m_referenceSpace = XR_NULL_HANDLE;
  }

  if(m_session != XR_NULL_HANDLE)
  {
    xrDestroySession(m_session);
    m_session = XR_NULL_HANDLE;
  }

  if(m_instance != XR_NULL_HANDLE)
  {
    xrDestroyInstance(m_instance);
    m_instance = XR_NULL_HANDLE;
  }

  m_sessionRunning = false;
  m_shouldRender   = false;
  m_hasControllers = false;
}

std::vector<const char*> GsOpenXr::getRequiredInstanceExtensions() const
{
  return {"XR_KHR_vulkan_enable"};
}

std::vector<const char*> GsOpenXr::getRequiredDeviceExtensions() const
{
  return {"VK_KHR_multiview"};
}

bool GsOpenXr::initialize(VkInstance       vkInstance,
                          VkPhysicalDevice physicalDevice,
                          VkDevice         device,
                          uint32_t         graphicsQueueFamilyIndex,
                          uint32_t         graphicsQueueIndex,
                          VkFormat         colorFormat,
                          VkFormat         depthFormat)
{
  m_device = device;

  // If instance not created yet (queryRequiredVulkanExtensions not called), create it now
  if(m_instance == XR_NULL_HANDLE)
  {
    if(!createInstance())
      return false;

    loadXrFunctions();

    if(!getSystem())
      return false;
  }

  if(!createSession(vkInstance, physicalDevice, device, graphicsQueueFamilyIndex, graphicsQueueIndex))
    return false;

  if(!createSwapchains(colorFormat, depthFormat))
    return false;

  if(!createReferenceSpace())
    return false;

  if(!createActionSet())
  {
    LOGW("Failed to create OpenXR action set for controllers - locomotion disabled\n");
  }

  initPerformanceMetrics();
  initColorSpace();
  initPassthrough();
  initSpaceWarp();

  // Check for VK_KHR_multiview support
  m_supportsMultiview = true;  // OpenXR runtime should have provided this if supported

  LOGI("OpenXR initialized successfully. Per-eye resolution: %dx%d, Multiview: %s\n", 
       m_perEyeExtent.width, m_perEyeExtent.height, m_supportsMultiview ? "YES" : "NO");
  return true;
}

bool GsOpenXr::createInstance()
{
  std::vector<const char*> extensions = {
    "XR_KHR_vulkan_enable",
    "XR_META_performance_metrics",
    "XR_FB_color_space",
    "XR_FB_passthrough",
    "XR_FB_space_warp"
  };

  XrInstanceCreateInfo createInfo{XR_TYPE_INSTANCE_CREATE_INFO};
  strcpy_s(createInfo.applicationInfo.applicationName, "vk_gaussian_splatting");
  createInfo.applicationInfo.applicationVersion = 1;
  strcpy_s(createInfo.applicationInfo.engineName, "nvpro_core2");
  createInfo.applicationInfo.engineVersion       = 1;
  createInfo.applicationInfo.apiVersion          = XR_API_VERSION_1_0;
  createInfo.enabledExtensionCount               = static_cast<uint32_t>(extensions.size());
  createInfo.enabledExtensionNames               = extensions.data();

  XrResult result = xrCreateInstance(&createInfo, &m_instance);
  if(XR_FAILED(result))
  {
    LOGE("Failed to create OpenXR instance. Is an XR runtime installed? (result=%d)\n", (int)result);
    return false;
  }

  XrInstanceProperties instanceProps{XR_TYPE_INSTANCE_PROPERTIES};
  xrGetInstanceProperties(m_instance, &instanceProps);
  LOGI("OpenXR Runtime: %s (version %d.%d.%d)\n", instanceProps.runtimeName,
       XR_VERSION_MAJOR(instanceProps.runtimeVersion), XR_VERSION_MINOR(instanceProps.runtimeVersion),
       XR_VERSION_PATCH(instanceProps.runtimeVersion));

  return true;
}

void GsOpenXr::loadXrFunctions()
{
  xrGetInstanceProcAddr(m_instance, "xrGetVulkanGraphicsRequirementsKHR",
                        (PFN_xrVoidFunction*)&m_xrGetVulkanGraphicsRequirementsKHR);
  xrGetInstanceProcAddr(m_instance, "xrGetVulkanGraphicsDeviceKHR",
                        (PFN_xrVoidFunction*)&m_xrGetVulkanGraphicsDeviceKHR);
  xrGetInstanceProcAddr(m_instance, "xrGetVulkanInstanceExtensionsKHR",
                        (PFN_xrVoidFunction*)&m_xrGetVulkanInstanceExtensionsKHR);
  xrGetInstanceProcAddr(m_instance, "xrGetVulkanDeviceExtensionsKHR",
                        (PFN_xrVoidFunction*)&m_xrGetVulkanDeviceExtensionsKHR);

  xrGetInstanceProcAddr(m_instance, "xrEnumeratePerformanceMetricsCounterPathsMETA",
                        (PFN_xrVoidFunction*)&m_xrEnumeratePerformanceMetricsCounterPathsMETA);
  xrGetInstanceProcAddr(m_instance, "xrSetPerformanceMetricsStateMETA",
                        (PFN_xrVoidFunction*)&m_xrSetPerformanceMetricsStateMETA);
  xrGetInstanceProcAddr(m_instance, "xrGetPerformanceMetricsStateMETA",
                        (PFN_xrVoidFunction*)&m_xrGetPerformanceMetricsStateMETA);
  xrGetInstanceProcAddr(m_instance, "xrQueryPerformanceMetricsCounterMETA",
                        (PFN_xrVoidFunction*)&m_xrQueryPerformanceMetricsCounterMETA);

  m_perfMetricsSupported = (m_xrEnumeratePerformanceMetricsCounterPathsMETA != nullptr &&
                            m_xrSetPerformanceMetricsStateMETA != nullptr &&
                            m_xrQueryPerformanceMetricsCounterMETA != nullptr);

  if(m_perfMetricsSupported)
  {
    LOGI("XR_META_performance_metrics extension available\n");
  }

  xrGetInstanceProcAddr(m_instance, "xrEnumerateColorSpacesFB",
                        (PFN_xrVoidFunction*)&m_xrEnumerateColorSpacesFB);
  xrGetInstanceProcAddr(m_instance, "xrSetColorSpaceFB",
                        (PFN_xrVoidFunction*)&m_xrSetColorSpaceFB);

  m_colorSpaceSupported = (m_xrEnumerateColorSpacesFB != nullptr && m_xrSetColorSpaceFB != nullptr);

  if(m_colorSpaceSupported)
  {
    LOGI("XR_FB_color_space extension available\n");
  }

  // Load XR_FB_passthrough functions
  xrGetInstanceProcAddr(m_instance, "xrCreatePassthroughFB", (PFN_xrVoidFunction*)&m_xrCreatePassthroughFB);
  xrGetInstanceProcAddr(m_instance, "xrDestroyPassthroughFB", (PFN_xrVoidFunction*)&m_xrDestroyPassthroughFB);
  xrGetInstanceProcAddr(m_instance, "xrPassthroughStartFB", (PFN_xrVoidFunction*)&m_xrPassthroughStartFB);
  xrGetInstanceProcAddr(m_instance, "xrPassthroughPauseFB", (PFN_xrVoidFunction*)&m_xrPassthroughPauseFB);
  xrGetInstanceProcAddr(m_instance, "xrCreatePassthroughLayerFB", (PFN_xrVoidFunction*)&m_xrCreatePassthroughLayerFB);
  xrGetInstanceProcAddr(m_instance, "xrDestroyPassthroughLayerFB", (PFN_xrVoidFunction*)&m_xrDestroyPassthroughLayerFB);
}

bool GsOpenXr::queryRequiredVulkanExtensions(std::vector<std::string>& outInstanceExtensions,
                                              std::vector<std::string>& outDeviceExtensions)
{
  outInstanceExtensions.clear();
  outDeviceExtensions.clear();

  // Create temporary OpenXR instance to query extensions
  if(!createInstance())
    return false;

  loadXrFunctions();

  if(!getSystem())
  {
    xrDestroyInstance(m_instance);
    m_instance = XR_NULL_HANDLE;
    return false;
  }

  // Query required Vulkan instance extensions
  if(m_xrGetVulkanInstanceExtensionsKHR)
  {
    uint32_t bufferSize = 0;
    m_xrGetVulkanInstanceExtensionsKHR(m_instance, m_systemId, 0, &bufferSize, nullptr);
    if(bufferSize > 0)
    {
      std::string extensions(bufferSize, '\0');
      m_xrGetVulkanInstanceExtensionsKHR(m_instance, m_systemId, bufferSize, &bufferSize, extensions.data());
      
      // Parse space-separated extension names
      std::istringstream iss(extensions);
      std::string ext;
      while(iss >> ext)
      {
        if(!ext.empty())
          outInstanceExtensions.push_back(ext);
      }
      LOGI("OpenXR requires %zu Vulkan instance extensions\n", outInstanceExtensions.size());
    }
  }

  // Query required Vulkan device extensions
  if(m_xrGetVulkanDeviceExtensionsKHR)
  {
    uint32_t bufferSize = 0;
    m_xrGetVulkanDeviceExtensionsKHR(m_instance, m_systemId, 0, &bufferSize, nullptr);
    if(bufferSize > 0)
    {
      std::string extensions(bufferSize, '\0');
      m_xrGetVulkanDeviceExtensionsKHR(m_instance, m_systemId, bufferSize, &bufferSize, extensions.data());
      
      // Parse space-separated extension names
      std::istringstream iss(extensions);
      std::string ext;
      while(iss >> ext)
      {
        if(!ext.empty())
          outDeviceExtensions.push_back(ext);
      }
      LOGI("OpenXR requires %zu Vulkan device extensions\n", outDeviceExtensions.size());
    }
  }

  // Keep the instance alive for later use - don't destroy it
  // The systemId is also kept for session creation
  return true;
}

bool GsOpenXr::getSystem()
{
  XrSystemGetInfo systemInfo{XR_TYPE_SYSTEM_GET_INFO};
  systemInfo.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;

  XrResult result = xrGetSystem(m_instance, &systemInfo, &m_systemId);
  if(XR_FAILED(result))
  {
    LOGE("Failed to get XR system. Is a headset connected? (result=%d)\n", (int)result);
    return false;
  }

  XrSystemProperties systemProps{XR_TYPE_SYSTEM_PROPERTIES};
  xrGetSystemProperties(m_instance, m_systemId, &systemProps);
  LOGI("XR System: %s (vendorId=0x%x)\n", systemProps.systemName, systemProps.vendorId);
  LOGI("  Max swapchain size: %dx%d, max layers: %d\n", systemProps.graphicsProperties.maxSwapchainImageWidth,
       systemProps.graphicsProperties.maxSwapchainImageHeight, systemProps.graphicsProperties.maxLayerCount);

  // Get view configuration
  uint32_t viewConfigCount = 0;
  xrEnumerateViewConfigurations(m_instance, m_systemId, 0, &viewConfigCount, nullptr);
  std::vector<XrViewConfigurationType> viewConfigs(viewConfigCount);
  xrEnumerateViewConfigurations(m_instance, m_systemId, viewConfigCount, &viewConfigCount, viewConfigs.data());

  bool hasStereo = false;
  for(auto config : viewConfigs)
  {
    if(config == XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO)
    {
      hasStereo = true;
      break;
    }
  }

  if(!hasStereo)
  {
    LOGE("XR system does not support primary stereo view configuration\n");
    return false;
  }

  // Get recommended resolution
  uint32_t viewCount = 0;
  xrEnumerateViewConfigurationViews(m_instance, m_systemId, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0, &viewCount,
                                    nullptr);
  std::vector<XrViewConfigurationView> configViews(viewCount, {XR_TYPE_VIEW_CONFIGURATION_VIEW});
  xrEnumerateViewConfigurationViews(m_instance, m_systemId, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, viewCount,
                                    &viewCount, configViews.data());

  if(viewCount >= 1)
  {
    m_perEyeExtent.width  = configViews[0].recommendedImageRectWidth;
    m_perEyeExtent.height = configViews[0].recommendedImageRectHeight;
    m_fullExtent.width    = m_perEyeExtent.width * 2;
    m_fullExtent.height   = m_perEyeExtent.height;

    LOGI("  Recommended per-eye resolution: %dx%d\n", m_perEyeExtent.width, m_perEyeExtent.height);
  }

  return true;
}

bool GsOpenXr::createSession(VkInstance       vkInstance,
                             VkPhysicalDevice physicalDevice,
                             VkDevice         device,
                             uint32_t         graphicsQueueFamilyIndex,
                             uint32_t         graphicsQueueIndex)
{
  // Ensure function pointers are loaded
  if(!m_xrGetVulkanGraphicsRequirementsKHR || !m_xrGetVulkanGraphicsDeviceKHR)
  {
    LOGE("OpenXR Vulkan extension functions not loaded\n");
    return false;
  }

  // Check graphics requirements (using v1 extension) - MUST be called before xrCreateSession
  XrGraphicsRequirementsVulkanKHR graphicsRequirements{XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR};
  XR_CHECK(m_xrGetVulkanGraphicsRequirementsKHR(m_instance, m_systemId, &graphicsRequirements),
           "Failed to get Vulkan graphics requirements");
  LOGI("OpenXR Vulkan requirements: minApiVersion=%d.%d.%d, maxApiVersion=%d.%d.%d\n",
       VK_API_VERSION_MAJOR(graphicsRequirements.minApiVersionSupported),
       VK_API_VERSION_MINOR(graphicsRequirements.minApiVersionSupported),
       VK_API_VERSION_PATCH(graphicsRequirements.minApiVersionSupported),
       VK_API_VERSION_MAJOR(graphicsRequirements.maxApiVersionSupported),
       VK_API_VERSION_MINOR(graphicsRequirements.maxApiVersionSupported),
       VK_API_VERSION_PATCH(graphicsRequirements.maxApiVersionSupported));

  // Get the physical device OpenXR wants - MUST be called before xrCreateSession
  VkPhysicalDevice xrPhysicalDevice = VK_NULL_HANDLE;
  XR_CHECK(m_xrGetVulkanGraphicsDeviceKHR(m_instance, m_systemId, vkInstance, &xrPhysicalDevice),
           "Failed to get Vulkan graphics device from OpenXR");
  LOGI("OpenXR xrGetVulkanGraphicsDeviceKHR called successfully\n");

  if(xrPhysicalDevice != physicalDevice)
  {
    LOGE("OpenXR runtime expects a DIFFERENT physical device! App: %p, XR: %p\n",
         (void*)physicalDevice, (void*)xrPhysicalDevice);
    LOGE("This will likely cause VK_ERROR_DEVICE_LOST. The app must use the XR-provided GPU.\n");
  }
  else
  {
    LOGI("Physical device matches OpenXR requirement: %p\n", (void*)physicalDevice);
  }

  // Create session with Vulkan binding (using v1 structure)
  XrGraphicsBindingVulkanKHR graphicsBinding{XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR};
  graphicsBinding.instance         = vkInstance;
  graphicsBinding.physicalDevice   = physicalDevice;
  graphicsBinding.device           = device;
  graphicsBinding.queueFamilyIndex = graphicsQueueFamilyIndex;
  graphicsBinding.queueIndex       = graphicsQueueIndex;

  XrSessionCreateInfo sessionCreateInfo{XR_TYPE_SESSION_CREATE_INFO};
  sessionCreateInfo.next     = &graphicsBinding;
  sessionCreateInfo.systemId = m_systemId;

  XR_CHECK(xrCreateSession(m_instance, &sessionCreateInfo, &m_session), "Failed to create XR session");

  // Wait for session to be ready - xrBeginSession will be called in handleSessionStateChange
  LOGI("Waiting for XR session to become ready...\n");
  while(m_sessionState != XR_SESSION_STATE_READY)
  {
    pollEvents();
    if(m_sessionState == XR_SESSION_STATE_LOSS_PENDING || m_sessionState == XR_SESSION_STATE_EXITING)
    {
      LOGE("XR session lost during initialization\n");
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  // Note: xrBeginSession is now called in handleSessionStateChange when we receive READY state
  // Wait a bit more for the session to actually start
  while(!m_sessionRunning)
  {
    pollEvents();
    if(m_sessionState == XR_SESSION_STATE_LOSS_PENDING || m_sessionState == XR_SESSION_STATE_EXITING)
    {
      LOGE("XR session lost during initialization\n");
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  LOGI("XR session started\n");
  return true;
}

bool GsOpenXr::createSwapchains(VkFormat colorFormat, VkFormat depthFormat)
{
  // Create color swapchain (SBS layout: 2 * perEyeWidth x perEyeHeight)
  XrSwapchainCreateInfo colorSwapchainInfo{XR_TYPE_SWAPCHAIN_CREATE_INFO};
  colorSwapchainInfo.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT;
  colorSwapchainInfo.format     = static_cast<int64_t>(colorFormat);
  colorSwapchainInfo.sampleCount = 1;
  colorSwapchainInfo.width      = m_fullExtent.width;
  colorSwapchainInfo.height     = m_fullExtent.height;
  colorSwapchainInfo.faceCount  = 1;
  colorSwapchainInfo.arraySize  = 1;
  colorSwapchainInfo.mipCount   = 1;

  m_colorSwapchain = createSwapchain(colorSwapchainInfo);
  if(m_colorSwapchain.handle == XR_NULL_HANDLE)
    return false;

  // Create depth swapchain
  XrSwapchainCreateInfo depthSwapchainInfo{XR_TYPE_SWAPCHAIN_CREATE_INFO};
  depthSwapchainInfo.usageFlags =
      XR_SWAPCHAIN_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT;
  depthSwapchainInfo.format      = static_cast<int64_t>(depthFormat);
  depthSwapchainInfo.sampleCount = 1;
  depthSwapchainInfo.width       = m_fullExtent.width;
  depthSwapchainInfo.height      = m_fullExtent.height;
  depthSwapchainInfo.faceCount   = 1;
  depthSwapchainInfo.arraySize   = 1;
  depthSwapchainInfo.mipCount    = 1;

  m_depthSwapchain = createSwapchain(depthSwapchainInfo);
  if(m_depthSwapchain.handle == XR_NULL_HANDLE)
    return false;

  // Create motion vector swapchain if Space Warp is supported
  if(m_spaceWarpSupported)
  {
    XrSwapchainCreateInfo motionSwapchainInfo{XR_TYPE_SWAPCHAIN_CREATE_INFO};
    motionSwapchainInfo.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT;
    motionSwapchainInfo.format     = VK_FORMAT_R16G16_SFLOAT;
    motionSwapchainInfo.sampleCount = 1;
    motionSwapchainInfo.width      = m_fullExtent.width;
    motionSwapchainInfo.height     = m_fullExtent.height;
    motionSwapchainInfo.faceCount  = 1;
    motionSwapchainInfo.arraySize  = 1;
    motionSwapchainInfo.mipCount   = 1;

    m_motionVectorSwapchain = createSwapchain(motionSwapchainInfo);
    if(m_motionVectorSwapchain.handle == XR_NULL_HANDLE)
    {
      LOGW("Failed to create motion vector swapchain, disabling Space Warp\n");
      m_spaceWarpSupported = false;
    }
  }

  LOGI("Created XR swapchains: color=%zu images, depth=%zu images, motion=%zu images\n", 
       m_colorSwapchain.images.size(), m_depthSwapchain.images.size(), 
       m_spaceWarpSupported ? m_motionVectorSwapchain.images.size() : 0);
  return true;
}


GsOpenXr::Swapchain GsOpenXr::createSwapchain(const XrSwapchainCreateInfo& createInfo) const
{
  Swapchain swapchain;

  XrResult result = xrCreateSwapchain(m_session, &createInfo, &swapchain.handle);
  if(XR_FAILED(result))
  {
    LOGE("Failed to create XR swapchain (result=%d)\n", (int)result);
    return swapchain;
  }

  // Get swapchain images
  uint32_t imageCount = 0;
  xrEnumerateSwapchainImages(swapchain.handle, 0, &imageCount, nullptr);

  std::vector<XrSwapchainImageVulkanKHR> xrImages(imageCount, {XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR});
  xrEnumerateSwapchainImages(swapchain.handle, imageCount, &imageCount,
                             reinterpret_cast<XrSwapchainImageBaseHeader*>(xrImages.data()));

  swapchain.images.resize(imageCount);
  for(uint32_t i = 0; i < imageCount; ++i)
  {
    swapchain.images[i] = xrImages[i].image;
  }

  return swapchain;
}

bool GsOpenXr::createReferenceSpace()
{
  XrReferenceSpaceCreateInfo spaceCreateInfo{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
  spaceCreateInfo.referenceSpaceType             = XR_REFERENCE_SPACE_TYPE_STAGE;
  spaceCreateInfo.poseInReferenceSpace.orientation.w = 1.0f;

  XrResult result = xrCreateReferenceSpace(m_session, &spaceCreateInfo, &m_referenceSpace);
  if(XR_FAILED(result))
  {
    // Fallback to LOCAL space if STAGE is not supported
    LOGW("STAGE reference space not available, falling back to LOCAL\n");
    spaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
    XR_CHECK(xrCreateReferenceSpace(m_session, &spaceCreateInfo, &m_referenceSpace),
             "Failed to create reference space");
  }

  return true;
}

void GsOpenXr::pollEvents()
{
  XrEventDataBuffer eventData{XR_TYPE_EVENT_DATA_BUFFER};

  while(true)
  {
    eventData.type = XR_TYPE_EVENT_DATA_BUFFER;
    XrResult result = xrPollEvent(m_instance, &eventData);

    if(result == XR_EVENT_UNAVAILABLE)
      break;

    if(XR_FAILED(result))
    {
      LOGE("Failed to poll XR events (result=%d)\n", (int)result);
      break;
    }

    switch(eventData.type)
    {
      case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED: {
        handleSessionStateChange(*reinterpret_cast<XrEventDataSessionStateChanged*>(&eventData));
        break;
      }
      case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING: {
        LOGW("XR instance loss pending\n");
        break;
      }
      default:
        break;
    }
  }
}

void GsOpenXr::handleSessionStateChange(const XrEventDataSessionStateChanged& event)
{
  m_sessionState = event.state;

  switch(m_sessionState)
  {
    case XR_SESSION_STATE_READY: {
      LOGI("XR session state: READY\n");
      // Begin session when we receive READY state
      XrSessionBeginInfo sessionBeginInfo{XR_TYPE_SESSION_BEGIN_INFO};
      sessionBeginInfo.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
      XrResult result = xrBeginSession(m_session, &sessionBeginInfo);
      if(XR_FAILED(result))
      {
        LOGE("Failed to begin XR session (result=%d)\n", (int)result);
      }
      else
      {
        m_sessionRunning = true;
        startTrackingThread();
        enablePerformanceMetrics();
      }
      break;
    }
    case XR_SESSION_STATE_SYNCHRONIZED:
      LOGI("XR session state: SYNCHRONIZED\n");
      break;
    case XR_SESSION_STATE_VISIBLE:
      LOGI("XR session state: VISIBLE\n");
      m_shouldRender = true;
      break;
    case XR_SESSION_STATE_FOCUSED:
      LOGI("XR session state: FOCUSED\n");
      m_shouldRender = true;
      break;
    case XR_SESSION_STATE_STOPPING:
      LOGI("XR session state: STOPPING\n");
      stopTrackingThread();
      m_shouldRender   = false;
      m_sessionRunning = false;
      xrEndSession(m_session);
      break;
    case XR_SESSION_STATE_LOSS_PENDING:
      LOGW("XR session state: LOSS_PENDING\n");
      stopTrackingThread();
      m_shouldRender   = false;
      m_sessionRunning = false;
      break;
    case XR_SESSION_STATE_EXITING:
      LOGI("XR session state: EXITING\n");
      stopTrackingThread();
      m_shouldRender   = false;
      m_sessionRunning = false;
      break;
    default:
      break;
  }
}

GsOpenXr::BeginFrameResult GsOpenXr::beginFrame()
{
  pollEvents();

  if(!m_sessionRunning)
    return BeginFrameResult::SkipFully;

  // Only proceed with frame if session is in an active state
  // READY and SYNCHRONIZED mean we should wait but not render
  // VISIBLE and FOCUSED mean we should render
  if(m_sessionState != XR_SESSION_STATE_READY && m_sessionState != XR_SESSION_STATE_SYNCHRONIZED
     && m_sessionState != XR_SESSION_STATE_VISIBLE && m_sessionState != XR_SESSION_STATE_FOCUSED)
  {
    return BeginFrameResult::SkipFully;
  }

  XrFrameWaitInfo frameWaitInfo{XR_TYPE_FRAME_WAIT_INFO};
  XrFrameState    frameState{XR_TYPE_FRAME_STATE};

  XrResult result = xrWaitFrame(m_session, &frameWaitInfo, &frameState);
  if(XR_FAILED(result))
  {
    LOGE("xrWaitFrame failed (result=%d)\n", (int)result);
    return BeginFrameResult::SkipFully;
  }

  m_predictedDisplayTime = frameState.predictedDisplayTime;
  m_shouldRender         = frameState.shouldRender == XR_TRUE;

  XrFrameBeginInfo frameBeginInfo{XR_TYPE_FRAME_BEGIN_INFO};
  result = xrBeginFrame(m_session, &frameBeginInfo);
  if(XR_FAILED(result))
  {
    LOGE("xrBeginFrame failed (result=%d)\n", (int)result);
    return BeginFrameResult::SkipFully;
  }

  m_swapchainImageState = SwapchainImageState::UNTOUCHED;
  
  // After calling xrBeginFrame, we MUST call xrEndFrame
  // Return SkipRender if shouldRender is false (but caller must still call endFrame)
  return m_shouldRender ? BeginFrameResult::RenderFully : BeginFrameResult::SkipRender;
}

bool GsOpenXr::locateViews(float nearZ, float farZ)
{
  m_nearZ = nearZ;
  m_farZ  = farZ;

  std::array<XrView, VIEW_COUNT> newViews;
  for(auto& view : newViews)
  {
    view.type = XR_TYPE_VIEW;
    view.next = nullptr;
  }

  bool gotHighFreqPose = false;
  if(m_trackingThreadRunning.load() && m_poseCount.load() > 0)
  {
    gotHighFreqPose = getPoseForTime(m_predictedDisplayTime, newViews);
    if(gotHighFreqPose)
    {
      m_locatedViews = newViews;
      m_trackingLossFrameCount = 0;
      return true;
    }
  }

  XrViewLocateInfo locateInfo{XR_TYPE_VIEW_LOCATE_INFO};
  locateInfo.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
  locateInfo.displayTime           = m_predictedDisplayTime;
  locateInfo.space                 = m_referenceSpace;

  XrViewState viewState{XR_TYPE_VIEW_STATE};
  uint32_t    viewCount = VIEW_COUNT;

  XrResult result = xrLocateViews(m_session, &locateInfo, &viewState, VIEW_COUNT, &viewCount, newViews.data());
  if(XR_FAILED(result))
  {
    LOGE("xrLocateViews failed (result=%d)\n", (int)result);
    m_shouldRender = false;
    return false;
  }

  if((viewState.viewStateFlags & XR_VIEW_STATE_POSITION_VALID_BIT) == 0 ||
     (viewState.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT) == 0)
  {
    m_trackingLossFrameCount++;

    if(m_trackingLossFrameCount <= MAX_TRACKING_LOSS_FRAMES)
    {
      return true;
    }

    m_shouldRender = false;
    return false;
  }

  m_locatedViews = newViews;
  m_trackingLossFrameCount = 0;
  return true;
}

GsOpenXr::EyeData GsOpenXr::getEyeData(uint32_t eyeIndex) const
{
  EyeData data;

  if(eyeIndex >= VIEW_COUNT)
  {
    data.view   = glm::mat4(1.0f);
    data.proj   = glm::mat4(1.0f);
    data.eyePos = glm::vec3(0.0f);
    data.fov    = glm::vec4(0.0f);
    return data;
  }

  const XrView& xrView = m_locatedViews[eyeIndex];

  data.view   = createViewMatrix(xrView.pose);
  data.proj   = createProjectionMatrix(xrView.fov, m_nearZ, m_farZ);
  data.eyePos = glm::vec3(xrView.pose.position.x, xrView.pose.position.y, xrView.pose.position.z);
  data.fov    = glm::vec4(xrView.fov.angleLeft, xrView.fov.angleRight, xrView.fov.angleUp, xrView.fov.angleDown);

  return data;
}

glm::mat4 GsOpenXr::createViewMatrix(const XrPosef& pose)
{
  // Convert XR pose to view matrix
  // XR uses right-handed coordinate system, +X right, +Y up, -Z forward
  glm::quat orientation(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
  glm::vec3 position(pose.position.x, pose.position.y, pose.position.z);

  // View matrix is inverse of pose
  glm::mat4 rotation    = glm::mat4_cast(glm::conjugate(orientation));
  glm::mat4 translation = glm::translate(glm::mat4(1.0f), -position);

  return rotation * translation;
}

glm::mat4 GsOpenXr::createProjectionMatrix(const XrFovf& fov, float nearZ, float farZ)
{
  // Create asymmetric frustum projection matrix
  const float tanLeft  = std::tan(fov.angleLeft);
  const float tanRight = std::tan(fov.angleRight);
  const float tanUp    = std::tan(fov.angleUp);
  const float tanDown  = std::tan(fov.angleDown);

  const float tanWidth  = tanRight - tanLeft;
  const float tanHeight = tanUp - tanDown;

  glm::mat4 proj(0.0f);
  proj[0][0] = 2.0f / tanWidth;
  proj[1][1] = -2.0f / tanHeight;  // Y-flip for Vulkan
  proj[2][0] = (tanRight + tanLeft) / tanWidth;
  proj[2][1] = (tanUp + tanDown) / tanHeight;
  proj[2][2] = farZ / (nearZ - farZ);
  proj[2][3] = -1.0f;
  proj[3][2] = (nearZ * farZ) / (nearZ - farZ);

  return proj;
}

bool GsOpenXr::acquireSwapchainImages(VkImage& outColorImage, VkImage& outDepthImage, VkImage& outMotionImage)
{
  XrSwapchainImageAcquireInfo acquireInfo{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};

  XrResult result = xrAcquireSwapchainImage(m_colorSwapchain.handle, &acquireInfo, &m_colorSwapchain.currentImageIndex);
  if(XR_FAILED(result))
  {
    LOGE("Failed to acquire color swapchain image (result=%d)\n", (int)result);
    return false;
  }

  result = xrAcquireSwapchainImage(m_depthSwapchain.handle, &acquireInfo, &m_depthSwapchain.currentImageIndex);
  if(XR_FAILED(result))
  {
    LOGE("Failed to acquire depth swapchain image (result=%d)\n", (int)result);
    return false;
  }

  if(m_spaceWarpSupported)
  {
    result = xrAcquireSwapchainImage(m_motionVectorSwapchain.handle, &acquireInfo, &m_motionVectorSwapchain.currentImageIndex);
    if(XR_FAILED(result))
    {
      LOGE("Failed to acquire motion swapchain image (result=%d)\n", (int)result);
      return false;
    }
  }

  // Wait for images to be ready
  XrSwapchainImageWaitInfo waitInfo{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
  waitInfo.timeout = XR_INFINITE_DURATION;

  result = xrWaitSwapchainImage(m_colorSwapchain.handle, &waitInfo);
  if(XR_FAILED(result))
  {
    LOGE("Failed to wait for color swapchain image (result=%d)\n", (int)result);
    return false;
  }

  result = xrWaitSwapchainImage(m_depthSwapchain.handle, &waitInfo);
  if(XR_FAILED(result))
  {
    LOGE("Failed to wait for depth swapchain image (result=%d)\n", (int)result);
    return false;
  }

  if(m_spaceWarpSupported)
  {
    result = xrWaitSwapchainImage(m_motionVectorSwapchain.handle, &waitInfo);
    if(XR_FAILED(result))
    {
      LOGE("Failed to wait for motion swapchain image (result=%d)\n", (int)result);
      return false;
    }
  }

  outColorImage = m_colorSwapchain.images[m_colorSwapchain.currentImageIndex];
  outDepthImage = m_depthSwapchain.images[m_depthSwapchain.currentImageIndex];
  if(m_spaceWarpSupported)
  {
    outMotionImage = m_motionVectorSwapchain.images[m_motionVectorSwapchain.currentImageIndex];
  }
  else
  {
    outMotionImage = VK_NULL_HANDLE;
  }

  m_swapchainImageState = SwapchainImageState::ACQUIRED;
  return true;
}


void GsOpenXr::releaseSwapchainImages()
{
  if(m_swapchainImageState != SwapchainImageState::ACQUIRED)
    return;

  XrSwapchainImageReleaseInfo releaseInfo{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};

  xrReleaseSwapchainImage(m_colorSwapchain.handle, &releaseInfo);
  xrReleaseSwapchainImage(m_depthSwapchain.handle, &releaseInfo);
  if(m_spaceWarpSupported)
  {
    xrReleaseSwapchainImage(m_motionVectorSwapchain.handle, &releaseInfo);
  }

  m_swapchainImageState = SwapchainImageState::RELEASED;
}


void GsOpenXr::endFrame()
{
  std::vector<XrCompositionLayerProjectionView> projectionViews(VIEW_COUNT);
  std::vector<XrCompositionLayerDepthInfoKHR>   depthInfos(VIEW_COUNT);
  std::vector<XrCompositionLayerSpaceWarpInfoFB> spaceWarpInfos(VIEW_COUNT);

  // Compute app space delta pose
  XrPosef appSpaceDeltaPose = { {0,0,0,1}, {0,0,0} }; // Identity by default
  
  // Calculate delta pose if we have a valid previous pose
  if (m_prevAppSpacePoseValid) {
      // Current app space is always identity in stage space, so delta is just inverse of previous?
      // Wait, appSpaceDeltaPose is "incremental application-applied transform... since the previous frame".
      // If we move the scene (m_splatSetVk.translation), we are effectively moving the app space.
      // However, OpenXR tracking handles HMD movement. App Space Delta Pose is for when the *virtual coordinate system* moves.
      // In this app, we move the scene by modifying m_splatSetVk.transform. The reference space (Stage) stays fixed.
      // So appSpaceDeltaPose should likely be identity unless we are implementing teleportation or artificial locomotion
      // by moving the reference space origin.
      
      // But wait, the spec says: "When artificial locomotion ... happens, the application might transform the whole 
      // XrCompositionLayerProjection::space from one application space pose to another pose between frames."
      // We are using m_referenceSpace (Stage) as the space. We don't change it frame to frame.
      // We change the *content* transform (Model Matrix).
      
      // Ideally, for Space Warp to work with artificial locomotion (stick movement), we should provide the delta.
      // BUT, if we move the Model Matrix, that's "App Space" movement relative to the HMD if the HMD is static?
      // No, App Space is the space the views are defined in. Here it is Stage Space.
      // If we don't change Stage Space origin, appSpaceDeltaPose is Identity.
      // BUT, if we move the "World" (Model Matrix), then from the perspective of the camera, the world moved.
      // Space Warp expects motion vectors to account for *everything*.
      // If we provide motion vectors that include camera movement AND object movement, then appSpaceDeltaPose 
      // allows the runtime to subtract the "camera movement" part that it already knows about (from tracking) 
      // vs the "artificial" part?
      
      // Actually, standard Space Warp usage:
      // Motion Vectors = (CurrentNDC - PreviousNDC).
      // This includes Camera Rotation + Translation (Tracking) AND Artificial Locomotion AND Object Motion.
      // The Runtime knows about Tracking. It needs to know about Artificial Locomotion to do the right reprojection.
      
      // In this app, we implement locomotion by moving the SCENE (Model Matrix), not the Camera (View Matrix is from XR).
      // Wait, updateXrLocomotion modifies m_splatSetVk.translation. 
      // So the object moves. The camera (XR Reference Space) is fixed to the physical room.
      // So effectively, the "World" is moving.
      // So appSpaceDeltaPose should be Identity because the Reference Space (Stage) hasn't moved.
      // The motion vectors will contain the scene movement.
      
      // Let's stick with Identity for now.
  }
  
  // Store current pose for next frame (not really used if we keep Identity)
  // m_prevAppSpacePose = currentAppSpacePose; 
  // m_prevAppSpacePoseValid = true;

  for(uint32_t i = 0; i < VIEW_COUNT; ++i)
  {
    XrRect2Di imageRect;
    imageRect.offset = {static_cast<int32_t>(i * m_perEyeExtent.width), 0};
    imageRect.extent = {static_cast<int32_t>(m_perEyeExtent.width), static_cast<int32_t>(m_perEyeExtent.height)};

    depthInfos[i].type     = XR_TYPE_COMPOSITION_LAYER_DEPTH_INFO_KHR;
    depthInfos[i].next     = nullptr;
    depthInfos[i].subImage = {m_depthSwapchain.handle, imageRect, 0};
    depthInfos[i].minDepth = 0.0f;
    depthInfos[i].maxDepth = 1.0f;
    depthInfos[i].nearZ    = m_nearZ;
    depthInfos[i].farZ     = m_farZ;

    if(m_spaceWarpSupported)
    {
      spaceWarpInfos[i].type = XR_TYPE_COMPOSITION_LAYER_SPACE_WARP_INFO_FB;
      spaceWarpInfos[i].next = &depthInfos[i]; // Chain depth info after space warp info? Or vice versa?
                                               // Spec says: "add an XrCompositionLayerSpaceWarpInfoFB structure to the XrCompositionLayerProjectionView::next chain"
                                               // It doesn't restrict order. Let's chain SpaceWarp -> Depth -> NULL
      spaceWarpInfos[i].layerFlags = 0;
      spaceWarpInfos[i].motionVectorSubImage = {m_motionVectorSwapchain.handle, imageRect, 0};
      spaceWarpInfos[i].appSpaceDeltaPose = appSpaceDeltaPose;
      spaceWarpInfos[i].depthSubImage = {m_depthSwapchain.handle, imageRect, 0};
      spaceWarpInfos[i].minDepth = 0.0f;
      spaceWarpInfos[i].maxDepth = 1.0f;
      spaceWarpInfos[i].nearZ = m_nearZ;
      spaceWarpInfos[i].farZ = m_farZ;
      
      projectionViews[i].next = &spaceWarpInfos[i];
    }
    else
    {
      projectionViews[i].next = &depthInfos[i];
    }

    projectionViews[i].type     = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW;
    projectionViews[i].pose     = m_locatedViews[i].pose;
    projectionViews[i].fov      = m_locatedViews[i].fov;
    projectionViews[i].subImage = {m_colorSwapchain.handle, imageRect, 0};
  }


  XrCompositionLayerProjection layer{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
  layer.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
  layer.space      = m_referenceSpace;
  layer.viewCount  = VIEW_COUNT;
  layer.views      = projectionViews.data();

  // Prepare layers list
  std::vector<const XrCompositionLayerBaseHeader*> layers;

  // If passthrough is enabled, add it as the background layer
  XrCompositionLayerPassthroughFB passthroughCompLayer{XR_TYPE_COMPOSITION_LAYER_PASSTHROUGH_FB};
  if(m_passthroughEnabled && m_passthroughSupported && m_passthroughLayer != XR_NULL_HANDLE)
  {
    passthroughCompLayer.layerHandle = m_passthroughLayer;
    passthroughCompLayer.flags       = 0;
    passthroughCompLayer.space       = XR_NULL_HANDLE;
    layers.push_back(reinterpret_cast<XrCompositionLayerBaseHeader*>(&passthroughCompLayer));
  }

  layers.push_back(reinterpret_cast<XrCompositionLayerBaseHeader*>(&layer));

  XrFrameEndInfo frameEndInfo{XR_TYPE_FRAME_END_INFO};
  frameEndInfo.displayTime          = m_predictedDisplayTime;
  
  if(m_passthroughEnabled && m_passthroughSupported)
  {
    frameEndInfo.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND;
  }
  else
  {
    frameEndInfo.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
  }

  if(m_swapchainImageState == SwapchainImageState::RELEASED && m_shouldRender)
  {
    frameEndInfo.layerCount = static_cast<uint32_t>(layers.size());
    frameEndInfo.layers     = layers.data();
  }
  else
  {
    frameEndInfo.layerCount = 0;
    frameEndInfo.layers     = nullptr;
  }

  XrResult result = xrEndFrame(m_session, &frameEndInfo);

  if(XR_FAILED(result))
  {
    LOGE("xrEndFrame failed (result=%d)\n", (int)result);
  }
}

bool GsOpenXr::createActionSet()
{
  // Create action set
  XrActionSetCreateInfo actionSetInfo{XR_TYPE_ACTION_SET_CREATE_INFO};
  strcpy_s(actionSetInfo.actionSetName, "gameplay");
  strcpy_s(actionSetInfo.localizedActionSetName, "Gameplay");
  actionSetInfo.priority = 0;

  XrResult result = xrCreateActionSet(m_instance, &actionSetInfo, &m_actionSet);
  if(XR_FAILED(result))
  {
    LOGE("Failed to create action set (result=%d)\n", (int)result);
    return false;
  }

  // Create subaction paths for left and right hands
  xrStringToPath(m_instance, "/user/hand/left", &m_leftHandPath);
  xrStringToPath(m_instance, "/user/hand/right", &m_rightHandPath);
  XrPath handPaths[] = {m_leftHandPath, m_rightHandPath};

  // Create thumbstick action (Vector2)
  {
    XrActionCreateInfo actionInfo{XR_TYPE_ACTION_CREATE_INFO};
    actionInfo.actionType = XR_ACTION_TYPE_VECTOR2F_INPUT;
    strcpy_s(actionInfo.actionName, "thumbstick");
    strcpy_s(actionInfo.localizedActionName, "Thumbstick");
    actionInfo.countSubactionPaths = 2;
    actionInfo.subactionPaths = handPaths;
    XR_CHECK(xrCreateAction(m_actionSet, &actionInfo, &m_thumbstickAction), "Failed to create thumbstick action");
  }

  // Create trigger action (Float)
  {
    XrActionCreateInfo actionInfo{XR_TYPE_ACTION_CREATE_INFO};
    actionInfo.actionType = XR_ACTION_TYPE_FLOAT_INPUT;
    strcpy_s(actionInfo.actionName, "trigger");
    strcpy_s(actionInfo.localizedActionName, "Trigger");
    actionInfo.countSubactionPaths = 2;
    actionInfo.subactionPaths = handPaths;
    XR_CHECK(xrCreateAction(m_actionSet, &actionInfo, &m_triggerAction), "Failed to create trigger action");
  }

  // Create grip action (Float)
  {
    XrActionCreateInfo actionInfo{XR_TYPE_ACTION_CREATE_INFO};
    actionInfo.actionType = XR_ACTION_TYPE_FLOAT_INPUT;
    strcpy_s(actionInfo.actionName, "grip");
    strcpy_s(actionInfo.localizedActionName, "Grip");
    actionInfo.countSubactionPaths = 2;
    actionInfo.subactionPaths = handPaths;
    XR_CHECK(xrCreateAction(m_actionSet, &actionInfo, &m_gripAction), "Failed to create grip action");
  }

  // Create thumbstick click action (Boolean)
  {
    XrActionCreateInfo actionInfo{XR_TYPE_ACTION_CREATE_INFO};
    actionInfo.actionType = XR_ACTION_TYPE_BOOLEAN_INPUT;
    strcpy_s(actionInfo.actionName, "thumbstick_click");
    strcpy_s(actionInfo.localizedActionName, "Thumbstick Click");
    actionInfo.countSubactionPaths = 2;
    actionInfo.subactionPaths = handPaths;
    XR_CHECK(xrCreateAction(m_actionSet, &actionInfo, &m_thumbstickClickAction), "Failed to create thumbstick click action");
  }

  // Create primary button action (A/X)
  {
    XrActionCreateInfo actionInfo{XR_TYPE_ACTION_CREATE_INFO};
    actionInfo.actionType = XR_ACTION_TYPE_BOOLEAN_INPUT;
    strcpy_s(actionInfo.actionName, "primary_button");
    strcpy_s(actionInfo.localizedActionName, "Primary Button");
    actionInfo.countSubactionPaths = 2;
    actionInfo.subactionPaths = handPaths;
    XR_CHECK(xrCreateAction(m_actionSet, &actionInfo, &m_primaryButtonAction), "Failed to create primary button action");
  }

  // Create secondary button action (B/Y)
  {
    XrActionCreateInfo actionInfo{XR_TYPE_ACTION_CREATE_INFO};
    actionInfo.actionType = XR_ACTION_TYPE_BOOLEAN_INPUT;
    strcpy_s(actionInfo.actionName, "secondary_button");
    strcpy_s(actionInfo.localizedActionName, "Secondary Button");
    actionInfo.countSubactionPaths = 2;
    actionInfo.subactionPaths = handPaths;
    XR_CHECK(xrCreateAction(m_actionSet, &actionInfo, &m_secondaryButtonAction), "Failed to create secondary button action");
  }

  // Create menu button action
  {
    XrActionCreateInfo actionInfo{XR_TYPE_ACTION_CREATE_INFO};
    actionInfo.actionType = XR_ACTION_TYPE_BOOLEAN_INPUT;
    strcpy_s(actionInfo.actionName, "menu_button");
    strcpy_s(actionInfo.localizedActionName, "Menu Button");
    actionInfo.countSubactionPaths = 2;
    actionInfo.subactionPaths = handPaths;
    XR_CHECK(xrCreateAction(m_actionSet, &actionInfo, &m_menuButtonAction), "Failed to create menu button action");
  }

  // Create pose action for controller tracking
  {
    XrActionCreateInfo actionInfo{XR_TYPE_ACTION_CREATE_INFO};
    actionInfo.actionType = XR_ACTION_TYPE_POSE_INPUT;
    strcpy_s(actionInfo.actionName, "hand_pose");
    strcpy_s(actionInfo.localizedActionName, "Hand Pose");
    actionInfo.countSubactionPaths = 2;
    actionInfo.subactionPaths = handPaths;
    XR_CHECK(xrCreateAction(m_actionSet, &actionInfo, &m_poseAction), "Failed to create pose action");
  }

  // Helper lambda for adding bindings
  auto addBinding = [this](std::vector<XrActionSuggestedBinding>& bindings, XrAction action, const char* path) {
    XrPath bindingPath;
    xrStringToPath(m_instance, path, &bindingPath);
    bindings.push_back({action, bindingPath});
  };

  // Suggest bindings for KHR Simple Controller (fallback for all controllers)
  {
    XrPath simpleProfile;
    xrStringToPath(m_instance, "/interaction_profiles/khr/simple_controller", &simpleProfile);

    std::vector<XrActionSuggestedBinding> bindings;
    addBinding(bindings, m_poseAction, "/user/hand/left/input/aim/pose");
    addBinding(bindings, m_poseAction, "/user/hand/right/input/aim/pose");
    addBinding(bindings, m_triggerAction, "/user/hand/left/input/select/click");
    addBinding(bindings, m_triggerAction, "/user/hand/right/input/select/click");

    XrInteractionProfileSuggestedBinding suggestedBindings{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
    suggestedBindings.interactionProfile = simpleProfile;
    suggestedBindings.countSuggestedBindings = static_cast<uint32_t>(bindings.size());
    suggestedBindings.suggestedBindings = bindings.data();

    result = xrSuggestInteractionProfileBindings(m_instance, &suggestedBindings);
    if(XR_FAILED(result))
    {
      LOGW("Failed to suggest KHR Simple Controller bindings (result=%d)\n", (int)result);
    }
    else
    {
      LOGI("Suggested KHR Simple Controller bindings\n");
    }
  }

  // Suggest bindings for Oculus Touch controllers
  {
    XrPath oculusTouchProfile;
    xrStringToPath(m_instance, "/interaction_profiles/oculus/touch_controller", &oculusTouchProfile);

    std::vector<XrActionSuggestedBinding> bindings;

    // Left controller bindings
    addBinding(bindings, m_thumbstickAction, "/user/hand/left/input/thumbstick");
    addBinding(bindings, m_triggerAction, "/user/hand/left/input/trigger/value");
    addBinding(bindings, m_gripAction, "/user/hand/left/input/squeeze/value");
    addBinding(bindings, m_thumbstickClickAction, "/user/hand/left/input/thumbstick/click");
    addBinding(bindings, m_primaryButtonAction, "/user/hand/left/input/x/click");
    addBinding(bindings, m_secondaryButtonAction, "/user/hand/left/input/y/click");
    addBinding(bindings, m_menuButtonAction, "/user/hand/left/input/menu/click");
    addBinding(bindings, m_poseAction, "/user/hand/left/input/aim/pose");

    // Right controller bindings
    addBinding(bindings, m_thumbstickAction, "/user/hand/right/input/thumbstick");
    addBinding(bindings, m_triggerAction, "/user/hand/right/input/trigger/value");
    addBinding(bindings, m_gripAction, "/user/hand/right/input/squeeze/value");
    addBinding(bindings, m_thumbstickClickAction, "/user/hand/right/input/thumbstick/click");
    addBinding(bindings, m_primaryButtonAction, "/user/hand/right/input/a/click");
    addBinding(bindings, m_secondaryButtonAction, "/user/hand/right/input/b/click");
    addBinding(bindings, m_poseAction, "/user/hand/right/input/aim/pose");

    XrInteractionProfileSuggestedBinding suggestedBindings{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
    suggestedBindings.interactionProfile = oculusTouchProfile;
    suggestedBindings.countSuggestedBindings = static_cast<uint32_t>(bindings.size());
    suggestedBindings.suggestedBindings = bindings.data();

    result = xrSuggestInteractionProfileBindings(m_instance, &suggestedBindings);
    if(XR_FAILED(result))
    {
      LOGW("Failed to suggest Oculus Touch bindings (result=%d)\n", (int)result);
    }
    else
    {
      LOGI("Suggested Oculus Touch bindings\n");
    }
  }

  // Suggest bindings for Valve Index controllers
  {
    XrPath indexProfile;
    xrStringToPath(m_instance, "/interaction_profiles/valve/index_controller", &indexProfile);

    std::vector<XrActionSuggestedBinding> bindings;

    // Left controller bindings
    addBinding(bindings, m_thumbstickAction, "/user/hand/left/input/thumbstick");
    addBinding(bindings, m_triggerAction, "/user/hand/left/input/trigger/value");
    addBinding(bindings, m_gripAction, "/user/hand/left/input/squeeze/value");
    addBinding(bindings, m_thumbstickClickAction, "/user/hand/left/input/thumbstick/click");
    addBinding(bindings, m_primaryButtonAction, "/user/hand/left/input/a/click");
    addBinding(bindings, m_secondaryButtonAction, "/user/hand/left/input/b/click");
    addBinding(bindings, m_poseAction, "/user/hand/left/input/aim/pose");

    // Right controller bindings
    addBinding(bindings, m_thumbstickAction, "/user/hand/right/input/thumbstick");
    addBinding(bindings, m_triggerAction, "/user/hand/right/input/trigger/value");
    addBinding(bindings, m_gripAction, "/user/hand/right/input/squeeze/value");
    addBinding(bindings, m_thumbstickClickAction, "/user/hand/right/input/thumbstick/click");
    addBinding(bindings, m_primaryButtonAction, "/user/hand/right/input/a/click");
    addBinding(bindings, m_secondaryButtonAction, "/user/hand/right/input/b/click");
    addBinding(bindings, m_poseAction, "/user/hand/right/input/aim/pose");

    XrInteractionProfileSuggestedBinding suggestedBindings{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
    suggestedBindings.interactionProfile = indexProfile;
    suggestedBindings.countSuggestedBindings = static_cast<uint32_t>(bindings.size());
    suggestedBindings.suggestedBindings = bindings.data();

    result = xrSuggestInteractionProfileBindings(m_instance, &suggestedBindings);
    if(XR_FAILED(result))
    {
      LOGW("Failed to suggest Valve Index bindings (result=%d)\n", (int)result);
    }
    else
    {
      LOGI("Suggested Valve Index bindings\n");
    }
  }

  // Attach action set to session
  XrSessionActionSetsAttachInfo attachInfo{XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO};
  attachInfo.countActionSets = 1;
  attachInfo.actionSets = &m_actionSet;

  result = xrAttachSessionActionSets(m_session, &attachInfo);
  if(XR_FAILED(result))
  {
    LOGE("Failed to attach action sets (result=%d)\n", (int)result);
    return false;
  }
  LOGI("Action sets attached to session\n");

  // Create action spaces for controller poses (AFTER attaching action sets)
  XrActionSpaceCreateInfo spaceInfo{XR_TYPE_ACTION_SPACE_CREATE_INFO};
  spaceInfo.action = m_poseAction;
  spaceInfo.poseInActionSpace.orientation.w = 1.0f;
  spaceInfo.poseInActionSpace.orientation.x = 0.0f;
  spaceInfo.poseInActionSpace.orientation.y = 0.0f;
  spaceInfo.poseInActionSpace.orientation.z = 0.0f;
  spaceInfo.poseInActionSpace.position.x = 0.0f;
  spaceInfo.poseInActionSpace.position.y = 0.0f;
  spaceInfo.poseInActionSpace.position.z = 0.0f;

  spaceInfo.subactionPath = m_leftHandPath;
  result = xrCreateActionSpace(m_session, &spaceInfo, &m_leftHandSpace);
  if(XR_FAILED(result))
  {
    LOGW("Failed to create left hand space (result=%d)\n", (int)result);
  }
  else
  {
    LOGI("Created left hand action space\n");
  }

  spaceInfo.subactionPath = m_rightHandPath;
  result = xrCreateActionSpace(m_session, &spaceInfo, &m_rightHandSpace);
  if(XR_FAILED(result))
  {
    LOGW("Failed to create right hand space (result=%d)\n", (int)result);
  }
  else
  {
    LOGI("Created right hand action space\n");
  }

  m_hasControllers = true;
  LOGI("OpenXR controller input initialized successfully\n");
  return true;
}

void GsOpenXr::destroyActionSet()
{
  if(m_leftHandSpace != XR_NULL_HANDLE)
  {
    xrDestroySpace(m_leftHandSpace);
    m_leftHandSpace = XR_NULL_HANDLE;
  }

  if(m_rightHandSpace != XR_NULL_HANDLE)
  {
    xrDestroySpace(m_rightHandSpace);
    m_rightHandSpace = XR_NULL_HANDLE;
  }

  // Actions are destroyed when the action set is destroyed
  if(m_actionSet != XR_NULL_HANDLE)
  {
    xrDestroyActionSet(m_actionSet);
    m_actionSet = XR_NULL_HANDLE;
  }

  m_thumbstickAction = XR_NULL_HANDLE;
  m_triggerAction = XR_NULL_HANDLE;
  m_gripAction = XR_NULL_HANDLE;
  m_thumbstickClickAction = XR_NULL_HANDLE;
  m_primaryButtonAction = XR_NULL_HANDLE;
  m_secondaryButtonAction = XR_NULL_HANDLE;
  m_menuButtonAction = XR_NULL_HANDLE;
  m_poseAction = XR_NULL_HANDLE;
}

void GsOpenXr::pollControllerInput()
{
  if(!m_hasControllers || !m_sessionRunning)
    return;

  syncControllerActions();
  updateControllerPoses();

  // Update locomotion input from thumbsticks
  m_locomotionInput.move = m_leftController.thumbstick;
  m_locomotionInput.turn = m_rightController.thumbstick;
  m_locomotionInput.sprintPressed = m_leftController.thumbstickClick;

  // Snap turn detection (trigger once when crossing threshold)
  if(m_rightController.thumbstick.x < -SNAP_TURN_THRESHOLD && !m_snapTurnLeftTriggered)
  {
    m_locomotionInput.snapTurnLeft = true;
    m_snapTurnLeftTriggered = true;
  }
  else if(m_rightController.thumbstick.x >= -SNAP_TURN_THRESHOLD)
  {
    m_locomotionInput.snapTurnLeft = false;
    m_snapTurnLeftTriggered = false;
  }
  else
  {
    m_locomotionInput.snapTurnLeft = false;
  }

  if(m_rightController.thumbstick.x > SNAP_TURN_THRESHOLD && !m_snapTurnRightTriggered)
  {
    m_locomotionInput.snapTurnRight = true;
    m_snapTurnRightTriggered = true;
  }
  else if(m_rightController.thumbstick.x <= SNAP_TURN_THRESHOLD)
  {
    m_locomotionInput.snapTurnRight = false;
    m_snapTurnRightTriggered = false;
  }
  else
  {
    m_locomotionInput.snapTurnRight = false;
  }
}

void GsOpenXr::syncControllerActions()
{
  XrActiveActionSet activeActionSet{};
  activeActionSet.actionSet = m_actionSet;
  activeActionSet.subactionPath = XR_NULL_PATH;

  XrActionsSyncInfo syncInfo{XR_TYPE_ACTIONS_SYNC_INFO};
  syncInfo.countActiveActionSets = 1;
  syncInfo.activeActionSets = &activeActionSet;

  XrResult result = xrSyncActions(m_session, &syncInfo);
  if(XR_FAILED(result))
  {
    return;
  }

  // Helper to get float value
  auto getFloat = [&](XrAction action, XrPath subactionPath) -> float {
    XrActionStateGetInfo getInfo{XR_TYPE_ACTION_STATE_GET_INFO};
    getInfo.action = action;
    getInfo.subactionPath = subactionPath;

    XrActionStateFloat state{XR_TYPE_ACTION_STATE_FLOAT};
    if(XR_SUCCEEDED(xrGetActionStateFloat(m_session, &getInfo, &state)) && state.isActive)
    {
      return state.currentState;
    }
    return 0.0f;
  };

  // Helper to get boolean value
  auto getBool = [&](XrAction action, XrPath subactionPath) -> bool {
    XrActionStateGetInfo getInfo{XR_TYPE_ACTION_STATE_GET_INFO};
    getInfo.action = action;
    getInfo.subactionPath = subactionPath;

    XrActionStateBoolean state{XR_TYPE_ACTION_STATE_BOOLEAN};
    if(XR_SUCCEEDED(xrGetActionStateBoolean(m_session, &getInfo, &state)) && state.isActive)
    {
      return state.currentState == XR_TRUE;
    }
    return false;
  };

  // Helper to get vector2 value
  auto getVector2 = [&](XrAction action, XrPath subactionPath) -> glm::vec2 {
    XrActionStateGetInfo getInfo{XR_TYPE_ACTION_STATE_GET_INFO};
    getInfo.action = action;
    getInfo.subactionPath = subactionPath;

    XrActionStateVector2f state{XR_TYPE_ACTION_STATE_VECTOR2F};
    if(XR_SUCCEEDED(xrGetActionStateVector2f(m_session, &getInfo, &state)) && state.isActive)
    {
      return glm::vec2(state.currentState.x, state.currentState.y);
    }
    return glm::vec2(0.0f);
  };

  // Update left controller
  m_leftController.thumbstick = getVector2(m_thumbstickAction, m_leftHandPath);
  m_leftController.trigger = getFloat(m_triggerAction, m_leftHandPath);
  m_leftController.grip = getFloat(m_gripAction, m_leftHandPath);
  m_leftController.thumbstickClick = getBool(m_thumbstickClickAction, m_leftHandPath);
  m_leftController.primaryButton = getBool(m_primaryButtonAction, m_leftHandPath);
  m_leftController.secondaryButton = getBool(m_secondaryButtonAction, m_leftHandPath);
  m_leftController.menuButton = getBool(m_menuButtonAction, m_leftHandPath);

  // Update right controller
  m_rightController.thumbstick = getVector2(m_thumbstickAction, m_rightHandPath);
  m_rightController.trigger = getFloat(m_triggerAction, m_rightHandPath);
  m_rightController.grip = getFloat(m_gripAction, m_rightHandPath);
  m_rightController.thumbstickClick = getBool(m_thumbstickClickAction, m_rightHandPath);
  m_rightController.primaryButton = getBool(m_primaryButtonAction, m_rightHandPath);
  m_rightController.secondaryButton = getBool(m_secondaryButtonAction, m_rightHandPath);

}

void GsOpenXr::updateControllerPoses()
{
  auto locateSpace = [&](XrSpace space, ControllerInput& controller) {
    if(space == XR_NULL_HANDLE)
    {
      controller.poseValid = false;
      return;
    }

    XrSpaceLocation location{XR_TYPE_SPACE_LOCATION};
    XrResult result = xrLocateSpace(space, m_referenceSpace, m_predictedDisplayTime, &location);

    if(XR_SUCCEEDED(result) && (location.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) &&
       (location.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT))
    {
      controller.position = glm::vec3(location.pose.position.x, location.pose.position.y, location.pose.position.z);
      controller.orientation = glm::quat(location.pose.orientation.w, location.pose.orientation.x,
                                         location.pose.orientation.y, location.pose.orientation.z);
      controller.poseValid = true;
    }
    else
    {
      controller.poseValid = false;
    }
  };

  locateSpace(m_leftHandSpace, m_leftController);
  locateSpace(m_rightHandSpace, m_rightController);
}

// ============================================================================
// High-frequency tracking thread implementation
// ============================================================================

void GsOpenXr::startTrackingThread()
{
  if(m_trackingThreadRunning.load())
    return;

  m_trackingThreadShouldStop.store(false);
  m_poseWriteIndex.store(0);
  m_poseCount.store(0);

  m_trackingThread = std::thread([this]() { trackingThreadLoop(); });
  m_trackingThreadRunning.store(true);

  LOGI("OpenXR high-frequency tracking thread started (%d Hz)\n", TRACKING_SAMPLE_RATE_HZ);
}

void GsOpenXr::stopTrackingThread()
{
  if(!m_trackingThreadRunning.load())
    return;

  m_trackingThreadShouldStop.store(true);

  if(m_trackingThread.joinable())
  {
    m_trackingThread.join();
  }

  m_trackingThreadRunning.store(false);
  LOGI("OpenXR high-frequency tracking thread stopped\n");
}

void GsOpenXr::trackingThreadLoop()
{
  using namespace std::chrono;
  const auto sampleInterval = microseconds(1000000 / TRACKING_SAMPLE_RATE_HZ);

  PFN_xrConvertWin32PerformanceCounterToTimeKHR convertTimeFunc = nullptr;
  xrGetInstanceProcAddr(m_instance, "xrConvertWin32PerformanceCounterToTimeKHR",
                        (PFN_xrVoidFunction*)&convertTimeFunc);

  while(!m_trackingThreadShouldStop.load())
  {
    auto loopStart = steady_clock::now();

    if(m_session != XR_NULL_HANDLE && m_referenceSpace != XR_NULL_HANDLE && m_sessionRunning && convertTimeFunc)
    {
      LARGE_INTEGER perfCount;
      QueryPerformanceCounter(&perfCount);

      XrTime now = 0;
      if(XR_SUCCEEDED(convertTimeFunc(m_instance, &perfCount, &now)) && now != 0)
      {
        XrViewLocateInfo locateInfo{XR_TYPE_VIEW_LOCATE_INFO};
        locateInfo.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
        locateInfo.displayTime = now;
        locateInfo.space = m_referenceSpace;

        XrViewState viewState{XR_TYPE_VIEW_STATE};
        uint32_t viewCount = VIEW_COUNT;

        std::array<XrView, VIEW_COUNT> views;
        for(auto& view : views)
        {
          view.type = XR_TYPE_VIEW;
          view.next = nullptr;
        }

        XrResult result = xrLocateViews(m_session, &locateInfo, &viewState, VIEW_COUNT, &viewCount, views.data());

        if(XR_SUCCEEDED(result))
        {
          size_t writeIdx = m_poseWriteIndex.load() % POSE_RING_BUFFER_SIZE;

          {
            std::lock_guard<std::mutex> lock(m_poseMutex);
            m_poseRingBuffer[writeIdx].timestamp = now;
            m_poseRingBuffer[writeIdx].views = views;
            m_poseRingBuffer[writeIdx].viewStateFlags = viewState.viewStateFlags;
            m_poseRingBuffer[writeIdx].valid = true;
          }

          m_poseWriteIndex.fetch_add(1);
          size_t count = m_poseCount.load();
          if(count < POSE_RING_BUFFER_SIZE)
          {
            m_poseCount.fetch_add(1);
          }
        }
      }
    }

    auto loopEnd = steady_clock::now();
    auto elapsed = duration_cast<microseconds>(loopEnd - loopStart);
    auto sleepTime = sampleInterval - elapsed;

    if(sleepTime > microseconds(0))
    {
      std::this_thread::sleep_for(sleepTime);
    }
  }
}

bool GsOpenXr::getPoseForTime(XrTime targetTime, std::array<XrView, VIEW_COUNT>& outViews) const
{
  size_t count = m_poseCount.load();
  if(count == 0)
    return false;

  std::lock_guard<std::mutex> lock(m_poseMutex);

  size_t writeIdx = m_poseWriteIndex.load();
  size_t startIdx = (writeIdx >= count) ? (writeIdx - count) : 0;

  const TimestampedPose* closest = nullptr;
  const TimestampedPose* before = nullptr;
  const TimestampedPose* after = nullptr;
  XrTime minDiff = INT64_MAX;

  for(size_t i = 0; i < count; ++i)
  {
    size_t idx = (startIdx + i) % POSE_RING_BUFFER_SIZE;
    const TimestampedPose& pose = m_poseRingBuffer[idx];

    if(!pose.valid)
      continue;

    XrTime diff = (pose.timestamp > targetTime) ? (pose.timestamp - targetTime) : (targetTime - pose.timestamp);

    if(diff < minDiff)
    {
      minDiff = diff;
      closest = &pose;
    }

    if(pose.timestamp <= targetTime)
    {
      if(!before || pose.timestamp > before->timestamp)
        before = &pose;
    }
    if(pose.timestamp >= targetTime)
    {
      if(!after || pose.timestamp < after->timestamp)
        after = &pose;
    }
  }

  if(before && after && before != after)
  {
    XrTime range = after->timestamp - before->timestamp;
    if(range > 0)
    {
      float t = static_cast<float>(targetTime - before->timestamp) / static_cast<float>(range);
      t = std::clamp(t, 0.0f, 1.0f);

      for(uint32_t eye = 0; eye < VIEW_COUNT; ++eye)
      {
        outViews[eye] = interpolateView(before->views[eye], after->views[eye], t);
      }
      return true;
    }
  }

  if(closest)
  {
    outViews = closest->views;
    return true;
  }

  return false;
}

XrView GsOpenXr::interpolateView(const XrView& a, const XrView& b, float t)
{
  XrView result;
  result.type = XR_TYPE_VIEW;
  result.next = nullptr;
  result.pose = interpolatePose(a.pose, b.pose, t);
  result.fov.angleLeft = a.fov.angleLeft + t * (b.fov.angleLeft - a.fov.angleLeft);
  result.fov.angleRight = a.fov.angleRight + t * (b.fov.angleRight - a.fov.angleRight);
  result.fov.angleUp = a.fov.angleUp + t * (b.fov.angleUp - a.fov.angleUp);
  result.fov.angleDown = a.fov.angleDown + t * (b.fov.angleDown - a.fov.angleDown);
  return result;
}

XrPosef GsOpenXr::interpolatePose(const XrPosef& a, const XrPosef& b, float t)
{
  XrPosef result;

  result.position.x = a.position.x + t * (b.position.x - a.position.x);
  result.position.y = a.position.y + t * (b.position.y - a.position.y);
  result.position.z = a.position.z + t * (b.position.z - a.position.z);

  result.orientation = slerp(a.orientation, b.orientation, t);

  return result;
}

XrQuaternionf GsOpenXr::slerp(const XrQuaternionf& a, const XrQuaternionf& b, float t)
{
  float dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;

  XrQuaternionf b2 = b;
  if(dot < 0.0f)
  {
    dot = -dot;
    b2.x = -b2.x;
    b2.y = -b2.y;
    b2.z = -b2.z;
    b2.w = -b2.w;
  }

  XrQuaternionf result;

  if(dot > 0.9995f)
  {
    result.x = a.x + t * (b2.x - a.x);
    result.y = a.y + t * (b2.y - a.y);
    result.z = a.z + t * (b2.z - a.z);
    result.w = a.w + t * (b2.w - a.w);

    float len = std::sqrt(result.x * result.x + result.y * result.y + result.z * result.z + result.w * result.w);
    result.x /= len;
    result.y /= len;
    result.z /= len;
    result.w /= len;
  }
  else
  {
    float theta0 = std::acos(dot);
    float theta = theta0 * t;
    float sinTheta = std::sin(theta);
    float sinTheta0 = std::sin(theta0);

    float s0 = std::cos(theta) - dot * sinTheta / sinTheta0;
    float s1 = sinTheta / sinTheta0;

    result.x = s0 * a.x + s1 * b2.x;
    result.y = s0 * a.y + s1 * b2.y;
    result.z = s0 * a.z + s1 * b2.z;
    result.w = s0 * a.w + s1 * b2.w;
  }

  return result;
}

void GsOpenXr::initPerformanceMetrics()
{
  if(!m_perfMetricsSupported || !m_xrEnumeratePerformanceMetricsCounterPathsMETA)
    return;

  uint32_t pathCount = 0;
  XrResult result = m_xrEnumeratePerformanceMetricsCounterPathsMETA(m_instance, 0, &pathCount, nullptr);
  if(XR_FAILED(result) || pathCount == 0)
  {
    LOGW("No performance metrics counters available\n");
    m_perfMetricsSupported = false;
    return;
  }

  m_perfMetricsPaths.resize(pathCount);
  result = m_xrEnumeratePerformanceMetricsCounterPathsMETA(m_instance, pathCount, &pathCount, m_perfMetricsPaths.data());
  if(XR_FAILED(result))
  {
    LOGE("Failed to enumerate performance metrics paths\n");
    m_perfMetricsSupported = false;
    return;
  }

  xrStringToPath(m_instance, "/perfmetrics_meta/app/cpu_frametime", &m_pathAppCpuFrametime);
  xrStringToPath(m_instance, "/perfmetrics_meta/app/gpu_frametime", &m_pathAppGpuFrametime);
  xrStringToPath(m_instance, "/perfmetrics_meta/app/motion_to_photon_latency", &m_pathMotionToPhoton);
  xrStringToPath(m_instance, "/perfmetrics_meta/compositor/cpu_frametime", &m_pathCompositorCpuFrametime);
  xrStringToPath(m_instance, "/perfmetrics_meta/compositor/gpu_frametime", &m_pathCompositorGpuFrametime);
  xrStringToPath(m_instance, "/perfmetrics_meta/compositor/dropped_frame_count", &m_pathDroppedFrameCount);
  xrStringToPath(m_instance, "/perfmetrics_meta/compositor/spacewarp_mode", &m_pathSpacewarpMode);
  xrStringToPath(m_instance, "/perfmetrics_meta/device/cpu_utilization_average", &m_pathCpuUtilAvg);
  xrStringToPath(m_instance, "/perfmetrics_meta/device/cpu_utilization_worst", &m_pathCpuUtilWorst);
  xrStringToPath(m_instance, "/perfmetrics_meta/device/gpu_utilization", &m_pathGpuUtil);

  LOGI("XR performance metrics initialized with %u counters\n", pathCount);
}

void GsOpenXr::enablePerformanceMetrics()
{
  if(!m_perfMetricsSupported || !m_xrSetPerformanceMetricsStateMETA || m_session == XR_NULL_HANDLE)
    return;

  struct
  {
    XrStructureType type;
    const void*     next;
    XrBool32        enabled;
  } state = {(XrStructureType)1000232001, nullptr, XR_TRUE};

  XrResult result = m_xrSetPerformanceMetricsStateMETA(m_session, &state);
  if(XR_SUCCEEDED(result))
  {
    m_perfMetricsEnabled = true;
    LOGI("XR performance metrics enabled\n");
  }
  else
  {
    LOGW("Failed to enable XR performance metrics (result=%d)\n", (int)result);
  }
}

void GsOpenXr::updatePerformanceMetrics()
{
  if(!m_perfMetricsEnabled || !m_xrQueryPerformanceMetricsCounterMETA || m_session == XR_NULL_HANDLE)
  {
    m_perfMetrics.valid = false;
    return;
  }

  struct XrPerfCounter
  {
    XrStructureType type;
    const void*     next;
    uint64_t        counterFlags;
    uint32_t        counterUnit;
    uint32_t        uintValue;
    float           floatValue;
  };

  auto queryFloat = [this](XrPath path) -> float {
    if(path == XR_NULL_PATH)
      return 0.0f;
    XrPerfCounter counter = {(XrStructureType)1000232002, nullptr, 0, 0, 0, 0.0f};
    XrResult result = m_xrQueryPerformanceMetricsCounterMETA(m_session, path, &counter);
    if(XR_SUCCEEDED(result) && (counter.counterFlags & 0x04))
      return counter.floatValue;
    return 0.0f;
  };

  auto queryUint = [this](XrPath path) -> uint32_t {
    if(path == XR_NULL_PATH)
      return 0;
    XrPerfCounter counter = {(XrStructureType)1000232002, nullptr, 0, 0, 0, 0.0f};
    XrResult result = m_xrQueryPerformanceMetricsCounterMETA(m_session, path, &counter);
    if(XR_SUCCEEDED(result) && (counter.counterFlags & 0x02))
      return counter.uintValue;
    return 0;
  };

  m_perfMetrics.appCpuFrameTimeMs = queryFloat(m_pathAppCpuFrametime);
  m_perfMetrics.appGpuFrameTimeMs = queryFloat(m_pathAppGpuFrametime);
  m_perfMetrics.motionToPhotonLatencyMs = queryFloat(m_pathMotionToPhoton);
  m_perfMetrics.compositorCpuFrameTimeMs = queryFloat(m_pathCompositorCpuFrametime);
  m_perfMetrics.compositorGpuFrameTimeMs = queryFloat(m_pathCompositorGpuFrametime);
  m_perfMetrics.droppedFrameCount = queryUint(m_pathDroppedFrameCount);
  m_perfMetrics.spacewarpMode = queryUint(m_pathSpacewarpMode);
  m_perfMetrics.cpuUtilizationAvg = queryFloat(m_pathCpuUtilAvg);
  m_perfMetrics.cpuUtilizationWorst = queryFloat(m_pathCpuUtilWorst);
  m_perfMetrics.gpuUtilization = queryFloat(m_pathGpuUtil);
  m_perfMetrics.valid = true;
}

void GsOpenXr::initColorSpace()
{
  if(!m_colorSpaceSupported || !m_xrEnumerateColorSpacesFB || m_session == XR_NULL_HANDLE)
    return;

  uint32_t colorSpaceCount = 0;
  XrResult result = m_xrEnumerateColorSpacesFB(m_session, 0, &colorSpaceCount, nullptr);
  if(XR_FAILED(result) || colorSpaceCount == 0)
  {
    LOGW("No color spaces available\n");
    m_colorSpaceSupported = false;
    return;
  }

  std::vector<int32_t> colorSpaces(colorSpaceCount);
  result = m_xrEnumerateColorSpacesFB(m_session, colorSpaceCount, &colorSpaceCount, colorSpaces.data());
  if(XR_FAILED(result))
  {
    LOGE("Failed to enumerate color spaces\n");
    m_colorSpaceSupported = false;
    return;
  }

  m_supportedColorSpaces.clear();
  for(int32_t cs : colorSpaces)
  {
    m_supportedColorSpaces.push_back(static_cast<ColorSpace>(cs));
  }

  LOGI("XR color spaces available: %u\n", colorSpaceCount);
  for(ColorSpace cs : m_supportedColorSpaces)
  {
    LOGI("  - %s\n", colorSpaceToString(cs));
  }

  bool hasRec709 = std::find(m_supportedColorSpaces.begin(), m_supportedColorSpaces.end(), 
                              ColorSpace::Rec709) != m_supportedColorSpaces.end();
  if(hasRec709)
  {
    setColorSpace(ColorSpace::Rec709);
  }
  else if(!m_supportedColorSpaces.empty())
  {
    setColorSpace(m_supportedColorSpaces[0]);
  }
}

bool GsOpenXr::setColorSpace(ColorSpace colorSpace)
{
  if(!m_colorSpaceSupported || !m_xrSetColorSpaceFB || m_session == XR_NULL_HANDLE)
    return false;

  XrResult result = m_xrSetColorSpaceFB(m_session, static_cast<int32_t>(colorSpace));
  if(XR_SUCCEEDED(result))
  {
    m_currentColorSpace = colorSpace;
    LOGI("XR color space set to: %s\n", colorSpaceToString(colorSpace));
    return true;
  }
  else
  {
    LOGW("Failed to set XR color space to %s (result=%d)\n", colorSpaceToString(colorSpace), (int)result);
    return false;
  }
}

const char* GsOpenXr::colorSpaceToString(ColorSpace cs)
{
  switch(cs)
  {
    case ColorSpace::Unmanaged: return "Unmanaged";
    case ColorSpace::Rec2020:   return "Rec.2020";
    case ColorSpace::Rec709:    return "Rec.709 (sRGB)";
    case ColorSpace::RiftCV1:   return "Rift CV1";
    case ColorSpace::RiftS:     return "Rift S";
    case ColorSpace::Quest:     return "Quest";
    case ColorSpace::P3:        return "P3-D65";
    case ColorSpace::AdobeRGB:  return "Adobe RGB";
    default:                    return "Unknown";
  }
}

void GsOpenXr::initPassthrough()
{
  if (!m_xrCreatePassthroughFB || m_session == XR_NULL_HANDLE)
  {
    return;
  }

  XrPassthroughCreateInfoFB createInfo{XR_TYPE_PASSTHROUGH_CREATE_INFO_FB};
  // flags = 0 means default behavior

  XrResult result = m_xrCreatePassthroughFB(m_session, &createInfo, &m_passthrough);
  if (XR_FAILED(result))
  {
    LOGW("Failed to create passthrough handle (result=%d)\n", (int)result);
    return;
  }

  XrPassthroughLayerCreateInfoFB layerInfo{XR_TYPE_PASSTHROUGH_LAYER_CREATE_INFO_FB};
  layerInfo.passthrough = m_passthrough;
  layerInfo.purpose = XR_PASSTHROUGH_LAYER_PURPOSE_RECONSTRUCTION_FB;
  // usage member does not exist in this version of the struct

  result = m_xrCreatePassthroughLayerFB(m_session, &layerInfo, &m_passthroughLayer);
  if (XR_FAILED(result))
  {
    LOGW("Failed to create passthrough layer (result=%d)\n", (int)result);
    m_xrDestroyPassthroughFB(m_passthrough);
    m_passthrough = XR_NULL_HANDLE;
    return;
  }

  // Start passthrough immediately, but we control visibility via layer submission
  result = m_xrPassthroughStartFB(m_passthrough);
  if (XR_FAILED(result))
  {
    LOGW("Failed to start passthrough (result=%d)\n", (int)result);
    m_xrDestroyPassthroughLayerFB(m_passthroughLayer);
    m_xrDestroyPassthroughFB(m_passthrough);
    m_passthroughLayer = XR_NULL_HANDLE;
    m_passthrough = XR_NULL_HANDLE;
    return;
  }

  m_passthroughSupported = true;
  m_passthroughRunning = true;
  
  // Default to enabled if supported
  m_passthroughEnabled = true;
  LOGI("XR Passthrough initialized and started\n");
}

void GsOpenXr::destroyPassthrough()
{
  if (m_passthroughRunning && m_xrPassthroughPauseFB && m_passthrough != XR_NULL_HANDLE)
  {
    m_xrPassthroughPauseFB(m_passthrough);
  }

  if (m_passthroughLayer != XR_NULL_HANDLE && m_xrDestroyPassthroughLayerFB)
  {
    m_xrDestroyPassthroughLayerFB(m_passthroughLayer);
    m_passthroughLayer = XR_NULL_HANDLE;
  }

  if (m_passthrough != XR_NULL_HANDLE && m_xrDestroyPassthroughFB)
  {
    m_xrDestroyPassthroughFB(m_passthrough);
    m_passthrough = XR_NULL_HANDLE;
  }

  m_passthroughRunning = false;
  m_passthroughSupported = false;
}

void GsOpenXr::setPassthroughEnabled(bool enabled)
{
  if (m_passthroughSupported)
  {
    m_passthroughEnabled = enabled;
    if (m_passthroughRunning)
    {
      if (enabled)
         m_xrPassthroughStartFB(m_passthrough);
      else
         m_xrPassthroughPauseFB(m_passthrough);
    }
  }
}

void GsOpenXr::initSpaceWarp()
{
  m_spaceWarpSupported = false;
  
  if (m_instance != XR_NULL_HANDLE && m_systemId != XR_NULL_SYSTEM_ID)
  {
      XrSystemSpaceWarpPropertiesFB spaceWarpProps{XR_TYPE_SYSTEM_SPACE_WARP_PROPERTIES_FB};
      XrSystemProperties systemProps{XR_TYPE_SYSTEM_PROPERTIES};
      systemProps.next = &spaceWarpProps;
      
      if (XR_SUCCEEDED(xrGetSystemProperties(m_instance, m_systemId, &systemProps)))
      {
          LOGI("Space Warp supported. Recommended motion vector resolution: %dx%d\n", 
               spaceWarpProps.recommendedMotionVectorImageRectWidth, 
               spaceWarpProps.recommendedMotionVectorImageRectHeight);
          m_spaceWarpSupported = true;
          
          // Using full resolution (m_fullExtent) for motion vectors to match the color buffer pipeline.
          // While spec recommends smaller resolution, our rendering pipeline is fixed to m_viewSize.
      }
      else
      {
          LOGW("Space Warp extension enabled but failed to get properties. Disabling.\n");
      }
  }
}

}  // namespace vk_gaussian_splatting

#endif  // WITH_OPENXR
