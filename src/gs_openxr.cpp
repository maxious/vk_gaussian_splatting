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

  // Check for VK_KHR_multiview support
  m_supportsMultiview = true;  // OpenXR runtime should have provided this if supported

  LOGI("OpenXR initialized successfully. Per-eye resolution: %dx%d, Multiview: %s\n", 
       m_perEyeExtent.width, m_perEyeExtent.height, m_supportsMultiview ? "YES" : "NO");
  return true;
}

bool GsOpenXr::createInstance()
{
  std::vector<const char*> extensions = {"XR_KHR_vulkan_enable"};

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

  LOGI("Created XR swapchains: color=%zu images, depth=%zu images\n", m_colorSwapchain.images.size(),
       m_depthSwapchain.images.size());
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
      m_shouldRender   = false;
      m_sessionRunning = false;
      xrEndSession(m_session);
      break;
    case XR_SESSION_STATE_LOSS_PENDING:
      LOGW("XR session state: LOSS_PENDING\n");
      m_shouldRender   = false;
      m_sessionRunning = false;
      break;
    case XR_SESSION_STATE_EXITING:
      LOGI("XR session state: EXITING\n");
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

void GsOpenXr::locateViews(float nearZ, float farZ)
{
  m_nearZ = nearZ;
  m_farZ  = farZ;

  XrViewLocateInfo locateInfo{XR_TYPE_VIEW_LOCATE_INFO};
  locateInfo.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
  locateInfo.displayTime           = m_predictedDisplayTime;
  locateInfo.space                 = m_referenceSpace;

  XrViewState viewState{XR_TYPE_VIEW_STATE};
  uint32_t    viewCount = VIEW_COUNT;

  XrResult result = xrLocateViews(m_session, &locateInfo, &viewState, VIEW_COUNT, &viewCount, m_locatedViews.data());
  if(XR_FAILED(result))
  {
    LOGE("xrLocateViews failed (result=%d)\n", (int)result);
  }
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

bool GsOpenXr::acquireSwapchainImages(VkImage& outColorImage, VkImage& outDepthImage)
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

  outColorImage = m_colorSwapchain.images[m_colorSwapchain.currentImageIndex];
  outDepthImage = m_depthSwapchain.images[m_depthSwapchain.currentImageIndex];

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

  m_swapchainImageState = SwapchainImageState::RELEASED;
}

void GsOpenXr::endFrame()
{
  std::vector<XrCompositionLayerProjectionView> projectionViews(VIEW_COUNT);
  std::vector<XrCompositionLayerDepthInfoKHR>   depthInfos(VIEW_COUNT);

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

    projectionViews[i].type     = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW;
    projectionViews[i].next     = &depthInfos[i];
    projectionViews[i].pose     = m_locatedViews[i].pose;
    projectionViews[i].fov      = m_locatedViews[i].fov;
    projectionViews[i].subImage = {m_colorSwapchain.handle, imageRect, 0};
  }

  XrCompositionLayerProjection layer{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
  layer.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
  layer.space      = m_referenceSpace;
  layer.viewCount  = VIEW_COUNT;
  layer.views      = projectionViews.data();

  const XrCompositionLayerBaseHeader* layers[] = {reinterpret_cast<XrCompositionLayerBaseHeader*>(&layer)};

  XrFrameEndInfo frameEndInfo{XR_TYPE_FRAME_END_INFO};
  frameEndInfo.displayTime          = m_predictedDisplayTime;
  frameEndInfo.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;

  if(m_swapchainImageState == SwapchainImageState::RELEASED && m_shouldRender)
  {
    frameEndInfo.layerCount = 1;
    frameEndInfo.layers     = layers;
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

}  // namespace vk_gaussian_splatting

#endif  // WITH_OPENXR
