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
}

std::vector<const char*> GsOpenXr::getRequiredInstanceExtensions() const
{
  return {"XR_KHR_vulkan_enable2"};
}

std::vector<const char*> GsOpenXr::getRequiredDeviceExtensions() const
{
  return {};
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

  if(!createInstance())
    return false;

  loadXrFunctions();

  if(!getSystem())
    return false;

  if(!createSession(vkInstance, physicalDevice, device, graphicsQueueFamilyIndex, graphicsQueueIndex))
    return false;

  if(!createSwapchains(colorFormat, depthFormat))
    return false;

  if(!createReferenceSpace())
    return false;

  LOGI("OpenXR initialized successfully. Per-eye resolution: %dx%d\n", m_perEyeExtent.width, m_perEyeExtent.height);
  return true;
}

bool GsOpenXr::createInstance()
{
  std::vector<const char*> extensions = {"XR_KHR_vulkan_enable2"};

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
  xrGetInstanceProcAddr(m_instance, "xrGetVulkanGraphicsRequirements2KHR",
                        (PFN_xrVoidFunction*)&m_xrGetVulkanGraphicsRequirements2KHR);
  xrGetInstanceProcAddr(m_instance, "xrGetVulkanGraphicsDevice2KHR",
                        (PFN_xrVoidFunction*)&m_xrGetVulkanGraphicsDevice2KHR);
  xrGetInstanceProcAddr(m_instance, "xrCreateVulkanInstanceKHR",
                        (PFN_xrVoidFunction*)&m_xrCreateVulkanInstanceKHR);
  xrGetInstanceProcAddr(m_instance, "xrCreateVulkanDeviceKHR",
                        (PFN_xrVoidFunction*)&m_xrCreateVulkanDeviceKHR);
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
  // Check graphics requirements
  XrGraphicsRequirementsVulkan2KHR graphicsRequirements{XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN2_KHR};
  XR_CHECK(m_xrGetVulkanGraphicsRequirements2KHR(m_instance, m_systemId, &graphicsRequirements),
           "Failed to get Vulkan graphics requirements");

  // Create session with Vulkan binding
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

  // Wait for session to be ready
  LOGI("Waiting for XR session to become ready...\n");
  while(m_sessionState != XR_SESSION_STATE_READY)
  {
    pollEvents();
    if(m_sessionState == XR_SESSION_STATE_LOSS_PENDING || m_sessionState == XR_SESSION_STATE_EXITING)
    {
      LOGE("XR session lost during initialization\n");
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // Begin session
  XrSessionBeginInfo sessionBeginInfo{XR_TYPE_SESSION_BEGIN_INFO};
  sessionBeginInfo.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
  XR_CHECK(xrBeginSession(m_session, &sessionBeginInfo), "Failed to begin XR session");

  m_sessionRunning = true;
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
    case XR_SESSION_STATE_READY:
      LOGI("XR session state: READY\n");
      break;
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

bool GsOpenXr::beginFrame()
{
  pollEvents();

  if(!m_sessionRunning)
    return false;

  XrFrameWaitInfo frameWaitInfo{XR_TYPE_FRAME_WAIT_INFO};
  XrFrameState    frameState{XR_TYPE_FRAME_STATE};

  XrResult result = xrWaitFrame(m_session, &frameWaitInfo, &frameState);
  if(XR_FAILED(result))
  {
    LOGE("xrWaitFrame failed (result=%d)\n", (int)result);
    return false;
  }

  m_predictedDisplayTime = frameState.predictedDisplayTime;
  m_shouldRender         = frameState.shouldRender == XR_TRUE;

  XrFrameBeginInfo frameBeginInfo{XR_TYPE_FRAME_BEGIN_INFO};
  result = xrBeginFrame(m_session, &frameBeginInfo);
  if(XR_FAILED(result))
  {
    LOGE("xrBeginFrame failed (result=%d)\n", (int)result);
    return false;
  }

  m_swapchainImageState = SwapchainImageState::UNTOUCHED;
  return m_shouldRender;
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

}  // namespace vk_gaussian_splatting

#endif  // WITH_OPENXR
