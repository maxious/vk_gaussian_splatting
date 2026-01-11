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

#pragma once

#ifdef WITH_OPENXR

// Vulkan headers must be included before OpenXR platform headers
#include <vulkan/vulkan.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#ifndef XR_USE_PLATFORM_WIN32
#define XR_USE_PLATFORM_WIN32
#endif
#else
// Linux: use Xlib platform
#include <X11/Xlib.h>
#ifndef XR_USE_PLATFORM_XLIB
#define XR_USE_PLATFORM_XLIB
#endif
#endif

#ifndef XR_USE_GRAPHICS_API_VULKAN
#define XR_USE_GRAPHICS_API_VULKAN
#endif
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>
#include <glm/gtc/quaternion.hpp>
#include <array>
#include <vector>
#include <optional>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>



// Define hand tracking extension names if not available
#ifndef XR_EXT_HAND_TRACKING_EXTENSION_NAME
#define XR_EXT_HAND_TRACKING_EXTENSION_NAME "XR_EXT_hand_tracking"
#endif
#ifndef XR_FB_HAND_TRACKING_MESH_EXTENSION_NAME
#define XR_FB_HAND_TRACKING_MESH_EXTENSION_NAME "XR_FB_hand_tracking_mesh"
#endif
#ifndef XR_FB_HAND_TRACKING_AIM_EXTENSION_NAME
#define XR_FB_HAND_TRACKING_AIM_EXTENSION_NAME "XR_FB_hand_tracking_aim"
#endif
#ifndef XR_FB_HAND_TRACKING_CAPSULES_EXTENSION_NAME
#define XR_FB_HAND_TRACKING_CAPSULES_EXTENSION_NAME "XR_FB_hand_tracking_capsules"
#endif
#ifndef XR_EXT_HAND_TRACKING_DATA_SOURCE_EXTENSION_NAME
#define XR_EXT_HAND_TRACKING_DATA_SOURCE_EXTENSION_NAME "XR_EXT_hand_tracking_data_source"
#endif

namespace vk_viewer {

class GsOpenXr
{
public:
  static constexpr uint32_t VIEW_COUNT = 2;

  struct EyeData
  {
    glm::mat4 view;    // world -> view matrix
    glm::mat4 proj;    // view -> clip matrix (Vulkan-style, Y-flipped)
    glm::vec3 eyePos;  // world-space eye position
    glm::vec4 fov;     // angleLeft, angleRight, angleUp, angleDown (radians)
  };

  GsOpenXr();
  ~GsOpenXr();

  // Phase 1: Query required Vulkan extensions from OpenXR (call before creating Vulkan instance)
  // Returns true if OpenXR is available, fills outInstanceExtensions and outDeviceExtensions
  bool queryRequiredVulkanExtensions(std::vector<std::string>& outInstanceExtensions,
                                     std::vector<std::string>& outDeviceExtensions);

  // Legacy: Returns OpenXR extension names (not Vulkan extensions)
  std::vector<const char*> getRequiredInstanceExtensions() const;
  std::vector<const char*> getRequiredDeviceExtensions() const;

  // Phase 2: Initialize OpenXR with existing Vulkan resources
  bool initialize(VkInstance     vkInstance,
                  VkPhysicalDevice physicalDevice,
                  VkDevice       device,
                  uint32_t       graphicsQueueFamilyIndex,
                  uint32_t       graphicsQueueIndex,
                  VkFormat       colorFormat,
                  VkFormat       depthFormat);

  void shutdown();

  // Get the rendering resolution
  VkExtent2D getPerEyeExtent() const { return m_perEyeExtent; }
  VkExtent2D getFullExtent() const { return m_fullExtent; }  // 2 * perEye.width x perEye.height

  // Begin frame result
  enum class BeginFrameResult
  {
    RenderFully,  // Render and submit frame
    SkipRender,   // Don't render, but still call endFrame
    SkipFully     // Session not ready, don't call endFrame
  };

  // Frame lifecycle
  BeginFrameResult beginFrame();
  
  // Locate views - call after beginFrame()
  // Returns true if views were successfully located and are valid for rendering
  bool locateViews(float nearZ, float farZ);
  
  // Get per-eye view/projection data
  EyeData getEyeData(uint32_t eyeIndex) const;

  // Swapchain image management
  bool acquireSwapchainImages(VkImage& outColorImage, VkImage& outDepthImage, VkImage& outMotionImage);
  void releaseSwapchainImages();


  // End frame and submit to compositor
  void endFrame();

  // State queries
  bool isValid() const { return m_session != XR_NULL_HANDLE; }
  bool isSessionRunning() const { return m_sessionRunning; }
  bool shouldRender() const { return m_shouldRender; }
  bool supportsMultiview() const { return m_supportsMultiview; }

  // Get the predicted display time for the current frame (useful for motion prediction)
  XrTime getPredictedDisplayTime() const { return m_predictedDisplayTime; }

  // Controller input data
  struct ControllerInput
  {
    glm::vec2 thumbstick{0.0f, 0.0f};  // X: left/right, Y: forward/back
    float     trigger{0.0f};           // 0.0 to 1.0
    float     grip{0.0f};              // 0.0 to 1.0
    bool      thumbstickClick{false};
    bool      primaryButton{false};    // A/X button
    bool      secondaryButton{false};  // B/Y button
    bool      menuButton{false};
    glm::vec3 position{0.0f};
    glm::quat orientation{1.0f, 0.0f, 0.0f, 0.0f};
    bool      poseValid{false};
  };

  struct LocomotionInput
  {
    glm::vec2 move{0.0f, 0.0f};        // Left thumbstick: forward/back, strafe
    glm::vec2 turn{0.0f, 0.0f};        // Right thumbstick: turn (X only typically)
    bool      sprintPressed{false};    // Left thumbstick click
    bool      snapTurnLeft{false};     // Snap turn triggers
    bool      snapTurnRight{false};
  };

  // Poll controller input - call after beginFrame()
  void pollControllerInput();
  void pollHandInput();
  
  // Get controller data
  const ControllerInput& getLeftController() const { return m_leftController; }
  const ControllerInput& getRightController() const { return m_rightController; }
  const LocomotionInput& getLocomotionInput() const { return m_locomotionInput; }
  
  // Check if controllers are available
  bool hasControllers() const { return m_hasControllers; }

  // Hand tracking data
  enum class Hand { Left = 0, Right = 1 };

  struct HandJoint {
    glm::vec3 position{0.0f};
    glm::quat orientation{1,0,0,0};
    bool      positionValid{false};
    bool      orientationValid{false};
  };

  struct HandInput {
    bool tracked{false};           // any joints valid
    bool indexPinching{false};     // from FB aim
    glm::vec3 wristPos{0.0f};      // XR_HAND_JOINT_WRIST_EXT
    glm::quat wristRot{1,0,0,0};
    glm::vec3 indexTipPos{0.0f};   // XR_HAND_JOINT_INDEX_TIP_EXT
    glm::quat indexTipRot{1,0,0,0};

    // Optional: expose all joints for rendering / advanced logic
    std::array<HandJoint, XR_HAND_JOINT_COUNT_EXT> joints;
    std::array<XrPosef, XR_HAND_JOINT_COUNT_EXT> jointPoses;
  };

  // Get hand input data
  const HandInput& getHandInput(Hand hand) const;
  bool handsSupported() const { return m_handTrackingSupported; }

  // Get hand tracker for mesh access
  XrHandTrackerEXT getHandTracker(Hand hand) const;

  // Get hand mesh function
  XrResult getHandMeshFB(XrHandTrackerEXT handTracker, XrHandTrackingMeshFB* mesh);

public:
  struct PerformanceMetrics
  {
    float appCpuFrameTimeMs = 0.0f;
    float appGpuFrameTimeMs = 0.0f;
    float motionToPhotonLatencyMs = 0.0f;
    float compositorCpuFrameTimeMs = 0.0f;
    float compositorGpuFrameTimeMs = 0.0f;
    uint32_t droppedFrameCount = 0;
    uint32_t spacewarpMode = 0;
    float cpuUtilizationAvg = 0.0f;
    float cpuUtilizationWorst = 0.0f;
    float gpuUtilization = 0.0f;
    bool valid = false;
  };

  bool isPerformanceMetricsSupported() const { return m_perfMetricsSupported; }
  const PerformanceMetrics& getPerformanceMetrics() const { return m_perfMetrics; }
  void updatePerformanceMetrics();

  enum class ColorSpace
  {
    Unmanaged = 0,
    Rec2020 = 1,
    Rec709 = 2,
    RiftCV1 = 3,
    RiftS = 4,
    Quest = 5,
    P3 = 6,
    AdobeRGB = 7
  };

  bool isColorSpaceSupported() const { return m_colorSpaceSupported; }
  bool isSpaceWarpSupported() const { return m_spaceWarpSupported; }
  ColorSpace getNativeColorSpace() const { return m_nativeColorSpace; }
  ColorSpace getCurrentColorSpace() const { return m_currentColorSpace; }
  const std::vector<ColorSpace>& getSupportedColorSpaces() const { return m_supportedColorSpaces; }
  bool setColorSpace(ColorSpace colorSpace);
  static const char* colorSpaceToString(ColorSpace cs);

  // Utility
  static glm::mat4 createViewMatrix(const XrPosef& pose);
  static glm::mat4 createProjectionMatrix(const XrFovf& fov, float nearZ, float farZ);

private:
  // OpenXR handles
  XrInstance     m_instance      = XR_NULL_HANDLE;
  XrSystemId     m_systemId      = XR_NULL_SYSTEM_ID;
  XrSession      m_session       = XR_NULL_HANDLE;
  XrSpace        m_referenceSpace = XR_NULL_HANDLE;

  // Session state
  XrSessionState m_sessionState  = XR_SESSION_STATE_UNKNOWN;
  bool           m_sessionRunning = false;
  bool           m_shouldRender   = false;
  bool           m_supportsMultiview = false;

  // Swapchain structures
  struct Swapchain
  {
    XrSwapchain              handle = XR_NULL_HANDLE;
    std::vector<VkImage>     images;
    uint32_t                 currentImageIndex = 0;
  };

  Swapchain m_colorSwapchain;
  Swapchain m_depthSwapchain;

  enum class SwapchainImageState
  {
    UNTOUCHED,
    ACQUIRED,
    RELEASED
  };
  SwapchainImageState m_swapchainImageState = SwapchainImageState::UNTOUCHED;

  // View configuration
  VkExtent2D m_perEyeExtent{0, 0};
  VkExtent2D m_fullExtent{0, 0};

  // Frame state
  XrTime                        m_predictedDisplayTime = 0;
  std::array<XrView, VIEW_COUNT> m_locatedViews;
  float                         m_nearZ = 0.01f;
  float                         m_farZ  = 100.0f;

  // Cached Vulkan handles
  VkDevice m_device = VK_NULL_HANDLE;

  // Function pointers for Vulkan-OpenXR interop (v1 extension - XR_KHR_vulkan_enable)
  PFN_xrGetVulkanGraphicsRequirementsKHR  m_xrGetVulkanGraphicsRequirementsKHR  = nullptr;
  PFN_xrGetVulkanGraphicsDeviceKHR        m_xrGetVulkanGraphicsDeviceKHR        = nullptr;
  PFN_xrGetVulkanInstanceExtensionsKHR    m_xrGetVulkanInstanceExtensionsKHR    = nullptr;
  PFN_xrGetVulkanDeviceExtensionsKHR      m_xrGetVulkanDeviceExtensionsKHR      = nullptr;

  // Hand tracking function pointers
  PFN_xrCreateHandTrackerEXT  m_xrCreateHandTrackerEXT  = nullptr;
  PFN_xrDestroyHandTrackerEXT m_xrDestroyHandTrackerEXT = nullptr;
  PFN_xrLocateHandJointsEXT   m_xrLocateHandJointsEXT   = nullptr;
  PFN_xrGetHandMeshFB         m_xrGetHandMeshFB         = nullptr;

  // Extension availability flags (set during instance creation)
  bool m_extPerformanceMetricsAvailable = false;
  bool m_extColorSpaceAvailable = false;
  bool m_extSpaceWarpAvailable = false;
  bool m_extEnvironmentDepthAvailable = false;
  bool m_extDepthExtensionAvailable = false;
  bool m_extWin32PerfCounterAvailable = false;
  bool m_extHandTrackingAvailable = false;
  bool m_extHandTrackingMeshAvailable = false;
  bool m_extHandTrackingAimAvailable = false;
  bool m_extHandTrackingCapsulesAvailable = false;
  bool m_extHandTrackingDataSourceAvailable = false;


  // Helper methods
  bool createInstance();
  bool getSystem();
  bool createSession(VkInstance vkInstance, VkPhysicalDevice physicalDevice, VkDevice device,
                     uint32_t graphicsQueueFamilyIndex, uint32_t graphicsQueueIndex);
  bool createSwapchains(VkFormat colorFormat, VkFormat depthFormat);
  bool createReferenceSpace();
  Swapchain createSwapchain(const XrSwapchainCreateInfo& createInfo) const;

  void pollEvents();
  void handleSessionStateChange(const XrEventDataSessionStateChanged& event);

  void loadXrFunctions();

  // Controller input system
  bool createActionSet();
  void destroyActionSet();
  void syncControllerActions();
  void updateControllerPoses();

  // Action set and actions
  XrActionSet m_actionSet = XR_NULL_HANDLE;

  // Controller actions
  XrAction m_thumbstickAction     = XR_NULL_HANDLE;
  XrAction m_triggerAction        = XR_NULL_HANDLE;
  XrAction m_gripAction           = XR_NULL_HANDLE;
  XrAction m_thumbstickClickAction = XR_NULL_HANDLE;
  XrAction m_primaryButtonAction  = XR_NULL_HANDLE;
  XrAction m_secondaryButtonAction = XR_NULL_HANDLE;
  XrAction m_menuButtonAction     = XR_NULL_HANDLE;
  XrAction m_poseAction           = XR_NULL_HANDLE;

  // Action spaces for controller poses
  XrSpace m_leftHandSpace  = XR_NULL_HANDLE;
  XrSpace m_rightHandSpace = XR_NULL_HANDLE;

  // Subaction paths
  XrPath m_leftHandPath  = XR_NULL_PATH;
  XrPath m_rightHandPath = XR_NULL_PATH;

  // Controller state
  ControllerInput m_leftController;
  ControllerInput m_rightController;
  LocomotionInput m_locomotionInput;
  bool            m_hasControllers = false;

  // Snap turn state
  bool m_snapTurnLeftTriggered  = false;
  bool m_snapTurnRightTriggered = false;
  static constexpr float SNAP_TURN_THRESHOLD = 0.7f;
  
  // Tracking state
  uint32_t m_trackingLossFrameCount = 0;
  static constexpr uint32_t MAX_TRACKING_LOSS_FRAMES = 60;

  // ============================================================================
  // High-frequency async tracking thread
  // ============================================================================
  // Samples poses at ~1000 Hz independently of the 72 Hz frame loop.
  // This allows us to use the freshest tracking data when rendering.

  struct TimestampedPose
  {
    XrTime                        timestamp = 0;
    std::array<XrView, VIEW_COUNT> views;
    XrViewStateFlags              viewStateFlags = 0;
    bool                          valid = false;
  };

  static constexpr size_t POSE_RING_BUFFER_SIZE = 128;  // ~128ms at 1000Hz
  static constexpr int    TRACKING_SAMPLE_RATE_HZ = 1000;

  std::array<TimestampedPose, POSE_RING_BUFFER_SIZE> m_poseRingBuffer;
  std::atomic<size_t>   m_poseWriteIndex{0};
  std::atomic<size_t>   m_poseCount{0};  // Number of valid samples in buffer
  mutable std::mutex    m_poseMutex;     // Protects ring buffer access during interpolation

  std::thread           m_trackingThread;
  std::atomic<bool>     m_trackingThreadRunning{false};
  std::atomic<bool>     m_trackingThreadShouldStop{false};

  // Tracking thread methods
  void startTrackingThread();
  void stopTrackingThread();
  void trackingThreadLoop();
  
  // Get the best pose for a given display time (interpolates if needed)
  bool getPoseForTime(XrTime targetTime, std::array<XrView, VIEW_COUNT>& outViews) const;
  
  // Linear interpolation helper for XrView
  static XrView interpolateView(const XrView& a, const XrView& b, float t);
  static XrPosef interpolatePose(const XrPosef& a, const XrPosef& b, float t);
  static XrQuaternionf slerp(const XrQuaternionf& a, const XrQuaternionf& b, float t);

  // XR_META_performance_metrics support
  bool m_perfMetricsSupported = false;
  bool m_perfMetricsEnabled = false;
  PerformanceMetrics m_perfMetrics;
  
  std::vector<XrPath> m_perfMetricsPaths;
  XrPath m_pathAppCpuFrametime = XR_NULL_PATH;
  XrPath m_pathAppGpuFrametime = XR_NULL_PATH;
  XrPath m_pathMotionToPhoton = XR_NULL_PATH;
  XrPath m_pathCompositorCpuFrametime = XR_NULL_PATH;
  XrPath m_pathCompositorGpuFrametime = XR_NULL_PATH;
  XrPath m_pathDroppedFrameCount = XR_NULL_PATH;
  XrPath m_pathSpacewarpMode = XR_NULL_PATH;
  XrPath m_pathCpuUtilAvg = XR_NULL_PATH;
  XrPath m_pathCpuUtilWorst = XR_NULL_PATH;
  XrPath m_pathGpuUtil = XR_NULL_PATH;

  using PFN_xrEnumeratePerformanceMetricsCounterPathsMETA = XrResult(XRAPI_PTR*)(XrInstance, uint32_t, uint32_t*, XrPath*);
  using PFN_xrSetPerformanceMetricsStateMETA = XrResult(XRAPI_PTR*)(XrSession, const void*);
  using PFN_xrGetPerformanceMetricsStateMETA = XrResult(XRAPI_PTR*)(XrSession, void*);
  using PFN_xrQueryPerformanceMetricsCounterMETA = XrResult(XRAPI_PTR*)(XrSession, XrPath, void*);

  PFN_xrEnumeratePerformanceMetricsCounterPathsMETA m_xrEnumeratePerformanceMetricsCounterPathsMETA = nullptr;
  PFN_xrSetPerformanceMetricsStateMETA m_xrSetPerformanceMetricsStateMETA = nullptr;
  PFN_xrGetPerformanceMetricsStateMETA m_xrGetPerformanceMetricsStateMETA = nullptr;
  PFN_xrQueryPerformanceMetricsCounterMETA m_xrQueryPerformanceMetricsCounterMETA = nullptr;

  void initPerformanceMetrics();
  void enablePerformanceMetrics();

  // XR_FB_color_space support
  bool m_colorSpaceSupported = false;
  ColorSpace m_nativeColorSpace = ColorSpace::Unmanaged;
  ColorSpace m_currentColorSpace = ColorSpace::Unmanaged;
  std::vector<ColorSpace> m_supportedColorSpaces;

  using PFN_xrEnumerateColorSpacesFB = XrResult(XRAPI_PTR*)(XrSession, uint32_t, uint32_t*, int32_t*);
  using PFN_xrSetColorSpaceFB = XrResult(XRAPI_PTR*)(XrSession, int32_t);

  PFN_xrEnumerateColorSpacesFB m_xrEnumerateColorSpacesFB = nullptr;
  PFN_xrSetColorSpaceFB m_xrSetColorSpaceFB = nullptr;

  void initColorSpace();

  // XR_FB_space_warp support
  bool m_spaceWarpSupported = false;
  Swapchain m_motionVectorSwapchain;
  
  // App space pose from previous frame for delta calculation
  XrPosef m_prevAppSpacePose = { {0,0,0,1}, {0,0,0} };
  bool m_prevAppSpacePoseValid = false;

  void initSpaceWarp();

  // XR_META_environment_depth support
  bool m_environmentDepthSupported = false;
  bool m_environmentDepthEnabled = false;
  bool m_environmentDepthRunning = false;

  // Hand tracking support
  bool m_handTrackingSupported = false;
  HandInput m_handInputs[2]; // [Left, Right]

  // Hand trackers and buffers
  XrHandTrackerEXT m_handTracker[2] = {XR_NULL_HANDLE, XR_NULL_HANDLE};
  XrHandJointLocationEXT m_jointLocations[2][XR_HAND_JOINT_COUNT_EXT]{};
  XrHandJointVelocityEXT m_jointVelocities[2][XR_HAND_JOINT_COUNT_EXT]{};
  XrHandTrackingAimStateFB m_aimState[2]{};

  XrEnvironmentDepthProviderMETA m_environmentDepthProvider = XR_NULL_HANDLE;
  XrEnvironmentDepthSwapchainMETA m_environmentDepthSwapchain = XR_NULL_HANDLE;
  std::vector<VkImage> m_environmentDepthImages;

  PFN_xrCreateEnvironmentDepthProviderMETA m_xrCreateEnvironmentDepthProviderMETA = nullptr;
  PFN_xrDestroyEnvironmentDepthProviderMETA m_xrDestroyEnvironmentDepthProviderMETA = nullptr;
  PFN_xrStartEnvironmentDepthProviderMETA m_xrStartEnvironmentDepthProviderMETA = nullptr;
  PFN_xrStopEnvironmentDepthProviderMETA m_xrStopEnvironmentDepthProviderMETA = nullptr;
  PFN_xrCreateEnvironmentDepthSwapchainMETA m_xrCreateEnvironmentDepthSwapchainMETA = nullptr;
  PFN_xrDestroyEnvironmentDepthSwapchainMETA m_xrDestroyEnvironmentDepthSwapchainMETA = nullptr;
  PFN_xrGetEnvironmentDepthSwapchainStateMETA m_xrGetEnvironmentDepthSwapchainStateMETA = nullptr;
  PFN_xrAcquireEnvironmentDepthImageMETA m_xrAcquireEnvironmentDepthImageMETA = nullptr;
  PFN_xrEnumerateEnvironmentDepthSwapchainImagesMETA m_xrEnumerateEnvironmentDepthSwapchainImagesMETA = nullptr;
  PFN_xrSetEnvironmentDepthHandRemovalMETA m_xrSetEnvironmentDepthHandRemovalMETA = nullptr;

  void initEnvironmentDepth();
  void destroyEnvironmentDepth();

public:
  bool isEnvironmentDepthSupported() const { return m_environmentDepthSupported; }
  bool isEnvironmentDepthEnabled() const { return m_environmentDepthEnabled; }

  bool acquireEnvironmentDepthImage(VkImage& outDepthImage, XrEnvironmentDepthImageMETA& outDepthInfo);
};

}  // namespace vk_viewer

#endif  // WITH_OPENXR
