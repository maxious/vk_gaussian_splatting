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

#include <vk_viewer_ui.h>

// Define the dynamic dispatcher storage
// This handles LNK2001: unresolved external symbol "class vk::detail::DispatchLoaderDynamic vk::detail::defaultDispatchLoaderDynamic"
#include <vulkan/vulkan.hpp>
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

using namespace vk_viewer;

// Static member definition (declared in vk_viewer.h)
bool VkViewer::s_stochasticGsSupported = false;

// create, setup and run an nvapp::Application
// with a VkViewer element.
int main(int argc, char** argv)
{
  nvutils::Logger::getInstance().breakOnError(false);
  // Enable file flushing so logs are preserved on crash
  nvutils::Logger::getInstance().setFileFlush(true);
  
  // Check for debug log level via environment variable
  if(const char* debugEnv = std::getenv("VK_VIEWER_DEBUG"))
  {
    if(std::string(debugEnv) == "1")
    {
      nvutils::Logger::getInstance().setMinimumLogLevel(nvutils::Logger::LogLevel::eDEBUG);
    }
  }

  nvutils::ProfilerManager              profilerManager;
  nvutils::ParameterRegistry            parameterRegistry;
  nvutils::ParameterParser              parameterParser(nvutils::getExecutablePath().stem().string(), {".txt"});
  nvutils::ParameterSequencer::InitInfo sequencerInfo{// sequencer always requires a parser and registry
                                                      .parameterParser   = &parameterParser,
                                                      .parameterRegistry = &parameterRegistry,
                                                      // sequencer uses the profiler for benchmarking
                                                      .profilerManager = &profilerManager};

  nvvk::Context                vkContext;  // The Vulkan context
  nvvk::ContextInitInfo        vkSetup;    // Information to create the Vulkan context
  nvapp::Application           application;
  nvapp::ApplicationCreateInfo appInfo;  // Information to create the application
  appInfo.vSync = true;
  bool                         benchmarkMode = false;
  std::string                  tcpDepthServers;

  /////////////////////////////////
  // Parse the command line to get the application creation information
  // those parameter will have no effect if changed via benchmark script
  // see VkViewer constructor for other options
  parameterRegistry.addVector({"size", "Size of the window to be created"}, &appInfo.windowSize);
  parameterRegistry.add({"vsync"}, &appInfo.vSync);
  parameterRegistry.add({"verbose", "Verbose output of the Vulkan context"}, &vkSetup.verbose);
  parameterRegistry.add({"validation", "Enable validation layers"}, &vkSetup.enableValidationLayers);
  parameterRegistry.add({"benchmark", "Enable benchmarking, prevents async loadings and turns off vsync"}, &benchmarkMode);
  parameterRegistry.add({"forcegpu", "Force the use of a specific GPU by probviding its ID"}, &vkSetup.forceGPU);
  parameterRegistry.add({"tcp-depth-servers", "Comma separated list of TCP depth servers"}, &tcpDepthServers);

  registerCommandLineParameters(&parameterRegistry);

  /////////////////////////////////
  // Create elements of the application, including the core of the sample (vkViewer)

  // The VkViewerUI includes the core VkViewer class by inheritance
  auto vkViewer = std::make_shared<VkViewerUI>(&profilerManager, &parameterRegistry, &benchmarkMode);

  // add a few more parameters to registry and parser to handle sequencer settings
  sequencerInfo.registerScriptParameters(parameterRegistry, parameterParser);

  // extends reporting output with memory consumption information
  sequencerInfo.postCallbacks.emplace_back(
      [&](const nvutils::ParameterSequencer::State& /* unused */) { vkViewer->benchmarkAdvance(); });

  // After the creation of the elements we have more parameters in the registry than before (from vkViewer).
  // Therefore add the entire registry to the commandline parser again, to add new ones.
  parameterParser.add(parameterRegistry);
  // commandline parsing
  parameterParser.parse(argc, argv);
  // backup the default applications parameters, including those modified by command line
  storeDefaultParameters();
  // set more verbose for benchmark usage later on
  parameterParser.setVerbose(true);

  // this element requires sequencerInfo that is potentially updated by parameterParser
  auto elemSequencer = std::make_shared<nvapp::ElementSequencer>(sequencerInfo);

  /////////////////////////////////
  // Vulkan creation context information
  vkSetup.enableAllFeatures = true;

  // - Instance extensions
  vkSetup.instanceExtensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

  // - Device extensions
  static VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR baryFeaturesKHR = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
  static VkPhysicalDeviceMeshShaderFeaturesEXT meshFeaturesEXT = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
  };

  static VkPhysicalDeviceFragmentShadingRateFeaturesKHR fragFeaturesKHR = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR,
  };
  vkSetup.deviceExtensions.emplace_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);  // for vk_radix_sort (vrdx)
  vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  vkSetup.deviceExtensions.emplace_back(VK_EXT_MESH_SHADER_EXTENSION_NAME, &meshFeaturesEXT, true);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME, &fragFeaturesKHR, true);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME, &baryFeaturesKHR, true);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);  // for ImGui
  vkSetup.deviceExtensions.emplace_back(VK_KHR_MAINTENANCE_8_EXTENSION_NAME);

  // Activate the ray tracing extension
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  vkSetup.deviceExtensions.emplace_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature, true);  // To build acceleration structures
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  vkSetup.deviceExtensions.emplace_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rtPipelineFeature, false);  // To use vkCmdTraceRaysKHR
  vkSetup.deviceExtensions.emplace_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);  // Required by ray tracing pipeline

  VkPhysicalDeviceShaderClockFeaturesKHR clockFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
  vkSetup.deviceExtensions.emplace_back(VK_KHR_SHADER_CLOCK_EXTENSION_NAME, &clockFeatures);

  // 64-bit SSBO atomics for stochastic Gaussian splat rendering.
  // Enabled automatically by enableAllFeatures=true (Vulkan 1.2 core feature).
  // Only require the extension name for the driver to expose it.
  vkSetup.deviceExtensions.emplace_back(VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME);

  VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV serFeatures = {
      .sType                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV,
      .rayTracingInvocationReorder = VK_TRUE,
  };
  vkSetup.deviceExtensions.emplace_back(VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME, &serFeatures, false);

  // Blackwell native sphere primitives (VK_NV_ray_tracing_linear_swept_spheres)
  // Enables hardware-accelerated ray-sphere intersection for Gaussian splats
  VkPhysicalDeviceRayTracingLinearSweptSpheresFeaturesNV sphereFeatures = {
      .sType                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_LINEAR_SWEPT_SPHERES_FEATURES_NV,
      .spheres                     = VK_TRUE,
      .linearSweptSpheres          = VK_FALSE,  // We only need spheres, not LSS
  };
  vkSetup.deviceExtensions.emplace_back(VK_NV_RAY_TRACING_LINEAR_SWEPT_SPHERES_EXTENSION_NAME, &sphereFeatures, false);

  // VK_NV_partitioned_acceleration_structure (PTLAS) for sparse FreeTimeGS updates
  // Enables GPU-driven partial TLAS updates - only rebuild changed partitions
  // Ideal for 4D Gaussian splats where <10% of splats animate per frame
  VkPhysicalDevicePartitionedAccelerationStructureFeaturesNV ptlasFeatures = {
      .sType                            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PARTITIONED_ACCELERATION_STRUCTURE_FEATURES_NV,
      .partitionedAccelerationStructure = VK_TRUE,
  };
  vkSetup.deviceExtensions.emplace_back(VK_NV_PARTITIONED_ACCELERATION_STRUCTURE_EXTENSION_NAME, &ptlasFeatures, false);

#ifdef WITH_DLSS_RR
  // Required for DLSS-RR CUDA-Vulkan interop
  vkSetup.deviceExtensions.emplace_back(VK_NVX_BINARY_IMPORT_EXTENSION_NAME, nullptr, false);
  vkSetup.deviceExtensions.emplace_back(VK_NVX_IMAGE_VIEW_HANDLE_EXTENSION_NAME, nullptr, false);
#endif

#ifdef WITH_VULKAN_VIDEO
  // Vulkan Video extensions for hardware-accelerated video decode
  // These are optional - will gracefully degrade to FFmpeg if not available
  vkSetup.instanceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
  vkSetup.instanceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);

  vkSetup.queues.push_back(VK_QUEUE_VIDEO_DECODE_BIT_KHR);

  static VkPhysicalDeviceVideoMaintenance1FeaturesKHR videoMaintenanceFeatures = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_MAINTENANCE_1_FEATURES_KHR,
      .videoMaintenance1 = VK_TRUE
  };
  static VkPhysicalDeviceSynchronization2FeaturesKHR sync2Features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR,
      .synchronization2 = VK_TRUE
  };

  static VkVideoProfileInfoKHR videoProfiles[] = {
      {
          .sType               = VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR,
          .videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR,
          .chromaSubsampling   = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR,
          .lumaBitDepth        = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
          .chromaBitDepth      = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
      },
      {
          .sType               = VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR,
          .videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR,
          .chromaSubsampling   = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR,
          .lumaBitDepth        = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
          .chromaBitDepth      = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
      },
      {
          .sType               = VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR,
          .videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR,
          .chromaSubsampling   = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR,
          .lumaBitDepth        = VK_VIDEO_COMPONENT_BIT_DEPTH_10_BIT_KHR,
          .chromaBitDepth      = VK_VIDEO_COMPONENT_BIT_DEPTH_10_BIT_KHR,
      }};

  static VkVideoProfileListInfoKHR videoProfileList = {
      .sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR, .profileCount = 3, .pProfiles = videoProfiles};

  vkSetup.deviceExtensions.emplace_back(VK_KHR_VIDEO_QUEUE_EXTENSION_NAME, nullptr, true);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME, nullptr, true);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_VIDEO_DECODE_H265_EXTENSION_NAME, nullptr, true);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_VIDEO_DECODE_H264_EXTENSION_NAME, nullptr, true);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_VIDEO_MAINTENANCE_1_EXTENSION_NAME, &videoMaintenanceFeatures, true);
  
  // YCbCr conversion for sampling decoded video frames
  vkSetup.deviceExtensions.emplace_back(VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME, nullptr, false);

  // FFmpeg Vulkan HW acceleration requirements
  vkSetup.deviceExtensions.emplace_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME, nullptr, false);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME, nullptr, false);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, nullptr, false);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME, nullptr, false);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME, &sync2Features, false);
#endif

  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }

#ifdef WITH_OPENXR
  // Query OpenXR required Vulkan extensions before creating Vulkan context
  // Store extension strings in static vectors to keep pointers valid
  static std::vector<std::string> xrInstanceExts, xrDeviceExts;
  if(vkViewer->queryOpenXrVulkanExtensions(xrInstanceExts, xrDeviceExts))
  {
    for(const auto& ext : xrInstanceExts)
    {
      vkSetup.instanceExtensions.emplace_back(ext.c_str());
    }
    for(const auto& ext : xrDeviceExts)
    {
      vkSetup.deviceExtensions.emplace_back(ext.c_str(), nullptr, false);
    }
  }
#endif

  // Setting up the validation layers
  nvvk::ValidationSettings vvlInfo{};
  // Keep core object-lifetime checks enabled; they are essential when tracking
  // invalid handles during resize and shutdown.
  vvlInfo.setPreset(nvvk::ValidationSettings::LayerPresets::eStandard);
  vvlInfo.validate_sync             = VK_TRUE;
  vvlInfo.validate_best_practices   = VK_TRUE;
  vkSetup.instanceCreateInfoExt = vvlInfo.buildPNextChain();  // Adding the validation layer settings

  // Create Vulkan context
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }

  VkViewer::s_stochasticGsSupported = (vkContext.getPhysicalDeviceFeatures12().shaderBufferInt64Atomics == VK_TRUE);

#ifdef WITH_VULKAN_VIDEO
  LOGI("Vulkan Video Extensions Status:\n");
  auto logExt = [&](const char* name) {
    LOGI("  %s: %s\n", name, vkContext.hasExtensionEnabled(name) ? "Enabled" : "Not Supported/Enabled");
  };
  logExt(VK_KHR_VIDEO_QUEUE_EXTENSION_NAME);
  logExt(VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME);
  logExt(VK_KHR_VIDEO_DECODE_H264_EXTENSION_NAME);
  logExt(VK_KHR_VIDEO_DECODE_H265_EXTENSION_NAME);
  logExt(VK_KHR_VIDEO_MAINTENANCE_1_EXTENSION_NAME);
  logExt(VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME);
  logExt(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);

  LOGI("Vulkan Queues Info:\n");
  for(const auto& q : vkContext.getQueueInfos())
  {
    LOGI("  Family: %d, Index: %d\n", q.familyIndex, q.queueIndex);
  }
#endif

  /////////////////////////////////
  // Application setup
  appInfo.name                  = TARGET_NAME;
  appInfo.instance              = vkContext.getInstance();
  appInfo.device                = vkContext.getDevice();
  appInfo.physicalDevice        = vkContext.getPhysicalDevice();
  appInfo.queues                = vkContext.getQueueInfos();
  appInfo.hasUndockableViewport = true;
  appInfo.useMenu               = !benchmarkMode;  // we hide the menu in benchmark mode

  // Setting up the layout of the application
  appInfo.dockSetup = [](ImGuiID viewportID) {
    // right side panel container
    ImGuiID assetsID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Right, 0.20F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Assets", assetsID);
    ImGuiID propertiesID = ImGui::DockBuilderSplitNode(assetsID, ImGuiDir_Down, 0.75F, nullptr, &assetsID);
    ImGui::DockBuilderDockWindow("Properties", propertiesID);

    // bottom panel container
    ImGuiID memoryID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Down, 0.45F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Memory Statistics", memoryID);
    ImGui::DockBuilderDockWindow("ComfyUI 3D Generator", memoryID);
    ImGuiID profilerID = ImGui::DockBuilderSplitNode(memoryID, ImGuiDir_Right, 0.33F, nullptr, &memoryID);
    ImGui::DockBuilderDockWindow("Profiler", profilerID);
    ImGuiID renderingID = ImGui::DockBuilderSplitNode(profilerID, ImGuiDir_Down, 0.30F, nullptr, &profilerID);
    ImGui::DockBuilderDockWindow("Rendering Statistics", renderingID);
  };

  //
  vkViewer->guiRegisterIniFileHandlers();

  // Initializes the application
  application.init(appInfo);

  // Add all application elements including our sample specific vkViewer
  // onAttach will be invoked on elements at this stage
  application.addElement(elemSequencer);
  application.addElement(vkViewer);

#ifdef WITH_TCP_DEPTH
  if(!tcpDepthServers.empty())
  {
    if(!prmScene.sceneToLoadFilename.empty() && isImageFile(prmScene.sceneToLoadFilename.string()))
    {
      vkViewer->requestSingleImageDepth(prmScene.sceneToLoadFilename.string(), tcpDepthServers);
      prmScene.sceneToLoadFilename.clear();
    }
    else
    {
      vkViewer->enableTcpDepth(tcpDepthServers, prmScene.sceneToLoadFilename.string());
    }
  }
  else if(!prmScene.sceneToLoadFilename.empty() && isImageFile(prmScene.sceneToLoadFilename.string()))
  {
    LOGW("Ignoring image input without --tcp-depth-servers: %s\n", prmScene.sceneToLoadFilename.string().c_str());
    prmScene.sceneToLoadFilename.clear();
  }
#endif

  application.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>("", fmt::format("({})", "GLSL")));

  auto elemCamera = std::make_shared<nvapp::ElementCamera>();
  elemCamera->setCameraManipulator(vkViewer->cameraManip);
  application.addElement(elemCamera);

  if(benchmarkMode)
  {
    // In this mode we do not display the GUI elements
    application.setVsync(false);
  }
  else
  {
    application.addElement(std::make_shared<nvgpu_monitor::ElementGpuMonitor>());

    // setup the profiler element and view
    auto profilerViewSettings = std::make_shared<nvapp::ElementProfiler::ViewSettings>(
        nvapp::ElementProfiler::ViewSettings{.name       = "Profiler",
                                             .defaultTab = nvapp::ElementProfiler::TABLE,
                                             .pieChart   = {.cpuTotal = false, .levels = true},
                                             .lineChart  = {.cpuLine = false}});

    // setting are optional, but can be used to expose to sample code (like hiding views for benchmark)
    application.addElement(std::make_shared<nvapp::ElementProfiler>(&profilerManager, profilerViewSettings));
  }

  //
  application.run();

  // Cleanup
  application.deinit();
  vkContext.deinit();

  return 0;
}
